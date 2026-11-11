"""Native-coordinate inference, pooled metrics, and training-only EMA/BN tools."""
from contextlib import nullcontext
import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def autocast(device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" and torch.cuda.is_bf16_supported() else nullcontext()


@torch.no_grad()
def predict_probability(model, native_rgb, tta=True, geometry="native"):
    """Input is normalized BCHW at its native (unpadded) size.

    'padded' is a compatibility profile for the historical standalone checkpoint.
    New experiments should use 'native': flip before padding, crop before unflip.
    """
    height, width = native_rgb.shape[-2:]
    if geometry not in ("native", "padded"):
        raise ValueError("geometry must be native or padded")
    work = native_rgb
    if geometry == "padded":
        work = F.pad(work, (0, (-width) % 32, 0, (-height) % 32), mode="replicate")
    h, w = work.shape[-2:]
    result = torch.zeros((len(work), 1, h, w), device=work.device, dtype=torch.float32)
    transforms = [(), (-1,), (-2,), (-2, -1)] if tta else [()]
    for axes in transforms:
        transformed = work.flip(axes) if axes else work
        transformed = F.pad(transformed, (0, (-w) % 32, 0, (-h) % 32), mode="replicate")
        with autocast(work.device):
            probability = model(transformed).float().sigmoid()
        probability = probability[..., :h, :w]
        result += probability.flip(axes) if axes else probability
    return (result / len(transforms))[..., :height, :width]


def pixel_counts(probability, target, threshold=0.5):
    p, y = probability >= threshold, target.bool()
    return {"tp": int((p & y).sum()), "fp": int((p & ~y).sum()),
            "fn": int((~p & y).sum()), "tn": int((~p & ~y).sum())}


def scores(counts):
    tp, fp, fn, tn = [counts[k] for k in ("tp", "fp", "fn", "tn")]
    fg = tp / max(tp + fp + fn, 1)
    bg = tn / max(tn + fp + fn, 1)
    return {"mIoU": (fg + bg) / 2, "CrackIoU": fg,
            "CrackF1": 2 * tp / max(2 * tp + fp + fn, 1),
            "CrackPrecision": tp / max(tp + fp, 1), "CrackRecall": tp / max(tp + fn, 1),
            "aAcc": (tp + tn) / max(tp + fp + fn + tn, 1)}


@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5):
    model.eval()
    total = dict.fromkeys(("tp", "fp", "fn", "tn"), 0)
    for batch in loader:
        for i, (height, width) in enumerate(batch["native_size"].tolist()):
            image = batch["image"][i:i + 1, :, :height, :width].to(device)
            p = predict_probability(model, image)
            y = batch["mask"][i:i + 1, :, :height, :width].to(device)
            counts = pixel_counts(p, y, threshold)
            for key in total:
                total[key] += counts[key]
    return {"counts": total, **scores(total)}


@torch.no_grad()
def update_ema(average, model, decay):
    source = dict(model.named_parameters())
    for name, p in average.named_parameters():
        p.lerp_(source[name], 1 - decay)
    buffers = dict(model.named_buffers())
    for name, p in average.named_buffers():
        p.copy_(buffers[name])


@torch.no_grad()
def refresh_bn(model, loader, device, batches=84):
    model.eval()
    layers = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    momenta = [m.momentum for m in layers]
    for m in layers:
        m.reset_running_stats()
        m.momentum = None
        m.train()
    for index, batch in enumerate(loader):
        if index >= batches:
            break
        with autocast(device):
            model(batch["image"].to(device))
    for m, momentum in zip(layers, momenta):
        m.momentum = momentum
    model.eval()
