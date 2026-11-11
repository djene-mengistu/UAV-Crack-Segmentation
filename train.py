"""Small supervised reference trainer; released checkpoints reproduce inference."""
import argparse
import copy
import json
import math
from pathlib import Path
import time
import torch
from torch.utils.data import DataLoader
from dataset import CrackDataset, collate_samples, read_manifest
from losses import segmentation_loss, ensemble_loss
from model import build_model, load_encoder
from utils import autocast, evaluate, refresh_bn, seed_everything, update_ema


def optimizer_for(model, config):
    groups = {}
    for name, parameter in model.named_parameters():
        lr = config.get("refinement_lr", config["lr"]) if name.startswith("detail.") else (config["encoder_lr"] if "encoder." in name else config["lr"])
        decay = config.get("weight_decay", 0.01) if parameter.ndim > 1 else 0.0
        groups.setdefault((lr, decay), []).append(parameter)
    return torch.optim.AdamW([{"params": p, "lr": lr, "initial_lr": lr, "weight_decay": decay} for (lr, decay), p in groups.items()])


def train(config, root, manifest_path, output, init=None, encoder_weights=None, full=False, device="cuda"):
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use --device cpu for a small smoke test")
    seed_everything(config.get("seed", 2026))
    torch.set_num_threads(4)
    manifest = read_manifest(manifest_path)
    training = manifest["train"] + manifest.get("val", []) if full else manifest["train"]
    validation = [] if full else manifest.get("val", [])
    if not training or (not full and not validation):
        raise ValueError("Provide nonempty train/val records, or use --full for a fixed all-label run")
    batch_size = config.get("batch_size", 4)
    if batch_size < 2 or len(training) % batch_size == 1:
        raise ValueError("UPer PPM BatchNorm needs batches >=2. Choose a batch size without a singleton final batch; samples are never silently dropped.")
    context_weight = config.get("ensemble_weight", 0.0)
    dataset = CrackDataset(root, training, training=True, use_context=context_weight > 0, student_weight=config.get("student_weight", 0.14))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=config.get("workers", 2), collate_fn=collate_samples)
    clean = CrackDataset(root, training, training=False)
    bn_loader = DataLoader(clean, batch_size=2, shuffle=True, drop_last=True, collate_fn=collate_samples)
    val_loader = DataLoader(CrackDataset(root, validation), batch_size=batch_size, collate_fn=collate_samples) if validation else None
    model = build_model(**config["model"]).to(device)
    if init:
        checkpoint = torch.load(init, map_location="cpu", weights_only=True)
        source_config = checkpoint["model_config"]
        if source_config == config["model"]:
            model.load_state_dict(checkpoint["state_dict"], strict=True)
        elif not source_config["refinement"] and config["model"]["refinement"] and all(source_config[k] == config["model"][k] for k in ("decoder_width", "first_stride")):
            model.base.load_state_dict(checkpoint["state_dict"], strict=True)
        else:
            raise ValueError("Initialization architecture differs; only adding a zero-initialized refinement head is supported")
    elif encoder_weights:
        load_encoder(model, encoder_weights)
    average = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = optimizer_for(model, config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    (output / "provenance.json").write_text(json.dumps({"full_training": full, "training_images": [r["image"] for r in training], "validation_images": [r["image"] for r in validation]}, indent=2))
    epochs = config.get("epochs", 120)
    accumulation = config.get("accumulation", 1)
    best, best_epoch, steps, history = -1.0, 0, 0, []

    def checkpoint(metrics, epoch, variant):
        return {"state_dict": (model if variant == "live" else average).state_dict(), "model_config": config["model"],
                "epoch": epoch, "metrics": metrics, "weights_variant": variant,
                "inference": {"threshold": config.get("threshold", 0.5), "tta": True, "geometry": "native"}}

    if init and not full:
        initial = evaluate(model, val_loader, device, config.get("threshold", 0.5))
        best = initial["mIoU"]
        torch.save(checkpoint(initial, 0, "live"), output / "best.pt")
    for epoch in range(epochs):
        started = time.time()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        seen, loss_sum = 0, 0.0
        components = {k: 0.0 for k in ("bce", "dice", "centerline", "weighted_ensemble")}
        for i, batch in enumerate(loader):
            progress = (epoch + i / len(loader)) / epochs
            floor = config.get("lr_floor", 0.05)
            factor = min(1.0, (epoch + i / len(loader) + 0.01) / config.get("warmup_epochs", 3)) * (floor + (1 - floor) * (1 + math.cos(math.pi * progress)) / 2)
            for group in optimizer.param_groups:
                group["lr"] = group["initial_lr"] * factor
            image, target, valid, skeleton = [batch[k].to(device) for k in ("image", "mask", "valid", "skeleton")]
            with autocast(device):
                logits = model(image)
                loss, parts = segmentation_loss(logits, target, valid, skeleton, config.get("centerline_weight", 0.1))
                coupled = ensemble_loss(logits, batch["fixed_context"].to(device), target, valid, config.get("student_weight", 0.14)) if context_weight else logits.new_zeros(())
                loss = loss + context_weight * coupled
                parts["weighted_ensemble"] = context_weight * coupled
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite training loss")
            group_size = min(accumulation, len(loader) - (i // accumulation) * accumulation)
            (loss / group_size).backward()
            n = len(image)
            seen += n
            loss_sum += float(loss.detach()) * n
            for key in components:
                components[key] += float(parts[key].detach()) * n
            if (i + 1) % accumulation == 0 or i + 1 == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                steps += 1
                update_ema(average, model, min(config.get("ema_decay", 0.99), 1 - 1 / (steps + 1)))
        assert seen == len(training)
        metrics, variant = {}, "ema"
        if not full:
            live = evaluate(model, val_loader, device, config.get("threshold", 0.5))
            refresh_bn(average, bn_loader, device, config.get("bn_refresh_batches", 84))
            ema = evaluate(average, val_loader, device, config.get("threshold", 0.5))
            variant, metrics = max((("live", live), ("ema", ema)), key=lambda pair: pair[1]["mIoU"])
            if metrics["mIoU"] > best:
                best, best_epoch = metrics["mIoU"], epoch + 1
                torch.save(checkpoint(metrics, epoch + 1, variant), output / "best.pt")
        row = {"epoch": epoch + 1, "loss": loss_sum / seen, "components": {k: v / seen for k, v in components.items()},
               "images_seen": seen, "seconds": time.time() - started, "validation": metrics, "selected_weights": variant}
        history.append(row)
        (output / "history.json").write_text(json.dumps(history, indent=2))
        torch.save(checkpoint(metrics, epoch + 1, variant), output / "last.pt")
        print(json.dumps(row), flush=True)
        patience = config.get("early_stopping_patience", 0)
        if not full and patience and epoch + 1 >= config.get("early_stopping_min_epochs", 0) and epoch + 1 - best_epoch >= patience:
            break
    if full:
        refresh_bn(average, bn_loader, device, config.get("bn_refresh_batches", 84))
        torch.save(checkpoint({}, epochs, "ema"), output / "final.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.json")
    parser.add_argument("--data-root", required=True, help="TODO: your own dataset root")
    parser.add_argument("--manifest", required=True, help="TODO: replace example image/mask paths")
    parser.add_argument("--output", default="runs/baseline")
    parser.add_argument("--init", help="Optional released full-model checkpoint")
    parser.add_argument("--encoder-weights", help="Optional ImageNet MiT-B0 encoder initialization")
    parser.add_argument("--full", action="store_true", help="Train on train+val for a fixed epoch budget; no validation score")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    train(json.loads(Path(args.config).read_text()), args.data_root, args.manifest, args.output, args.init, args.encoder_weights, args.full, args.device)
