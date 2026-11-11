"""Generic image/mask loader. Adapt the example manifest to your dataset paths."""
import json
from pathlib import Path
import random
import numpy as np
from PIL import Image, ImageEnhance
from skimage.morphology import skeletonize
import torch
from torch.nn import functional as F

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def read_manifest(path):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    train_groups = {r["group"] for r in manifest.get("train", []) if "group" in r}
    val_groups = {r["group"] for r in manifest.get("val", []) if "group" in r}
    if train_groups & val_groups:
        raise ValueError("Training and validation source groups overlap")
    train_images = {r["image"] for r in manifest.get("train", [])}
    if train_images & {r["image"] for r in manifest.get("val", [])}:
        raise ValueError("Training and validation images overlap")
    return manifest


def image_tensor(rgb):
    image = torch.from_numpy(np.asarray(rgb).copy()).permute(2, 0, 1).float() / 255
    return (image - image.new_tensor(MEAN)[:, None, None]) / image.new_tensor(STD)[:, None, None]


class CrackDataset(torch.utils.data.Dataset):
    def __init__(self, root, records, training=False, use_context=False, student_weight=0.14):
        self.root, self.records, self.training = Path(root), records, training
        self.use_context, self.student_weight = use_context, student_weight

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        image = np.asarray(Image.open(self.root / row["image"]).convert("RGB")).copy()
        if "mask" not in row:
            raise ValueError("Training/validation records need a mask; use predict.py for unlabeled images")
        raw = np.asarray(Image.open(self.root / row["mask"]).convert("L"))
        if not np.isin(raw, [0, 1, 255]).all():
            raise ValueError("Masks must be binary, encoded as 0/1 or 0/255")
        mask = (raw > 0).astype(np.float32)
        if image.shape[:2] != mask.shape:
            raise ValueError("RGB and mask sizes differ; do not resize masks implicitly")
        skeleton = skeletonize(mask > 0).astype(np.float32)
        fixed = None
        if self.use_context:
            if "fixed_context" not in row:
                raise ValueError("Ensemble supervision requires a fixed_context .npy path for every training record")
            fixed = np.asarray(np.load(self.root / row["fixed_context"], allow_pickle=False), dtype=np.float32)
            if fixed.shape != mask.shape or not np.isfinite(fixed).all() or fixed.min() < 0 or fixed.max() > 1 - self.student_weight + 1e-6:
                raise ValueError("fixed_context must be a native-size finite weighted sum of OTHER model probabilities")
        if self.training:
            for axis in (0, 1):
                if random.random() < 0.5:
                    image, mask, skeleton = [np.flip(a, axis).copy() for a in (image, mask, skeleton)]
                    if fixed is not None:
                        fixed = np.flip(fixed, axis).copy()
            if random.random() < 0.8:
                pil = ImageEnhance.Brightness(Image.fromarray(image)).enhance(random.uniform(0.8, 1.2))
                image = np.asarray(ImageEnhance.Contrast(pil).enhance(random.uniform(0.8, 1.2)))
        height, width = mask.shape
        pad = (0, (-width) % 32, 0, (-height) % 32)

        def padded(a):
            return F.pad(torch.from_numpy(a.copy())[None], pad)

        result = {"image": F.pad(image_tensor(image), pad, mode="replicate"), "mask": padded(mask),
                "skeleton": padded(skeleton), "valid": padded(np.ones_like(mask)),
                "native_size": torch.tensor([height, width]), "index": index}
        if fixed is not None:
            result["fixed_context"] = padded(fixed)
        return result


def collate_samples(samples):
    """Keep native coordinates when images need different amounts of padding."""
    height = max(s["image"].shape[-2] for s in samples)
    width = max(s["image"].shape[-1] for s in samples)
    output = {}
    for key in samples[0]:
        values = []
        for sample in samples:
            value = sample[key]
            if key in ("image", "mask", "skeleton", "valid", "fixed_context"):
                pad = (0, width - value.shape[-1], 0, height - value.shape[-2])
                value = F.pad(value, pad, mode="replicate") if key == "image" else F.pad(value, pad)
            values.append(value)
        output[key] = torch.stack(values) if isinstance(values[0], torch.Tensor) else torch.tensor(values)
    return output
