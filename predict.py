"""Predict individual images or a manifest split, optionally evaluating real masks."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from dataset import image_tensor, read_manifest
from model import build_model
from utils import pixel_counts, predict_probability, scores


def predict(checkpoint_path, images=None, manifest_path=None, split="test", root=".", output="predictions", threshold=None, device="cuda", no_tta=False, batch_size=None):
    device = torch.device(device)
    torch.set_num_threads(4)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = build_model(**checkpoint["model_config"]).to(device).eval()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    profile = checkpoint.get("inference", {})
    threshold = profile.get("threshold", 0.5) if threshold is None else threshold
    geometry = profile.get("geometry", "native")
    batch_size = profile.get("batch_size", 1) if batch_size is None else batch_size
    if batch_size < 1 or not 0 <= threshold <= 1:
        raise ValueError("batch_size must be positive and threshold must be in [0,1]")
    if manifest_path:
        records = read_manifest(manifest_path)[split]
    elif images:
        directory = Path(images)
        records = [{"image": str(p.resolve())} for p in sorted(directory.rglob("*")) if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")]
    else:
        raise ValueError("Provide --images or --manifest")
    if not records:
        raise ValueError("No input images found")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    total = dict.fromkeys(("tp", "fp", "fn", "tn"), 0)
    evaluated = 0
    names = set()
    for offset in range(0, len(records), batch_size):
        rows = records[offset:offset + batch_size]
        images_in_batch, output_names = [], []
        for row in rows:
            path = Path(root) / row["image"]
            name = row.get("output_name", path.stem + ".png")
            if Path(name).name != name or Path(name).suffix.lower() != ".png" or name in names:
                raise ValueError("output_name must be a unique PNG basename; add it explicitly for repeated image stems")
            names.add(name)
            output_names.append(name)
            images_in_batch.append(image_tensor(np.asarray(Image.open(path).convert("RGB"))))
        if len({tuple(x.shape) for x in images_in_batch}) != 1:
            raise ValueError("Inference batches require equal native sizes; use --batch-size 1 for mixed-size input")
        with torch.no_grad():
            probabilities = predict_probability(model, torch.stack(images_in_batch).to(device), not no_tta, geometry)[:, 0].cpu()
        for row, name, probability in zip(rows, output_names, probabilities):
            Image.fromarray((probability.numpy() >= threshold).astype(np.uint8)).save(output / name)
            if "mask" in row:
                raw = np.asarray(Image.open(Path(root) / row["mask"]).convert("L"))
                if raw.shape != probability.shape or not np.isin(raw, [0, 1, 255]).all():
                    raise ValueError("Evaluation mask must be a matching-size binary mask")
                counts = pixel_counts(probability, torch.from_numpy(raw > 0), threshold)
                for key in total:
                    total[key] += counts[key]
                evaluated += 1
    report = {"images": len(records), "evaluated_images": evaluated, "threshold": threshold, "geometry": geometry,
              "tta": not no_tta, "batch_size": batch_size, "metrics": scores(total) if evaluated else None, "metric_unit": "fraction", "encoding": "grayscale 0/1"}
    (output / "prediction_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--images", help="Folder of RGB images")
    parser.add_argument("--manifest", help="Optional JSON manifest with image/mask records")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--data-root", default=".")
    parser.add_argument("--output", default="predictions")
    parser.add_argument("--threshold", type=float, help="Default comes from checkpoint; 0.44 is an ensemble threshold, not a measured single-model optimum")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-tta", action="store_true")
    parser.add_argument("--batch-size", type=int, help="Default from checkpoint: 1 baseline, 4 refined; changes can alter BF16 rounding")
    args = parser.parse_args()
    predict(args.checkpoint, args.images, args.manifest, args.split, args.data_root, args.output, args.threshold, args.device, args.no_tta, args.batch_size)
