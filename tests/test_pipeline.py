import json
from unittest.mock import patch
import numpy as np
from PIL import Image
import pytest
import torch
from torch.nn import functional as F
from dataset import CrackDataset, collate_samples, read_manifest
from losses import segmentation_loss, ensemble_loss, soft_skeleton
from model import build_model
from utils import predict_probability, pixel_counts, scores


def test_refinement_starts_as_identity():
    torch.set_num_threads(2)
    model = build_model(refinement=True, decoder_width=32).eval()
    x = torch.randn(1, 3, 64, 96)
    with torch.no_grad():
        base, refined = model.base(x), model(x)
    assert refined.shape == (1, 1, 64, 96)
    torch.testing.assert_close(base, refined, rtol=0, atol=0)


def test_padding_has_no_loss_or_gradient():
    torch.manual_seed(1)
    x = torch.randn(1, 1, 17, 25, requires_grad=True)
    y = torch.zeros_like(x)
    y[:, :, 3:14, 12] = 1
    valid = torch.ones_like(y)
    reference, _ = segmentation_loss(x, y, valid, y)
    padded = F.pad(x.detach(), (0, 7, 0, 15), value=4).requires_grad_(True)
    actual, _ = segmentation_loss(padded, F.pad(y, (0, 7, 0, 15)), F.pad(valid, (0, 7, 0, 15)), F.pad(y, (0, 7, 0, 15)))
    torch.testing.assert_close(reference, actual)
    actual.backward()
    assert padded.grad[:, :, 17:].abs().sum() == 0
    assert padded.grad[:, :, :, 25:].abs().sum() == 0
    assert padded.grad[:, :, :17, :25].abs().sum() > 0


def test_skeleton_bounds_and_target_guard():
    sk = soft_skeleton(torch.randn(2, 1, 16, 24) * 3)
    assert sk.min() >= 0 and sk.max() <= 1
    x = torch.zeros(1, 1, 16, 24)
    with pytest.raises(ValueError, match="binary"):
        segmentation_loss(x, x + 0.5, torch.ones_like(x), x)
    with pytest.raises(ValueError, match="skeleton"):
        segmentation_loss(x, x, torch.ones_like(x))


def test_other_model_and_targets_are_detached():
    logits = torch.randn(2, 1, 8, 12, requires_grad=True)
    fixed = torch.full_like(logits, 0.3, requires_grad=True)
    target = torch.ones_like(logits, requires_grad=True)
    ensemble_loss(logits, fixed, target, torch.ones_like(logits)).backward()
    assert logits.grad.abs().sum() > 0
    assert fixed.grad is None and target.grad is None


def test_loader_alignment_and_batch_padding(tmp_path):
    rgb = np.zeros((17, 25, 3), np.uint8)
    mask = np.zeros((17, 25), np.uint8)
    rgb[2:12, 3, 0] = 255
    mask[2:12, 3] = 1
    Image.fromarray(rgb).save(tmp_path / "image.png")
    Image.fromarray(mask).save(tmp_path / "mask.png")
    np.save(tmp_path / "context.npy", mask.astype(np.float32) * 0.7)
    rows = [{"image": "image.png", "mask": "mask.png", "fixed_context": "context.npy"}]
    with patch("dataset.random.random", side_effect=[0.1, 0.1, 0.99]):
        item = CrackDataset(tmp_path, rows, training=True, use_context=True)[0]
    expected = torch.from_numpy(mask[::-1, ::-1].copy()).float()
    torch.testing.assert_close(item["mask"][0, :17, :25], expected)
    torch.testing.assert_close(item["fixed_context"][0, :17, :25], expected * 0.7)
    assert item["valid"].sum() == 17 * 25
    assert item["valid"][0, 17:].sum() == 0
    assert collate_samples([item, item])["image"].shape == (2, 3, 32, 32)


def test_split_leakage_is_rejected(tmp_path):
    file = tmp_path / "manifest.json"
    file.write_text(json.dumps({"train": [{"image": "a", "group": "g"}], "val": [{"image": "b", "group": "g"}]}))
    with pytest.raises(ValueError, match="groups overlap"):
        read_manifest(file)


def test_tta_geometry_and_metrics():
    class Identity(torch.nn.Module):
        def forward(self, x):
            return x[:, :1]
    x = torch.randn(1, 3, 17, 25)
    for geometry in ("native", "padded"):
        p = predict_probability(Identity(), x, geometry=geometry)
        torch.testing.assert_close(p, x[:, :1].sigmoid())
    counts = pixel_counts(torch.tensor([0.8, 0.7, 0.2, 0.1]), torch.tensor([1, 0, 1, 0]))
    assert counts == {"tp": 1, "fp": 1, "fn": 1, "tn": 1}
    assert scores(counts)["mIoU"] == pytest.approx(1 / 3)


def test_train_predict_roundtrip(tmp_path):
    from train import train
    from predict import predict
    rows = []
    for i in range(4):
        image = np.full((32, 32, 3), i * 40 + 30, np.uint8)
        mask = np.zeros((32, 32), np.uint8)
        mask[4:28, 14:17] = 1
        Image.fromarray(image).save(tmp_path / f"rgb{i}.png")
        Image.fromarray(mask).save(tmp_path / f"gt{i}.png")
        rows.append({"image": f"rgb{i}.png", "mask": f"gt{i}.png", "group": str(i)})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"train": rows[:2], "val": rows[2:]}))
    config = {"model": {"refinement": False, "decoder_width": 32, "first_stride": 2}, "epochs": 1,
              "batch_size": 2, "workers": 0, "lr": 1e-4, "encoder_lr": 1e-5, "bn_refresh_batches": 1}
    train(config, tmp_path, manifest, tmp_path / "run", device="cpu")
    report = predict(tmp_path / "run/best.pt", manifest_path=manifest, split="val", root=tmp_path, output=tmp_path / "out", device="cpu")
    assert report["images"] == report["evaluated_images"] == 2
    assert set(np.unique(Image.open(tmp_path / "out/rgb2.png"))) <= {0, 1}
    # Exercise baseline-to-refiner initialization and optional frozen context.
    np.save(tmp_path / "context.npy", np.full((32, 32), 0.4, np.float32))
    for row in rows[:2]:
        row["fixed_context"] = "context.npy"
    manifest.write_text(json.dumps({"train": rows[:2], "val": rows[2:]}))
    refined_config = {**config, "model": {**config["model"], "refinement": True}, "ensemble_weight": 0.5}
    train(refined_config, tmp_path, manifest, tmp_path / "refined", init=tmp_path / "run/best.pt", device="cpu")
    assert torch.load(tmp_path / "refined/last.pt", weights_only=True)["model_config"]["refinement"]
    # Also exercise the explicit all-label stage: no held-out metrics are claimed.
    train(config, tmp_path, manifest, tmp_path / "full", full=True, device="cpu")
    final = torch.load(tmp_path / "full/final.pt", weights_only=True)
    assert final["metrics"] == {}
    assert json.loads((tmp_path / "full/provenance.json").read_text())["validation_images"] == []
