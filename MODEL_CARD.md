# Model card

## Released checkpoints

| File | Role | Official mIoU | Training history |
|---|---|---:|---|
| `checkpoints/mit_b0_standalone.pt` | MiT-B0/UPer-128 baseline | 81.73% | EMA epoch 52, 934 labeled training / 266 validation images, no all-label refit |
| `checkpoints/mit_b0_native_refined.pt` | Refined MiT member, 14% weight in the best ensemble | No standalone submission | Inherits co-trained MiT weights; 1000/200 development split; selected refinement epoch 11; then 2 supervised refit epochs on all 1200 labels |

The original single-model official result was reported by the author from platform feedback: mIoU 81.73, crack F1 79.48, crack IoU 65.95, precision 80.50, recall 78.49, accuracy 97.62 (all percentages). Its original submitted archive SHA256 is `3060879227ac7f86de78c45526b51135d7be1bddb8ada5f2cbd71a3789109305`.

The refined member has no measured official single-model score. Its default threshold 0.50 is an inference convention, not a claimed optimal standalone threshold. It participated in the 82.83% ensemble. No hidden test labels were available for local evaluation.

The checkpoint exports preserve every state tensor, remove private paths/configuration histories and include only architecture, inference settings and safe provenance. Original source checkpoint SHA256 values:

- Baseline: `a018c58430fee6691e397df8c3e9c25a6ce9b4f7b14e3bb92baae1be9a7e3edd`.
- Refined: `32971901f34f09304f9be93135a789691ef89449ec7dc9d3ba8e695ce20b905d`.

Exported files have different hashes because their metadata/serialization differ. Use `checkpoints/SHA256SUMS` to verify the actual released files. Loading uses `weights_only=True` and strict architecture checks.

## Full submission context

The [djene-XDU-V2 entry](http://vlp.chd.edu.cn/tasks/method/327) was recorded at 82.83% mIoU, 80.96 F1, 68.00 crack IoU, 80.85 precision, 81.06 recall and 97.76 accuracy. A public leaderboard check on 2026-11-11 at 04:01 UTC recorded the highest displayed mIoU at that time; rankings can change.

| Full ensemble member | Probability weight |
|---|---:|
| Earlier MiT-B0 | 0.14 |
| Earlier ConvNeXt | 0.14 |
| Recent ConvNeXt | 0.14 |
| Updated MiT-B0 with native RGB/logit refinement | 0.14 |
| DINOv3-pretrained ConvNeXt-S | 0.24 |
| Earlier ResNet34 U-Net refit | 0.15 |
| Fresh crop-trained ResNet34 U-Net refit | 0.05 |

Each member uses native four-flip averaging; weighted probabilities are thresholded at 0.44. The exact winning archive SHA256 is `2e66852112e1f17c97773726f1a49c535d970d6cbd97602b9a9a9f2b7d1c06cf`. The full seven checkpoints, ensemble prediction caches and historical orchestration are not included in this deliberately compact release.

## Training protocol and limitations

The refined member's development learning rates are 2e-6 encoder, 1e-5 decoder and 2e-4 refinement. All components train. BCE+Dice+0.1 soft-clDice is combined with 0.5 times ensemble BCE+Dice against true labels. Frozen predictions from other models provide additive ensemble context; there is no backpropagation into those teachers. Final all-label refit uses 6e-7/3e-6/3e-5 learning rates and supervised connectivity loss only, for two epochs.

Earlier MiT/ConvNeXt members were trained with cross-supervision/consistency and confident pseudo-label information from unlabeled submission images. The final system is therefore **transductive**, including inherited weights, even though the last refinement/refit uses real labeled data. This must be disclosed when comparing with strictly supervised methods or benchmarks that prohibit transductive use.

The same 200 development samples were used repeatedly to select experiments. Tiny leaderboard gains are reported platform measurements, not evidence of statistical significance or isolated causal benefits. Controlled retraining, independent validation and ablations are needed for a manuscript.

The included trainer offers a clean supervised recipe and an optional fixed-context ensemble-loss hook. It does not include the historical CPS pipeline, exact original split lists, parent co-training initialization or all ensemble teachers. The supervised configs are examples, not exact from-scratch reproduction claims. The baseline's original inference uses padded-image flip geometry; the refined model uses native-before-padding geometry. These profiles deliberately remain different to preserve original predictions.

Images and masks remain subject to the dataset provider's terms. The model is intended for research crack segmentation on UAV imagery; its predictions have not been validated as structural safety assessments.
