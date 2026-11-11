# Release verification

Checks performed on 2026-11-11 using Python 3.11, PyTorch 2.1.0+cu118 and NVIDIA RTX 4090. No hidden test labels were used.  

## Saved model compatibility

Both public architectures loaded every original state tensor with `strict=True`. Their logits matched the corresponding original research implementations exactly on a native task image padded to 384×672, under CUDA BF16.

| Comparison | Images / pixels checked | Differing binary-mask pixels |
|---|---:|---:|
| Baseline versus the actual 81.73% submitted mask archive | 300 / 76,204,800 | **0** |
| Refined member versus its original probability cache, threshold 0.50 | 300 / 76,204,800 | **0** |

Baseline probabilities matched exactly, using historical padded-image TTA, batch size 1 and threshold 0.51. Refined-member probabilities used native-before-padding TTA and batch size 4. Their maximum absolute cache difference was 0.0009989143, with zero changed binary pixels at threshold 0.50. This is a component check, not reconstruction of the seven-model 82.83% result. Floating-point/kernel differences can affect threshold-border pixels in another environment.

## Code tests

Run `python -m pytest tests -q`. The eight tests cover:

- Native refinement output size and exact identity at zero initialization.
- Loss invariance to bottom/right padding and zero gradients in invalid pixels.
- Soft-skeleton bounds, binary-target enforcement and missing-skeleton errors.
- Student gradients with detached frozen-model predictions and detached target supervision.
- Joint image/mask/context flips, validity regions and batch collation.
- Rejection of overlapping training/validation source groups.
- Flip-TTA inversion and pooled two-class metric definitions.
- A real one-epoch training/checkpoint/prediction round trip and an all-label refit smoke test with no held-out score.

 