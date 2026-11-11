# Submission description

## Short version: paste into the submission page

We use a heterogeneous ensemble of MiT-B0/UPer, ConvNeXt and ResNet-based crack segmentation models. A native-resolution RGB-conditioned residual head refines the MiT predictions. Training combines binary cross-entropy, Dice and centerline connectivity supervision, with an additional ground-truth-supervised ensemble objective for refinement. Earlier ensemble members use transductive cross-supervision and consistency learning with unlabeled submission images. Inference averages four flip views at native resolution, combines model probabilities and applies a calibrated threshold. The full system achieves 82.83% mIoU on UAV-CrackX.

## Optional implementation details

The refined MiT-B0 uses first-stage stride 2, a 128-channel UPer decoder and a zero-initialized RGB/logit residual head. Development uses 1000 labeled training and 200 capture-group-disjoint validation images, followed by a fixed two-epoch supervised refit on all 1200 labels. Refinement minimizes BCE + Dice + 0.1 soft-clDice, plus 0.5 BCE+Dice on an ensemble probability with detached other-model contributions and true labeled targets. The final refit uses supervised connectivity loss only. The ensemble includes a DINOv3-pretrained ConvNeXt member and two ResNet34 U-Net members. Inference uses identity/horizontal/vertical/both-flip averaging, replicated padding, native-size cropping and threshold 0.44.

## Code-link wording, after uploading

Code and two MiT checkpoints: https://github.com/djene-mengistu/UAV-Crack-Segmentation

This URL is the intended repository location. It is not published by this local preparation step; use it in the submission only after pushing. The release reproduces a standalone 81.73% MiT model and provides the refined MiT component of the 82.83% ensemble. It does not label a single model as achieving 82.83%.
