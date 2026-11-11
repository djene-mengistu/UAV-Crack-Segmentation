# Checkpoint license and pretrained lineage

The two checkpoints in `checkpoints/` descend from MiT-B0 ImageNet pretrained weights from NVIDIA's SegFormer lineage, reused through the author's ProMSC architecture. They are distributed for **non-commercial research or evaluation** under the applicable NVIDIA SegFormer license, reproduced in full in [licenses/SegFormer.txt](licenses/SegFormer.txt).

The source license's use limitation and redistribution requirements continue to apply to pretrained-derived weights. The repository's MIT license for original code does not grant broader rights to these checkpoints. NVIDIA retains its upstream rights. The new decoder/refinement training does not remove the upstream limitation.

Upstream source: https://github.com/NVlabs/SegFormer

License source: https://github.com/NVlabs/SegFormer/blob/master/LICENSE

No dataset images, annotations or third-party pretrained download archives are bundled. Obtain those separately under their provider's terms.
