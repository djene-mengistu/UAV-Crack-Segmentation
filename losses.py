"""Binary BCE + soft Dice + optional topology supervision, excluding padding."""
import torch
from torch.nn import functional as F


def soft_skeleton(probability, iterations=5):
    p = probability.float().clamp(0, 1)

    def erode(x):
        return torch.minimum(-F.max_pool2d(-x, (3, 1), 1, (1, 0)), -F.max_pool2d(-x, (1, 3), 1, (0, 1)))

    def opened(x):
        return F.max_pool2d(erode(x), 3, 1, 1)

    skeleton = F.relu(p - opened(p)).clamp(0, 1)
    for _ in range(iterations):
        p = erode(p)
        delta = F.relu(p - opened(p))
        skeleton = (skeleton + (1 - skeleton) * delta).clamp(0, 1)
    return skeleton


def centerline_loss(probability, target, valid, skeleton):
    if skeleton is None:
        raise ValueError("Provide a binary GT skeleton when centerline_weight > 0")
    # Morphology is computed on the unpadded rectangle, never across padding.
    per_image = []
    for p, y, v, sk in zip(probability, target, valid, skeleton):
        positions = v[0].bool().nonzero()
        if not len(positions):
            per_image.append(p.sum() * 0)
            continue
        top, left = positions.amin(0).tolist()
        bottom, right = (positions.amax(0) + 1).tolist()
        p, y, v, sk = [a[None, :, top:bottom, left:right] for a in (p, y, v, sk)]
        if not bool(v.bool().all()):
            raise ValueError("Centerline morphology supports rectangular valid regions, not partial annotations")
        if bool(((y != 0) & (y != 1)).any()) or bool(((sk != 0) & (sk != 1)).any()):
            raise ValueError("Centerline targets and skeletons must be binary")
        predicted_skeleton = soft_skeleton(p)
        precision = ((predicted_skeleton * y).sum() + 1e-6) / (predicted_skeleton.sum() + 1e-6)
        recall = ((sk * p).sum() + 1e-6) / (sk.sum() + 1e-6)
        per_image.append(1 - (2 * precision * recall + 1e-6) / (precision + recall + 1e-6))
    return torch.stack(per_image).mean()


def segmentation_loss(logits, target, valid, skeleton=None, centerline_weight=0.1):
    logits, target, valid = logits.float(), target.float(), valid.float()
    p = logits.sigmoid()
    dims = (1, 2, 3)
    bce = ((F.binary_cross_entropy_with_logits(logits, target, reduction="none") * valid).sum(dims) / valid.sum(dims).clamp_min(1)).mean()
    dice = (1 - (2 * (p * target * valid).sum(dims) + 1) / ((p * valid).sum(dims) + (target * valid).sum(dims) + 1)).mean()
    topology = centerline_loss(p, target, valid, skeleton) if centerline_weight else logits.sum() * 0
    return bce + dice + centerline_weight * topology, {"bce": bce, "dice": dice, "centerline": topology}


def ensemble_loss(logits, fixed_context, target, valid, student_weight=0.14):
    """Optional frozen-other-model objective. True labels supervise the mixture.

    fixed_context = sum(w_i * p_i) for OTHER models, with sum(w_i)=1-student_weight.
    Cached maps must undergo the same spatial transformations as RGB and labels.
    This is an extension hook; the minimal supervised trainer does not need it.
    """
    if not 0 < student_weight <= 1:
        raise ValueError("student_weight must be in (0,1]")
    q = (fixed_context.detach().float() + student_weight * logits.float().sigmoid()).clamp(1e-6, 1 - 1e-6)
    y, v = target.detach().float(), valid.float()
    dims = (1, 2, 3)
    bce = (-(y * q.log() + (1 - y) * torch.log1p(-q)) * v).sum(dims) / v.sum(dims).clamp_min(1)
    dice = 1 - (2 * (q * y * v).sum(dims) + 1) / ((q * v).sum(dims) + (y * v).sum(dims) + 1)
    return (bce + dice).mean()
