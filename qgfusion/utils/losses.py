"""
Loss functions for the three output heads.

Risk level: occupancy/completion losses are LOW risk (standard voxel-wise classification
losses). The detection Hungarian matcher is the fiddly part -- flagged below.
"""
import torch.nn.functional as F


def occupancy_loss(logits, gt, ignore_index: int = -1):
    """Voxel-wise cross-entropy. TODO: swap for focal loss if class imbalance dominates --
    occupancy grids are typically >90% empty space, plain CE will under-weight rare classes."""
    return F.cross_entropy(logits, gt, ignore_index=ignore_index)


def completion_loss(logits, gt, ignore_index: int = -1):
    return F.cross_entropy(logits, gt, ignore_index=ignore_index)


def detection_loss(preds: dict, targets: list, matcher=None) -> dict:
    """
    Args:
        preds: output of DetectionHead -> {"cls_logits": (B,N,K), "boxes": (B,N,8), "velocity": (B,N,2)}
        targets: list[dict] of length B, each with "labels" (M,), "boxes" (M,8), "velocity" (M,2)
        matcher: Hungarian matcher instance (TODO below)
    Returns:
        dict of loss terms
    """
    if matcher is None:
        raise NotImplementedError(
            "TODO(risk=MEDIUM, fiddly): implement a Hungarian matcher -- "
            "scipy.optimize.linear_sum_assignment over a cost matrix of "
            "(classification cost + L1 box cost + 3D-IoU/GIoU cost) -- to assign each of the "
            "N predicted Gaussians to a GT box before computing per-pair losses. This is the "
            "standard DETR/FUTR3D recipe; FUTR3D's matcher implementation is the closest prior "
            "art to this design (per proposal Sec 4.2) and worth using as a reference rather "
            "than building from scratch."
        )
    # matched_preds, matched_targets = matcher(preds, targets)
    # cls_loss = F.cross_entropy(matched_preds["cls_logits"], matched_targets["labels"])
    # box_loss = F.l1_loss(matched_preds["boxes"], matched_targets["boxes"])
    # vel_loss = F.l1_loss(matched_preds["velocity"], matched_targets["velocity"])
    # return {"cls_loss": cls_loss, "box_loss": box_loss, "vel_loss": vel_loss}
