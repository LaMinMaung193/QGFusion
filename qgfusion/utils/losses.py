"""
Loss functions for the three output heads.

Risk level: occupancy/completion losses are LOW risk (standard voxel-wise classification
losses). The detection loss depends on the Hungarian matcher (utils/matcher.py).
"""
import torch
import torch.nn.functional as F


def occupancy_loss(logits, gt, ignore_index: int = -1, weight=None, gamma: float = 2.0):
    """Voxel-wise focal loss for occupancy prediction.
    Focal loss down-weights easy examples (free space) and focuses on hard ones (occupied).
    gamma=0 reduces to standard cross-entropy. gamma=2 is the standard focal loss setting.
    logits: (B, C, X, Y, Z),  gt: (B, X, Y, Z) int64
    weight: (C,) optional per-class weights
    gamma: focal loss exponent (2.0 recommended)
    """
    import torch
    B, C = logits.shape[:2]
    logits_2d = logits.reshape(B, C, -1)     # (B, C, N)
    gt_1d = gt.reshape(B, -1).long()          # (B, N)

    # Standard CE loss per voxel (unreduced)
    ce = F.cross_entropy(logits_2d, gt_1d, weight=weight,
                         ignore_index=ignore_index, reduction='none')  # (B, N)

    # Focal weight: (1 - p_t)^gamma
    with torch.no_grad():
        probs = F.softmax(logits_2d, dim=1)   # (B, C, N)
        gt_clamped = gt_1d.clone()
        gt_clamped[gt_clamped == ignore_index] = 0
        pt = probs.gather(1, gt_clamped.unsqueeze(1)).squeeze(1)  # (B, N)
        focal_weight = (1.0 - pt) ** gamma

    # Zero out ignored positions
    if ignore_index >= 0:
        mask = (gt_1d != ignore_index).float()
        focal_weight = focal_weight * mask
        ce = ce * mask
        n_valid = mask.sum().clamp(min=1)
    else:
        n_valid = torch.tensor(gt_1d.numel(), dtype=torch.float)

    return (focal_weight * ce).sum() / n_valid


def completion_loss(logits, gt, ignore_index: int = -1):
    """Same as occupancy_loss. logits: (B, C, X, Y, Z), gt: (B, X, Y, Z) int64"""
    B, C = logits.shape[:2]
    return F.cross_entropy(
        logits.reshape(B, C, -1),
        gt.reshape(B, -1).long(),
        ignore_index=ignore_index,
    )


def gt_boxes_to_targets(gt_boxes: torch.Tensor, num_boxes: torch.Tensor) -> list:
    """Convert collate_fn's padded (B, max_N, 9) GT box batch + per-sample valid counts into
    the list-of-dicts format HungarianMatcher/detection_loss expect. Column 8 is class_id
    (see nuscenes_dataset.py::_load_boxes), columns 0-7 are the box itself."""
    targets = []
    for b in range(gt_boxes.shape[0]):
        n = int(num_boxes[b].item())
        targets.append({
            "boxes": gt_boxes[b, :n, :8],
            "labels": gt_boxes[b, :n, 8].long(),
        })
    return targets


def detection_loss(preds: dict, targets: list, matcher) -> dict:
    """
    Args:
        preds: output of DetectionHead -> {"cls_logits": (B,N,K), "boxes": (B,N,8), "velocity": (B,N,2)}
        targets: list[dict] of length B, each with "labels" (M,), "boxes" (M,8) --
                 build via gt_boxes_to_targets() from a collated batch
        matcher: a HungarianMatcher instance (utils/matcher.py)
    Returns:
        dict of loss terms: cls_loss, box_loss

    Note: classification loss is computed only on matched pairs -- see the design note in
    matcher.py about the missing "no object" background class for unmatched predictions.
    """
    indices = matcher(preds, targets)
    device = preds["cls_logits"].device

    cls_loss_total = torch.zeros((), device=device)
    box_loss_total = torch.zeros((), device=device)
    num_matched = 0

    for b, (pred_idx, tgt_idx) in enumerate(indices):
        if pred_idx.numel() == 0:
            continue
        pred_idx, tgt_idx = pred_idx.to(device), tgt_idx.to(device)

        matched_logits = preds["cls_logits"][b, pred_idx]
        matched_labels = targets[b]["labels"][tgt_idx].to(device)
        cls_loss_total = cls_loss_total + F.cross_entropy(matched_logits, matched_labels, reduction="sum")

        matched_boxes = preds["boxes"][b, pred_idx]
        matched_gt_boxes = targets[b]["boxes"][tgt_idx].to(device)
        box_loss_total = box_loss_total + F.l1_loss(matched_boxes, matched_gt_boxes, reduction="sum")

        num_matched += pred_idx.numel()

    num_matched = max(num_matched, 1)
    return {
        "cls_loss": cls_loss_total / num_matched,
        "box_loss": box_loss_total / num_matched,
    }


# ---------------------------------------------------------------------------
# Occupancy GT helpers for the two supervised heads
# ---------------------------------------------------------------------------

def downsample_occ_gt(gt, factor=2):
    """Majority-vote (mode) pooling on an integer voxel GT grid.
    Args:
        gt: (B, X, Y, Z) int64 semantic labels
        factor: spatial downsampling factor applied uniformly to all three axes
    Returns:
        (B, X//factor, Y//factor, Z//factor) int64
    """
    import torch
    B, X, Y, Z = gt.shape
    Xo, Yo, Zo = X // factor, Y // factor, Z // factor
    gt_r = gt.reshape(B, Xo, factor, Yo, factor, Zo, factor)
    gt_r = gt_r.permute(0, 1, 3, 5, 2, 4, 6).reshape(B, Xo, Yo, Zo, factor ** 3)
    return torch.mode(gt_r, dim=-1).values


def make_completion_gt(occ_gt):
    """Convert Occ3D semantics (B,200,200,16), values 0-17 → coarse completion GT
    (B,100,100,8), values 0=free, 1=occupied, for SceneCompletionHead."""
    import torch
    coarse = downsample_occ_gt(occ_gt, factor=2)
    out = torch.ones_like(coarse)   # default: occupied
    out[coarse == 17] = 0           # Occ3D class 17 = free/empty
    return out
