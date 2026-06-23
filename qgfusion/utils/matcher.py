"""
Hungarian matcher for the detection head -- standard DETR/FUTR3D-style bipartite matching
between predicted Gaussians and GT boxes, consumed by detection_loss() in losses.py.

Cost combines classification cost (negative predicted probability of the GT class) and L1
box regression cost (center + size + orientation), matching the convention used by FUTR3D
and most DETR3D-family detectors -- FUTR3D is the closest prior art to this design's
query-based fusion (per proposal Sec 4.2).

Design note (flagging deliberately, not hiding it): this first-pass version does NOT use an
explicit "no object" / background class for unmatched predictions, unlike standard DETR.
Adding one would mean extending DetectionHead.cls_head from num_classes to num_classes+1
outputs -- an architecture change, not just a loss-function change. For now, classification
loss is only computed on matched pairs (see detection_loss in losses.py), which means
unmatched Gaussians aren't explicitly penalized for false positives. This is a real scope
decision worth revisiting once basic training is confirmed to flow correctly, not an
oversight.
"""
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment


class HungarianMatcher(nn.Module):
    def __init__(self, cost_class: float = 1.0, cost_bbox: float = 2.5):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox

    @torch.no_grad()
    def forward(self, preds: dict, targets: list) -> list:
        """
        Args:
            preds: {"cls_logits": (B,N,K), "boxes": (B,N,8), "velocity": (B,N,2)}
            targets: list[dict], length B, each with "labels": (M,) long, "boxes": (M,8) float
                     (use gt_boxes_to_targets() in losses.py to build this from a collated batch)
        Returns:
            list of (pred_idx, tgt_idx) index tensors, one pair per batch element -- the
            optimal bipartite assignment between predicted Gaussians and GT boxes under the
            combined cost.
        """
        B = preds["cls_logits"].shape[0]
        cls_probs = preds["cls_logits"].softmax(-1)  # (B, N, K)

        indices = []
        for b in range(B):
            tgt_labels = targets[b]["labels"]
            tgt_boxes = targets[b]["boxes"]
            M = tgt_labels.shape[0]

            if M == 0:
                indices.append((torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)))
                continue

            cost_class = -cls_probs[b][:, tgt_labels]  # (N, M)
            # Normalize center (cols 0:3) and size (cols 3:6) separately so
            # large metric values don't dominate the classification cost.
            pred_ctr = preds["boxes"][b][:, :3] / 40.0   # pc_range half-width
            tgt_ctr  = tgt_boxes[:, :3] / 40.0
            pred_sz  = preds["boxes"][b][:, 3:6] / 10.0  # max realistic object size
            tgt_sz   = tgt_boxes[:, 3:6] / 10.0
            pred_norm = torch.cat([pred_ctr, pred_sz, preds["boxes"][b][:, 6:]], dim=-1)
            tgt_norm  = torch.cat([tgt_ctr,  tgt_sz,  tgt_boxes[:, 6:]], dim=-1)
            cost_bbox = torch.cdist(pred_norm, tgt_norm, p=1)  # (N, M)
            cost = (self.cost_class * cost_class + self.cost_bbox * cost_bbox).cpu()

            pred_idx, tgt_idx = linear_sum_assignment(cost)
            indices.append((torch.as_tensor(pred_idx, dtype=torch.long), torch.as_tensor(tgt_idx, dtype=torch.long)))

        return indices
