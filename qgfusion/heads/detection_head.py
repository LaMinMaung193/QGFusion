"""
3D Detection Head -- Secondary task, OUTPUT HEADS column.

DETR-style: decode each fused query's Gaussian into a 3D box directly (no NMS needed if
queries are well-separated, matching the proposal's query-based design). Reuses the
Gaussian's own position/scale as box center/size priors, with residual prediction on top --
the standard DETR3D/FUTR3D recipe rather than predicting boxes from scratch.

Risk level (per proposal Sec 7): LOW (architecture). Real training requires a Hungarian
matcher -- see utils/losses.py TODO, that part is the genuinely fiddly piece.
"""
import torch
import torch.nn as nn

from qgfusion.utils.gaussian_utils import GaussianScene


class DetectionHead(nn.Module):
    def __init__(self, feature_dim: int = 128, num_classes: int = 10):
        super().__init__()
        self.cls_head = nn.Linear(feature_dim, num_classes)
        self.center_residual = nn.Linear(feature_dim, 3)  # residual on Gaussian position -> box center
        self.size_residual = nn.Linear(feature_dim, 3)    # residual on log-scale -> box size
        self.yaw_head = nn.Linear(feature_dim, 2)         # (sin, cos) parameterization
        self.velocity_head = nn.Linear(feature_dim, 2)    # (vx, vy), nuScenes convention

    def forward(self, gaussians: GaussianScene) -> dict:
        """
        Returns dict with:
            cls_logits: (B, N, num_classes)
            boxes:      (B, N, 8)  -> (x, y, z, w, l, h, sin_yaw, cos_yaw) -- re-check this
                        ordering against whichever eval toolkit you target; nuscenes-devkit
                        expects a specific box convention, see README > Box convention.
            velocity:   (B, N, 2)
        """
        feats = gaussians.features
        cls_logits = self.cls_head(feats)
        center = gaussians.position + self.center_residual(feats)
        size = torch.exp(self.size_residual(feats).clamp(min=-1, max=2.7))
        yaw_sc = nn.functional.normalize(self.yaw_head(feats), dim=-1)
        velocity = self.velocity_head(feats)

        boxes = torch.cat([center, size, yaw_sc], dim=-1)  # (B, N, 8)
        return {"cls_logits": cls_logits, "boxes": boxes, "velocity": velocity}
