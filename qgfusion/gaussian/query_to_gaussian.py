"""
Query-to-Gaussian Generator -- Stage 4 (GAUSSIAN REPR. column in diagram)

Implements Option B from the proposal (recommended): DETR-style prediction head with a
separate sub-head per Gaussian attribute, for more stable training than a single shared MLP.

Risk level (per proposal Sec 7): MEDIUM -- "Gaussian parameter regression"
"""
import torch
import torch.nn as nn

from qgfusion.utils.gaussian_utils import GaussianScene


class QueryToGaussianGenerator(nn.Module):
    def __init__(self, embed_dim: int = 256, feature_dim: int = 128, predict_velocity: bool = False):
        super().__init__()
        self.predict_velocity = predict_velocity

        def head(out_dim):
            return nn.Sequential(
                nn.Linear(embed_dim, embed_dim), nn.ReLU(inplace=True), nn.Linear(embed_dim, out_dim)
            )

        self.pos_head = head(3)        # (x, y, z)
        self.scale_head = head(3)      # log-scale; exp() applied below
        self.rot_head = head(4)        # quaternion (w, x, y, z); normalized below
        self.opacity_head = head(1)    # raw logit; sigmoid applied below
        self.feature_head = head(feature_dim)
        if predict_velocity:
            self.velocity_head = head(3)

    def forward(self, Qf: torch.Tensor) -> GaussianScene:
        """
        Args:
            Qf: (B, N, embed_dim) fused scene queries
        Returns:
            GaussianScene with N Gaussians per batch element.
        """
        position = self.pos_head(Qf)
        scale = torch.exp(self.scale_head(Qf).clamp(max=10))
        rotation = nn.functional.normalize(self.rot_head(Qf), dim=-1)
        opacity = torch.sigmoid(self.opacity_head(Qf)).squeeze(-1)
        features = self.feature_head(Qf)
        velocity = self.velocity_head(Qf) if self.predict_velocity else None

        return GaussianScene(
            position=position,
            scale=scale,
            rotation=rotation,
            opacity=opacity,
            features=features,
            velocity=velocity,
        )
