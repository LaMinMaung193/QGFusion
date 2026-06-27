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

        # Learned reference points — each query starts from a different position
        # in normalized [-1, 1] space, then scaled to pc_range during forward
        self.pos_anchor = nn.Parameter(torch.randn(1, 1, 3) * 0.5)
        # Per-query offset from anchor (different for each of N queries)
        self.pos_head = head(3)        # (x, y, z) residual on anchor
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
        # Each query gets a unique position: learned anchor + MLP residual
        # Anchor is broadcast across batch; residual varies per query via Qf
        B, N, _ = Qf.shape
        # Create N evenly-spaced reference points in BEV
        idx = torch.arange(N, device=Qf.device).float()
        # Distribute queries across X-Y plane in a grid pattern
        grid_w = int(N ** 0.5) + 1
        ref_x = ((idx % grid_w) / grid_w - 0.5) * 2   # [-1, 1]
        ref_y = ((idx // grid_w) / grid_w - 0.5) * 2  # [-1, 1]
        ref_z = torch.zeros_like(ref_x)
        refs = torch.stack([ref_x, ref_y, ref_z], dim=-1).unsqueeze(0)  # (1, N, 3)
        # Scale to pc_range: X in [-40,40], Y in [-40,40], Z in [-1,5.4]
        scale = torch.tensor([40.0, 40.0, 3.2], device=Qf.device)
        refs = refs * scale  # (1, N, 3)
        position = refs.expand(B, -1, -1) + self.pos_head(Qf) * 5.0
        scale = torch.exp(self.scale_head(Qf).clamp(min=-1, max=2))
        rotation = nn.functional.normalize(self.rot_head(Qf), dim=-1)
        opacity = torch.sigmoid(self.opacity_head(Qf)).squeeze(-1) * 0.9 + 0.05
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
