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
        # Distribute queries in 3D: X-Y grid + Z levels
        n_z = 4  # number of height levels
        n_xy = N // n_z
        grid_w = int(n_xy ** 0.5) + 1
        idx_xy = idx % n_xy
        idx_z  = (idx // n_xy).clamp(max=n_z - 1)
        ref_x = ((idx_xy % grid_w) / grid_w - 0.5) * 2
        ref_y = ((idx_xy // grid_w) / grid_w - 0.5) * 2
        ref_z = (idx_z / (n_z - 1) - 0.5) * 2  # [-1, 1] across Z levels
        refs = torch.stack([ref_x, ref_y, ref_z], dim=-1).unsqueeze(0)  # (1, N, 3)
        # Scale to pc_range
        scale_xyz = torch.tensor([40.0, 40.0, 3.2], device=Qf.device)
        refs = refs * scale_xyz  # (1, N, 3)
        position = refs.expand(B, -1, -1) + self.pos_head(Qf) * 5.0
        # Clamp to pc_range to keep Gaussians inside the scene
        pc_min = torch.tensor([-40.0, -40.0, -1.0], device=Qf.device)
        pc_max = torch.tensor([ 40.0,  40.0,  5.4], device=Qf.device)
        position = torch.clamp(position, pc_min, pc_max)
        # Softplus ensures scale > 0 without hard clamp saturation
        # beta=1, threshold=20: smooth for small values, linear for large
        scale_raw = self.scale_head(Qf)  # (B, N, 3)
        scale = torch.nn.functional.softplus(scale_raw, beta=2.0) + 0.5
        scale = scale.clamp(max=5.0)  # max 5m per axis
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
