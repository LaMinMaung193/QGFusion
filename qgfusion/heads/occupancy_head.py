"""
Occupancy Head -- Primary task, OUTPUT HEADS column.

Implements voxel rasterization with anisotropic Gaussian splatting.
Each voxel accumulates features from nearby Gaussians weighted by:
  w = opacity * exp(-0.5 * sum((dx/sx)^2 + (dy/sy)^2 + (dz/sz)^2))
where sx, sy, sz are the per-axis Gaussian scales.

A distance cutoff (max_sigma_multiplier) ensures each Gaussian only
influences nearby voxels, preventing uniform blurring across the scene.
"""
import torch
import torch.nn as nn
from qgfusion.utils.gaussian_utils import GaussianScene


class OccupancyHead(nn.Module):
    def __init__(
        self,
        feature_dim: int = 128,
        num_classes: int = 18,
        voxel_size=(0.4, 0.4, 0.4),
        pc_range=(-40.0, -40.0, -1.0, 40.0, 40.0, 5.4),
        max_sigma_multiplier: float = 3.0,
    ):
        super().__init__()
        self.voxel_size = voxel_size
        self.pc_range = pc_range
        self.max_sigma_mult = max_sigma_multiplier
        self.grid_size = [
            round((pc_range[3 + i] - pc_range[i]) / voxel_size[i])
            for i in range(3)
        ]  # (X, Y, Z)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, num_classes),
        )

    def _voxel_centers(self, device):
        X, Y, Z = self.grid_size
        xs = torch.linspace(self.pc_range[0], self.pc_range[3], X, device=device)
        ys = torch.linspace(self.pc_range[1], self.pc_range[4], Y, device=device)
        zs = torch.linspace(self.pc_range[2], self.pc_range[5], Z, device=device)
        gx, gy, gz = torch.meshgrid(xs, ys, zs, indexing="ij")
        return torch.stack([gx, gy, gz], dim=-1).view(-1, 3)  # (V, 3)

    def forward(self, gaussians: GaussianScene, chunk_size: int = 4000) -> torch.Tensor:
        """
        Args:
            gaussians: GaussianScene with N Gaussians per batch element
            chunk_size: voxels per gradient-checkpointed chunk
        Returns:
            occupancy_logits: (B, num_classes, X, Y, Z)
        """
        from torch.utils.checkpoint import checkpoint as grad_ckpt

        def _splat_chunk(c_chunk, pos_b, scale_b, opacity_b, feats_b):
            # c_chunk: (Vc, 3)   pos_b: (N, 3)   scale_b: (N, 3)
            # Anisotropic: compute per-axis squared distance / sigma^2
            diff = c_chunk.unsqueeze(1) - pos_b.unsqueeze(0)  # (Vc, N, 3)
            sigma2 = (scale_b ** 2).clamp(min=1e-4)           # (N, 3)
            # Mahalanobis-style distance for axis-aligned Gaussians
            mahal2 = (diff ** 2 / sigma2.unsqueeze(0)).sum(-1) # (Vc, N)
            # Distance cutoff: zero out contributions beyond max_sigma_mult * sigma
            cutoff = self.max_sigma_mult ** 2
            mask = (mahal2 <= cutoff).float()
            weight = opacity_b * torch.exp(-0.5 * mahal2) * mask  # (Vc, N)
            # Softmax over Gaussians so weights sum to 1 even with cutoff
            # Use softmax instead of normalize: handles zero-weight case gracefully
            # Add small epsilon before softmax so voxels with no coverage get
            # uniform average of all Gaussian features (not garbage from near-zero div)
            weight = weight + 1e-6  # ensure no voxel is fully zero
            weight = weight / weight.sum(-1, keepdim=True)
            return weight @ feats_b  # (Vc, C)

        B, N, _ = gaussians.position.shape
        device = gaussians.position.device
        centers = self._voxel_centers(device)
        V = centers.shape[0]
        C = gaussians.features.shape[-1]

        occ_feat = torch.zeros(B, V, C, device=device)

        for b in range(B):
            pos_b     = gaussians.position[b]   # (N, 3)
            scale_b   = gaussians.scale[b]      # (N, 3)
            opacity_b = gaussians.opacity[b]    # (N,)
            feats_b   = gaussians.features[b]   # (N, C)

            # Filter Gaussians outside pc_range to avoid wasting compute
            in_range = (
                (pos_b[:, 0] >= self.pc_range[0]) & (pos_b[:, 0] <= self.pc_range[3]) &
                (pos_b[:, 1] >= self.pc_range[1]) & (pos_b[:, 1] <= self.pc_range[4]) &
                (pos_b[:, 2] >= self.pc_range[2]) & (pos_b[:, 2] <= self.pc_range[5])
            )
            if in_range.sum() == 0:
                # No Gaussians in range — leave voxels at zero (will predict free)
                continue
            pos_b     = pos_b[in_range]
            scale_b   = scale_b[in_range]
            opacity_b = opacity_b[in_range]
            feats_b   = feats_b[in_range]

            for start in range(0, V, chunk_size):
                end = min(start + chunk_size, V)
                c_chunk = centers[start:end]
                occ_feat[b, start:end] = grad_ckpt(
                    _splat_chunk, c_chunk, pos_b, scale_b, opacity_b, feats_b,
                    use_reentrant=False,
                )

        logits = self.classifier(occ_feat)
        X, Y, Z = self.grid_size
        return logits.view(B, X, Y, Z, -1).permute(0, 4, 1, 2, 3)
