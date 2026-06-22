"""
Occupancy Head -- Primary task, OUTPUT HEADS column.

Implements Option 1 from the proposal (recommended): voxel rasterization. Gaussians are
splatted into a voxel grid (weighted by opacity and an isotropic Gaussian falloff from each
voxel center to the Gaussian's mean), then an MLP turns the accumulated per-voxel feature
into class logits. Compatible with standard mIoU/IoU benchmarks (e.g. Occ3D-nuScenes).

Risk level (per proposal Sec 7): LOW (head itself), but real quality depends heavily on the
MEDIUM-risk Gaussian generator upstream being well-calibrated.
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
    ):
        super().__init__()
        self.voxel_size = voxel_size
        self.pc_range = pc_range
        self.grid_size = [round((pc_range[3 + i] - pc_range[i]) / voxel_size[i]) for i in range(3)]  # (X, Y, Z)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim), nn.ReLU(inplace=True),
            nn.Linear(feature_dim, num_classes),
        )

    def _voxel_centers(self, device):
        X, Y, Z = self.grid_size
        xs = torch.linspace(self.pc_range[0], self.pc_range[3], X, device=device)
        ys = torch.linspace(self.pc_range[1], self.pc_range[4], Y, device=device)
        zs = torch.linspace(self.pc_range[2], self.pc_range[5], Z, device=device)
        gx, gy, gz = torch.meshgrid(xs, ys, zs, indexing="ij")
        return torch.stack([gx, gy, gz], dim=-1).view(-1, 3)  # (X*Y*Z, 3)

    def forward(self, gaussians: GaussianScene, chunk_size: int = 4000) -> torch.Tensor:
        """
        Args:
            gaussians: GaussianScene, N Gaussians per batch element
            chunk_size: voxels per chunk. Gradient checkpointing means each chunk's
                intermediates are recomputed during backward rather than stored, so
                total backward memory is O(chunk_size * N) not O(V * N).
                4000 is safe on a 24GB card with batch_size=1 and N=300 queries.
        Returns:
            occupancy_logits: (B, num_classes, X, Y, Z)
        """
        from torch.utils.checkpoint import checkpoint as grad_ckpt

        def _splat_chunk(c_chunk, pos_b, scale_b, opacity_b, feats_b):
            dist2 = ((c_chunk.unsqueeze(1) - pos_b.unsqueeze(0)) ** 2).sum(-1)
            sigma2 = (scale_b.mean(-1) ** 2).clamp(min=1e-4)
            weight = opacity_b * torch.exp(-0.5 * dist2 / sigma2.unsqueeze(0))
            return weight @ feats_b

        B, N, _ = gaussians.position.shape
        device = gaussians.position.device
        centers = self._voxel_centers(device)
        V, C = centers.shape[0], gaussians.features.shape[-1]
        occ_feat = torch.zeros(B, V, C, device=device)

        for b in range(B):
            pos_b     = gaussians.position[b]
            scale_b   = gaussians.scale[b]
            opacity_b = gaussians.opacity[b]
            feats_b   = gaussians.features[b]
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
