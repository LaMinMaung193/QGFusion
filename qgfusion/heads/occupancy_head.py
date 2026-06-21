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

    def forward(self, gaussians: GaussianScene, chunk_size: int = 20000) -> torch.Tensor:
        """
        Args:
            gaussians: GaussianScene, N Gaussians per batch element
            chunk_size: number of voxels processed at once. Bounds peak memory to
                roughly chunk_size * N * 3 floats regardless of total grid resolution --
                at full nuScenes occupancy resolution (0.4m, ~640k voxels) the unchunked
                (V, N, 3) distance tensor is multiple GB, so this matters even on a GPU.
        Returns:
            occupancy_logits: (B, num_classes, X, Y, Z)

        TODO(risk=MEDIUM): the splat below uses an isotropic distance falloff (mean scale
        across axes) for clarity. For the real model, swap in the full anisotropic Gaussian
        (rotation + per-axis scale, via gaussian_utils.quaternion_to_rotmat) per proposal
        Sec 4.6 Option 1.
        """
        B, N, _ = gaussians.position.shape
        device = gaussians.position.device
        centers = self._voxel_centers(device)  # (V, 3)
        V, C = centers.shape[0], gaussians.features.shape[-1]

        occ_feat = torch.zeros(B, V, C, device=device)

        for b in range(B):
            pos_b = gaussians.position[b]  # (N, 3)
            sigma2 = (gaussians.scale[b].mean(-1) ** 2).clamp(min=1e-4)  # (N,)
            opacity_b = gaussians.opacity[b]  # (N,)
            feats_b = gaussians.features[b]  # (N, C)

            for start in range(0, V, chunk_size):
                end = min(start + chunk_size, V)
                c_chunk = centers[start:end]  # (v, 3)
                dist2 = ((c_chunk.unsqueeze(1) - pos_b.unsqueeze(0)) ** 2).sum(-1)  # (v, N)
                weight = opacity_b.unsqueeze(0) * torch.exp(-0.5 * dist2 / sigma2.unsqueeze(0))  # (v, N)
                occ_feat[b, start:end] = weight @ feats_b  # (v, C)

        logits = self.classifier(occ_feat)  # (B, V, num_classes)
        X, Y, Z = self.grid_size
        return logits.view(B, X, Y, Z, -1).permute(0, 4, 1, 2, 3)
