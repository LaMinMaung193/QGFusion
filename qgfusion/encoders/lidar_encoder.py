"""
LiDAR Encoder -- Stage 1 (ENCODER column, red lane in diagram)

Backbone: CenterPoint-style sparse voxel encoder.
Input:  point cloud, ~30k pts/frame, (B, P, 4+) [x, y, z, intensity, ...]
Output: voxel features Fl, 3D geometric encoding

Risk level (per proposal Sec 7): LOW (encoder itself); sparse-conv backend integration is
infra work, not research risk.

Two backends are supported:
  - "spconv": real sparse convolution (CenterPoint backbone). Requires `spconv` installed.
              NOT wired up yet -- see TODO in __init__.
  - "dense":  dense voxel-grid + 3D conv stack. Slower / more memory than spconv, but has
              zero extra dependencies -- this is what lets tools/test_forward.py run on a
              fresh machine before spconv is set up. Swap to "spconv" once your cluster
              environment is ready for real training.
"""
import torch
import torch.nn as nn


class _DenseVoxelEncoder(nn.Module):
    """Dependency-free fallback: naive voxelize (mean-pool per voxel) -> dense 3D conv stack."""

    def __init__(self, in_channels: int = 1, out_channels: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, 32, 3, stride=2, padding=1), nn.BatchNorm3d(32), nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, 3, stride=2, padding=1), nn.BatchNorm3d(64), nn.ReLU(inplace=True),
            nn.Conv3d(64, out_channels, 3, stride=2, padding=1), nn.BatchNorm3d(out_channels), nn.ReLU(inplace=True),
        )

    def voxelize(self, points: torch.Tensor, pc_range, voxel_size):
        """Naive mean-pool voxelization, implemented in pure PyTorch (no torch-scatter dep).
        TODO(risk=LOW): this loops over the batch dim in Python -- fine for the smoke test,
        but replace with a vectorized scatter (torch_scatter.scatter_mean, or spconv's own
        voxelization) before using on full-size point clouds for real training."""
        B, P, C = points.shape
        device = points.device
        pc_range_t = torch.tensor(pc_range, device=device, dtype=points.dtype)
        voxel_size_t = torch.tensor(voxel_size, device=device, dtype=points.dtype)
        grid_size = ((pc_range_t[3:] - pc_range_t[:3]) / voxel_size_t).long()  # (X, Y, Z)
        X, Y, Z = grid_size.tolist()

        xyz = points[..., :3]
        feat = points[..., 3:4] if C > 3 else torch.ones(B, P, 1, device=device, dtype=points.dtype)
        feat_dim = feat.shape[-1]

        idx = ((xyz - pc_range_t[:3]) / voxel_size_t).long()
        valid = ((idx >= 0) & (idx < grid_size)).all(dim=-1)

        voxels = torch.zeros(B, feat_dim, Z, Y, X, device=device, dtype=points.dtype)
        counts = torch.zeros(B, 1, Z, Y, X, device=device, dtype=points.dtype)

        for b in range(B):
            v_idx = idx[b][valid[b]]
            v_feat = feat[b][valid[b]]
            if v_idx.numel() == 0:
                continue
            flat_idx = v_idx[:, 2] * (Y * X) + v_idx[:, 1] * X + v_idx[:, 0]
            voxels[b].view(feat_dim, -1).index_add_(1, flat_idx, v_feat.t())
            ones = torch.ones(flat_idx.shape[0], device=device, dtype=points.dtype).unsqueeze(0)
            counts[b].view(1, -1).index_add_(1, flat_idx, ones)

        return voxels / counts.clamp(min=1)

    def forward(self, points: torch.Tensor, pc_range, voxel_size):
        voxels = self.voxelize(points, pc_range, voxel_size)
        return self.net(voxels)


class LiDAREncoder(nn.Module):
    """CenterPoint-style LiDAR encoder producing voxel features Fl."""

    def __init__(self, out_channels: int = 256, backend: str = "dense", pretrained_ckpt: str = None):
        super().__init__()
        self.backend = backend
        self.out_channels = out_channels

        if backend == "spconv":
            # TODO(risk=LOW, infra): wire in a real spconv VoxelNet/CenterPoint backbone, e.g.
            #   from mmdet3d.models.backbones import SparseEncoder
            #   self.net = SparseEncoder(...)
            # Left unimplemented so this file doesn't hard-require spconv to be importable.
            self.net = None
        elif backend == "dense":
            self.net = _DenseVoxelEncoder(out_channels=out_channels)
        else:
            raise ValueError(f"Unknown LiDAR backend: {backend}")

        if pretrained_ckpt is not None:
            raise NotImplementedError("Load a CenterPoint pretrained checkpoint here.")

    def forward(
        self,
        points: torch.Tensor,
        pc_range=(-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
        voxel_size=(0.8, 0.8, 0.8),
    ):
        """
        Args:
            points: (B, P, C) raw LiDAR points, C >= 4 (x, y, z, intensity, ...)
        Returns:
            Fl: voxel features.
                - "spconv": SparseConvTensor (TODO once backend wired up)
                - "dense":  dense Tensor (B, C_out, D', H', W')
        """
        if self.backend == "spconv":
            raise NotImplementedError(
                "spconv backend not wired up yet -- see TODO in __init__, "
                "or pass backend='dense' to run the skeleton end-to-end for now."
            )
        return self.net(points, pc_range, voxel_size)
