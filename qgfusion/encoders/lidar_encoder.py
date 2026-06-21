"""
LiDAR Encoder -- Stage 1 (ENCODER column, red lane in diagram)

Backbone: CenterPoint-style sparse voxel encoder.
Input:  point cloud, ~30k pts/frame, (B, P, 4+) [x, y, z, intensity, ...]
Output: voxel features Fl, 3D geometric encoding

Risk level (per proposal Sec 7): LOW (encoder itself); sparse-conv backend integration is
infra work, not research risk.

Two backends are supported:
  - "spconv": real sparse convolution (VoxelBackBone8x, the standard SECOND/CenterPoint/
              OpenPCDet architecture). Requires `spconv` installed + a CUDA GPU -- spconv's
              conv kernels are CUDA-only, there is no CPU fallback for the conv ops
              themselves (only voxelization has a CPU path). This is the path to use for
              real training.
  - "dense":  dense voxel-grid + 3D conv stack. Slower / more memory than spconv, but has
              zero extra dependencies and runs on CPU -- this is what lets
              tools/test_forward.py run on a fresh machine before spconv is set up.
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
        but replace with a vectorized scatter before using on full-size point clouds."""
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


class _SparseVoxelEncoder(nn.Module):
    """Real sparse voxel encoder backed by spconv 2.x.

    Architecture is VoxelBackBone8x, the standard design shared across SECOND, CenterPoint,
    and OpenPCDet's other detectors: 4 stages of submanifold + strided sparse conv (each
    strided stage halves spatial resolution), followed by a final stride-(2,1,1) conv that
    compresses just the z dimension before densifying.
    """

    def __init__(self, in_channels: int = 4, out_channels: int = 256):
        super().__init__()
        import spconv.pytorch as spconv

        self._spconv = spconv

        def block(in_c, out_c, indice_key, stride=1, padding=1, subm=True):
            if subm:
                conv = spconv.SubMConv3d(in_c, out_c, 3, padding=padding, indice_key=indice_key, bias=False)
            else:
                conv = spconv.SparseConv3d(
                    in_c, out_c, 3, stride=stride, padding=padding, indice_key=indice_key, bias=False
                )
            return spconv.SparseSequential(conv, nn.BatchNorm1d(out_c), nn.ReLU(inplace=True))

        self.conv_input = block(in_channels, 16, "subm0")
        self.conv1 = block(16, 16, "subm1")

        self.conv2 = spconv.SparseSequential(
            block(16, 32, "spconv2", stride=2, padding=1, subm=False),
            block(32, 32, "subm2"),
            block(32, 32, "subm2"),
        )
        self.conv3 = spconv.SparseSequential(
            block(32, 64, "spconv3", stride=2, padding=1, subm=False),
            block(64, 64, "subm3"),
            block(64, 64, "subm3"),
        )
        self.conv4 = spconv.SparseSequential(
            block(64, 64, "spconv4", stride=2, padding=(0, 1, 1), subm=False),
            block(64, 64, "subm4"),
            block(64, 64, "subm4"),
        )
        # final conv: compress z only (stride (2,1,1)), expand to out_channels
        self.conv_out = spconv.SparseSequential(
            spconv.SparseConv3d(
                64, out_channels, (3, 1, 1), stride=(2, 1, 1), padding=0, indice_key="spconv_down2", bias=False
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, voxel_features, voxel_coords, spatial_shape, batch_size):
        """
        Args:
            voxel_features: (num_voxels, in_channels), already mean-pooled per voxel
            voxel_coords: (num_voxels, 4) int32 -- (batch_idx, z, y, x)
            spatial_shape: [D, H, W] in (z, y, x) order
            batch_size: int
        Returns:
            (B, out_channels, D', H', W') dense tensor
        """
        x = self._spconv.SparseConvTensor(voxel_features, voxel_coords, spatial_shape, batch_size)
        x = self.conv_input(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv_out(x)
        return x.dense()


class LiDAREncoder(nn.Module):
    """CenterPoint-style LiDAR encoder producing voxel features Fl."""

    def __init__(
        self,
        out_channels: int = 256,
        backend: str = "dense",
        pretrained_ckpt: str = None,
        spconv_voxel_size=(0.1, 0.1, 0.2),
        spconv_pc_range=(-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
        spconv_max_points_per_voxel: int = 10,
        spconv_max_voxels: int = 40000,
    ):
        super().__init__()
        self.backend = backend
        self.out_channels = out_channels

        if backend == "spconv":
            self.net = _SparseVoxelEncoder(in_channels=4, out_channels=out_channels)
            self._voxel_size = spconv_voxel_size
            self._pc_range = spconv_pc_range
            self._max_points_per_voxel = spconv_max_points_per_voxel
            self._max_voxels = spconv_max_voxels
            self._voxel_generators = {}
        elif backend == "dense":
            self.net = _DenseVoxelEncoder(out_channels=out_channels)
            self._dense_pc_range = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0)
            self._dense_voxel_size = (0.8, 0.8, 0.8)
        else:
            raise ValueError(f"Unknown LiDAR backend: {backend}")

        if pretrained_ckpt is not None:
            raise NotImplementedError("Load a CenterPoint pretrained checkpoint here.")

    def _get_voxel_generator(self, device: torch.device, num_point_features: int):
        from spconv.pytorch.utils import PointToVoxel

        key = str(device)
        if key not in self._voxel_generators:
            self._voxel_generators[key] = PointToVoxel(
                vsize_xyz=list(self._voxel_size),
                coors_range_xyz=list(self._pc_range),
                num_point_features=num_point_features,
                max_num_voxels=self._max_voxels,
                max_num_points_per_voxel=self._max_points_per_voxel,
                device=device,
            )
        return self._voxel_generators[key]

    def _voxelize_batch(self, points: torch.Tensor):
        B, P, C = points.shape
        device = points.device
        vg = self._get_voxel_generator(device, num_point_features=C)

        feats_list, coords_list = [], []
        for b in range(B):
            voxels, coords, num_per_voxel = vg(points[b])
            voxel_feat = voxels.sum(dim=1) / num_per_voxel.float().clamp(min=1).unsqueeze(-1)
            batch_col = torch.full((coords.shape[0], 1), b, dtype=torch.int32, device=device)
            coords_list.append(torch.cat([batch_col, coords], dim=1))
            feats_list.append(voxel_feat)

        voxel_features = torch.cat(feats_list, dim=0)
        voxel_coords = torch.cat(coords_list, dim=0)
        spatial_shape = [int(s) for s in vg.grid_size]  # already matches coords axis order -- do not reverse
        return voxel_features, voxel_coords, spatial_shape

    def forward(self, points: torch.Tensor, pc_range=None, voxel_size=None):
        if self.backend == "spconv":
            voxel_features, voxel_coords, spatial_shape = self._voxelize_batch(points)
            return self.net(voxel_features, voxel_coords, spatial_shape, points.shape[0])

        pc_range = pc_range or self._dense_pc_range
        voxel_size = voxel_size or self._dense_voxel_size
        return self.net(points, pc_range, voxel_size)
