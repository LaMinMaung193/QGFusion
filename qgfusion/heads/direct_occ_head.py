"""
Ablation A2 — Direct Query Occupancy Head (no Gaussian intermediate).
Takes fused queries Qf (B, N, embed_dim) and predicts voxel occupancy
by projecting each query to a spatial position and splatting features
directly without the Gaussian generator.

Used as the baseline for Ablation A: proves whether Gaussian representation
adds value over direct query decoding.
"""
import torch
import torch.nn as nn


class DirectOccHead(nn.Module):
    """
    Direct query-to-occupancy without Gaussian intermediate.
    Each query contributes to nearby voxels based on a learned position.
    Simpler than GaussianOccHead: no scale/opacity/rotation.
    """
    def __init__(
        self,
        embed_dim: int = 256,
        num_classes: int = 18,
        voxel_size=(0.4, 0.4, 0.4),
        pc_range=(-40.0, -40.0, -1.0, 40.0, 40.0, 5.4),
        num_queries: int = 300,
    ):
        super().__init__()
        self.pc_range = pc_range
        self.voxel_size = voxel_size
        self.grid_size = [
            round((pc_range[3+i] - pc_range[i]) / voxel_size[i])
            for i in range(3)
        ]
        # Each query gets a learned position in BEV
        n_z = 4
        n_xy = num_queries // n_z
        grid_w = int(n_xy**0.5) + 1
        idx = torch.arange(num_queries).float()
        idx_xy = idx % n_xy
        idx_z  = (idx // n_xy).clamp(max=n_z-1)
        ref_x = ((idx_xy % grid_w) / grid_w - 0.5) * 2 * 40.0
        ref_y = ((idx_xy // grid_w) / grid_w - 0.5) * 2 * 40.0
        ref_z = (idx_z / (n_z-1) - 0.5) * 2 * 3.2
        self.register_buffer('ref_pos', torch.stack([ref_x, ref_y, ref_z], dim=-1))

        self.pos_head     = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, 2))  # XY only
        self.feature_head = nn.Sequential(
            nn.Linear(embed_dim, 128), nn.ReLU())
        self.classifier   = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, num_classes))

        self.sigma = 1.0  # 1m sigma: tight enough to differentiate voxels

    def _voxel_centers(self, device):
        X, Y, Z = self.grid_size
        xs = torch.linspace(self.pc_range[0], self.pc_range[3], X, device=device)
        ys = torch.linspace(self.pc_range[1], self.pc_range[4], Y, device=device)
        zs = torch.linspace(self.pc_range[2], self.pc_range[5], Z, device=device)
        gx, gy, gz = torch.meshgrid(xs, ys, zs, indexing='ij')
        return torch.stack([gx, gy, gz], dim=-1).view(-1, 3)

    def forward(self, Qf: torch.Tensor, chunk_size: int = 4000) -> torch.Tensor:
        from torch.utils.checkpoint import checkpoint as grad_ckpt

        B, N, _ = Qf.shape
        device = Qf.device

        # Query positions: fixed Z from ref, learned XY residual
        pos_xy_res = self.pos_head(Qf) * 5.0       # (B, N, 2)
        pos_xy = self.ref_pos[:, :2].unsqueeze(0) + pos_xy_res  # (B, N, 2)
        pos_z  = self.ref_pos[:, 2:3].unsqueeze(0).expand(B, -1, -1)  # (B, N, 1)
        pos = torch.cat([pos_xy, pos_z], dim=-1)    # (B, N, 3)
        pos = torch.clamp(pos,
            torch.tensor([self.pc_range[0], self.pc_range[1], self.pc_range[2]], device=device),
            torch.tensor([self.pc_range[3], self.pc_range[4], self.pc_range[5]], device=device))

        feats = self.feature_head(Qf)               # (B, N, 128)
        centers = self._voxel_centers(device)        # (V, 3)
        V = centers.shape[0]
        occ_feat = torch.zeros(B, V, 128, device=device)

        def _splat(c_chunk, pos_b, feats_b):
            diff  = c_chunk.unsqueeze(1) - pos_b.unsqueeze(0)  # (Vc, N, 3)
            dist2 = (diff**2).sum(-1)                            # (Vc, N)
            w     = torch.exp(-0.5 * dist2 / self.sigma**2)     # (Vc, N)
            return w @ feats_b                                   # (Vc, 128)

        for b in range(B):
            for start in range(0, V, chunk_size):
                end = min(start + chunk_size, V)
                occ_feat[b, start:end] = grad_ckpt(
                    _splat, centers[start:end], pos[b], feats[b],
                    use_reentrant=False)

        logits = self.classifier(occ_feat)
        X, Y, Z = self.grid_size
        return logits.view(B, X, Y, Z, -1).permute(0, 4, 1, 2, 3)
