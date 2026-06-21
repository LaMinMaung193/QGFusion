"""
Radar Encoder -- Stage 1 (ENCODER column, green lane in diagram)

Approach: per-point MLP over (x, y, vx, vy, rcs).
Input:  radar points, ~100 pts/frame, (B, P, 5) -> (x, y, vx, vy, rcs)
Output: motion-aware per-point embeddings Fr, (B, P, out_channels)

Risk level (per proposal Sec 7): LOW
"""
import torch
import torch.nn as nn


class RadarEncoder(nn.Module):
    """Per-point MLP encoder over sparse radar returns (x, y, vx, vy, rcs)."""

    def __init__(self, in_channels: int = 5, out_channels: int = 128, hidden: int = 128):
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Linear(in_channels, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, out_channels),
        )

    def forward(self, radar_points: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            radar_points: (B, P, 5) -> (x, y, vx, vy, rcs). Pad with zeros + mask for variable P.
            mask: (B, P) bool, True where valid (optional)
        Returns:
            Fr: per-point motion-aware embeddings, (B, P, out_channels)
        """
        Fr = self.point_mlp(radar_points)
        if mask is not None:
            Fr = Fr * mask.unsqueeze(-1)
        # TODO(risk=LOW): add an explicit BEV scatter (max/mean pool per BEV cell) here if the
        # Query Proposal Network should attend to a dense grid instead of a raw point-token set.
        return Fr
