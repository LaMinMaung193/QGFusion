"""
Gaussian scene representation -- shared dataclass + helpers used by the Query-to-Gaussian
Generator and all three output heads.
"""
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class GaussianScene:
    """A set of N 3D Gaussian primitives per batch element. G = {G1, ..., GN} in the proposal."""

    position: torch.Tensor             # (B, N, 3)  world-frame center (x, y, z)
    scale: torch.Tensor                # (B, N, 3)  std-dev along each axis (sx, sy, sz), > 0
    rotation: torch.Tensor             # (B, N, 4)  unit quaternion (w, x, y, z)
    opacity: torch.Tensor              # (B, N)     in [0, 1]
    features: torch.Tensor             # (B, N, C)  semantic / appearance embedding
    velocity: Optional[torch.Tensor] = None  # (B, N, 3), only if predict_velocity=True

    @property
    def num_gaussians(self) -> int:
        return self.position.shape[1]

    def to(self, device):
        kwargs = {}
        for f in ("position", "scale", "rotation", "opacity", "features", "velocity"):
            v = getattr(self, f)
            kwargs[f] = v.to(device) if v is not None else None
        return GaussianScene(**kwargs)


def quaternion_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """(..., 4) unit quaternion (w, x, y, z) -> (..., 3, 3) rotation matrix.

    Intended for the occupancy head's voxel rasterizer, to orient each Gaussian's covariance
    anisotropically instead of the isotropic approximation currently used in OccupancyHead
    (see that file's TODO).
    """
    w, x, y, z = q.unbind(-1)
    lead_shape = q.shape[:-1]
    R = torch.stack(
        [
            1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x ** 2 + y ** 2),
        ],
        dim=-1,
    ).view(*lead_shape, 3, 3)
    return R
