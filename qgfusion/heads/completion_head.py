"""
Scene Completion Head -- Support task, OUTPUT HEADS column.

Predicts free-space / occupied / unknown at a coarser resolution than the main Occupancy
Head, used as an auxiliary supervision signal (proposal Sec 3, "Support Task"). Subclasses
OccupancyHead rather than reimplementing the splat, since the proposal frames this as a
relaxed/auxiliary version of the same query -- not a separate architecture.

Risk level (per proposal Sec 7): LOW
"""
from qgfusion.heads.occupancy_head import OccupancyHead


class SceneCompletionHead(OccupancyHead):
    """Coarser-resolution OccupancyHead with free-space-oriented classes (free / occupied /
    unknown) instead of semantic occupancy classes."""

    def __init__(
        self,
        feature_dim: int = 128,
        voxel_size=(0.8, 0.8, 0.8),
        pc_range=(-40.0, -40.0, -1.0, 40.0, 40.0, 5.4),
    ):
        super().__init__(feature_dim=feature_dim, num_classes=3, voxel_size=voxel_size, pc_range=pc_range)
