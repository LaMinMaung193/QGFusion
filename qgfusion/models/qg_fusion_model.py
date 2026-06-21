"""
Top-level model -- wires the full pipeline together exactly as drawn in the architecture
diagram:

    Camera/LiDAR/Radar -> Encoders -> Query Generation -> Adaptive Fusion
        -> Query-to-Gaussian -> {Occupancy, Detection, Completion} heads
"""
import torch
import torch.nn as nn

from qgfusion.encoders.camera_encoder import CameraEncoder
from qgfusion.encoders.lidar_encoder import LiDAREncoder
from qgfusion.encoders.radar_encoder import RadarEncoder
from qgfusion.query_generation.query_proposal_network import QueryProposalNetwork
from qgfusion.fusion.adaptive_query_fusion import AdaptiveQueryFusion
from qgfusion.gaussian.query_to_gaussian import QueryToGaussianGenerator
from qgfusion.heads.occupancy_head import OccupancyHead
from qgfusion.heads.detection_head import DetectionHead
from qgfusion.heads.completion_head import SceneCompletionHead


class QGFusionModel(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg["model"]

        self.camera_encoder = CameraEncoder(
            out_channels=m["camera_channels"], pretrained_backbone=m.get("camera_pretrained", "imagenet")
        )
        self.lidar_encoder = LiDAREncoder(out_channels=m["lidar_channels"], backend=m.get("lidar_backend", "dense"))
        self.radar_encoder = RadarEncoder(out_channels=m["radar_channels"])

        embed_dim = m["embed_dim"]
        num_queries = m["num_queries"]
        self.qpn_camera = QueryProposalNetwork(m["camera_channels"], embed_dim, num_queries)
        self.qpn_lidar = QueryProposalNetwork(m["lidar_channels"], embed_dim, num_queries)
        self.qpn_radar = QueryProposalNetwork(m["radar_channels"], embed_dim, num_queries)

        self.fusion = AdaptiveQueryFusion(embed_dim=embed_dim, num_layers=m.get("fusion_layers", 4))
        self.gaussian_gen = QueryToGaussianGenerator(
            embed_dim=embed_dim,
            feature_dim=m["gaussian_feature_dim"],
            predict_velocity=m.get("predict_velocity", False),
        )

        self.occ_head = OccupancyHead(
            feature_dim=m["gaussian_feature_dim"],
            num_classes=m["occ_num_classes"],
            voxel_size=m["occ_voxel_size"],
            pc_range=m["pc_range"],
        )
        self.det_head = DetectionHead(feature_dim=m["gaussian_feature_dim"], num_classes=m["det_num_classes"])
        self.completion_head = SceneCompletionHead(feature_dim=m["gaussian_feature_dim"], pc_range=m["pc_range"])

    def forward(self, batch: dict) -> dict:
        """
        Args:
            batch: dict with keys
                "camera_images": (B, N_cam, 3, H, W)
                "lidar_points":  (B, P, 4)
                "radar_points":  (B, P_r, 5)
        Returns:
            dict with "occupancy_logits", "detection", "completion_logits", and "gaussians"
            (the raw GaussianScene -- useful for visualization/debugging).
        """
        Fc = self.camera_encoder(batch["camera_images"])  # list[(B, N_cam, C, H, W)]
        Fl = self.lidar_encoder(batch["lidar_points"])  # (B, C, D, H, W) for "dense" backend
        Fr = self.radar_encoder(batch["radar_points"])  # (B, P_r, C)

        # --- flatten each modality's native feature map into a (B, L, C) token sequence ---
        # TODO(risk=LOW): replace with proper positional encodings; this is shape-correct but
        # drops spatial position info that the QPN's cross-attention would benefit from.
        Bc, Ncam, Cc, Hc, Wc = Fc[-1].shape
        Fc_tokens = Fc[-1].permute(0, 1, 3, 4, 2).reshape(Bc, Ncam * Hc * Wc, Cc)

        Bl, Cl, Dl, Hl, Wl = Fl.shape
        Fl_tokens = Fl.permute(0, 2, 3, 4, 1).reshape(Bl, Dl * Hl * Wl, Cl)

        Fr_tokens = Fr  # already (B, P_r, C)

        Qc = self.qpn_camera(Fc_tokens)
        Ql = self.qpn_lidar(Fl_tokens)
        Qr = self.qpn_radar(Fr_tokens)

        Qf = self.fusion(Qc, Ql, Qr)
        gaussians = self.gaussian_gen(Qf)

        occupancy_logits = self.occ_head(gaussians)
        detection = self.det_head(gaussians)
        completion_logits = self.completion_head(gaussians)

        return {
            "occupancy_logits": occupancy_logits,
            "detection": detection,
            "completion_logits": completion_logits,
            "gaussians": gaussians,
        }
