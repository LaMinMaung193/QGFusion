"""
Smoke test: runs the full QGFusionModel forward pass on synthetic tensors of the right
shapes, with zero dependency on nuScenes data or spconv. Run this first after cloning to
confirm the module wiring is correct before touching real data.

Usage:
    python tools/test_forward.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import yaml
import torch

from qgfusion.models.qg_fusion_model import QGFusionModel


def main():
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Smoke test only checks shape wiring, not weight quality -- skip the pretrained download
    # so this runs offline. Real training should use cfg["model"]["camera_pretrained"]="imagenet".
    cfg["model"]["camera_pretrained"] = "none"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Building model on device={device} ...")
    model = QGFusionModel(cfg).to(device)
    model.eval()

    B = 1
    batch = {
        "camera_images": torch.randn(
            B, cfg["dataset"]["num_cameras"], 3, *cfg["dataset"]["img_size"], device=device
        ),
        "lidar_points": torch.randn(B, 5000, 4, device=device) * 10,  # rough nuScenes-ish point spread
        "radar_points": torch.randn(B, 100, 5, device=device) * 5,
    }

    print("Running forward pass...")
    with torch.no_grad():
        out = model(batch)

    print("\nForward pass OK. Output shapes:")
    print(f"  occupancy_logits  : {tuple(out['occupancy_logits'].shape)}")
    print(f"  detection.cls     : {tuple(out['detection']['cls_logits'].shape)}")
    print(f"  detection.boxes   : {tuple(out['detection']['boxes'].shape)}")
    print(f"  detection.velocity: {tuple(out['detection']['velocity'].shape)}")
    print(f"  completion_logits : {tuple(out['completion_logits'].shape)}")
    print(f"  gaussians.N       : {out['gaussians'].num_gaussians}")


if __name__ == "__main__":
    main()
