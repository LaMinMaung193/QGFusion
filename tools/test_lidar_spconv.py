"""
Standalone test for the spconv LiDAR backend, isolated from the rest of the model so any
spconv-specific issues (install, CUDA arch mismatch, API drift) are easy to debug separately
from the full pipeline.

Usage (run on a CUDA machine with spconv installed):
    python tools/test_lidar_spconv.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from qgfusion.encoders.lidar_encoder import LiDAREncoder


def main():
    assert torch.cuda.is_available(), "spconv's conv kernels are CUDA-only -- run this on the 3090, not CPU."
    device = "cuda"

    encoder = LiDAREncoder(out_channels=256, backend="spconv").to(device)

    B, P = 2, 20000  # roughly one nuScenes LiDAR_TOP sweep per sample
    points = torch.zeros(B, P, 4, device=device)
    points[..., :2] = (torch.rand(B, P, 2, device=device) - 0.5) * 100  # x, y in [-50, 50]
    points[..., 2] = (torch.rand(B, P, device=device) - 0.5) * 6  # z in [-3, 3]
    points[..., 3] = torch.rand(B, P, device=device)  # intensity in [0, 1]

    print("Running spconv forward pass...")
    out = encoder(points)
    print(f"Output shape: {tuple(out.shape)}  (expect (B={B}, C=256, D', H', W'))")
    print("spconv backend OK.")


if __name__ == "__main__":
    main()
