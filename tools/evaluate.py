"""
Phase 5 — Quantitative Evaluation
===================================
Computes per-class voxel IoU (mIoU) for occupancy, and free-space IoU for
scene completion, following the official Occ3D-nuScenes evaluation protocol.

Occ3D evaluation protocol:
  - Only evaluate voxels observed by LiDAR OR camera (mask_lidar | mask_camera)
  - Class 17 = free/empty space
  - Classes 0-16 = 17 semantic occupied classes
  - mIoU computed over all 18 classes on observed voxels only

Usage:
    # Evaluate mini_val checkpoint (Phase 4C result)
    python tools/evaluate.py \\
        --config configs/default.yaml \\
        --checkpoint checkpoints/epoch_023_step_7752.pt \\
        --split mini_val

    # Evaluate trainval checkpoint once Phase 4D finishes
    python tools/evaluate.py \\
        --config configs/trainval.yaml \\
        --checkpoint checkpoints_trainval/epoch_023_step_XXXXX.pt \\
        --split val \\
        --blacklist bad_sample_tokens.txt

    # Quick sanity check on 10 samples
    python tools/evaluate.py \\
        --config configs/default.yaml \\
        --checkpoint checkpoints/epoch_023_step_7752.pt \\
        --split mini_val \\
        --max-samples 10
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from qgfusion.models.qg_fusion_model import QGFusionModel
from qgfusion.datasets.nuscenes_dataset import NuScenesMultiModalDataset, collate_fn

# ── Occ3D-nuScenes class names (index 0-17) ──────────────────────────────────
OCC3D_CLASSES = [
    "barrier",            # 0
    "bicycle",            # 1
    "bus",                # 2
    "car",                # 3
    "construction_veh",   # 4
    "motorcycle",         # 5
    "pedestrian",         # 6
    "traffic_cone",       # 7
    "trailer",            # 8
    "truck",              # 9
    "driveable_surface",  # 10
    "other_flat",         # 11
    "sidewalk",           # 12
    "terrain",            # 13
    "manmade",            # 14
    "vegetation",         # 15
    "free",               # 16  ← NOTE: Occ3D labels free as 17 in raw GT,
                          #     but some versions use 16. We handle both below.
    "free_17",            # 17  (raw Occ3D uses 17 for free/empty)
]

# Occ3D raw GT uses class 17 for free space. Classes 0-16 are occupied.
# We evaluate all 18 classes (0-17) with observed-voxel masking.
NUM_CLASSES = 18
FREE_CLASS  = 17  # raw Occ3D free label


class OccupancyEvaluator:
    """
    Accumulates per-class intersection and union across all samples,
    then computes IoU at the end.

    Follows Occ3D protocol:
      - Only count voxels where (mask_lidar | mask_camera) == 1
      - Ignore voxels with GT label == 255 (if any — some versions use this)
    """
    def __init__(self, num_classes: int = NUM_CLASSES):
        self.num_classes = num_classes
        self.intersection = np.zeros(num_classes, dtype=np.int64)
        self.union        = np.zeros(num_classes, dtype=np.int64)
        self.gt_count     = np.zeros(num_classes, dtype=np.int64)
        self.pred_count   = np.zeros(num_classes, dtype=np.int64)
        self.n_samples    = 0

    def update(self, pred: np.ndarray, gt: np.ndarray,
               mask_lidar: np.ndarray = None, mask_camera: np.ndarray = None):
        """
        Args:
            pred:         (X, Y, Z) int predicted class labels
            gt:           (X, Y, Z) int GT class labels
            mask_lidar:   (X, Y, Z) uint8 binary — 1 = observed by LiDAR
            mask_camera:  (X, Y, Z) uint8 binary — 1 = observed by camera
        """
        assert pred.shape == gt.shape, f"Shape mismatch: pred {pred.shape} vs gt {gt.shape}"

        # Build observation mask — evaluate only observed voxels
        if mask_lidar is not None and mask_camera is not None:
            obs_mask = (mask_lidar.astype(bool) | mask_camera.astype(bool))
        elif mask_lidar is not None:
            obs_mask = mask_lidar.astype(bool)
        elif mask_camera is not None:
            obs_mask = mask_camera.astype(bool)
        else:
            obs_mask = np.ones_like(gt, dtype=bool)  # evaluate all voxels

        # Flatten and apply mask
        pred_flat = pred[obs_mask].astype(np.int32)
        gt_flat   = gt[obs_mask].astype(np.int32)

        # Ignore invalid GT labels (255 used in some benchmark versions)
        valid = (gt_flat >= 0) & (gt_flat < self.num_classes)
        pred_flat = pred_flat[valid]
        gt_flat   = gt_flat[valid]

        # Accumulate per-class intersection and union
        for c in range(self.num_classes):
            pred_c = (pred_flat == c)
            gt_c   = (gt_flat   == c)
            self.intersection[c] += int((pred_c & gt_c).sum())
            self.union[c]        += int((pred_c | gt_c).sum())
            self.gt_count[c]     += int(gt_c.sum())
            self.pred_count[c]   += int(pred_c.sum())

        self.n_samples += 1

    def compute(self) -> dict:
        """Returns dict with per-class IoU and mIoU."""
        iou = np.zeros(self.num_classes)
        for c in range(self.num_classes):
            if self.union[c] == 0:
                iou[c] = float('nan')  # class never appeared — exclude from mean
            else:
                iou[c] = self.intersection[c] / self.union[c]

        # mIoU: mean over classes that appeared in GT
        valid_iou = iou[~np.isnan(iou)]
        miou = float(valid_iou.mean()) if len(valid_iou) > 0 else 0.0

        return {
            "miou":          miou,
            "per_class_iou": iou,
            "gt_count":      self.gt_count,
            "pred_count":    self.pred_count,
            "n_samples":     self.n_samples,
        }

    def print_results(self, results: dict):
        iou = results["per_class_iou"]
        print("\n" + "=" * 65)
        print(f"  Occupancy Evaluation — {results['n_samples']} samples")
        print("=" * 65)
        print(f"  {'Class':<22}  {'IoU':>8}  {'GT voxels':>12}  {'Pred voxels':>12}")
        print("  " + "-" * 61)
        for c in range(self.num_classes):
            name = OCC3D_CLASSES[c] if c < len(OCC3D_CLASSES) else f"class_{c}"
            iou_str = f"{iou[c]*100:.2f}%" if not np.isnan(iou[c]) else "  N/A  "
            print(f"  {name:<22}  {iou_str:>8}  "
                  f"{results['gt_count'][c]:>12,}  {results['pred_count'][c]:>12,}")
        print("  " + "-" * 61)
        print(f"  {'mIoU':<22}  {results['miou']*100:.2f}%")
        print("=" * 65)


class CompletionEvaluator:
    """
    Evaluates scene completion on the downsampled 100x100x8 grid.
    Binary: free (0) vs occupied (1).
    Uses same observation mask as occupancy.
    """
    def __init__(self):
        self.intersection = np.zeros(2, dtype=np.int64)
        self.union        = np.zeros(2, dtype=np.int64)
        self.n_samples    = 0

    def update(self, pred_logits: torch.Tensor, gt_occ: torch.Tensor):
        """
        Args:
            pred_logits: (1, 2, X//2, Y//2, Z//2) completion logits
            gt_occ:      (1, X, Y, Z) raw Occ3D semantics
        """
        from qgfusion.utils.losses import make_completion_gt
        comp_gt = make_completion_gt(gt_occ)  # (1, 100, 100, 8) binary
        comp_pred = pred_logits.argmax(dim=1)  # (1, 100, 100, 8)

        pred_np = comp_pred[0].cpu().numpy().astype(np.int32)
        gt_np   = comp_gt[0].cpu().numpy().astype(np.int32)

        for c in range(2):
            pred_c = (pred_np == c)
            gt_c   = (gt_np   == c)
            self.intersection[c] += int((pred_c & gt_c).sum())
            self.union[c]        += int((pred_c | gt_c).sum())

        self.n_samples += 1

    def compute(self) -> dict:
        iou = np.zeros(2)
        for c in range(2):
            iou[c] = self.intersection[c] / self.union[c] if self.union[c] > 0 else 0.0
        return {
            "free_iou":     float(iou[0]),
            "occupied_iou": float(iou[1]),
            "mean_iou":     float(iou.mean()),
            "n_samples":    self.n_samples,
        }

    def print_results(self, results: dict):
        print("\n" + "=" * 40)
        print(f"  Scene Completion — {results['n_samples']} samples")
        print("=" * 40)
        print(f"  Free IoU:      {results['free_iou']*100:.2f}%")
        print(f"  Occupied IoU:  {results['occupied_iou']*100:.2f}%")
        print(f"  Mean IoU:      {results['mean_iou']*100:.2f}%")
        print("=" * 40)


def build_occ3d_index(occ_gt_root: str) -> dict:
    """Build token -> full .npz path index for fast lookup."""
    index = {}
    for scene_dir in os.listdir(occ_gt_root):
        scene_path = os.path.join(occ_gt_root, scene_dir)
        if not os.path.isdir(scene_path):
            continue
        for fname in os.listdir(scene_path):
            if fname.endswith(".npz"):
                token = fname[:-4]
                index[token] = os.path.join(scene_path, fname)
    return index


def load_occ3d_gt(sample_token: str, occ_gt_root: str, index: dict = None):
    """
    Load Occ3D GT for a sample token.
    Returns (semantics, mask_lidar, mask_camera) or (None, None, None) if missing.
    Handles both flat {token}.npz and scene/{token}.npz layouts.
    """
    # Use pre-built index for fast lookup
    if index is not None:
        path = index.get(sample_token, None)
    else:
        path = None
        for scene_dir in os.listdir(occ_gt_root):
            candidate = os.path.join(occ_gt_root, scene_dir, f"{sample_token}.npz")
            if os.path.exists(candidate):
                path = candidate
                break

    if path is None:
        return None, None, None

    try:
        data = np.load(path)
        semantics    = data["semantics"]     # (200, 200, 16) uint8
        mask_lidar   = data.get("mask_lidar",   None)
        mask_camera  = data.get("mask_camera",  None)
        if mask_lidar is not None:
            mask_lidar = mask_lidar.astype(np.uint8)
        if mask_camera is not None:
            mask_camera = mask_camera.astype(np.uint8)
        return semantics, mask_lidar, mask_camera
    except Exception as e:
        print(f"  [warn] failed to load GT for {sample_token}: {e}")
        return None, None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     required=True,
                        help="Path to config yaml (e.g. configs/default.yaml)")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to model checkpoint .pt file")
    parser.add_argument("--split",      default="mini_val",
                        help="Dataset split to evaluate (mini_val, val)")
    parser.add_argument("--blacklist",  default=None,
                        help="Path to bad_sample_tokens.txt (for trainval)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit evaluation to first N samples (for quick checks)")
    parser.add_argument("--no-completion", action="store_true",
                        help="Skip scene completion evaluation")
    parser.add_argument("--output",     default=None,
                        help="Save results to this .txt file")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Split: {args.split}")

    # ── Load model ────────────────────────────────────────────────────────────
    model = QGFusionModel(cfg).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint (epoch {ckpt.get('epoch', '?')}, "
          f"step {ckpt.get('step', '?')})")

    # ── Build dataset ─────────────────────────────────────────────────────────
    ds = NuScenesMultiModalDataset(
        dataroot=cfg["dataset"]["dataroot"],
        version=cfg["dataset"]["version"],
        split=args.split,
        occ_gt_root=cfg["dataset"]["occ_gt_root"],
        num_cameras=cfg["dataset"]["num_cameras"],
        img_size=tuple(cfg["dataset"]["img_size"]),
        blacklist=args.blacklist,
    )
    print(f"Dataset: {len(ds)} samples")

    loader = DataLoader(
        ds, batch_size=1, shuffle=False,
        num_workers=2, collate_fn=collate_fn,
        pin_memory=(device == "cuda"),
    )

    # ── Evaluators ────────────────────────────────────────────────────────────
    occ_eval  = OccupancyEvaluator(num_classes=NUM_CLASSES)
    comp_eval = CompletionEvaluator()

    occ_gt_root   = cfg["dataset"]["occ_gt_root"]
    print(f"Building Occ3D GT index from {occ_gt_root}...")
    occ3d_index = build_occ3d_index(occ_gt_root)
    print(f"  Found {len(occ3d_index)} GT files")
    n_processed   = 0
    n_with_gt     = 0
    n_skipped_gt  = 0
    t_start       = time.time()

    print(f"\nRunning evaluation...")
    print("-" * 50)

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if args.max_samples and n_processed >= args.max_samples:
                break

            sample_token = batch["sample_tokens"][0]

            # Skip samples without Occ3D GT
            if batch.get("gt_occupancy") is None:
                n_skipped_gt += 1
                n_processed  += 1
                continue

            # Load GT with masks directly from file (for mask_lidar/mask_camera)
            semantics, mask_lidar, mask_camera = load_occ3d_gt(
                sample_token, occ_gt_root, index=occ3d_index)
            if semantics is None:
                n_skipped_gt += 1
                n_processed  += 1
                continue

            # Forward pass
            batch_gpu = {k: (v.to(device) if torch.is_tensor(v) else v)
                         for k, v in batch.items()}
            out = model(batch_gpu)

            # ── Occupancy IoU ─────────────────────────────────────────────
            occ_logits = out["occupancy_logits"]          # (1, 18, 200, 200, 16)
            occ_pred   = occ_logits.argmax(dim=1)         # (1, 200, 200, 16)
            pred_np    = occ_pred[0].cpu().numpy()        # (200, 200, 16)

            occ_eval.update(pred_np, semantics, mask_lidar, mask_camera)

            # ── Scene Completion IoU ──────────────────────────────────────
            if not args.no_completion:
                comp_eval.update(out["completion_logits"], batch_gpu["gt_occupancy"])

            n_with_gt += 1
            n_processed += 1

            # Progress
            if n_with_gt % 10 == 0:
                elapsed = time.time() - t_start
                ms_per  = elapsed / n_with_gt * 1000
                print(f"  [{n_with_gt:4d} samples with GT | {n_processed:4d} total] "
                      f"{ms_per:.0f}ms/sample")

    # ── Print results ─────────────────────────────────────────────────────────
    print(f"\nProcessed {n_processed} samples total, "
          f"{n_with_gt} with Occ3D GT, "
          f"{n_skipped_gt} skipped (no GT)")

    occ_results  = occ_eval.compute()
    occ_eval.print_results(occ_results)

    if not args.no_completion and comp_eval.n_samples > 0:
        comp_results = comp_eval.compute()
        comp_eval.print_results(comp_results)
    else:
        comp_results = None

    # ── Save results ──────────────────────────────────────────────────────────
    if args.output:
        with open(args.output, "w") as f:
            f.write(f"Checkpoint: {args.checkpoint}\n")
            f.write(f"Split: {args.split}\n")
            f.write(f"Samples with GT: {n_with_gt}\n\n")
            f.write(f"mIoU: {occ_results['miou']*100:.2f}%\n\n")
            for c in range(NUM_CLASSES):
                name = OCC3D_CLASSES[c] if c < len(OCC3D_CLASSES) else f"class_{c}"
                iou  = occ_results["per_class_iou"][c]
                iou_str = f"{iou*100:.2f}%" if not np.isnan(iou) else "N/A"
                f.write(f"  {name:<22}  {iou_str}\n")
            if comp_results:
                f.write(f"\nCompletion free IoU:     {comp_results['free_iou']*100:.2f}%\n")
                f.write(f"Completion occupied IoU: {comp_results['occupied_iou']*100:.2f}%\n")
                f.write(f"Completion mean IoU:     {comp_results['mean_iou']*100:.2f}%\n")
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()