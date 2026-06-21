"""
Quick dataset sanity check -- loads two real samples and prints every tensor's shape/dtype,
without needing the full training loop. Run this against nuScenes-mini first (works
immediately, no setup needed beyond what's already on disk), then against the consolidated
trainval root once that's set up.

Usage:
    python tools/test_dataset.py --dataroot /media/user/Transcend/nuScenes/v1.0-mini \
        --version v1.0-mini --split mini_train

    python tools/test_dataset.py --dataroot <consolidated_trainval_root> \
        --version v1.0-trainval --split train \
        --occ-gt-root /media/user/Transcend/data/occ3d/gts
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qgfusion.datasets.nuscenes_dataset import NuScenesMultiModalDataset, collate_fn


def _print_item(d: dict, indent: str = "  "):
    for k, v in d.items():
        if hasattr(v, "shape"):
            print(f"{indent}{k}: shape={tuple(v.shape)} dtype={v.dtype}")
        else:
            print(f"{indent}{k}: {v}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--split", default="mini_train", help="mini_train/mini_val for v1.0-mini; train/val for v1.0-trainval")
    parser.add_argument("--occ-gt-root", default=None)
    args = parser.parse_args()

    print(f"Loading {args.version} ({args.split}) from {args.dataroot} ...")
    ds = NuScenesMultiModalDataset(
        dataroot=args.dataroot,
        version=args.version,
        split=args.split,
        occ_gt_root=args.occ_gt_root,
    )
    print(f"Dataset has {len(ds)} samples.\n")

    print("Sample 0:")
    sample0 = ds[0]
    _print_item(sample0)

    print("\nSample 1:")
    sample1 = ds[1]
    _print_item(sample1)

    print("\nCollated batch of 2:")
    batch = collate_fn([sample0, sample1])
    _print_item(batch)

    print("\nDataset loader OK.")


if __name__ == "__main__":
    main()