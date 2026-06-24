"""
Creates a unified nuScenes trainval dataroot by symlinking sensor files
from the 5 blob folders into a single directory tree.

Run once before Phase 4D training. Takes ~5 minutes (symlinks only, no copying).

Usage:
    python tools/consolidate_trainval.py
    python tools/consolidate_trainval.py --out /media/user/Transcend/data123/v1.0-trainval-unified
"""
import os
import argparse

BLOB_DIRS = [
    "/media/user/Transcend/data123/v1 (1).0-trainval01_blobs",
    "/media/user/Transcend/data123/v1.0-trainval02_blobs",
    "/media/user/Transcend/data123/v1.0-trainval03_blobs",
    "/media/user/Transcend/data123/v1.0-trainval04_blobs",
    "/media/user/Transcend/data123/v1.0-trainval05_blobs",
]
META_DIR = "/media/user/Transcend/data123/v1.0-trainval_meta"

SENSOR_SUBDIRS = [
    "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
    "LIDAR_TOP",
    "RADAR_FRONT", "RADAR_FRONT_LEFT", "RADAR_FRONT_RIGHT",
    "RADAR_BACK_LEFT", "RADAR_BACK_RIGHT",
]

def consolidate(out_root):
    os.makedirs(out_root, exist_ok=True)
    print(f"Building unified trainval root at: {out_root}\n")

    # Symlink metadata directory
    meta_src = os.path.join(META_DIR, "v1.0-trainval")
    meta_dst = os.path.join(out_root, "v1.0-trainval")
    if not os.path.exists(meta_dst):
        os.symlink(meta_src, meta_dst)
        print(f"  [meta]  linked v1.0-trainval -> {meta_src}")
    else:
        print(f"  [meta]  v1.0-trainval already exists, skipping")

    # Symlink maps directory
    maps_src = os.path.join(META_DIR, "maps")
    maps_dst = os.path.join(out_root, "maps")
    if not os.path.exists(maps_dst):
        os.symlink(maps_src, maps_dst)
        print(f"  [maps]  linked maps -> {maps_src}")
    else:
        print(f"  [maps]  maps already exists, skipping")

    print()

    # Create samples/ and sweeps/ and symlink each file from all blobs
    total_linked = 0
    total_skipped = 0
    total_missing = 0

    for split_dir in ["samples"]:  # sweeps not needed for single-frame model; saves inodes
        for sensor in SENSOR_SUBDIRS:
            dst_sensor_dir = os.path.join(out_root, split_dir, sensor)
            os.makedirs(dst_sensor_dir, exist_ok=True)

            linked = 0
            skipped = 0
            missing = 0

            for blob in BLOB_DIRS:
                src_sensor_dir = os.path.join(blob, split_dir, sensor)
                if not os.path.isdir(src_sensor_dir):
                    missing += 1
                    continue
                for fname in os.listdir(src_sensor_dir):
                    src = os.path.join(src_sensor_dir, fname)
                    dst = os.path.join(dst_sensor_dir, fname)
                    if os.path.exists(dst):
                        skipped += 1
                        continue
                    os.symlink(src, dst)
                    linked += 1

            total_linked += linked
            total_skipped += skipped
            total_missing += missing
            status = f"{linked:5d} linked  {skipped:5d} skipped"
            if missing:
                status += f"  ({missing} blobs missing this sensor — normal)"
            print(f"  {split_dir}/{sensor}: {status}")

    print(f"\nTotal: {total_linked} symlinks created, {total_skipped} already existed")
    print(f"\nVerify with:")
    print(f"  python -c \"from nuscenes.nuscenes import NuScenes; "
          f"n = NuScenes('v1.0-trainval', '{out_root}', verbose=True); "
          f"print(f'Scenes: {{len(n.scene)}}, Samples: {{len(n.sample)}}')\"")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="/media/user/Transcend/data123/v1.0-trainval-unified",
        help="Output path for unified dataroot"
    )
    args = parser.parse_args()
    consolidate(args.out)
