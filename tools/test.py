"""
Evaluation skeleton.

TODO: wire up
  - mIoU for the occupancy head (standard voxel-wise IoU per class)
  - mAP / NDS for the detection head, via nuscenes-devkit's eval.detection tools
  - free-space IoU for the completion head
once GT loading (datasets/nuscenes_dataset.py) and the detection matcher (utils/losses.py)
are in place and you have a checkpoint to evaluate.
"""

if __name__ == "__main__":
    raise NotImplementedError(
        "Fill in once train.py's TODOs are resolved and you have a checkpoint to evaluate."
    )
