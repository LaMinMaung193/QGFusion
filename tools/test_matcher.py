"""
Standalone test for the Hungarian matcher + detection_loss, using a synthetic case where the
correct assignment is known in advance, so a passing test means the matching logic is
actually correct -- not just that it runs without crashing.

Usage:
    python tools/test_matcher.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from qgfusion.utils.matcher import HungarianMatcher
from qgfusion.utils.losses import detection_loss, gt_boxes_to_targets


def test_known_assignment():
    """3 predictions, 2 GT boxes. Construct predictions so pred[1] is near-perfect for GT[0]
    and pred[2] is near-perfect for GT[1], pred[0] is junk -- matcher should pick (1,0) and (2,1)."""
    num_classes = 10
    cls_logits = torch.zeros(1, 3, num_classes)
    cls_logits[0, 1, 3] = 10.0  # confidently predicts class 3
    cls_logits[0, 2, 7] = 10.0  # confidently predicts class 7

    boxes = torch.tensor([
        [[50.0, 50.0, 50.0, 1, 1, 1, 0, 1],   # pred 0: junk, far away
         [1.0, 2.0, 3.0, 4, 5, 6, 0.1, 0.9],   # pred 1: near GT 0
         [10.0, 20.0, 30.0, 1, 2, 3, 0.5, 0.5]]  # pred 2: near GT 1
    ])

    preds = {"cls_logits": cls_logits, "boxes": boxes, "velocity": torch.zeros(1, 3, 2)}
    targets = [{
        "labels": torch.tensor([3, 7]),
        "boxes": torch.tensor([
            [1.0, 2.0, 3.0, 4, 5, 6, 0.1, 0.9],   # GT 0 -- matches pred 1
            [10.0, 20.0, 30.0, 1, 2, 3, 0.5, 0.5],  # GT 1 -- matches pred 2
        ]),
    }]

    matcher = HungarianMatcher()
    indices = matcher(preds, targets)
    pred_idx, tgt_idx = indices[0]
    print(f"Matched pred indices: {pred_idx.tolist()}, GT indices: {tgt_idx.tolist()}")

    matched_pairs = set(zip(pred_idx.tolist(), tgt_idx.tolist()))
    assert matched_pairs == {(1, 0), (2, 1)}, f"Expected {{(1,0),(2,1)}}, got {matched_pairs}"
    print("Assignment correct.")

    loss = detection_loss(preds, targets, matcher)
    print(f"cls_loss={loss['cls_loss'].item():.4f}  box_loss={loss['box_loss'].item():.4f}")
    assert loss["box_loss"].item() < 1e-4, "box_loss should be ~0 for near-perfect matched boxes"
    assert loss["cls_loss"].item() < 0.1, "cls_loss should be small for confident correct predictions"
    print("Losses sane for a near-perfect match.")


def test_empty_targets():
    """0 GT boxes -- matcher and loss should handle this without crashing (common case: a
    frame with no annotated objects)."""
    preds = {
        "cls_logits": torch.randn(1, 5, 10),
        "boxes": torch.randn(1, 5, 8),
        "velocity": torch.randn(1, 5, 2),
    }
    targets = [{"labels": torch.zeros(0, dtype=torch.long), "boxes": torch.zeros(0, 8)}]

    matcher = HungarianMatcher()
    loss = detection_loss(preds, targets, matcher)
    print(f"Empty-target loss: cls_loss={loss['cls_loss'].item():.4f}  box_loss={loss['box_loss'].item():.4f}")
    print("Empty targets handled OK.")


def test_collate_bridge():
    """gt_boxes_to_targets() correctly un-pads a collated batch using num_boxes."""
    gt_boxes = torch.zeros(2, 5, 9)  # batch of 2, padded to max 5 boxes
    gt_boxes[0, :2] = torch.tensor([[1, 1, 1, 1, 1, 1, 0, 1, 3], [2, 2, 2, 1, 1, 1, 0, 1, 5]])
    gt_boxes[1, :1] = torch.tensor([[3, 3, 3, 1, 1, 1, 0, 1, 7]])
    num_boxes = torch.tensor([2, 1])

    targets = gt_boxes_to_targets(gt_boxes, num_boxes)
    assert targets[0]["boxes"].shape == (2, 8) and targets[0]["labels"].tolist() == [3, 5]
    assert targets[1]["boxes"].shape == (1, 8) and targets[1]["labels"].tolist() == [7]
    print("gt_boxes_to_targets correctly un-pads using num_boxes.")


if __name__ == "__main__":
    test_known_assignment()
    print()
    test_empty_targets()
    print()
    test_collate_bridge()
    print("\nAll matcher tests passed.")
