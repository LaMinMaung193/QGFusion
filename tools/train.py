"""
Training loop skeleton. The model forward/backward wiring is complete; what's left as TODO
is everything dataset-specific (real data loading -- see datasets/nuscenes_dataset.py TODOs)
and the detection Hungarian matcher (see utils/losses.py TODO).

Usage:
    python tools/train.py --config configs/default.yaml
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import yaml
import torch
from torch.utils.data import DataLoader

from qgfusion.models.qg_fusion_model import QGFusionModel
from qgfusion.datasets.nuscenes_dataset import NuScenesMultiModalDataset, collate_fn
from qgfusion.utils.losses import occupancy_loss, completion_loss, detection_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = QGFusionModel(cfg).to(device)

    # NOTE: this will raise NotImplementedError until nuscenes_dataset.py's _load_* methods
    # are filled in -- run tools/test_forward.py first to validate the model itself.
    train_set = NuScenesMultiModalDataset(
        dataroot=cfg["dataset"]["dataroot"],
        version=cfg["dataset"]["version"],
        split="train",
        occ_gt_root=cfg["dataset"]["occ_gt_root"],
        num_cameras=cfg["dataset"]["num_cameras"],
        img_size=tuple(cfg["dataset"]["img_size"]),
    )
    train_loader = DataLoader(
        train_set,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        collate_fn=collate_fn,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"]
    )

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        for batch in train_loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

            out = model(batch)

            # TODO: replace with real GT once dataset GT loading is implemented
            loss_occ = occupancy_loss(out["occupancy_logits"], batch["gt_occupancy"])
            loss_completion = completion_loss(out["completion_logits"], batch["gt_completion"])
            loss_det = detection_loss(out["detection"], batch["gt_boxes"])  # needs matcher, see losses.py

            loss = loss_occ + loss_completion + sum(loss_det.values())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            optimizer.step()

        print(f"epoch {epoch}: loss={loss.item():.4f}")
        # TODO: checkpointing, val loop with mIoU/mAP/NDS metrics, logging (wandb/tensorboard)


if __name__ == "__main__":
    main()
