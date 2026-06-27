"""
QG-Fusion training loop.

Usage:
    python tools/train.py --config configs/default.yaml

    # disable wandb for a quick test run:
    python tools/train.py --config configs/default.yaml --no-wandb

    # resume from checkpoint:
    python tools/train.py --config configs/default.yaml --resume checkpoints/epoch_005_step_1938.pt
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import yaml
import torch
from torch.utils.data import DataLoader

from qgfusion.models.qg_fusion_model import QGFusionModel
from qgfusion.datasets.nuscenes_dataset import NuScenesMultiModalDataset, collate_fn
from qgfusion.utils.losses import (
    occupancy_loss, completion_loss, detection_loss,
    gt_boxes_to_targets, make_completion_gt,
)

# Occ3D class frequencies (approx): free(17) ~86%, occupied classes ~14% combined
# Upweight occupied classes by ~6x to counter free-space dominance
OCC_CLASS_WEIGHTS = None  # built at first use

def get_occ_weights(device):
    global OCC_CLASS_WEIGHTS
    if OCC_CLASS_WEIGHTS is None:
        import torch
        # Weight = 1/freq, normalized. Free class(17) weight=1, occupied=6
        w = torch.ones(18, device=device)
        w[:17] = 3.0  # all occupied classes upweighted
        OCC_CLASS_WEIGHTS = w
    return OCC_CLASS_WEIGHTS
from qgfusion.utils.matcher import HungarianMatcher

LOSS_WEIGHTS = {"occ": 1.0, "det_cls": 0.5, "det_box": 0.05, "completion": 0.25}


class _NoopLogger:
    def log(self, *a, **kw): pass
    def finish(self): pass

def init_logger(cfg, enabled):
    if not enabled:
        print("[logger] wandb disabled -- using console only")
        return _NoopLogger()
    try:
        import wandb
        wc = cfg.get("wandb", {})
        run = wandb.init(
            project=wc.get("project", "qgfusion"),
            name=wc.get("name", None),
            config=cfg,
            resume="allow",
        )
        print(f"[logger] wandb run: {run.name}  ({run.url})")
        return wandb
    except ImportError:
        print("[logger] wandb not installed -- falling back to console.")
        return _NoopLogger()


def save_checkpoint(model, optimizer, epoch, step, loss, cfg, ckpt_dir):
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"epoch_{epoch:03d}_step_{step}.pt")
    torch.save({
        "epoch": epoch, "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss, "cfg": cfg,
    }, path)
    print(f"  [ckpt] saved → {path}")
    return path


def load_checkpoint(path, model, optimizer=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    print(f"  [ckpt] resumed from {path}  (epoch {ckpt['epoch']}, step {ckpt['step']})")
    return ckpt["epoch"], ckpt["step"]



def gaussian_health_check(out, step, logger):
    g = out.get("gaussians", None)
    if g is None:
        return
    stats = {}
    if hasattr(g, "scale") and g.scale is not None:
        s = g.scale.detach()
        stats["scale_min"] = s.min().item()
        stats["scale_mean"] = s.mean().item()
        stats["scale_max"] = s.max().item()
    if hasattr(g, "opacity") and g.opacity is not None:
        o = g.opacity.detach()
        stats["opacity_mean"] = o.mean().item()
    if hasattr(g, "position") and g.position is not None:
        stats["pos_abs_max"] = g.position.detach().abs().max().item()
    if stats:
        print(f"  [GAUSS] s{step:05d}  "
              f"scale=({stats.get('scale_min',0):.3e}, {stats.get('scale_mean',0):.3e}, {stats.get('scale_max',0):.3e})  "
              f"opacity={stats.get('opacity_mean',0):.3f}  "
              f"pos_abs_max={stats.get('pos_abs_max',0):.1f}")
        if stats.get("scale_min", 1.0) < 1e-3:
            print(f"  !! [COLLAPSE] scale_min={stats['scale_min']:.2e}")
        if stats.get("opacity_mean", 1.0) < 0.05:
            print(f"  !! [COLLAPSE] opacity_mean={stats['opacity_mean']:.4f}")
        logger.log({"train/gauss_" + k: v for k, v in stats.items()}, step=step)

def compute_losses(out, batch, matcher, device):
    losses = {}
    if batch["gt_occupancy"] is not None:
        losses["occ"] = occupancy_loss(out["occupancy_logits"], batch["gt_occupancy"],
                                            weight=get_occ_weights(out["occupancy_logits"].device))
        comp_gt = make_completion_gt(batch["gt_occupancy"])
        losses["completion"] = completion_loss(out["completion_logits"], comp_gt)
    pc_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
    filtered_boxes = batch["gt_boxes"].clone()
    filtered_num   = batch["num_boxes"].clone()
    for b in range(filtered_boxes.shape[0]):
        n = int(batch["num_boxes"][b].item())
        boxes_b = filtered_boxes[b, :n]
        mask = (
            (boxes_b[:, 0] >= pc_range[0]) & (boxes_b[:, 0] <= pc_range[3]) &
            (boxes_b[:, 1] >= pc_range[1]) & (boxes_b[:, 1] <= pc_range[4]) &
            (boxes_b[:, 2] >= pc_range[2]) & (boxes_b[:, 2] <= pc_range[5])
        )
        kept = boxes_b[mask]
        filtered_boxes[b, :kept.shape[0]] = kept
        filtered_boxes[b, kept.shape[0]:n] = 0
        filtered_num[b] = kept.shape[0]
    targets = gt_boxes_to_targets(filtered_boxes, filtered_num)
    det = detection_loss(out["detection"], targets, matcher)
    losses["det_cls"] = det["cls_loss"]
    losses["det_box"] = det["box_loss"]
    total = sum(LOSS_WEIGHTS.get(k, 1.0) * v for k, v in losses.items())
    # Gaussian regularisers: prevent scale explosion and opacity collapse
    if "gaussians" in out:
        mean_opacity = out["gaussians"].opacity.mean()
        opacity_reg = torch.relu(0.3 - mean_opacity) * 2.0
        total = total + opacity_reg
        mean_scale = out["gaussians"].scale.mean()
        scale_reg = torch.relu(mean_scale - 5.0) * 0.1  # penalise explosion
        scale_floor_reg = torch.relu(1.0 - mean_scale) * 0.5  # penalise collapse to floor
        total = total + scale_reg + scale_floor_reg
    # Guard: if no loss terms fired (e.g. all boxes filtered), return a differentiable zero
    if not isinstance(total, torch.Tensor) or total.grad_fn is None:
        total = sum(p.sum() * 0 for p in out["occupancy_logits"].flatten()[:1])
    return losses, total


def run_val(model, val_loader, matcher, device, epoch, global_step, logger):
    model.eval()
    sums = {}
    count = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            out = model(batch)
            losses, total = compute_losses(out, batch, matcher, device)
            for k, v in losses.items():
                sums[k] = sums.get(k, 0.0) + v.item()
            sums["total"] = sums.get("total", 0.0) + total.item()
            count += 1
    means = {k: v / max(count, 1) for k, v in sums.items()}
    if "total" not in means and means:
        means["total"] = sum(LOSS_WEIGHTS.get(k, 1.0) * v for k, v in means.items() if k != "total")
    if "total" not in means and means:
        means["total"] = sum(LOSS_WEIGHTS.get(k, 1.0) * v for k, v in means.items() if k != "total")
    logger.log({f"val/{k}": v for k, v in means.items()}, step=global_step)
    loss_str = "  ".join(f"{k}={v:.4f}" for k, v in means.items())
    print(f"  [val] epoch {epoch:03d}  {loss_str}")
    return means


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataroot", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--occ-gt-root", default=None, dest="occ_gt_root")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--ckpt-dir", default="checkpoints")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--ckpt-every", type=int, default=1)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--blacklist", default=None, help="Path to bad_sample_tokens.txt")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.dataroot:    cfg["dataset"]["dataroot"]    = args.dataroot
    if args.version:     cfg["dataset"]["version"]     = args.version
    if args.split:       cfg["train"]["split"]         = args.split
    if args.occ_gt_root: cfg["dataset"]["occ_gt_root"] = args.occ_gt_root

    split  = cfg["train"].get("split", "train")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    wandb_enabled = not args.no_wandb and cfg.get("wandb", {}).get("enabled", True)
    logger = init_logger(cfg, wandb_enabled)

    model     = QGFusionModel(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=cfg["train"]["lr"],
                                  weight_decay=cfg["train"]["weight_decay"])
    matcher   = HungarianMatcher(cost_class=1.0, cost_bbox=2.5)

    start_epoch, global_step = 0, 0
    if args.resume:
        start_epoch, global_step = load_checkpoint(args.resume, model, optimizer)
        start_epoch += 1

    train_set = NuScenesMultiModalDataset(
        dataroot=cfg["dataset"]["dataroot"],
        version=cfg["dataset"]["version"],
        split=split,
        occ_gt_root=cfg["dataset"]["occ_gt_root"],
        num_cameras=cfg["dataset"]["num_cameras"],
        img_size=tuple(cfg["dataset"]["img_size"]),
        blacklist=args.blacklist,
    )
    train_loader = DataLoader(
        train_set, batch_size=cfg["train"]["batch_size"],
        shuffle=True, num_workers=cfg["train"]["num_workers"],
        collate_fn=collate_fn, pin_memory=(device == "cuda"),
    )
    print(f"Train: {len(train_set)} samples ({split}), "
          f"batch={cfg['train']['batch_size']}, {len(train_loader)} steps/epoch")

    val_cfg    = cfg.get("val", {})
    val_split  = val_cfg.get("split", split.replace("train", "val"))
    val_every  = val_cfg.get("val_every", 1)
    val_loader = None
    try:
        val_set = NuScenesMultiModalDataset(
            dataroot=cfg["dataset"]["dataroot"],
            version=cfg["dataset"]["version"],
            split=val_split,
            occ_gt_root=cfg["dataset"]["occ_gt_root"],
            num_cameras=cfg["dataset"]["num_cameras"],
            img_size=tuple(cfg["dataset"]["img_size"]),
            blacklist=args.blacklist,
        )
        val_loader = DataLoader(
            val_set, batch_size=cfg["train"]["batch_size"],
            shuffle=False, num_workers=cfg["train"]["num_workers"],
            collate_fn=collate_fn, pin_memory=(device == "cuda"),
        )
        print(f"Val:   {len(val_set)} samples ({val_split}), every {val_every} epoch(s)")
    except Exception as e:
        print(f"[val] Could not build val loader ({e}) -- skipping")

    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            out = model(batch)
            losses, total_loss = compute_losses(out, batch, matcher, device)

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            optimizer.step()

            epoch_loss += total_loss.item()
            global_step += 1

            if step % 50 == 0:
                gaussian_health_check(out, global_step, logger)

            if step % args.log_every == 0:
                loss_str = "  ".join(f"{k}={v.item():.4f}" for k, v in losses.items())
                print(f"  e{epoch:03d} s{step:04d}/{len(train_loader)}  "
                      f"total={total_loss.item():.4f}  [{loss_str}]")
                logger.log(
                    {**{"train/" + k: v.item() for k, v in losses.items()},
                    "train/total": total_loss.item(), "train/step": global_step},
                    step=global_step,
                )

        elapsed = time.time() - t0
        mean_loss = epoch_loss / len(train_loader)
        print(f"epoch {epoch:03d} | mean={mean_loss:.4f} | "
              f"{elapsed:.0f}s ({elapsed/len(train_loader)*1000:.0f}ms/step)")
        logger.log({"train/epoch_mean_loss": mean_loss}, step=global_step)

        if val_loader is not None and (epoch + 1) % val_every == 0:
            run_val(model, val_loader, matcher, device, epoch, global_step, logger)

        if (epoch + 1) % args.ckpt_every == 0:
            save_checkpoint(model, optimizer, epoch, global_step,
                            mean_loss, cfg, args.ckpt_dir)

    logger.finish()


if __name__ == "__main__":
    main()
