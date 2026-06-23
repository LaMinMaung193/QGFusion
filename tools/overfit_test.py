"""
Phase 3 — Overfit / Sanity Test
================================
Fixes N_SAMPLES samples from nuScenes-mini and trains until loss -> near zero.
Confirms the model can memorize before committing to a full training run.

Usage:
    python tools/overfit_test.py --config configs/default.yaml
    python tools/overfit_test.py --config configs/default.yaml --no-wandb
    python tools/overfit_test.py --config configs/default.yaml --steps 1000 --n-samples 5
    python tools/overfit_test.py --config configs/default.yaml --resume checkpoints/overfit_step_0500.pt

What to watch:
    - All loss terms should trend downward; total should reach < 0.5 by step ~300-500
    - [GAUSS] lines: scale_min should stay > 0.001, opacity_mean should stay > 0.05
    - If scale_min < 0.001 or opacity_mean < 0.02 -> Gaussian collapse; see fix notes below

Collapse fixes (if needed):
    query_to_gaussian.py:
        scale:   .clamp(min=-3, max=10)   # prevents exp -> 0
        opacity: add floor:  opacity = opacity * 0.9 + 0.05
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import yaml
import torch
from torch.utils.data import DataLoader, Subset

from qgfusion.models.qg_fusion_model import QGFusionModel
from qgfusion.datasets.nuscenes_dataset import NuScenesMultiModalDataset, collate_fn
from qgfusion.utils.losses import (
    occupancy_loss, completion_loss, detection_loss,
    gt_boxes_to_targets, make_completion_gt,
)
from qgfusion.utils.matcher import HungarianMatcher

LOSS_WEIGHTS = {"occ": 1.0, "det_cls": 0.5, "det_box": 0.05, "completion": 0.25}

COLLAPSE_SCALE_MIN   = 1e-3
COLLAPSE_OPACITY_MIN = 0.02
EXPLODE_SCALE_MAX    = 1e4


class _NoopLogger:
    def log(self, *a, **kw): pass
    def finish(self): pass


def init_logger(cfg, enabled, run_name="overfit"):
    if not enabled:
        print("[logger] wandb disabled")
        return _NoopLogger()
    try:
        import wandb
        wc = cfg.get("wandb", {})
        run = wandb.init(
            project=wc.get("project", "qgfusion"),
            name=run_name,
            config=cfg,
            resume="allow",
        )
        print(f"[logger] wandb: {run.name}  ({run.url})")
        return wandb
    except ImportError:
        print("[logger] wandb not installed")
        return _NoopLogger()


def compute_losses(out, batch, matcher, device):
    losses = {}
    if batch["gt_occupancy"] is not None:
        losses["occ"] = occupancy_loss(out["occupancy_logits"], batch["gt_occupancy"])
        comp_gt = make_completion_gt(batch["gt_occupancy"])
        losses["completion"] = completion_loss(out["completion_logits"], comp_gt)
    # Filter GT boxes to pc_range so out-of-range annotations don't inflate det_box loss
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
    # Gaussian regularisers
    if "gaussians" in out:
        mean_opacity = out["gaussians"].opacity.mean()
        opacity_reg = torch.relu(0.3 - mean_opacity) * 2.0
        total = total + opacity_reg
        mean_scale = out["gaussians"].scale.mean()
        scale_reg = torch.relu(mean_scale - 5.0) * 0.1
        total = total + scale_reg
    return losses, total


def gaussian_health(out):
    g = out.get("gaussians", None)
    if g is None:
        return {}
    stats = {}
    if hasattr(g, "scale") and g.scale is not None:
        s = g.scale.detach()
        stats["scale_min"]  = s.min().item()
        stats["scale_max"]  = s.max().item()
        stats["scale_mean"] = s.mean().item()
    if hasattr(g, "opacity") and g.opacity is not None:
        o = g.opacity.detach()
        stats["opacity_min"]  = o.min().item()
        stats["opacity_max"]  = o.max().item()
        stats["opacity_mean"] = o.mean().item()
    if hasattr(g, "position") and g.position is not None:
        p = g.position.detach()
        stats["pos_abs_max"] = p.abs().max().item()
    return stats


def check_collapse(stats, step):
    if not stats:
        return
    if stats.get("scale_min", 1.0) < COLLAPSE_SCALE_MIN:
        print(f"  !! [COLLAPSE] step {step}: scale_min={stats['scale_min']:.2e} "
              f"< {COLLAPSE_SCALE_MIN:.0e}  ->  clamp scale_head min=-3 in query_to_gaussian.py")
    if stats.get("scale_max", 0.0) > EXPLODE_SCALE_MAX:
        print(f"  !! [EXPLODE]  step {step}: scale_max={stats['scale_max']:.2e} "
              f"> {EXPLODE_SCALE_MAX:.0e}  ->  reduce scale_head clamp max")
    if stats.get("opacity_mean", 1.0) < COLLAPSE_OPACITY_MIN:
        print(f"  !! [COLLAPSE] step {step}: opacity_mean={stats['opacity_mean']:.4f} "
              f"< {COLLAPSE_OPACITY_MIN}  ->  add opacity floor in query_to_gaussian.py")


def save_checkpoint(model, optimizer, step, loss, cfg, ckpt_dir):
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"overfit_step_{step:04d}.pt")
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "cfg": cfg,
    }, path)
    print(f"  [ckpt] saved -> {path}")
    return path


def load_checkpoint(path, model, optimizer=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    start = ckpt.get("step", 0) + 1
    print(f"  [ckpt] resumed from {path}  (step {ckpt.get('step', '?')})")
    return start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      default="configs/default.yaml")
    parser.add_argument("--n-samples",   type=int, default=5)
    parser.add_argument("--steps",       type=int, default=500)
    parser.add_argument("--lr",          type=float, default=3e-4)
    parser.add_argument("--log-every",   type=int, default=10)
    parser.add_argument("--gauss-every", type=int, default=50)
    parser.add_argument("--ckpt-every",  type=int, default=250)
    parser.add_argument("--ckpt-dir",    default="checkpoints")
    parser.add_argument("--sample-indices", type=int, nargs="+", default=None)
    parser.add_argument("--resume",      default=None)
    parser.add_argument("--no-wandb",    action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    wandb_enabled = not args.no_wandb and cfg.get("wandb", {}).get("enabled", True)
    logger = init_logger(cfg, wandb_enabled, run_name="overfit")

    model     = QGFusionModel(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    matcher   = HungarianMatcher(cost_class=1.0, cost_bbox=2.5)

    start_step = 0
    if args.resume:
        start_step = load_checkpoint(args.resume, model, optimizer)

    full_set = NuScenesMultiModalDataset(
        dataroot=cfg["dataset"]["dataroot"],
        version=cfg["dataset"]["version"],
        split=cfg["train"].get("split", "mini_train"),
        occ_gt_root=cfg["dataset"]["occ_gt_root"],
        num_cameras=cfg["dataset"]["num_cameras"],
        img_size=tuple(cfg["dataset"]["img_size"]),
    )

    if args.sample_indices is not None:
        indices = args.sample_indices
    else:
        n = min(args.n_samples, len(full_set))
        indices = list(range(n))

    subset = Subset(full_set, indices)
    loader = DataLoader(
        subset, batch_size=1, shuffle=False,
        num_workers=2, collate_fn=collate_fn, pin_memory=(device == "cuda"),
    )

    print(f"Overfitting on {len(indices)} fixed samples: indices {indices}")
    print(f"Steps: {args.steps}  LR: {args.lr}  log_every: {args.log_every}  gauss_every: {args.gauss_every}")
    print("-" * 70)

    model.train()
    step      = start_step
    data_iter = iter(loader)
    t_start   = time.time()
    loss_history = []

    while step < args.steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        out = model(batch)
        losses, total_loss = compute_losses(out, batch, matcher, device)

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 35.0)
        optimizer.step()

        loss_val = total_loss.item()
        loss_history.append((step, loss_val))

        if step % args.log_every == 0:
            loss_str = "  ".join(f"{k}={v.item():.4f}" for k, v in losses.items())
            elapsed  = time.time() - t_start
            ms_per   = elapsed / max(step - start_step, 1) * 1000
            print(f"  s{step:04d}/{args.steps}  total={loss_val:.4f}  [{loss_str}]  {ms_per:.0f}ms/step")
            logger.log(
                {**{"overfit/" + k: v.item() for k, v in losses.items()},
                 "overfit/total": loss_val},
                step=step,
            )

        if step % args.gauss_every == 0:
            stats = gaussian_health(out)
            if stats:
                print(f"  [GAUSS] s{step:04d}  "
                      f"scale=({stats.get('scale_min',0):.3e}, {stats.get('scale_mean',0):.3e}, {stats.get('scale_max',0):.3e})  "
                      f"opacity=({stats.get('opacity_min',0):.3f}, {stats.get('opacity_mean',0):.3f}, {stats.get('opacity_max',0):.3f})  "
                      f"pos_abs_max={stats.get('pos_abs_max',0):.1f}")
                check_collapse(stats, step)
                logger.log({"overfit/gauss_" + k: v for k, v in stats.items()}, step=step)

        if args.ckpt_every > 0 and step > 0 and step % args.ckpt_every == 0:
            save_checkpoint(model, optimizer, step, loss_val, cfg, args.ckpt_dir)

        step += 1

    print("=" * 70)
    print(f"Overfit test complete -- {args.steps} steps on {len(indices)} samples")
    if loss_history:
        first_10 = [l for _, l in loss_history[:10]]
        last_10  = [l for _, l in loss_history[-10:]]
        print(f"  Loss first 10 steps: mean={sum(first_10)/len(first_10):.4f}  min={min(first_10):.4f}  max={max(first_10):.4f}")
        print(f"  Loss last  10 steps: mean={sum(last_10)/len(last_10):.4f}  min={min(last_10):.4f}  max={max(last_10):.4f}")
        drop = (sum(first_10)/len(first_10)) - (sum(last_10)/len(last_10))
        verdict = "model is learning" if drop > 0.5 else "loss barely moved -- check bugs"
        print(f"  Total drop: {drop:.4f}  ({verdict})")
    save_checkpoint(model, optimizer, step - 1, loss_history[-1][1] if loss_history else 0, cfg, args.ckpt_dir)
    logger.finish()
    print("=" * 70)


if __name__ == "__main__":
    main()
