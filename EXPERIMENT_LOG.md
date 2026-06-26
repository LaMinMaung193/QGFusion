# QG-Fusion Experiment Log
### Query-to-Gaussian Multi-Modal Scene Representation for 3D Occupancy

**Author:** La Min Maung (Liam)  
**Supervisor:** Prof. Rachael Chiang, CCU  
**Repository:** https://github.com/LaMinMaung193/QGFusion  
**Last updated:** 2026-06-24

---

## 1. System Configuration

| Item | Value |
|------|-------|
| GPU | NVIDIA RTX 3090 24GB |
| CPU | — |
| RAM | — |
| OS | Ubuntu 20.04 |
| CUDA | 12.8 |
| Python | 3.8 |
| PyTorch | — |
| spconv | 2.3.6 (cu120) |
| wandb project | qgfusion |

---

## 2. Dataset

| Item | Value |
|------|-------|
| Primary dataset | nuScenes v1.0 |
| Occupancy GT | Occ3D-nuScenes |
| Mini split (train) | 323 samples |
| Mini split (val) | 81 samples |
| Full trainval (train) | — |
| Full trainval (val) | — |
| Occ3D GT coverage (mini_train) | Index 39 onward (indices 0-38 have no GT file) |
| Occ3D GT scenes on disk | 850 scene folders |
| LiDAR points per sweep | ~34,700 |
| Radar points per frame | ~250 (merged across 5 sensors) |
| Boxes per frame (mini) | 68-77 |
| GT box coordinate frame | Ego-vehicle frame |
| GT box format | (x, y, z, w, l, h, sin_yaw, cos_yaw, class_id) |
| GT box center range (mini samples 39-43) | x,y: -53 to +155m (some outside pc_range) |
| GT box size range | 0.40 - 14.01m |
| pc_range | [-40, -40, -1, 40, 40, 5.4] meters |
| Occ3D voxel size | 0.4 x 0.4 x 0.4 m -> 200 x 200 x 16 grid |
| Completion voxel size | 0.8 x 0.8 x 0.8 m -> 100 x 100 x 8 grid |
| Occ3D semantic classes | 18 (0-16 occupied, 17 = free/empty) |
| nuScenes detection classes | 10 |

---

## 3. Model Configuration (default.yaml - Phase 4 validated)

| Hyperparameter | Value | Notes |
|----------------|-------|-------|
| embed_dim | 256 | |
| num_queries | 300 | Per proposal: keep <= 300-500 |
| camera_channels | 256 | |
| camera_backbone | ResNet-50 + FPN | ImageNet pretrained; upgrade to DINOv2 planned |
| lidar_channels | 256 | |
| lidar_backend | spconv | Switched from dense in Phase 4B |
| radar_channels | 128 | |
| gaussian_feature_dim | 128 | |
| predict_velocity | false | Future work |
| fusion_layers | 4 | |
| occ_num_classes | 18 | |
| det_num_classes | 10 | |
| Total parameters | 32,927,735 (32.9M) | Measured 2026-06-23 |

---

## 4. Training Configuration

| Hyperparameter | Phase 3 (overfit) | Phase 4 (full mini) | Notes |
|----------------|-------------------|---------------------|-------|
| batch_size | 1 | 1 | Single 3090 constraint |
| optimizer | AdamW | AdamW | |
| learning rate | 3e-4 | 2e-4 | |
| weight_decay | 0.0 | 0.01 | |
| grad_clip | 35.0 | 35.0 | |
| epochs | 500 steps | 24 | |
| num_workers | 2 | 4 | |
| AMP | No | No | Not needed: 3.34GB peak, 20.66GB headroom |
| LR scheduler | None | None | |
| loss: occ weight | 1.0 | 1.0 | |
| loss: det_cls weight | 0.5 | 0.5 | |
| loss: det_box weight | 0.05 | 0.05 | Reduced from 0.25 in Phase 3 |
| loss: completion weight | 0.25 | 0.25 | |
| opacity regulariser | relu(0.3 - mean_opacity) x 2.0 | same | |
| scale regulariser | relu(mean_scale - 5.0) x 0.1 | same | Added Phase 4 |

---

## 5. VRAM Audit (Phase 4A - 2026-06-24)

| Mode | Forward peak | Fwd+Bwd peak | Headroom |
|------|-------------|--------------|---------|
| No AMP (fp32) | 2.15 GB | 3.34 GB | 20.66 GB |
| AMP (fp16) | — | 1.85 GB | 22.15 GB |

**Decision:** AMP skipped. 20GB headroom sufficient; avoids spconv fp16 issues.

---

## 6. Phase 3 - Overfit Sanity Test Results

**Date:** 2026-06-22
**Samples:** indices [39, 40, 41, 42, 43], 500 steps, LR=3e-4

| Metric | Step 0 | Step 500 |
|--------|--------|----------|
| total loss | 5.94 | 0.73 |
| occ | 2.83 | ~0.30 |
| completion | 1.13 | ~0.17 |
| det_cls | 2.32 | ~0.10 |
| det_box | 33.4 | ~4.8 |
| scale_mean | 1.27 | ~1.0 |
| opacity_mean | 0.43 | ~0.31 |
| ms/step | 833 (cold) | 389 (steady) |

---

## 7. Phase 4 - Full Training Results (nuScenes-mini, 24 epochs)

**Date:** 2026-06-24
**Log:** logs/phase4_mini_train_v2.log
**Final checkpoint:** checkpoints/epoch_023_step_7752.pt

### 7A - Run Info

| Item | Value |
|------|-------|
| Epochs | 24 |
| Steps/epoch | 323 |
| Total steps | 7,752 |
| ms/step | 215 ms |
| Time/epoch | ~70 s |
| Total time | ~28 min |
| Peak VRAM | 3.34 GB |
| LiDAR backend | spconv |

### 7B - Loss Curves

| Epoch | Train total | Val occ | Val completion | Val det_cls | Val det_box |
|-------|-------------|---------|----------------|-------------|-------------|
| 0 | 3.45 | 1.237 | 0.477 | 1.198 | 28.71 |
| 1 | 3.25 | 1.091 | 0.383 | 1.207 | 29.32 |
| 22 | 1.48 | 0.209 | 0.168 | 2.161 | 43.61 |
| 23 | 1.48 | 0.209 | 0.168 | 3.283 | 39.03 |

**Key observations:**
- Occ val: 1.237 -> 0.209 (-83%) strong convergence
- Completion val: 0.477 -> 0.168 (-65%) strong convergence
- Det_cls val: 1.198 -> 3.283 overfitting (expected on 323-sample mini)
- Det_box val: 28.71 -> 39.03 overfitting (expected on mini)

### 7C - Gaussian Health (epoch 23)

| Stat | Value | Status |
|------|-------|--------|
| scale_min | 0.135 (exp(-2) floor) | Bimodal |
| scale_mean | 1.1-1.8 | Healthy |
| scale_max | 20.09 (exp(3) ceiling) | Bimodal |
| opacity_mean | 0.47-0.58 | Healthy, no collapse |
| pos_abs_max | 29-33 m | Gaussians spread across scene |

---

## 8. Phase 5 - Quantitative Evaluation Results

*(To be filled)*

### 8A - Occupancy mIoU (Occ3D-nuScenes, mini val)

| Class | IoU |
|-------|-----|
| barrier | — |
| bicycle | — |
| bus | — |
| car | — |
| construction vehicle | — |
| motorcycle | — |
| pedestrian | — |
| traffic cone | — |
| trailer | — |
| truck | — |
| driveable surface | — |
| other flat | — |
| sidewalk | — |
| terrain | — |
| manmade | — |
| vegetation | — |
| free | — |
| **mIoU** | — |

### 8B - Detection (nuScenes devkit)

| Metric | Value |
|--------|-------|
| mAP | — |
| NDS | — |
| mATE | — |
| mASE | — |
| mAOE | — |
| mAVE | — |

### 8C - Scene Completion

| Metric | Value |
|--------|-------|
| Completion IoU (free-space) | — |
| Completion IoU (occupied) | — |

---

## 9. Phase 6 - Ablation Study Results

*(To be filled)*

### Ablation A - Gaussian Representation (Core Contribution)

| Variant | mIoU | mAP | NDS | Completion IoU | Notes |
|---------|------|-----|-----|----------------|-------|
| A1: Full model (Queries -> Gaussians -> Heads) | — | — | — | — | Proposed method |
| A2: No Gaussian (Queries -> Heads directly) | — | — | — | — | Direct fusion baseline |
| Delta (A1 - A2) | — | — | — | — | Proves Gaussian value |

### Ablation B - Radar Contribution

| Variant | mIoU | mAP | NDS | Notes |
|---------|------|-----|-----|-------|
| B1: Camera + LiDAR + Radar | — | — | — | Full model |
| B2: Camera + LiDAR only | — | — | — | Radar ablated |
| Delta | — | — | — | |

### Ablation C - Query Count

| num_queries | mIoU | mAP | ms/step |
|-------------|------|-----|---------|
| 100 | — | — | — |
| 200 | — | — | — |
| 300 | — | — | — |
| 500 | — | — | — |

### Ablation D - Fusion Strategy

| Fusion type | mIoU | mAP |
|-------------|------|-----|
| AdaptiveQueryFusion (learned gating) | — | — |
| Simple mean pooling | — | — |
| Concatenation + linear | — | — |

---

## 10. Phase 8 - Baseline Comparison

*(To be filled)*

| Method | Backbone | mIoU | mAP | NDS | Params | Source |
|--------|----------|------|-----|-----|--------|--------|
| BEVFusion (MIT) | Swin-T + VoxelNet | — | — | — | — | Paper |
| SparseOcc | — | — | — | — | — | Paper |
| FUTR3D | — | — | — | — | — | Paper |
| QG-Fusion (ours) | ResNet-50 + spconv | — | — | — | 32.9M | This work |

---

## 11. Bugs and Fixes Log

| Phase | File | Bug | Fix |
|-------|------|-----|-----|
| 0 | lidar_encoder.py | spconv Z/X axis swap | Removed extra reversed() |
| 2 | losses.py | F.cross_entropy 5D Python 3.8 incompatibility | Reshape to (B,C,-1) |
| 2 | losses.py | Double .permute() from sed patch | Reverted |
| 2 | nuscenes_dataset.py | Occ3D path assumed {token}/labels.npz | Fixed to {token}.npz |
| 2 | occupancy_head.py | OOM on backward | Gradient checkpointing chunk_size=4000 |
| 2 | train.py | dict|dict Python 3.9+ syntax | Replaced with {**d1,**d2} |
| 3 | query_to_gaussian.py | Scale explosion clamp(max=10) | Tightened to clamp(min=-2,max=3) |
| 3 | query_to_gaussian.py | Opacity collapse to ~0 | Floor: * 0.9 + 0.05 |
| 3 | detection_head.py | Box size = Gaussian scale x residual -> 1000m | Decouple: exp(clamp(min=-1,max=2.7)) |
| 3 | matcher.py | Raw L1 cost dominates cls cost | Normalise div40/div10 |
| 3 | train.py | GT boxes outside pc_range inflate loss | Filter to pc_range before matching |
| 3 | train.py | backward() crash when all boxes filtered | Grad guard: differentiable zero |
| 4 | train.py | Scale saturating at exp(3) in full training | Scale regulariser added to train.py |
| 4 | train.py | Opacity collapse in full training | Opacity regulariser missing from train.py - added |
| 4 | train.py | Val total=0.0000 despite nonzero losses | Compute total from weighted sum in run_val |

---

## 12. Design Decisions

| # | Decision | Choice | Rationale | Revisit? |
|---|----------|--------|-----------|----------|
| 1 | Occupancy GT | Occ3D-nuScenes | Standard benchmark | No |
| 2 | Completion GT | Downsample occ_gt div2, class 17->free | Simple, consistent | No |
| 3 | det_box loss weight | 0.05 | Prevent box regression dominating early | After Phase 5 |
| 4 | Scale clamp | exp(min=-2,max=3) = [0.135m, 20.1m] | Physically meaningful | After Phase 5 |
| 5 | Box size param | exp(clamp(min=-1,max=2.7)) = [0.37m,14.9m] | Covers nuScenes range 0.4-14m | No |
| 6 | Matcher normalisation | div40 center, div10 size | Prevents L1 dominating cls cost | No |
| 7 | Camera backbone | ResNet-50 + FPN | Fast for dev; upgrade to DINOv2 for paper | Yes - Phase 5+ |
| 8 | LiDAR backend | spconv (switched Phase 4B) | Correct per proposal | No |
| 9 | AMP | Skipped | 20GB headroom, avoids spconv fp16 issues | Revisit if batch increases |
| 10 | Background class | Not added | Deferred; architecture change | Yes - Phase 5B |
| 11 | QPN positional encoding | Dropped | Deferred; TODO | After Phase 5 |
| 12 | Opacity regulariser | relu(0.3-mean_opacity) x 2.0 | Prevents occ head zeroing opacities | No |
| 13 | Scale regulariser | relu(mean_scale-5.0) x 0.1 | Prevents scale ceiling saturation | No |

---

## 13. Paper Notes

**Key contribution:**
> QG-Fusion: multi-modal queries fused and decoded into Gaussian primitives as an explicit
> intermediate 3D scene representation for occupancy, detection, and completion.

**Novelty:**
1. Gaussian primitives as intermediate representation - explicit, interpretable
2. Modality-aware query generation before fusion
3. Single framework for three output tasks via shared Gaussian scene

**Numbers for paper:**
- Total parameters: 32.9M
- Training time/epoch: ~70s (215ms/step) on RTX 3090
- Peak VRAM: 3.34 GB (fp32, batch=1)
- Val occ after 24 epochs mini: 0.209
- Val completion after 24 epochs mini: 0.168
- mIoU (mini val): Phase 5

**Limitations to acknowledge:**
- Single-GPU; full-scale is extension
- Detection overfits on mini (323 samples insufficient)
- No background class in detection head
- QPN positional encoding dropped
- Gaussian scale bimodal on mini (resolves on larger data)

**Reviewer questions to prepare:**
1. Why Gaussians over voxels? explicit, continuous, compact, differentiable
2. Why queries over direct BEV? modality-aware, sparse, DETR-compatible
3. Radar contribution? Ablation B
4. Compute overhead? 215ms/step, 32.9M params
5. vs GaussianFormer/SDGOCC? Phase 8 baseline table
6. Detection overfitting? Mini limitation; occupancy is primary metric

**Target venues:**
- IROS 2027 / ICRA 2027 (primary)
- ECCV 2026 workshop (faster)
- RA-L (journal backup)
---

## 14. Phase 5 — Evaluation Results (Mini Checkpoint, Weighted Loss)

**Date:** 2026-06-25
**Checkpoint:** checkpoints/epoch_023_step_7752.pt (weighted loss, 24 epochs mini)
**Split:** mini_val (41 samples with Occ3D GT out of 81 total)

### Occupancy mIoU — mini_val

| Class | IoU | GT voxels | Notes |
|-------|-----|-----------|-------|
| barrier | 0.00% | 1,646 | |
| bicycle | N/A | 0 | Not present in eval set |
| bus | 0.00% | 281 | |
| car | 0.00% | 40,826 | |
| construction_veh | 0.00% | 160,941 | |
| motorcycle | N/A | 0 | |
| pedestrian | 0.00% | 8,537 | |
| traffic_cone | 0.00% | 9,334 | |
| trailer | N/A | 0 | |
| truck | N/A | 0 | |
| driveable_surface | 0.00% | 29,255 | |
| other_flat | 0.00% | 303,834 | 762 pred voxels — barely activating |
| sidewalk | N/A | 0 | |
| terrain | 0.00% | 75,970 | |
| manmade | 0.00% | 113,373 | |
| vegetation | 0.00% | 308,368 | |
| free | 0.00% | 767,927 | |
| free_17 | 74.57% | 5,336,368 | Model predicts mostly free |
| **mIoU** | **5.74%** | | Inflated by free class only |

### Scene Completion — mini_val
| Metric | Value |
|--------|-------|
| Free IoU | 91.92% |
| Occupied IoU | 0.00% |
| Mean IoU | 45.96% |

### Diagnosis
- Model predicts class 17 (free) for almost all voxels
- 6x class weight insufficient at mini scale (~284 GT samples only)
- Trainval expected to show non-zero occupied IoU — that is the paper result
- Mini = proof of concept only

---

## 15. Architecture Note — OccupancyHead Rasterization Fix (Paper-worthy)

### What we found
Original implementation used isotropic mean-sigma splatting:
    sigma = scale.mean(-1)  # scalar per Gaussian
    weight = opacity * exp(-0.5 * dist^2 / sigma^2)

This caused every voxel to receive contributions from all 300 Gaussians
(each covering ~1700 voxels at scale_mean=2.6m). Result: uniform feature
vectors everywhere → model always predicted free (class 17), pred_count=0
for all occupied classes even after 14 epochs on 14,749 samples.

### The fix
Anisotropic splatting with distance cutoff and weight normalization:
    mahal2 = (dx/sx)^2 + (dy/sy)^2 + (dz/sz)^2   # per-axis
    mask = (mahal2 <= 3.0^2)                         # 3-sigma cutoff
    weight = opacity * exp(-0.5 * mahal2) * mask
    weight = weight / weight.sum()                   # normalize

Smoke test with random weights:
  Before fix: pred classes = [17] (always free)
  After fix:  pred classes = [6, 9] (pedestrian, truck — spatially varied)

### Paper framing
Frame as a design contribution in Section 4 (Method):
"Unlike prior works that use isotropic Gaussian splatting [cite], we use
anisotropic axis-aligned Gaussians with a Mahalanobis distance cutoff,
which prevents distant Gaussians from uniformly blurring voxel features
and enables spatially localized occupancy predictions."

This is directly relevant to the paper's core claim: Gaussian representation
as an effective intermediate for occupancy. The anisotropic formulation is
what makes the Gaussians actually useful for this task.

### Evidence to report in paper
- Ablation: isotropic vs anisotropic splatting (pred_count=0 vs non-zero IoU)
- This becomes Ablation F in the ablation study table
- Add to Section 11 (Design Decisions) as decision #14
