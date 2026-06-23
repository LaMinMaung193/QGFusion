# QG-Fusion Experiment Log
### Query-to-Gaussian Multi-Modal Scene Representation for 3D Occupancy

**Author:** La Min Maung (Liam)  
**Supervisor:** Prof. Rachael Chiang, CCU  
**Repository:** https://github.com/LaMinMaung193/QGFusion  
**Last updated:** 2026-06-23

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
| Occ3D GT coverage (mini_train) | Index 39 onward (indices 0–38 have no GT file) |
| Occ3D GT scenes on disk | 850 scene folders |
| LiDAR points per sweep | ~34,700 |
| Radar points per frame | ~250 (merged across 5 sensors) |
| Boxes per frame (mini) | 68–77 |
| GT box coordinate frame | Ego-vehicle frame |
| GT box format | (x, y, z, w, l, h, sin_yaw, cos_yaw, class_id) |
| pc_range | [-40, -40, -1, 40, 40, 5.4] meters |
| Occ3D voxel size | 0.4 × 0.4 × 0.4 m → 200 × 200 × 16 grid |
| Completion voxel size | 0.8 × 0.8 × 0.8 m → 100 × 100 × 8 grid |
| Occ3D semantic classes | 18 (0–16 occupied, 17 = free/empty) |
| nuScenes detection classes | 10 |

---

## 3. Model Configuration (default.yaml — Phase 3 validated)

| Hyperparameter | Value | Notes |
|----------------|-------|-------|
| embed_dim | 256 | |
| num_queries | 300 | Per proposal: keep ≤ 300–500 |
| camera_channels | 256 | |
| camera_backbone | ResNet-50 + FPN | ImageNet pretrained; upgrade to DINOv2 planned |
| lidar_channels | 256 | |
| lidar_backend | dense | Upgrade to spconv before Phase 5 |
| radar_channels | 128 | |
| gaussian_feature_dim | 128 | |
| predict_velocity | false | Future work |
| fusion_layers | 4 | |
| occ_num_classes | 18 | |
| det_num_classes | 10 | |
| **Total parameters** | **32,927,735 (32.9M)** | Measured 2026-06-23 |

---

## 4. Training Configuration

| Hyperparameter | Phase 3 (overfit) | Phase 4 (full) | Notes |
|----------------|-------------------|-----------------|-------|
| batch_size | 1 | 1 | Single 3090 constraint |
| optimizer | AdamW | AdamW | |
| learning rate | 3e-4 | 2e-4 | Production LR lower |
| weight_decay | 0.0 | 0.01 | No decay in overfit test |
| grad_clip | 35.0 | 35.0 | |
| epochs | — (500 steps) | — | |
| num_workers | 2 | 4 | |
| AMP (mixed precision) | No | Planned | |
| LR scheduler | None | — | TBD |
| loss: occ weight | 1.0 | 1.0 | |
| loss: det_cls weight | 0.5 | 0.5 | |
| loss: det_box weight | 0.05 | 0.05 | Reduced from 0.25 in Phase 3 |
| loss: completion weight | 0.25 | 0.25 | |

---

## 5. Phase 3 — Overfit Sanity Test Results

**Date:** 2026-06-22  
**Samples:** 5 fixed samples, indices [39, 40, 41, 42, 43] from mini_train  
**Steps:** 500  
**LR:** 3e-4, no weight decay, no scheduler

| Metric | Step 0 | Step 500 | Notes |
|--------|--------|----------|-------|
| total loss | 5.94 | 0.73 | |
| occ loss | 2.83 | ~0.30 | |
| completion loss | 1.13 | ~0.17 | |
| det_cls loss | 2.32 | ~0.10 | |
| det_box loss | 33.4 | ~4.8 | Not fully converged at 500 steps |
| scale_min | 0.77 | ~0.14 | exp(-2) = 0.135 floor |
| scale_mean | 1.27 | ~1.0 | Healthy range |
| scale_max | 1.65 | ~20 | exp(3) = 20.1 ceiling; bimodal distribution |
| opacity_mean | 0.43 | ~0.31 | Stable, above collapse threshold |
| pos_abs_max (m) | 0.4 | ~20 | Gaussians spreading to cover scene |
| ms/step | 833 (cold) | 389 (steady) | |

**Conclusion:** Model is learning. Gradients flow through full pipeline. No Gaussian collapse.

---

## 6. Phase 4 — Full Training Results

*(To be filled)*

### 6A — nuScenes-mini Training Run

| Item | Value |
|------|-------|
| Date | — |
| Epochs trained | — |
| Steps per epoch | 323 |
| ms/step (AMP off) | — |
| ms/step (AMP on) | — |
| Peak VRAM (GB) | — |
| Checkpoint | — |
| wandb run | — |

**Loss curves (final epoch):**

| Loss term | Train | Val |
|-----------|-------|-----|
| total | — | — |
| occ | — | — |
| completion | — | — |
| det_cls | — | — |
| det_box | — | — |

**Gaussian health (final checkpoint):**

| Stat | Value |
|------|-------|
| scale_min | — |
| scale_mean | — |
| scale_max | — |
| opacity_mean | — |
| pos_abs_max | — |

### 6B — Full Trainval Run (if completed)

| Item | Value |
|------|-------|
| Date | — |
| Epochs trained | — |
| Total training time | — |
| Final checkpoint | — |

---

## 7. Phase 5 — Quantitative Evaluation Results

*(To be filled after Phase 5)*

### 7A — Occupancy (mIoU, Occ3D-nuScenes)

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

### 7B — Detection (nuScenes devkit eval)

| Metric | Value |
|--------|-------|
| mAP | — |
| NDS | — |
| mATE | — |
| mASE | — |
| mAOE | — |
| mAVE | — |

### 7C — Scene Completion

| Metric | Value |
|--------|-------|
| Completion IoU (free-space) | — |
| Completion IoU (occupied) | — |

---

## 8. Phase 6 — Ablation Study Results

*(To be filled after Phase 6)*

### Ablation A — Gaussian Representation (Core Contribution)

| Variant | mIoU | mAP | NDS | Completion IoU | Notes |
|---------|------|-----|-----|----------------|-------|
| A1: Full model (Queries → Gaussians → Heads) | — | — | — | — | **Proposed method** |
| A2: No Gaussian (Queries → Heads directly) | — | — | — | — | Baseline: direct fusion |
| Delta (A1 − A2) | — | — | — | — | Proves Gaussian value |

### Ablation B — Radar Contribution

| Variant | mIoU | mAP | NDS | Notes |
|---------|------|-----|-----|-------|
| B1: Camera + LiDAR + Radar | — | — | — | Full model |
| B2: Camera + LiDAR only | — | — | — | Radar ablated |
| Delta (B1 − B2) | — | — | — | |

### Ablation C — Query Count

| num_queries | mIoU | mAP | ms/step | Notes |
|-------------|------|-----|---------|-------|
| 100 | — | — | — | |
| 200 | — | — | — | |
| 300 | — | — | — | Default |
| 500 | — | — | — | |

### Ablation D — Fusion Strategy

| Fusion type | mIoU | mAP | Notes |
|-------------|------|-----|-------|
| AdaptiveQueryFusion (learned gating) | — | — | Default |
| Simple mean pooling | — | — | Degenerate baseline |
| Concatenation + linear projection | — | — | Strong baseline |

### Ablation E — Gaussian Count (if time allows)

| num_gaussians | mIoU | mAP | Notes |
|---------------|------|-----|-------|
| 100 | — | — | |
| 300 | — | — | Default |
| 500 | — | — | |
| 1000 | — | — | |

---

## 9. Phase 8 — Baseline Comparison

*(To be filled after Phase 8)*

| Method | Backbone | mIoU | mAP | NDS | Params | Source |
|--------|----------|------|-----|-----|--------|--------|
| BEVFusion (MIT) | Swin-T + VoxelNet | — | — | — | — | Paper |
| SparseOcc | — | — | — | — | — | Paper |
| FUTR3D | — | — | — | — | — | Paper |
| **QG-Fusion (ours)** | ResNet-50 + MLP | — | — | — | 32.9M | This work |

---

## 10. Bugs and Fixes Log

| Phase | File | Bug | Fix |
|-------|------|-----|-----|
| 0 | `lidar_encoder.py` | spconv `PointToVoxel` Z/X axis swap | Removed extra `reversed()` call |
| 2 | `losses.py` | `F.cross_entropy` 5D incompatibility Python 3.8 | Reshape to `(B, C, -1)` |
| 2 | `losses.py` | Double `.permute()` from sed patch | Reverted to single permute |
| 2 | `datasets/nuscenes_dataset.py` | Occ3D assumed `{token}/labels.npz`, actual is `{token}.npz` | Fixed path construction |
| 2 | `heads/occupancy_head.py` | OOM: backward stored all 320 chunk intermediates | Gradient checkpointing, chunk_size=4000 |
| 2 | `train.py` | `dict \| dict` Python 3.9+ syntax | Replaced with `{**d1, **d2}` |
| 3 | `gaussian/query_to_gaussian.py` | Scale clamp max=10 → exp(10)=22km, explosion | Tightened to `clamp(min=-2, max=3)` |
| 3 | `gaussian/query_to_gaussian.py` | Opacity collapse to ~0 under occ head pressure | Added floor: `* 0.9 + 0.05` |
| 3 | `heads/detection_head.py` | Box size = Gaussian scale × residual → sizes 100–1000m | Decouple: `exp(clamp(min=-1, max=2.7))` |
| 3 | `utils/matcher.py` | Raw L1 box cost in meters dominates cls cost | Normalise by pc_range(÷40) and size(÷10) |
| 3 | `tools/train.py` | GT boxes beyond pc_range inflate det_box loss | Filter to pc_range before matching |
| 3 | `tools/train.py` | `total_loss.backward()` crash when all boxes filtered | Grad guard: differentiable zero fallback |

---

## 11. Design Decisions

| # | Decision | Choice Made | Rationale | Revisit? |
|---|----------|-------------|-----------|----------|
| 1 | Occupancy GT source | Occ3D-nuScenes | Standard benchmark, widely used | No |
| 2 | Completion GT | Downsample occ_gt ÷2, class 17→free | Simple, consistent with occ GT | No |
| 3 | det_box loss weight | 0.05 (reduced from 0.25) | Prevents box regression dominating early training | After Phase 5 |
| 4 | Scale clamp range | exp(min=-2, max=3) → [0.135m, 20.1m] | Physically meaningful for driving scene | After Phase 5 |
| 5 | Box size parameterization | exp(clamp(min=-1, max=2.7)) → [0.37m, 14.9m] | Covers nuScenes object size range (0.4–14m) | No |
| 6 | Matcher bbox normalisation | ÷40 (center), ÷10 (size) | Prevents L1 metric dominating cls cost | No |
| 7 | Camera backbone | ResNet-50 + FPN (current) | Fast for development; upgrade to DINOv2 for paper | **Yes — Phase 4** |
| 8 | LiDAR backend | dense (current) | Fast for development; switch to spconv for paper | **Yes — Phase 4** |
| 9 | Background class (detection) | Not yet added | Architecture change deferred | **Yes — Phase 5** |
| 10 | Positional encoding in QPN | Dropped (spatial info lost) | Deferred; flagged as TODO | After Phase 5 |

---

## 12. Paper Notes

*Running notes for eventual manuscript — add to this as ideas come up.*

**Key contribution framing:**
> We introduce Query-to-Gaussian Scene Representation (QG-Fusion): a multi-modal perception
> framework where modality-specific queries are fused and decoded into structured Gaussian
> primitives, which serve as an explicit intermediate 3D scene representation for occupancy
> prediction, detection, and scene completion.

**Novelty points to emphasize:**
1. Gaussian primitives as intermediate representation (not just output) — explicit, interpretable
2. Modality-aware query generation before fusion (not post-hoc modality weighting)
3. Single framework for occupancy + detection + completion via shared Gaussian scene

**Limitations to acknowledge honestly:**
- Single-GPU training; full-scale results are an extension
- No background/no-object class in detection head (affects mAP)
- Positional encoding in QPN token flattening currently dropped
- Gaussian scale distribution bimodal in early training

**Potential reviewer questions to prepare for:**
1. Why Gaussians over voxels? → explicit, continuous, compact, differentiable
2. Why queries over direct BEV features? → modality-aware, sparse, DETR-compatible
3. How does radar actually help? → Ablation B answers this
4. What is the computational overhead of the Gaussian stage? → Report ms/step breakdown
5. How does this compare to GaussianFormer / SDGOCC? → Baseline table

**Target venues (discuss with Prof. Chiang after Phase 5):**
- IROS 2027 (robotics/AV focus, good fit)
- ICRA 2027
- ECCV 2026 workshop (faster)
- RA-L (journal, longer timeline)