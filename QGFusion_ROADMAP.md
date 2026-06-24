# QG-Fusion Research Roadmap
### Query-to-Gaussian Multi-Modal Scene Representation for 3D Occupancy

**Target:** Conference or journal paper accepted (Prof. Rachael Chiang, CCU)
**Core scientific claim:** Query-to-Gaussian representation is a more effective intermediate
representation than direct BEV fusion for multi-modal 3D occupancy perception.
**Dataset:** nuScenes + Occ3D-nuScenes  
**Hardware:** RTX 3090 24GB, Ubuntu 20.04, CUDA 12.8, Python 3.8

---

## Current Status

**Phases 0–4: COMPLETE**  
Full training on nuScenes-mini confirmed stable. Val occ loss −83%, completion −65% over 24 epochs.
Gaussian health maintained throughout. 215ms/step, 3.34GB VRAM.

**Next step: Phase 5 — Quantitative Evaluation**

---

## Phase 0 — Foundation (COMPLETE)

- [x] Repo skeleton, all six pipeline stages as separate modules
- [x] Camera encoder (ResNet-50 + FPN, ImageNet pretrained)
- [x] LiDAR encoder: dense backend + spconv backend (VoxelBackBone8x), GPU-verified
- [x] Radar encoder (per-point MLP + BEV projection)
- [x] Query Proposal Network (cross-attention, all three modalities)
- [x] Adaptive Query Fusion (learned gating, 4 transformer layers)
- [x] Query-to-Gaussian Generator (separate heads: pos, scale, rot, opacity, feature)
- [x] Occupancy / Detection / Completion heads
- [x] Full model wiring, smoke-tested on RTX 3090 with spconv backend
- **Bug fixed:** spconv axis-ordering (Z/X swap in `PointToVoxel` → `SparseConvTensor`)

---

## Phase 1 — Data Pipeline (COMPLETE)

- [x] nuScenes-mini camera (6, 3, 256, 704), LiDAR (~34.7k pts), radar (~250 pts) validated
- [x] Occ3D-nuScenes GT loader — flat `{token}.npz` layout confirmed
- [x] Box loader validated (68–77 boxes/frame), `collate_fn` with padding
- [x] GT coverage: Occ3D GT present from mini_train index 39 onward (indices 0–38 have no GT)
- **Bug fixed:** Occ3D uses flat token layout, not `{token}/labels.npz` subdirectory

---

## Phase 2 — Training Infrastructure (COMPLETE)

- [x] Hungarian matcher (DETR/FUTR3D-style), verified with synthetic test
- [x] All loss terms: occ (CE), completion (CE), det_cls (CE), det_box (L1)
- [x] Gradient-checkpointed OccupancyHead (chunk_size=4000, saves ~2.3GB)
- [x] `train.py`: full loop, checkpointing, resume, wandb logging, val loop
- [x] 24 epochs on nuScenes-mini confirmed at 204ms/step
- **Known gap:** no background class in DetectionHead (unmatched queries unpunished)

---

## Phase 3 — Overfit Sanity Test (COMPLETE)

**Result:** total loss 5.8 → 0.73 on 5 fixed samples (indices 39–43), 500 steps.
All 4 loss terms converging. No Gaussian collapse.

### Bugs found and fixed
1. `query_to_gaussian.py`: scale clamp tightened to `(min=-2, max=3)`; opacity floor `* 0.9 + 0.05`
2. `detection_head.py`: box size decoupled from Gaussian scale → `exp(clamp(min=-1, max=2.7))`
3. `matcher.py`: bbox cost normalised by pc_range (÷40) and object size (÷10)
4. GT boxes outside pc_range filtered before matching (nuScenes annotations extend beyond ±40m)
5. `det_box` loss weight reduced 0.25 → 0.05 (prevents box regression dominating early training)
6. `train.py`: grad guard for samples where all boxes are filtered (returns differentiable zero)

### Known limitations entering Phase 4
- Scale distribution bimodal (some Gaussians at min/max clamp) — self-corrects with more data
- No background class in detection head — revisit in Phase 4
- `det_box` not fully converged at 500 steps — expected at this scale

---

## Phase 4 — Full-Scale Training (COMPLETE)

**Goal:** stable, non-diverging training run on full nuScenes-mini (then trainval).

### 4A — Mixed precision + memory audit (COMPLETE)
- [x] Profiled peak VRAM: 3.34 GB fwd+bwd (fp32), 1.85 GB with AMP
- [x] Decision: AMP skipped — 20.66 GB headroom sufficient; avoids spconv fp16 issues

### 4B — LiDAR backend switch (COMPLETE)
- [x] Switched `configs/default.yaml`: `lidar_backend: dense` → `lidar_backend: spconv`
- [x] Verified: 5-step smoke test clean, loss matches dense baseline

### 4C — Full mini_train run (COMPLETE)
- [x] 24 epochs on nuScenes-mini (323 train / 81 val), batch=1, lr=2e-4
- [x] 215ms/step, ~70s/epoch, ~28min total, 3.34 GB peak VRAM
- [x] Final checkpoint: `checkpoints/epoch_023_step_7752.pt`
- [x] Gaussian health monitor every 50 steps added to `train.py`
- [x] Gaussian regularisers (opacity + scale) added to `train.py`
- [x] Val `total=0.0` logging bug fixed

**Results:**
- Val occ: 1.237 → 0.209 (−83%) — strong convergence ✓
- Val completion: 0.477 → 0.168 (−65%) — strong convergence ✓
- Val det_cls: 1.198 → 3.283 — overfitting (expected on 323-sample mini) ⚠
- Val det_box: 28.71 → 39.03 — overfitting (expected on mini) ⚠
- Gaussian health: opacity_mean 0.47–0.58 stable, scale_mean 1.1–1.8 healthy ✓

**Bugs fixed in Phase 4:**
- Scale saturating at exp(3) ceiling during full training → scale regulariser added to `train.py`
- Opacity collapsing during full training → opacity regulariser was missing from `train.py`, added
- Val loop `total=0.0` despite nonzero individual losses → fixed weighted sum in `run_val`

### 4D — Scale to full nuScenes trainval (DEFERRED)
- [ ] Deferred pending Phase 5 results on mini
- [ ] Decision: run full trainval only if mini numbers are promising and time allows
- **Note:** full trainval on a single 3090 will take ~3–5 days. Plan accordingly.

### Notes for paper
- Report which backbone you actually used and why
- Report training time and GPU memory — reviewers care about this
- If you can only train on mini, be honest about it; frame as "proof of concept" and
  note that full-scale results are a direct extension

**Exit criteria:** ✓ checkpoint trained on nuScenes-mini with stable loss curves for all
three heads over 24 epochs.

---

## Phase 5 — Quantitative Evaluation

**Goal:** produce correct, trustworthy numbers.
A bug that inflates scores is worse than honest low numbers.

### 5A — Occupancy evaluation
- [ ] Implement per-class voxel IoU (mIoU) — standard for Occ3D-nuScenes benchmark
- [ ] Confirm class mapping: Occ3D uses 17 semantic classes + 1 free class (class 17)
- [ ] Handle ignore_index correctly — unknown/unobserved voxels must not count toward IoU
- [ ] Run on mini_val (81 samples) first to verify correctness
- [ ] Record per-class IoU table (not just mean) — paper tables show per-class breakdown
- [ ] Sanity check: free-space class (17) IoU should be highest (dominant class)

### 5B — Detection evaluation
- [ ] Verify box convention: current output is `(x,y,z,w,l,h,sin_yaw,cos_yaw)`
  — confirm this matches what `nuscenes-devkit` eval expects before trusting any number
- [ ] Implement mAP and NDS via `nuscenes-devkit` official eval tools
- [ ] Add background class (no-object) to DetectionHead — required for honest mAP
  - Change `cls_head` from `num_classes` to `num_classes+1` outputs
  - Update loss to penalise unmatched predictions

### 5C — Scene completion evaluation
- [ ] Free-space IoU on downsampled 100×100×8 grid
- [ ] Cross-check downsampling logic (`make_completion_gt`) against GT distribution

### 5D — Results table
- [ ] Format: one row = one model variant, columns = mIoU / mAP / NDS / completion-IoU
- [ ] Show results to Prof. Chiang before moving to ablations

**Exit criteria:** results table with numbers you can defend, shown to Prof. Chiang.

---

## Phase 6 — Ablation Study (Most Important for Publication)

**Goal:** prove that the Gaussian representation adds value. This is your contribution.

Design all ablation variants NOW so each can be trained in a single pass.
Each variant is a config change + checkpoint — no code rewrite needed.

### Ablation A — The core contribution (mandatory)
**"Does the Gaussian stage actually help?"**
- [ ] Variant A1 (full model): Queries → Gaussians → Heads ← *your method*
- [ ] Variant A2 (no Gaussian): Queries → Heads directly (bypass `query_to_gaussian.py`,
  pass `Qf` features directly into each head)
- This ablation alone justifies the paper. If A1 > A2: Gaussians add value. Done.

### Ablation B — Radar contribution
**"Does radar actually help?"**
- [ ] Variant B1: Camera + LiDAR + Radar ← full model
- [ ] Variant B2: Camera + LiDAR only (zero out radar input or skip radar encoder)
- Hypothesis: radar helps detection (velocity) more than occupancy

### Ablation C — Query count sensitivity
**"How many queries do you need?"**
- [ ] N = 100, 200, 300 (default), 500
- Expect diminishing returns after ~300 — confirms the proposal's N≤300 recommendation

### Ablation D — Fusion strategy
**"Does learned adaptive fusion matter?"**
- [ ] AdaptiveQueryFusion (current, learned gating)
- [ ] Simple mean pooling across modalities (degenerate baseline)
- [ ] Concatenation + linear projection (strong baseline)

### Ablation E — Gaussian count (if time allows)
- [ ] N_gaussian = 100, 300 (default), 500, 1000

### Notes for paper
- Run all ablations on the same train/val split and same number of epochs
- Report standard deviation if you can run 3 seeds (even 2 is better than 1)
- Table format: one row per variant, columns = all metrics

**Exit criteria:** ablation table with at least A and B complete. C and D strengthen the paper.

---

## Phase 7 — Visualization

**Goal:** show what the model is doing internally. Reviewers respond strongly to visualizations.

- [ ] Visualize Gaussian positions and scales overlaid on BEV (bird's-eye view)
  — use matplotlib scatter with ellipse patches scaled by Gaussian σ
- [ ] Visualize predicted occupancy vs. ground truth (side-by-side voxel grid, 3 classes:
  free / occupied / unknown)
- [ ] Visualize query attention heatmaps on camera images (where does each modality's QPN
  attend to extract queries?)
- [ ] For detection: show matched vs. unmatched predictions on a sample frame
- [ ] Pick 3–5 diverse scenes: clear scene, crowded intersection, partially occluded objects

### Tools to build
- [ ] `tools/visualize_gaussians.py` — BEV Gaussian scatter
- [ ] `tools/visualize_occupancy.py` — predicted vs GT voxel grid
- [ ] These can be lightweight scripts using matplotlib + the checkpoint loader

**Exit criteria:** at least 3 paper-quality figures ready for the manuscript.

---

## Phase 8 — Baseline Comparison

**Goal:** situate your method relative to published work.

### Recommended baselines
- [ ] **BEVFusion (MIT)** — strongest camera+LiDAR fusion baseline, numbers available in paper
- [ ] **SparseOcc** — occupancy-focused, good reference for mIoU
- [ ] **FUTR3D** — closest architectural ancestor (query-based multi-modal)
- [ ] **Your own ablation A2** (no Gaussian) — this is the fairest direct comparison

### Strategy
You do not need to rerun all baselines from scratch. Options in order of effort:
1. Cite published numbers directly (note dataset/split carefully — must match exactly)
2. Run official checkpoints on your val split if available
3. Reimplement only if numbers seem inconsistent with your setup

### What to write if your numbers are lower than baselines
This is expected for a single-GPU research prototype. Frame as:
- "Our method achieves competitive performance while introducing an interpretable
  Gaussian intermediate representation"
- Emphasize: visualization quality, modular design, explicit scene structure
- Show that ablation A1 > A2 — this proves the contribution regardless of absolute numbers

**Exit criteria:** comparison table with at least 2 published baselines + your method.

---

## Phase 9 — Paper Writing

**Goal:** conference-ready manuscript (target: IROS, ICRA, ECCV, or ICCV workshop).

### Recommended venue tier (be realistic)
- Tier 1 target: **IROS 2027** or **ICRA 2027** (robotics/AV focus, good fit)
- Tier 2 target: **CoRL workshop** or **ECCV workshop** (faster turnaround)
- Backup: **arXiv preprint** → journal extension (RA-L)

### Paper structure
1. Abstract — claim + result + 1 key number
2. Introduction — the problem, the gap, your approach, contributions (3 bullet points)
3. Related Work — occupancy prediction, Gaussian representations, multi-modal fusion (1.5 pages)
4. Method — architecture diagram + each module (3 pages)
5. Experiments — Phase 5 results table + Phase 6 ablation table (1.5 pages)
6. Visualization — Phase 7 figures (0.5 page)
7. Conclusion + limitations

### Writing tasks
- [ ] Draft related work section (can be done in parallel with Phase 6)
- [ ] Update architecture diagram to reflect actual implementation
  (current diagram shows DINOv2 but code uses ResNet-50 — sync these)
- [ ] Draft method section from the proposal — ~60% is already written
- [ ] Fill results tables from Phase 5–8 outputs
- [ ] Prof. Chiang review → revise → submit

---

## Open Decisions Log

| # | Decision | Status | Notes |
|---|----------|--------|-------|
| 1 | Camera backbone: ResNet-50 vs DINOv2 | Open | DINOv2 gives better numbers; ResNet-50 trains faster |
| 2 | Full trainval vs mini only | Open | Deferred — decide after Phase 5 mini results |
| 3 | Box convention (x,y,z,w,l,h,sin,cos) vs devkit format | Open | Verify before Phase 5B |
| 4 | Background class in DetectionHead | Open | Required for honest mAP; implement in Phase 5B |
| 5 | Target venue and deadline | Open | Discuss with Prof. Chiang after Phase 5 |
| 6 | Number of ablation seeds | Open | 1 seed is minimum; 2–3 strengthens claims |

---

## Key Notes for the Paper

**Architecture decisions to justify in writing:**
- Why queries as intermediate representation (not direct BEV fusion)
- Why Gaussians (not voxel features or point clouds)
- Why separate QPN per modality (not shared)
- Why AdaptiveQueryFusion gating (not simple concatenation)

**Honest limitations to acknowledge:**
- Single-GPU training limits scale; full trainval is an extension
- No background class in current detection head (affects mAP)
- Gaussian scale distribution bimodal in current training (noted in Phase 3)
- Positional encoding in QPN token flattening is currently dropped (TODO)
- Detection overfits on mini split (323 samples insufficient for 10-class detection)

**Numbers to record during training (for the paper):**
- Total parameters: **32.9M** ✓
- Training time per epoch: **~70s (215ms/step) on RTX 3090** ✓
- Peak GPU memory: **3.34 GB (fp32, batch=1)** ✓
- Val occ after 24 epochs mini: **0.209** ✓
- Val completion after 24 epochs mini: **0.168** ✓
- mIoU (mini val): *(Phase 5)*
- mAP / NDS: *(Phase 5)*

---

## Publication Readiness Estimate

| After Phase | Readiness | What's missing |
|-------------|-----------|----------------|
| Phase 3 | 0% | No evaluation results |
| Phase 4 (now) | 20% | Training stable but no metric numbers |
| Phase 5 | 45% | Numbers but no ablations |
| Phase 6 | 70% | Ablations done, no comparison |
| Phase 7–8 | 85% | Visuals + baselines done |
| Phase 9 | 100% | Paper written and submitted |