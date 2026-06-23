# QG-Fusion Implementation Roadmap

A phase-by-phase guide from "repo skeleton" to "results you can show Professor Chiang."
Check items off as you go. When you come back to me after a break, just say which phase
you're on and what's failing/unclear — you don't need to re-explain the whole project.

**Where you are right now:** Phases 0, 1, 2 and 3 are fully complete. Starting Phase 4.

---

## Phase 3 — COMPLETE

### Overfit sanity test results (5 samples, indices 39-43, 500 steps)
- Total loss: 5.8 → 0.73
- occ: ~0.30, completion: ~0.17, det_cls: ~0.10, det_box: 33 → 4.8
- No Gaussian collapse: opacity_mean stable 0.30–0.55, scale_mean 1–5m

### Bugs found and fixed
1. `query_to_gaussian.py`: scale clamp tightened to (min=-2, max=3); opacity floor added (* 0.9 + 0.05)
2. `detection_head.py`: box size decoupled from Gaussian scale → exp(clamp(min=-1, max=2.7))
3. `matcher.py`: bbox cost normalised by pc_range/object size to prevent L1 dominating cls cost
4. GT boxes outside pc_range filtered before matching (nuScenes annotations extend beyond ±40m)
5. det_box loss weight reduced 0.25 → 0.05 (prevents box regression from dominating early training)
6. Occ3D GT only present from dataset index 39 onward in mini_train split

### Known limitations entering Phase 4
- scale distribution bimodal (some Gaussians at min clamp, some at max) — will self-correct with more data
- det_box not fully converged at 500 steps — expected, needs full training
- No background/no-object class in detection head — unmatched predictions unpunished


---

## Phase 2 — Training Infrastructure (FULLY DONE)

Goal: a `train.py` that can actually take a gradient step on real data.

- [x] Hungarian matcher (`utils/matcher.py`) — DETR/FUTR3D-style bipartite matching,
      verified with known-assignment synthetic test (`tools/test_matcher.py`).
      Deliberate scope cut: no "no object" background class yet (needs DetectionHead
      architecture change to num_classes+1). Revisit post-Phase 3.
- [x] Completion GT gap resolved — majority-vote `downsample_occ_gt()` + `make_completion_gt()`
      in `losses.py`. Maps Occ3D 200×200×16 → 100×100×8, class 17→free(0), 0-16→occupied(1).
- [x] Loss weighting: occ=1.0, det_cls=0.5, det_box=0.25, completion=0.25. Revisit after
      Phase 3 shows which term dominates.
- [x] `occupancy_loss` / `completion_loss` fixed to `reshape(B, C, -1)` for reliable 5D
      cross_entropy across PyTorch versions.
- [x] `OccupancyHead.forward()` upgraded to gradient-checkpointed chunk loop (chunk_size=4000).
      Backward now recomputes intermediates per-chunk instead of storing all 320 simultaneously.
      Fits full pipeline on RTX 3090 24GB with batch_size=1.
- [x] `_load_occupancy_gt` robust to missing Occ3D scenes (returns None gracefully).
      Occ3D uses flat `{token}.npz` layout, not `{token}/labels.npz` as initially assumed.
- [x] `collate_fn` hardened: returns None occupancy for whole batch if any sample missing.
- [x] `train.py` fully wired: all five loss terms, weighted sum, checkpointing + resume,
      `--no-wandb` flag, CLI path overrides.
- [x] wandb logging — per-step train losses streamed to wandb, val losses logged at epoch
      end with `val/` prefix for separate panels. Project: qgfusion.
- [x] Validation loop — runs `mini_val` (81 samples) every epoch in eval mode, no gradients.
- [x] `configs/default.yaml` — added `val:` and `wandb:` sections, paths point at mini.
- [x] 24 epochs completed on nuScenes-mini, all five loss terms finite, 204ms/step on
      RTX 3090. Val loop confirmed working. wandb run synced successfully.

**Bugs caught and fixed in Phase 2:**
- Occ3D flat `.npz` layout (not subdirectory-per-token as assumed)
- Double `.permute()` from sed patch applying to already-correct code
- OOM from occupancy head backward retaining all chunk intermediates simultaneously
- `F.cross_entropy` 5D incompatibility on Python 3.8 / PyTorch 2.x
- `dict | dict` merge syntax requires Python 3.9+ (used `{**d1, **d2}` instead)
- wandb step ordering warning (mixed global_step and epoch as step= arg)

---

Goal: a `train.py` that can actually take a gradient step on real data.

- [x] Hungarian matcher (`utils/matcher.py`) — standard DETR/FUTR3D-style bipartite matching,
      verified with known-assignment synthetic test (`tools/test_matcher.py`). Design note:
      no "no object" background class for unmatched predictions yet -- revisit post-Phase 3.
- [x] `gt_boxes_to_targets()` bridge from collated batch format to matcher input
- [x] `downsample_occ_gt()` + `make_completion_gt()` in `losses.py` — majority-vote pool
      Occ3D 200×200×16 → 100×100×8 for SceneCompletionHead, class mapping 17→free, 0-16→occupied
- [x] `occupancy_loss` / `completion_loss` fixed to `reshape(B, C, -1)` form for reliable
      5D cross-entropy across PyTorch versions
- [x] `OccupancyHead.forward()` upgraded to gradient-checkpointed chunk loop -- backward
      now recomputes chunk intermediates instead of storing all 320 simultaneously (~2.3GB
      saved), making the full pipeline fit on a single 3090 24GB with batch_size=1
- [x] `_load_occupancy_gt` made robust to missing Occ3D scenes (returns None gracefully)
- [x] `collate_fn` hardened: returns None occupancy for whole batch if any sample missing
- [x] `train.py` fully wired: all five loss terms, weighted sum, checkpointing, resume
- [x] `configs/default.yaml` updated: mini paths, split key, batch_size=1
- [x] Validated: 24 epochs on nuScenes-mini completed successfully, all loss terms finite,
      204ms/step on RTX 3090. Phase 2 exit criteria met.

**Bugs caught and fixed in this phase:**
- Occ3D flat `.npz` layout (not subdirectory-per-token as assumed)
- Double `.permute()` from sed patch applying to already-correct code
- OOM from occupancy head backward storing all chunk intermediates simultaneously
- `F.cross_entropy` 5D incompatibility across PyTorch versions

---

## Phase 1 — Data Pipeline (DONE)

Goal: load real nuScenes samples into the exact tensor shapes the model already expects.

- [x] Occupancy GT source decided: Occ3D-nuScenes
- [x] `_load_lidar` — validated against real mini data (~34.7k pts/sweep, matches known
      nuScenes sweep density)
- [x] `_load_cameras` — validated (6, 3, 256, 704)
- [x] `_load_radar` — validated (~250 merged points across 5 sensors, matches expected
      radar sparsity)
- [x] `_load_boxes` — validated (68-77 boxes/frame, normal density for nuScenes)
- [x] `_load_occupancy_gt` — validated against Occ3D-nuScenes GT (200,200,16) on mini scenes
- [x] `collate_fn` pads `gt_boxes` correctly (padded to batch max, `num_boxes` tracks real count)
- [x] Full batch sanity-checked end to end on real data, both with and without occupancy GT

**Exit criteria:** met. Every loader confirmed against real nuScenes-mini data, not just
synthetic tensors.

**Known follow-up, not blocking:** trainval is split across blob folders (01-05 of a normal
10) plus a separate metadata folder, needs the symlink consolidation script before training
at scale. Not needed yet -- mini is sufficient through Phase 2's matcher work.

---

## Phase 0 — Foundation (DONE)

Goal: a model that runs end-to-end on synthetic data, on your actual GPU.

- [x] Repo skeleton built, all six diagram stages as separate modules
- [x] Camera encoder (ResNet-50 + FPN) — functional
- [x] Radar encoder (per-point MLP) — functional
- [x] LiDAR encoder, dense backend — functional, CPU-runnable
- [x] LiDAR encoder, spconv backend — implemented (VoxelBackBone8x) and GPU-tested on the
      3090. Caught and fixed a real bug along the way: `PointToVoxel`'s `grid_size` turned
      out to already be in the axis order `SparseConvTensor` expects, so an extra
      `reversed()` call was silently swapping the Z and X axes. Output now correctly comes
      out as `(B, 256, 1, 128, 128)` — vertical axis compressed to 1, horizontal axes
      symmetric, as the VoxelBackBone8x design intends.
- [x] Query Proposal Network — complete
- [x] Adaptive Query Fusion — complete (gating mechanism is an untested guess, see Phase 6)
- [x] Query-to-Gaussian Generator — complete
- [x] Occupancy / Detection / Completion heads — complete
- [x] Full model wiring (`qg_fusion_model.py`) — smoke-tested on CPU and on your 3090
- [x] `tools/test_lidar_spconv.py` passes on the 3090 with correct output shape

**Exit criteria:** `test_forward.py` passes with `lidar_backend: spconv`, on GPU, with real
voxelization (not the dense fallback).

**If you get stuck here:** paste me the traceback. spconv indexing (z/y/x order,
`indice_key` mismatches) is the most likely failure point.

---

## Phase 1 — Data Pipeline (next up)

Goal: load real nuScenes samples into the exact tensor shapes the model already expects.

- [x] Decide occupancy GT source — **open decision, bring this to me or Prof. Chiang**:
      the config defaults to 18 occupancy classes, which matches Occ3D-nuScenes. Confirm
      that's the benchmark you're targeting before writing `_load_occupancy_gt` — switching
      later means re-touching loss/eval code too.
- [x] Implement `NuScenesMultiModalDataset._load_lidar` (`nuscenes_dataset.py`)
- [x] Implement `_load_cameras` (image load + resize + normalize)
- [x] Implement `_load_radar` (merge 5 sensors, transform to ego frame)
- [x] Implement `_load_boxes` (GT 3D boxes → tensor matching `DetectionHead`'s box convention)
- [x] Implement `_load_occupancy_gt` (once Occ3D-nuScenes vs. alternative is decided)
- [x] Update `collate_fn` to pad variable-length `gt_boxes` across the batch
- [x] Sanity check: load one batch, print every tensor's shape/dtype/range, eyeball it

**Exit criteria:** `DataLoader` produces a batch where every tensor's shape matches what
`QGFusionModel.forward()` already expects, with no `NotImplementedError`s left.

**Priority:** `_load_lidar` and `_load_cameras` first (purely mechanical, devkit calls are
already spelled out in the TODOs). GT loading can come slightly after since you don't need
it to test that the encoders/forward pass work on real data.

---

## Phase 2 — Training Infrastructure

Goal: a `train.py` that can actually take a gradient step on real data.

- [x] Implement the Hungarian matcher for `detection_loss` (`utils/matcher.py`) -- standard
      DETR/FUTR3D-style bipartite matching, verified with a synthetic known-assignment test
      (`tools/test_matcher.py`). **Deliberate scope cut, not an oversight:** no explicit
      "no object" background class for unmatched predictions yet -- that needs
      `DetectionHead.cls_head` extended to `num_classes+1`, an architecture change. Revisit
      once basic training is confirmed to flow.
- [x] **New gap found while wiring this up:** `OccupancyHead` and `SceneCompletionHead` use
      different voxel resolutions (0.4m → 200×200×16 vs 0.8m → 100×100×8), but Occ3D GT only
      provides the 0.4m grid. `_load_occupancy_gt` has no downsampling step to produce a
      matching coarser GT for the completion head. Needs a decision: downsample (e.g.
      majority-vote pooling) the Occ3D semantics into the completion grid, or rethink what
      the completion head's GT source should be.
- [x] Decide loss weighting between occupancy / detection / completion (currently summed
      equally — this is a real hyperparameter, not just plumbing; revisit after Phase 3's
      overfit test shows which loss dominates)
- [x] Add checkpointing (save/resume) to `train.py`
- [x] Add logging (wandb or tensorboard — pick whichever you already use)
- [x] Add a validation loop stub that runs every N epochs (metrics come in Phase 5)
- [x] Wire `train.py` to actually use `gt_boxes_to_targets()` + the matcher (currently still
      calls the old placeholder signature)

**Exit criteria:** `train.py` runs for a few steps on a real batch without crashing, loss is
a finite number, `.backward()` doesn't error.

---

## Phase 3 — Sanity / Overfit Test (do not skip this)

Goal: prove the model *can* learn before spending real compute on it. This is the single
highest-leverage step for catching silent bugs — wrong box convention, broken matcher,
label misalignment — before they cost you a multi-day training run.

- [x] Take 1 scene (or even a handful of samples) from nuScenes-mini
- [x] Train for enough steps/epochs to deliberately overfit
- [x] Confirm loss → near zero and predictions visually match GT on those same samples
- [x] Watch specifically for **Gaussian collapse** (opacity → 0 for all Gaussians, or scale
      exploding/collapsing to a point) — this is the medium-risk failure mode flagged in
      your proposal for the Gaussian parameter regression stage. If you see it: tighten the
      `scale_head`'s clamp, consider an opacity floor, or initialize queries less randomly.

**Exit criteria:** the model memorizes a tiny dataset. If it can't, nothing downstream is
worth running — come back here before moving to Phase 4.


---

## Phase 4 — Full-Scale Training

Goal: a real training run on nuScenes, sized appropriately for a single 3090 (24GB) —
worth noting this is a meaningfully different resource envelope than the "GPU cluster"
your proposal's scalability section assumed, so budget accordingly.

- [ ] Start with nuScenes-mini to validate throughput (samples/sec) before committing to
      full trainval
- [ ] Enable mixed precision (`torch.cuda.amp`) — close to mandatory for fitting this model
      at reasonable batch size on a single 24GB card
- [ ] If you hit OOM: gradient accumulation first, then consider reducing `num_queries` or
      occupancy grid resolution before reducing batch size below 2
- [ ] Scale up to full nuScenes trainval once mini-set throughput/memory looks sane
- [ ] Track training curves — loss per head, not just total loss, so you can see if one
      task is starving the others

**Exit criteria:** a checkpoint trained on (ideally) full trainval, with stable
(non-diverging, non-collapsed) loss curves for all three heads.

---

## Phase 5 — Evaluation

Goal: numbers you can compare against literature.

- [ ] Implement occupancy mIoU (standard per-class voxel IoU)
- [ ] Implement detection mAP/NDS via `nuscenes-devkit`'s eval tools — **this is where the
      box convention TODO matters**; verify `(x,y,z,w,l,h,sin_yaw,cos_yaw)` against what the
      devkit eval expects before trusting any numbers here
- [ ] Implement completion free-space IoU
- [ ] Run eval on val split, record numbers

**Exit criteria:** a results table you'd be comfortable showing Prof. Chiang, even if the
numbers aren't great yet — getting *correct, trustworthy* numbers matters more than good
numbers at this stage. A bug in eval that inflates scores is worse than an honest bad score.

---

## Phase 6 — Research Validation (the actual contribution)

Goal: turn "a model that runs" into "evidence for the architectural choices in your
proposal." This is where the medium-risk items from Sec 7 get resolved with real ablations,
not guesses.

- [ ] Ablate the adaptive fusion gate vs. equal-weight fusion — does the learned gating in
      `AdaptiveQueryFusion` actually help, or is it dead weight?
- [ ] Ablate isotropic vs. anisotropic Gaussian splat in `OccupancyHead` (the anisotropic
      version using `quaternion_to_rotmat` is still a TODO — implement it for this ablation)
- [ ] Test with vs. without positional encoding in the token-flattening step
      (`qg_fusion_model.py` — currently drops spatial position info)
- [ ] Test with vs. without velocity prediction (`predict_velocity` in config)
- [ ] Compare against baselines: SDGOCC, MV2DFusion, and a vanilla CenterPoint/BEVFusion
      reference, using numbers from their papers or your own reruns if reproducibility is a
      concern

**Exit criteria:** for each medium-risk design choice in the proposal, you have an ablation
result, not just an architectural assumption.

---

## Phase 7 — Deferred / Stretch (explicitly out of scope until everything above works)

- [ ] Temporal modeling
- [ ] Dynamic Gaussian Flow (proposal Sec 8)

Don't start these until Phase 6 is in good shape — they touch every stage of the pipeline
and will be much easier to get right once the single-frame version is solid.

---

## Open Decisions Log

Things worth explicitly deciding (with me, or with Prof. Chiang) rather than letting
defaults silently decide for you:

1. **Occupancy GT benchmark** (Occ3D-nuScenes assumed by current config — confirm before
   Phase 1's GT loading work)
2. **Loss weighting strategy** across the three heads (revisit after Phase 3)
3. **Compute budget reality check** — single 3090 vs. the cluster-scale assumptions in your
   proposal's scalability section; worth a short note in your eventual writeup about what
   had to be scaled down and why
4. **Box convention** — confirm against `nuscenes-devkit` eval tools before trusting any
   Phase 5 detection numbers

---

## How to use this with me

When you come back, tell me the phase number and what's happening (passing/failing/stuck).
I'll have the context I need from that alone — no need to re-paste the whole project
history. If something here turns out wrong once you're deeper into a phase (e.g. the
occupancy class count needs to change), let me know and I'll update both the code and this
roadmap together so they stay in sync.
