# QG-Fusion: Query-to-Gaussian Multi-Modal Scene Representation

Implementation skeleton for the architecture in `query_gaussian_architecture_v3.png` /
the research proposal. Every file below maps directly onto one column of that diagram.

## Repo structure -> diagram mapping

| Diagram column      | Code |
|---|---|
| INPUT / ENCODER      | `qgfusion/encoders/{camera,lidar,radar}_encoder.py` |
| QUERY GENERATION     | `qgfusion/query_generation/query_proposal_network.py` (shared across all 3 modalities) |
| FUSION               | `qgfusion/fusion/adaptive_query_fusion.py` |
| GAUSSIAN REPR.       | `qgfusion/gaussian/query_to_gaussian.py`, `qgfusion/utils/gaussian_utils.py` |
| OUTPUT HEADS         | `qgfusion/heads/{occupancy,detection,completion}_head.py` |
| (full pipeline)      | `qgfusion/models/qg_fusion_model.py` |
| (data)               | `qgfusion/datasets/nuscenes_dataset.py` |
| (training)           | `qgfusion/utils/losses.py`, `tools/train.py`, `tools/test.py` |

## Setup

```bash
pip install -r requirements.txt --break-system-packages   # or use a venv
```

The LiDAR encoder defaults to `backend: "dense"` in `configs/default.yaml`, which has zero
extra dependencies and is what `tools/test_forward.py` exercises. Once you're ready to train
for real, install `spconv` (matching your CUDA version) and switch to `backend: "spconv"` --
see the TODO in `encoders/lidar_encoder.py` for what needs wiring up there.

## First thing to run

```bash
python tools/test_forward.py
```

This builds the full model and runs one forward pass on synthetic tensors of the correct
shapes (no nuScenes, no spconv required). It should print output shapes for occupancy,
detection, and completion. **Run this before touching real data** -- it's the fastest way
to catch wiring bugs (dimension mismatches between modules) independent of dataset I/O.

## Implementation roadmap, by risk (mirrors proposal Sec 7)

**Low risk -- mostly done, some TODOs to fill in:**
- Encoders (`camera_encoder.py`, `radar_encoder.py` are functional; `lidar_encoder.py`'s
  dense backend works, spconv backend needs wiring per its TODO)
- Query generation (`query_proposal_network.py` is complete)
- Output head architectures (`occupancy_head.py`, `detection_head.py`,
  `completion_head.py` are complete; quality depends on what feeds them)
- Dataset I/O (`nuscenes_dataset.py` -- every `_load_*` method has the exact devkit call
  to use in its TODO; this is mechanical, not research-risky)

**Medium risk -- needs validation, not just implementation:**
- Query fusion stability (`adaptive_query_fusion.py` -- the gating mechanism is a guess;
  ablate against equal weighting early)
- Gaussian parameter regression (`query_to_gaussian.py` -- watch for scale/opacity
  collapse during early training, a known failure mode for Gaussian-based representations)
- Occupancy head's anisotropic splat (`occupancy_head.py` TODO -- currently isotropic for
  a working baseline; the real proposal calls for using rotation + per-axis scale)

**High risk -- deliberately out of scope for this skeleton:**
- Temporal modeling (not included; would touch every stage)
- Dynamic Gaussian Flow (proposal Sec 8, explicitly deferred future work)

## Known gaps you'll hit immediately

1. **Detection loss needs a Hungarian matcher** -- `utils/losses.py::detection_loss` raises
   `NotImplementedError` until you implement one. This is standard DETR/FUTR3D machinery;
   worth referencing FUTR3D's matcher rather than writing from scratch.
2. **Dataset `_load_*` methods raise `NotImplementedError`** -- by design, so the smoke
   test can validate the model independent of data loading. Fill these in next once you've
   confirmed `test_forward.py` passes.
3. **Box convention** -- `detection_head.py` outputs `(x, y, z, w, l, h, sin_yaw, cos_yaw)`.
   Double check this against whatever you use for `nuscenes-devkit`'s eval tools before
   wiring up `tools/test.py`'s mAP/NDS computation -- box conventions are a classic source
   of silent bugs.
4. **Token flattening in `qg_fusion_model.py`** drops positional encoding -- fine for the
   smoke test, but the Query Proposal Network's cross-attention will likely benefit from
   explicit position embeddings (camera: pixel coords + camera id; LiDAR: voxel coords)
   before this becomes the real training pipeline.

## Suggested next session

Given everything above, a natural next step is implementing `nuscenes_dataset.py`'s
`_load_lidar` and `_load_cameras` first (most mechanical, unblocks everything downstream),
then the Hungarian matcher (more involved, but isolated -- doesn't touch the rest of the
model).
