# Workflow guide

This directory contains seven isolated runtimes that together implement the six
reproduction stages. The boundaries are intentionally file based: each stage
writes a manifest, structured arrays, masks, or frames, and the next stage
validates the identity and shape of those inputs before running.

## Directory contract

```text
workflow/
├── common/                         shared setup and checking helpers
├── step_one_WiLoR_and_DynHaMR/     hand tracking and camera-relative refinement
├── step_two_Retarget_and_Smooth/   gripper targets and temporal smoothing
├── step_three_SAM3/                arm and optional shadow masks
├── step_four_ProPainter/           human removal
├── step_five_MuJoCo_and_mink/      base search, IK, collision checks, rendering
└── step_six_DepthAnything3_and_Composite/
```

Runtime environments are created under each stage's ignored `envs/`
directory. Pinned upstream code is cloned under `repos/`. Weights are downloaded
to each stage's ignored `assets/` directory.

## Setup sequence

Set `CONDA_BIN`, `PROJECT`, `WORKFLOW`, `CLIP`, and `TEST_ROOT` as shown in the
root README. Then use the following order:

```bash
# Pinned source
bash workflow/step_one_WiLoR_and_DynHaMR/prepare.sh sources
bash workflow/prepare_all.sh sources

# Python environments
bash workflow/step_one_WiLoR_and_DynHaMR/prepare.sh wilor-env
bash workflow/step_one_WiLoR_and_DynHaMR/prepare.sh dynhamr-base
bash workflow/prepare_all.sh env

# Licensed model/data resources
bash workflow/step_one_WiLoR_and_DynHaMR/prepare.sh wilor-weights
bash workflow/step_one_WiLoR_and_DynHaMR/prepare.sh mano /path/to/MANO_RIGHT.pkl
bash workflow/step_one_WiLoR_and_DynHaMR/prepare.sh dynhamr-assets
bash workflow/step_one_WiLoR_and_DynHaMR/prepare.sh bmc
bash workflow/prepare_all.sh weights

# Loading and numerical checks
bash workflow/step_one_WiLoR_and_DynHaMR/prepare.sh wilor-check
bash workflow/prepare_all.sh check
bash workflow/prepare_all.sh model-check
```

The optional upstream `dynhamr-bundles` command downloads the full HaMeR,
DROID, and HMP resource set. It is not required for the camera-relative adapter
used here. Never treat a model load check as evidence that full-clip inference
or visual quality passed.

## Stage 1: Hand tracks

Pipeline within the stage:

1. `export_wilor.py` keeps raw battles including zero-detection frames, model
   MANO parameters, 2D keypoints, crop-to-pixel transformation, and fitted
   camera translation.
2. `src/sam3_hand_masks.py` uses the local SAM 3 runtime to reject detections
   whose keypoints are overwhelmingly outside a `hand` mask.
3. `track_wilor.py` associates candidates into left/right tracks, resolving
   same-handed duplicates by score and preserving questionable jumps.
4. `refine_dynhamr.py` runs a documented camera-relative adaptation of the
   Dyn-HaMR optimization components. It does not fabricate a world camera
   trajectory or claim to run the unmodified upstream full pipeline.

Outputs are `left_track.npz` and `right_track.npz`. Required fields include
`keypoints_cam_m[T,21,3]`, validity flags, MANO parameters, frame indices,
timestamps, and intrinsic matrices. Units are meters in RGB camera coordinates.

## Stage 2: Gripper targets

```bash
bash workflow/step_two_Retarget_and_Smooth/run_retarget.sh   --input /path/to/stage1 --clip "$CLIP" --output /path/to/new/output
```

The virtual fingertip is `0.7 * index_tip + 0.3 * middle_tip`. The TCP, handed
gripper axes, opening, and rotation are derived from the 21 landmarks. Bad
velocity outliers are replaced, then position and width are smoothed with
Savitzky-Golay; orientation uses weighted SLERP.

Short observation gaps of at most 10 frames are interpolated and flagged.
Boundary gaps and longer gaps fail rather than being silently extrapolated.
Width is nonnegative but is not clamped to a particular hardware range.

## Stage 3: Arm masks

```bash
bash workflow/step_three_SAM3/run_segment.sh   --clip "$CLIP" --config workflow/step_three_SAM3/configs/arm.yaml   --output /path/to/new/output

# Optional shadow candidate
bash workflow/step_three_SAM3/run_segment.sh   --clip "$CLIP" --config workflow/step_three_SAM3/configs/shadow.yaml   --output /path/to/new/shadow-output
```

Text-prompted SAM 3 masks are propagated bidirectionally, merged, filtered,
and resized/validated against the RGB timeline. `configs/arm.yaml` is the
default. `configs/shadow.yaml` adds spatially adjacent candidates; its visual
review showed missed regions and blocky false positives, so it is recorded as a
limitation rather than treated as a universally correct mask.

## Stage 4: Inpainting

Use `run_full.sh` for a complete clip:

```bash
bash workflow/step_four_ProPainter/run_full.sh   /path/to/new/output /path/to/prepared/arm-shadow-input
```

The prepared input must include paired 450 RGB frames and masks, a manifest,
and a frame mapping. The accepted OOM-safe preset processes at 640x360 with a
bounded subvideo length. It is a quality/capacity tradeoff, not the original
1280x720 background resolution. The output still contains hand/shadow ghosting
and local blur in sampled review; do not relabel those artifacts as accepted.

The tested one-file upstream integration patch sets
`macro_block_size=1` in `imageio.mimwrite`, preventing 640x360 output from
silently being encoded at 640x368.

## Stage 5: Robot synthesis

```bash
bash workflow/step_five_MuJoCo_and_mink/run.sh   --targets /path/to/stage2 --background /path/to/stage4   --config workflow/step_five_MuJoCo_and_mink/configs/panda_production.yaml   --workers 8 --output /path/to/new/output

workflow/step_five_MuJoCo_and_mink/envs/mujoco_mink/bin/python   workflow/step_five_MuJoCo_and_mink/src/check_production.py   --output /path/to/new/output
```

`run.sh` is the formal entrance and calls `run_production.py`.
`run_debug.sh` and `src/run_pipeline.py` reproduce the earlier simplified
baseline and are not substitutes.

The search evaluates arm base poses across a grid, solves constrained IK on
keyframes, rejects joint-limit violations, robot/cross-arm collisions, camera
keepout intersections, and excessive image occupancy, then verifies each
selected dual-arm combination over every frame. MuJoCo emits the same camera
grid, robot RGB, float Z depth, robot mask, and gripper mask.

The tested single-morphology setup uses two Panda arms. Physical input scale and
real-scene object collision are outside its validated scope.

## Stage 6: Depth and composite

```bash
bash workflow/step_six_DepthAnything3_and_Composite/run.sh   --robot /path/to/stage5 --clip "$CLIP" --check-inputs

bash workflow/step_six_DepthAnything3_and_Composite/run.sh   --robot /path/to/stage5 --clip "$CLIP"   --checkpoint workflow/step_six_DepthAnything3_and_Composite/assets/DA3METRIC-LARGE   --output /path/to/new/output

workflow/step_six_DepthAnything3_and_Composite/envs/da3/bin/python   workflow/step_six_DepthAnything3_and_Composite/src/check_pipeline.py   --output /path/to/new/output
```

DA3METRIC-LARGE runs on the original RGB. Its output is converted to metric
optical-axis Z with the actual preprocessed focal length and mapped to the robot
camera grid. The robot arm is rendered over the background; gripper occlusion is
resolved with floating-point Z comparisons outside the dilated pure-ARM mask.
The optional shadow-inclusive mask is not allowed to erase the synthesized
robot.

The depth-scale inspection is a consistency diagnostic, not sensor calibration.
Native sensor depth must be checked for units, distortion, and RGB alignment
before it can replace the DA3 estimate.

## Accepted run

The evidence in `test_data/_test1019` comes from one complete 450-frame run.
`04_run_full_pipeline.sh` is a run-specific convenience entry; stage success
markers are valid only for identical input paths and configurations. A new clip
or changed config must use new output paths and a fresh verification run.
