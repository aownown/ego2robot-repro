# Six-stage acceptance run — 2026-10-19

Input: one complete clip, 450 frames at 1280x720 and 30 fps (15 seconds).

## Completion

All six stages and independent checks completed. Seven isolated Conda
environments loaded their models, and 468 input checksum entries passed.

| Stage | Result | Primary output |
|---|---|---|
| 1. Three-dimensional hands | Completed | Camera-relative left/right hand tracks |
| 2. Targets and smoothing | Completed | TCP, rotation, width, flags |
| 3. Arm and shadow masks | Completed | Processed arm and optional shadow masks |
| 4. ProPainter | Completed | 640x360 full-clip background |
| 5. MuJoCo robot | Completed | Robot RGB, depth, masks, trajectory |
| 6. DA3 composite | Completed | Composite RGB and depth |

## Independent checks

- WiLoR produced 918 raw detections. Eighteen frames contained same-handed
  duplicates; tracking retained 900 detections.
- The formal stage-five grid evaluated 11,025 raw candidates per arm. Five
  thousand one hundred thirty left and 5,220 right candidates survived distance
  checks; 186 and 2,155, respectively, passed keyframe constraints. All 25
  selected dual-arm combinations passed full 450-frame verification.
- Robot geometry was valid on 450/450 frames. The paper-style gate passed 443
  frames, and the stricter motion gate passed 271.
- The right gripper exceeded the configured width on 177 frames. The left arm
  exceeded it on zero.
- Robot self-collision, cross-arm collision, and camera-keepout collision counts
  were 0/0/0. Joint-jump count was 0.
- Stage six independently checked 3,600 input hashes and every frame.
- Twenty-three intermediate videos and four final videos were verified at 450
  frames and 30 fps.
- Six sampled frames (0, 58, 150, 225, 325, 449) aligned across stages.

## Released quality flags

```text
visual_quality_accepted      = false
physical_scale_validated     = false
world_trajectory_recovered   = false
```

The sampled visual review saw hand/shadow ghosting and local blur in the
ProPainter background, including interaction regions. DA3 depth came from RGB,
not the original sensor stream. Raw sensor depth remains unit-unverified and is
not aligned to RGB. Robot collision checks cover robot self, cross-arm, and
camera keepout; they do not validate collision with the reconstructed real scene.
No real-robot transfer was tested.

## Evidence files

- `status.json`: concise overall status and key counts.
- `results.json`: exhaustive machine-readable stage evidence.
- `visual_review.json`: sampled observations and quality boundaries.
- `intermediate_videos.json`: video frame counts, dimensions, and frame rate.
- `input_integrity.json`: frame mappings, depth data, and checksums.
- `provenance/`: tested revisions, diffs, and installed package versions.
