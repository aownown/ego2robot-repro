# Data contract

The pipeline uses a file boundary between stages. Every path below is relative
to the output of the named stage.

| Boundary | Required content |
|---|---|
| Input → stages 1, 3, 6 | RGB frames, frame alignment, sensor timestamps, camera intrinsics |
| Stage 1 → 2 | Two `[T,21,3]` camera-frame keypoint tracks in meters, MANO parameters, validity flags |
| Stage 2 → 5 | `[T,3]` TCP, `[T,3,3]` gripper-to-camera rotation, `[T]` width, flags |
| Stage 3 → 4 | Binary PNG masks aligned one-to-one with RGB frames |
| Stage 4 → 5, 6 | Inpainted RGB with frame mapping, manifest, and hashes |
| Stage 4 → 6 | Final background RGB on the selected output grid |
| Stage 5 → 6 | Robot RGB, float optical-Z depth, robot mask, gripper mask, camera/intrinsics |
| Stage 6 | Composite RGB, scene depth, occlusion mask, source hashes, validation |

## Coordinate conventions

- Stage-one and stage-two motion remains in RGB camera coordinates.
- Stage-two quaternions use `xyzw` order and rotate gripper-local vectors into
  camera coordinates.
- Robot and DA3 depths are metric optical-axis Z, not Euclidean ray distance.
- A lowered output resolution changes the effective focal length and principal
  point; all intrinsics must be transformed with the image grid.

## Quality semantics

- `finite=true` means no NaN/Inf; it does not mean trustworthy.
- `observed_valid` distinguishes an observation from interpolation or
  optimization inference.
- `gap_interpolated` and `replacement_source` are preserved.
- `geometry_valid`, `paper_style_valid`, and `strict_valid` are separate robot
  checks.
- `status=succeeded` means process completion. It does not imply visual accuracy,
  physical scale, world recovery, real-scene collision validity, or hardware
  transfer.

Each stage rejects mismatched frame identity, shape, timestamp, or mask coverage
instead of silently sorting filenames.
