# Stage 6: Depth-aware compositing

This stage estimates scene depth from original RGB, aligns it with the MuJoCo
robot camera grid, and resolves occlusion.

## Run

```bash
bash run.sh --robot /path/to/stage5 --clip "$CLIP" --check-inputs
bash run.sh --robot /path/to/stage5 --clip "$CLIP" \
  --checkpoint assets/DA3METRIC-LARGE --output /path/to/new/output

envs/da3/bin/python src/check_pipeline.py --output /path/to/new/output
envs/da3/bin/python src/inspect_depth_scale.py --output /path/to/new/output
```

## Depth conversion

The metric branch uses the preprocessed focal length and the official model
conversion factor. The unscaled focal length from the original 1280x720 image
must not be applied directly to a resized network output. Scene depth is mapped
back onto the robot camera grid before comparison.

## Composite rule

- Robot arm pixels always overwrite the background.
- Gripper pixels are occluded only when valid scene Z is nearer than robot Z and
  the pixel lies outside a dilated pure-ARM human-protection mask.
- Missing depths fail validation rather than defaulting to a convenient value.

## Boundary

DA3 depth is inferred from RGB and is not measured sensor depth. The final
report calls the scale check a consistency diagnostic. `sensor depth units
verified`, `aligned sensor depth`, and `physical scale validated` remain
outside this release.
