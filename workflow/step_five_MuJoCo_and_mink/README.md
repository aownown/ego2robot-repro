# Stage 5: Dual-arm robot synthesis

This stage searches robot base placements, solves constrained IK, and renders a
synchronized robot stream.

```bash
bash run.sh --targets /path/to/stage2 --background /path/to/stage4 \
  --config configs/panda_production.yaml --workers 8 --output /path/to/new/output

envs/mujoco_mink/bin/python src/check_production.py --output /path/to/new/output
```

`run.sh` uses the formal `run_production.py`. The older debug pipeline is
available through `run_debug.sh` for historical comparison only.

## Search and verification

- Search a base-position grid around the target workspace.
- Reject candidates that fail constrained IK on representative keyframes.
- Check joint limits, robot self-collision, cross-arm collision, camera keepout,
  and image/workspace occupancy.
- Re-run all checks over the full 450-frame clip for selected dual-arm pairs.
- Render the accepted combination into robot RGB, float Z depth, robot mask,
  gripper mask, and viewing videos.

The tested candidate set had 11,025 raw placements per arm: 5,130 and 5,220
distance survivors for left/right, 186 and 2,155 accepted keyframe candidates,
and 25 selected dual-arm combinations that passed full verification. Of the 450
frames, 271 met the strict motion/geometric gate; the right gripper width
exceeded the configured model limit on 177 frames. That limitation is preserved
rather than silently clipped away.

## Boundary

The implementation uses two Panda arms with unverified physical hand scale.
Self/cross-arm/camera checks do not establish collision validity against the
real scene. The robot stream feeds visual data synthesis only and has not been
executed on real hardware.
