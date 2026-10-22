# Stage 2: Retarget and smooth

This stage maps the 21 landmarks of each tracked hand to a parallel-gripper TCP,
orientation, and opening width.

## Run

```bash
bash run_retarget.sh --input /path/to/stage1 --clip "$CLIP" --output /path/to/new/output
```

No neural checkpoint or GPU is required. Python 3.11 with NumPy, SciPy, and
OpenCV is used by the environment setup.

## Geometry

- virtual fingertip: `0.7 * index_tip + 0.3 * middle_tip`
- x axis: thumb tip to virtual fingertip
- y axis: world-down projection orthogonal to x
- z axis: `x cross y`
- TCP: midpoint between thumb and virtual fingertip
- opening: thumb-to-virtual-fingertip distance

Raw positions and quaternions are retained alongside smoothed values. Validation
frames must remain observable; a missing boundary or gap longer than 10 frames
fails. Short internal gaps receive linear TCP/width interpolation and SLERP and
are explicitly marked.

## Output contract

`left_targets.npz` and `right_targets.npz` include:

- `tcp_cam_m[T,3]`
- `rotation_cam_gripper[T,3,3]`
- `quaternion_xyzw[T,4]`
- `gripper_width_m[T]`
- raw values and validity/interpolation/replacement flags
- frame indices, sensor timestamps, and source frame indices

Coordinates remain camera-relative. Robot TCP transformation, gripper stroke
limits, and world calibration are handled later or remain unvalidated.

## Accepted clip metrics

Left and right TCP second-difference RMS dropped from 5.431 to 0.473 and 6.029
to 0.530 mm/frame². Maximum adjacent rotation dropped from 16.145° to 2.891° and
22.242° to 4.510°. These are trajectory-change metrics, not ground-truth
accuracy. One right-hand sample exceeded the configured robot-width range; the
flags and 177 clipped frames in the downstream report must remain visible.
