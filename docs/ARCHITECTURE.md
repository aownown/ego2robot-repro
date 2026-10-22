# Architecture

## Pipeline branches

The implementation has one shared input and two branches that rejoin at robot
rendering:

1. The action branch runs WiLoR, SAM 3 hand gating, deterministic track
   association, a documented camera-relative Dyn-HaMR adaptation, retargeting,
   and temporal smoothing.
2. The visual branch runs SAM 3 arm segmentation, an optional shadow proposal,
   mask merge, and ProPainter background restoration.
3. Base candidates are evaluated on keyframes against constrained IK, joint
   limits, robot collisions, cross-arm collisions, camera keepout, and image
   occupancy. Accepted dual-arm combinations are rechecked over the full clip.
4. MuJoCo emits robot RGB, geometric masks, gripper masks, and float Z depth.
5. DA3 estimates scene depth from the original RGB. A deterministic compositor
   compares scene and robot Z while protecting the arm-only human mask.

## Why separate environments

WiLoR and Dyn-HaMR, SAM 3, ProPainter, MuJoCo/mink, and DA3 have incompatible
Python, PyTorch, CUDA, and optional-extension constraints. Isolation keeps those
requirements explicit and avoids silently changing an already accepted model
runtime.

## Evidence before output

Every stage writes a manifest with input paths, hashes, settings, dimensions,
and quality flags. Independent validation then checks file identity, frame
timeline, grid alignment, binary masks, rotations, collision constraints,
video properties, and compositing pixels. The acceptance report records both
successful and unresolved quality concerns.

## Coordinate and unit boundary

Hand targets remain in RGB camera coordinates. Robot rendering uses a calibrated
camera projection. This reproduction does not recover world camera motion or
verify the source hand trajectory's physical scale. Estimated DA3 depth is not a
replacement for aligned sensor depth.
