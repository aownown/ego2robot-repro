# Roadmap

## Done in the first public milestone

- Six-stage camera-relative pipeline for one 450-frame, 30 fps clip.
- Pinned upstream sources and checksum/integrity checks.
- Separate environments for seven required runtimes.
- Full robot candidate search, constrained dual-arm IK, and MuJoCo rendering.
- DA3 depth from RGB plus depth-aware robot compositing.
- Independent checks for masks, timeline, geometry, videos, and input hashes.

## Next validation milestones

1. Replace camera-relative 3D hand tracks with calibrated world trajectories.
2. Incorporate aligned, unit-verified sensor depth and validate physical scale.
3. Improve hand-shadow removal and measure temporal visual quality.
4. Add scene-object collision reconstruction and real-robot safety checks.
5. Support a second robot morphology and an external contribution path.
6. Add language/task segmentation and export to a policy-training data format.
7. Publish reproducibility results on additional egocentric clips.

These are transparent research tasks, not claims already achieved by this release.
