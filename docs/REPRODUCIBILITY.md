# Reproduction scope and evidence

## This release reproduces

- One Path B input: raw RGB to hand tracks and visual robot synthesis.
- The six interdependent stages listed in `README.md`.
- A complete 450-frame acceptance run on a camera-relative clip.
- Structured input/output contracts, variant flags, failure handling, and
  independent validation scripts.

## This release does not reproduce

- The paper's four-source, 18,561-hour dataset.
- Fifteen robot morphologies or downstream VLA pretraining.
- RoboTwin, EBench, or real-robot generalization numbers.
- World camera pose recovery or a unit-validated sensor-depth pipeline.
- Language-driven long-video segmentation and policy-data export.

## Evidence map

| Claim | Evidence |
|---|---|
| All stages completed | `reports/status.json` |
| Detailed stage metrics | `reports/results.json` |
| Visual observations and false quality flags | `reports/visual_review.json` |
| Video dimensions, frame counts, and FPS | `reports/intermediate_videos.json` |
| Input and output hashes | `reports/input_integrity.json`, `outputs/06_composite/source_hashes.json` on the artifact host |
| Environment versions | `reports/provenance/` |
| Human-readable summary | `reports/TEST_REPORT.md` |

## Quality boundaries

Three independent status flags remain false: visual quality, physical scale,
and world trajectory recovery. Robot self/cross-arm/camera checks do not imply
real-scene collision validity. A depth preview and metric DA3 output do not
imply sensor-depth alignment. Numerical completion is reproducible evidence,
not automatic scientific acceptance.
