# Data and model resources

## Included in Git

- Original orchestration, adapters, configuration, and checks.
- Aggregate acceptance reports, environment metadata, and provenance diffs.
- Two representative frames documenting the final composite and RGB-based depth.
- No source video, raw sensor data, full frame sequence, or learned weights.

## Input contract

The stage-one entry needs more than a video:

```text
test_data/
├── manifest.json
├── camera_parameters.json
├── frame_alignment.csv
└── rgb/frames/frame_000000.jpg ...
```

The manifest records source-frame mapping, timestamps, resolution, frame count,
and camera intrinsics. Later stages consume exact file identities and hashes;
re-encoding a video without the timeline is insufficient.

## Resource acquisition

- WiLoR weights: official model URLs in `workflow/step_one_WiLoR_and_DynHaMR/prepare.sh`.
- MANO: registration and terms acceptance at https://mano.is.tue.mpg.de/.
- SAM 3: official Hugging Face model; access may require authentication.
- ProPainter: official release weights.
- DA3METRIC-LARGE: `depth-anything/DA3METRIC-LARGE` at the pinned revision.
- MuJoCo and Panda assets: official pinned Menagerie source.

Tokens belong in environment variables or the tool's user-level credential
store. They must not appear in Git, reports, command logs, or shell history that
is shared.
