# Stage 3: Arm and shadow masks

SAM 3 produces text-prompted masks used to remove the person's arm before robot
rendering. The default configuration is `configs/arm.yaml`.

```bash
bash run_segment.sh --clip "$CLIP" --config configs/arm.yaml --output /path/to/new/output

# Optional experimental shadow mask
bash run_segment.sh --clip "$CLIP" --config configs/shadow.yaml --output /path/to/new/shadow
```

## Implementation

1. Build a SAM 3 video predictor from the official pinned source.
2. Propagate and merge text-prompted `arm` detections in both directions.
3. Preserve the raw mask and a filtered binary mask.
4. Verify frame count, timeline, dimensions, and empty-mask behavior.
5. Optionally generate `shadow`, select spatially adjacent candidates, and merge
   them with arm masks in a separate prepared run.

The formal mask entry is the processed binary mask under `masks/`; raw masks and
preview videos are for diagnosis. SAM 3 access may require Hugging Face terms.

## Boundary

The accepted pipeline uses the pure arm mask for human protection in depth
compositing, regardless of whether shadow candidates are used for inpainting.
Shadow segmentation showed misses and blocky false positives in sampled review.
It is retained as an experimental branch, not evidence of a solved human-shadow
problem.
