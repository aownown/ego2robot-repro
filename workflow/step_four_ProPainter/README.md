# Stage 4: Human removal

ProPainter restores the background under arm masks. The formal complete-clip
entry is `run_full.sh`.

```bash
bash run_full.sh /path/to/new/output /path/to/prepared/arm-shadow-input
```

The prepared input must contain matching RGB and mask frames, `manifest.json`,
and `frame_mapping.csv`; it is not sufficient to sort files independently.

## Full-run preset

The accepted run used:

```text
resolution: 640x360
neighbor_length: 10
ref_stride: 10
subvideo_length: 80
mask_dilation: 4
raft_iter: 20
fps: 30
fp16
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

This preset lets a 450-frame input run within the tested GPU capacity. It is a
deliberate quality/capacity tradeoff. Lowering only total frame count does not
guarantee lower transformer memory because multiple adjacent frames enter an
attention operation.

## Verified integration fix

`test_data/_test1019/reports/provenance/step_four_ProPainter__ProPainter.diff`
records the tested one-line-per-call integration change:
`macro_block_size=1` prevents a nominal 640x360 output from becoming 640x368.
The model itself is unchanged.

## Known limitations

Sampled review still shows hand/shadow ghosting and local blur. The final
composite inherits those artifacts. Background quality is explicitly
`false`/unaccepted in the run metadata.
