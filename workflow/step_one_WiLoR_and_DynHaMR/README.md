# Stage 1: 3D hand pose

This stage converts RGB frames into two camera-relative MANO hand tracks.

## Entries

- `prepare.sh`: pinned source, environments, licensed resources, and load checks.
- `run_wilor_export.sh`: raw WiLoR detections.
- `run_sam3_hand_masks.sh`: SAM 3 `hand` candidate gate.
- `run_wilor_tracks.sh`: deterministic left/right association and duplicate handling.
- `run_dynhamr.sh`: camera-relative temporal refinement.

## Main files

| File | Role |
|---|---|
| `src/export_wilor.py` | Preserve raw detections and camera projection data |
| `src/camera_init.py` | Crop-to-pixel geometry and camera-translation fit |
| `src/sam3_hand_masks.py` | Reject detections mostly outside a hand mask |
| `src/association.py` | Select two side-consistent tracks |
| `src/track_wilor.py` | Write track arrays and rejection evidence |
| `src/dynhamr_adapter.py` | Convert WiLoR to documented Dyn-HaMR inputs |
| `src/dynhamr_a1_loss.py` | Camera-relative temporal objective |
| `src/refine_dynhamr.py` | Run refinement and verify outputs |

## Contract

Input needs JPEG frames, RGB calibration, frame mapping, and timestamps.
Outputs are `left_track.npz` and `right_track.npz`; keypoints are
`[T,21,3]` meters in the RGB camera frame. Timestamp, frame, MANO, observation,
and interpolation flags must survive downstream.

## Required licenses

WiLoR source is CC BY-NC-ND 4.0. MANO and checkpoints are obtained from their
official providers. Dyn-HaMR and Hand-BMC source are MIT. Review all upstream
terms before use.

## Known boundary

The temporal path is an adaptation of upstream optimization components in a
camera-relative proxy frame. It is not the upstream full HMP/SLAM pipeline and
does not recover world camera motion. The full HaMeR/DROID/HMP bundle is optional.
