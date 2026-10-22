# Input data

The public repository contains the input contract and checksums, not the source
video or full frame sequence. The tested input is an external 15-second
first-person RGB-D capture from 6–21 seconds of its source recording.

## Expected layout

```text
test_data/
├── manifest.json
├── camera_parameters.json
├── frame_alignment.csv
└── rgb/frames/frame_000000.jpg ...
```

The original derived sample contained 450 JPEG frames at 1280x720, 30 fps, with
separate original and NSL depth arrays on an 848x480 native grid. The source
timeline was matched by nearest header timestamp, with a maximum observed RGB
timestamp delta of 266 microseconds.

Important limits from the capture:

- raw depth units/disparity interpretation were not confirmed;
- original depth was not spatially aligned to RGB;
- the RGB rotation fields were not a valid usable rotation matrix;
- no world camera trajectory or robot action annotations were included.

For a new input, generate a new manifest and frame mapping, preserve sensor
timestamps, validate camera calibration, and use new output directories. Do not
replace an accepted run in place.
