# Upstream patches

The repository fetches upstream code at pinned revisions. It does not vendor
third-party source.

## ProPainter image output geometry

`test_data/_test1019/reports/provenance/step_four_ProPainter__ProPainter.diff`
records the tested one-file change to ProPainter. It passes
`macro_block_size=1` to `imageio.mimwrite` so a 640×360 output is not silently
encoded at 640×368. This is a small integration fix; it does not modify the
model or change its upstream license.

## Compatibility fix included in this repository

`workflow/step_two_Retarget_and_Smooth/src/check.py` passes each Euler angle as
its own row, matching SciPy's current `Rotation.from_euler` interface. This was
found while testing installation from a clean environment.

## Submodule state

Some captured provenance records show dirty submodule or bytecode state from
upstream checkouts. The setup scripts re-fetch official revisions and initialize
submodules. Do not treat those historical dirty-state markers as a source
patent or redistribution grant.
