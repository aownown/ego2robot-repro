#!/usr/bin/env python3
"""Inventory environment and resource files. This is not a runtime validation."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGES = [
    "step_one_WiLoR_and_DynHaMR",
    "step_two_Retarget_and_Smooth",
    "step_three_SAM3",
    "step_four_ProPainter",
    "step_five_MuJoCo_and_mink",
    "step_six_DepthAnything3_and_Composite",
]
for stage in STAGES:
    root = ROOT / stage
    envs = sorted(p.name for p in (root / "envs").glob("*") if (p / "bin/python").is_file())
    print(f"{stage}: environments={envs or 'missing'}")
    check = root / "logs/check.log"
    print(f"  check log: {check if check.is_file() else 'missing'}")
print("This command checks file presence only; full inference is separate.")
