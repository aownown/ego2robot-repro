#!/usr/bin/env python3
"""Portable repository checks that do not require CUDA or model weights."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ERRORS: list[str] = []

required = [
    "README.md", "LICENSE", "THIRD_PARTY.md", "DATA.md", "CHANGELOG.md",
    "CITATION.cff", "workflow/sources.lock.json",
    "workflow/DATA_CONTRACT.md", "docs/RELEASING.md",
    "test_data/_test1019/reports/status.json",
]
for rel in required:
    if not (ROOT / rel).is_file():
        ERRORS.append(f"missing required file: {rel}")

for rel in ("workflow/sources.lock.json", "test_data/_test1019/reports/status.json"):
    path = ROOT / rel
    if path.is_file():
        try:
            json.loads(path.read_text())
        except Exception as exc:
            ERRORS.append(f"invalid JSON {rel}: {exc}")

for path in ROOT.rglob("*"):
    if ".venv" in path.parts or "envs" in path.parts or "repos" in path.parts:
        continue
    if path.is_file() and path.suffix in {".py", ".sh", ".md", ".json", ".yaml", ".yml"}:
        text = path.read_text(errors="replace")
        home_prefix = "/home/" + "user/"
        cloud_prefix = "/home/" + "dataset-assist-"
        if home_prefix in text or cloud_prefix in text:
            ERRORS.append(f"machine-specific home path: {path.relative_to(ROOT)}")
        if "BEGIN " in text and "PRIVATE" + " KEY" in text:
            ERRORS.append(f"private key material: {path.relative_to(ROOT)}")
        if path.suffix in {".py", ".sh", ".md", ".json"} and re.search(r"[\u4e00-\u9fff]", text):
            ERRORS.append(f"non-English prose remains: {path.relative_to(ROOT)}")

max_file = 120 * 1024 * 1024
for path in ROOT.rglob("*"):
    if path.is_file() and path.stat().st_size > max_file:
        ERRORS.append(f"file exceeds GitHub limit: {path.relative_to(ROOT)}")

status_file = ROOT / "test_data/_test1019/reports/status.json"
if status_file.is_file():
    status = json.loads(status_file.read_text())
    if status.get("status") != "all_six_stages_and_checks_completed":
        ERRORS.append("acceptance status is not complete")
    for key in ("visual_quality_accepted", "physical_scale_validated", "world_trajectory_recovered"):
        if status.get(key) is not False:
            ERRORS.append(f"quality boundary must remain explicit bool false: {key}")

if ERRORS:
    print("Repository check failed:")
    for error in ERRORS:
        print(f" - {error}")
    sys.exit(1)

print("Repository checks passed")
