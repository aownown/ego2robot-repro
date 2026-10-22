#!/usr/bin/env python3
"""Validate stage/report metadata without requiring model runtimes."""
from __future__ import annotations
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
status=json.loads((ROOT/'test_data/_test1019/reports/status.json').read_text())
assert status['input_rgb_frames']==450
assert status['environments_ready']==7
assert status['intermediate_videos_verified']==23
assert status['final_videos_verified']==4
assert status['source_file_hashes_verified']==3600
sources=json.loads((ROOT/'workflow/sources.lock.json').read_text())
assert len(sources['repositories'])==8
assert all(len(repo['revision'])==40 for repo in sources['repositories'])
print('Metadata checks passed')
