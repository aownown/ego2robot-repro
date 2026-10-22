# Contributing

Small, reproducible contributions are welcome:

- bug fixes with a minimal reproduction and the exact command used;
- environment or GPU compatibility fixes tied to a tested version;
- adapters for another robot embodiment, with units and frame conventions stated;
- validation improvements that do not hide failures behind defaults;
- documentation corrections with links to source evidence.

Before opening a pull request, run:

```bash
python3 tools/repo_check.py
python3 tools/check_links.py
python3 -m compileall -q workflow test_data/_test1019/scripts
bash tests/smoke.sh
```

Do not commit datasets, generated frames, model weights, environments, credentials, or code copied from a third party whose license forbids redistribution. New upstream dependencies must be pinned in `workflow/sources.lock.json` and documented in `THIRD_PARTY.md`.
