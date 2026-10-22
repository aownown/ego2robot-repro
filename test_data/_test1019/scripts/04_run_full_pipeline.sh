#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/test_env.sh"
exec "$TEST_ROOT/tools/bin/python" -u "$TEST_ROOT/scripts/run_pipeline.py" "$@"
