#!/usr/bin/env bash
# Step one already has its own validated preparation commands; preserve it.
set -euo pipefail
WORKFLOW_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
stages=(step_two_Retarget_and_Smooth step_three_SAM3 step_four_ProPainter step_five_MuJoCo_and_mink step_six_DepthAnything3_and_Composite)
action=${1:-status}
case "$action" in
    status) exec /usr/bin/python3 "$WORKFLOW_ROOT/common/status.py" ;;
    sources|env|weights|check|model-check|all) ;;
    *) echo 'Usage: bash prepare_all.sh {status|sources|env|weights|check|model-check|all}' >&2; exit 2 ;;
esac
echo 'Processing stages two through six; stage one keeps its own prepare.sh interface.'
for stage in "${stages[@]}"; do
    echo "$stage: $action"
    bash "$WORKFLOW_ROOT/$stage/prepare.sh" "$action"
done
