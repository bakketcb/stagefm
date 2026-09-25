#!/usr/bin/env bash
# Evaluate one fitted arm on the external cohort.
#
#   scripts/launch_eval.sh                                  # the primary result
#   scripts/launch_eval.sh main unconstrained_posthoc       # the control arm
#   scripts/launch_eval.sh supplementary_ascertainment_analysis
#
# The evaluation reads the thresholds written by the matching training run and
# refits nothing: the external sites contribute to no estimate.
set -euo pipefail

EXPERIMENT="${1:-main}"
ARM="${2:-stagefm}"
PYTHON="${PYTHON:-python3}"

"${PYTHON}" -m stagefm.cli.eval \
  --experiment "${EXPERIMENT}" \
  --arm "${ARM}" \
  --run-dir "${RUN_DIR:-runs/${EXPERIMENT}/${ARM}}"
