#!/usr/bin/env bash
# Launch one training run.
#
#   scripts/launch_train.sh                      # the paper's primary result
#   scripts/launch_train.sh ablation_without_risk_control
#   scripts/launch_train.sh main finetuned_independent_heads
#
# The reported profile is one node with four accelerators (Methods Sec. 4.7), which
# is what the WORLD_SIZE default below matches. Accelerator memory is not managed by
# this script; the reported run held 80 GB per node.
set -euo pipefail

EXPERIMENT="${1:-main}"
ARM="${2:-stagefm}"
WORLD_SIZE="${WORLD_SIZE:-4}"
PYTHON="${PYTHON:-python3}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

if command -v torchrun >/dev/null 2>&1 && [ "${WORLD_SIZE}" -gt 1 ]; then
  torchrun --standalone --nproc_per_node="${WORLD_SIZE}" -m stagefm.cli.train \
    --experiment "${EXPERIMENT}" --arm "${ARM}" \
    experiment.seed="${SEED:-0}"
else
  echo "torchrun unavailable or WORLD_SIZE=1; running single-process"
  "${PYTHON}" -m stagefm.cli.train --experiment "${EXPERIMENT}" --arm "${ARM}" experiment.seed="${SEED:-0}"
fi
