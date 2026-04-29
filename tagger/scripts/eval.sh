#!/usr/bin/env bash
# Evaluate a trained checkpoint on the test set.
# Usage: ./tagger/scripts/eval.sh logs/hza_tagger_*/checkpoints/best.ckpt

set -euo pipefail

CKPT="${1:?Usage: eval.sh <checkpoint.ckpt>}"
CONFIG=tagger/configs/hza_train.yaml

echo "==> Evaluating checkpoint: ${CKPT}"
salt test \
    --config "${CONFIG}" \
    --ckpt_path "${CKPT}" \
    "$@"
