#!/usr/bin/env bash
# Launch SALT training for the HZa binary tagger.
# Adjust --accelerator and --devices for your hardware.

set -euo pipefail

CONFIG=tagger/configs/hza_train.yaml
NAME=hza_tagger_$(date +%Y%m%d_%H%M%S)

echo "==> Starting training: ${NAME}"
salt fit \
    --config "${CONFIG}" \
    --trainer.logger.name "${NAME}" \
    "$@"
