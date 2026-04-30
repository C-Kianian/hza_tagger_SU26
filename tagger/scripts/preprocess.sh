#!/usr/bin/env bash
# Generate the normalisation dictionary from the training H5 file.
# This must be run before the first training.

set -euo pipefail

CONFIG=tagger/configs/hza_train.yaml

echo "==> Computing normalisation statistics …"
salt preprocess --config "${CONFIG}"

echo "Norm dict written to tagger/configs/norm_dict.yaml"
