#!/usr/bin/env bash
# Initialize the SALT submodule and install it in the current environment.
# Run once from the repo root after cloning.

set -euo pipefail

echo "==> Initialising SALT submodule …"
git submodule update --init --recursive tagger/salt

echo "==> Installing SALT (editable) …"
pip install -e tagger/salt

echo "==> Verifying salt CLI …"
salt --help | head -3

echo "Done. You can now run: salt fit --config tagger/configs/hza_train.yaml"
