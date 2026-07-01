#!/usr/bin/env bash
# Generate the normalisation dictionary from the training H5 file.
# This must be run before the first training.
#
# SALT 0.11 does not have a `salt preprocess` subcommand; we use a
# standalone script instead.

set -euo pipefail

# arg for the mass input as: 2_0, 1_5 etc
POSITIONAL=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mass=*)
    	    MASS="${1#*=}"
    	    shift
            ;;
	--mass)
            MASS="$2"
            shift 2
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done

set -- "${POSITIONAL[@]}"

# Resolve Python from the active conda env
PYTHON="${CONDA_PREFIX:+${CONDA_PREFIX}/bin/python}"
PYTHON="${PYTHON:-$(command -v python3 2>/dev/null || command -v python)}"

TRAIN_FILE=${1:-data/train.h5}
# fall back to test_out.h5 if train.h5 hasn't been created yet
[[ -f "${TRAIN_FILE}" ]] || TRAIN_FILE="$(ls data/*.h5 2>/dev/null | head -1 || true)"
[[ -f "${TRAIN_FILE:-}" ]] || { echo "ERROR: no H5 file found in data/. Run the converter first."; exit 1; }
VARIABLES=tagger/configs/hza_variables.yaml

# mass specific norm dict, avoids recalculating
if [[ -n "${MASS:-}" ]]; then
    NORM_DICT="tagger/configs/norm_dict_mA${MASS}.yaml"
    echo "Mass specified, creating mass specific dict: ${NORM_DICT}"
else
    echo "No mass specified, defaulting to generic name: tagger/configs/norm_dict.yaml"
    NORM_DICT=tagger/configs/norm_dict.yaml
fi


echo "==> Computing normalisation statistics from ${TRAIN_FILE} …"
"${PYTHON}" tagger/scripts/create_norm_dict.py \
    --input   "${TRAIN_FILE}" \
    --config  "${VARIABLES}" \
    --output  "${NORM_DICT}"

echo "Norm dict written to ${NORM_DICT}"
