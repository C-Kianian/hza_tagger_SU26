#!/usr/bin/env bash
# Launch SALT training for the HZa binary tagger.
# Adjust --accelerator and --devices for your hardware.
#
# Comet.ml logging: put your API key in .env at the project root:
#   echo "COMET_API_KEY=your_key_here" > .env
# train.sh sources .env automatically and passes the key to CometLogger.
# Without a key the run falls back to offline mode (logs saved under logs/).
#
# Auto-discovers train/val/test H5 files from the data/ directory.
# Override via environment variables or positional args.
#
# Usage:
#   bash atlas_2025_model/scripts/train.sh                          			# auto-discover everything
#   bash atlas_2025_model/scripts/train.sh data/train.h5 data/val.h5 data/test.h5	# specify all file paths
#   bash atlas_2025_model/scripts/train.sh --rw 			 		# auto reweight a classification task
#   bash atlas_2025_model/scripts/train.sh --mask=atlas_mask       			# Apply a mask to the events, ie. atlas_mask
#   bash atlas_2025_model/scripts/train.sh --rename=some_name_here 			# the name to rename the standard hza_tagger_YMD_HMS out directory
#   bash atlas_2025_model/scripts/train.sh --regression            			# to train the regression mass estimator model
#
# Environment overrides:
#   TRAIN_FILE, VAL_FILE, TEST_FILE   explicit H5 paths
#   CONFIG                            YAML config (default: tagger/configs/hza_train.yaml)

set -euo pipefail

die()  { echo "ERROR: $*" >&2; exit 1; }
info() { echo "[train] $*"; }

# Resolve Python from the active conda env
PYTHON="${CONDA_PREFIX:+${CONDA_PREFIX}/bin/python}"
PYTHON="${PYTHON:-$(command -v python3 2>/dev/null || command -v python)}"

# == get args ================================================================
REGRESS=false
RENAME=""
RW=false # to reweight

POSITIONAL=()

while [[ $# -gt 0 ]]; do
    case "$1" in
    	--rename=*)   RENAME="${1#*=}" ;;
	--rw)         RW=true ;;
	--regression) REGRESS=true ;;
	*)
            POSITIONAL+=("$1")
            ;;
    esac
    shift
done

set -- "${POSITIONAL[@]}"

# ── Load secrets from .env if present ────────────────────────────────────────
if [[ -f ".env" ]]; then
    # export only lines that look like KEY=value (skip comments / blank lines)
    set -o allexport
    # shellcheck disable=SC1091
    source <(grep -E '^[A-Z_]+=.+' .env)
    set +o allexport
    info "Loaded .env"
fi

# ── Comet API key check ───────────────────────────────────────────────────────
if [[ -z "${COMET_API_KEY:-}" ]]; then
    info "COMET_API_KEY not set — running in offline mode (logs under logs/)."
    EXTRA_LOGGER_ARGS="--trainer.logger.init_args.offline true"
else
    info "COMET_API_KEY found — logging to Comet.ml."
    EXTRA_LOGGER_ARGS="--trainer.logger.init_args.offline false"
fi

# ── Resolve CONFIG ────────────────────────────────────────────────────────────
CONFIG="${CONFIG:-atlas_2025_model/configs/jet_classification_train.yaml}"
if [[ "$REGRESS" == true ]]; then
    echo "==> Using train yaml for regression …"
    CONFIG=atlas_2025_model/configs/mass_regression_train.yaml
fi
[[ -f "${CONFIG}" ]] || die "Config not found: ${CONFIG}"

export CONFIG="${CONFIG}" #export env variable

# ── Auto-discover H5 files ────────────────────────────────────────────────────
_pick_h5() {
    local val="${1}"; shift
    if [[ -n "${val}" ]]; then
        [[ -f "${val}" ]] || die "File not found: ${val}"
        echo "${val}"; return
    fi
    for c in "$@"; do [[ -f "${c}" ]] && echo "${c}" && return; done
    local f; f="$(ls data/*.h5 2>/dev/null | head -1 || true)"
    [[ -n "${f}" ]] || die "No H5 file found in data/. Run the converter first."
    echo "${f}"
}

if [[ -n "${1:-}" && "${1}" != --* ]]; then
    TRAIN_FILE="${1}"
    VAL_FILE="${2:-${1}}"
    TEST_FILE="${3:-${1}}"
else
    TRAIN_FILE="$(_pick_h5 "${TRAIN_FILE:-}" data/train.h5 data/test_out.h5)"
    VAL_FILE="$(_pick_h5   "${VAL_FILE:-}"   data/val.h5   data/test_out.h5)"
    TEST_FILE="$(_pick_h5  "${TEST_FILE:-}"  data/test.h5  data/test_out.h5)"
fi

# ── Clean file args before passing to salt ────────────────────────────────────
if [[ -n "${1:-}" && "${1}" != --* ]]; then
    # Positional args detected
    TRAIN_FILE="${1}"
    VAL_FILE="${2:-${1}}"
    TEST_FILE="${3:-${1}}"
    shift 3
else
    # pick fallback files if they aren't already populated
    TRAIN_FILE="$(_pick_h5 "${TRAIN_FILE}" data/train.h5 data/test_out.h5)"
    VAL_FILE="$(_pick_h5   "${VAL_FILE}"   data/val.h5   data/test_out.h5)"
    TEST_FILE="$(_pick_h5  "${TEST_FILE}"  data/test.h5  data/test_out.h5)"
fi

#export env variable
export TRAIN_FILE="${TRAIN_FILE}"
export VAL_FILE="${VAL_FILE}"
export TEST_FILE="${TEST_FILE}"

# == norm dict =================================================================
NORM_DICT=atlas_2025_model/configs/norm_dict_classification_atlas.yaml
if [[ "$REGRESS" == true ]]; then
    echo "==> Using norm dict yaml for regression …"
    NORM_DICT=atlas_2025_model/configs/norm_dict_regression_atlas.yaml
fi

NAME=atlas_2025_classifier_$(date +%Y%m%d_%H%M%S)
if [[ "$REGRESS" == true ]]; then
    NAME=atlas_2025_regressor_$(date +%Y%m%d_%H%M%S)
fi
EXTRA_DATA_ARGS="--data.norm_dict ${NORM_DICT}"

export REGRESS="${REGRESS}"
export NORM_DICT="${NORM_DICT}"

# pre training output
info "Config:     ${CONFIG}"
info "Train file: ${TRAIN_FILE}"
info "Val file:   ${VAL_FILE}"
info "Test file:  ${TEST_FILE}"
info "Run name:   ${NAME}"
info "Norm dict:  ${NORM_DICT}"

# == Traning regression  ==========================================================
if [[ "$REGRESS" == true ]]; then
   echo ""
   # begin training for regression
   echo "==> Starting training: ${NAME}"
   # shellcheck disable=SC2086
   salt fit \
       --config "${CONFIG}" \
       --data.train_file "${TRAIN_FILE}" \
       --data.val_file   "${VAL_FILE}" \
       --data.test_file  "${TEST_FILE}" \
       --trainer.logger.init_args.experiment_name "${NAME}" \
       ${EXTRA_LOGGER_ARGS} \
       ${EXTRA_DATA_ARGS} \
       --force \
       "$@"	

# == reweighting =================================================================
else
    if [[ "$RW" == true ]]; then    
        echo "==> Computing reweighting from ${TRAIN_FILE} …"
        read -r W_POS < <("${PYTHON}" tagger/scripts/calc_reweight_vals.py --file "${TRAIN_FILE}" --bce)
    else
        W_POS=1.0
    fi
    
    #export env variable
    export W_POS=${W_POS}
    
    info "Positive (signal) rw: ${W_POS}"
    echo ""
    EXTRA_LOSS_ARGS="--model.model.init_args.tasks.init_args.modules.init_args.loss.init_args.pos_weight=[${W_POS}]"

# == Traning classification ======================================================
   # begin training for classification
   echo "==> Starting training: ${NAME}"
   # shellcheck disable=SC2086
   salt fit \
       --config "${CONFIG}" \
       --data.train_file "${TRAIN_FILE}" \
       --data.val_file   "${VAL_FILE}" \
       --data.test_file  "${TEST_FILE}" \
       --trainer.logger.init_args.experiment_name "${NAME}" \
       ${EXTRA_LOGGER_ARGS} \
       ${EXTRA_LOSS_ARGS} \
       ${EXTRA_DATA_ARGS} \
       --force \
       "$@"
fi

# == Option to rename the output dir ===========================================
# 1. Capture the exit code of the training process
TRAIN_STATUS=$?

# 2. Find the most recently modified directory matching the lightning pattern
LATEST_DIR=$(ls -td atlas_logs/*class* 2>/dev/null | head -n 1)
if [[ "$REGRESS" == true ]]; then
    LATEST_DIR=$(ls -td atlas_logs/*regress* 2>/dev/null | head -n 1)
fi


# 3. Rename it to your specified $NAME variable
if [[ -d "$LATEST_DIR" && "$LATEST_DIR" != "atlas_logs/${RENAME}" ]]; then
    echo "==> Clean up: Moving output directory to atlas_logs/${RENAME}"

    # Safety check: if target directory exists, append a small safety flag
    if [[ -d "atlas_logs/${RENAME}" ]]; then
        SAFE_NAME="atlas_logs/${RENAME}_fallback_$(date +%H%M%S)"
        echo "    [Warning] atlas_logs/${RENAME} already exists! Saving to ${SAFE_NAME} instead."
        mv "$LATEST_DIR" "$SAFE_NAME"
    else
        mv "$LATEST_DIR" "atlas_logs/${RENAME}"
    fi
else
    echo "==> Clean up: No matching timestamped directory found to rename."
fi

# 4. Exit with the original training status so batch scripts know if it failed
exit $TRAIN_STATUS
