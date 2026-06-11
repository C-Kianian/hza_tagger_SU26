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
#   bash tagger/scripts/train.sh                          # auto-discover everything
#   bash tagger/scripts/train.sh data/train.h5 data/val.h5 data/test.h5
#
# Environment overrides:
#   TRAIN_FILE, VAL_FILE, TEST_FILE   explicit H5 paths
#   CONFIG                            YAML config (default: tagger/configs/hza_train.yaml)

set -euo pipefail

die()  { echo "ERROR: $*" >&2; exit 1; }
info() { echo "[train] $*"; }

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
CONFIG="${CONFIG:-tagger/configs/hza_train.yaml}"
[[ -f "${CONFIG}" ]] || die "Config not found: ${CONFIG}"

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
    
    # SHIFT REMOVES THESE FROM "$@" SO SALT DOESN'T SEE THEM AGAIN
    shift 3 
else
    TRAIN_FILE="$(_pick_h5 "${TRAIN_FILE:-}" data/train.h5 data/test_out.h5)"
    VAL_FILE="$(_pick_h5   "${VAL_FILE:-}"   data/val.h5   data/test_out.h5)"
    TEST_FILE="$(_pick_h5  "${TEST_FILE:-}"  data/test.h5  data/test_out.h5)"
fi

NAME=hza_tagger_$(date +%Y%m%d_%H%M%S)

info "Config:     ${CONFIG}"
info "Train file: ${TRAIN_FILE}"
info "Val file:   ${VAL_FILE}"
info "Test file:  ${TEST_FILE}"
info "Run name:   ${NAME}"
echo ""

echo "==> Starting training: ${NAME}"
# shellcheck disable=SC2086
salt fit \
    --config "${CONFIG}" \
    --data.train_file "${TRAIN_FILE}" \
    --data.val_file   "${VAL_FILE}" \
    --data.test_file  "${TEST_FILE}" \
    --trainer.logger.init_args.experiment_name "${NAME}" \
    ${EXTRA_LOGGER_ARGS} \
    --force \
    "$@"

