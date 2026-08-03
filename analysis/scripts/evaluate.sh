#!/usr/bin/env bash
# evaluate.sh — score a test H5 file with the best checkpoint and produce plots.
#
# Auto-discovers paths from the standard project layout.  Override any of the
# variables below via environment variables or positional arguments.
#
# Usage
#   evaluate.sh [TEST_FILE] [CKPT] [--run RUN_NAME]
#
# Examples
#   evaluate.sh --modeldir path/to/my/ckpts/dir (ie. logs/hza_tagger_YYMMDD)
#   evaluate.sh --plotsdir path/to/my/plots     (where you wants plots stored, otherwise named: plots_<config_name>)
#   evaluate.sh data/test.h5 --run my_training_run
#   evaluate.sh data/test.h5 logs/my_run/ckpts/best.ckpt
#
# Environment overrides (all optional):
#   TEST_FILE    path to input test H5
#   CKPT         path to checkpoint (.ckpt)
#   TRAIN_CFG    path to training config YAML
#   SCORES_FILE  path for output scores H5
#   PLOT_DIR     output directory for plots

set -euo pipefail

# ── Locate the project root (directory containing this script's …/hza_tagger) ─
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

# add arg for which  dir to search for model checkpoint
MODEL_DIR="${MODEL_DIR:-}"
PLOTS_DIR="${PLOTS_DIR:-}"
TRAIN_CFG="${TRAIN_CFG:-}" # arg for the training config
PLOTS=false

POSITIONAL=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --modeldir)
            MODEL_DIR="$2"
            shift 2
            ;;
	      --modeldir=*)
	          MODEL_DIR="${1#*=}"
	          shift 1
	          ;;
        --config)
            TRAIN_CFG="$2"
	          shift 2
	          ;;
	      --config=*)
	          TRAIN_CFG="${1#*=}"
	          shift 1
	          ;;
        --plotsdir)
            PLOTS_DIR="$2"
            shift 2
            ;;
	      --plotsdir=*)
	          PLOTS_DIR="${1#*=}"
	          shift 1
	          ;;
        --plot)
            PLOTS=true;
            shift 1
            ;;
	      *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done

set -- "${POSITIONAL[@]}"

# ── Helpers ───────────────────────────────────────────────────────────────────
die()  { echo "ERROR: $*" >&2; exit 1; }
info() { echo "[evaluate] $*"; }

# ── Resolve Python (prefer the active conda env, then PATH) ──────────────────
_find_python() {
    # If already inside a conda env, use that Python
    if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
        echo "${CONDA_PREFIX}/bin/python"; return
    fi
    # Fall back to whatever python3/python is on PATH
    command -v python3 2>/dev/null || command -v python 2>/dev/null \
        || die "No Python found. Activate the hza_tagger conda env first."
}
PYTHON="$(_find_python)"
info "Python:     ${PYTHON}"

# ── Resolve TEST_FILE ─────────────────────────────────────────────────────────
if [[ -n "${1:-}" ]]; then
    TEST_FILE="${1}"
elif [[ -z "${TEST_FILE:-}" ]]; then
    # Try the canonical split first, then any h5 in data/
    for candidate in data/test.h5 data/test_out.h5; do
        [[ -f "${candidate}" ]] && TEST_FILE="${candidate}" && break
    done
    if [[ -z "${TEST_FILE:-}" ]]; then
        TEST_FILE="$(ls data/*.h5 2>/dev/null | head -1 || true)"
    fi
fi
[[ -f "${TEST_FILE:-}" ]] || die "No test H5 file found. Pass it as argument or set TEST_FILE."
info "Test file:  ${TEST_FILE}"

# ── Resolve CKPT ──────────────────────────────────────────────────────────────
if [[ -n "${MODEL_DIR}" ]]; then
    [[ -d "${MODEL_DIR}" ]] || die "Dir not found: ${MODEL_DIR}"
    info "Ckpts dir:     ${MODEL_DIR}"
fi

if [[ -n "${2:-}" ]]; then
    CKPT="${2}"
elif [[ -z "${CKPT:-}" ]]; then
    # SALT 0.11 saves to logs/<run>/ckpts/epoch=NNN-val_loss=X.ckpt
    # Pick the checkpoint with the lowest val_loss by parsing the filename.
    # Also handles the conventional best.ckpt name for other SALT versions.
    _best_by_loss() {
        local MODEL_DIR="${MODEL_DIR:-*}"
        ls -1 ${MODEL_DIR}/ckpts/*.ckpt ${MODEL_DIR}ckpts/*.ckpt logs/ckpts/*.ckpt logs/*/version_*/ckpts/*.ckpt logs/checkpoints/best.ckpt 2>/dev/null \
            | awk -F'val_loss=' '
                NF==2 { val=$2; sub(/\.ckpt$/,"",val); print val, $0 }
                NF==1 { print "best", $0 }
              ' \
            | sort -n \
            | head -1 \
            | awk '{print $2}'
    }
    CKPT="$(_best_by_loss || true)"
fi
[[ -f "${CKPT:-}" ]] || die "No checkpoint found. Run training first, or set CKPT."
info "Checkpoint: ${CKPT}"

# ── Resolve TRAIN_CFG ─────────────────────────────────────────────────────────
[[ -f "${TRAIN_CFG}" ]] || die "Training config not found: ${TRAIN_CFG}"
info "Config:     ${TRAIN_CFG}"

# get info from config
ATLAS=$(python common/parse_yaml.py --in-name atlas --config "${TRAIN_CFG}")
REGRESS=$(python common/parse_yaml.py --in-name regress --config "${TRAIN_CFG}")
CLASS=$(python common/parse_yaml.py --in-name tag --config "${TRAIN_CFG}" || python common/parse_yaml.py --in-name class --config "${TRAIN_CFG}")
CFG_NAME=$(python common/parse_yaml.py --get_value name --config "${TRAIN_CFG}")
# ── Derive output paths ───────────────────────────────────────────────────────
# Put scores next to the test file: test.h5 → test_scores.h5
_base="$(basename "${TEST_FILE}" .h5)"
_dir="$(dirname "${TEST_FILE}")"
SCORES_FILE="${SCORES_FILE:-${_dir}/${_base}_scores.h5}"
PLOT_DIR="${PLOT_DIR:-plots_${CFG_NAME}}"

info "Scores:     ${SCORES_FILE}"
info "Plots dir:  ${PLOT_DIR}"
echo ""

# ── Step 1: score ─────────────────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Step 1 / 2  —  Scoring test file"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
"${PYTHON}" analysis/scripts/eval_to_h5.py \
    --input  	 "${TEST_FILE}" \
    --ckpt   	 "${CKPT}" \
    --config 	 "${TRAIN_CFG}" \
    --output 	 "${SCORES_FILE}" \
    --atlas  	 "${ATLAS}" \
    --regression "${REGRESS}"

# ── Step 2: plots ─────────────────────────────────────────────────────────────
echo ""
if [[ "${PLOTS}" != true ]]; then
    echo "Plotting step not specified, ending without plotting"
    exit 0
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Step 2 / 2  —  Producing plots"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
CFG_NAME=$(python common/parse_yaml.py --get name --config "${TRAIN_CFG}")
if [[ "${REGRESS}" == true ]]; then
  "${PYTHON}" analysis/scripts/model_plotting/plots_regression.py \
    --scores "${SCORES_FILE}" \
    --outdir "${PLOT_DIR}"
fi
if [[ "${CLASS}" == true ]]; then # out jet classifier case
      "${PYTHON}" analysis/scripts/model_plotting/plots_classification.py \
          --scores "${SCORES_FILE}" \
          --outdir "${PLOT_DIR}" \
	        --config "${TRAIN_CFG}"
fi


echo ""
echo "✓  Done.  Plots written to ${PLOT_DIR}"
