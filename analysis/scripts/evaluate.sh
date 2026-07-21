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
#   evaluate.sh --dir path/to/my/ckpts/dir (ie. logs/hza_tagger_YYMMDD)
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
DIR="${DIR:-}"
TRAIN_CFG="${TRAIN_CFG:-}" # arg for the training config
PLOTS=false

POSITIONAL=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --modeldir)
            DIR="$2"
            shift 2
            ;;
	 --modeldir=*)
	    DIR="${1#*=}"
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
    	 --plot)       PLOTS=true; shift ;;
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
if [[ -n "${DIR}" ]]; then
    [[ -d "${DIR}" ]] || die "Dir not found: ${DIR}"
    info "Ckpts dir:     ${DIR}"
fi

if [[ -n "${2:-}" ]]; then
    CKPT="${2}"
elif [[ -z "${CKPT:-}" ]]; then
    # SALT 0.11 saves to logs/<run>/ckpts/epoch=NNN-val_loss=X.ckpt
    # Pick the checkpoint with the lowest val_loss by parsing the filename.
    # Also handles the conventional best.ckpt name for other SALT versions.
    _best_by_loss() {
        local DIR="${DIR:-*}"
        ls -1 ${DIR}/ckpts/*.ckpt ${DIR}ckpts/*.ckpt logs/ckpts/*.ckpt logs/*/version_*/ckpts/*.ckpt logs/checkpoints/best.ckpt 2>/dev/null \
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
ATLAS=$(python common/parse_yaml.py --contains atlas --config "${TRAIN_CFG}")
REGRESS=$(python common/parse_yaml.py --contains regress --config "${TRAIN_CFG}")
CFG_NAME=$(python common/parse_yaml.py --get name --config "${TRAIN_CFG}")
# ── Derive output paths ───────────────────────────────────────────────────────
# Put scores next to the test file: test.h5 → test_scores.h5
_base="$(basename "${TEST_FILE}" .h5)"
_dir="$(dirname "${TEST_FILE}")"
SCORES_FILE="${SCORES_FILE:-${_dir}/${_base}_salt_scores.h5}"
if [[ "${REGRESS}" == true ]]; then
    SCORES_FILE="${SCORES_FILE:-${_dir}/${_base}_regression_scores.h5}"
fi
if [[ "${ATLAS}" == true ]]; then
    SCORES_FILE="${SCORES_FILE:-${_dir}/${_base}_atlas_classification_scores.h5}"
    if [[ "${REGRESS}" == true ]]; then
        SCORES_FILE="${SCORES_FILE:-${_dir}/${_base}_atlas_regression_scores.h5}"
    fi
fi
#PLOT_DIR="${PLOT_DIR:-analysis/plots}"
PLOT_DIR="${PLOT_DIR:-analysis/plots${_base:+_${_base}}_salt}"

info "Scores:     ${SCORES_FILE}"
#info "Plots dir:  ${PLOT_DIR}"
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
if [[ "${REGRESS}" == true ]]; then
    if [[ "${ATLAS}" == true ]]; then # atlas regression case
	PLOT_DIR="${PLOT_DIR}_atlas_regression"
        "${PYTHON}" analysis/scripts/plots_regression.py \
        --scores "${SCORES_FILE}" \
        --outdir "${PLOT_DIR}" \
    	--eval   "atlas_regression_a_mass"
    else # our regression model case
	PLOT_DIR="${PLOT_DIR}_atlas_classifier"
        "${PYTHON}" analysis/scripts/plots_regression.py \
        --scores "${SCORES_FILE}" \
        --outdir "${PLOT_DIR}" \
    	--eval  "regression_a_mass"
    fi
else # out jet classifier case
      PLOT_DIR="${PLOT_DIR}_${CFG_NAME}"
      if [[ "${ATLAS}" == true ]]; then
    	  "${PYTHON}" analysis/scripts/shap_atlas.py \
    	  --input   "${SCORES_FILE}" \
    	  --ckpt    "${CKPT}" \
    	  --output  "${PLOT_DIR}/atlas_shap_summary.png" \
    	  --config  "${TRAIN_CFG}" \
    	  --nsamples 10000
      fi 
      "${PYTHON}" analysis/scripts/plots.py \
          --scores "${SCORES_FILE}" \
          --outdir "${PLOT_DIR}" \
	  --atlas  "${ATLAS}"
fi


echo ""
echo "✓  Done.  Plots written to ${PLOT_DIR}"
