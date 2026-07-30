#!/usr/bin/env bash
# condor submission helper script called by train_condor.sh
set -euo pipefail

die()  { echo "ERROR [submit]: $*" >&2; exit 1; }
info() { echo "[submit] $*"; }

# ── Validate required environment variables ───────────────────────────────────
[[ -n "${NAME:-}" ]]       || die "NAME variable is not set."
[[ -n "${CONFIG:-}" ]]     || die "CONFIG variable is not set."
[[ -n "${TRAIN_FILE:-}" ]] || die "TRAIN_FILE variable is not set."
[[ -n "${VAL_FILE:-}" ]]   || die "VAL_FILE variable is not set."
[[ -n "${TEST_FILE:-}" ]]  || die "TEST_FILE variable is not set."

info "Generating HTCondor batch submission for: ${NAME}"

# final output dir
if [[ -n "${RENAME:-}" ]]; then
    FINAL_DIR="logs/${RENAME}"
    if [[ -d "${FINAL_DIR}" ]]; then
        FINAL_DIR="logs/${RENAME}_fallback_$(date +%H%M%S)"
        info "[Warning] logs/${RENAME} exists! Saving to ${FINAL_DIR}"
    fi
else
    FINAL_DIR="logs/${NAME}"
fi

mkdir -p "${FINAL_DIR}"

WORKER_SCRIPT="${FINAL_DIR}/run.sh"
SUB_FILE="${FINAL_DIR}/job.sub"

# script for condor worker to run
cat << EOF > "${WORKER_SCRIPT}"
#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${CONDA_PREFIX:-}" && -f "${CONDA_PREFIX}/bin/activate" ]]; then
    source "${CONDA_PREFIX}/bin/activate"
fi

echo "================================================================================"
echo " Worker Host: \$(hostname)"
echo " Start Time:  \$(date)"
echo "================================================================================"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv
fi
echo "================================================================================"

# salt command args
SALT_CMD=(
    salt fit
    --config "${CONFIG}"
    --data.train_file "${TRAIN_FILE}"
    --data.val_file "${VAL_FILE}"
    --data.test_file "${TEST_FILE}"
    --trainer.accelerator gpu
    --trainer.devices 1
    --force
)
EOF
# don't think accelerator or devices args make a difference?

# add args if set
if [[ -n "${EXTRA_LOGGER_ARGS:-}" ]]; then
    echo "SALT_CMD+=(${EXTRA_LOGGER_ARGS})" >> "${WORKER_SCRIPT}"
fi
if [[ -n "${EXTRA_LOSS_ARGS:-}" ]]; then
    echo "SALT_CMD+=(${EXTRA_LOSS_ARGS})" >> "${WORKER_SCRIPT}"
fi
if [[ -n "${EXTRA_DATA_ARGS:-}" ]]; then
    echo "SALT_CMD+=(${EXTRA_DATA_ARGS})" >> "${WORKER_SCRIPT}"
fi

# write salt args to run.sh file
cat << EOF >> "${WORKER_SCRIPT}"

# train
"\${SALT_CMD[@]}"

TRAIN_STATUS=\$?

# move outputs to dir
LATEST_SALT_DIR=\$(ls -td logs/hza_tagger_* 2>/dev/null | grep -v "${FINAL_DIR}" | head -n 1 || true)

if [[ -n "\${LATEST_SALT_DIR:-}" && -d "\${LATEST_SALT_DIR:-}" ]]; then
    echo "==> Unifying logs: Moving salt outputs into ${FINAL_DIR}"
    shopt -s dotglob
    mv "\${LATEST_SALT_DIR}"/* "${FINAL_DIR}/" 2>/dev/null || true
    shopt -u dotglob
    rm -rf "\${LATEST_SALT_DIR}"
fi

exit \$TRAIN_STATUS
EOF
chmod +x "${WORKER_SCRIPT}"

# make submit file + worker requirements, job.sub, 2 hrs = 7200
cat << EOF > "${SUB_FILE}"
universe              = vanilla
executable            = ${WORKER_SCRIPT}
initialdir            = ${PWD}
output                = ${FINAL_DIR}/condor.out
error                 = ${FINAL_DIR}/condor.err
log                   = ${FINAL_DIR}/condor.log
getenv                = True
request_gpus          = 1
request_cpus          = 4
request_memory        = 8GB
Requirements          = (GPUs_Capability >= 7.0)
+RequestRuntime       = 7200
queue 1
EOF

# submit
info "Submitting job to HTCondor..."
SUB_OUTPUT=$(condor_submit "${SUB_FILE}")
echo "${SUB_OUTPUT}"

CLUSTER_ID=$(echo "${SUB_OUTPUT}" | grep -oE 'cluster [0-9]+' | awk '{print $2}' || true)

if [[ -n "${CLUSTER_ID}" ]]; then
    echo "=========================================================================================================================="
    echo " Training submitted to HTCondor queue!"
    echo " Final Directory: ${FINAL_DIR}"
    echo " Monitor output:  tail -f ${FINAL_DIR}/condor.out"
    echo " Check queue:     condor_q ${CLUSTER_ID}"
    echo " Remove job:      condor_rm ${CLUSTER_ID}"
    echo " Useful naf info: https://docs.desy.de/naf/documentation/gpu-on-naf/"
    echo " Useful naf commands:"
    echo " condor_status -constraint 'GPUs >= 1'  "
    echo " condor_status -constraint 'gpus >= 1' -af:h Name GPUs_Capability GPUs_DeviceName  GPUs_DriverVersion GPUs_GlobalMemoryMb "
    echo "=========================================================================================================================="
fi
