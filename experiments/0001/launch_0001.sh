#!/usr/bin/env bash
# Stage C launcher for experiment 0001 (G1 Location baseline).
#
# Starts the formal run inside a detached tmux session, tees the full console
# log into logs/<run>/console.log and records the *python* exit code (not the
# exit code of tee) into logs/<run>/exit_code.txt.
#
# Usage:
#   experiments/0001/launch_0001.sh [run_id]
#
# Default run id: 0001_location_seed2027_<Asia/Shanghai timestamp>.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="${REPO_ROOT}/../env.sh"
PYTHON="${UV_PROJECT_ENVIRONMENT:-/root/gpufree-data/smp-g1/venv}/bin/python"

NUM_ENVS="${NUM_ENVS:-1024}"
MAX_ITERATIONS="${MAX_ITERATIONS:-10000}"
SEED="${SEED:-2027}"
DEVICE="${DEVICE:-cuda:0}"

RUN_ID="${1:-0001_location_seed${SEED}_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)}"
SESSION="smp_g1_0001_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
LOG_DIR="${REPO_ROOT}/logs/${RUN_ID}"
CKPT_DIR="${REPO_ROOT}/ckpt/${RUN_ID}"

if [[ -e "${LOG_DIR}" || -e "${CKPT_DIR}" ]]; then
  echo "refusing to overwrite existing run: ${RUN_ID}" >&2
  exit 2
fi
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 2
fi
mkdir -p "${LOG_DIR}"

INNER="cd ${REPO_ROOT} && set -o pipefail && source ${ENV_FILE} && \
${PYTHON} experiments/0001/run_0001.py --num-envs ${NUM_ENVS} \
  --max-iterations ${MAX_ITERATIONS} --seed ${SEED} --device ${DEVICE} \
  --run ${RUN_ID} 2>&1 | tee ${LOG_DIR}/console.log; \
rc=\${PIPESTATUS[0]}; echo \"python_exit_code=\${rc}\" | tee ${LOG_DIR}/exit_code.txt; \
echo \"[launcher] finished run ${RUN_ID} exit=\${rc}\""

tmux new-session -d -s "${SESSION}" "${INNER}"
echo "session=${SESSION}"
echo "run_id=${RUN_ID}"
echo "log_dir=${LOG_DIR}"
echo "ckpt_dir=${CKPT_DIR}"
echo "console=${LOG_DIR}/console.log"
