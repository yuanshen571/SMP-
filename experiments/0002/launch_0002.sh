#!/usr/bin/env bash
# Formal experiment 0002; start once, then verify completion separately.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_ROOT}/experiments/0002/activate_0002.sh"
PYTHON="${UV_PROJECT_ENVIRONMENT}/bin/python"
NUM_ENVS="${NUM_ENVS:-1024}"
MAX_ITERATIONS="${MAX_ITERATIONS:-10000}"
SEED="${SEED:-2027}"
DEVICE="${DEVICE:-cuda:0}"

if [[ "${NUM_ENVS}" != 1024 || "${MAX_ITERATIONS}" != 10000 || "${SEED}" != 2027 ]]; then
  printf 'formal 0002 requires NUM_ENVS=1024 MAX_ITERATIONS=10000 SEED=2027; use run_0002.py for smoke checks\n' >&2
  exit 2
fi

if [[ "${1:-}" == --worker ]]; then
  RUN_ID="${2:?run id required}"
  LOG_DIR="${SMP_0002_ROOT}/logs/${RUN_ID}"
  cd "${REPO_ROOT}"
  set +e
  "${PYTHON}" experiments/0002/run_0002.py \
    --num-envs "${NUM_ENVS}" --max-iterations "${MAX_ITERATIONS}" \
    --seed "${SEED}" --device "${DEVICE}" --run "${RUN_ID}" \
    --log-root "${SMP_0002_ROOT}/logs" --ckpt-root "${SMP_0002_ROOT}/ckpt" \
    2>&1 | tee "${LOG_DIR}/console.log"
  statuses=("${PIPESTATUS[@]}")
  set -e
  rc="${statuses[0]}"
  printf 'python_exit_code=%s\ntee_exit_code=%s\n' "${rc}" "${statuses[1]}" > "${LOG_DIR}/exit_code.txt"
  printf '[launcher] finished run %s python_exit=%s tee_exit=%s\n' "${RUN_ID}" "${rc}" "${statuses[1]}"
  if [[ "${rc}" -ne 0 ]]; then exit "${rc}"; fi
  exit "${statuses[1]}"
fi

RUN_ID="${1:-0002_location_seed${SEED}_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)}"
if [[ ! "${RUN_ID}" =~ ^0002_[A-Za-z0-9_-]+$ ]]; then
  printf 'run id must start with 0002_ and contain only letters, digits, _ or -\n' >&2
  exit 2
fi
SESSION="smp_g1_${RUN_ID}"
LOG_DIR="${SMP_0002_ROOT}/logs/${RUN_ID}"
CKPT_DIR="${SMP_0002_ROOT}/ckpt/${RUN_ID}"
[[ -x "${PYTHON}" ]] || { printf '0002 Python is not installed: %s\n' "${PYTHON}" >&2; exit 2; }
if [[ -e "${LOG_DIR}" || -e "${CKPT_DIR}" ]]; then
  printf 'refusing to overwrite existing run: %s\n' "${RUN_ID}" >&2
  exit 2
fi
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  printf 'tmux session already exists: %s\n' "${SESSION}" >&2
  exit 2
fi
mkdir -p "${LOG_DIR}"
printf -v INNER '%q ' env "NUM_ENVS=${NUM_ENVS}" "MAX_ITERATIONS=${MAX_ITERATIONS}" "SEED=${SEED}" "DEVICE=${DEVICE}" bash "${REPO_ROOT}/experiments/0002/launch_0002.sh" --worker "${RUN_ID}"
tmux new-session -d -s "${SESSION}" "${INNER}"
printf 'session=%s\nrun_id=%s\nlog_dir=%s\nckpt_dir=%s\n' "${SESSION}" "${RUN_ID}" "${LOG_DIR}" "${CKPT_DIR}"
