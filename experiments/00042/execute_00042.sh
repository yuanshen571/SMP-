#!/usr/bin/env bash
# Run this one-experiment workflow inside tmux in the shared server checkout.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO}/experiments/00042/activate_00042.sh"
SMP_G1_ROOT="${SMP_00042_ROOT}"
PYTHON="${SMP_G1_ROOT}/venv/bin/python"
: "${SMP_INITIAL_CHECKPOINT:?set the common untrained checkpoint used by 00031/0004}"
: "${SMP_00031_RUN_INFO:?set the actual completed 00031 run_info.json}"
export SMP_INITIAL_CHECKPOINT SMP_00031_RUN_INFO
[[ -f "${SMP_INITIAL_CHECKPOINT}" && -f "${SMP_00031_RUN_INFO}" ]] || exit 2
cd "${REPO}"
[[ "$(git branch --show-current)" == 00042 ]] || { printf 'expected branch 00042\n' >&2; exit 2; }
[[ -z "$(git status --porcelain)" ]] || { printf 'working tree is dirty; stop without reset\n' >&2; exit 2; }
exec 9>"${SMP_G1_ROOT}/.0003_00031_queue.lock"
flock -n 9 || { printf 'another baseline/TS2Vec/cluster workflow is still running\n' >&2; exit 2; }
# Pin this checkout for every subsequent stage; never switch a running repo.
COMMIT="$(git rev-parse HEAD)"
mkdir -p "${SMP_G1_ROOT}/logs"
STAMP="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
EXECUTION="$(mktemp -d "${SMP_G1_ROOT}/logs/execute_00042_${STAMP}_XXXXXX")"
TAG="$(basename "${EXECUTION}")"
TAG="${TAG#execute_00042_}"
STAGE=preflight
on_exit() {
  local rc=$?
  printf 'execution_exit_code=%s\nlast_stage=%s\n' "${rc}" "${STAGE}" > "${EXECUTION}/exit_code.txt"
  printf 'EXECUTION_EXIT_CODE=%s\nEXECUTION_DIRECTORY=%s\n' "${rc}" "${EXECUTION}"
}
trap on_exit EXIT
"${PYTHON}" - "${EXECUTION}/execution_info.json" "${COMMIT}" "${SMP_INITIAL_CHECKPOINT}" "${SMP_00031_RUN_INFO}" <<'PY'
import json,sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    'experiment':'00042','git_commit':sys.argv[2],
    'control_experiment':'0004','control_commit':'07d60d9a1058f73860f00a2f6aae9bd37e277fa2',
    'weight_cap':1.5,'output_dims':64,
    'initial_checkpoint':sys.argv[3],'baseline_info':sys.argv[4],
},indent=2)+'\n',encoding='utf-8')
PY
check_source() {
  [[ "$(git branch --show-current)" == 00042 && "$(git rev-parse HEAD)" == "${COMMIT}" && -z "$(git status --porcelain)" ]] || {
    printf 'source changed during execution; stop without restoring or overwriting it\n' >&2
    return 2
  }
}
run_logged() {
  local folder="$1"
  shift
  check_source
  mkdir -p "${folder}"
  set +e
  "$@" 2>&1 | tee "${folder}/console.log"
  local statuses=("${PIPESTATUS[@]}")
  set -e
  printf 'python_exit_code=%s\ntee_exit_code=%s\n' "${statuses[0]}" "${statuses[1]}" > "${folder}/exit_code.txt"
  if [[ "${statuses[0]}" != 0 ]]; then return "${statuses[0]}"; fi
  if [[ "${statuses[1]}" != 0 ]]; then return "${statuses[1]}"; fi
}
run_logged "${EXECUTION}/preflight_initial" "${PYTHON}" experiments/00042/preflight.py \
  --baseline-info "${SMP_00031_RUN_INFO}" --initial-checkpoint "${SMP_INITIAL_CHECKPOINT}" \
  --out "${EXECUTION}/preflight.json"
STAGE=contract_tests
printf '%s\n' "${STAGE}" > "${EXECUTION}/progress.txt"
for test in test_reward_formula test_passive_metrics test_physics test_runtime test_initialization test_clusters; do
  run_logged "${EXECUTION}/${test}" "${PYTHON}" "experiments/00042/${test}.py"
done
STAGE=environment_acceptance
printf '%s\n' "${STAGE}" > "${EXECUTION}/progress.txt"
run_logged "${EXECUTION}/accept_console" "${PYTHON}" experiments/00042/accept_00042.py \
  --num-envs 64 --steps 2400 --baseline-info "${SMP_00031_RUN_INFO}" --out "${EXECUTION}/accept"
STAGE=smoke
printf '%s\n' "${STAGE}" > "${EXECUTION}/progress.txt"
smoke="00042_smoke_seed2027_${TAG}"
[[ ! -e "${SMP_G1_ROOT}/logs/${smoke}" && ! -e "${SMP_G1_ROOT}/ckpt/${smoke}" ]] || exit 2
run_logged "${SMP_G1_ROOT}/logs/${smoke}" "${PYTHON}" experiments/00042/run_00042.py \
  --num-envs 1024 --max-iterations 2 --seed 2027 --initial-checkpoint "${SMP_INITIAL_CHECKPOINT}" \
  --baseline-info "${SMP_00031_RUN_INFO}" --run "${smoke}" \
  --log-root "${SMP_G1_ROOT}/logs" --ckpt-root "${SMP_G1_ROOT}/ckpt"
run_logged "${EXECUTION}/smoke_verify" "${PYTHON}" experiments/00042/verify_completion.py \
  --iterations 2 --log-dir "${SMP_G1_ROOT}/logs/${smoke}" --ckpt-dir "${SMP_G1_ROOT}/ckpt/${smoke}"
STAGE=preflight_before_formal
printf '%s\n' "${STAGE}" > "${EXECUTION}/progress.txt"
run_logged "${EXECUTION}/preflight_before_formal" "${PYTHON}" experiments/00042/preflight.py \
  --baseline-info "${SMP_00031_RUN_INFO}" --initial-checkpoint "${SMP_INITIAL_CHECKPOINT}" \
  --expected "${EXECUTION}/preflight.json" --out "${EXECUTION}/preflight_before_formal.json"
STAGE=formal_training
printf '%s\n' "${STAGE}" > "${EXECUTION}/progress.txt"
run="00042_location_seed2027_${TAG}"
[[ ! -e "${SMP_G1_ROOT}/logs/${run}" && ! -e "${SMP_G1_ROOT}/ckpt/${run}" ]] || exit 2
mkdir "${SMP_G1_ROOT}/logs/${run}"
printf 'formal_run=%s\nsmoke_run=%s\n' "${run}" "${smoke}" > "${EXECUTION}/runs.txt"
check_source
# Fix formal dimensions explicitly; ignore caller's smoke overrides.
NUM_ENVS=1024 MAX_ITERATIONS=10000 SEED=2027 DEVICE=cuda:0 \
  bash experiments/00042/launch_00042.sh --worker "${run}"
STAGE=formal_verification
printf '%s\n' "${STAGE}" > "${EXECUTION}/progress.txt"
run_logged "${EXECUTION}/formal_verify" "${PYTHON}" experiments/00042/verify_completion.py \
  --log-dir "${SMP_G1_ROOT}/logs/${run}" --ckpt-dir "${SMP_G1_ROOT}/ckpt/${run}" \
  --out "${SMP_G1_ROOT}/logs/${run}/COMPLETION.md"
STAGE=complete
printf 'complete\n' > "${EXECUTION}/progress.txt"
printf '00042 PASS\n' > "${EXECUTION}/completed.txt"
