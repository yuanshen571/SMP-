#!/usr/bin/env bash
# Run in tmux. The same shared checkout switches only after a process exits.
set -euo pipefail
SMP_G1_ROOT="${SMP_G1_ROOT:-/root/gpufree-data/smp-g1}"
REPO="${SMP_G1_ROOT}/repo"
source "${REPO}/experiments/0004/activate_0004.sh"
PYTHON="${SMP_G1_ROOT}/venv/bin/python"
: "${SMP_INITIAL_CHECKPOINT:?set the common untrained checkpoint used by 00031}"
: "${SMP_00031_RUN_INFO:?set the actual completed 00031 run_info.json}"
export SMP_INITIAL_CHECKPOINT SMP_00031_RUN_INFO
[[ -f "${SMP_INITIAL_CHECKPOINT}" && -f "${SMP_00031_RUN_INFO}" ]] || exit 2
cd "${REPO}"
[[ -z "$(git status --porcelain)" ]] || { printf 'working tree is dirty; stop without reset\n' >&2; exit 2; }
exec 9>"${SMP_G1_ROOT}/.0003_00031_queue.lock"
flock -n 9 || { printf 'another baseline/TS2Vec/cluster queue is still running\n' >&2; exit 2; }
git fetch origin refs/heads/0004:refs/remotes/origin/0004 refs/heads/00041:refs/remotes/origin/00041
COMMIT_0004="$(git rev-parse origin/0004)"
COMMIT_00041="$(git rev-parse origin/00041)"
STAMP="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
QUEUE="$(mktemp -d "${SMP_G1_ROOT}/logs/queue_0004_00041_${STAMP}_XXXXXX")"
STAGE=preflight
on_exit() {
  local rc=$?
  printf 'queue_exit_code=%s\nlast_stage=%s\n' "${rc}" "${STAGE}" > "${QUEUE}/exit_code.txt"
  printf 'QUEUE_EXIT_CODE=%s\nQUEUE_DIRECTORY=%s\n' "${rc}" "${QUEUE}"
}
trap on_exit EXIT
"${PYTHON}" - "${QUEUE}/queue_info.json" "${COMMIT_0004}" "${COMMIT_00041}" "${SMP_INITIAL_CHECKPOINT}" "${SMP_00031_RUN_INFO}" <<'PY'
import json,sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({'order':['0004','00041'],'commits':dict(zip(['0004','00041'],sys.argv[2:4])),'initial_checkpoint':sys.argv[4],'baseline_info':sys.argv[5]},indent=2)+'\n',encoding='utf-8')
PY
run_logged() {
  local folder="$1"
  shift
  mkdir -p "${folder}"
  set +e
  "$@" 2>&1 | tee "${folder}/console.log"
  local statuses=("${PIPESTATUS[@]}")
  set -e
  printf 'python_exit_code=%s\ntee_exit_code=%s\n' "${statuses[0]}" "${statuses[1]}" > "${folder}/exit_code.txt"
  if [[ "${statuses[0]}" != 0 ]]; then return "${statuses[0]}"; fi
  if [[ "${statuses[1]}" != 0 ]]; then return "${statuses[1]}"; fi
}
# Freeze both existing modules before the first training process starts.
run_logged "${QUEUE}/preflight_initial" "${PYTHON}" experiments/0004/preflight.py \
  --baseline-info "${SMP_00031_RUN_INFO}" --initial-checkpoint "${SMP_INITIAL_CHECKPOINT}" \
  --out "${QUEUE}/preflight.json"
for experiment in 0004 00041; do
  if [[ "${experiment}" == 0004 ]]; then expected="${COMMIT_0004}"; else expected="${COMMIT_00041}"; fi
  if git show-ref --verify --quiet "refs/heads/${experiment}"; then
    git switch "${experiment}"
    git merge --ff-only "${expected}"
  else
    git switch -c "${experiment}" "${expected}"
  fi
  [[ "$(git rev-parse HEAD)" == "${expected}" ]] || { printf 'branch diverged; stop without reset\n' >&2; exit 2; }
  STAGE="${experiment}_preflight"
  printf '%s\n' "${STAGE}" > "${QUEUE}/progress.txt"
  run_logged "${QUEUE}/${experiment}_preflight" "${PYTHON}" "experiments/${experiment}/preflight.py" \
    --baseline-info "${SMP_00031_RUN_INFO}" --initial-checkpoint "${SMP_INITIAL_CHECKPOINT}" \
    --expected "${QUEUE}/preflight.json" --out "${QUEUE}/${experiment}_preflight.json"
  STAGE="${experiment}_acceptance"
  printf '%s\n' "${STAGE}" > "${QUEUE}/progress.txt"
  for test in test_reward_formula test_passive_metrics test_physics test_runtime test_initialization test_clusters; do
    run_logged "${QUEUE}/${experiment}_${test}" "${PYTHON}" "experiments/${experiment}/${test}.py"
  done
  run_logged "${QUEUE}/${experiment}_accept_console" "${PYTHON}" "experiments/${experiment}/accept_${experiment}.py" \
    --num-envs 64 --steps 2400 --baseline-info "${SMP_00031_RUN_INFO}" --out "${QUEUE}/${experiment}_accept"
  STAGE="${experiment}_smoke"
  printf '%s\n' "${STAGE}" > "${QUEUE}/progress.txt"
  smoke="${experiment}_smoke_seed2027_${STAMP}"
  run_logged "${SMP_G1_ROOT}/logs/${smoke}" "${PYTHON}" "experiments/${experiment}/run_${experiment}.py" \
    --num-envs 1024 --max-iterations 2 --seed 2027 --initial-checkpoint "${SMP_INITIAL_CHECKPOINT}" \
    --baseline-info "${SMP_00031_RUN_INFO}" --run "${smoke}" \
    --log-root "${SMP_G1_ROOT}/logs" --ckpt-root "${SMP_G1_ROOT}/ckpt"
  run_logged "${QUEUE}/${experiment}_smoke_verify" "${PYTHON}" "experiments/${experiment}/verify_completion.py" \
    --iterations 2 --log-dir "${SMP_G1_ROOT}/logs/${smoke}" --ckpt-dir "${SMP_G1_ROOT}/ckpt/${smoke}"
  STAGE="${experiment}_formal_training"
  printf '%s\n' "${STAGE}" > "${QUEUE}/progress.txt"
  run="${experiment}_location_seed2027_${STAMP}"
  mkdir "${SMP_G1_ROOT}/logs/${run}"
  bash "experiments/${experiment}/launch_${experiment}.sh" --worker "${run}"
  run_logged "${QUEUE}/${experiment}_formal_verify" "${PYTHON}" "experiments/${experiment}/verify_completion.py" \
    --log-dir "${SMP_G1_ROOT}/logs/${run}" --ckpt-dir "${SMP_G1_ROOT}/ckpt/${run}" \
    --out "${SMP_G1_ROOT}/logs/${run}/COMPLETION.md"
  printf '%s PASS\n' "${experiment}" >> "${QUEUE}/completed.txt"
done
STAGE=complete
printf 'complete\n' > "${QUEUE}/progress.txt"
