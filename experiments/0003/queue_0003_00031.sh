#!/usr/bin/env bash
# Run inside a persistent tmux session; one shared checkout, sequential processes.
set -euo pipefail
SMP_G1_ROOT="${SMP_G1_ROOT:-/root/gpufree-data/smp-g1}"
REPO="${SMP_G1_ROOT}/repo"
source "${REPO}/experiments/0003/activate_0003.sh"
PYTHON="${SMP_G1_ROOT}/venv/bin/python"
: "${SMP_INITIAL_CHECKPOINT:?set one common 0002 initial_untrained.pt}"
[[ -f "${SMP_INITIAL_CHECKPOINT}" ]] || exit 2
cd "${REPO}"
[[ -z "$(git status --porcelain)" ]] || { printf 'working tree is dirty; stop\n' >&2; exit 2; }
exec 9>"${SMP_G1_ROOT}/.0003_00031_queue.lock"
flock -n 9 || { printf 'another queue is already running\n' >&2; exit 2; }
git fetch origin refs/heads/0003:refs/remotes/origin/0003 refs/heads/00031:refs/remotes/origin/00031
COMMIT_0003="$(git rev-parse origin/0003)"
COMMIT_00031="$(git rev-parse origin/00031)"
STAMP="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
QUEUE="${SMP_G1_ROOT}/logs/queue_0003_00031_${STAMP}"
mkdir "${QUEUE}"
trap 'rc=$?; printf "queue_exit_code=%s\n" "${rc}" > "${QUEUE}/exit_code.txt"' EXIT
"${PYTHON}" - "${QUEUE}/queue_info.json" "${COMMIT_0003}" "${COMMIT_00031}" "${SMP_INITIAL_CHECKPOINT}" <<'PY'
import json,sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({'order':['0003','00031'],'commits':dict(zip(['0003','00031'],sys.argv[2:4])),'initial_checkpoint':sys.argv[4]},indent=2)+'\n',encoding='utf-8')
PY
run_logged() {
  local folder="$1"; shift
  mkdir -p "${folder}"
  set +e
  "$@" 2>&1 | tee "${folder}/console.log"
  local statuses=("${PIPESTATUS[@]}")
  set -e
  printf 'python_exit_code=%s\ntee_exit_code=%s\n' "${statuses[0]}" "${statuses[1]}" > "${folder}/exit_code.txt"
  [[ "${statuses[0]}" == 0 && "${statuses[1]}" == 0 ]]
}
for experiment in 0003 00031; do
  if [[ "${experiment}" == 0003 ]]; then expected="${COMMIT_0003}"; else expected="${COMMIT_00031}"; fi
  if git show-ref --verify --quiet "refs/heads/${experiment}"; then
    git switch "${experiment}"
    git merge --ff-only "${expected}"
  else
    git switch -c "${experiment}" "${expected}"
  fi
  [[ "$(git rev-parse HEAD)" == "${expected}" ]] || { printf 'local branch diverged; stop without reset\n' >&2; exit 2; }
  printf '%s acceptance\n' "${experiment}" > "${QUEUE}/progress.txt"
  run_logged "${QUEUE}/${experiment}_tests" "${PYTHON}" "experiments/${experiment}/test_reward_formula.py"
  "${PYTHON}" "experiments/${experiment}/test_passive_metrics.py"
  "${PYTHON}" "experiments/${experiment}/test_physics.py"
  "${PYTHON}" "experiments/${experiment}/test_runtime.py"
  "${PYTHON}" "experiments/${experiment}/test_initialization.py"
  "${PYTHON}" "experiments/${experiment}/accept_${experiment}.py" --num-envs 64 --steps 2400 --out "${QUEUE}/${experiment}_accept"
  smoke="${experiment}_smoke_seed2027_${STAMP}"
  run_logged "${SMP_G1_ROOT}/logs/${smoke}" "${PYTHON}" "experiments/${experiment}/run_${experiment}.py" \
    --num-envs 1024 --max-iterations 2 --seed 2027 --initial-checkpoint "${SMP_INITIAL_CHECKPOINT}" \
    --run "${smoke}" --log-root "${SMP_G1_ROOT}/logs" --ckpt-root "${SMP_G1_ROOT}/ckpt"
  "${PYTHON}" "experiments/${experiment}/verify_completion.py" --iterations 2 \
    --log-dir "${SMP_G1_ROOT}/logs/${smoke}" --ckpt-dir "${SMP_G1_ROOT}/ckpt/${smoke}"
  printf '%s formal_training\n' "${experiment}" > "${QUEUE}/progress.txt"
  run="${experiment}_location_seed2027_${STAMP}"
  mkdir "${SMP_G1_ROOT}/logs/${run}"
  bash "experiments/${experiment}/launch_${experiment}.sh" --worker "${run}"
  "${PYTHON}" "experiments/${experiment}/verify_completion.py" \
    --log-dir "${SMP_G1_ROOT}/logs/${run}" --ckpt-dir "${SMP_G1_ROOT}/ckpt/${run}"
  printf '%s PASS\n' "${experiment}" >> "${QUEUE}/completed.txt"
done
printf 'complete\n' > "${QUEUE}/progress.txt"
printf 'QUEUE_COMPLETE=%s\n' "${QUEUE}"
