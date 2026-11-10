#!/usr/bin/env bash
# Run this workflow in a server tmux session, beginning on branch 0005.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO}"
[[ "$(git branch --show-current)" == 0005 && -z "$(git status --porcelain)" ]] || {
  printf 'start on clean branch 0005 (seed 2028); do not reset or overwrite\n' >&2
  exit 2
}
source experiments/0005/activate_0005.sh
SMP_G1_ROOT="$(dirname "${REPO}")"
PYTHON="${SMP_G1_ROOT}/venv/bin/python"
exec 9>"${SMP_G1_ROOT}/.0003_00031_queue.lock"
flock -n 9 || { printf 'another SMP workflow holds the shared lock\n' >&2; exit 2; }
# Obtain the other branch once, before any training, then pin both revisions.
git fetch origin refs/heads/00051:refs/remotes/origin/00051
COMMIT_A="$(git rev-parse HEAD)"
COMMIT_B="$(git rev-parse refs/remotes/origin/00051)"
mkdir -p "${SMP_G1_ROOT}/logs" "${SMP_G1_ROOT}/ckpt"
STAMP="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
EXECUTION="$(mktemp -d "${SMP_G1_ROOT}/logs/seed2028_0005_00051_${STAMP}_XXXXXX")"
TAG="$(basename "${EXECUTION}")"
STAGE=prepare
CURRENT=0005
CURRENT_COMMIT="${COMMIT_A}"
on_exit() {
  local rc=$?
  printf 'queue_exit_code=%s\nlast_experiment=%s\nlast_stage=%s\n' "${rc}" "${CURRENT}" "${STAGE}" > "${EXECUTION}/exit_code.txt"
  printf 'QUEUE_EXIT_CODE=%s\nQUEUE_DIRECTORY=%s\n' "${rc}" "${EXECUTION}"
}
trap on_exit EXIT
printf '0005_commit=%s\n00051_commit=%s\nseed=2028\n' "${COMMIT_A}" "${COMMIT_B}" > "${EXECUTION}/revisions.txt"
check_source() {
  [[ "$(git branch --show-current)" == "${CURRENT}" && "$(git rev-parse HEAD)" == "${CURRENT_COMMIT}" && -z "$(git status --porcelain)" ]] || {
    printf 'source changed during execution; stop and preserve it\n' >&2
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
mark_stage() {
  STAGE="$1"
  printf 'experiment=%s\nstage=%s\n' "${CURRENT}" "${STAGE}" > "${EXECUTION}/progress.txt"
}
for CURRENT in 0005 00051; do
  if [[ "${CURRENT}" == 00051 ]]; then
    check_source_before_switch="$(git rev-parse HEAD)"
    [[ "$(git branch --show-current)" == 0005 && "${check_source_before_switch}" == "${COMMIT_A}" && -z "$(git status --porcelain)" ]] || exit 2
    if git show-ref --verify --quiet refs/heads/00051; then
      git switch 00051
      git merge --ff-only "${COMMIT_B}"
    else
      git switch -c 00051 "${COMMIT_B}"
    fi
    CURRENT_COMMIT="${COMMIT_B}"
  fi
  source "experiments/${CURRENT}/activate_${CURRENT}.sh"
  check_source
  prefix="experiments/${CURRENT}"
  extra=()
  if [[ "${CURRENT}" == 00051 ]]; then
    extra=(--initial-checkpoint "${CONTROL_CKPT}" --control-info "${CONTROL_INFO}")
  fi
  mark_stage preflight
  run_logged "${EXECUTION}/${CURRENT}_preflight" "${PYTHON}" "${prefix}/contract.py" --environment --out "${EXECUTION}/${CURRENT}_environment.json"
  mark_stage tests
  tests=(test_reward_formula test_passive_metrics test_initialization)
  if [[ "${CURRENT}" == 00051 ]]; then tests+=(test_physics test_runtime); fi
  for test in "${tests[@]}"; do
    run_logged "${EXECUTION}/${CURRENT}_${test}" "${PYTHON}" "${prefix}/${test}.py"
  done
  mark_stage environment_acceptance
  run_logged "${EXECUTION}/${CURRENT}_accept_console" "${PYTHON}" "${prefix}/accept_${CURRENT}.py" --num-envs 64 --steps 2400 --seed 2028 --out "${EXECUTION}/${CURRENT}_accept"
  mark_stage smoke
  smoke="${CURRENT}_smoke_seed2028_${TAG}"
  [[ ! -e "${SMP_G1_ROOT}/logs/${smoke}" && ! -e "${SMP_G1_ROOT}/ckpt/${smoke}" ]] || exit 2
  run_logged "${SMP_G1_ROOT}/logs/${smoke}" "${PYTHON}" "${prefix}/run_${CURRENT}.py" --num-envs 1024 --max-iterations 2 --seed 2028 --device cuda:0 --run "${smoke}" --log-root "${SMP_G1_ROOT}/logs" --ckpt-root "${SMP_G1_ROOT}/ckpt" "${extra[@]}"
  run_logged "${EXECUTION}/${CURRENT}_smoke_verify" "${PYTHON}" "${prefix}/verify_completion.py" --iterations 2 --log-dir "${SMP_G1_ROOT}/logs/${smoke}" --ckpt-dir "${SMP_G1_ROOT}/ckpt/${smoke}"
  mark_stage formal_preflight
  run_logged "${EXECUTION}/${CURRENT}_formal_preflight" "${PYTHON}" "${prefix}/contract.py" --environment
  mark_stage formal_training
  run="${CURRENT}_location_seed2028_${TAG}"
  [[ ! -e "${SMP_G1_ROOT}/logs/${run}" && ! -e "${SMP_G1_ROOT}/ckpt/${run}" ]] || exit 2
  printf '%s=%s\n' "${CURRENT}" "${run}" >> "${EXECUTION}/runs.txt"
  run_logged "${SMP_G1_ROOT}/logs/${run}" "${PYTHON}" "${prefix}/run_${CURRENT}.py" --num-envs 1024 --max-iterations 10000 --seed 2028 --device cuda:0 --run "${run}" --log-root "${SMP_G1_ROOT}/logs" --ckpt-root "${SMP_G1_ROOT}/ckpt" "${extra[@]}"
  mark_stage formal_verification
  run_logged "${EXECUTION}/${CURRENT}_formal_verify" "${PYTHON}" "${prefix}/verify_completion.py" --log-dir "${SMP_G1_ROOT}/logs/${run}" --ckpt-dir "${SMP_G1_ROOT}/ckpt/${run}" --out "${SMP_G1_ROOT}/logs/${run}/COMPLETION.md"
  if [[ "${CURRENT}" == 0005 ]]; then
    CONTROL_INFO="${SMP_G1_ROOT}/logs/${run}/run_info.json"
    CONTROL_CKPT="${SMP_G1_ROOT}/ckpt/${run}/initial_untrained.pt"
    export CONTROL_INFO CONTROL_CKPT
  else
    "${PYTHON}" - "${CONTROL_INFO}" "${SMP_G1_ROOT}/logs/${run}/run_info.json" "${EXECUTION}/COMPLETION.md" <<'PY'
import json,sys
from pathlib import Path
a,b=(json.loads(Path(p).read_text()) for p in sys.argv[1:3])
assert a['seed']==b['seed']==2028
assert a['initial_models_sha256']==b['initial_models_sha256']
assert a['saved_initial_checkpoint_sha256']==b['initial_checkpoint_sha256']
assert a['completed_iterations']==b['completed_iterations']==10000
assert a['total_samples']==b['total_samples']==245760000
Path(sys.argv[3]).write_text('# 种子2028：0005与00051完成\n\n两组各10000轮、245760000样本，单组严格完成核验均通过。共同未训练Actor/Critic逐张量指纹一致，各自独立优化器。\n\n- 0005日志：'+a['log_dir']+'\n- 00051日志：'+b['log_dir']+'\n- 公共模型指纹：'+a['initial_models_sha256']+'\n',encoding='utf-8')
PY
  fi
done
mark_stage complete
printf '0005=PASS\n00051=PASS\n' > "${EXECUTION}/completed.txt"
