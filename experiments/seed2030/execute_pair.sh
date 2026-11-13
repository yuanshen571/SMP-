#!/usr/bin/env bash
# Run this workflow in a server tmux session, beginning on branch 0007.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO}"
[[ "$(git branch --show-current)" == 0007 && -z "$(git status --porcelain)" ]] || {
  printf 'start on clean branch 0007 (seed 2030); do not reset or overwrite\n' >&2
  exit 2
}
SMP_G1_ROOT="$(dirname "${REPO}")"
exec 9>"${SMP_G1_ROOT}/.0003_00031_queue.lock"
flock -n 9 || { printf 'another SMP workflow holds the shared lock\n' >&2; exit 2; }
source experiments/0007/activate_0007.sh
PYTHON="${SMP_G1_ROOT}/venv/bin/python"
EXPECTED_TARGET="${1:?provide the delivered full 00073 commit SHA}"
[[ "${EXPECTED_TARGET}" =~ ^[0-9a-f]{40}$ ]] || exit 2
# Obtain the other branch once, before any training, then pin both revisions.
git fetch origin refs/heads/00073:refs/remotes/origin/00073
COMMIT_A="$(git rev-parse HEAD)"
COMMIT_B="$(git rev-parse refs/remotes/origin/00073)"
[[ "${COMMIT_B}" == "${EXPECTED_TARGET}" ]] || { printf '00073 differs from pinned delivery; stop\n' >&2; exit 2; }
QUEUE_FILE="${REPO}/experiments/seed2030/execute_pair.sh"
SCRIPT_SHA="$(sha256sum "${QUEUE_FILE}" | cut -d' ' -f1)"
TARGET_SCRIPT_SHA="$(git show "${COMMIT_B}:experiments/seed2030/execute_pair.sh" | sha256sum | cut -d' ' -f1)"
[[ "${SCRIPT_SHA}" == "${TARGET_SCRIPT_SHA}" ]] || { printf 'queue differs across branches; stop\n' >&2; exit 2; }
"${PYTHON}" - "${COMMIT_B}" <<'PYTARGET'
import json,subprocess,sys
spec=json.loads(subprocess.check_output(['git','show',sys.argv[1]+':experiments/00073/EXPERIMENT.json']))
assert spec['experiment']=='00073' and spec['seed']==2030
assert spec['control']['experiment']=='0007' and spec['control']['seed']==2030
assert spec['noise_rescue']['max_generations']==30 and spec['noise_rescue']['rms_limit']==0.15
assert spec['physical_filter']['repair_small_joints'] is False
assert spec['mixed_cache_enabled'] is spec['clustering_enabled'] is False
PYTARGET
mkdir -p "${SMP_G1_ROOT}/logs" "${SMP_G1_ROOT}/ckpt"
STAMP="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
EXECUTION="$(mktemp -d "${SMP_G1_ROOT}/logs/seed2030_0007_00073_${STAMP}_XXXXXX")"
TAG="$(basename "${EXECUTION}")"
STAGE=prepare
CURRENT=0007
CURRENT_COMMIT="${COMMIT_A}"
on_exit() {
  local rc=$?
  printf 'queue_exit_code=%s\nlast_experiment=%s\nlast_stage=%s\n' "${rc}" "${CURRENT}" "${STAGE}" > "${EXECUTION}/exit_code.txt"
  printf 'QUEUE_EXIT_CODE=%s\nQUEUE_DIRECTORY=%s\n' "${rc}" "${EXECUTION}"
}
trap on_exit EXIT
GUARD="${SMP_G1_ROOT}/logs/.seed2030_0007_00073_registered"
if ! (set -o noclobber; printf 'queue=%s\n0007_commit=%s\n00073_commit=%s\n' "${EXECUTION}" "${COMMIT_A}" "${COMMIT_B}" > "${GUARD}") 2>/dev/null; then
  printf 'this seed-2030 pair was already registered; inspect its existing queue, do not launch twice\n' >&2
  exit 2
fi
printf '0007_commit=%s\n00073_commit=%s\nseed=2030\n' "${COMMIT_A}" "${COMMIT_B}" > "${EXECUTION}/revisions.txt"
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
for CURRENT in 0007 00073; do
  if [[ "${CURRENT}" == 00073 ]]; then
    check_source_before_switch="$(git rev-parse HEAD)"
    [[ "$(git branch --show-current)" == 0007 && "${check_source_before_switch}" == "${COMMIT_A}" && -z "$(git status --porcelain)" ]] || exit 2
    if git show-ref --verify --quiet refs/heads/00073; then
      git switch 00073
      git merge --ff-only "${COMMIT_B}"
    else
      git switch -c 00073 "${COMMIT_B}"
    fi
    CURRENT_COMMIT="${COMMIT_B}"
  fi
  source "experiments/${CURRENT}/activate_${CURRENT}.sh"
  check_source
  prefix="experiments/${CURRENT}"
  extra=()
  if [[ "${CURRENT}" == 00073 ]]; then
    extra=(--initial-checkpoint "${CONTROL_CKPT}" --control-info "${CONTROL_INFO}")
  fi
  mark_stage preflight
  run_logged "${EXECUTION}/${CURRENT}_preflight" "${PYTHON}" "${prefix}/contract.py" --environment --out "${EXECUTION}/${CURRENT}_environment.json"
  mark_stage tests
  tests=(test_variant test_reward_formula test_passive_metrics test_initialization)
  if [[ "${CURRENT}" == 00073 ]]; then tests+=(test_physics test_runtime test_scoring); fi
  for test in "${tests[@]}"; do
    run_logged "${EXECUTION}/${CURRENT}_${test}" "${PYTHON}" "${prefix}/${test}.py"
  done
  if [[ "${CURRENT}" == 00073 ]]; then
    mark_stage frozen_prior_sampler
    run_logged "${EXECUTION}/00073_sampler_console" "${PYTHON}" "${prefix}/accept_sampler.py" --out "${EXECUTION}/00073_sampler.json"
  fi
  mark_stage environment_acceptance
  run_logged "${EXECUTION}/${CURRENT}_accept_console" "${PYTHON}" "${prefix}/accept_${CURRENT}.py" --num-envs 64 --steps 2400 --seed 2030 --out "${EXECUTION}/${CURRENT}_accept"
  mark_stage smoke
  smoke="${CURRENT}_smoke_seed2030_${TAG}"
  [[ ! -e "${SMP_G1_ROOT}/logs/${smoke}" && ! -e "${SMP_G1_ROOT}/ckpt/${smoke}" ]] || exit 2
  run_logged "${SMP_G1_ROOT}/logs/${smoke}" "${PYTHON}" "${prefix}/run_${CURRENT}.py" --num-envs 1024 --max-iterations 2 --seed 2030 --device cuda:0 --run "${smoke}" --log-root "${SMP_G1_ROOT}/logs" --ckpt-root "${SMP_G1_ROOT}/ckpt" "${extra[@]}"
  run_logged "${EXECUTION}/${CURRENT}_smoke_verify" "${PYTHON}" "${prefix}/verify_completion.py" --iterations 2 --log-dir "${SMP_G1_ROOT}/logs/${smoke}" --ckpt-dir "${SMP_G1_ROOT}/ckpt/${smoke}"
  mark_stage formal_preflight
  run_logged "${EXECUTION}/${CURRENT}_formal_preflight" "${PYTHON}" "${prefix}/contract.py" --environment
  mark_stage formal_training
  run="${CURRENT}_location_seed2030_${TAG}"
  [[ ! -e "${SMP_G1_ROOT}/logs/${run}" && ! -e "${SMP_G1_ROOT}/ckpt/${run}" ]] || exit 2
  printf '%s=%s\n' "${CURRENT}" "${run}" >> "${EXECUTION}/runs.txt"
  run_logged "${SMP_G1_ROOT}/logs/${run}" "${PYTHON}" "${prefix}/run_${CURRENT}.py" --num-envs 1024 --max-iterations 10000 --seed 2030 --device cuda:0 --run "${run}" --log-root "${SMP_G1_ROOT}/logs" --ckpt-root "${SMP_G1_ROOT}/ckpt" "${extra[@]}"
  mark_stage formal_verification
  run_logged "${EXECUTION}/${CURRENT}_formal_verify" "${PYTHON}" "${prefix}/verify_completion.py" --log-dir "${SMP_G1_ROOT}/logs/${run}" --ckpt-dir "${SMP_G1_ROOT}/ckpt/${run}" --out "${SMP_G1_ROOT}/logs/${run}/COMPLETION.md"
  if [[ "${CURRENT}" == 0007 ]]; then
    CONTROL_INFO="${SMP_G1_ROOT}/logs/${run}/run_info.json"
    CONTROL_CKPT="${SMP_G1_ROOT}/ckpt/${run}/initial_untrained.pt"
    export CONTROL_INFO CONTROL_CKPT
  else
    "${PYTHON}" - "${CONTROL_INFO}" "${SMP_G1_ROOT}/logs/${run}/run_info.json" "${EXECUTION}/COMPLETION.md" <<'PY'
import json,sys
from pathlib import Path
a,b=(json.loads(Path(p).read_text()) for p in sys.argv[1:3])
assert a['experiment']=='0007' and b['experiment']=='00073'
assert a['seed']==b['seed']==2030
assert a['physical_filter_enabled'] is False and b['physical_filter_enabled'] is True
assert b['noise_rescue_enabled'] is True
assert b['noise_rescue_config']['max_generations']==30 and b['noise_rescue_config']['rms_limit']==0.15
assert b['control_info']==sys.argv[1]
assert a['initial_models_sha256']==b['initial_models_sha256']
assert a['saved_initial_checkpoint_sha256']==b['initial_checkpoint_sha256']
assert a['completed_iterations']==b['completed_iterations']==10000
assert a['total_samples']==b['total_samples']==245760000
Path(sys.argv[3]).write_text('# 种子2030：0007与00073完成\n\n两组各10000轮、245760000样本，单组严格完成核验均通过。共同未训练Actor/Critic逐张量指纹一致，各自独立优化器。\n\n- 0007日志：'+a['log_dir']+'\n- 00073日志：'+b['log_dir']+'\n- 公共模型指纹：'+a['initial_models_sha256']+'\n',encoding='utf-8')
PY
  fi
done
mark_stage complete
printf '0007=PASS\n00073=PASS\n' > "${EXECUTION}/completed.txt"
