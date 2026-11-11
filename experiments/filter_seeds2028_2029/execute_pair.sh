#!/usr/bin/env bash
# Start on 00052; use the existing 0005/0006 untrained controls, not new baselines.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO}"
[[ "$(git branch --show-current)" == 00052 && -z "$(git status --porcelain)" ]] || {
  printf 'start on clean branch 00052; preserve any existing changes\n' >&2
  exit 2
}
SMP_G1_ROOT="$(dirname "${REPO}")"
exec 9>"${SMP_G1_ROOT}/.0003_00031_queue.lock"
flock -n 9 || { printf 'another SMP workflow holds the shared lock\n' >&2; exit 2; }
source experiments/00052/activate_00052.sh
PYTHON="${SMP_G1_ROOT}/venv/bin/python"
git fetch origin refs/heads/00062:refs/remotes/origin/00062
COMMIT_A="$(git rev-parse HEAD)"
COMMIT_B="$(git rev-parse refs/remotes/origin/00062)"
mkdir -p "${SMP_G1_ROOT}/logs" "${SMP_G1_ROOT}/ckpt"
STAMP="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
EXECUTION="$(mktemp -d "${SMP_G1_ROOT}/logs/filter_00052_00062_${STAMP}_XXXXXX")"
TAG="$(basename "${EXECUTION}")"
STAGE=prepare
CURRENT=00052
CURRENT_COMMIT="${COMMIT_A}"
on_exit() {
  local rc=$?
  printf 'queue_exit_code=%s\nlast_experiment=%s\nlast_stage=%s\n' "${rc}" "${CURRENT}" "${STAGE}" > "${EXECUTION}/exit_code.txt"
  printf 'QUEUE_EXIT_CODE=%s\nQUEUE_DIRECTORY=%s\n' "${rc}" "${EXECUTION}"
}
trap on_exit EXIT
printf '00052_commit=%s\n00062_commit=%s\n' "${COMMIT_A}" "${COMMIT_B}" > "${EXECUTION}/revisions.txt"
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
for CURRENT in 00052 00062; do
  if [[ "${CURRENT}" == 00062 ]]; then
    [[ "$(git branch --show-current)" == 00052 && "$(git rev-parse HEAD)" == "${COMMIT_A}" && -z "$(git status --porcelain)" ]] || exit 2
    if git show-ref --verify --quiet refs/heads/00062; then
      git switch 00062
      git merge --ff-only "${COMMIT_B}"
    else
      git switch -c 00062 "${COMMIT_B}"
    fi
    CURRENT_COMMIT="${COMMIT_B}"
  fi
  source "experiments/${CURRENT}/activate_${CURRENT}.sh"
  check_source
  prefix="experiments/${CURRENT}"
  if [[ "${CURRENT}" == 00052 ]]; then SMP_EXPERIMENT_SEED=2028; else SMP_EXPERIMENT_SEED=2029; fi
  CONTROL_INFO="$("${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["control"]["run_info"])' "${prefix}/EXPERIMENT.json")"
  CONTROL_CKPT="$("${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["control"]["initial_checkpoint"])' "${prefix}/EXPERIMENT.json")"
  extra=(--initial-checkpoint "${CONTROL_CKPT}" --control-info "${CONTROL_INFO}")
  mark_stage preflight
  run_logged "${EXECUTION}/${CURRENT}_preflight" "${PYTHON}" "${prefix}/contract.py" --environment --out "${EXECUTION}/${CURRENT}_environment.json"
  # Pin the known, completed same-seed baseline. No simulation or RNG draws.
  mark_stage control_initialization
  "${PYTHON}" - "${prefix}" "${EXECUTION}/${CURRENT}_control.json" <<'PY'
import json,sys
from pathlib import Path
prefix=Path(sys.argv[1]).resolve()
sys.path.insert(0,str(prefix))
from initialization import validate_control,sha256
spec=json.loads((prefix/'EXPERIMENT.json').read_text(encoding='utf-8'))
control=spec['control']
info=validate_control(control['initial_checkpoint'],control['run_info'])
assert info['initial_models_sha256']==control['initial_models_sha256']
assert info['experiment']==control['experiment'] and info['seed']==spec['seed']==control['seed']
Path(sys.argv[2]).write_text(json.dumps(dict(experiment=spec['experiment'],seed=spec['seed'],control=control,control_info_sha256=sha256(control['run_info']),initial_checkpoint_sha256=sha256(control['initial_checkpoint'])),indent=2)+'\n',encoding='utf-8')
print('EXISTING_SAME_SEED_UNTRAINED_CONTROL=PASS')
PY
  mark_stage tests
  for test in test_reward_formula test_passive_metrics test_initialization test_physics test_runtime; do
    run_logged "${EXECUTION}/${CURRENT}_${test}" "${PYTHON}" "${prefix}/${test}.py"
  done
  mark_stage environment_acceptance
  run_logged "${EXECUTION}/${CURRENT}_accept_console" "${PYTHON}" "${prefix}/accept_${CURRENT}.py" --num-envs 64 --steps 2400 --seed "${SMP_EXPERIMENT_SEED}" --out "${EXECUTION}/${CURRENT}_accept"
  mark_stage smoke
  smoke="${CURRENT}_smoke_seed${SMP_EXPERIMENT_SEED}_${TAG}"
  [[ ! -e "${SMP_G1_ROOT}/logs/${smoke}" && ! -e "${SMP_G1_ROOT}/ckpt/${smoke}" ]] || exit 2
  run_logged "${SMP_G1_ROOT}/logs/${smoke}" "${PYTHON}" "${prefix}/run_${CURRENT}.py" --num-envs 1024 --max-iterations 2 --seed "${SMP_EXPERIMENT_SEED}" --device cuda:0 --run "${smoke}" --log-root "${SMP_G1_ROOT}/logs" --ckpt-root "${SMP_G1_ROOT}/ckpt" "${extra[@]}"
  run_logged "${EXECUTION}/${CURRENT}_smoke_verify" "${PYTHON}" "${prefix}/verify_completion.py" --iterations 2 --log-dir "${SMP_G1_ROOT}/logs/${smoke}" --ckpt-dir "${SMP_G1_ROOT}/ckpt/${smoke}"
  mark_stage formal_preflight
  run_logged "${EXECUTION}/${CURRENT}_formal_preflight" "${PYTHON}" "${prefix}/contract.py" --environment
  mark_stage formal_training
  run="${CURRENT}_location_seed${SMP_EXPERIMENT_SEED}_${TAG}"
  [[ ! -e "${SMP_G1_ROOT}/logs/${run}" && ! -e "${SMP_G1_ROOT}/ckpt/${run}" ]] || exit 2
  printf '%s=%s\n' "${CURRENT}" "${run}" >> "${EXECUTION}/runs.txt"
  run_logged "${SMP_G1_ROOT}/logs/${run}" "${PYTHON}" "${prefix}/run_${CURRENT}.py" --num-envs 1024 --max-iterations 10000 --seed "${SMP_EXPERIMENT_SEED}" --device cuda:0 --run "${run}" --log-root "${SMP_G1_ROOT}/logs" --ckpt-root "${SMP_G1_ROOT}/ckpt" "${extra[@]}"
  mark_stage formal_verification
  run_logged "${EXECUTION}/${CURRENT}_formal_verify" "${PYTHON}" "${prefix}/verify_completion.py" --log-dir "${SMP_G1_ROOT}/logs/${run}" --ckpt-dir "${SMP_G1_ROOT}/ckpt/${run}" --out "${SMP_G1_ROOT}/logs/${run}/COMPLETION.md"
done
mark_stage complete
"${PYTHON}" - "${SMP_G1_ROOT}" "${EXECUTION}" <<'PY'
import json,sys
from pathlib import Path
root,execution=map(Path,sys.argv[1:])
runs=dict(line.split('=',1) for line in (execution/'runs.txt').read_text().splitlines())
lines=['# 00052与00062完成','', '各1024环境、10000轮、245760000训练样本；只做物理筛选，关节修正关闭。','']
for exp,seed in [('00052',2028),('00062',2029)]:
    info=json.loads((root/'logs'/runs[exp]/'run_info.json').read_text())
    control=json.loads(Path(info['control_info']).read_text())
    assert info['experiment']==exp and info['seed']==seed
    assert info['completed_iterations']==10000 and info['total_samples']==245760000
    assert info['initial_models_sha256']==control['initial_models_sha256']
    assert info['gsi_physics_config']['repair'] is False
    assert all(info['gsi_physics_totals'][k]==0 for k in ('repair_attempted','repair_qualified','repair_inserted'))
    lines.append(f"- {exp}（seed{seed}）：{info['log_dir']}；公共初始模型指纹 {info['initial_models_sha256']}")
(execution/'COMPLETION.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
PY
printf '00052=PASS\n00062=PASS\n' > "${EXECUTION}/completed.txt"
