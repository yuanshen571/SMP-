#!/usr/bin/env bash
# One shared repo/venv; no remote work is performed by preparing this file.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO}"
[[ "$(git branch --show-current)" == 00054 && -z "$(git status --porcelain)" ]] || {
  printf 'start on clean branch 00054; preserve any existing changes\n' >&2
  exit 2
}
SMP_G1_ROOT="$(dirname "${REPO}")"
exec 9>"${SMP_G1_ROOT}/.0003_00031_queue.lock"
flock -n 9 || { printf 'another SMP workflow holds the shared lock\n' >&2; exit 2; }
source experiments/00054/activate_00054.sh
PYTHON="${SMP_G1_ROOT}/venv/bin/python"
REVISION="$(git rev-parse HEAD)"
mkdir -p "${SMP_G1_ROOT}/logs" "${SMP_G1_ROOT}/ckpt"
STAMP="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
EXECUTION="$(mktemp -d "${SMP_G1_ROOT}/logs/00054_pipeline_${STAMP}_XXXXXX")"
TAG="$(basename "${EXECUTION}")"
STAGE=prepare
on_exit() {
  local rc=$?
  printf 'pipeline_exit_code=%s\nlast_stage=%s\n' "${rc}" "${STAGE}" > "${EXECUTION}/exit_code.txt"
  printf 'PIPELINE_EXIT_CODE=%s\nEXECUTION_DIRECTORY=%s\n' "${rc}" "${EXECUTION}"
}
trap on_exit EXIT
GUARD="${SMP_G1_ROOT}/logs/.00054_registered"
if ! (set -o noclobber; printf 'pipeline=%s\ncommit=%s\n' "${EXECUTION}" "${REVISION}" > "${GUARD}") 2>/dev/null; then
  printf '00054 was already registered; inspect the existing run, do not launch twice\n' >&2
  exit 2
fi
printf 'branch=00054\ncommit=%s\n' "${REVISION}" > "${EXECUTION}/revision.txt"
check_source() {
  [[ "$(git branch --show-current)" == 00054 && "$(git rev-parse HEAD)" == "${REVISION}" && -z "$(git status --porcelain)" ]] || {
    printf 'source changed; stop and preserve the workspace\n' >&2
    return 2
  }
}
mark_stage() {
  STAGE="$1"
  printf 'experiment=00054\nstage=%s\n' "${STAGE}" > "${EXECUTION}/progress.txt"
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
PREFIX=experiments/00054
CONTROL_INFO="$("${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["control"]["run_info"])' "${PREFIX}/EXPERIMENT.json")"
CONTROL_CKPT="$("${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["control"]["initial_checkpoint"])' "${PREFIX}/EXPERIMENT.json")"
extra=(--initial-checkpoint "${CONTROL_CKPT}" --control-info "${CONTROL_INFO}")
mark_stage source_and_environment
run_logged "${EXECUTION}/environment" "${PYTHON}" "${PREFIX}/contract.py" --environment --out "${EXECUTION}/environment.json"
mark_stage common_initialization
"${PYTHON}" - "${PREFIX}" "${EXECUTION}/control.json" <<'PY'
import json,sys
from pathlib import Path
prefix=Path(sys.argv[1]).resolve()
sys.path.insert(0,str(prefix))
from initialization import validate_control,sha256
spec=json.loads((prefix/'EXPERIMENT.json').read_text(encoding='utf-8'))
c=spec['control']
info=validate_control(c['initial_checkpoint'],c['run_info'])
assert info['initial_models_sha256']==c['initial_models_sha256']
assert info['seed']==spec['seed']==2028
Path(sys.argv[2]).write_text(json.dumps(dict(control=c,source_sha256=sha256(c['initial_checkpoint'])),indent=2)+'\n',encoding='utf-8')
print('COMMON_SEED2028_INITIALIZATION=PASS')
PY
mark_stage protocol_tests
for test in test_variant test_reward_formula test_passive_metrics test_initialization test_physics test_runtime test_scoring test_completion_policy; do
  run_logged "${EXECUTION}/${test}" "${PYTHON}" "${PREFIX}/${test}.py"
done
mark_stage frozen_prior_sampler
run_logged "${EXECUTION}/sampler_console" "${PYTHON}" "${PREFIX}/accept_sampler.py" --out "${EXECUTION}/sampler.json"
mark_stage environment_acceptance
run_logged "${EXECUTION}/accept_console" "${PYTHON}" "${PREFIX}/accept_00054.py" --num-envs 64 --steps 2400 --seed 2028 --out "${EXECUTION}/accept"
mark_stage smoke
smoke="00054_smoke_seed2028_${TAG}"
[[ ! -e "${SMP_G1_ROOT}/logs/${smoke}" && ! -e "${SMP_G1_ROOT}/ckpt/${smoke}" ]] || exit 2
run_logged "${SMP_G1_ROOT}/logs/${smoke}" "${PYTHON}" "${PREFIX}/run_00054.py" --num-envs 1024 --max-iterations 2 --seed 2028 --device cuda:0 --run "${smoke}" --log-root "${SMP_G1_ROOT}/logs" --ckpt-root "${SMP_G1_ROOT}/ckpt" "${extra[@]}"
run_logged "${EXECUTION}/smoke_verify" "${PYTHON}" "${PREFIX}/verify_completion.py" --iterations 2 --log-dir "${SMP_G1_ROOT}/logs/${smoke}" --ckpt-dir "${SMP_G1_ROOT}/ckpt/${smoke}"
mark_stage formal_preflight
run_logged "${EXECUTION}/formal_preflight" "${PYTHON}" "${PREFIX}/contract.py" --environment
mark_stage formal_training
run="00054_location_seed2028_${TAG}"
[[ ! -e "${SMP_G1_ROOT}/logs/${run}" && ! -e "${SMP_G1_ROOT}/ckpt/${run}" ]] || exit 2
printf '00054=%s\n' "${run}" > "${EXECUTION}/runs.txt"
run_logged "${SMP_G1_ROOT}/logs/${run}" "${PYTHON}" "${PREFIX}/run_00054.py" --num-envs 1024 --max-iterations 10000 --seed 2028 --device cuda:0 --run "${run}" --log-root "${SMP_G1_ROOT}/logs" --ckpt-root "${SMP_G1_ROOT}/ckpt" "${extra[@]}"
mark_stage formal_verification
run_logged "${EXECUTION}/formal_verify" "${PYTHON}" "${PREFIX}/verify_completion.py" --log-dir "${SMP_G1_ROOT}/logs/${run}" --ckpt-dir "${SMP_G1_ROOT}/ckpt/${run}" --out "${SMP_G1_ROOT}/logs/${run}/COMPLETION.md"
check_source
mark_stage complete
printf '00054=PASS\nrun=%s\ncompletion=%s\n' "${run}" "${SMP_G1_ROOT}/logs/${run}/COMPLETION.md" > "${EXECUTION}/COMPLETED.txt"
