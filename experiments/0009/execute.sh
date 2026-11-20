#!/usr/bin/env bash
# Single fresh baseline run; never invokes a paired/rescue queue.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO}"
[[ "$(git branch --show-current)" == 0009 && -z "$(git status --porcelain)" ]] || {
  printf 'start on clean branch 0009; preserve any existing changes\n' >&2
  exit 2
}
SMP_G1_ROOT="$(dirname "${REPO}")"
exec 9>"${SMP_G1_ROOT}/.0003_00031_queue.lock"
flock -n 9 || { printf 'another SMP workflow holds the shared lock\n' >&2; exit 2; }
source experiments/0009/activate_0009.sh
PYTHON="${SMP_G1_ROOT}/venv/bin/python"
COMMIT="$(git rev-parse HEAD)"
PREFIX=experiments/0009
mkdir -p "${SMP_G1_ROOT}/logs" "${SMP_G1_ROOT}/ckpt"
STAMP="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
EXECUTION="$(mktemp -d "${SMP_G1_ROOT}/logs/0009_execution_${STAMP}_XXXXXX")"
TAG="$(basename "${EXECUTION}")"
STAGE=prepare
on_exit() {
  local rc=$?
  printf 'execution_exit_code=%s\nlast_stage=%s\n' "${rc}" "${STAGE}" > "${EXECUTION}/exit_code.txt"
  printf 'EXPERIMENT_0009_EXIT_CODE=%s\nEXECUTION_DIRECTORY=%s\n' "${rc}" "${EXECUTION}"
}
trap on_exit EXIT
GUARD="${SMP_G1_ROOT}/logs/.0009_registered"
if ! (set -o noclobber; printf 'execution=%s\ncommit=%s\n' "${EXECUTION}" "${COMMIT}" > "${GUARD}") 2>/dev/null; then
  printf '0009 already registered; inspect existing execution, do not launch twice\n' >&2
  exit 2
fi
printf 'experiment=0009\ncommit=%s\nseed=2027\niterations=20000\nsamples=491520000\n' "${COMMIT}" > "${EXECUTION}/revisions.txt"
check_source() {
  [[ "$(git branch --show-current)" == 0009 && "$(git rev-parse HEAD)" == "${COMMIT}" && -z "$(git status --porcelain)" ]] || {
    printf 'source changed during execution; stop and preserve it\n' >&2
    return 2
  }
}
mark_stage() {
  STAGE="$1"
  printf 'experiment=0009\nstage=%s\n' "${STAGE}" > "${EXECUTION}/progress.txt"
}
run_logged() {
  local folder="$1"
  shift
  check_source
  [[ ! -e "${folder}" ]] || { printf 'refuse existing stage directory: %s\n' "${folder}" >&2; return 2; }
  mkdir -p "${folder}"
  set +e
  "$@" 2>&1 | tee "${folder}/console.log"
  local statuses=("${PIPESTATUS[@]}")
  set -e
  printf 'python_exit_code=%s\ntee_exit_code=%s\n' "${statuses[0]}" "${statuses[1]}" > "${folder}/exit_code.txt"
  if [[ "${statuses[0]}" != 0 ]]; then return "${statuses[0]}"; fi
  if [[ "${statuses[1]}" != 0 ]]; then return "${statuses[1]}"; fi
}
mark_stage preflight
run_logged "${EXECUTION}/preflight" "${PYTHON}" "${PREFIX}/contract.py" --environment --out "${EXECUTION}/environment.json"
mark_stage tests
for test in test_variant test_reward_formula test_passive_metrics test_initialization test_archive; do
  run_logged "${EXECUTION}/${test}" "${PYTHON}" "${PREFIX}/${test}.py"
done
mark_stage frozen_prior_replay
run_logged "${EXECUTION}/sampler_replay" "${PYTHON}" "${PREFIX}/accept_sampler.py" --out "${EXECUTION}/sampler_replay.json"
mark_stage environment_acceptance
run_logged "${EXECUTION}/accept_console" "${PYTHON}" "${PREFIX}/accept_0009.py" --num-envs 64 --steps 2400 --seed 2027 --out "${EXECUTION}/acceptance"
mark_stage smoke
SMOKE="0009_smoke_seed2027_${TAG}"
[[ ! -e "${SMP_G1_ROOT}/ckpt/${SMOKE}" ]] || exit 2
run_logged "${SMP_G1_ROOT}/logs/${SMOKE}" "${PYTHON}" "${PREFIX}/run_0009.py" --num-envs 1024 --max-iterations 2 --stage smoke --seed 2027 --device cuda:0 --run "${SMOKE}" --log-root "${SMP_G1_ROOT}/logs" --ckpt-root "${SMP_G1_ROOT}/ckpt"
run_logged "${EXECUTION}/smoke_verify" "${PYTHON}" "${PREFIX}/verify_completion.py" --iterations 2 --log-dir "${SMP_G1_ROOT}/logs/${SMOKE}" --ckpt-dir "${SMP_G1_ROOT}/ckpt/${SMOKE}"
mark_stage smoke_archive_verification
run_logged "${EXECUTION}/smoke_archive_verify" "${PYTHON}" "${PREFIX}/verify_archive.py" --iterations 2 --log-dir "${SMP_G1_ROOT}/logs/${SMOKE}" --out "${EXECUTION}/smoke_archive_check.json"
# Remove only this verified smoke run's large payload, keeping its audit/manifest.
"${PYTHON}" - "${SMP_G1_ROOT}/logs/${SMOKE}/run_info.json" "${EXECUTION}" <<'PYCLEAN'
import json,shutil,sys
from pathlib import Path
info=json.loads(Path(sys.argv[1]).read_text())
execution=Path(sys.argv[2])
audit=json.loads((execution/'smoke_archive_check.json').read_text())
archive=Path(info['gsi_archive_dir']).resolve()
root=Path('/gpufree-share/smp_gsi_paired_0009/smoke').resolve()
assert info['experiment']=='0009' and info['stage']=='smoke'
assert archive.parent==root and archive.name==info['run_id']
assert audit['status']=='PASS' and Path(audit['directory']).resolve()==archive
shutil.copyfile(archive/'manifest.json',execution/'smoke_archive_manifest.json')
shutil.rmtree(archive)
(execution/'smoke_archive_cleanup.txt').write_text('Only this verified smoke payload was deleted: '+str(archive)+'\n',encoding='utf-8')
PYCLEAN
mark_stage formal_preflight
run_logged "${EXECUTION}/formal_preflight" "${PYTHON}" "${PREFIX}/contract.py" --environment
mark_stage formal_training
RUN="0009_location_seed2027_${TAG}"
[[ ! -e "${SMP_G1_ROOT}/ckpt/${RUN}" ]] || exit 2
printf 'formal_run=%s\nsmoke_run=%s\n' "${RUN}" "${SMOKE}" > "${EXECUTION}/runs.txt"
run_logged "${SMP_G1_ROOT}/logs/${RUN}" "${PYTHON}" "${PREFIX}/run_0009.py" --num-envs 1024 --max-iterations 20000 --stage formal --seed 2027 --device cuda:0 --run "${RUN}" --log-root "${SMP_G1_ROOT}/logs" --ckpt-root "${SMP_G1_ROOT}/ckpt"
mark_stage formal_verification
run_logged "${EXECUTION}/formal_verify" "${PYTHON}" "${PREFIX}/verify_completion.py" --iterations 20000 --log-dir "${SMP_G1_ROOT}/logs/${RUN}" --ckpt-dir "${SMP_G1_ROOT}/ckpt/${RUN}" --out "${SMP_G1_ROOT}/logs/${RUN}/COMPLETION.md"
mark_stage formal_archive_verification
run_logged "${EXECUTION}/formal_archive_verify" "${PYTHON}" "${PREFIX}/verify_archive.py" --iterations 20000 --log-dir "${SMP_G1_ROOT}/logs/${RUN}" --out "${EXECUTION}/archive_check.json"
check_source
cp "${SMP_G1_ROOT}/logs/${RUN}/COMPLETION.md" "${EXECUTION}/COMPLETION.md"
mark_stage complete
printf '0009=PASS\n' > "${EXECUTION}/completed.txt"
