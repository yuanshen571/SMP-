#!/usr/bin/env bash
# Single paired-rescue experiment, only after an explicitly selected 00095 run.
set -euo pipefail
[[ $# == 1 && -f "$1" ]] || { printf 'usage: bash experiments/00097/execute.sh /absolute/path/to/ORIGINAL_00096/run_info.json\n' >&2; exit 2; }
REPEAT_INFO="$(realpath "$1")"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO}"
[[ "$(git branch --show-current)" == "00097" && -z "$(git status --porcelain)" ]] || { printf 'clean delivery branch 00097 required\n' >&2; exit 2; }
SMP_G1_ROOT="$(dirname "${REPO}")"
exec 9>"${SMP_G1_ROOT}/.0003_00031_queue.lock"
flock -n 9 || { printf 'another SMP workflow holds the shared lock\n' >&2; exit 2; }
source experiments/00097/activate_00097.sh
PYTHON="${SMP_G1_ROOT}/venv/bin/python"
PREFIX=experiments/00097
CONTROL_INFO="$("${PYTHON}" "${PREFIX}/repeat_binding.py" --repeat-info "${REPEAT_INFO}" --print-control)"
COMMIT="$(git rev-parse HEAD)"
mkdir -p "${SMP_G1_ROOT}/logs" "${SMP_G1_ROOT}/ckpt"
STAMP="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
EXECUTION="$(mktemp -d "${SMP_G1_ROOT}/logs/00097_execution_${STAMP}_XXXXXX")"
TAG="$(basename "${EXECUTION}")"
STAGE=prepare
on_exit() {
  local rc=$?
  printf 'execution_exit_code=%s\nlast_stage=%s\n' "${rc}" "${STAGE}" > "${EXECUTION}/exit_code.txt"
  printf 'EXPERIMENT_00097_EXIT_CODE=%s\nEXECUTION_DIRECTORY=%s\n' "${rc}" "${EXECUTION}"
}
trap on_exit EXIT
GUARD="${SMP_G1_ROOT}/logs/.00097_registered"
if ! (set -o noclobber; printf 'execution=%s\ncommit=%s\ncontrol_info=%s\nrepeat_info=%s\n' "${EXECUTION}" "${COMMIT}" "${CONTROL_INFO}" "${REPEAT_INFO}" > "${GUARD}") 2>/dev/null; then
  printf '00097 already registered; preserve and inspect previous execution\n' >&2; exit 2
fi
check_source() {
  [[ "$(git branch --show-current)" == "00097" && "$(git rev-parse HEAD)" == "${COMMIT}" && -z "$(git status --porcelain)" ]] || { printf 'source changed; stop and preserve it\n' >&2; return 2; }
}
mark_stage() { STAGE="$1"; printf 'experiment=00097\nstage=%s\n' "${STAGE}" > "${EXECUTION}/progress.txt"; }
run_logged() {
  local folder="$1"; shift
  check_source
  [[ ! -e "${folder}" ]] || { printf 'refuse existing stage directory\n' >&2; return 2; }
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
mark_stage repeat_binding
run_logged "${EXECUTION}/repeat_binding" "${PYTHON}" "${PREFIX}/repeat_binding.py" --repeat-info "${REPEAT_INFO}" --out "${EXECUTION}/repeat_binding.json"
mark_stage source_archive_validation
run_logged "${EXECUTION}/source" "${PYTHON}" "${PREFIX}/validate_source.py" --control-info "${CONTROL_INFO}" --out "${EXECUTION}/source_binding.json"
# Verify the actual baseline model/TensorBoard completion using its own verifier.
CONTROL_LOG="$(dirname "${CONTROL_INFO}")"
CONTROL_CKPT="$("${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["ckpt_dir"])' "${CONTROL_INFO}")"
run_logged "${EXECUTION}/control_completion" "${PYTHON}" experiments/00095/verify_completion.py --iterations 20000 --log-dir "${CONTROL_LOG}" --ckpt-dir "${CONTROL_CKPT}" --out "${EXECUTION}/CONTROL_COMPLETION.md"
mark_stage tests
for test in test_variant test_repeat_binding test_reward_formula test_passive_metrics test_initialization test_search test_policy test_scoring test_physics test_pairing; do
  run_logged "${EXECUTION}/${test}" "${PYTHON}" "${PREFIX}/${test}.py"
done
mark_stage frozen_prior_replay
run_logged "${EXECUTION}/sampler_replay" "${PYTHON}" "${PREFIX}/accept_sampler.py" --control-info "${CONTROL_INFO}" --out "${EXECUTION}/sampler_replay.json"
mark_stage smoke
SMOKE="00097_smoke_seed2026_${TAG}"
run_logged "${SMP_G1_ROOT}/logs/${SMOKE}" "${PYTHON}" "${PREFIX}/run_00097.py" --num-envs 1024 --max-iterations 2 --stage smoke --seed 2026 --device cuda:0 --control-info "${CONTROL_INFO}" --repeat-info "${REPEAT_INFO}" --run "${SMOKE}" --log-root "${SMP_G1_ROOT}/logs" --ckpt-root "${SMP_G1_ROOT}/ckpt"
run_logged "${EXECUTION}/smoke_verify" "${PYTHON}" "${PREFIX}/verify_completion.py" --iterations 2 --log-dir "${SMP_G1_ROOT}/logs/${SMOKE}" --ckpt-dir "${SMP_G1_ROOT}/ckpt/${SMOKE}"
run_logged "${EXECUTION}/smoke_pairing" "${PYTHON}" "${PREFIX}/verify_pairing.py" --iterations 2 --log-dir "${SMP_G1_ROOT}/logs/${SMOKE}" --out "${EXECUTION}/smoke_pairing.json"
mark_stage formal_training
RUN="00097_location_seed2026_${TAG}"
[[ ! -e "${SMP_G1_ROOT}/ckpt/${RUN}" ]] || exit 2
printf 'formal_run=%s\nsmoke_run=%s\ncontrol_info=%s\ncommit=%s\n' "${RUN}" "${SMOKE}" "${CONTROL_INFO}" "${COMMIT}" > "${EXECUTION}/runs.txt"
run_logged "${SMP_G1_ROOT}/logs/${RUN}" "${PYTHON}" "${PREFIX}/run_00097.py" --num-envs 1024 --max-iterations 20000 --stage formal --seed 2026 --device cuda:0 --control-info "${CONTROL_INFO}" --repeat-info "${REPEAT_INFO}" --run "${RUN}" --log-root "${SMP_G1_ROOT}/logs" --ckpt-root "${SMP_G1_ROOT}/ckpt"
mark_stage formal_verification
run_logged "${EXECUTION}/formal_verify" "${PYTHON}" "${PREFIX}/verify_completion.py" --iterations 20000 --log-dir "${SMP_G1_ROOT}/logs/${RUN}" --ckpt-dir "${SMP_G1_ROOT}/ckpt/${RUN}" --out "${SMP_G1_ROOT}/logs/${RUN}/COMPLETION.md"
run_logged "${EXECUTION}/formal_pairing" "${PYTHON}" "${PREFIX}/verify_pairing.py" --iterations 20000 --log-dir "${SMP_G1_ROOT}/logs/${RUN}" --out "${EXECUTION}/pairing.json"
# Re-read the entire baseline archive after training; do not modify/delete it.
mark_stage source_archive_recheck
run_logged "${EXECUTION}/source_after" "${PYTHON}" "${PREFIX}/validate_source.py" --control-info "${CONTROL_INFO}" --out "${EXECUTION}/source_after.json"
check_source
cp "${SMP_G1_ROOT}/logs/${RUN}/COMPLETION.md" "${EXECUTION}/COMPLETION.md"
"${PYTHON}" - "${SMP_G1_ROOT}/logs/${RUN}/run_info.json" "${EXECUTION}" <<'PYRESULT'
import json,sys
from pathlib import Path
info_path=Path(sys.argv[1]).resolve()
info=json.loads(info_path.read_text(encoding="utf-8"))
assert info["completed_iterations"]==20000 and info["total_samples"]==491520000
result={"status":"PASS","experiment":info["experiment"],"seed":info["seed"],
        "run_info":str(info_path),"ckpt_dir":info["ckpt_dir"],
        "run_id":info["run_id"],"execution_directory":str(Path(sys.argv[2]).resolve())}
Path(sys.argv[2],"result.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
PYRESULT
mark_stage complete
printf '00097=PASS\n' > "${EXECUTION}/completed.txt"
