#!/usr/bin/env bash
# One shared repo/environment and one lock for all four fresh formal runs.
set -euo pipefail
[[ $# == 0 ]] || { printf 'usage: bash experiments/00093_00096/execute_queue.sh\n' >&2; exit 2; }
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO}"
BRANCH="$(git branch --show-current)"
[[ "${BRANCH}" =~ ^0009(3-00096|3|4|5|6)$ && -z "$(git status --porcelain)" ]] || { printf 'clean delivery branch required\n' >&2; exit 2; }
SMP_G1_ROOT="$(dirname "${REPO}")"
exec 9>"${SMP_G1_ROOT}/.0003_00031_queue.lock"
flock -n 9 || { printf 'another SMP workflow holds the shared lock\n' >&2; exit 2; }
source experiments/00093/activate_00093.sh
PYTHON="${SMP_G1_ROOT}/venv/bin/python"
COMMIT="$(git rev-parse HEAD)"
mkdir -p "${SMP_G1_ROOT}/logs" "${SMP_G1_ROOT}/ckpt"
STAMP="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
QUEUE="$(mktemp -d "${SMP_G1_ROOT}/logs/00093_00096_queue_${STAMP}_XXXXXX")"
STAGE=prepare
on_exit() {
  local rc=$?
  printf 'queue_exit_code=%s\nlast_stage=%s\n' "${rc}" "${STAGE}" > "${QUEUE}/exit_code.txt"
  printf 'QUEUE_EXIT_CODE=%s\nQUEUE_DIRECTORY=%s\n' "${rc}" "${QUEUE}"
}
trap on_exit EXIT
GUARD="${SMP_G1_ROOT}/logs/.00093_00096_queue_registered"
if ! (set -o noclobber; printf 'queue=%s\ncommit=%s\n' "${QUEUE}" "${COMMIT}" > "${GUARD}") 2>/dev/null; then
  printf 'queue already registered; inspect previous result, do not relaunch\n' >&2; exit 2
fi
export SMP_PAIRED_QUEUE_DIRECTORY="${QUEUE}"
PREFIX=experiments/00093_00096
check_source() {
  [[ "$(git branch --show-current)" == "${BRANCH}" && "$(git rev-parse HEAD)" == "${COMMIT}" && -z "$(git status --porcelain)" ]] || { printf 'source changed during queue\n' >&2; return 2; }
}
"${PYTHON}" "${PREFIX}/queue_state.py" preflight --queue "${QUEUE}" --root "${SMP_G1_ROOT}"
run_stage() {
  local exp="$1" control="${2:-}"
  STAGE="${exp}"
  check_source
  "${PYTHON}" "${PREFIX}/queue_state.py" running --queue "${QUEUE}" --experiment "${exp}"
  local args=()
  if [[ -n "${control}" ]]; then args+=("${control}"); fi
  set +e
  bash "experiments/${exp}/execute.sh" "${args[@]}" 2>&1 | tee "${QUEUE}/${exp}_console.log"
  local statuses=("${PIPESTATUS[@]}")
  set -e
  printf 'stage_exit_code=%s\ntee_exit_code=%s\n' "${statuses[0]}" "${statuses[1]}" > "${QUEUE}/${exp}_exit_code.txt"
  if [[ "${statuses[0]}" != 0 ]]; then return "${statuses[0]}"; fi
  if [[ "${statuses[1]}" != 0 ]]; then return "${statuses[1]}"; fi
  check_source
  "${PYTHON}" "${PREFIX}/queue_state.py" completed --queue "${QUEUE}" --root "${SMP_G1_ROOT}" --experiment "${exp}"
}
run_stage 00093
CONTROL_2028="$("${PYTHON}" "${PREFIX}/queue_state.py" control --queue "${QUEUE}" --experiment 00093)"
run_stage 00094 "${CONTROL_2028}"
run_stage 00095
CONTROL_2026="$("${PYTHON}" "${PREFIX}/queue_state.py" control --queue "${QUEUE}" --experiment 00095)"
run_stage 00096 "${CONTROL_2026}"
check_source
"${PYTHON}" "${PREFIX}/queue_state.py" finish --queue "${QUEUE}"
STAGE=complete
printf '00093=PASS\n00094=PASS\n00095=PASS\n00096=PASS\n' > "${QUEUE}/completed.txt"
