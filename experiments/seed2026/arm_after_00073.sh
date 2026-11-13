#!/usr/bin/env bash
# Run an external copy: arm_after_00073.sh <0008_SHA> <00083_SHA> <formal_log_dir> <learner_pid> [finite_coordinator_pid ...]
set -euo pipefail
REVISION="${1:?provide full immutable 0008 SHA}"
RESCUE="${2:?provide full immutable 00083 SHA}"
RUN="${3:?provide exact active 00073 formal log directory}"
shift 3
[[ "${REVISION}" =~ ^[0-9a-f]{40}$ && "${RESCUE}" =~ ^[0-9a-f]{40}$ && "$#" -gt 0 ]] || exit 2
SMP_G1_ROOT="${SMP_G1_ROOT:-/root/gpufree-data/smp-g1}"
TOOLS="${SMP_G1_ROOT}/tools/handoff_00073_to_seed2026_${REVISION:0:12}"
mkdir -p "${TOOLS}"
TEMPORARY="$(mktemp "${TOOLS}/download_XXXXXX")"
trap 'rm -f "${TEMPORARY}"' EXIT
curl --fail --location --retry 2 --connect-timeout 20 --max-time 120 \
  "https://raw.githubusercontent.com/yuanshen571/SMP-/${REVISION}/experiments/seed2026/chain_after_00073.py" -o "${TEMPORARY}"
if [[ -e "${TOOLS}/chain_after_00073.py" ]]; then
  cmp -s "${TEMPORARY}" "${TOOLS}/chain_after_00073.py" || { printf 'existing watcher differs; preserve and stop\n' >&2; exit 2; }
else
  cp "${TEMPORARY}" "${TOOLS}/chain_after_00073.py"
fi
ARGS=(python3 "${TOOLS}/chain_after_00073.py" --root "${SMP_G1_ROOT}" --run "${RUN}" --target-revision "${REVISION}" --rescue-revision "${RESCUE}")
for pid in "$@"; do
  [[ "${pid}" =~ ^[0-9]+$ ]] || exit 2
  ARGS+=(--wait-pid "${pid}")
done
"${ARGS[@]}" --check-only
SESSION=g1_00073_to_seed2026
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  printf 'watcher session already exists: %s\n' "${SESSION}" >&2
  exit 2
fi
printf -v INNER '%q ' "${ARGS[@]}"
printf -v LOG '%q' "${TOOLS}/watcher_console.log"
tmux new-session -d -s "${SESSION}" "exec ${INNER} >> ${LOG} 2>&1"
sleep 1
if ! tmux has-session -t "${SESSION}" 2>/dev/null; then
  cat "${TOOLS}/watcher_console.log"
  printf 'watcher exited; inspect state, do not relaunch automatically\n' >&2
  exit 2
fi
printf 'HANDOFF_SESSION=%s\nHANDOFF_LOG=%s\nHANDOFF_STATE=%s/logs/handoff_00073_to_seed2026/state.json\n' \
  "${SESSION}" "${TOOLS}/watcher_console.log" "${SMP_G1_ROOT}"
