#!/usr/bin/env bash
# Bind the viewer to loopback. SSH-forward port 6010 from your local machine.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_ROOT}/experiments/0005/activate_0005.sh"
TB_BIN="${UV_PROJECT_ENVIRONMENT}/bin/tensorboard"
PORT="${1:-6010}"
[[ "${PORT}" =~ ^[0-9]+$ ]] && ((PORT >= 1024 && PORT <= 65535)) || exit 2
SESSION="${TB_SESSION:-tb_0005_${PORT}}"
LOG="${SMP_0005_ROOT}/logs/_tensorboard_0005_${PORT}.log"
[[ -x "${TB_BIN}" ]] || { printf 'missing TensorBoard executable\n' >&2; exit 2; }
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  printf 'tmux session already exists: %s\n' "${SESSION}" >&2
  exit 2
fi
mkdir -p "${SMP_0005_ROOT}/logs"
printf -v INNER '%q ' "${TB_BIN}" --logdir "${SMP_0005_ROOT}/logs" --port "${PORT}" --host 127.0.0.1 --reload_interval 10
printf -v LOG_QUOTED '%q' "${LOG}"
tmux new-session -d -s "${SESSION}" "${INNER} > ${LOG_QUOTED} 2>&1"
printf 'session=%s\nurl=http://127.0.0.1:%s\nlog=%s\n' "${SESSION}" "${PORT}" "${LOG}"
