#!/usr/bin/env bash
# TensorBoard for experiment 0001 (read-only, bound to loopback).
#
# Usage: experiments/0001/launch_tensorboard.sh [port]
# Access from the local machine:
#   ssh -p 31207 -N -L 6009:127.0.0.1:6009 root@120.209.70.195
# then open http://127.0.0.1:6009
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="${REPO_ROOT}/../env.sh"
TB_BIN="${UV_PROJECT_ENVIRONMENT:-/root/gpufree-data/smp-g1/venv}/bin/tensorboard"
PORT="${1:-6009}"
SESSION="${TB_SESSION:-tb_0001_${PORT}}"
LOG="${REPO_ROOT}/logs/_tensorboard_${PORT}.log"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 2
fi

mkdir -p "${REPO_ROOT}/logs"
INNER="cd ${REPO_ROOT} && ${TB_BIN} --logdir logs --port ${PORT} \
  --host 127.0.0.1 --reload_interval 10 2>&1 | tee ${LOG}"
tmux new-session -d -s "${SESSION}" "${INNER}"
echo "session=${SESSION}"
echo "url=http://127.0.0.1:${PORT}"
echo "log=${LOG}"
