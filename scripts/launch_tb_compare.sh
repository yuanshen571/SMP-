#!/usr/bin/env bash
# Overlay viewer: one TensorBoard showing every run with shared tag names.
#
#   bash scripts/launch_tb_compare.sh [port]
#
# Starts two tmux sessions:
#   tb_compare_mirror   - mirrors runs into /tmp/tb_compare (viewer-only copy)
#   tb_compare_<port>   - TensorBoard on 127.0.0.1:<port>
#
# Forward the port from your own machine, e.g.
#   ssh -N -L 6012:127.0.0.1:6012 <server>
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="$(dirname "${REPO_ROOT}")"

PYTHON="${UV_PROJECT_ENVIRONMENT:-${ROOT}/venv}/bin/python"
[[ -x "${PYTHON}" ]] || PYTHON="${ROOT}/venv-0002/bin/python"
[[ -x "${PYTHON}" ]] || { printf 'no python found (set UV_PROJECT_ENVIRONMENT)\n' >&2; exit 2; }
TB_BIN="$(dirname "${PYTHON}")/tensorboard"
[[ -x "${TB_BIN}" ]] || { printf 'missing tensorboard in %s\n' "$(dirname "${PYTHON}")" >&2; exit 2; }
"${PYTHON}" -c "import pkg_resources" 2>/dev/null || {
  printf 'tensorboard cannot import pkg_resources; run:\n  uv pip install --python %s "setuptools<81"\n' "${PYTHON}" >&2
  exit 2
}

PORT="${1:-6012}"
[[ "${PORT}" =~ ^[0-9]+$ ]] && ((PORT >= 1024 && PORT <= 65535)) || exit 2

MIRROR_DIR="${TB_COMPARE_OUT:-/tmp/tb_compare_live}"
if [[ -n "${TB_COMPARE_ROOTS:-}" ]]; then
  ROOTS="${TB_COMPARE_ROOTS}"
else
  # every experiment keeps its own checkout (repo, repo-0002, ...), so look for
  # run directories in the shared log root and in each sibling checkout.
  ROOTS="${ROOT}/logs"
  for candidate in "${ROOT}"/repo*/logs; do
    [[ -d "${candidate}" ]] && ROOTS="${ROOTS},${candidate}"
  done
fi
INTERVAL="${TB_COMPARE_INTERVAL:-20}"
LOG_DIR="${ROOT}/logs"
mkdir -p "${LOG_DIR}"

MIRROR_SESSION="tb_compare_mirror"
VIEW_SESSION="tb_compare_${PORT}"

for session in "${MIRROR_SESSION}" "${VIEW_SESSION}"; do
  if tmux has-session -t "${session}" 2>/dev/null; then
    tmux kill-session -t "${session}"
  fi
done

printf -v MIRROR_CMD '%q ' "${PYTHON}" "${REPO_ROOT}/scripts/tb_compare.py" \
  --roots "${ROOTS}" --out "${MIRROR_DIR}" --interval "${INTERVAL}"
tmux new-session -d -s "${MIRROR_SESSION}" "${MIRROR_CMD} > $(printf '%q' "${LOG_DIR}/_tb_compare_mirror.log") 2>&1"

printf -v VIEW_CMD '%q ' "${TB_BIN}" --logdir "${MIRROR_DIR}" --port "${PORT}" \
  --host 127.0.0.1 --reload_interval 5
tmux new-session -d -s "${VIEW_SESSION}" "${VIEW_CMD} > $(printf '%q' "${LOG_DIR}/_tb_compare_${PORT}.log") 2>&1"

printf 'mirror=%s (%s)\nport=%s\nurl=http://127.0.0.1:%s\n' \
  "${MIRROR_SESSION}" "${MIRROR_DIR}" "${PORT}" "${PORT}"
printf 'note: the first mirror pass can take ~30 s before the runs appear\n'
