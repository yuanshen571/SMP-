#!/usr/bin/env bash
# Activate the single shared server-B environment, not an experiment-specific venv.
SMP_0010_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SMP_0010_ROOT="$(dirname "${SMP_0010_REPO}")"
if [[ "${SMP_0010_ROOT}" != /home/yang/SMP+ || ! -f "${SMP_0010_ROOT}/env.sh" || ! -x "${SMP_0010_ROOT}/venv/bin/python" || ! -f "${SMP_0010_ROOT}/setup/shared_environment.json" ]]; then
  printf 'server B shared setup required: run experiments/0010/setup_server_b.sh first\n' >&2
  return 2 2>/dev/null || exit 2
fi
source "${SMP_0010_ROOT}/env.sh"
export SMP_0010_REPO SMP_0010_ROOT
