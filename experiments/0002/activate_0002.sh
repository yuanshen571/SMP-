#!/usr/bin/env bash
# Source this file from each server process. Do not modify the 0001 env.sh.
SMP_0002_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SMP_0002_ROOT="$(dirname "${SMP_0002_REPO}")"
if [[ ! -f "${SMP_0002_ROOT}/env.sh" ]]; then
  printf 'missing shared environment file: %s/env.sh\n' "${SMP_0002_ROOT}" >&2
  return 2 2>/dev/null || exit 2
fi
source "${SMP_0002_ROOT}/env.sh"
# Isolate the editable smp install; dependency versions stay pinned by uv.lock.
export UV_PROJECT_ENVIRONMENT="${SMP_0002_VENV:-${SMP_0002_ROOT}/venv-0002}"
export PATH="${UV_PROJECT_ENVIRONMENT}/bin:${PATH}"
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export WANDB_MODE=disabled
export SMP_0002_REPO SMP_0002_ROOT
