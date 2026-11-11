#!/usr/bin/env bash
# Reuse the one existing server environment; never create or install another.
SMP_0006_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SMP_0006_ROOT="$(dirname "${SMP_0006_REPO}")"
if [[ ! -f "${SMP_0006_ROOT}/env.sh" || ! -x "${SMP_0006_ROOT}/venv/bin/python" ]]; then
  printf 'existing shared env.sh and venv required\n' >&2
  return 2 2>/dev/null || exit 2
fi
source "${SMP_0006_ROOT}/env.sh"
export UV_PROJECT_ENVIRONMENT="${SMP_0006_ROOT}/venv"
export PATH="${UV_PROJECT_ENVIRONMENT}/bin:${PATH}"
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8 WANDB_MODE=disabled
export SMP_0006_REPO SMP_0006_ROOT
