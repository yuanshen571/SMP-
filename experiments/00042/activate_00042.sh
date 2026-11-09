#!/usr/bin/env bash
# Source in the one existing server repo. Reuse the one existing venv.
SMP_00042_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SMP_00042_ROOT="$(dirname "${SMP_00042_REPO}")"
if [[ ! -f "${SMP_00042_ROOT}/env.sh" || ! -x "${SMP_00042_ROOT}/venv/bin/python" ]]; then
  printf 'existing shared env.sh and venv are required; do not create another environment\n' >&2
  return 2 2>/dev/null || exit 2
fi
source "${SMP_00042_ROOT}/env.sh"
export UV_PROJECT_ENVIRONMENT="${SMP_00042_ROOT}/venv"
export PATH="${UV_PROJECT_ENVIRONMENT}/bin:${PATH}"
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8 WANDB_MODE=disabled
export SMP_00042_REPO SMP_00042_ROOT
