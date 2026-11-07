#!/usr/bin/env bash
# Source in the one existing server repo. Reuse the one existing venv.
SMP_0003_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SMP_0003_ROOT="$(dirname "${SMP_0003_REPO}")"
if [[ ! -f "${SMP_0003_ROOT}/env.sh" || ! -x "${SMP_0003_ROOT}/venv/bin/python" ]]; then
  printf 'existing shared env.sh and venv are required; do not create another environment\n' >&2
  return 2 2>/dev/null || exit 2
fi
source "${SMP_0003_ROOT}/env.sh"
export UV_PROJECT_ENVIRONMENT="${SMP_0003_ROOT}/venv"
export PATH="${UV_PROJECT_ENVIRONMENT}/bin:${PATH}"
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8 WANDB_MODE=disabled
export SMP_0003_REPO SMP_0003_ROOT
