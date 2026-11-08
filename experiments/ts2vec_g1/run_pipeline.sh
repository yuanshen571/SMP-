#!/usr/bin/env bash
# One shared repo + venv; no dependency installation and no PPO entry point.
set -euo pipefail
TSG1_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TSG1_ROOT="$(dirname "${TSG1_REPO}")"
if [[ $# != 2 ]]; then
  printf 'Usage: %s /absolute/new/run_directory /absolute/00031/run_info.json\n' "$0" >&2
  exit 2
fi
TSG1_RUN="$1"
TSG1_REFERENCE="$2"
if [[ "${TSG1_RUN}" != /root/gpufree-share/* || ! -f "${TSG1_REFERENCE}" ]]; then
  printf 'Require a run directory under /root/gpufree-share and existing 00031 run_info.json\n' >&2
  exit 2
fi
if [[ ! -f "${TSG1_ROOT}/env.sh" || ! -x "${TSG1_ROOT}/venv/bin/python" ]]; then
  printf 'Existing shared env.sh and venv are required; do not create another environment\n' >&2
  exit 2
fi
source "${TSG1_ROOT}/env.sh"
export PATH="${TSG1_ROOT}/venv/bin:${PATH}"
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8 WANDB_MODE=disabled
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
cd "${TSG1_REPO}"
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  printf 'Tracked repository changes detected; stop without resetting files\n' >&2
  exit 2
fi
mkdir -p "${TSG1_RUN}"
exec 9>"${TSG1_ROOT}/.ts2vec_g1.lock"
flock -n 9 || { printf 'Another G1 TS2Vec pipeline holds the lock\n' >&2; exit 3; }
exec 8>"${TSG1_ROOT}/.0003_00031_queue.lock"
flock -n 8 || { printf 'The shared 0003/00031 queue is still running\n' >&2; exit 3; }
if [[ -e "${TSG1_RUN}/result.json" ]]; then
  printf 'Completed output already exists; verify it without rerunning the queue\n' >&2
  exit 3
fi
TSG1_PYTHON="${TSG1_ROOT}/venv/bin/python"
TSG1_STAGE=initialization
on_exit() {
  local code=$?
  printf 'PIPELINE_EXIT_CODE=%s\nLAST_STAGE=%s\nRUN_DIRECTORY=%s\n' "${code}" "${TSG1_STAGE}" "${TSG1_RUN}" | tee "${TSG1_RUN}/EXIT_CODE.txt"
  if [[ "${code}" != 0 ]]; then
    printf 'FAILED_STAGE=%s\n' "${TSG1_STAGE}" > "${TSG1_RUN}/FAILED_STAGE.txt"
  fi
}
trap on_exit EXIT
stage() {
  TSG1_STAGE="$1"
  shift
  printf 'STAGE=%s\n' "${TSG1_STAGE}" | tee "${TSG1_RUN}/progress.txt"
  set +e
  "$@" 2>&1 | tee -a "${TSG1_RUN}/console.log"
  local status=("${PIPESTATUS[@]}")
  set -e
  printf '%s PROCESS=%s TEE=%s\n' "${TSG1_STAGE}" "${status[0]}" "${status[1]}" >> "${TSG1_RUN}/stage_exit_codes.txt"
  if [[ "${status[0]}" != 0 ]]; then return "${status[0]}"; fi
  if [[ "${status[1]}" != 0 ]]; then return "${status[1]}"; fi
}
stage preflight "${TSG1_PYTHON}" -m experiments.ts2vec_g1.preflight --run "${TSG1_RUN}" --reference-00031 "${TSG1_REFERENCE}"
stage acceptance "${TSG1_PYTHON}" -m experiments.ts2vec_g1.test_module
stage dataset "${TSG1_PYTHON}" -m experiments.ts2vec_g1.generate --run "${TSG1_RUN}" --reference-00031 "${TSG1_REFERENCE}"
stage train64 "${TSG1_PYTHON}" -m experiments.ts2vec_g1.train --run "${TSG1_RUN}" --dimension 64
stage train320 "${TSG1_PYTHON}" -m experiments.ts2vec_g1.train --run "${TSG1_RUN}" --dimension 320
stage verification "${TSG1_PYTHON}" -m experiments.ts2vec_g1.verify --run "${TSG1_RUN}"
TSG1_STAGE=complete
printf 'STAGE=complete\n' > "${TSG1_RUN}/progress.txt"
