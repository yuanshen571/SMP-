#!/usr/bin/env bash
set -euo pipefail
TSG1_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ $# != 1 || ! -f "$1" ]]; then
  printf 'Usage: %s /absolute/existing/00031/run_info.json\n' "$0" >&2
  exit 2
fi
if [[ "$(git -C "${TSG1_REPO}" branch --show-current)" != 'TS2Vec+K-means' ]]; then
  printf 'Switch the shared clean repo to the TS2Vec+K-means branch first\n' >&2
  exit 2
fi
TSG1_PARENT='/root/gpufree-share/TS2Vec+K-means'
mkdir -p "${TSG1_PARENT}"
TSG1_STAMP="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
TSG1_RUN="$(mktemp -d "${TSG1_PARENT}/g1_seed2027_${TSG1_STAMP}_XXXXXX")"
TSG1_SESSION="g1_ts2vec_${TSG1_STAMP}_$(basename "${TSG1_RUN}" | tail -c 7)"
TSG1_REFERENCE="$(realpath "$1")"
printf -v TSG1_COMMAND '%q ' bash "${TSG1_REPO}/experiments/ts2vec_g1/run_pipeline.sh" "${TSG1_RUN}" "${TSG1_REFERENCE}"
printf -v TSG1_BOOT_LOG '%q' "${TSG1_RUN}/launcher.log"
tmux new-session -d -s "${TSG1_SESSION}" "${TSG1_COMMAND} >> ${TSG1_BOOT_LOG} 2>&1"
printf 'RUN_DIRECTORY=%s\nTMUX_SESSION=%s\n' "${TSG1_RUN}" "${TSG1_SESSION}"
printf 'RUN_DIRECTORY=%s\nTMUX_SESSION=%s\n' "${TSG1_RUN}" "${TSG1_SESSION}" > "${TSG1_RUN}/LAUNCH.txt"
