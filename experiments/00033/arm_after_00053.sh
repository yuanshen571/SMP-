#!/usr/bin/env bash
# Copy outside repo, then: bash arm_after_00053.sh <00053_pipeline_dir> <00033_full_SHA>
set -euo pipefail
PIPELINE="${1:?provide exact running 00053_pipeline directory}"
REVISION="${2:?provide full immutable 00033 commit SHA}"
[[ "${REVISION}" =~ ^[0-9a-f]{40}$ ]] || exit 2
SMP_G1_ROOT="${SMP_G1_ROOT:-/root/gpufree-data/smp-g1}"
TOOLS="${SMP_G1_ROOT}/tools/handoff_00053_to_00033_${REVISION:0:12}"
mkdir -p "${TOOLS}"
fetch_file() {
  local name="$1" temporary
  temporary="$(mktemp "${TOOLS}/download_XXXXXX")"
  if ! curl --fail --location --retry 2 --connect-timeout 20 --max-time 120 \
    "https://raw.githubusercontent.com/yuanshen571/SMP-/${REVISION}/experiments/00033/${name}" -o "${temporary}"; then
    rm -f "${temporary}"
    return 2
  fi
  if [[ -e "${TOOLS}/${name}" ]]; then
    cmp -s "${temporary}" "${TOOLS}/${name}" || { rm -f "${temporary}"; printf 'existing handoff file differs; preserve it and stop\n' >&2; return 2; }
    rm -f "${temporary}"
  else
    mv "${temporary}" "${TOOLS}/${name}"
  fi
}
fetch_file chain_after_00053.py
fetch_file EXPERIMENT.json
# This is a CPU/stdlib identity check, while 00053 continues undisturbed.
python3 - "${TOOLS}/EXPERIMENT.json" "${PIPELINE}" "${SMP_G1_ROOT}" <<'PY'
import hashlib, json, re, sys
from pathlib import Path
spec=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
root, pipeline=Path(sys.argv[3]).resolve(),Path(sys.argv[2]).resolve()
assert pipeline.parent==root/'logs' and re.fullmatch(r'00053_pipeline_[A-Za-z0-9_]+',pipeline.name)
assert (pipeline/'revision.txt').is_file()
assert spec['experiment']=='00033' and spec['seed']==2027
control=spec['control']
p=Path(control['initial_checkpoint'])
assert p.is_file() and Path(control['run_info']).is_file(), '2027 common initial resources missing'
h=hashlib.sha256()
with p.open('rb') as f:
    for part in iter(lambda:f.read(1<<20),b''): h.update(part)
assert h.hexdigest()==control['initial_checkpoint_sha256'], 'wrong 2027 untrained checkpoint'
print('2027_INITIAL_FILE_IDENTITY=PASS')
PY
SESSION="g1_00053_to_00033_$(basename "${PIPELINE}")"
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  printf 'handoff session already exists: %s\n' "${SESSION}" >&2
  exit 2
fi
printf -v INNER '%q ' python3 "${TOOLS}/chain_after_00053.py" --root "${SMP_G1_ROOT}" --pipeline "${PIPELINE}" --target-revision "${REVISION}"
printf -v LOG '%q' "${TOOLS}/watcher_console.log"
tmux new-session -d -s "${SESSION}" "exec ${INNER} >> ${LOG} 2>&1"
sleep 1
if ! tmux has-session -t "${SESSION}" 2>/dev/null; then
  cat "${TOOLS}/watcher_console.log"
  printf 'handoff did not remain active; inspect log, do not restart training\n' >&2
  exit 2
fi
printf 'HANDOFF_SESSION=%s\nHANDOFF_LOG=%s\nHANDOFF_STATE=%s/logs/handoff_00053_to_00033_%s/state.json\n' \
  "${SESSION}" "${TOOLS}/watcher_console.log" "${SMP_G1_ROOT}" "$(basename "${PIPELINE}")"
