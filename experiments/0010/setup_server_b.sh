#!/usr/bin/env bash
# One common uv environment for all subsequent experiments on server B.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SMP_B_ROOT='/home/yang/SMP+'
[[ "${REPO}" == "${SMP_B_ROOT}/repo" && "$(id -un)" == yang && "$(id -u)" != 0 && "$(hostname -s)" == zxq ]] || { printf 'server B yang@zxq and /home/yang/SMP+/repo required\n' >&2; exit 2; }
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || exit 2
cd "${REPO}"
[[ "$(git branch --show-current)" == 0010 && -z "$(git status --porcelain)" ]] || { printf 'clean NEW server-B 0010 branch required\n' >&2; exit 2; }
for tool in git flock nvidia-smi sha256sum df; do command -v "${tool}" >/dev/null || { printf 'missing prerequisite: %s\n' "${tool}" >&2; exit 2; }; done
command -v cc >/dev/null || { printf 'C compiler required for JIT; request administrator installation\n' >&2; exit 2; }
command -v c++ >/dev/null || { printf 'C++ compiler required for JIT; request administrator installation\n' >&2; exit 2; }
exec 9>"${SMP_B_ROOT}/.0003_00031_queue.lock"
flock -n 9 || { printf 'another server B SMP workflow is active\n' >&2; exit 2; }
mkdir -p "${SMP_B_ROOT}/tools" "${SMP_B_ROOT}/setup" "${SMP_B_ROOT}/logs" "${SMP_B_ROOT}/ckpt" /home/yang/rollout
STAMP="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
SETUP="$(mktemp -d "${SMP_B_ROOT}/logs/server_b_setup_${STAMP}_XXXXXX")"
on_exit() { local rc=$?; printf 'setup_exit_code=%s\n' "${rc}" > "${SETUP}/exit_code.txt"; printf 'SERVER_B_SETUP_EXIT_CODE=%s\nSETUP_DIRECTORY=%s\n' "${rc}" "${SETUP}"; }
trap on_exit EXIT
exec > >(tee "${SETUP}/console.log") 2>&1
[[ "$(sha256sum uv.lock | cut -d ' ' -f 1)" == ce041c7774b1d4abf42e868fa3b4af2973954cec61d35c3b4e58a7e27464d9ce ]] || exit 2
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
nvidia-smi --query-gpu=name --format=csv,noheader | grep -q 'RTX 3090' || { printf 'wrong GPU\n' >&2; exit 2; }
FREE="$(df --output=avail -B1 "${SMP_B_ROOT}" | tail -n 1 | tr -d ' ')"
[[ "${FREE}" -ge 80000000000 ]] || { printf 'reserve 80 GB before first environment installation\n' >&2; exit 2; }
if [[ -e "${SMP_B_ROOT}/env.sh" ]]; then
  cmp -s experiments/0010/server_b_env.sh "${SMP_B_ROOT}/env.sh" || { printf 'existing env.sh differs; preserve it and report before changing\n' >&2; exit 2; }
else
  cp experiments/0010/server_b_env.sh "${SMP_B_ROOT}/env.sh"
fi
source "${SMP_B_ROOT}/env.sh"
mkdir -p "${UV_CACHE_DIR}" "${UV_PYTHON_INSTALL_DIR}" "${HF_HOME}" "${TORCH_HOME}" "${XDG_CACHE_HOME}" "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}" "${PIP_CACHE_DIR}" "${WARP_CACHE_PATH}" "${UV_PYTHON_BIN_DIR}"
UV="${SMP_B_ROOT}/tools/uv"
if [[ ! -e "${UV}" ]]; then
  INSTALLER="${SETUP}/uv-0.13.7-installer.sh"
  if command -v curl >/dev/null; then
    curl --fail --location --silent --show-error --retry 3 --connect-timeout 20 https://astral.sh/uv/0.13.7/install.sh -o "${INSTALLER}"
  elif command -v wget >/dev/null; then
    wget -O "${INSTALLER}" https://astral.sh/uv/0.13.7/install.sh
  else
    printf 'curl or wget is required; no system package installation attempted\n' >&2; exit 2
  fi
  sha256sum "${INSTALLER}" > "${SETUP}/uv_installer.sha256"
  env UV_UNMANAGED_INSTALL="${SMP_B_ROOT}/tools" sh "${INSTALLER}"
fi
[[ -x "${UV}" && "$("${UV}" --version | awk '{print $2}')" == 0.13.7 ]] || { printf 'pinned uv 0.13.7 required\n' >&2; exit 2; }
if [[ ! -f "${SMP_B_ROOT}/setup/shared_environment.json" ]]; then
  # Create/recover only this common environment; never write into system Python.
  "${UV}" python install 3.13.18
  "${UV}" sync --frozen --no-dev --python 3.13.18
fi
# Re-source now that the wheel library directories exist.
source "${SMP_B_ROOT}/env.sh"
PYTHON="${SMP_B_ROOT}/venv/bin/python"
"${PYTHON}" experiments/0010/contract.py --environment --out "${SETUP}/environment.json"
"${UV}" pip freeze --python "${PYTHON}" > "${SETUP}/freeze.txt"
"${PYTHON}" - "${SMP_B_ROOT}" "${SETUP}" <<'PYMARK'
import hashlib,json,sys
from pathlib import Path
root,out=map(Path,sys.argv[1:])
record=json.loads((out/'environment.json').read_text())
record.update(schema='smp-server-b-shared-env-v1',python_required='3.13.18',
 uv_lock_sha256=hashlib.sha256((root/'repo/uv.lock').read_bytes()).hexdigest(),
 setup_directory=str(out))
marker=root/'setup/shared_environment.json'
if marker.exists():
 previous=json.loads(marker.read_text())
 assert previous['uv_lock_sha256']==record['uv_lock_sha256']
 assert previous['versions']==record['versions']
else:
 temporary=marker.with_suffix('.json.partial')
 temporary.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
 temporary.replace(marker)
print('SHARED_SERVER_B_ENVIRONMENT=PASS')
PYMARK
printf 'source /home/yang/SMP+/env.sh\n' > "${SETUP}/ACTIVATE.txt"
printf 'Environment ready. Training was not started by this setup script.\n'
