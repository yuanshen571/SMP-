"""Server-B identity, shared-path and locked-stack checks; no task changes."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import pwd
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path("/home/yang/SMP+")
ARCHIVE_ROOT = Path("/home/yang/rollout")


def check_identity(repo):
  repo = Path(repo).resolve()
  assert os.geteuid() != 0 and pwd.getpwuid(os.geteuid()).pw_name == "yang", (
    "run as non-root yang on server B"
  )
  assert platform.node().split(".")[0] == "zxq", (
    "this task is for server B (zxq), not server A"
  )
  assert platform.system() == "Linux" and platform.machine() == "x86_64"
  assert repo == PROJECT_ROOT / "repo", (
    f"expected shared server B repo: {PROJECT_ROOT}/repo"
  )
  assert ARCHIVE_ROOT.is_dir(), f"missing archive parent: {ARCHIVE_ROOT}"
  return repo


def check_environment(repo):
  import torch

  repo = check_identity(repo)
  assert Path(sys.prefix).resolve() == PROJECT_ROOT / "venv", (
    "use the one shared uv venv"
  )
  assert sys.version_info[:3] == (3, 13, 18), sys.version
  import smp

  assert Path(smp.__file__).resolve() == repo / "src/smp/__init__.py", (
    "editable smp must point to the shared repo"
  )
  expected = {
    "mjlab": "1.6.0",
    "mujoco": "3.11.0",
    "mujoco-warp": "3.11.0",
    "rsl-rl-lib": "5.4.2",
    "tensorboard": "2.20.0",
    "warp-lang": "1.17.0",
  }
  versions = {
    name: importlib.metadata.version(name)
    for name in [*expected, "torch", "numpy", "triton"]
  }
  for name, value in expected.items():
    assert versions[name] == value, (name, versions[name], value)
  assert versions["torch"].split("+")[0] == "2.10.0"
  assert torch.cuda.is_available() and torch.version.cuda == "12.8"
  gpu = torch.cuda.get_device_name(0)
  assert "RTX 3090" in gpu, f"expected RTX 3090 on server B: {gpu}"
  assert torch.cuda.get_device_capability(0) == (8, 6)
  vram = torch.cuda.get_device_properties(0).total_memory
  assert 22 * 2**30 <= vram <= 25 * 2**30
  # Actual CUDA allocation and kernel execution, not just a driver string check.
  sample = torch.arange(32, device="cuda:0", dtype=torch.float32)
  assert torch.isfinite(sample.square()).all()
  torch.cuda.synchronize()
  uv = PROJECT_ROOT / "tools/uv"
  assert uv.is_file() and os.access(uv, os.X_OK)
  uv_version = subprocess.check_output([str(uv), "--version"], text=True).strip()
  assert uv_version.split()[1] == "0.13.7"
  subprocess.run([str(uv), "pip", "check", "--python", sys.executable], check=True)
  drivers = subprocess.check_output(
    [
      "nvidia-smi",
      "--query-gpu=name,driver_version,memory.total",
      "--format=csv,noheader",
    ],
    text=True,
  ).strip()
  return dict(
    server="B",
    hostname=platform.node(),
    user=pwd.getpwuid(os.geteuid()).pw_name,
    project_root=str(PROJECT_ROOT),
    shared_env=str(Path(sys.prefix)),
    archive_mount=str(ARCHIVE_ROOT),
    archive_free_bytes=shutil.disk_usage(ARCHIVE_ROOT).free,
    versions=versions,
    python=sys.version,
    gpu=gpu,
    vram_bytes=vram,
    cuda=torch.version.cuda,
    uv=uv_version,
    nvidia_smi=drivers,
    compile_threads=os.environ.get("TORCHINDUCTOR_COMPILE_THREADS"),
  )
