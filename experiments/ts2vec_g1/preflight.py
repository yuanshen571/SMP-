"""Read-only validation in the existing shared venv; never installs packages."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import importlib.util
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from experiments.ts2vec_g1.common import (
  HERE,
  ROOT,
  config,
  git_commit,
  now,
  official,
  read_json,
  sha256,
  write_json,
)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--run", type=Path, required=True)
  parser.add_argument("--reference-00031", type=Path, required=True)
  args = parser.parse_args()
  run = args.run.resolve()
  cfg = config()
  expected_venv = ROOT.parent / "venv"
  if Path(sys.prefix).resolve() != expected_venv.resolve():
    raise RuntimeError("must reuse the one existing smp-g1/venv")
  smp_spec = importlib.util.find_spec("smp")
  if (
    smp_spec is None or Path(smp_spec.origin).resolve() != ROOT / "src/smp/__init__.py"
  ):
    raise RuntimeError("editable smp installation must point to the shared repo/src")
  if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for formal generation/training")
  versions = {
    name: metadata.version(name)
    for name in (
      "torch",
      "numpy",
      "scipy",
      "scikit-learn",
      "mjlab",
      "mujoco",
      "mujoco-warp",
      "threadpoolctl",
    )
  }
  if versions["mjlab"] != "1.6.0" or versions["mujoco"] != "3.11.0":
    raise RuntimeError(
      "00031 SDK versions differ; stop without installing/replacing packages"
    )
  if not versions["torch"].startswith("2.10."):
    raise RuntimeError("00031 torch major/minor differs")
  if sha256(ROOT / "uv.lock") != cfg["uv_lock_sha256"]:
    raise RuntimeError("shared uv.lock differs from 00031")
  if sha256(ROOT / cfg["prior_path"]) != cfg["prior_sha256"]:
    raise RuntimeError("frozen GSI prior differs from 00031")
  implementation = read_json(HERE / "IMPLEMENTATION.json")
  for name, digest in implementation["baseline_files"].items():
    if sha256(ROOT / name) != digest:
      raise RuntimeError(f"00031 sampling/filter/repair source changed: {name}")
  reference = read_json(args.reference_00031)
  if (
    reference.get("experiment") != "00031"
    or reference.get("completed_iterations") != 10000
  ):
    raise RuntimeError("select an actual completed 00031 run_info.json")
  if reference["gsi_physics_config"] != cfg["physical_filter"]:
    raise RuntimeError("reference physical conditions differ")
  subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
  official()
  # Small dependency/GPU check only; no TS2Vec or PPO learning update.
  t = torch.arange(64, device="cuda", dtype=torch.float32)
  if not bool(torch.isfinite(t.square()).all()):
    raise RuntimeError("GPU finite-value check failed")
  if not np.isfinite(t.cpu().numpy()).all():
    raise RuntimeError("GPU/CPU transfer check failed")
  run.mkdir(parents=True, exist_ok=True)
  write_json(
    run / "preflight.json",
    dict(
      status="PASS",
      created_at=now(),
      python=platform.python_version(),
      python_executable=sys.executable,
      venv=sys.prefix,
      versions=versions,
      gpu=torch.cuda.get_device_name(0),
      git_commit=git_commit(),
      config=cfg,
      reference_00031=str(args.reference_00031.resolve()),
      reference_sha256=sha256(args.reference_00031),
      prior_sha256=sha256(ROOT / cfg["prior_path"]),
      uv_lock_sha256=sha256(ROOT / "uv.lock"),
    ),
  )
  print("G1_TS2VEC_PREFLIGHT=PASS", flush=True)


if __name__ == "__main__":
  main()
