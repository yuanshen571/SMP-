"""Check exact baseline source, the new experiment identity and shared SDK."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parent


def check_files(*, quiet=False):
  spec = json.loads((EXP / "EXPERIMENT.json").read_text())
  expected = json.loads((EXP / "BASELINE_FILES.json").read_text())
  for name, digest in expected.items():
    actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    assert actual == digest, f"baseline file changed: {name}"
  tracked = subprocess.check_output(["git", "ls-files", "src"], cwd=ROOT, text=True)
  assert set(tracked.splitlines()) == {p for p in expected if p.startswith("src/")}
  assert spec["experiment"] == EXP.name and spec["seed"] == 2026
  assert spec["num_envs"] == 1024 and spec["num_steps_per_env"] == 24
  assert spec["iterations"] == 10000 and spec["samples"] == 245760000
  assert spec["physical_filter_enabled"] is False
  assert spec["mixed_cache_enabled"] is spec["clustering_enabled"] is False
  if not quiet:
    print(
      f"BASELINE_SOURCE_CHECK=PASS ({len(expected)} files, {spec['repeats_experiment']}, seed 2026)"
    )
  return spec


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--environment", action="store_true")
  parser.add_argument("--out", type=Path)
  args = parser.parse_args()
  spec = check_files()
  result = {
    "experiment": spec["experiment"],
    "seed": 2026,
    "baseline_commit": spec["baseline_commit"],
  }
  if args.environment:
    import torch

    assert Path(sys.prefix).resolve() == (ROOT.parent / "venv").resolve(), (
      "reuse shared venv"
    )
    package = importlib.util.find_spec("smp")
    assert (
      package is not None
      and Path(package.origin).resolve() == ROOT / "src/smp/__init__.py"
    ), "editable smp must point to this shared repo"
    versions = {
      name: importlib.metadata.version(name)
      for name in (
        "torch",
        "numpy",
        "mujoco",
        "mjlab",
        "mujoco-warp",
        "rsl-rl-lib",
        "tensorboard",
      )
    }
    assert versions["mjlab"] == "1.6.0" and versions["mujoco"] == "3.11.0"
    assert versions["mujoco-warp"] == "3.11.0" and versions["rsl-rl-lib"] == "5.4.2"
    assert versions["torch"].startswith("2.10.")
    assert torch.cuda.is_available(), "server CUDA required"
    gpu = torch.cuda.get_device_name(0)
    assert "L40" in gpu, f"expected the agreed L40, found {gpu}"
    uv = ROOT.parent / "tools/uv"
    executable = str(uv) if uv.is_file() else shutil.which("uv")
    if executable:
      subprocess.run(
        [executable, "pip", "check", "--python", sys.executable], check=True
      )
    else:
      subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
    result.update(
      versions=versions,
      gpu=gpu,
      python=sys.version,
      shared_env=str(Path(sys.prefix).resolve()),
    )
    print("SHARED_ENVIRONMENT_CHECK=PASS")
  if args.out:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
  main()
