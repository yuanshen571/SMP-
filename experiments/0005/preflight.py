"""Read-only shared-environment and actual 00031 baseline preflight."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import importlib.util
import inspect
import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
  h = hashlib.sha256()
  with Path(path).open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
      h.update(block)
  return h.hexdigest()


def check_dependencies():
  candidate = ROOT.parent / "tools/uv"
  uv = str(candidate) if candidate.is_file() else shutil.which("uv")
  if uv:
    command = [uv, "pip", "check", "--python", sys.executable]
  elif importlib.util.find_spec("pip") is not None:
    command = [sys.executable, "-m", "pip", "check"]
  else:
    raise RuntimeError("need existing uv or pip; do not install an environment")
  subprocess.run(command, check=True)
  return command


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--baseline-info", type=Path, required=True)
  p.add_argument("--initial-checkpoint", type=Path, required=True)
  p.add_argument("--out", type=Path, required=True)
  p.add_argument("--expected", type=Path)
  args = p.parse_args()
  baseline = json.loads(args.baseline_info.read_text(encoding="utf-8"))
  if baseline["experiment"] != "00031" or baseline["completed_iterations"] != 10000:
    raise ValueError("select actual completed 00031 run_info.json")
  if sha256(args.initial_checkpoint) != baseline["initial_checkpoint_sha256"]:
    raise ValueError("different common untrained checkpoint")
  state = torch.load(args.initial_checkpoint, map_location="cpu", weights_only=False)
  if (
    args.initial_checkpoint.name != "initial_untrained.pt"
    or state.get("infos", {}).get("stage") != "initial_untrained"
    or state["optimizer_state_dict"]["state"]
  ):
    raise ValueError("initial model must be preserved untrained source")
  for name in ["actor_state_dict", "critic_state_dict"]:
    if int(state[name]["obs_normalizer.count"].item()) != 0:
      raise ValueError("common initial model has nonzero observation count")
  spec = json.loads(Path(__file__).with_name("EXPERIMENT.json").read_text())
  for file, key in [
    (ROOT / "uv.lock", "uv_lock_sha256"),
    (ROOT / "datasets/pretrain_ckpt/pretrained_lafan_run.pt", "prior_sha256"),
  ]:
    if sha256(file) != spec[key] or baseline[key] != spec[key]:
      raise ValueError(f"baseline resource changed: {file}")
  protected = json.loads(Path(__file__).with_name("BASELINE_FILES.json").read_text())
  for name, digest in protected.items():
    if sha256(ROOT / name) != digest:
      raise ValueError(f"protected baseline file changed: {name}")
  if Path(sys.prefix).resolve() != (ROOT.parent / "venv").resolve():
    raise ValueError("reuse the one existing smp-g1/venv")
  smp = importlib.util.find_spec("smp")
  if smp is None or Path(smp.origin).resolve() != ROOT / "src/smp/__init__.py":
    raise ValueError("editable smp must point at shared repo/src")
  versions = {
    name: metadata.version(name)
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
  if (
    versions["mjlab"] != "1.6.0"
    or versions["mujoco"] != "3.11.0"
    or not versions["torch"].startswith("2.10.")
  ):
    raise ValueError("SDK differs from 00031; stop without reinstalling")
  if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required")
  from mjlab.envs import ManagerBasedRlEnv

  source = inspect.getsource(ManagerBasedRlEnv.step)
  points = [
    source.find(token)
    for token in (
      "self.reward_manager.compute",
      "self._reset_idx",
      "self.sim.forward()",
      "self.observation_manager.compute",
    )
  ]
  if any(i < 0 for i in points) or points != sorted(points):
    raise ValueError("SDK step ordering incompatible with real-state collection hooks")
  check = check_dependencies()
  result = dict(
    status="PASS",
    versions=versions,
    gpu=torch.cuda.get_device_name(0),
    baseline_info=str(args.baseline_info.resolve()),
    baseline_sha256=sha256(args.baseline_info),
    common_initial_sha256=sha256(args.initial_checkpoint),
    sdk_step_sha256=hashlib.sha256(source.encode()).hexdigest(),
    dependency_check=check,
  )
  if args.expected:
    expected = json.loads(args.expected.read_text())
    for key in (
      "versions",
      "baseline_sha256",
      "common_initial_sha256",
      "sdk_step_sha256",
    ):
      if result[key] != expected[key]:
        raise AssertionError(f"preflight changed: {key}")
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
  print("MIXED_PREFLIGHT=PASS")


if __name__ == "__main__":
  main()
