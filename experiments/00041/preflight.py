"""Read-only shared-environment, baseline, and both frozen-module preflight."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
from smp.rl.gsi_clusters import (  # noqa: E402
  frozen_inference,
  load_module,
  sha256,
  validate_artifacts,
)


def check_dependencies(uv_path=None):
  candidate = Path(uv_path) if uv_path else ROOT.parent / "tools/uv"
  existing_uv = str(candidate) if candidate.is_file() else shutil.which("uv")
  if existing_uv:
    command = [existing_uv, "pip", "check", "--python", sys.executable]
  elif importlib.util.find_spec("pip") is not None:
    command = [sys.executable, "-m", "pip", "check"]
  else:
    raise RuntimeError(
      "need the existing uv tool or pip; do not install another environment"
    )
  subprocess.run(command, check=True)
  return command


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--baseline-info", type=Path, required=True)
  parser.add_argument("--initial-checkpoint", type=Path, required=True)
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument(
    "--expected",
    type=Path,
    help="original queue preflight for immutable paired artifacts",
  )
  args = parser.parse_args()
  baseline = json.loads(args.baseline_info.read_text(encoding="utf-8"))
  if baseline["experiment"] != "00031" or baseline["completed_iterations"] != 10000:
    raise ValueError("select the actual completed 00031 baseline metadata")
  if sha256(args.initial_checkpoint) != baseline["initial_checkpoint_sha256"]:
    raise ValueError("initial checkpoint differs from the one used by 00031")
  checkpoint = torch.load(
    args.initial_checkpoint, map_location="cpu", weights_only=False
  )
  if (
    checkpoint.get("infos", {}).get("stage") != "initial_untrained"
    or checkpoint["optimizer_state_dict"]["state"]
  ):
    raise ValueError("common checkpoint must be untrained with a fresh optimizer")
  if args.initial_checkpoint.name != "initial_untrained.pt":
    raise ValueError("use the preserved initial_untrained.pt source")
  spec = json.loads(
    Path(__file__).with_name("EXPERIMENT.json").read_text(encoding="utf-8")
  )
  for file, key in [
    (ROOT / "uv.lock", "uv_lock_sha256"),
    (ROOT / "datasets/pretrain_ckpt/pretrained_lafan_run.pt", "prior_sha256"),
  ]:
    if sha256(file) != spec[key] or baseline[key] != spec[key]:
      raise ValueError(f"baseline resource changed: {file}")
  with Path(__file__).with_name("BASELINE_FILES.json").open(encoding="utf-8") as stream:
    protected = json.load(stream)
  for file, digest in protected.items():
    if sha256(ROOT / file) != digest:
      raise ValueError(f"unchanged baseline source differs: {file}")
  if Path(sys.prefix).resolve() != (ROOT.parent / "venv").resolve():
    raise ValueError(
      "reuse the one existing smp-g1/venv; do not create another environment"
    )
  smp = importlib.util.find_spec("smp")
  if smp is None or Path(smp.origin).resolve() != ROOT / "src/smp/__init__.py":
    raise ValueError("editable smp must point at the one shared repo/src")
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
    raise ValueError("shared SDK differs from 00031; stop without reinstalling")
  if not torch.cuda.is_available():
    raise RuntimeError("formal experiments require CUDA")
  dependency_command = check_dependencies()
  modules = []
  code = spec["cluster_sampling"]["code_dir"]
  for dim in (64, 320):
    directory = f"/root/gpufree-data/dim{dim}"
    manifest = validate_artifacts(code, directory, dim)
    frozen, loaded_manifest = load_module(code, directory, dim, "cpu")
    probe = np.zeros((4, 10, 59), dtype=np.float32)
    probe[:, :, 2], probe[:, :, 3], probe[:, :, 8] = 0.8, 1, 1
    original = probe.copy()
    with frozen_inference("cpu"):
      embedding = frozen.encode(probe)
      predicted = frozen.predict(probe)
    if (
      embedding.shape != (4, dim)
      or not np.isfinite(embedding).all()
      or predicted.shape != (4,)
    ):
      raise AssertionError("saved module inference failed")
    if not np.array_equal(original, probe):
      raise AssertionError("frozen encoding must not mutate its input")
    manifest["encoder_state_sha256"] = loaded_manifest["encoder_state_sha256"]
    modules.append(manifest)
  if (
    modules[0]["dataset_sha256"] != modules[1]["dataset_sha256"]
    or modules[0]["normalizer_sha256"] != modules[1]["normalizer_sha256"]
  ):
    raise AssertionError(
      "64D/320D modules must share their dataset and 59-channel normalization"
    )
  result = dict(
    status="PASS",
    versions=versions,
    gpu=torch.cuda.get_device_name(0),
    baseline_info=str(args.baseline_info.resolve()),
    baseline_sha256=sha256(args.baseline_info),
    common_initial_sha256=sha256(args.initial_checkpoint),
    dependency_check=dependency_command,
    modules=modules,
  )
  if args.expected:
    expected = json.loads(args.expected.read_text(encoding="utf-8"))
    for key in ("versions", "baseline_sha256", "common_initial_sha256", "modules"):
      if result[key] != expected[key]:
        raise AssertionError(f"paired-experiment preflight changed: {key}")
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(
    json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
  )
  print("G1_CLUSTER_PREFLIGHT=PASS")


if __name__ == "__main__":
  main()
