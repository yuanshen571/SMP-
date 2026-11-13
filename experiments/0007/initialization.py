"""Seed-2030 paired initialization; reads model tensors without consuming RNG."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch


def sha256(path):
  digest = hashlib.sha256()
  with Path(path).open("rb") as stream:
    for block in iter(lambda: stream.read(1 << 20), b""):
      digest.update(block)
  return digest.hexdigest()


def initial_models_digest(payload):
  digest = hashlib.sha256()
  for component in ("actor_state_dict", "critic_state_dict"):
    for name, tensor in sorted(payload[component].items()):
      value = tensor.detach().cpu().contiguous()
      digest.update(f"{component}:{name}:{value.dtype}:{tuple(value.shape)}".encode())
      digest.update(value.numpy().tobytes())
  return digest.hexdigest()


def assert_untrained(payload, *, require_provenance=False):
  assert not payload["optimizer_state_dict"]["state"], "optimizer has updates"
  for component in ("actor_state_dict", "critic_state_dict"):
    state = payload[component]
    assert state and all(bool(torch.isfinite(v).all()) for v in state.values())
    assert int(state["obs_normalizer.count"].item()) == 0, "trained normalizer"
  if require_provenance:
    info = payload.get("infos", {})
    assert info.get("stage") == "initial_untrained"
    assert info.get("initialization_seed") == 2030, "expected a seed-2030 source"
    assert info.get("experiment") == "0007", "expected the new 0007 baseline"


def validate_control(path, control_info):
  """Require the actual completed baseline, never an old 2027 or smoke model."""
  path, control_info = Path(path).resolve(), Path(control_info).resolve()
  info = json.loads(control_info.read_text(encoding="utf-8"))
  assert info["experiment"] == "0007" and info["seed"] == 2030
  assert info["initialization_seed"] == 2030
  assert info["initialization_mode"] == "fresh_seed"
  assert info["completed_iterations"] == info["max_iterations"] == 10000
  assert info["num_envs"] == 1024 and info["num_steps_per_env"] == 24
  assert info["total_samples"] == 245760000
  assert path == Path(info["ckpt_dir"]).resolve() / "initial_untrained.pt"
  assert path.name == "initial_untrained.pt"
  assert sha256(path) == info["saved_initial_checkpoint_sha256"]
  exits = dict(
    line.split("=", 1)
    for line in (control_info.parent / "exit_code.txt").read_text().splitlines()
    if "=" in line
  )
  assert exits.get("python_exit_code") == exits.get("tee_exit_code") == "0"
  payload = torch.load(path, map_location="cpu", weights_only=False)
  assert_untrained(payload, require_provenance=True)
  assert initial_models_digest(payload) == info["initial_models_sha256"]
  return info


def load_common_initial(runner, path, control_info):
  validate_control(path, control_info)
  payload = torch.load(path, map_location="cpu", weights_only=False)
  runner.alg.load(
    payload,
    {"actor": True, "critic": True, "optimizer": False, "iteration": False},
    strict=True,
  )
  assert runner.current_learning_iteration == 0
  assert not runner.alg.optimizer.state
  assert initial_models_digest(runner.alg.save()) == initial_models_digest(payload)
  return initial_models_digest(payload)
