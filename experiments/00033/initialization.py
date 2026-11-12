"""Load the pinned legacy seed-2027 0002 initialization without RNG draws."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

CONTROL = json.loads(
  Path(__file__).with_name("EXPERIMENT.json").read_text(encoding="utf-8")
)["control"]


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
    # The original 0002 payload predates experiment/initialization_seed fields.
    # Its exact archived file and tensor hashes establish identity instead.
    if "initialization_seed" in info:
      assert info["initialization_seed"] == 2027
    if "experiment" in info:
      assert info["experiment"] == "0002"
    assert info.get("env_state", {}).get("common_step_counter") == 0


def validate_control(path, control_info):
  path, control_info = Path(path).resolve(), Path(control_info).resolve()
  info = json.loads(control_info.read_text(encoding="utf-8"))
  assert info["experiment"] == "0002" and info["seed"] == 2027
  assert info["completed_iterations"] == info["max_iterations"] == 10000
  assert info["num_envs"] == 1024 and info["num_steps_per_env"] == 24
  assert info["total_samples"] == 245760000
  assert path == Path(info["ckpt_dir"]).resolve() / "initial_untrained.pt"
  digest = sha256(path)
  assert digest == CONTROL["initial_checkpoint_sha256"], (
    "not the archived 2027 initial file"
  )
  index = json.loads((path.parent / "index.json").read_text(encoding="utf-8"))
  initial_entries = [row for row in index if row["file"] == "initial_untrained.pt"]
  assert len(initial_entries) == 1
  assert initial_entries[0]["sha256"] == digest
  assert initial_entries[0]["completed_iterations"] == 0
  assert initial_entries[0]["samples_on_save"] == 0
  exits = dict(
    line.split("=", 1)
    for line in (control_info.parent / "exit_code.txt").read_text().splitlines()
    if "=" in line
  )
  assert exits.get("python_exit_code") == exits.get("tee_exit_code") == "0"
  payload = torch.load(path, map_location="cpu", weights_only=False)
  assert_untrained(payload, require_provenance=True)
  assert payload["infos"]["run_id"] == info["run_id"]
  models_digest = initial_models_digest(payload)
  assert models_digest == CONTROL["initial_models_sha256"]
  # Return derived metadata without changing the legacy source files.
  return dict(
    info,
    initial_models_sha256=models_digest,
    initialization_seed=2027,
    initialization_mode="archived_legacy_0002",
  )


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
