"""Test paired loading, RNG neutrality and rejection of wrong/used sources."""

import copy
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
from initialization import (
  assert_untrained,
  initial_models_digest,
  load_common_initial,
  sha256,
  validate_control,
)


class Algorithm:
  def __init__(self):
    self.actor = torch.nn.Module()
    self.actor.mlp = torch.nn.Linear(3, 2)
    self.actor.register_buffer("dummy", torch.zeros(1))
    self.actor.obs_normalizer = torch.nn.Module()
    self.actor.obs_normalizer.register_buffer("count", torch.tensor(0.0))
    self.critic = copy.deepcopy(self.actor)
    self.optimizer = torch.optim.Adam(
      list(self.actor.parameters()) + list(self.critic.parameters()), lr=0.001
    )

  def save(self):
    return {
      "actor_state_dict": self.actor.state_dict(),
      "critic_state_dict": self.critic.state_dict(),
      "optimizer_state_dict": self.optimizer.state_dict(),
    }

  def load(self, payload, cfg, strict):
    assert cfg == {
      "actor": True,
      "critic": True,
      "optimizer": False,
      "iteration": False,
    }
    self.actor.load_state_dict(payload["actor_state_dict"], strict=strict)
    self.critic.load_state_dict(payload["critic_state_dict"], strict=strict)


with tempfile.TemporaryDirectory(prefix="seed2028-initial-protocol-") as tmp:
  root = Path(tmp)
  path, info_path = root / "initial_untrained.pt", root / "run_info.json"
  torch.manual_seed(2028)
  payload = Algorithm().save()
  payload["infos"] = {
    "stage": "initial_untrained",
    "experiment": "0005",
    "initialization_seed": 2028,
  }
  digest = initial_models_digest(payload)
  torch.manual_seed(2027)
  assert digest != initial_models_digest(Algorithm().save()), (
    "different seed must change this model"
  )
  # Different source optimizer hyperparameters must not overwrite the new optimizer.
  payload["optimizer_state_dict"]["param_groups"][0]["lr"] = 0.123
  torch.save(payload, path)
  info = dict(
    experiment="0005",
    seed=2028,
    initialization_seed=2028,
    initialization_mode="fresh_seed",
    completed_iterations=10000,
    max_iterations=10000,
    num_envs=1024,
    num_steps_per_env=24,
    total_samples=245760000,
    ckpt_dir=str(root),
    saved_initial_checkpoint_sha256=sha256(path),
    initial_models_sha256=digest,
  )
  info_path.write_text(json.dumps(info))
  (root / "exit_code.txt").write_text("python_exit_code=0\ntee_exit_code=0\n")
  runner = SimpleNamespace(alg=Algorithm(), current_learning_iteration=0)
  before = torch.get_rng_state().clone()
  assert validate_control(path, info_path)["seed"] == 2028
  assert load_common_initial(runner, path, info_path) == digest
  assert torch.equal(before, torch.get_rng_state()), "validation/loading consumed RNG"
  assert runner.alg.optimizer.param_groups[0]["lr"] == 0.001
  assert not runner.alg.optimizer.state and runner.current_learning_iteration == 0
  for name, value in [
    ("seed", 2027),
    ("seed", 2029),
    ("initialization_seed", 2027),
    ("initialization_seed", 2029),
    ("experiment", "00031"),
    ("experiment", "0006"),
    ("experiment", "00051"),
    ("completed_iterations", 2),
    ("initial_models_sha256", "wrong"),
    ("saved_initial_checkpoint_sha256", "wrong"),
  ]:
    bad = dict(info, **{name: value})
    info_path.write_text(json.dumps(bad))
    try:
      validate_control(path, info_path)
    except AssertionError:
      pass
    else:
      raise AssertionError(f"accepted wrong source metadata: {name}")
  info_path.write_text(json.dumps(info))
  for name in ("stage", "seed", "previous_seed", "normalizer", "optimizer", "nan"):
    bad = copy.deepcopy(payload)
    if name == "stage":
      bad["infos"]["stage"] = "trained"
    if name == "seed":
      bad["infos"]["initialization_seed"] = 2027
    if name == "previous_seed":
      bad["infos"]["initialization_seed"] = 2029
    if name == "normalizer":
      bad["actor_state_dict"]["obs_normalizer.count"] = torch.tensor(1.0)
    if name == "optimizer":
      bad["optimizer_state_dict"]["state"] = {0: {"step": torch.tensor(1.0)}}
    if name == "nan":
      bad["actor_state_dict"]["mlp.weight"][0, 0] = float("nan")
    try:
      assert_untrained(bad, require_provenance=True)
    except AssertionError:
      pass
    else:
      raise AssertionError(f"accepted used or invalid model: {name}")
print("SEED2028_INITIALIZATION_TESTS=PASS (CPU protocol; no simulator or training)")
