"""Untrained-model audit must not alter the training RNG or accept used models."""

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
)

torch.manual_seed(2027)
actor = {"mlp.weight": torch.randn(3, 2), "obs_normalizer.count": torch.tensor(0.0)}
payload = dict(
  actor_state_dict=actor,
  critic_state_dict=copy.deepcopy(actor),
  optimizer_state_dict={"state": {}},
  infos=dict(stage="initial_untrained", experiment="0009", initialization_seed=2027),
)
before = torch.get_rng_state().clone()
digest = initial_models_digest(payload)
assert_untrained(payload, require_provenance=True)
assert torch.equal(before, torch.get_rng_state())
assert digest == initial_models_digest(copy.deepcopy(payload))
for field in ("optimizer", "normalizer", "nan", "seed", "experiment"):
  bad = copy.deepcopy(payload)
  if field == "optimizer":
    bad["optimizer_state_dict"]["state"] = {0: {"step": 1}}
  if field == "normalizer":
    bad["actor_state_dict"]["obs_normalizer.count"].fill_(1)
  if field == "nan":
    bad["critic_state_dict"]["mlp.weight"][0, 0] = float("nan")
  if field == "seed":
    bad["infos"]["initialization_seed"] = 2026
  if field == "experiment":
    bad["infos"]["experiment"] = "0002"
  try:
    assert_untrained(bad, require_provenance=True)
  except AssertionError:
    pass
  else:
    raise AssertionError(f"accepted invalid initial payload: {field}")
print("INITIALIZATION_PROTOCOL=PASS (CPU, no policy training)")

# Exercise the actual models-only loader with a fresh optimizer and dynamic SHA.


class Algorithm:
  def __init__(self):
    self.actor = torch.nn.Module()
    self.actor.mlp = torch.nn.Linear(3, 2)
    self.actor.obs_normalizer = torch.nn.Module()
    self.actor.obs_normalizer.register_buffer("count", torch.tensor(0.0))
    self.critic = copy.deepcopy(self.actor)
    self.optimizer = torch.optim.Adam(
      [*self.actor.parameters(), *self.critic.parameters()], lr=0.001
    )

  def save(self):
    return dict(
      actor_state_dict=self.actor.state_dict(),
      critic_state_dict=self.critic.state_dict(),
      optimizer_state_dict=self.optimizer.state_dict(),
    )

  def load(self, payload, cfg, strict):
    assert cfg == dict(actor=True, critic=True, optimizer=False, iteration=False)
    self.actor.load_state_dict(payload["actor_state_dict"], strict=strict)
    self.critic.load_state_dict(payload["critic_state_dict"], strict=strict)


with tempfile.TemporaryDirectory() as tmp:
  root = Path(tmp)
  path = root / "initial_untrained.pt"
  meta = root / "run_info.json"
  original = Algorithm().save()
  original["infos"] = dict(
    stage="initial_untrained", experiment="0009", initialization_seed=2027
  )
  original["optimizer_state_dict"]["param_groups"][0]["lr"] = 0.123
  torch.save(original, path)
  info = dict(
    experiment="0009",
    seed=2027,
    initialization_seed=2027,
    initialization_mode="fresh_seed",
    completed_iterations=20000,
    max_iterations=20000,
    num_envs=1024,
    num_steps_per_env=24,
    total_samples=491520000,
    ckpt_dir=str(root),
    saved_initial_checkpoint_sha256=sha256(path),
    initial_models_sha256=initial_models_digest(original),
  )
  meta.write_text(json.dumps(info))
  (root / "exit_code.txt").write_text("python_exit_code=0\ntee_exit_code=0\n")
  runner = SimpleNamespace(alg=Algorithm(), current_learning_iteration=0)
  before = torch.get_rng_state().clone()
  assert load_common_initial(runner, path, meta) == info["initial_models_sha256"]
  assert torch.equal(before, torch.get_rng_state())
  assert runner.alg.optimizer.param_groups[0]["lr"] == 0.001
  assert runner.current_learning_iteration == 0 and not runner.alg.optimizer.state
print(
  "COMMON_INITIAL_LOADER=PASS: actual dynamic fingerprint, models-only, fresh optimizer and RNG unchanged"
)
