"""Untrained-model audit must not alter the training RNG or accept used models."""

import copy

import torch
from initialization import assert_untrained, initial_models_digest

torch.manual_seed(2028)
actor = {"mlp.weight": torch.randn(3, 2), "obs_normalizer.count": torch.tensor(0.0)}
payload = dict(
  actor_state_dict=actor,
  critic_state_dict=copy.deepcopy(actor),
  optimizer_state_dict={"state": {}},
  infos=dict(stage="initial_untrained", experiment="0010", initialization_seed=2028),
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
