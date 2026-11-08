"""Exercise the real initialization helpers with CPU model/optimizer states."""

import ast
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch

source = Path(__file__).with_name("run_0004.py")
functions = [
  node
  for node in ast.parse(source.read_text(encoding="utf-8")).body
  if isinstance(node, ast.FunctionDef)
  and node.name in ("initial_models_digest", "load_common_initial")
]
namespace = {"hashlib": hashlib, "torch": torch}
exec(
  compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace
)


def model():
  m = torch.nn.Module()
  m.mlp = torch.nn.Linear(3, 2)
  m.obs_normalizer = torch.nn.Module()
  m.obs_normalizer.register_buffer("count", torch.tensor(0.0))
  return m


class Algorithm:
  def __init__(self):
    self.actor = model()
    self.critic = model()
    self.optimizer = torch.optim.Adam(
      list(self.actor.parameters()) + list(self.critic.parameters()), lr=0.001
    )

  def load(self, payload, cfg, strict):
    assert cfg == {
      "actor": True,
      "critic": True,
      "optimizer": False,
      "iteration": False,
    }
    self.actor.load_state_dict(payload["actor_state_dict"], strict=strict)
    self.critic.load_state_dict(payload["critic_state_dict"], strict=strict)

  def save(self):
    return {
      "actor_state_dict": self.actor.state_dict(),
      "critic_state_dict": self.critic.state_dict(),
      "optimizer_state_dict": self.optimizer.state_dict(),
    }


with TemporaryDirectory() as temp:
  path = Path(temp) / "initial_untrained.pt"
  original = Algorithm()
  payload = original.save()
  payload["infos"] = {"stage": "initial_untrained"}
  payload["optimizer_state_dict"]["param_groups"][0]["lr"] = 0.123
  torch.save(payload, path)
  runner = SimpleNamespace(alg=Algorithm(), current_learning_iteration=0)
  digest = namespace["load_common_initial"](runner, path)
  assert digest == namespace["initial_models_digest"](payload)
  assert runner.alg.optimizer.param_groups[0]["lr"] == 0.001
  assert not runner.alg.optimizer.state and runner.current_learning_iteration == 0
  for bad in ["stage", "normalizer", "optimizer"]:
    corrupted = torch.load(path, weights_only=False)
    if bad == "stage":
      corrupted["infos"]["stage"] = "trained"
    if bad == "normalizer":
      corrupted["actor_state_dict"]["obs_normalizer.count"] = torch.tensor(1.0)
    if bad == "optimizer":
      corrupted["optimizer_state_dict"]["state"] = {0: {"step": torch.tensor(1.0)}}
    torch.save(corrupted, path)
    try:
      namespace["load_common_initial"](runner, path)
    except AssertionError:
      pass
    else:
      raise AssertionError(f"trained initial source was accepted: {bad}")
    torch.save(payload, path)
print("COMMON_INITIALIZATION_TESTS=PASS")
