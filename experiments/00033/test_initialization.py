"""Validate legacy 0002 identity, untrained state, strict loading and RNG neutrality."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import initialization as module
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
    self.actor.obs_normalizer = torch.nn.Module()
    self.actor.obs_normalizer.register_buffer("count", torch.tensor(0.0))
    self.critic = copy.deepcopy(self.actor)
    self.optimizer = torch.optim.Adam(
      list(self.actor.parameters()) + list(self.critic.parameters()), lr=0.001
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


class InitialTests(unittest.TestCase):
  def test_pinned_legacy_source_and_rng_neutrality(self):
    with tempfile.TemporaryDirectory() as temp:
      root = Path(temp)
      checkpoint, meta = root / "initial_untrained.pt", root / "run_info.json"
      torch.manual_seed(2027)
      payload = Algorithm().save()
      payload["infos"] = dict(
        stage="initial_untrained",
        run_id="0002_fixture",
        env_state=dict(common_step_counter=0),
      )
      payload["optimizer_state_dict"]["param_groups"][0]["lr"] = 0.123
      torch.save(payload, checkpoint)
      spec = dict(
        initial_checkpoint_sha256=sha256(checkpoint),
        initial_models_sha256=initial_models_digest(payload),
      )
      info = dict(
        experiment="0002",
        seed=2027,
        completed_iterations=10000,
        max_iterations=10000,
        num_envs=1024,
        num_steps_per_env=24,
        total_samples=245760000,
        ckpt_dir=str(root),
        run_id="0002_fixture",
      )
      meta.write_text(json.dumps(info))
      index = [
        dict(
          file="initial_untrained.pt",
          sha256=sha256(checkpoint),
          completed_iterations=0,
          samples_on_save=0,
        )
      ]
      (root / "index.json").write_text(json.dumps(index))
      (root / "exit_code.txt").write_text("python_exit_code=0\ntee_exit_code=0\n")
      with patch.object(module, "CONTROL", spec):
        runner = SimpleNamespace(alg=Algorithm(), current_learning_iteration=0)
        rng = torch.get_rng_state().clone()
        self.assertEqual(
          load_common_initial(runner, checkpoint, meta), spec["initial_models_sha256"]
        )
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        self.assertEqual(runner.alg.optimizer.param_groups[0]["lr"], 0.001)
        self.assertFalse(runner.alg.optimizer.state)
        for key, value in (
          ("seed", 2028),
          ("seed", 2029),
          ("experiment", "0005"),
          ("completed_iterations", 2),
        ):
          meta.write_text(json.dumps(dict(info, **{key: value})))
          with self.assertRaises(AssertionError):
            validate_control(checkpoint, meta)
        meta.write_text(json.dumps(info))
        with patch.object(
          module, "CONTROL", dict(spec, initial_checkpoint_sha256="wrong")
        ):
          with self.assertRaises(AssertionError):
            validate_control(checkpoint, meta)
        with patch.object(module, "CONTROL", dict(spec, initial_models_sha256="wrong")):
          with self.assertRaises(AssertionError):
            validate_control(checkpoint, meta)
        index[0]["completed_iterations"] = 1
        (root / "index.json").write_text(json.dumps(index))
        with self.assertRaises(AssertionError):
          validate_control(checkpoint, meta)
      for name in ("optimizer", "normalizer", "nan", "stage", "seed", "counter"):
        bad = copy.deepcopy(payload)
        if name == "optimizer":
          bad["optimizer_state_dict"]["state"] = {0: dict(step=torch.tensor(1.0))}
        if name == "normalizer":
          bad["actor_state_dict"]["obs_normalizer.count"] = torch.tensor(1.0)
        if name == "nan":
          bad["actor_state_dict"]["mlp.weight"][0, 0] = float("nan")
        if name == "stage":
          bad["infos"]["stage"] = "trained"
        if name == "seed":
          bad["infos"]["initialization_seed"] = 2028
        if name == "counter":
          bad["infos"]["env_state"]["common_step_counter"] = 1
        with self.assertRaises(AssertionError):
          assert_untrained(bad, require_provenance=True)


if __name__ == "__main__":
  unittest.main(verbosity=2)
