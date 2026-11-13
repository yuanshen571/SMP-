"""Run the actual CLI/main preflight with a real CPU checkpoint, before SDK imports."""

from __future__ import annotations

import argparse
import ast
import copy
import json
import tempfile
import unittest
from pathlib import Path

import initialization
import torch

HERE = Path(__file__).resolve().parent
SPEC = json.loads((HERE / "EXPERIMENT.json").read_text(encoding="utf-8"))
RUNNER = HERE / f"run_{SPEC['experiment']}.py"


def startup_main():
  source = ast.parse(RUNNER.read_text(encoding="utf-8"))
  parse = copy.deepcopy(
    next(
      n
      for n in source.body
      if isinstance(n, ast.FunctionDef) and n.name == "parse_args"
    )
  )
  main = copy.deepcopy(
    next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == "main")
  )
  stop = next(
    i
    for i, n in enumerate(main.body)
    if isinstance(n, ast.Assign)
    and any(isinstance(t, ast.Name) and t.id == "env_cfg" for t in n.targets)
  )
  main.body = main.body[:stop] + [
    ast.Return(
      value=ast.Constant(value="STARTUP_PREFLIGHT_REACHED_ENVIRONMENT_BOUNDARY")
    )
  ]
  module = ast.parse("from __future__ import annotations\n")
  module.body.extend([parse, main])
  namespace = dict(
    __file__=str(RUNNER),
    argparse=argparse,
    Path=Path,
    json=json,
    validate_control=initialization.validate_control,
  )
  if hasattr(initialization, "validate_startup_control"):
    namespace["validate_startup_control"] = initialization.validate_startup_control
  exec(compile(ast.fix_missing_locations(module), str(RUNNER), "exec"), namespace)
  return namespace["main"]


def fixture(root):
  expected = SPEC["control"]
  path, info_path = root / "initial_untrained.pt", root / "run_info.json"
  state = {
    "layer.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
    "obs_normalizer.count": torch.tensor(0.0),
  }
  payload = dict(
    actor_state_dict=copy.deepcopy(state),
    critic_state_dict=copy.deepcopy(state),
    optimizer_state_dict=dict(state={}, param_groups=[]),
    infos=dict(
      stage="initial_untrained",
      experiment=expected["experiment"],
      initialization_seed=expected["seed"],
    ),
  )
  torch.save(payload, path)
  info = dict(
    experiment=expected["experiment"],
    seed=expected["seed"],
    initialization_seed=expected["seed"],
    initialization_mode="fresh_seed",
    completed_iterations=10000,
    max_iterations=10000,
    num_envs=1024,
    num_steps_per_env=24,
    total_samples=245760000,
    ckpt_dir=str(root),
    saved_initial_checkpoint_sha256=initialization.sha256(path),
    initial_models_sha256=initialization.initial_models_digest(payload),
  )
  info_path.write_text(json.dumps(info), encoding="utf-8")
  (root / "exit_code.txt").write_text(
    "python_exit_code=0\ntee_exit_code=0\n", encoding="utf-8"
  )
  return path, info_path, info


def invoke(path, info_path):
  return startup_main()(
    [
      "--seed",
      str(SPEC["seed"]),
      "--initial-checkpoint",
      str(path),
      "--control-info",
      str(info_path),
    ]
  )


class StartupTests(unittest.TestCase):
  def test_real_main_accepts_runtime_fingerprint_without_static_pin(self):
    self.assertNotIn("initial_models_sha256", SPEC["control"])
    with tempfile.TemporaryDirectory() as temp:
      path, info_path, info = fixture(Path(temp))
      before = torch.get_rng_state().clone()
      result = invoke(path, info_path)
      self.assertEqual(result, "STARTUP_PREFLIGHT_REACHED_ENVIRONMENT_BOUNDARY")
      self.assertTrue(torch.equal(before, torch.get_rng_state()))

  def test_optional_static_pins_are_enforced_when_present(self):
    with tempfile.TemporaryDirectory() as temp:
      path, info_path, info = fixture(Path(temp))
      control = dict(
        SPEC["control"],
        initial_models_sha256=info["initial_models_sha256"],
        initial_checkpoint_sha256=info["saved_initial_checkpoint_sha256"],
      )
      initialization.validate_startup_control(path, info_path, control)
      for key, value in (
        ("initial_models_sha256", "wrong"),
        ("initial_models_sha256", None),
        ("initial_checkpoint_sha256", "wrong"),
        ("seed", SPEC["seed"] + 1),
        ("experiment", "wrong"),
      ):
        with self.assertRaises(AssertionError):
          initialization.validate_startup_control(
            path, info_path, dict(control, **{key: value})
          )

  def test_real_main_rejects_incomplete_or_wrong_runtime_identity(self):
    with tempfile.TemporaryDirectory() as temp:
      path, info_path, info = fixture(Path(temp))
      for key, value in (
        ("completed_iterations", 2),
        ("seed", SPEC["seed"] + 1),
        ("initial_models_sha256", "wrong"),
        ("saved_initial_checkpoint_sha256", "wrong"),
      ):
        info_path.write_text(json.dumps(dict(info, **{key: value})), encoding="utf-8")
        with self.assertRaises(AssertionError):
          invoke(path, info_path)
      info_path.write_text(json.dumps(info), encoding="utf-8")
      (Path(temp) / "exit_code.txt").write_text("python_exit_code=1\ntee_exit_code=0\n")
      with self.assertRaises(AssertionError):
        invoke(path, info_path)

  def test_real_main_rejects_modified_checkpoint(self):
    with tempfile.TemporaryDirectory() as temp:
      path, info_path, _ = fixture(Path(temp))
      with path.open("ab") as stream:
        stream.write(b"modified")
      with self.assertRaises(AssertionError):
        invoke(path, info_path)


if __name__ == "__main__":
  unittest.main(verbosity=2)
