#!/usr/bin/env python3
"""Dependency-free checks of the actual formula/config, not a GPU/prior test.

Compile just the reward function ASTs with a scalar guidance stub. This lets
us exercise both paths on the preparation machine without installing torch,
mjlab or CUDA. Runtime metrics/environment tests remain mandatory on server.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import subprocess
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parent
BASELINE = "ff8b7f3a90d9b87fa230b5b9777b75ee51cb2d5d"


def only_function(path, name, namespace):
  tree = ast.parse(path.read_text(encoding="utf-8"))
  fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
  module = ast.parse("from __future__ import annotations\n")
  module.body.append(fn)
  exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
  return namespace[name]


def main():
  calls = []
  guidance_value = [0.25]

  def guidance(env, **kw):
    calls.append((env, kw))
    return guidance_value[0]

  core = only_function(
    ROOT / "src/smp/rl/rewards.py",
    "task_smp_additive",
    {"smp_guidance_reward": guidance},
  )
  factory = only_function(
    EXP / "passive_metrics.py",
    "build_logged_task_smp_additive",
    {"smp_guidance_reward": guidance},
  )
  logged = factory()
  task_value = [0.5]
  task_calls = []

  def task_fn(env, **kwargs):
    task_calls.append((env, kwargs))
    return task_value[0]

  terms = ((task_fn, 1.0, {"command_name": "location"}),)
  records = []
  observer = SimpleNamespace(record_step=lambda *args: records.append(args))
  # Include zeros and asymmetric values to detect an accidental product/unweighted sum.
  for t, s in (
    (0.0, 0.0),
    (0.0, 1.0),
    (1.0, 0.0),
    (0.5, 0.25),
    (0.01, 0.8),
    (1.0, 1.0),
  ):
    task_value[0], guidance_value[0] = t, s
    expected = 0.5 * t + 0.5 * s
    for fn, metrics in ((core, None), (logged, None), (logged, observer)):
      logged.metrics = metrics
      before_g, before_t = len(calls), len(task_calls)
      value = fn("env", terms, fixed_timesteps=(8, 15, 22), ws=4)
      assert math.isclose(value, expected, abs_tol=1e-12), (t, s, value)
      assert len(calls) == before_g + 1 and len(task_calls) == before_t + 1
      assert calls[-1] == ("env", {"fixed_timesteps": (8, 15, 22), "ws": 4})
      if metrics is not None:
        assert records[-1] == ("env", t, s, expected, terms)
  print("PASS core/logged/no-observer formulas and one task/guidance call per step")

  cfg_path = ROOT / "src/smp/rl/tasks/location/location_env_cfg.py"
  tree = ast.parse(cfg_path.read_text(encoding="utf-8"))
  assigns = [
    n
    for n in ast.walk(tree)
    if isinstance(n, ast.Assign)
    and any(
      isinstance(target, ast.Subscript) and ast.unparse(target.value) == "cfg.rewards"
      for target in n.targets
    )
  ]
  assert len(assigns) == 1
  assert ast.literal_eval(assigns[0].targets[0].slice) == "task_smp_additive"
  config = eval(
    compile(ast.Expression(assigns[0].value), str(cfg_path), "eval"),
    {
      "RewardTermCfg": lambda **kw: kw,
      "task_smp_additive": core,
      "mdp": SimpleNamespace(location_position=task_fn),
    },
  )
  assert config["func"] is core and config["weight"] == 1.0
  params = config["params"]
  assert params["ws"] == 4
  assert params["task_reward_weight"] == params["smp_reward_weight"] == 0.5
  assert params["task_terms"] == (
    (task_fn, 1.0, {"command_name": "location", "pos_err_scale": 0.25}),
  )
  for t, s in ((0.3, 0.7), (0.0, 1.0)):
    task_value[0], guidance_value[0] = t, s
    assert math.isclose(config["func"]("env", **params), 0.5 * t + 0.5 * s)
  print("PASS Location weights: task/SMP=.5/.5; position/velocity/facing=1/0/0")

  manifest = json.loads((EXP / "EXPERIMENT.json").read_text(encoding="utf-8"))
  for name, key in (
    ("uv.lock", "uv_lock_sha256"),
    ("datasets/pretrain_ckpt/pretrained_lafan_run.pt", "prior_sha256"),
  ):
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == manifest[key], name
  # All existing reward functions, including the guidance normalizer and
  # multiplicative rewards of other tasks, must match the 0001 source exactly.
  original = subprocess.check_output(
    ["git", "show", f"{BASELINE}:src/smp/rl/rewards.py"], cwd=ROOT, text=True
  )
  new = ast.parse((ROOT / "src/smp/rl/rewards.py").read_text(encoding="utf-8"))
  original_tree = ast.parse(original)
  new.body = [
    n
    for n in new.body
    if not (isinstance(n, ast.FunctionDef) and n.name == "task_smp_additive")
  ]
  assert ast.dump(new) == ast.dump(original_tree)
  # Check that the rest of Location's configuration has no other semantic diff.
  original_cfg = ast.parse(
    subprocess.check_output(
      ["git", "show", f"{BASELINE}:src/smp/rl/tasks/location/location_env_cfg.py"],
      cwd=ROOT,
      text=True,
    )
  )
  for node in ast.walk(tree):
    if isinstance(node, ast.Name) and node.id == "task_smp_additive":
      node.id = "task_smp_product"
    if isinstance(node, ast.alias) and node.name == "task_smp_additive":
      node.name = "task_smp_product"
  for cfg_tree in (tree, original_cfg):
    cfg_tree.body = [n for n in cfg_tree.body if not isinstance(n, ast.Expr)]
    fn = next(n for n in cfg_tree.body if isinstance(n, ast.FunctionDef))
    fn.body = [
      n
      for n in fn.body
      if not (
        isinstance(n, ast.Assign)
        and any(
          isinstance(t, ast.Subscript) and ast.unparse(t.value) == "cfg.rewards"
          for t in n.targets
        )
      )
    ]
  assert ast.dump(tree) == ast.dump(original_cfg)
  changed = subprocess.check_output(
    ["git", "diff", "--name-only", BASELINE, "--", "src/"], cwd=ROOT, text=True
  ).splitlines()
  assert set(changed) <= {
    "src/smp/rl/rewards.py",
    "src/smp/rl/tasks/location/location_env_cfg.py",
    "src/smp/rl/events.py",
    "src/smp/rl/gsi_physics.py",
    "src/smp/rl/gsi_physics_runtime.py",
    "src/smp/rl/gsi_clusters.py",
  }, changed
  print(
    "PASS frozen prior/lock, unchanged rewards/PPO/other tasks, GSI changes isolated"
  )
  print("REWARD_FORMULA_CHECK=PASS (scalar stubs; no simulator or GPU was run)")


if __name__ == "__main__":
  main()
