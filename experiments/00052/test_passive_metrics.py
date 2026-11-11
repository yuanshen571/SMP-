"""Unit checks for the passive metrics layer (no simulator required).

Proves the two properties EXECUTION.md §7 demands from the logging path:

1. it consumes **no** random numbers (torch RNG state is untouched), and
2. the episode counters satisfy the §8 identities on synthetic data.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch

_spec = importlib.util.spec_from_file_location(
  "smp_experiment_00052_metrics", Path(__file__).with_name("passive_metrics.py")
)
assert _spec is not None and _spec.loader is not None
pm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pm)


class _Writer:
  def __init__(self) -> None:
    self.scalars: dict[str, list[tuple[float, int]]] = {}
    self.flushes = 0

  def add_scalar(self, tag: str, value: float, step: int) -> None:
    self.scalars.setdefault(tag, []).append((float(value), int(step)))

  def flush(self) -> None:
    self.flushes += 1


class _Data:
  def __init__(self, n: int) -> None:
    self.root_link_pos_w = torch.zeros(n, 3)
    self.root_link_lin_vel_w = torch.zeros(n, 3)


class _Term:
  def __init__(self, n: int) -> None:
    self.tar_pos_w = torch.zeros(n, 2)


def make_env(n: int = 16) -> SimpleNamespace:
  return SimpleNamespace(
    num_envs=n,
    device=torch.device("cpu"),
    step_dt=0.02,
    command_manager=SimpleNamespace(get_term=lambda name: _Term(n)),
    scene={"robot": SimpleNamespace(data=_Data(n))},
    reward_manager=SimpleNamespace(compute=lambda dt: torch.zeros(n)),
    reset_terminated=torch.zeros(n, dtype=torch.bool),
    reset_time_outs=torch.zeros(n, dtype=torch.bool),
    _reset_idx=lambda ids=None: None,
  )


def test_rng_neutral() -> None:
  env = make_env()
  env._smp_physics_totals = {"generated": 5120, "inserted": 4096, "repair_inserted": 0}
  metrics = pm.PassiveMetrics(env, steps_per_env=24)
  writer = _Writer()
  task_terms = (
    (lambda e, **kw: torch.zeros(e.num_envs), 1.0, {"command_name": "location"}),
  )

  torch.manual_seed(1234)
  control = torch.randn(8)

  torch.manual_seed(1234)
  task = torch.full((env.num_envs,), 0.5)
  smp = torch.full((env.num_envs,), 0.25)
  additive = 0.5 * task + 0.5 * smp
  for _ in range(5):
    metrics.record_step(env, task, smp, additive, task_terms)
    metrics.record_env_reward(additive * 0.02)
  metrics.on_reset_idx(env, None)
  metrics.on_iteration(writer, 0, 0.1, 0.05)
  for name, value in env._smp_physics_totals.items():
    assert writer.scalars[f"00052/GSI/{name}"] == [(float(value), 1)]
  assert torch.equal(torch.randn(8), control), "logging path consumed RNG"
  print(f"PASS rng-neutral (wrote {len(writer.scalars)} tags)")


def test_count_identities() -> None:
  n = 12
  env = make_env(n)
  metrics = pm.PassiveMetrics(env, steps_per_env=24)
  writer = _Writer()
  task_terms = (
    (lambda e, **kw: torch.zeros(e.num_envs), 1.0, {"command_name": "location"}),
  )
  zero = torch.zeros(n)
  metrics.record_step(env, zero, zero, zero, task_terms)
  metrics.record_env_reward(zero)

  # env 0..3 -> failure, 4..7 -> timeout only, 8..9 -> brand new (startup), 10..11: no end
  ids = torch.arange(10)
  env.reset_terminated = torch.zeros(n, dtype=torch.bool)
  env.reset_time_outs = torch.zeros(n, dtype=torch.bool)
  env.reset_terminated[:4] = True
  env.reset_time_outs[4:8] = True
  metrics.ep_startup[:] = False
  metrics.ep_startup[8:10] = True
  metrics.on_reset_idx(env, ids)
  metrics.on_iteration(writer, 0, 1.0, 1.0)

  s = {k: v[-1][0] for k, v in writer.scalars.items()}
  assert s["00052/Count/ended_episode"] == 10, s
  assert s["00052/Count/failure_episode"] == 4, s
  assert s["00052/Count/timeout_only_episode"] == 4, s
  assert s["00052/Count/startup_incomplete_episode"] == 2, s
  assert s["00052/Count/complete_episode"] == 8, s
  assert s["00052/Count/complete_failure_episode"] == 4, s
  assert s["00052/Count/complete_timeout_only_episode"] == 4, s
  assert s["00052/Reward/additive_raw_mean"] == 0.0
  assert s["00052/Progress/completed_iterations"] == 1
  assert s["00052/Progress/training_samples"] == 1 * 24 * n
  print("PASS count identities")


def test_additive_mean_identity() -> None:
  n = 4
  env = make_env(n)
  metrics = pm.PassiveMetrics(env, steps_per_env=24)
  writer = _Writer()
  task_terms = (
    (lambda e, **kw: torch.zeros(e.num_envs), 1.0, {"command_name": "location"}),
  )
  task = torch.tensor([1.0, 1.0, 0.0, 0.0])
  smp = torch.tensor([0.0, 1.0, 1.0, 0.0])
  metrics.record_step(env, task, smp, 0.5 * task + 0.5 * smp, task_terms)
  metrics.record_env_reward((0.5 * task + 0.5 * smp) * 0.02)
  metrics.on_iteration(writer, 0, 1.0, 1.0)
  s = {k: v[-1][0] for k, v in writer.scalars.items()}
  assert abs(s["00052/Reward/task_raw_mean"] - 0.5) < 1e-9
  assert abs(s["00052/Reward/smp_raw_mean"] - 0.5) < 1e-9
  assert (
    abs(s["00052/Reward/additive_raw_mean"] - 0.5) < 1e-9
  )  # weighted sum, not multiplication
  assert "00052/Episode/complete_task_return_raw_mean" not in s
  assert s["00052/Progress/training_samples"] == 24 * n
  print("PASS additive-mean identity and skip-empty-denominator")


def test_simultaneous_termination_counts_as_failure() -> None:
  n = 4
  env = make_env(n)
  metrics = pm.PassiveMetrics(env, steps_per_env=24)
  writer = _Writer()
  task_terms = (
    (lambda e, **kw: torch.zeros(e.num_envs), 1.0, {"command_name": "location"}),
  )
  zero = torch.zeros(n)
  metrics.record_step(env, zero, zero, zero, task_terms)
  metrics.record_env_reward(zero)
  env.reset_terminated = torch.zeros(n, dtype=torch.bool)
  env.reset_time_outs = torch.zeros(n, dtype=torch.bool)
  env.reset_terminated[0] = True
  env.reset_time_outs[0] = True  # both flags on the same step -> failure
  env.reset_time_outs[1] = True  # timeout only
  env.reset_terminated[2] = True  # failure only
  metrics.on_reset_idx(env, torch.arange(3))
  metrics.on_iteration(writer, 0, 1.0, 1.0)
  s = {k: v[-1][0] for k, v in writer.scalars.items()}
  assert s["00052/Count/ended_episode"] == 3, s
  assert s["00052/Count/failure_episode"] == 2, s
  assert s["00052/Count/timeout_only_episode"] == 1, s
  print("PASS simultaneous termination+timeout counted as failure")


def test_iteration_without_episode_end() -> None:
  n = 3
  env = make_env(n)
  metrics = pm.PassiveMetrics(env, steps_per_env=24)
  writer = _Writer()
  task_terms = (
    (lambda e, **kw: torch.zeros(e.num_envs), 1.0, {"command_name": "location"}),
  )
  zero = torch.zeros(n)
  for _ in range(2):
    metrics.record_step(env, zero, zero, zero, task_terms)
    metrics.record_env_reward(zero)
  metrics.on_iteration(writer, 0, 1.0, 1.0)
  s = {k: v[-1][0] for k, v in writer.scalars.items()}
  for name in (
    "Count/ended_episode",
    "Count/failure_episode",
    "Count/timeout_only_episode",
    "Count/complete_episode",
    "Count/startup_incomplete_episode",
  ):
    assert s[f"00052/{name}"] == 0.0, (name, s)
  assert not any(k.startswith("00052/Episode/") for k in s), s
  assert not any(k.startswith(("00052/Episode/", "00052/Failure/")) for k in s), s
  print("PASS empty-denominator iteration writes zero counts and skips means")


if __name__ == "__main__":
  test_rng_neutral()
  test_count_identities()
  test_additive_mean_identity()
  test_simultaneous_termination_counts_as_failure()
  test_iteration_without_episode_end()
  print("all passive-metrics unit checks passed")
