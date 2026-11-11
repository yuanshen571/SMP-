"""Passive TensorBoard metrics for experiment ``00062`` (``Smp-Location-G1``).

This module implements the observation layer required by
`experiments/00062/EXECUTION.md`.  It is *passive*:

* the reward formula is untouched - ``r_additive = 0.5*r_task + 0.5*r_smp`` is computed
  exactly as before (``smp_guidance_reward`` is still called once per step);
* every reported value is cached from a tensor the environment already
  produced in that step (no recomputation, no extra ``reset``/``forward``);
* no random numbers are consumed by the logging path;
* the episode-end mask is read at ``_reset_idx`` entry, i.e. before the
  automatic reset clears/shuffles the termination buffers.

TensorBoard names live under the ``00062/`` namespace and are written with
``global_step = completed_iterations`` (1-based, so the last point is 10000).
"""

from __future__ import annotations

import time
from typing import Any, Callable

import torch

from smp.rl.rewards import smp_guidance_reward

NS = "00062"


def _as_ids(
  env_ids: torch.Tensor | slice | None, num_envs: int, device: torch.device
) -> torch.Tensor:
  if env_ids is None:
    return torch.arange(num_envs, device=device)
  if isinstance(env_ids, slice):
    return torch.arange(num_envs, device=device)[env_ids]
  return env_ids.to(device=device, dtype=torch.long)


class PassiveMetrics:
  """Per-step cache + per-iteration aggregation of the §8 metrics."""

  def __init__(
    self,
    env: Any,
    *,
    steps_per_env: int,
    command_name: str = "location",
    asset_name: str = "robot",
  ) -> None:
    self.env = env
    self.num_envs = int(env.num_envs)
    self.steps_per_env = int(steps_per_env)
    self.device = torch.device(env.device)
    self.step_dt = float(env.step_dt)
    self.command_name = command_name
    self.asset_name = asset_name

    self.writer: Any | None = None
    self._active = False  # True once the first environment step was seen
    self._cmd_term: Any | None = None
    self._robot: Any | None = None
    self._last_flush = 0.0

    n = self.num_envs
    f64 = torch.float64
    self.ep_task = torch.zeros(n, dtype=f64, device=self.device)
    self.ep_reward = torch.zeros(n, dtype=f64, device=self.device)
    self.ep_steps = torch.zeros(n, dtype=torch.long, device=self.device)
    self.ep_startup = torch.ones(n, dtype=torch.bool, device=self.device)

    self.total_iterations = 0
    self.total_samples = 0
    self.elapsed = 0.0
    self._reset_iteration()

  # ------------------------------------------------------------------ state

  def _reset_iteration(self) -> None:
    z = torch.zeros((), dtype=torch.float64, device=self.device)
    self.s_task = z.clone()
    self.s_smp = z.clone()
    self.s_additive = z.clone()
    self.s_env_reward = z.clone()
    self.s_distance = z.clone()
    self.s_speed = z.clone()
    self.n_steps = torch.zeros((), dtype=torch.long, device=self.device)
    self.it_ended = torch.zeros((), dtype=torch.long, device=self.device)
    self.it_failure = torch.zeros((), dtype=torch.long, device=self.device)
    self.it_timeout = torch.zeros((), dtype=torch.long, device=self.device)
    self.it_startup = torch.zeros((), dtype=torch.long, device=self.device)
    self.it_complete = torch.zeros((), dtype=torch.long, device=self.device)
    self.it_complete_failure = torch.zeros((), dtype=torch.long, device=self.device)
    self.it_complete_timeout = torch.zeros((), dtype=torch.long, device=self.device)
    self.s_complete_len = torch.zeros((), dtype=torch.float64, device=self.device)
    self.s_complete_fail_dur = torch.zeros((), dtype=torch.float64, device=self.device)
    self.s_complete_task_ret = torch.zeros((), dtype=torch.float64, device=self.device)
    self.s_complete_train_ret = torch.zeros((), dtype=torch.float64, device=self.device)

  # ------------------------------------------------------------- step cache

  def record_step(
    self,
    env: Any,
    task: torch.Tensor,
    smp: torch.Tensor,
    additive: torch.Tensor,
    task_terms: tuple[Any, ...],
  ) -> None:
    """Cache ``r_task``, ``r_smp``, ``r_additive`` and the step's state metrics."""
    if self._cmd_term is None:
      name = self.command_name
      for _func, _weight, kwargs in task_terms:
        name = kwargs.get("command_name", name)
      self._cmd_term = env.command_manager.get_term(name)
    if self._robot is None:
      self._robot = env.scene[self.asset_name]

    self._active = True
    with torch.no_grad():
      task_d = task.detach().double()
      additive_d = additive.detach().double()
      root_xy = self._robot.data.root_link_pos_w[:, :2]
      distance = (self._cmd_term.tar_pos_w - root_xy).norm(dim=-1).double()
      speed = self._robot.data.root_link_lin_vel_w[:, :2].norm(dim=-1).double()

      self.s_task += task_d.sum()
      self.s_smp += smp.detach().double().sum()
      self.s_additive += additive_d.sum()
      self.s_distance += distance.sum()
      self.s_speed += speed.sum()
      self.n_steps += self.num_envs

      self.ep_task += task_d
      self.ep_steps += 1

  def record_env_reward(self, reward: torch.Tensor) -> None:
    """Cache the reward the environment actually returns to the trainer."""
    if not self._active:
      return
    with torch.no_grad():
      reward_d = reward.detach().double()
      self.s_env_reward += reward_d.sum()
      self.ep_reward += reward_d

  # ---------------------------------------------------------- episode ends

  def on_reset_idx(self, env: Any, env_ids: Any) -> None:
    """Called at ``_reset_idx`` entry, before the auto-reset mutates state."""
    ids = _as_ids(env_ids, self.num_envs, self.device)
    if not self._active:
      # Construction-time / wrapper reset: nothing has been rolled out yet.
      self.ep_task[ids] = 0.0
      self.ep_reward[ids] = 0.0
      self.ep_steps[ids] = 0
      return

    terminated = env.reset_terminated[ids].bool()
    time_outs = env.reset_time_outs[ids].bool()
    startup = self.ep_startup[ids]
    lengths = self.ep_steps[ids]

    failure = terminated
    timeout_only = (~terminated) & time_outs
    complete = ~startup

    self.it_ended += int(ids.numel())
    self.it_failure += int(failure.sum())
    self.it_timeout += int(timeout_only.sum())
    self.it_startup += int(startup.sum())

    if bool(complete.any()):
      self.it_complete += int(complete.sum())
      self.it_complete_failure += int((complete & failure).sum())
      self.it_complete_timeout += int((complete & timeout_only).sum())
      self.s_complete_len += lengths[complete].double().sum()
      fail_dur = lengths[complete & failure].double() * self.step_dt
      self.s_complete_fail_dur += fail_dur.sum()
      self.s_complete_task_ret += self.ep_task[ids][complete].sum()
      self.s_complete_train_ret += self.ep_reward[ids][complete].sum()

    self.ep_task[ids] = 0.0
    self.ep_reward[ids] = 0.0
    self.ep_steps[ids] = 0
    self.ep_startup[ids] = False

  # ------------------------------------------------------------- integration

  def attach(self, runner: Any) -> None:
    """Hook ``logger.log`` so metrics are written once per completed iteration."""
    logger = runner.logger
    original = logger.log

    def log(*args: Any, **kwargs: Any) -> Any:
      result = original(*args, **kwargs)
      it = kwargs.get("it", args[0] if args else None)
      collect = kwargs.get("collect_time", args[3] if len(args) > 3 else 0.0)
      learn = kwargs.get("learn_time", args[4] if len(args) > 4 else 0.0)
      if it is not None:
        self.on_iteration(logger.writer, int(it), float(collect), float(learn))
      return result

    logger.log = log  # type: ignore[method-assign]

  def on_iteration(
    self, writer: Any | None, it: int, collect_time: float, learn_time: float
  ) -> None:
    step = it + 1
    self.total_iterations = step
    self.total_samples = step * self.steps_per_env * self.num_envs
    iteration_time = collect_time + learn_time
    self.elapsed += iteration_time
    if writer is not None:
      self._write(writer, step, iteration_time)
      now = time.time()
      if now - self._last_flush > 10.0:
        writer.flush()
        self._last_flush = now
    self._reset_iteration()

  def _write(self, writer: Any, step: int, iteration_time: float) -> None:
    add = writer.add_scalar
    n = max(int(self.n_steps.item()), 1)
    n_f = float(n)

    add(f"{NS}/Reward/task_raw_mean", self.s_task.item() / n_f, step)
    add(f"{NS}/Reward/smp_raw_mean", self.s_smp.item() / n_f, step)
    add(f"{NS}/Reward/additive_raw_mean", self.s_additive.item() / n_f, step)
    add(f"{NS}/Reward/env_reward_mean", self.s_env_reward.item() / n_f, step)
    add(f"{NS}/Task/target_distance_mean_m", self.s_distance.item() / n_f, step)
    add(f"{NS}/Motion/horizontal_speed_mean_mps", self.s_speed.item() / n_f, step)

    ended = int(self.it_ended.item())
    failure = int(self.it_failure.item())
    timeout_only = int(self.it_timeout.item())
    startup = int(self.it_startup.item())
    complete = int(self.it_complete.item())
    complete_failure = int(self.it_complete_failure.item())
    complete_timeout = int(self.it_complete_timeout.item())

    add(f"{NS}/Count/ended_episode", ended, step)
    add(f"{NS}/Count/failure_episode", failure, step)
    add(f"{NS}/Count/timeout_only_episode", timeout_only, step)
    add(f"{NS}/Count/complete_episode", complete, step)
    add(f"{NS}/Count/complete_failure_episode", complete_failure, step)
    add(f"{NS}/Count/complete_timeout_only_episode", complete_timeout, step)
    add(f"{NS}/Count/startup_incomplete_episode", startup, step)

    sum_len = self.s_complete_len.item()
    sum_fail_dur = self.s_complete_fail_dur.item()
    sum_task_ret = self.s_complete_task_ret.item()
    sum_train_ret = self.s_complete_train_ret.item()

    add(f"{NS}/Sum/complete_episode_length_steps", sum_len, step)
    add(f"{NS}/Sum/complete_failure_duration_s", sum_fail_dur, step)
    add(f"{NS}/Sum/complete_task_return_raw", sum_task_ret, step)
    add(f"{NS}/Sum/complete_training_return", sum_train_ret, step)

    add(f"{NS}/Progress/completed_iterations", step, step)
    add(f"{NS}/Progress/training_samples", self.total_samples, step)
    add(f"{NS}/Time/iteration_s", iteration_time, step)
    add(f"{NS}/Time/training_elapsed_s", self.elapsed, step)
    for name, value in getattr(self.env, "_smp_physics_totals", {}).items():
      add(f"{NS}/GSI/{name}", value, step)

    if complete > 0:
      add(f"{NS}/Episode/complete_task_return_raw_mean", sum_task_ret / complete, step)
      add(f"{NS}/Episode/complete_training_return_mean", sum_train_ret / complete, step)
      add(f"{NS}/Episode/complete_length_steps_mean", sum_len / complete, step)
      add(
        f"{NS}/Episode/complete_survival_s_mean",
        sum_len * self.step_dt / complete,
        step,
      )
      add(f"{NS}/Failure/complete_rate", complete_failure / complete, step)
    if complete_failure > 0:
      add(
        f"{NS}/Failure/complete_duration_s_mean", sum_fail_dur / complete_failure, step
      )
    if ended > 0:
      add(f"{NS}/Failure/all_ended_rate", failure / ended, step)

  def finalize(self, runner: Any) -> None:
    writer = getattr(runner.logger, "writer", None)
    if writer is not None:
      writer.flush()


def build_logged_task_smp_additive(
  command_name: str = "location",
) -> Callable[..., torch.Tensor]:
  """Drop-in replacement for ``smp.rl.rewards.task_smp_additive``.

  Identical math (one ``smp_guidance_reward`` call, same weights/params); the
  only addition is caching the three raw values for ``PassiveMetrics``.
  """

  def term(
    env: Any,
    task_terms: tuple[Any, ...],
    fixed_timesteps: tuple[int, ...] = (8, 15, 22),
    ws: float = 6.0,
    task_reward_weight: float = 0.5,
    smp_reward_weight: float = 0.5,
  ) -> torch.Tensor:
    task = sum(w * func(env, **kw) for func, w, kw in task_terms)
    smp = smp_guidance_reward(env, fixed_timesteps=fixed_timesteps, ws=ws)
    additive = task_reward_weight * task + smp_reward_weight * smp
    metrics = term.metrics
    if metrics is not None:
      metrics.record_step(env, task, smp, additive, task_terms)
    return additive

  term.metrics = None  # type: ignore[attr-defined]
  term.__name__ = "task_smp_additive"
  term.command_name = command_name  # type: ignore[attr-defined]
  return term


def install(env: Any, metrics: PassiveMetrics) -> PassiveMetrics:
  """Install the two remaining passive hooks (reward buffer + reset entry)."""

  reward_manager = env.reward_manager
  original_compute = reward_manager.compute

  def compute(dt: float) -> torch.Tensor:
    reward = original_compute(dt)
    metrics.record_env_reward(reward)
    return reward

  reward_manager.compute = compute  # type: ignore[method-assign]

  original_reset_idx = env._reset_idx

  def _reset_idx(env_ids: Any = None) -> None:
    metrics.on_reset_idx(env, env_ids)
    original_reset_idx(env_ids)

  env._reset_idx = _reset_idx  # type: ignore[method-assign]
  return metrics
