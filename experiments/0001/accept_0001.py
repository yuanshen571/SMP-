"""Stage B acceptance for experiment ``0001`` (no policy updates).

Runs a short, separate process that only steps the environment with random
actions.  It checks the facts EXECUTION.md §B asks for:

* default GSI full reset at startup, partial (subset) re-priming on resets and
  the FIFO ``gsi_refresh`` at 2400 common steps;
* termination / timeout masks and the 10-frame critic history;
* finite observations and rewards;
* reward decomposition and the ``scale_rewards_by_dt`` identity
  ``env_reward == weight * 0.02 * r_task * r_smp`` (single reward term);
* episode statistics across rollout iterations, including partial resets,
  simultaneous failure+timeout classification and the randomly offset first
  episode (``startup_incomplete_episode``), compared against an independent
  counter that hooks ``_reset_idx`` on its own.

No simulator trajectories or per-iteration tables are written; only a short
summary JSON/Markdown and a small TensorBoard event file.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

import smp.rl.tasks  # noqa: E402,F401
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.tasks.registry import load_env_cfg  # noqa: E402
from tensorboard.backend.event_processing.event_accumulator import (  # noqa: E402
  EventAccumulator,
)
from torch.utils.tensorboard import SummaryWriter  # noqa: E402

_spec = importlib.util.spec_from_file_location(
  "smp_experiment_0001_metrics", Path(__file__).with_name("passive_metrics.py")
)
assert _spec is not None and _spec.loader is not None
pm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pm)

TASK_ID = "Smp-Location-G1"
CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
  CHECKS.append((name, bool(ok), detail))
  print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


class NaiveTracker:
  """Independent episode counter; keeps its own per-env step/task/return sums."""

  def __init__(self, num_envs: int, step_dt: float, device: str | torch.device) -> None:
    self.num_envs = num_envs
    self.step_dt = step_dt
    self.device = torch.device(device)
    self.startup = torch.ones(num_envs, dtype=torch.bool, device=self.device)
    self.ep_product = torch.zeros(num_envs, dtype=torch.float64, device=self.device)
    self.ep_reward = torch.zeros(num_envs, dtype=torch.float64, device=self.device)
    self.ep_steps = torch.zeros(num_envs, dtype=torch.long, device=self.device)
    self.reset()

  def reset(self) -> None:
    self.ended = 0
    self.failure = 0
    self.timeout = 0
    self.startup_ended = 0
    self.complete = 0
    self.complete_failure = 0
    self.complete_timeout = 0
    self.sum_len = 0.0
    self.sum_fail_dur = 0.0
    self.sum_task_ret = 0.0
    self.sum_train_ret = 0.0

  def snapshot(self) -> dict[str, float]:
    return {
      "ended": self.ended,
      "failure": self.failure,
      "timeout": self.timeout,
      "startup": self.startup_ended,
      "complete": self.complete,
      "complete_failure": self.complete_failure,
      "complete_timeout": self.complete_timeout,
      "sum_len": self.sum_len,
      "sum_fail_dur": self.sum_fail_dur,
      "sum_train_ret": self.sum_train_ret,
    }

  def on_reset(self, env, env_ids) -> None:
    if env_ids is None:
      ids = torch.arange(self.num_envs, device=self.device)
    else:
      ids = env_ids.to(self.device, dtype=torch.long)
    term = env.reset_terminated[ids].bool()
    tout = env.reset_time_outs[ids].bool()
    startup = self.startup[ids]
    lengths = self.ep_steps[ids].clone()
    failure = term
    timeout_only = (~term) & tout
    complete = ~startup
    self.ended += int(ids.numel())
    self.failure += int(failure.sum())
    self.timeout += int(timeout_only.sum())
    self.startup_ended += int(startup.sum())
    if bool(complete.any()):
      self.complete += int(complete.sum())
      self.complete_failure += int((complete & failure).sum())
      self.complete_timeout += int((complete & timeout_only).sum())
      self.sum_len += float(lengths[complete].double().sum())
      self.sum_fail_dur += float(lengths[complete & failure].double().sum()) * self.step_dt
      self.sum_task_ret += float(self.ep_product[ids][complete].sum())
      self.sum_train_ret += float(self.ep_reward[ids][complete].sum())
    self.ep_product[ids] = 0.0
    self.ep_reward[ids] = 0.0
    self.ep_steps[ids] = 0
    self.startup[ids] = False


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--num-envs", type=int, default=64)
  parser.add_argument("--steps", type=int, default=2400, help="env.step calls")
  parser.add_argument("--steps-per-env", type=int, default=24)
  parser.add_argument("--seed", type=int, default=2027)
  parser.add_argument("--out", type=str, required=True, help="acceptance output dir")
  args = parser.parse_args(argv)

  out = Path(args.out).resolve()
  out.mkdir(parents=True, exist_ok=True)
  device = "cuda:0"

  env_cfg = load_env_cfg(TASK_ID, play=False)
  env_cfg.scene.num_envs = args.num_envs
  env_cfg.seed = args.seed
  weight = float(env_cfg.rewards["task_smp_product"].weight)

  holder: dict[str, torch.Tensor] = {}
  metrics_term = pm.build_logged_task_smp_product()

  def probe(env, task_terms, fixed_timesteps=(8, 15, 22), ws=6.0):
    value = metrics_term(
      env, task_terms=task_terms, fixed_timesteps=fixed_timesteps, ws=ws
    )
    holder["product"] = value.detach()
    tr = holder.get("tracker")
    if tr is not None:
      # accumulated during the reward stage, i.e. before auto-reset zeroes it
      tr.ep_product += value.detach().double()
    return value

  env_cfg.rewards["task_smp_product"].func = probe

  torch.manual_seed(args.seed)
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)

  # ---------------------------------------------------------------- startup
  check("num_envs", env.num_envs == args.num_envs, f"{env.num_envs}")
  check("step_dt == 0.02", abs(env.step_dt - 0.02) < 1e-9, f"{env.step_dt}")
  check(
    "episode horizon == 1000 steps",
    int(env.max_episode_length) == 1000,
    f"{env.max_episode_length}",
  )
  check(
    "physics 0.005 / decimation 4",
    abs(float(env.cfg.sim.mujoco.timestep) - 0.005) < 1e-12
    and int(env.cfg.decimation) == 4,
    f"dt={env.cfg.sim.mujoco.timestep} dec={env.cfg.decimation}",
  )
  check(
    "scale_rewards_by_dt default True",
    bool(env.cfg.scale_rewards_by_dt),
    f"{env.cfg.scale_rewards_by_dt}",
  )
  cmd = env.command_manager.get_term("location")
  check(
    "location resampling 5-10 s",
    tuple(cmd.cfg.resampling_time_range) == (5.0, 10.0),
    f"{tuple(cmd.cfg.resampling_time_range)}",
  )
  check(
    "target distance range 1-10 m",
    (float(cmd.cfg.tar_dist_min), float(cmd.cfg.tar_dist_max)) == (1.0, 10.0),
    f"{cmd.cfg.tar_dist_min}..{cmd.cfg.tar_dist_max}",
  )
  check(
    "critic history_length == 10",
    int(env.cfg.observations["critic"].history_length) == 10,
    f"{env.cfg.observations['critic'].history_length}",
  )

  obs = env.get_observations()
  actor, critic = obs["actor"], obs["critic"]
  check("actor obs finite", bool(torch.isfinite(actor).all()), f"shape={tuple(actor.shape)}")
  check("critic obs finite", bool(torch.isfinite(critic).all()), f"shape={tuple(critic.shape)}")
  check(
    "critic obs = 10 x actor obs",
    critic.shape[1] == 10 * actor.shape[1],
    f"critic={critic.shape[1]} actor={actor.shape[1]}",
  )

  buf = env._smp_buffer
  pool = env._smp_gsi_pool
  check("GSI pool size 4096", pool.shape[0] == 4096, f"shape={tuple(pool.shape)}")
  check(
    "GSI head starts at 0 (created lazily)", int(getattr(env, "_smp_gsi_head", 0)) == 0
  )
  check("feature window 10 frames", buf.window_size == 10, f"{buf.window_size}")
  check(
    "29 joints / 5 end-effectors (window dim 3+6+29+15+3+3=59)",
    (buf.num_joints, buf.num_ee) == (29, 5),
    f"{buf.num_joints}/{buf.num_ee}",
  )
  primed = float(buf.root_pos_w.abs().sum())
  check("startup GSI full reset primed buffer", primed > 0.0, f"|root_pos|sum={primed:.4f}")
  check(
    "prior window/feature dims [N,10,59]",
    tuple(pool.shape[1:]) == (10, 59),
    f"{tuple(pool.shape[1:])}",
  )

  # ------------------------------------------------------- stepping set-up
  tracker = NaiveTracker(env.num_envs, env.step_dt, env.device)
  holder["tracker"] = tracker
  metrics = pm.PassiveMetrics(env, steps_per_env=args.steps_per_env)
  metrics_term.metrics = metrics

  # The tracker must accumulate the step reward BEFORE auto-reset clears the
  # per-env buffers (same ordering the metrics layer uses).
  original_reward_compute = env.reward_manager.compute

  def tracker_reward_compute(dt):
    rew = original_reward_compute(dt)
    tracker.ep_reward += rew.detach().double()
    tracker.ep_steps += 1
    return rew

  env.reward_manager.compute = tracker_reward_compute  # type: ignore[method-assign]

  original_reset_idx = env._reset_idx

  def tracker_reset(env_ids=None):
    tracker.on_reset(env, env_ids)
    original_reset_idx(env_ids)

  env._reset_idx = tracker_reset  # type: ignore[method-assign]
  pm.install(env, metrics)  # wraps tracker_reset -> both see pre-reset state

  writer = SummaryWriter(log_dir=str(out))

  # mirror rsl_rl's init_at_random_ep_len (baseline behaviour): the first
  # episode of every env starts at a random offset
  env.episode_length_buf = torch.randint_like(
    env.episode_length_buf, high=int(env.max_episode_length)
  )
  tracker.ep_steps.zero_()
  tracker.startup[:] = True
  metrics.ep_startup[:] = True

  max_len = int(env.max_episode_length)
  gen = torch.Generator(device="cpu").manual_seed(args.seed + 1)
  partial_reset_seen = False
  reward_identity_fail = 0
  length_consistency_fail = 0
  simult = 0
  buffer_subset_checked = False
  buffer_subset_ok = True
  pool_before_refresh = None

  iterations = args.steps // args.steps_per_env
  per_iteration: list[dict] = []
  prev_snapshot = tracker.snapshot()

  for it in range(iterations):
    iter_start = time.time()
    collect = 0.0
    for _ in range(args.steps_per_env):
      k = int(env.common_step_counter)
      if k + 1 == 2400:
        pool_before_refresh = pool.clone()
      prev_len = env.episode_length_buf.clone()
      prev_buffer = buf.root_pos_w.clone()
      actions = (
        torch.randn(
          env.num_envs, env.action_manager.total_action_dim, generator=gen
        ).to(env.device)
        * 0.3
      )
      obs, rew, terminated, time_outs, extras = env.step(actions)
      collect += 0.0
      product = holder["product"]

      if not bool(torch.isfinite(rew).all()):
        check("reward finite", False, f"step {k + 1}")
      if not bool(torch.isfinite(obs["actor"]).all()):
        check("actor obs finite during rollout", False, f"step {k + 1}")

      expected = product * weight * env.step_dt
      if not bool(torch.allclose(rew, expected, rtol=1e-5, atol=1e-7)):
        reward_identity_fail += 1
        if reward_identity_fail == 1:
          print(
            f"  first reward identity mismatch at step {k + 1}: "
            f"max|d|={float((rew - expected).abs().max()):.3e}"
          )

      ended = terminated | time_outs
      after = env.episode_length_buf
      if bool(ended.any()):
        if not bool((after[ended] == 0).all()):
          length_consistency_fail += 1
        not_ended = ~ended
        if bool(not_ended.any()) and not bool(
          (after[not_ended] == prev_len[not_ended] + 1).all()
        ):
          length_consistency_fail += 1
        if bool((prev_len[time_outs] + 1 > max_len).any()):
          length_consistency_fail += 1
        n = int(ended.sum())
        if 0 < n < env.num_envs:
          partial_reset_seen = True
          if not buffer_subset_checked:
            # every env's window shifts by one frame per step; only reset envs
            # have their whole window re-primed from the GSI pool
            cur = buf.root_pos_w
            shifted_ok = True
            if bool((~ended).any()):
              shifted_ok = torch.allclose(
                cur[~ended][:, :-1], prev_buffer[~ended][:, 1:], atol=1e-6, rtol=0
              )
            reprimed = bool(
              (cur[ended][:, :-1] != prev_buffer[ended][:, 1:]).any(dim=(1, 2)).all()
            )
            buffer_subset_ok = bool(shifted_ok) and reprimed
            if not buffer_subset_ok:
              print(
                f"  buffer subset detail: shifted_ok={bool(shifted_ok)} "
                f"reprimed={reprimed} n_ended={n}"
              )
            buffer_subset_checked = True
        simult += int((terminated & time_outs).sum())


    iter_time = time.time() - iter_start
    metrics.on_iteration(writer, it, iter_time, 0.0)
    now = tracker.snapshot()
    per_iteration.append({"iteration": it + 1, "tracker": now, "prev": prev_snapshot})
    prev_snapshot = now

  # ------------------------------------------------------------- assertions
  check(
    "env reward == weight*dt*r_product on every step",
    reward_identity_fail == 0,
    f"{reward_identity_fail} mismatching steps",
  )
  check(
    "termination masks/length bookkeeping consistent",
    length_consistency_fail == 0,
    f"{length_consistency_fail} violations",
  )
  check("partial (subset) resets observed", partial_reset_seen)
  check(
    "partial reset re-primes only reset envs",
    buffer_subset_ok,
    "feature buffer rows of non-reset envs untouched",
  )
  print(
    f"[info] env-steps with termination and timeout both true: {simult} "
    "(classification rule covered by test_passive_metrics.py)"
  )

  if pool_before_refresh is not None:
    changed = (
      (pool[:1024] != pool_before_refresh[:1024])
      .any(dim=tuple(range(1, pool.dim())))
      .float()
      .mean()
    )
    untouched_same = bool((pool[1024:] == pool_before_refresh[1024:]).all())
    check("GSI refresh rewrote 1024 pool rows", float(changed) > 0.99, f"{float(changed):.3f}")
    check("GSI refresh left other rows intact", untouched_same)
    check("GSI head advanced to 1024", int(env._smp_gsi_head) == 1024, f"{env._smp_gsi_head}")
  else:
    check("GSI refresh checkpoint reached", False, "2400 env.step calls not executed")

  writer.flush()
  writer.close()
  del writer

  acc = EventAccumulator(str(out), size_guidance={"scalars": 0})
  acc.Reload()
  series: dict[str, dict[int, float]] = {}
  for tag in acc.Tags().get("scalars", []):
    if tag.startswith("0001/"):
      series[tag] = {int(e.step): float(e.value) for e in acc.Scalars(tag)}

  compare = (
    "Count/ended_episode",
    "Count/failure_episode",
    "Count/timeout_only_episode",
    "Count/complete_episode",
    "Count/complete_failure_episode",
    "Count/complete_timeout_only_episode",
    "Count/startup_incomplete_episode",
    "Sum/complete_episode_length_steps",
    "Sum/complete_failure_duration_s",
    "Sum/complete_training_return",
  )
  key = {
    "Count/ended_episode": "ended",
    "Count/failure_episode": "failure",
    "Count/timeout_only_episode": "timeout",
    "Count/complete_episode": "complete",
    "Count/complete_failure_episode": "complete_failure",
    "Count/complete_timeout_only_episode": "complete_timeout",
    "Count/startup_incomplete_episode": "startup",
    "Sum/complete_episode_length_steps": "sum_len",
    "Sum/complete_failure_duration_s": "sum_fail_dur",
    "Sum/complete_training_return": "sum_train_ret",
  }
  mismatch = 0
  for entry in per_iteration:
    step = entry["iteration"]
    for name in compare:
      expected_value = entry["tracker"][key[name]] - entry["prev"][key[name]]
      got = series.get(f"0001/{name}", {}).get(step)
      if got is None:
        print(f"  missing tag 0001/{name} at step {step}")
        mismatch += 1
        continue
      if abs(got - expected_value) > 1e-6 * max(1.0, abs(expected_value)):
        print(f"  MISMATCH {name} step {step}: metrics={got} tracker={expected_value}")
        mismatch += 1
  check(
    "metrics series match independent tracker",
    mismatch == 0,
    f"{mismatch} mismatches over {len(compare)} x {len(per_iteration)} values",
  )

  ended = series.get("0001/Count/ended_episode", {})
  failure = series.get("0001/Count/failure_episode", {})
  timeout = series.get("0001/Count/timeout_only_episode", {})
  complete = series.get("0001/Count/complete_episode", {})
  startup = series.get("0001/Count/startup_incomplete_episode", {})
  identity_bad = 0
  for step in ended:
    if failure.get(step, 0) + timeout.get(step, 0) != ended[step]:
      identity_bad += 1
    if complete.get(step, 0) + startup.get(step, 0) != ended[step]:
      identity_bad += 1
  check("count identities hold for every iteration", identity_bad == 0, f"{identity_bad} bad")
  check(
    "episodes end across iterations (no censoring)",
    len(ended) == iterations and max(ended.values(), default=0) > 0,
    f"iters={len(ended)} max_ended={max(ended.values(), default=0):.0f}",
  )
  check(
    "startup offset episodes classified separately",
    startup.get(1, 0) > 0,
    f"startup at it1={startup.get(1, 0):.0f}",
  )

  failures = [name for name, ok, _ in CHECKS if not ok]
  summary = {
    "task_id": TASK_ID,
    "num_envs": args.num_envs,
    "steps": args.steps,
    "iterations": iterations,
    "simultaneous_failure_timeout_env_steps": simult,
    "checks": [{"name": n, "ok": ok, "detail": d} for n, ok, d in CHECKS],
    "failures": failures,
  }
  (out / "acceptance.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
  )
  lines = ["# 0001 stage-B acceptance", ""]
  lines += [f"- [{'x' if ok else ' '}] {n}" + (f" — {d}" if d else "") for n, ok, d in CHECKS]
  (out / "acceptance.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
  print(f"\n{len(CHECKS) - len(failures)}/{len(CHECKS)} checks passed")
  return 1 if failures else 0


if __name__ == "__main__":
  raise SystemExit(main())
