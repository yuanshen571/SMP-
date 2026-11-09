"""Experiment 0005: real rollout windows selected by completed-episode lifetime.

No physics gate, repair, reward threshold, extra simulation or persistent raw
trajectories. A private GPU ring holds at most one episode horizon. Independent
priority reservoirs per episode length allow exact uniform window selection
AFTER the window's final episode-weighted mean is known.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

# A real post-forward frame, before any re-anchoring. Positions are env-relative.
# root(pos3, quat4, linvel3, angvel3), joints(pos29, vel29), EE positions(5x3).
FRAME_DIM = 86
META_COLUMNS = [
  "env",
  "episode_serial",
  "episode_start",
  "episode_end",
  "episode_steps",
  "collection_start",
  "collection_end",
  "candidate_start",
  "candidate_end",
  "future_steps",
  "episode_failed",
]


@dataclass(frozen=True)
class MixedConfig:
  capacity: int = 456
  replace_count: int = 228
  policy_probability: float = 0.1
  collect_after_iteration: int = 1000
  activate_after_iteration: int = 1500
  update_interval: int = 250
  window_frames: int = 10
  future_steps: int = 50
  collection_rng_seed: int = 2027005
  reset_rng_seed: int = 2027006

  def __post_init__(self):
    if self.capacity != 2 * self.replace_count or self.replace_count < 1:
      raise ValueError("need two equal, nonempty alternating halves")
    if not 0 < self.policy_probability < 1:
      raise ValueError("invalid reset mixture")
    if not 0 <= self.collect_after_iteration < self.activate_after_iteration:
      raise ValueError("collection must precede activation")
    if self.update_interval < 1 or self.window_frames != 10 or self.future_steps != 50:
      raise ValueError(
        "need positive update interval, ten real frames, 50-step lookahead"
      )


def config_from_spec(spec):
  return MixedConfig(**{key: spec[key] for key in MixedConfig.__dataclass_fields__})


def digest(array):
  value = np.ascontiguousarray(array)
  return hashlib.sha256(value.tobytes()).hexdigest()


class WindowReservoir:
  """Keep K smallest independent keys PER length, before mean is available.

  Any global top-K candidate must belong to the top-K of its own length bucket.
  Thus dropping larger keys is exact, not an approximate episode-first sampler.
  Only the final selected windows are kept beyond this collection window.
  """

  def __init__(self, start, end, quota, horizon, rng, frames=10, future=50):
    self.start, self.end, self.quota = start, end, quota
    self.rng, self.frames, self.future = rng, frames, future
    self.episode_count, self.sum_steps = 0, 0
    self.episode_histogram = np.zeros(horizon + 1, dtype=np.int64)
    self.candidate_counts = np.zeros(horizon + 1, dtype=np.int64)
    self.buckets = {}
    self.retained_count, self.peak_bytes = 0, 0

  def add_episode(self, *, env_id, serial, start, end, failed, load):
    length = end - start
    if not self.start <= end <= self.end or not 0 < length < len(
      self.episode_histogram
    ):
      raise ValueError("episode end/length outside the collection contract")
    self.episode_count += 1
    self.sum_steps += length
    self.episode_histogram[length] += 1
    first = max(start + self.frames, self.start + self.frames - 1)
    # Failure on exactly the 50th future step fails the survival condition.
    # A pure timeout on that step has completed the full observation horizon.
    last = end - self.future - int(failed)
    ends = np.arange(first, last + 1, dtype=np.int64)
    n = len(ends)
    self.candidate_counts[length] += n
    if not n:
      return
    old = self.buckets.get(length)
    old_n = 0 if old is None else len(old[0])
    keys = self.rng.random(n)
    all_keys = keys if old is None else np.concatenate([old[0], keys])
    take = min(self.quota, len(all_keys))
    keep = np.argpartition(all_keys, take - 1)[:take]
    keep = keep[np.argsort(all_keys[keep], kind="stable")]
    incoming = keep >= old_n
    if not incoming.any():
      return
    data = np.empty((take, self.frames, FRAME_DIM), dtype=np.float32)
    meta = np.empty((take, len(META_COLUMNS)), dtype=np.int64)
    if (~incoming).any():
      data[~incoming], meta[~incoming] = (
        old[1][keep[~incoming]],
        old[2][keep[~incoming]],
      )
    selected_ends = ends[keep[incoming] - old_n]
    loaded = np.asarray(load(selected_ends), dtype=np.float32)
    if loaded.shape != (len(selected_ends), self.frames, FRAME_DIM):
      raise ValueError("incorrect real-trajectory snapshot shape")
    if not np.isfinite(loaded).all():
      raise FloatingPointError(
        "non-finite rollout data; abort without screening/repair"
      )
    data[incoming] = loaded
    meta[incoming] = np.column_stack(
      [
        np.full(len(selected_ends), env_id),
        np.full(len(selected_ends), serial),
        np.full(len(selected_ends), start),
        np.full(len(selected_ends), end),
        np.full(len(selected_ends), length),
        np.full(len(selected_ends), self.start),
        np.full(len(selected_ends), self.end),
        selected_ends - self.frames + 1,
        selected_ends,
        np.full(len(selected_ends), self.future),
        np.full(len(selected_ends), int(failed)),
      ]
    )
    self.buckets[length] = (all_keys[keep], data, meta)
    self.retained_count += take - old_n
    self.peak_bytes = max(
      self.peak_bytes,
      self.retained_count * (self.frames * FRAME_DIM * 4 + len(META_COLUMNS) * 8 + 8),
    )

  def finish(self):
    if not self.episode_count:
      raise RuntimeError("no completed non-startup episodes in collection window")
    # Integer comparison preserves strict > without rounding mean seconds.
    eligible = [
      length for length in self.buckets if length * self.episode_count > self.sum_steps
    ]
    population = sum(int(self.candidate_counts[length]) for length in eligible)
    if population < self.quota:
      raise RuntimeError(
        f"eligible windows {population} < fixed quota {self.quota}; stop"
      )
    keys = np.concatenate([self.buckets[length][0] for length in eligible])
    keep = np.argpartition(keys, self.quota - 1)[: self.quota]
    keep = keep[np.argsort(keys[keep], kind="stable")]
    data = np.empty((self.quota, self.frames, FRAME_DIM), dtype=np.float32)
    meta = np.empty((self.quota, len(META_COLUMNS)), dtype=np.int64)
    offset = 0
    for length in eligible:
      bucket = self.buckets[length]
      selected = (keep >= offset) & (keep < offset + len(bucket[0]))
      data[selected] = bucket[1][keep[selected] - offset]
      meta[selected] = bucket[2][keep[selected] - offset]
      offset += len(bucket[0])
    audit = dict(
      collection_start_step=self.start,
      collection_end_step=self.end,
      complete_episodes=self.episode_count,
      sum_complete_episode_steps=self.sum_steps,
      mean_episode_steps=self.sum_steps / self.episode_count,
      eligible_windows=population,
      selected_windows=self.quota,
      candidate_windows=int(self.candidate_counts.sum()),
      episode_histogram={
        str(i): int(n) for i, n in enumerate(self.episode_histogram) if n
      },
      candidate_counts={
        str(i): int(n) for i, n in enumerate(self.candidate_counts) if n
      },
      peak_reservoir_bytes=self.peak_bytes,
    )
    return data, meta, audit


def capture_frame(env):
  robot = env.scene["robot"].data
  origin = env.scene.env_origins
  return torch.cat(
    [
      robot.root_link_pos_w - origin,
      robot.root_link_quat_w,
      robot.root_link_lin_vel_w,
      robot.root_link_ang_vel_w,
      robot.joint_pos,
      robot.joint_vel,
      (robot.body_link_pos_w[:, env._smp_ee_indexes] - origin[:, None]).flatten(1),
    ],
    dim=-1,
  ).detach()


def restore(env, ids, windows):
  """Restore real kinematics and actual joint velocities; translate root XY only.

  The same translation is applied to all ten root/EE positions. Orientations,
  velocities, joint states and every relative motion are otherwise untouched.
  """
  if windows.shape != (len(ids), 10, FRAME_DIM) or not bool(
    torch.isfinite(windows).all()
  ):
    raise ValueError("invalid rollout cache payload")
  raw = windows.clone()
  robot = env.scene["robot"]
  delta = robot.data.default_root_state[ids, :2] - raw[:, -1, :2]
  raw[:, :, :2] += delta[:, None]
  ee = raw[:, :, 71:86].reshape(len(ids), 10, 5, 3)
  ee[:, :, :, :2] += delta[:, None, None]
  root = raw[:, -1, :13].clone()
  root[:, :3] += env.scene.env_origins[ids]
  robot.write_root_state_to_sim(root, env_ids=ids)
  robot.write_joint_state_to_sim(raw[:, -1, 13:42], raw[:, -1, 42:71], env_ids=ids)
  env._smp_buffer.reset(
    ids,
    raw[:, :, :3],
    raw[:, :, 3:7],
    raw[:, :, 7:10],
    raw[:, :, 10:13],
    ee,
    raw[:, :, 13:42],
    raw[:, :, 42:71],
  )


class MixedCache:
  def __init__(self, env, *, config, steps_per_iteration=24, audit_path=None):
    self.env, self.config = env, config
    self.steps_per_iteration = steps_per_iteration
    if abs(float(env.step_dt) - 0.02) > 1e-12:
      raise ValueError("0005 lookahead requires the unchanged 50Hz control rate")
    self.horizon = int(env.max_episode_length)
    self.global_step = int(env.common_step_counter)
    if self.global_step != 0:
      raise ValueError(
        "0005 starts fresh; cannot reconstruct cache from a resumed PPO file"
      )
    self.episode_start = np.zeros(env.num_envs, dtype=np.int64)
    self.serial = np.zeros(env.num_envs, dtype=np.int64)
    self.startup = np.ones(env.num_envs, dtype=bool)
    self.collection_rng = np.random.default_rng(config.collection_rng_seed)
    self.reset_rng = torch.Generator(device=env.device).manual_seed(
      config.reset_rng_seed
    )
    self.window_start = config.collect_after_iteration * steps_per_iteration + 1
    self.window_end = config.activate_after_iteration * steps_per_iteration
    self.reservoir, self.history = None, None
    self.active, self.in_step = False, False
    self.pool, self.metadata, self.halves = None, None, None
    self.next_half = 0
    self.transactions = []
    self.audit_path = Path(audit_path) if audit_path else None
    self.stats = dict(
      transactions=0,
      inserted=0,
      gsi_reset_draws=0,
      mixed_gsi_reset_draws=0,
      policy_reset_draws=0,
      mixed_reset_calls=0,
      pre_activation_policy_draws=0,
      first_policy_iteration=0,
      truncated_episodes=0,
      peak_reservoir_bytes=0,
    )

  def count_gsi(self, n):
    self.stats["gsi_reset_draws"] += n

  def reset(self, ids, gsi_restore):
    if not self.active or self.pool is None:
      raise RuntimeError("policy reset before complete initial admission")
    n = len(ids)
    mask = (
      torch.rand(n, device=self.env.device, generator=self.reset_rng)
      < self.config.policy_probability
    )
    policy_ids, gsi_ids = ids[mask], ids[~mask]
    if len(gsi_ids):
      indices = torch.randint(
        0, len(self.env._smp_gsi_pool), (len(gsi_ids),), device=self.env.device
      )
      gsi_restore(self.env, gsi_ids, self.env._smp_gsi_pool[indices])
      self.count_gsi(len(gsi_ids))
      self.stats["mixed_gsi_reset_draws"] += len(gsi_ids)
    if len(policy_ids):
      indices = torch.randint(
        0,
        len(self.pool),
        (len(policy_ids),),
        device=self.env.device,
        generator=self.reset_rng,
      )
      restore(self.env, policy_ids, self.pool[indices])
      iteration = (self.global_step // self.steps_per_iteration) + 1
      if not self.stats["first_policy_iteration"]:
        self.stats["first_policy_iteration"] = iteration
      if iteration <= self.config.activate_after_iteration:
        self.stats["pre_activation_policy_draws"] += len(policy_ids)
        raise AssertionError("policy source used before the agreed activation boundary")
      self.stats["policy_reset_draws"] += len(policy_ids)
    self.stats["mixed_reset_calls"] += 1

  def on_reset(self, ids):
    ids = ids.detach().cpu().numpy().astype(np.int64, copy=False)
    if not len(ids):
      return
    step = self.global_step + int(self.in_step)
    if self.in_step:
      ended = (
        (self.env.reset_terminated | self.env.reset_time_outs)[ids]
        .detach()
        .cpu()
        .numpy()
      )
      failed = self.env.reset_terminated[ids].detach().cpu().numpy()
      if self.reservoir is not None:
        complete = (~self.startup[ids]) & ended
        for env_id, failure in zip(ids[complete], failed[complete], strict=True):
          start, serial = int(self.episode_start[env_id]), int(self.serial[env_id])

          def load(ends, env_id=int(env_id)):
            slots = (ends[:, None] + np.arange(-9, 1)) % self.horizon
            slots = torch.as_tensor(slots, device=self.env.device)
            return self.history[slots, env_id].cpu().numpy()

          self.reservoir.add_episode(
            env_id=int(env_id),
            serial=serial,
            start=start,
            end=step,
            failed=bool(failure),
            load=load,
          )
      self.stats["truncated_episodes"] += int((~ended).sum())
      self.startup[ids] = False
    elif self.global_step:
      self.stats["truncated_episodes"] += len(ids)
      self.startup[ids] = False
    self.episode_start[ids] = step
    self.serial[ids] += 1

  def before_step(self):
    if self.global_step + 1 == self.window_start:
      if self.history is None:
        self.history = torch.empty(
          (self.horizon, self.env.num_envs, FRAME_DIM),
          dtype=torch.float32,
          device=self.env.device,
        )
      quota = self.config.replace_count if self.active else self.config.capacity
      self.reservoir = WindowReservoir(
        self.window_start, self.window_end, quota, self.horizon, self.collection_rng
      )
    self.in_step = True

  def after_step(self):
    self.in_step = False
    self.global_step += 1
    if int(self.env.common_step_counter) != self.global_step:
      raise AssertionError("collector is not synchronized with real environment steps")
    if self.reservoir is not None:
      self.history[self.global_step % self.horizon] = capture_frame(self.env)
      self.stats["peak_reservoir_bytes"] = max(
        self.stats["peak_reservoir_bytes"], self.reservoir.peak_bytes
      )
      if self.global_step == self.window_end:
        self.finish_window()

  def finish_window(self):
    data, meta, audit = self.reservoir.finish()
    initial = not self.active
    if initial:
      slots = np.arange(self.config.capacity)
      new_pool = torch.as_tensor(data, device=self.env.device).clone()
      new_meta = meta.copy()
      order = self.collection_rng.permutation(self.config.capacity)
      self.halves = [
        order[: self.config.replace_count],
        order[self.config.replace_count :],
      ]
      next_half = 0
    else:
      slots = self.halves[self.next_half]
      new_pool, new_meta = self.pool.clone(), self.metadata.copy()
      new_pool[torch.as_tensor(slots, device=self.env.device)] = torch.as_tensor(
        data, device=self.env.device
      )
      new_meta[slots] = meta
      next_half = 1 - self.next_half
    audit.update(
      completed_iteration=self.global_step // self.steps_per_iteration,
      initial=initial,
      replaced_slots=slots.tolist(),
      selected_meta=meta.tolist(),
      pool_metadata_sha256=digest(new_meta),
      pool_payload_sha256=digest(new_pool.cpu().numpy()),
    )
    self.pool, self.metadata, self.next_half = new_pool, new_meta, next_half
    self.active = True
    self.stats["transactions"] += 1
    self.stats["inserted"] += len(data)
    self.stats["peak_reservoir_bytes"] = max(
      self.stats["peak_reservoir_bytes"], self.reservoir.peak_bytes
    )
    self.transactions.append(audit)
    self.reservoir = None
    self.window_start = self.window_end + 1
    self.window_end += self.config.update_interval * self.steps_per_iteration
    if self.audit_path:
      self.audit_path.parent.mkdir(parents=True, exist_ok=True)
      temporary = self.audit_path.with_suffix(".tmp")
      temporary.write_text(
        json.dumps(
          {"meta_columns": META_COLUMNS, "transactions": self.transactions}, indent=2
        )
        + "\n",
        encoding="utf-8",
      )
      temporary.replace(self.audit_path)
    print(
      f"[mixed cache] iteration={audit['completed_iteration']} inserted={len(data)} "
      f"eligible={audit['eligible_windows']} mean_lifetime_s={audit['mean_episode_steps'] * 0.02:.6f}",
      flush=True,
    )

  def summary(self):
    return dict(
      **self.stats,
      active=self.active,
      size=0 if self.pool is None else len(self.pool),
      next_half=self.next_half,
      policy_probability=self.config.policy_probability,
      history_bytes=0 if self.history is None else self.history.numel() * 4,
      last_update_iteration=0
      if not self.transactions
      else self.transactions[-1]["completed_iteration"],
      pool_payload_sha256=None
      if self.pool is None
      else digest(self.pool.cpu().numpy()),
    )


def install(env, *, config, steps_per_iteration=24, audit_path=None):
  cache = MixedCache(
    env, config=config, steps_per_iteration=steps_per_iteration, audit_path=audit_path
  )
  env._smp_mixed_cache = cache
  original_step, original_reset = env.step, env._reset_idx

  def step(action):
    cache.before_step()
    try:
      result = original_step(action)
    except BaseException:
      cache.in_step = False
      raise
    cache.after_step()
    return result

  def reset(ids=None):
    if ids is None:
      ids = torch.arange(env.num_envs, device=env.device)
    cache.on_reset(ids)
    return original_reset(ids)

  env.step, env._reset_idx = step, reset
  return cache
