"""Qualified GSI collection. Fixed quotas; fail instead of falling back."""

from __future__ import annotations

import math

import numpy as np
import torch

from smp.rl.gsi_physics import FilterConfig, PhysicalFilter


def initialize(env, *, repair: bool, batch_size: int):
  env._smp_physics = PhysicalFilter(
    env.sim.mj_model,
    FilterConfig(
      repair=repair,
      control_dt=float(env.cfg.sim.mujoco.timestep) * int(env.cfg.decimation),
    ),
  )
  env._smp_physics_batch = batch_size
  env._smp_physics_totals = {
    "updates": 0,
    "generated": 0,
    "qualified": 0,
    "inserted": 0,
    "surplus": 0,
    "numerical_failed": 0,
    "joint_failed": 0,
    "geometry_checked": 0,
    "self_failed": 0,
    "ground_failed": 0,
    "repair_attempted": 0,
    "repair_qualified": 0,
    "repair_inserted": 0,
    "repair_velocity_change_max_rad_s": 0.0,
    "repair_acceleration_change_max_rad_s2": 0.0,
    "repair_ee_delta_max_m": 0.0,
  }


def collect(env, count: int, draw):
  batch = env._smp_physics_batch
  if count <= 0 or batch <= 0:
    raise ValueError("GSI quotas and raw batch size must be positive")
  chunks, flags, remaining = [], [], count
  totals = env._smp_physics_totals
  before = dict(totals)
  inserted_repaired = 0
  # Emergency bound is not a fallback. No partial pool replacement on failure.
  for _ in range(math.ceil(100 * count / batch)):
    raw = draw(env, batch)
    if raw.dtype != torch.float32:
      raise TypeError("the frozen G1 prior must emit raw float32 features")
    array = raw.detach().to(device="cpu", dtype=torch.float32).numpy()
    corrected, mask, stats = env._smp_physics.process(array)
    for name, value in stats.items():
      if name in totals:
        totals[name] = (
          max(totals[name], value) if "_max_" in name else totals[name] + value
        )
    ids = np.flatnonzero(mask)[:remaining]
    if len(ids):
      original_q = array[ids, :, 9:38].astype(np.float64)
      physics = env._smp_physics
      was_repaired = (
        np.maximum(0, np.maximum(physics.lo - original_q, original_q - physics.hi)).max(
          axis=(1, 2)
        )
        > physics.config.joint_tolerance_rad
      )
      inserted_repaired += int(was_repaired.sum())
      chunks.append(
        torch.as_tensor(corrected[ids].copy(), device=raw.device, dtype=raw.dtype)
      )
      flags.append(torch.as_tensor(was_repaired, device=raw.device, dtype=torch.bool))
      remaining -= len(ids)
    if remaining == 0:
      result = torch.cat(chunks, dim=0)
      assert len(result) == count
      env._smp_physics_last_repaired = torch.cat(flags)
      totals["updates"] += 1
      totals["inserted"] += count
      totals["repair_inserted"] += inserted_repaired
      totals["surplus"] += totals["qualified"] - before["qualified"] - count
      print(
        f"[GSI physical] step={int(env.common_step_counter)} quota={count} "
        f"generated={totals['generated'] - before['generated']} "
        f"qualified={totals['qualified'] - before['qualified']} "
        f"inserted={count} repaired_inserted={inserted_repaired}",
        flush=True,
      )
      return result
  raise RuntimeError(
    f"GSI physical quota not filled: requested={count}, missing={remaining}; no fallback or partial replacement"
  )
