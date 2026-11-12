"""Initial-noise search for 00033; the caller supplies the frozen G1 bundle.

No policy, simulator or global random stream is used by the search. Independent
Sep-CMA-ES jobs share a fixed-size decoder batch; all original rows are retained.
"""

from __future__ import annotations

import importlib.util
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

_path = Path(__file__).with_name("vendor") / "_sepcma.py"
_spec = importlib.util.spec_from_file_location("_smp00033_sepcma", _path)
assert _spec is not None and _spec.loader is not None
_vendor = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _vendor
_spec.loader.exec_module(_vendor)
SepCMA = _vendor.SepCMA


@dataclass(frozen=True)
class SearchConfig:
  population_size: int = 32
  max_generations: int = 30
  sigma0: float = 0.05
  rms_limit: float = 0.15
  coordinate_bound: float = 1.0
  inference_batch_size: int = 1024

  def validate(self):
    assert self.population_size == 32
    assert self.max_generations == 30
    assert self.sigma0 == 0.05 and self.rms_limit == 0.15
    assert self.coordinate_bound == 1.0
    assert self.inference_batch_size >= self.population_size
    assert self.inference_batch_size % self.population_size == 0

  def as_dict(self):
    return asdict(self)


@torch.no_grad()
def decode(bundle, initial: torch.Tensor, posterior: torch.Tensor):
  """Native DDPM algebra with explicit, read-only posterior noise arrays."""
  model, scheduler, low, high, feature_dim, window_size = bundle
  T = int(scheduler.num_timesteps)
  assert initial.shape[1:] == (window_size, feature_dim)
  assert posterior.shape == (T - 1, *initial.shape)
  x = initial
  for t in reversed(range(T)):
    times = torch.full((len(x),), t, dtype=torch.long, device=x.device)
    eps = model(x, times)
    x0 = scheduler.sqrt_recip_alphas_cumprod[t] * x - (
      scheduler.sqrt_recipm1_alphas_cumprod[t] * eps
    )
    mean = scheduler.posterior_mean_coef1[t] * x0 + (
      scheduler.posterior_mean_coef2[t] * x
    )
    x = (
      mean
      if t == 0
      else mean + torch.sqrt(scheduler.posterior_variance[t]) * posterior[T - 1 - t]
    )
  return (x + 1.0) / 2.0 * (high - low) + low


@torch.no_grad()
def draw_recorded(bundle, batch_size: int, device):
  """Consume exactly the native sampler's draws, in their original order."""
  model, scheduler, low, high, F, W = bundle
  T = int(scheduler.num_timesteps)
  initial = torch.randn(batch_size, W, F, device=device)
  posterior = torch.empty((T - 1, batch_size, W, F), device=device)
  x = initial
  for t in reversed(range(T)):
    times = torch.full((batch_size,), t, dtype=torch.long, device=device)
    eps = model(x, times)
    x0 = scheduler.sqrt_recip_alphas_cumprod[t] * x - (
      scheduler.sqrt_recipm1_alphas_cumprod[t] * eps
    )
    mean = scheduler.posterior_mean_coef1[t] * x0 + (
      scheduler.posterior_mean_coef2[t] * x
    )
    if t:
      noise = torch.randn_like(x)
      posterior[T - 1 - t].copy_(noise)
      x = mean + torch.sqrt(scheduler.posterior_variance[t]) * noise
    else:
      x = mean
  raw = (x + 1.0) / 2.0 * (high - low) + low
  if raw.dtype != torch.float32:
    raise TypeError("00033 expects the native float32 prior")
  return raw, initial, posterior


class PhysicalScorer:
  """Continuous violations; admission success always uses the original gate."""

  def __init__(self, physics):
    self.physics = physics

  def evaluate(self, windows):
    from smp.rl.gsi_physics import rotations, segment_distance

    x = np.ascontiguousarray(windows, dtype=np.float32)
    ph = self.physics
    _, passed, _ = ph.process(x, repair=False)
    _, rot_ok = rotations(x.astype(np.float64))
    numeric = np.isfinite(x).all(axis=(1, 2)) & rot_ok.all(axis=1)
    with np.errstate(invalid="ignore", over="ignore"):
      ee = x[:, :, 38:53].reshape(-1, 10, 5, 3) + x[:, :, :3][:, :, None, :]
      velocity = np.diff(x[:, :, 9:38], axis=1) / np.float32(ph.config.control_dt)
      numeric &= np.isfinite(ee).all(axis=(1, 2, 3))
      numeric &= np.isfinite(velocity).all(axis=(1, 2))
    joint = np.full(len(x), np.nan)
    ground = np.full(len(x), np.nan)
    overlap = np.full(len(x), np.nan)
    if numeric.any():
      valid = x[numeric]
      angles = valid[:, :, 9:38].astype(np.float64)
      joint[numeric] = np.maximum(0, np.maximum(ph.lo - angles, angles - ph.hi)).max(
        axis=(1, 2)
      )
      p, q, _ = ph._kinematics(valid)
      ground[numeric] = (
        np.maximum(
          0, ph.ground_z - (np.minimum(p[:, :, 2], q[:, :, 2]) - ph.radius).min(axis=1)
        )
        .reshape(-1, 10)
        .max(axis=1)
        * 1000
      )
      depth = np.zeros(len(p))
      for i, j in ph.pairs:
        gap = (
          segment_distance(p[:, i], q[:, i], p[:, j], q[:, j])
          - ph.radius[i]
          - ph.radius[j]
        )
        depth = np.maximum(depth, np.maximum(0, -gap))
      overlap[numeric] = depth.reshape(-1, 10).max(axis=1) * 1000
    V = (
      (np.maximum(0, np.nan_to_num(joint, nan=1e3) - 1e-5) / (np.pi / 180)) ** 2
      + np.maximum(0, np.nan_to_num(ground, nan=1e4) - 19.72) ** 2
      + np.nan_to_num(overlap, nan=1e4) ** 2
    )
    fitness = np.where(numeric, np.where(passed, V, 10000 + V), 1e12)
    return dict(
      passed=passed,
      numeric=numeric,
      joint=joint,
      ground=ground,
      overlap=overlap,
      fitness=fitness,
    )


def _initial_stats():
  return dict(
    attempted=0,
    succeeded=0,
    exhausted=0,
    proposals=0,
    effective_candidates=0,
    radius_rejected=0,
    search_decode_calls=0,
    search_decoded_slots=0,
    replay_decode_calls=0,
    replay_decoded_slots=0,
    generations=0,
    success_generation_sum=0,
    success_generation_max=0,
    max_selected_rms=0.0,
    replay_max_abs=0.0,
    search_seconds=0.0,
  )


@torch.no_grad()
def search_batch(
  bundle,
  raw,
  initial,
  posterior,
  original_passed,
  scorer,
  *,
  seed: int,
  global_start: int,
  config: SearchConfig,
  optimizer_factory=SepCMA,
  log=print,
):
  """Search every failed row, never selecting the original as a fallback.

  A completed job frees its fixed block for the next failed row. Per-row RNGs
  and posterior tensors are independent of how other jobs finish.
  """
  config.validate()
  B, W, F = raw.shape
  assert (W, F) == (10, 59) and B == config.inference_batch_size
  assert initial.shape == raw.shape and posterior.shape[1:] == raw.shape
  P = config.population_size
  output = raw.detach().cpu().numpy().copy()
  final_passed = np.asarray(original_passed, dtype=bool).copy()
  failed = np.flatnonzero(~final_passed).tolist()
  stats = _initial_stats()
  stats["attempted"] = len(failed)
  if not failed:
    return output, final_passed, stats
  started = time.perf_counter()
  active = [None] * (B // P)
  pending = iter(failed)
  remaining = len(failed)
  winners = {}
  rounds = 0

  while remaining:
    for block, job in enumerate(active):
      if job is None:
        row = next(pending, None)
        if row is not None:
          opt = optimizer_factory(
            mean=np.zeros(W * F),
            sigma=config.sigma0,
            bounds=np.tile(
              [[-config.coordinate_bound, config.coordinate_bound]], (W * F, 1)
            ),
            seed=(seed + 10000 + global_start + row) % 2**32,
            population_size=P,
          )
          active[block] = dict(row=row, opt=opt, generation=0, best=None)
    slots = np.zeros((B, W, F), dtype=np.float32)
    sources = np.zeros(B, dtype=np.int64)
    allowed = np.zeros(B, dtype=bool)
    for block, job in enumerate(active):
      if job is None:
        continue
      candidates = np.stack([job["opt"].ask() for _ in range(P)])
      norms = np.sqrt(np.mean(candidates**2, axis=1))
      valid = norms <= config.rms_limit
      start = block * P
      sources[start : start + P] = job["row"]
      allowed[start : start + P] = valid
      slots[start : start + P][valid] = (
        candidates[valid].reshape(-1, W, F).astype(np.float32)
      )
      job.update(candidates=candidates, norms=norms, generation=job["generation"] + 1)
      stats["proposals"] += P
      stats["effective_candidates"] += int(valid.sum())
      stats["radius_rejected"] += int((~valid).sum())
      stats["generations"] += 1
    src = torch.as_tensor(sources, dtype=torch.long, device=raw.device)
    delta = torch.as_tensor(slots, device=raw.device)
    generated = decode(
      bundle, initial.index_select(0, src) + delta, posterior.index_select(1, src)
    )
    stats["search_decode_calls"] += 1
    stats["search_decoded_slots"] += B
    ids = np.flatnonzero(allowed)
    values = generated[torch.as_tensor(ids, device=raw.device)].cpu().numpy()
    metrics = scorer.evaluate(values) if len(ids) else None
    offsets = np.full(B, -1, dtype=np.int64)
    offsets[ids] = np.arange(len(ids))
    scores = np.full(B, 1e12 + 1e9)
    if len(ids):
      scores[ids] = metrics["fitness"]
    for block, job in enumerate(active):
      if job is None:
        continue
      start = block * P
      for slot in range(start, start + P):
        k = offsets[slot]
        if k < 0:
          continue
        passed = bool(metrics["passed"][k])
        best = job["best"]
        replace = best is None or (passed and not best["passed"])
        replace |= (
          best is not None and not best["passed"] and scores[slot] < best["fitness"]
        )
        if replace:
          job["best"] = dict(
            passed=passed,
            fitness=float(scores[slot]),
            features=values[k].copy(),
            delta=slots[slot].copy(),
            slot=slot,
            rms=float(job["norms"][slot - start]),
          )
      job["opt"].tell(
        [(job["candidates"][i], float(scores[start + i])) for i in range(P)]
      )
      best = job["best"]
      done = best is not None and best["passed"]
      if done or job["generation"] == config.max_generations:
        if best is None:
          raise RuntimeError(
            "no evaluated search candidate; original fallback is disabled"
          )
        row = job["row"]
        output[row] = best["features"]
        final_passed[row] = best["passed"]
        winners[row] = best
        stats["max_selected_rms"] = max(stats["max_selected_rms"], best["rms"])
        stats["succeeded" if done else "exhausted"] += 1
        if done:
          stats["success_generation_sum"] += job["generation"]
          stats["success_generation_max"] = max(
            stats["success_generation_max"], job["generation"]
          )
        active[block] = None
        remaining -= 1
    rounds += 1
    if log and (rounds == 1 or rounds % 5 == 0 or remaining == 0):
      log(
        f"[00033 rescue] original_start={global_start} rounds={rounds} "
        f"remaining={remaining}/{len(failed)} success={stats['succeeded']} "
        f"exhausted={stats['exhausted']}",
        flush=True,
      )

  # Replay each winner at its search-time slot in the same fixed decoder shape.
  # Winners that used the same slot at different times require different calls.
  queue = {}
  for row, best in winners.items():
    queue.setdefault(best["slot"], []).append(row)
  while queue:
    slots = np.zeros((B, W, F), dtype=np.float32)
    sources = np.zeros(B, dtype=np.int64)
    selected = []
    for slot in list(queue):
      row = queue[slot].pop(0)
      sources[slot] = row
      slots[slot] = winners[row]["delta"]
      selected.append((slot, row))
      if not queue[slot]:
        del queue[slot]
    src = torch.as_tensor(sources, device=raw.device)
    replay = (
      decode(
        bundle,
        initial.index_select(0, src) + torch.as_tensor(slots, device=raw.device),
        posterior.index_select(1, src),
      )
      .cpu()
      .numpy()
    )
    replay_windows = np.stack([replay[slot] for slot, _ in selected])
    replay_passed = scorer.evaluate(replay_windows)["passed"]
    for k, (slot, row) in enumerate(selected):
      error = float(np.max(np.abs(replay[slot] - output[row])))
      stats["replay_max_abs"] = max(stats["replay_max_abs"], error)
      if error > 1e-5 or bool(replay_passed[k]) != bool(final_passed[row]):
        raise RuntimeError(
          "search output/replay disagreement; no pool update committed"
        )
    stats["replay_decode_calls"] += 1
    stats["replay_decoded_slots"] += B
  stats["search_seconds"] = time.perf_counter() - started
  assert stats["attempted"] == stats["succeeded"] + stats["exhausted"]
  assert stats["proposals"] == P * stats["generations"]
  assert stats["proposals"] <= len(failed) * P * config.max_generations
  return output, final_passed, stats
