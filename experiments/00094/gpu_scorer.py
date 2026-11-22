"""Single-pass GPU scorer: replaces PhysicalScorer.evaluate without changing values.

The reference scorer computes the 469-pair geometry twice (once inside
``process`` for the boolean verdict, once here for the continuous penalty).
This version runs FK once and the pair distances once -- on the GPU -- and
derives both the verdict and the continuous terms from that single pass.

Validation (see gpu_score_validate.py) must show *identical* arrays for
``passed``, ``joint``, ``ground``, ``overlap`` and ``fitness``.
"""

from __future__ import annotations

import numpy as np
import torch

from gsi_physics import rotations


class GpuScorer:
  """Drop-in replacement for PhysicalScorer.evaluate with one geometry pass."""

  def __init__(self, physics):
    self.physics = physics
    self.ph = physics

  # ------------------------------------------------------------------ helpers
  def _gaps(self, p, q):
    ph = self.ph
    pp = torch.as_tensor(p, device=ph.device, dtype=torch.float64)
    qq = torch.as_tensor(q, device=ph.device, dtype=torch.float64)
    pi, pj = ph._pi, ph._pj
    P, Q = pp[:, pi], qq[:, pi]
    R, S = pp[:, pj], qq[:, pj]
    u, v, w = Q - P, S - R, P - R

    def dot(a, b):
      return (a * b).sum(-1)

    def div(a, b):
      return torch.where(b > 1e-25, a / b, torch.zeros_like(a))

    aa, bb, cc = dot(u, u), dot(u, v), dot(v, v)
    dd, ee = dot(u, w), dot(v, w)
    t0, t1 = div(ee, cc).clamp(0, 1), div(ee + bb, cc).clamp(0, 1)
    s0, s1 = div(-dd, aa).clamp(0, 1), div(bb - dd, aa).clamp(0, 1)
    best = torch.minimum(dot(w - t0[..., None] * v, w - t0[..., None] * v),
                         dot(w + u - t1[..., None] * v, w + u - t1[..., None] * v))
    z = w + s0[..., None] * u
    best = torch.minimum(best, dot(z, z))
    z = w + s1[..., None] * u - v
    best = torch.minimum(best, dot(z, z))
    den = aa * cc - bb * bb
    si, ti = div(bb * ee - cc * dd, den), div(aa * ee - bb * dd, den)
    interior = (den > 1e-25) & (si >= 0) & (si <= 1) & (ti >= 0) & (ti <= 1)
    z = w + si[..., None] * u - ti[..., None] * v
    best = torch.where(interior, torch.minimum(best, dot(z, z)), best)
    return torch.sqrt(torch.clamp(best, min=0)) - ph._rad[pi] - ph._rad[pj]

  # ----------------------------------------------------------------- evaluate
  def evaluate(self, windows):
    ph = self.ph
    x = np.ascontiguousarray(windows, dtype=np.float32)
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
      joint[numeric] = np.maximum(0, np.maximum(ph.lo - angles, angles - ph.hi)).max(axis=(1, 2))
      p, q, _ = ph._kinematics(valid)          # the single FK pass
      ground[numeric] = (
        np.maximum(0, ph.ground_z - (np.minimum(p[:, :, 2], q[:, :, 2]) - ph.radius).min(axis=1))
        .reshape(-1, 10).max(axis=1) * 1000
      )
      rows = p.shape[0]
      gaps = np.empty((rows, len(ph.pairs)), dtype=np.float64)
      for start in range(0, rows, ph.chunk_rows):
        stop = min(start + ph.chunk_rows, rows)
        gaps[start:stop] = self._gaps(p[start:stop], q[start:stop]).cpu().numpy()
      flat = np.maximum(0, -gaps)                      # (rows, pairs)
      # Only pairs that actually penetrate decide the maximum, and there are few
      # of them.  Re-evaluating exactly those with the reference numpy routine
      # makes the value bit-identical (torch sqrt differs by 1 ulp).
      from gsi_physics import segment_distance
      rows_i, pair_j = np.nonzero(flat > 0)
      depth_rows = np.zeros(flat.shape[0], dtype=np.float64)
      if rows_i.size:
        gi = np.array([ph.pairs[k][0] for k in pair_j], dtype=np.intp)
        gj = np.array([ph.pairs[k][1] for k in pair_j], dtype=np.intp)
        exact = (segment_distance(p[rows_i, gi], q[rows_i, gi], p[rows_i, gj], q[rows_i, gj])
                 - ph.radius[gi] - ph.radius[gj])
        np.maximum.at(depth_rows, rows_i, np.maximum(0, -exact))
      depth = depth_rows.reshape(-1, 10).max(axis=1)
      overlap[numeric] = depth * 1000
    V = (
      (np.maximum(0, np.nan_to_num(joint, nan=1e3) - 1e-5) / (np.pi / 180)) ** 2
      + np.maximum(0, np.nan_to_num(ground, nan=1e4) - 19.72) ** 2
      + np.nan_to_num(overlap, nan=1e4) ** 2
    )
    # verdict: same predicates as PhysicalFilter.process (numeric & joint & ground & self)
    passed = np.zeros(len(x), dtype=bool)
    if numeric.any():
      idx = np.flatnonzero(numeric)
      joint_ok = joint[numeric] <= ph.config.joint_tolerance_rad
      ground_ok = ground[numeric] <= ph.config.ground_limit_mm
      self_ok = overlap[numeric] <= 0.0
      passed[idx] = joint_ok & ground_ok & self_ok
    fitness = np.where(numeric, np.where(passed, V, 10000 + V), 1e12)
    return dict(passed=passed, numeric=numeric, joint=joint, ground=ground,
                overlap=overlap, fitness=fitness)
