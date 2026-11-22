"""GPU (torch float64) self-penetration check as a drop-in for PhysicalFilter.

Strategy: leave `PhysicalFilter.process` untouched.  We intercept the two things
it does around the 469-pair loop:

  * `_kinematics` is wrapped so we keep the computed capsule endpoints p, q;
  * `gsi_physics.segment_distance` is temporarily replaced by a dispatcher that,
    on its first call inside a process() batch, evaluates *all* 469 pairs on the
    GPU at once and then hands the loop the precomputed columns one by one.

Everything else (joint limits, ground depth, repair, numerics, diagnostics)
stays byte-for-byte the reference code, so verdicts must match exactly.
"""

from __future__ import annotations

import numpy as np
import torch

import gsi_physics
from gsi_physics import PhysicalFilter

_ORIGINAL_DISTANCE = gsi_physics.segment_distance


class _Ctx:
  """Holds the precomputed gap matrix for one process() call."""

  def __init__(self, filler):
    self._filler = filler
    self._cols = None
    self._k = 0

  def next_column(self):
    if self._cols is None:
      self._cols = self._filler()
      if self._k >= self._cols.shape[1]:
        raise RuntimeError("segment_distance called more times than pairs")
    col = self._cols[:, self._k]
    self._k += 1
    return col


_CTX = {"current": None}


def _dispatcher(p, q, r, s):
  ctx = _CTX["current"]
  if ctx is None:
    return _ORIGINAL_DISTANCE(p, q, r, s)
  return ctx.next_column()


gsi_physics.segment_distance = _dispatcher


class GpuPhysicalFilter(PhysicalFilter):
  """PhysicalFilter whose 469-pair distance loop runs on a CUDA device."""

  def __init__(self, model, config, device: str = "cuda:0", chunk_rows: int = 16384):
    super().__init__(model, config)
    self.device = torch.device(device)
    self.chunk_rows = int(chunk_rows)
    self._pi = torch.as_tensor([i for i, _ in self.pairs], device=self.device)
    self._pj = torch.as_tensor([j for _, j in self.pairs], device=self.device)
    self._rad = torch.as_tensor(self.radius, device=self.device, dtype=torch.float64)
    self._last_pq = None

  # ---------------------------------------------------------------- internals
  def _kinematics(self, windows):  # noqa: D401 - keep reference behaviour
    p, q, ee = super()._kinematics(windows)
    self._last_pq = (p, q)
    return p, q, ee

  def _gaps(self, p, q):
    P, Q = p[:, self._pi], q[:, self._pi]
    R, S = p[:, self._pj], q[:, self._pj]
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
    # NOTE: return the raw segment distance; PhysicalFilter.process subtracts the
    # two radii itself, so subtracting here would double-count.
    return torch.sqrt(torch.clamp(best, min=0))

  def _all_gaps(self):
    p, q = self._last_pq
    rows = p.shape[0]
    out = np.empty((rows, len(self.pairs)), dtype=np.float64)
    for start in range(0, rows, self.chunk_rows):
      stop = min(start + self.chunk_rows, rows)
      pt = torch.as_tensor(p[start:stop], device=self.device, dtype=torch.float64)
      qt = torch.as_tensor(q[start:stop], device=self.device, dtype=torch.float64)
      out[start:stop] = self._gaps(pt, qt).cpu().numpy()
    return out

  # ------------------------------------------------------------------ process
  def process(self, windows, *, repair: bool | None = None):
    self._last_pq = None
    _CTX["current"] = _Ctx(self._all_gaps)
    try:
      return super().process(windows, repair=repair)
    finally:
      _CTX["current"] = None
      self._last_pq = None
