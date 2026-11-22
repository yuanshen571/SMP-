"""CPU execution of borrowed Torch geometry; this is NOT CUDA acceptance."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import torch
from gpu_filter import GpuPhysicalFilter
from gpu_scorer import GpuScorer
from gsi_physics import segment_distance
from noise_rescue import PhysicalScorer
from test_scoring import Spheres


class TorchSpheres(Spheres):
  device = torch.device("cpu")
  chunk_rows = 64
  config = SimpleNamespace(
    control_dt=0.02, joint_tolerance_rad=1e-5, ground_limit_mm=19.72
  )
  _pi, _pj = torch.tensor([0]), torch.tensor([1])
  _rad = torch.tensor([0.1, 0.1], dtype=torch.float64)


class TorchGeometryTests(unittest.TestCase):
  def test_single_pass_score_matches_cpu_for_all_six_arrays(self):
    x = np.zeros((6, 10, 59), dtype=np.float32)
    x[:, :, 2] = 0.1
    x[:, :, 3] = x[:, :, 8] = 1
    x[:, :, 38] = 0.3
    x[1, 0, 9] = 1.001
    x[2, 0, 2] = 0.08
    x[3, 0, 38] = 0.15
    x[4, 0, 9] = 1.001
    x[4, 0, 38] = 0.15
    x[5, 0, 9] = np.nan
    reference = PhysicalScorer(Spheres()).evaluate(x)
    actual = GpuScorer(TorchSpheres()).evaluate(x)
    for key in ("passed", "numeric", "joint", "ground", "overlap", "fitness"):
      self.assertTrue(np.array_equal(reference[key], actual[key], equal_nan=True), key)

  def test_469_pair_torch_distances_match_segment_reference(self):
    rng = np.random.default_rng(2027)
    pairs = [(i, j) for i in range(33) for j in range(i + 1, 33)][:469]
    p = rng.normal(size=(8, 33, 3))
    q = p + rng.normal(size=p.shape)
    q[0] = p[0]  # zero-length segments/spheres
    q[1] = p[1] + np.array([1.0, 0.0, 0.0])  # parallel segments
    fake = SimpleNamespace(
      _pi=torch.tensor([i for i, _ in pairs]), _pj=torch.tensor([j for _, j in pairs])
    )
    got = GpuPhysicalFilter._gaps(fake, torch.tensor(p), torch.tensor(q)).numpy()
    expected = np.stack(
      [segment_distance(p[:, i], q[:, i], p[:, j], q[:, j]) for i, j in pairs], axis=1
    )
    np.testing.assert_allclose(got, expected, atol=2e-14, rtol=1e-14)


if __name__ == "__main__":
  unittest.main(verbosity=2)
