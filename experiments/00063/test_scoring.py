"""Continuous scoring versus closed-form sphere geometry and hard limits."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from noise_rescue import PhysicalScorer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


class Spheres:
  lo, hi = np.full(29, -1.0), np.full(29, 1.0)
  radius = np.array([0.1, 0.1])
  ground_z = 0.0
  pairs = [(0, 1)]
  config = SimpleNamespace(control_dt=0.02)

  def _kinematics(self, x):
    roots = x[:, :, :3].reshape(-1, 3).astype(np.float64)
    p = np.repeat(roots[:, None], 2, axis=1)
    p[:, 1, 0] += x[:, :, 38].reshape(-1)
    return p, p.copy(), None

  def process(self, x, repair=False):
    assert repair is False
    numeric = np.isfinite(x).all(axis=(1, 2))
    q = x[:, :, 9:38].astype(np.float64)
    joint = ((q >= -1.0 - 1e-5) & (q <= 1.0 + 1e-5)).all(axis=(1, 2))
    ground = (
      np.maximum(0, 0.1 - x[:, :, 2].astype(np.float64)).max(axis=1) * 1000 <= 19.72
    )
    contact = (x[:, :, 38].astype(np.float64) >= 0.2).all(axis=1)
    return x.copy(), numeric & joint & ground & contact, {}


class ScoringTests(unittest.TestCase):
  def test_full_window_metrics_and_joint_failure_does_not_skip_geometry(self):
    x = np.zeros((5, 10, 59), dtype=np.float32)
    x[:, :, 2] = 0.1
    x[:, :, 3] = x[:, :, 8] = 1
    x[:, :, 38] = 0.3
    x[1, 0, 9] = 1.01
    x[2, 0, 2] = 0.08  # 20mm; last frame is fine
    x[3, 0, 38] = 0.15  # spheres overlap by 50mm
    x[4, 0, 9] = 1.01
    x[4, 0, 2] = 0.08
    scored = PhysicalScorer(Spheres()).evaluate(x)
    self.assertEqual(scored["passed"].tolist(), [True, False, False, False, False])
    self.assertAlmostEqual(scored["joint"][1], 0.01, places=6)
    self.assertAlmostEqual(scored["ground"][2], 20, places=4)
    self.assertAlmostEqual(scored["overlap"][3], 50, places=4)
    self.assertGreater(scored["ground"][4], 19.72)
    self.assertGreater(scored["fitness"][4], scored["fitness"][1])
    self.assertEqual(scored["fitness"][0], 0)

  def test_large_tensorboard_counters_are_compared_after_scalar_quantization(self):
    from torch.utils.tensorboard import SummaryWriter
    from verify_completion import load_scalars, tensorboard_float

    value = 2**25 + 1
    with tempfile.TemporaryDirectory() as tmp:
      with SummaryWriter(tmp) as writer:
        writer.add_scalar("00063/NoiseRescue/effective_candidates", value, 1)
      stored = load_scalars(Path(tmp))["00063/NoiseRescue/effective_candidates"][1]
    self.assertNotEqual(stored, value)
    self.assertEqual(stored, tensorboard_float(value))


if __name__ == "__main__":
  unittest.main(verbosity=2)
