"""Real G1 geometry/repair tests; optional original GSI fixture stays outside Git."""

import argparse
import json
import sys
import unittest
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from smp.rl.gsi_physics import FilterConfig, PhysicalFilter, segment_distance  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=Path)
parser.add_argument("--gsi", type=Path)
args = parser.parse_args()
if args.model is None:
  import mjlab

  args.model = Path(mjlab.__file__).parent / "asset_zoo/robots/unitree_g1/xmls/g1.xml"
spec = mujoco.MjSpec.from_file(str(args.model))
model = spec.compile()
if not (model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE).any():
  spec.worldbody.add_geom(
    name="unit_test_ground", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.01]
  )
  model = spec.compile()


class PhysicsTests(unittest.TestCase):
  def setUp(self):
    self.strict = PhysicalFilter(model, FilterConfig())
    self.repair = PhysicalFilter(model, FilterConfig(repair=True))
    self.x = np.zeros((1, 10, 59), dtype=np.float32)
    self.x[:, :, 2] = 3.0
    self.x[:, :, 3] = 1.0
    self.x[:, :, 8] = 1.0

  def test_numeric_and_rotation(self):
    self.assertTrue(self.strict.process(self.x)[1].all())
    for channel, value in [(0, np.nan), (53, np.inf)]:
      x = self.x.copy()
      x[0, 3, channel] = value
      self.assertFalse(self.strict.process(x)[1].any())
    for d6 in [np.zeros(6), np.array([1, 0, 0, 1, 0, 0])]:
      x = self.x.copy()
      x[0, 2, 3:9] = d6
      self.assertFalse(self.strict.process(x)[1].any())

  def test_full_history_ground(self):
    x = self.x.copy()
    x[0, 0, 2] = -1
    self.assertEqual(x[0, -1, 2], 3)
    self.assertFalse(self.strict.process(x)[1].any())

  def test_derived_position_overflow(self):
    x = self.x.copy()
    x[:, :, 0] = 3e38
    x[:, :, 38] = 3e38
    self.assertTrue(np.isfinite(x).all())
    _, ok, stats = self.strict.process(x)
    self.assertFalse(ok.any())
    self.assertEqual(stats["numerical_failed"], 1)

  def test_history_limit_and_whole_window_repair_cap(self):
    x = self.x.copy()
    x[0, 0, 14] = self.strict.hi[5] + np.deg2rad(0.9)
    self.assertFalse(self.strict.process(x)[1].any())
    y, ok, d = self.repair.process(x)
    self.assertTrue(ok.all())
    self.assertEqual(d["repair_attempted"], 1)
    self.assertLess(abs(y[0, 0, 14] - self.strict.hi[5]), 1e-7)
    self.assertTrue(self.strict.process(y)[1].all())
    x[0, 7, 20] = self.strict.hi[11] + np.deg2rad(1.1)
    self.assertFalse(self.repair.process(x)[1].any())

  def test_original_small_tolerance_preserved(self):
    x = self.x.copy()
    x[0, 4, 14] = self.strict.hi[5] + 0.5e-5
    y, ok, d = self.strict.process(x)
    self.assertTrue(ok.all())
    self.assertTrue(np.array_equal(x, y))
    self.assertEqual(d["repair_attempted"], 0)

  def test_correlated_features_and_fresh_differences(self):
    x = self.x.copy()
    x[0, 5, 13] = self.strict.lo[4] - np.deg2rad(0.5)
    _, _, before = self.repair._kinematics(x)
    y, ok, d = self.repair.process(x)
    self.assertTrue(ok.all())
    _, _, after = self.repair._kinematics(y)
    self.assertTrue(
      np.allclose(
        y[:, :, 38:53] - x[:, :, 38:53], (after - before).reshape(1, 10, 15), atol=1e-9
      )
    )
    self.assertTrue(np.array_equal(y[:, :, :9], x[:, :, :9]))
    self.assertTrue(np.array_equal(y[:, :, 53:], x[:, :, 53:]))
    self.assertTrue(np.isfinite(np.diff(y[:, :, 9:38], axis=1) / 0.02).all())
    self.assertGreater(d["repair_ee_delta_max_m"], 0)
    # Scalar clipping is non-expansive for each adjacent-frame angular difference.
    self.assertTrue(
      (
        np.abs(np.diff(y[:, :, 9:38], axis=1))
        <= np.abs(np.diff(x[:, :, 9:38], axis=1)) + 1e-7
      ).all()
    )

  def test_capsules_spheres_and_touch(self):
    a = np.array([[0.0, 0, 0]])
    b = np.array([[1.0, 0, 0]])
    self.assertAlmostEqual(
      float(segment_distance(a, b, a + [0, 2, 0], b + [0, 2, 0])[0]), 2
    )
    self.assertAlmostEqual(float(segment_distance(a, a, b, b)[0]), 1)
    self.assertAlmostEqual(
      float(segment_distance(a, b, a + [0.5, -1, 0], a + [0.5, 1, 0])[0]), 0
    )
    self.assertEqual(len(self.strict.pairs), 469)


suite = unittest.defaultTestLoader.loadTestsFromTestCase(PhysicsTests)
result = unittest.TextTestRunner(verbosity=2).run(suite)
if not result.wasSuccessful():
  raise SystemExit(1)
if args.gsi:
  x = np.load(args.gsi, allow_pickle=False)
  strict = PhysicalFilter(model, FilterConfig())
  repair = PhysicalFilter(model, FilterConfig(repair=True))
  original = x.copy()
  y, passed, stats = strict.process(x)
  assert np.array_equal(original, y) and np.array_equal(original, x)
  corrected, qualified, repair_stats = repair.process(x)
  assert strict.process(corrected[qualified])[1].all()
  assert np.array_equal(x[:, :, :9], corrected[:, :, :9]) and np.array_equal(
    x[:, :, 53:], corrected[:, :, 53:]
  )
  print(
    json.dumps(
      {
        "fixture": str(args.gsi),
        "strict_passed": int(passed.sum()),
        "repair_passed": int(qualified.sum()),
        "strict": stats,
        "repair": repair_stats,
      },
      indent=2,
    )
  )
print("GSI_PHYSICS_TESTS=PASS")
