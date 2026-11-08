"""Exercise actual fixed-quota collector and FIFO update without simulation."""

import __future__

import argparse
import ast
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from smp.rl.gsi_physics_runtime import collect, initialize  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=Path)
args = parser.parse_args()
if args.model is None:
  import mjlab

  args.model = Path(mjlab.__file__).parent / "asset_zoo/robots/unitree_g1/xmls/g1.xml"
spec = mujoco.MjSpec.from_file(str(args.model))
model = spec.compile()
if not (model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE).any():
  spec.worldbody.add_geom(
    name="test_ground", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.01]
  )
  model = spec.compile()


class Gate:
  lo = np.full(29, -1e9)
  hi = np.full(29, 1e9)
  config = SimpleNamespace(joint_tolerance_rad=1e-5)
  reject = False

  def process(self, x):
    mask = np.zeros(len(x), dtype=bool)
    mask[::2] = not self.reject
    return x.copy(), mask, {"generated": len(x), "qualified": int(mask.sum())}


def environment():
  env = SimpleNamespace(
    sim=SimpleNamespace(mj_model=model),
    cfg=SimpleNamespace(
      sim=SimpleNamespace(mujoco=SimpleNamespace(timestep=0.005)), decimation=4
    ),
    common_step_counter=2400,
  )
  initialize(env, repair=False, batch_size=4)
  env._smp_physics = Gate()
  env.calls = 0
  env._smp_gsi_pool = torch.full((8, 10, 59), -1.0)
  env._smp_gsi_repaired_rows = torch.ones(8, dtype=torch.bool)
  return env


def draw(env, n):
  env.calls += 1
  x = torch.zeros(n, 10, 59)
  x[:, 0, 0] = env.calls * 10 + torch.arange(n)
  return x


def refresh():
  tree = ast.parse((ROOT / "src/smp/rl/events.py").read_text(encoding="utf-8"))
  fn = next(
    n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "gsi_refresh"
  )
  fn.decorator_list = []
  module = ast.Module(body=[fn], type_ignores=[])
  namespace = {"torch": torch, "_ddpm_sample": draw}
  exec(
    compile(
      module, "actual_gsi_refresh", "exec", flags=__future__.annotations.compiler_flag
    ),
    namespace,
  )
  return namespace["gsi_refresh"]


class RuntimeTests(unittest.TestCase):
  def test_fill_and_surplus(self):
    env = environment()
    x = collect(env, 5, draw)
    self.assertEqual(x[:, 0, 0].tolist(), [10, 12, 20, 22, 30])
    self.assertEqual(env._smp_physics_totals["generated"], 12)
    self.assertEqual(env._smp_physics_totals["qualified"], 6)
    self.assertEqual(env._smp_physics_totals["inserted"], 5)
    self.assertEqual(env._smp_physics_totals["surplus"], 1)

  def test_failure_keeps_old_pool(self):
    env = environment()
    env._smp_physics.reject = True
    before = env._smp_gsi_pool.clone()
    with self.assertRaises(RuntimeError):
      collect(env, 5, draw)
    self.assertTrue(torch.equal(before, env._smp_gsi_pool))
    self.assertEqual(env._smp_physics_totals["inserted"], 0)

  def test_fifo_wrap_and_metadata(self):
    env = environment()
    env._smp_gsi_head = 6
    refresh()(env, num_samples=4)
    self.assertEqual(env._smp_gsi_head, 2)
    self.assertEqual(env._smp_gsi_pool[[6, 7, 0, 1], 0, 0].tolist(), [10, 12, 20, 22])
    self.assertTrue((env._smp_gsi_pool[2:6] == -1).all())
    self.assertTrue(env._smp_gsi_repaired_rows[2:6].all())
    self.assertFalse(env._smp_gsi_repaired_rows[[6, 7, 0, 1]].any())

  def test_no_refresh_outside_interval(self):
    env = environment()
    env.common_step_counter = 2399
    before = env._smp_gsi_pool.clone()
    refresh()(env, num_samples=4)
    self.assertEqual(env.calls, 0)
    self.assertTrue(torch.equal(before, env._smp_gsi_pool))


result = unittest.TextTestRunner(verbosity=2).run(
  unittest.defaultTestLoader.loadTestsFromTestCase(RuntimeTests)
)
if not result.wasSuccessful():
  raise SystemExit(1)
print("GSI_RUNTIME_TESTS=PASS")
