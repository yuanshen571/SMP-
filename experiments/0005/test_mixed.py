"""CPU tests of exact admission, timing, reset dispatch and real-velocity restore."""

from __future__ import annotations
import __future__

import ast
import copy
import importlib.util
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from smp.rl.mixed_cache import (  # noqa: E402
  FRAME_DIM,
  META_COLUMNS,
  MixedConfig,
  WindowReservoir,
  digest,
  install,
  restore,
)


def payload(ends):
  x = np.zeros((len(ends), 10, FRAME_DIM), dtype=np.float32)
  x[:, :, 0] = ends[:, None] + np.arange(-9, 1)
  x[:, :, 2], x[:, :, 3], x[:, :, 42:71] = 1, 1, 7
  return x


def actual_reset():
  tree = ast.parse((ROOT / "src/smp/rl/events.py").read_text())
  fn = next(
    n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "gsi_reset"
  )
  fn.decorator_list = []
  ns = {"torch": torch, "_prime_sim_and_buffer": prime_gsi}
  exec(
    compile(
      ast.Module(body=[fn], type_ignores=[]),
      "actual_gsi_reset",
      "exec",
      flags=__future__.annotations.compiler_flag,
    ),
    ns,
  )
  return ns["gsi_reset"]


def prime_gsi(env, ids, windows):
  env.gsi_calls += len(ids)
  env.scene["robot"].data.joint_pos[ids] = windows[:, -1, 0, None]
  env.scene["robot"].data.joint_vel[ids] = -5


class Buffer:
  def __init__(self, n):
    self.raw = torch.zeros(n, 10, FRAME_DIM)

  def reset(self, ids, p, q, v, w, ee, jp, jv):
    self.raw[ids] = torch.cat([p, q, v, w, jp, jv, ee.flatten(2)], dim=-1)


class Robot:
  def __init__(self, n):
    self.data = SimpleNamespace(
      root_link_pos_w=torch.zeros(n, 3),
      root_link_quat_w=torch.tensor([[1.0, 0, 0, 0]]).repeat(n, 1),
      root_link_lin_vel_w=torch.zeros(n, 3),
      root_link_ang_vel_w=torch.zeros(n, 3),
      joint_pos=torch.zeros(n, 29),
      joint_vel=torch.zeros(n, 29),
      body_link_pos_w=torch.zeros(n, 5, 3),
      default_root_state=torch.zeros(n, 13),
    )
    self.data.default_root_state[:, 2:4] = 1

  def write_root_state_to_sim(self, state, env_ids):
    d = self.data
    d.root_link_pos_w[env_ids], d.root_link_quat_w[env_ids] = (
      state[:, :3],
      state[:, 3:7],
    )
    d.root_link_lin_vel_w[env_ids], d.root_link_ang_vel_w[env_ids] = (
      state[:, 7:10],
      state[:, 10:13],
    )

  def write_joint_state_to_sim(self, pos, vel, env_ids):
    self.data.joint_pos[env_ids], self.data.joint_vel[env_ids] = pos, vel


class Scene(dict):
  pass


class FakeEnv:
  def __init__(self, n=3):
    self.num_envs, self.device, self.step_dt = n, "cpu", 0.02
    self.max_episode_length, self.common_step_counter = 1000, 0
    self.scene = Scene(robot=Robot(n))
    self.scene.env_origins = torch.zeros(n, 3)
    self.scene.env_origins[:, 0] = torch.arange(n) * 100
    self._smp_ee_indexes = torch.arange(5)
    self._smp_buffer = Buffer(n)
    self._smp_gsi_pool = torch.zeros(4096, 10, 59)
    self._smp_gsi_pool[:, -1, 0] = torch.arange(4096)
    self.reset_terminated = torch.zeros(n, dtype=torch.bool)
    self.reset_time_outs = torch.zeros(n, dtype=torch.bool)
    self.age = torch.zeros(n, dtype=torch.long)
    self.periods = torch.tensor([80, 100, 120]).repeat((n + 2) // 3)[:n]
    self.gsi_calls, self.event_reset = 0, actual_reset()

  def _reset_idx(self, ids=None):
    if ids is None:
      ids = torch.arange(self.num_envs)
    self.event_reset(self, ids)
    self.age[ids] = 0

  def step(self, action=None):
    del action
    self.common_step_counter += 1
    s, d = self.common_step_counter, self.scene["robot"].data
    d.root_link_pos_w[:, 0] = self.scene.env_origins[:, 0] + s * 0.01
    d.root_link_pos_w[:, 2] = 1
    d.root_link_lin_vel_w[:, 0] = 0.5
    d.body_link_pos_w[:] = d.root_link_pos_w[:, None]
    d.joint_pos[:] = s * 0.001
    d.joint_vel[:] = 7 + s * 0.001  # Deliberately not the finite-difference value.
    self.age += 1
    self.reset_time_outs = self.age >= self.periods
    ids = self.reset_time_outs.nonzero().flatten()
    if len(ids):
      self._reset_idx(ids)
    return None


class MixedTests(unittest.TestCase):
  def test_final_mean_strict_and_window_provenance(self):
    r = WindowReservoir(1, 300, 5, 1000, np.random.default_rng(2))
    for env_id, length in enumerate([80, 140, 200]):
      r.add_episode(
        env_id=env_id, serial=0, start=0, end=length, failed=False, load=payload
      )
    data, meta, audit = r.finish()
    self.assertEqual(audit["mean_episode_steps"], 140)
    self.assertTrue((meta[:, 4] == 200).all())  # Exactly 140 is excluded.
    self.assertTrue((meta[:, 8] + 50 <= meta[:, 3]).all())
    self.assertTrue((meta[:, 7] >= 1).all())
    np.testing.assert_array_equal(data[:, :, 0], meta[:, 8, None] + np.arange(-9, 1))
    self.assertEqual(len(set(map(tuple, meta[:, [0, 1, 8]]))), 5)

  def test_reservoir_equals_offline_global_uniform_priorities(self):
    seed, quota = 123, 7
    rng = np.random.default_rng(seed)
    r = WindowReservoir(1, 400, quota, 1000, np.random.default_rng(seed))
    all_keys, all_rows = [], []
    for env_id, length in enumerate([65, 180, 90, 240, 180]):
      ends = np.arange(10, length - 50 + 1)
      keys = rng.random(len(ends))
      all_keys.extend(keys)
      all_rows.extend((env_id, int(end), length) for end in ends)
      r.add_episode(
        env_id=env_id, serial=1, start=0, end=length, failed=False, load=payload
      )
    _, meta, audit = r.finish()
    allowed = [
      i for i, row in enumerate(all_rows) if row[2] > audit["mean_episode_steps"]
    ]
    expected = sorted(allowed, key=lambda i: all_keys[i])[:quota]
    self.assertEqual(
      [(int(x[0]), int(x[8]), int(x[4])) for x in meta], [all_rows[i] for i in expected]
    )

  def test_failure_on_last_future_step_and_collection_boundary(self):
    for failed, last in [(False, 200), (True, 199)]:
      r = WindowReservoir(100, 300, 500, 1000, np.random.default_rng(1))
      r.add_episode(env_id=0, serial=1, start=50, end=250, failed=failed, load=payload)
      meta = r.buckets[200][2]
      self.assertEqual(int(meta[:, 8].min()), 109)
      self.assertEqual(int(meta[:, 8].max()), last)
      self.assertTrue((meta[:, 7] >= 100).all())

  def test_timing_episode_average_fifo_and_no_persistent_payloads(self):
    env = FakeEnv()
    cfg = MixedConfig(
      capacity=4,
      replace_count=2,
      collect_after_iteration=0,
      activate_after_iteration=10,
      update_interval=5,
    )
    with tempfile.TemporaryDirectory() as folder:
      path = Path(folder) / "mixed_cache_audit.json"
      cache = install(env, config=cfg, audit_path=path)
      for _ in range(239):
        env.step(None)
      self.assertFalse(cache.active)
      self.assertEqual(cache.stats["policy_reset_draws"], 0)
      env.step(None)
      self.assertTrue(cache.active)
      self.assertEqual(
        cache.transactions[0]["episode_histogram"], {"80": 2, "100": 1, "120": 1}
      )
      self.assertEqual(cache.transactions[0]["mean_episode_steps"], 95)
      before = cache.metadata.copy()
      for _ in range(120):
        env.step(None)
      a = cache.transactions[1]["replaced_slots"]
      untouched = [i for i in range(4) if i not in a]
      np.testing.assert_array_equal(cache.metadata[untouched], before[untouched])
      for _ in range(120):
        env.step(None)
      b = cache.transactions[2]["replaced_slots"]
      self.assertEqual(set(a) | set(b), set(range(4)))
      self.assertFalse(set(a) & set(b))
      for _ in range(120):
        env.step(None)
      self.assertEqual(cache.transactions[3]["replaced_slots"], a)
      self.assertEqual(cache.stats["inserted"], 10)
      self.assertEqual(cache.stats["pre_activation_policy_draws"], 0)
      self.assertEqual(
        [t["completed_iteration"] for t in cache.transactions], [10, 15, 20, 25]
      )
      audit = json.loads(path.read_text())
      self.assertEqual(audit["meta_columns"], META_COLUMNS)
      spec = importlib.util.spec_from_file_location(
        "mixed_audit_check", Path(__file__).with_name("check_mixed_audit.py")
      )
      check = importlib.util.module_from_spec(spec)
      spec.loader.exec_module(check)
      result = check.verify_audit(audit, cfg, 25, num_envs=3)
      self.assertEqual(result["inserted"], 10)
      damaged = copy.deepcopy(audit)
      damaged["transactions"][0]["selected_meta"][0][9] = 49
      with self.assertRaises(AssertionError):
        check.verify_audit(damaged, cfg, 25, num_envs=3)
      self.assertEqual(
        audit["transactions"][-1]["pool_metadata_sha256"], digest(cache.metadata)
      )
      self.assertEqual(
        [p.name for p in Path(folder).iterdir()], ["mixed_cache_audit.json"]
      )

  def test_pre_activation_global_rng_and_state_match_unmodified_path(self):
    cfg = MixedConfig(
      capacity=4,
      replace_count=2,
      collect_after_iteration=0,
      activate_after_iteration=10,
      update_interval=5,
    )
    states = []
    for enabled in (False, True):
      torch.manual_seed(2027)
      random.seed(77)
      np.random.seed(88)
      env = FakeEnv()
      if enabled:
        install(env, config=cfg)
      for _ in range(240):
        env.step(None)
      states.append(
        (
          torch.get_rng_state().clone(),
          env.scene["robot"].data.joint_pos.clone(),
          random.getstate(),
          np.random.get_state(),
        )
      )
    self.assertTrue(torch.equal(states[0][0], states[1][0]))
    self.assertTrue(torch.equal(states[0][1], states[1][1]))
    self.assertEqual(states[0][2], states[1][2])
    self.assertTrue(np.array_equal(states[0][3][1], states[1][3][1]))

  def test_actual_reset_mixture_and_true_velocity_partial_restore(self):
    env = FakeEnv(6)
    cfg = MixedConfig(
      capacity=4,
      replace_count=2,
      collect_after_iteration=0,
      activate_after_iteration=10,
      update_interval=5,
    )
    cache = install(env, config=cfg)
    for _ in range(240):
      env.step(None)
    before = env.scene["robot"].data.joint_vel.clone()
    windows = torch.from_numpy(payload(np.array([100, 200])))
    windows[:, :, 13:42] = 99  # No physics clamp or repair is allowed here.
    windows[:, :, 42:71] = 123
    restore(env, torch.tensor([1, 4]), windows)
    self.assertTrue((env.scene["robot"].data.joint_pos[[1, 4]] == 99).all())
    self.assertTrue((env.scene["robot"].data.joint_vel[[1, 4]] == 123).all())
    self.assertTrue(
      torch.equal(before[[0, 2, 3, 5]], env.scene["robot"].data.joint_vel[[0, 2, 3, 5]])
    )
    self.assertTrue((env._smp_buffer.raw[[1, 4], :, 42:71] == 123).all())
    self.assertTrue(
      torch.equal(
        env.scene["robot"].data.root_link_pos_w[[1, 4], :2],
        env.scene.env_origins[[1, 4], :2],
      )
    )
    n0 = cache.stats["policy_reset_draws"]
    for _ in range(3000):
      env.event_reset(env, torch.arange(6))
    fraction = (cache.stats["policy_reset_draws"] - n0) / 18000
    self.assertAlmostEqual(fraction, 0.1, delta=0.008)
    self.assertEqual(cache.stats["pre_activation_policy_draws"], 0)

  def test_formal_1000_1500_1750_2000_clock_and_real_frame_ring_wrap(self):
    env = FakeEnv()
    cache = install(env, config=MixedConfig())
    for step in range(1, 48001):
      env.step(None)
      if step == 24000:
        self.assertIsNone(cache.reservoir)
        self.assertIsNone(cache.history)
      if step == 35999:
        self.assertFalse(cache.active)
        self.assertEqual(cache.stats["policy_reset_draws"], 0)
      if step == 36000:
        self.assertTrue(cache.active)
        self.assertEqual(len(cache.pool), 456)
        self.assertEqual(cache.stats["policy_reset_draws"], 0)
    self.assertEqual(cache.stats["transactions"], 3)
    self.assertEqual(cache.stats["inserted"], 912)
    self.assertEqual(
      [x["completed_iteration"] for x in cache.transactions], [1500, 1750, 2000]
    )
    for row in cache.transactions:
      start, end = row["collection_start_step"], row["collection_end_step"]
      expected = {
        str(period): sum(
          start <= k * period <= end for k in range(2, end // period + 1)
        )
        for period in (80, 100, 120)
      }
      expected = {k: v for k, v in expected.items() if v}
      self.assertEqual(row["episode_histogram"], expected)
    ends = cache.metadata[:, 8, None] + np.arange(-9, 1)
    # Known raw post-step states prove that no reset/GSI frame or overwritten
    # history slot slipped into any of the final admitted ten-frame windows.
    np.testing.assert_allclose(
      cache.pool[:, :, 0].numpy(), ends * 0.01, atol=1e-4, rtol=0
    )
    np.testing.assert_allclose(
      cache.pool[:, :, 42].numpy(), 7 + ends * 0.001, atol=1e-5, rtol=0
    )
    self.assertEqual(cache.stats["pre_activation_policy_draws"], 0)

  def test_insufficient_mean_equal_and_invalid_control_rate_fail_explicitly(self):
    r = WindowReservoir(1, 300, 4, 1000, np.random.default_rng(1))
    for i in range(3):
      r.add_episode(env_id=i, serial=0, start=0, end=200, failed=False, load=payload)
    with self.assertRaises(RuntimeError):
      r.finish()  # strict > mean, never >=.
    env = FakeEnv()
    env.step_dt = 1 / 30
    with self.assertRaises(ValueError):
      install(env, config=MixedConfig())


if __name__ == "__main__":
  torch.set_num_threads(2)
  result = unittest.TextTestRunner(verbosity=2).run(
    unittest.defaultTestLoader.loadTestsFromTestCase(MixedTests)
  )
  if not result.wasSuccessful():
    raise SystemExit(1)
  print("MIXED_CACHE_TESTS=PASS")
