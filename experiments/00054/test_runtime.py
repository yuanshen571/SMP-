"""CPU protocol checks: real sampler/CMA, retention, early stop and FIFO."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import events_adapter  # noqa: E402
import noise_rescue as nr  # noqa: E402
import runtime  # noqa: E402

from smp.pretrain.scheduler import DDPMScheduler  # noqa: E402

torch.set_num_threads(1)


class LinearDenoiser(torch.nn.Module):
  def forward(self, x, t):
    return x * 0.01 + t[:, None, None] * 0.00001


def bundle():
  return (
    LinearDenoiser().eval(),
    DDPMScheduler(50),
    torch.tensor(-1.0),
    torch.tensor(1.0),
    59,
    10,
  )


def arrays(B=64):
  raw = torch.zeros(B, 10, 59)
  return raw, raw.clone(), torch.zeros(49, B, 10, 59)


def identity_decode(_bundle, initial, posterior):
  assert posterior.shape == (49, *initial.shape)
  return initial.clone()


class NeverPass:
  def evaluate(self, windows):
    # Worse candidates must not replace the original after an exhausted search.
    return dict(
      passed=np.zeros(len(windows), dtype=bool),
      fitness=10000 + 1 + np.abs(windows[:, 0, 0]),
    )


class EventuallyPass:
  def evaluate(self, windows):
    passed = windows[:, 0, 0] >= 0.015
    return dict(passed=passed, fitness=np.where(passed, 0.0, 10001.0))


class ScriptedOptimizer:
  mode = "worse"

  def __init__(self, mean, sigma, bounds, seed, population_size):
    self.dim, self.P, self.calls, self.generation = len(mean), population_size, 0, 0
    assert sigma == 0.05 and bounds.shape == (590, 2)

  def ask(self):
    i = self.calls % self.P
    self.calls += 1
    value = 0.01
    if self.mode == "early" and self.generation >= 1:
      value = 0.02
    if self.mode == "radius" and i % 2 == 0:
      value = 0.16
    if self.mode == "all_radius":
      value = 0.16
    if self.mode == "best_early" and self.generation > 0:
      value = 0.04
    if self.mode == "tie":
      value = 0.04 if i == 3 else (0.02 if i == 5 else 0.01)
    return np.full(self.dim, value)

  def tell(self, population):
    assert len(population) == self.P
    self.generation += 1


class SearchTests(unittest.TestCase):
  def test_recorded_sampler_matches_native_formula_and_rng(self):
    b = bundle()
    torch.manual_seed(2028)
    initial = torch.randn(64, 10, 59)
    x = initial
    for t in reversed(range(50)):
      ts = torch.full((64,), t, dtype=torch.long)
      x = b[1].step(b[0](x, ts), x, t)
    native_state = torch.get_rng_state().clone()
    torch.manual_seed(2028)
    raw, z0, noises = nr.draw_recorded(b, 64, "cpu")
    expected = (x + 1.0) / 2.0 * (b[3] - b[2]) + b[2]
    self.assertTrue(torch.equal(raw, expected))
    self.assertTrue(torch.equal(z0, initial))
    self.assertTrue(torch.equal(native_state, torch.get_rng_state()))
    before = noises.clone()
    rng = torch.get_rng_state().clone()
    replay = nr.decode(b, z0, noises)
    self.assertTrue(torch.equal(replay, raw))
    self.assertTrue(torch.equal(noises, before))
    self.assertTrue(torch.equal(rng, torch.get_rng_state()))

  def run_search(self, gate, factory=ScriptedOptimizer, failed=(1, 2)):
    raw, initial, noises = arrays()
    passed = np.ones(64, dtype=bool)
    passed[list(failed)] = False
    with patch.object(nr, "decode", identity_decode):
      out, mask, stats = nr.search_batch(
        None,
        raw,
        initial,
        noises,
        passed,
        gate,
        seed=2028,
        global_start=1024,
        config=nr.SearchConfig(inference_batch_size=64),
        optimizer_factory=factory,
        log=None,
      )
    return raw.numpy(), out, mask, stats

  def test_failed_search_keeps_original_even_when_candidates_are_worse(self):
    ScriptedOptimizer.mode = "worse"
    raw, out, passed, stats = self.run_search(NeverPass())
    self.assertEqual(stats["attempted"], 2)
    self.assertEqual(stats["exhausted"], 2)
    self.assertEqual(stats["succeeded"], 0)
    self.assertEqual(stats["proposals"], 2 * 30 * 32)
    self.assertEqual(out.tobytes(), raw.tobytes())
    self.assertFalse(passed[[1, 2]].any())
    self.assertTrue(np.array_equal(out[[0, 3]], raw[[0, 3]]))
    self.assertEqual(stats["replay_max_abs"], 0)
    self.assertEqual(stats["original_fallback_inserted"], 2)
    self.assertEqual(stats["replay_decode_calls"], 0)

  def test_stop_at_first_passing_generation(self):
    ScriptedOptimizer.mode = "early"
    _, out, passed, stats = self.run_search(EventuallyPass())
    self.assertEqual(stats["generations"], 4)
    self.assertEqual(stats["proposals"], 4 * 32)
    self.assertEqual(stats["succeeded"], 2)
    self.assertEqual(stats["success_generation_max"], 2)
    self.assertEqual(stats["original_fallback_inserted"], 0)
    self.assertTrue(passed.all())
    self.assertTrue(np.allclose(out[[1, 2]], 0.02))

  def test_neither_history_best_nor_last_failed_generation_is_admitted(self):
    ScriptedOptimizer.mode = "best_early"
    raw, out, _, stats = self.run_search(NeverPass(), failed=(1,))
    self.assertEqual(stats["exhausted"], 1)
    self.assertEqual(out[1].tobytes(), raw[1].tobytes())

  def test_first_passing_candidate_wins_without_noise_distance_tiebreak(self):
    ScriptedOptimizer.mode = "tie"
    _, out, mask, stats = self.run_search(EventuallyPass(), failed=(1,))
    self.assertTrue(mask.all())
    self.assertEqual(stats["generations"], 1)
    self.assertTrue(np.allclose(out[1], 0.04))

  def test_radius_rejects_candidates_without_changing_decoder_shape(self):
    ScriptedOptimizer.mode = "radius"
    _, out, _, stats = self.run_search(NeverPass(), failed=(1,))
    self.assertEqual(stats["radius_rejected"], 30 * 16)
    self.assertEqual(stats["effective_candidates"], 30 * 16)
    self.assertEqual(stats["search_decoded_slots"], 30 * 64)
    self.assertTrue((out[1] == 0).all())
    self.assertLessEqual(stats["max_selected_rms"], 0.15)

  def test_all_radius_rejections_still_search_full_budget_then_keep_original(self):
    ScriptedOptimizer.mode = "all_radius"
    raw, out, mask, stats = self.run_search(NeverPass(), failed=(1,))
    self.assertFalse(mask[1])
    self.assertEqual(stats["proposals"], 30 * 32)
    self.assertEqual(stats["radius_rejected"], 30 * 32)
    self.assertEqual(stats["effective_candidates"], 0)
    self.assertEqual(stats["original_fallback_inserted"], 1)
    self.assertEqual(out.tobytes(), raw.tobytes())

  def test_mixed_success_and_fallback_preserve_nonzero_complete_originals(self):
    raw, initial, noises = arrays()
    raw[:] = torch.arange(raw.numel(), dtype=torch.float32).reshape_as(raw) / 1e6
    raw[2, 0, 0] = -0.0
    saved = raw.clone()
    flags = np.ones(64, dtype=bool)
    flags[[1, 2]] = False
    initial[1] = 0.01
    initial[2] = -0.1
    ScriptedOptimizer.mode = "worse"
    with patch.object(nr, "decode", identity_decode):
      out, mask, stats = nr.search_batch(
        None,
        raw,
        initial,
        noises,
        flags,
        EventuallyPass(),
        seed=2028,
        global_start=0,
        config=nr.SearchConfig(inference_batch_size=64),
        optimizer_factory=ScriptedOptimizer,
        log=None,
      )
    self.assertTrue(mask[1])
    self.assertFalse(mask[2])
    self.assertEqual(stats["succeeded"], 1)
    self.assertEqual(stats["exhausted"], 1)
    self.assertEqual(stats["original_fallback_inserted"], 1)
    self.assertEqual(stats["proposals"], (1 + 30) * 32)
    self.assertEqual(out[2].tobytes(), saved[2].numpy().tobytes())
    self.assertEqual(out[flags].tobytes(), saved.numpy()[flags].tobytes())
    self.assertTrue(np.allclose(out[1], 0.02))
    self.assertEqual(raw.numpy().tobytes(), saved.numpy().tobytes())

  def test_already_valid_rows_do_not_start_search(self):
    _, out, passed, stats = self.run_search(NeverPass(), failed=())
    self.assertTrue(passed.all())
    self.assertEqual(stats["attempted"], 0)
    self.assertEqual(stats["search_decode_calls"], 0)
    self.assertTrue((out == 0).all())

  def test_real_sep_cma_does_not_consume_global_rng(self):
    torch.manual_seed(91)
    np.random.seed(92)
    before_torch = torch.get_rng_state().clone()
    before_numpy = np.random.get_state()
    _, out, _, stats = self.run_search(NeverPass(), factory=nr.SepCMA, failed=(1,))
    after_numpy = np.random.get_state()
    self.assertTrue(torch.equal(before_torch, torch.get_rng_state()))
    self.assertTrue(np.array_equal(before_numpy[1], after_numpy[1]))
    self.assertEqual(before_numpy[2:], after_numpy[2:])
    self.assertEqual(stats["proposals"], 30 * 32)
    self.assertTrue(np.isfinite(out).all())

  def test_refill_jobs_are_independent_of_other_cases(self):
    ScriptedOptimizer.mode = "early"
    _, out, mask, stats = self.run_search(EventuallyPass(), failed=(1, 2, 3, 4, 5))
    self.assertTrue(mask.all())
    self.assertEqual(stats["attempted"], 5)
    self.assertEqual(stats["proposals"], 5 * 2 * 32)
    self.assertTrue(np.allclose(out[[1, 2, 3, 4, 5]], 0.02))


class CollectorTests(unittest.TestCase):
  def test_configure_changes_only_startup_and_refresh_callbacks(self):
    marker = object()
    params = dict(
      gsi_batch_size=1024, gsi_buffer_size=4096, compile_mode="max-autotune"
    )
    refresh_params = dict(num_samples=1024, step_interval=2400)
    startup = SimpleNamespace(func=marker, mode="startup", params=params.copy())
    refresh = SimpleNamespace(
      func=marker, mode="interval", params=refresh_params.copy()
    )
    untouched = SimpleNamespace(func=marker, params={"a": 1})
    cfg = SimpleNamespace(
      events={"init_smp_state": startup, "gsi_refresh": refresh, "reset": untouched},
      rewards=marker,
      observations=marker,
      terminations=marker,
    )
    events_adapter.configure(cfg)
    self.assertIs(startup.func, events_adapter.init_smp_state)
    self.assertIs(refresh.func, events_adapter.gsi_refresh)
    self.assertEqual(
      startup.params, dict(params, physical_filter=True, repair_small_joints=False)
    )
    self.assertEqual(refresh.params, refresh_params)
    self.assertEqual(startup.mode, "startup")
    self.assertEqual(refresh.mode, "interval")
    self.assertIs(cfg.events["reset"], untouched)
    self.assertIs(cfg.rewards, marker)
    self.assertIs(cfg.observations, marker)
    self.assertIs(cfg.terminations, marker)

  def collect_fixture(self, corrupt=False):
    physics = SimpleNamespace(
      process=lambda raw, repair: (
        raw.copy(),
        np.arange(len(raw)) % 2 == 0,
        dict(generated=len(raw), qualified=len(raw) // 2),
      )
    )
    env = SimpleNamespace(
      _smp_noise_config=nr.SearchConfig(inference_batch_size=64),
      _smp_bundle=None,
      _smp_physics=physics,
      _smp_physics_totals=dict(updates=0, generated=0, qualified=0, inserted=0),
      _smp_noise_totals={},
      cfg=SimpleNamespace(seed=2028),
      device="cpu",
      common_step_counter=0,
    )
    env._smp_noise_totals = dict(
      original_rows=0,
      direct_inserted=0,
      inserted=0,
      raw_decode_calls=0,
      generation_seconds=0.0,
      collection_seconds=0.0,
      original_fallback_exact_verified=0,
      **nr._initial_stats(),
    )
    calls = []

    def draw(_bundle, n, device):
      calls.append(n)
      raw, initial, posterior = arrays(n)
      raw[:] = torch.arange(raw.numel(), dtype=torch.float32).reshape_as(raw) / 1e6
      return raw, initial, posterior

    def rescue(_bundle, raw, initial, posterior, passed, scorer, **kwargs):
      output = raw.numpy().copy()
      if corrupt:
        output[1, 0, 0] += 0.01
      stats = nr._initial_stats()
      stats.update(
        attempted=32,
        exhausted=32,
        original_fallback_inserted=32,
        proposals=32 * 30 * 32,
        generations=32 * 30,
        effective_candidates=32 * 30 * 32,
      )
      return output, passed.copy(), stats

    with (
      patch.object(runtime, "draw_recorded", draw),
      patch.object(runtime, "search_batch", rescue),
    ):
      x = runtime.collect(env, 128)
    return env, x, calls

  def test_exact_original_counts_all_failed_originals_retained(self):
    env, x, calls = self.collect_fixture()
    self.assertEqual(calls, [64, 64])
    self.assertEqual(len(x), 128)
    expected = torch.arange(64 * 10 * 59, dtype=torch.float32).reshape(64, 10, 59) / 1e6
    self.assertEqual(
      x.numpy().tobytes(), torch.cat([expected, expected]).numpy().tobytes()
    )
    self.assertEqual(env._smp_noise_totals["original_rows"], 128)
    self.assertEqual(env._smp_noise_totals["exhausted"], 64)
    self.assertEqual(env._smp_noise_totals["original_fallback_inserted"], 64)
    self.assertEqual(env._smp_noise_totals["original_fallback_exact_verified"], 64)
    self.assertEqual(env._smp_physics_totals["inserted"], 128)

  def test_collector_rejects_changed_historical_frame_on_fallback(self):
    with self.assertRaisesRegex(AssertionError, "complete original window"):
      self.collect_fixture(corrupt=True)

  def test_fifo_wrap_and_nonrefresh_steps(self):
    env = SimpleNamespace(
      common_step_counter=2400,
      _smp_gsi_pool=torch.full((128, 10, 59), -1.0),
      _smp_gsi_repaired_rows=torch.zeros(128, dtype=torch.bool),
      _smp_gsi_original_pass=torch.ones(128, dtype=torch.bool),
      _smp_gsi_physical_pass=torch.ones(128, dtype=torch.bool),
      _smp_gsi_head=100,
    )

    def collect(e, count):
      e._smp_physics_last_repaired = torch.zeros(count, dtype=torch.bool)
      e._smp_noise_last_original_pass = torch.zeros(count, dtype=torch.bool)
      e._smp_noise_last_final_pass = torch.zeros(count, dtype=torch.bool)
      return torch.ones(count, 10, 59)

    with patch.object(runtime, "collect", collect):
      events_adapter.gsi_refresh(env, num_samples=64)
      self.assertEqual(env._smp_gsi_head, 36)
      self.assertTrue((env._smp_gsi_pool[36:100] == -1).all())
      self.assertTrue((env._smp_gsi_pool[:36] == 1).all())
      self.assertTrue((env._smp_gsi_pool[100:] == 1).all())
      self.assertTrue(env._smp_gsi_physical_pass[36:100].all())
      self.assertFalse(env._smp_gsi_physical_pass[:36].any())
      before = env._smp_gsi_pool.clone()
      env.common_step_counter = 2401
      events_adapter.gsi_refresh(env, num_samples=64)
      self.assertTrue(torch.equal(before, env._smp_gsi_pool))


if __name__ == "__main__":
  unittest.main(verbosity=2)
