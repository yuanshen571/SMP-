"""Exercise 00094 decisions with scripted generation outcomes, no simulator/GPU."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import noise_rescue as nr
import numpy as np
from test_search import ScriptedOptimizer, arrays, identity_decode
from verify_pairing import verify_decisions


class GenerationOptimizer(ScriptedOptimizer):
  def ask(self):
    i = self.calls % self.P
    self.calls += 1
    return np.full(self.dim, (self.generation + 1) * 0.001 + i * 0.000001)


class GateScorer:
  def __init__(self, joint, *, success_at=None, invalid_first=False):
    self.joint = joint
    self.success_at = success_at
    self.invalid_first = invalid_first

  def evaluate(self, windows):
    code = windows[:, 0, 0]
    generation = np.rint(code * 1000).astype(int)
    joint = np.asarray(self.joint(generation), dtype=float)
    numeric = np.ones(len(windows), dtype=bool)
    if self.invalid_first:
      numeric[::32] = False
      joint[::32] = 0
    passed = np.zeros(len(windows), dtype=bool)
    if self.success_at is not None:
      passed = generation >= self.success_at
    # Prefer a later generation by COMBINED fitness, independently of minimum
    # joint excess. This deliberately separates stopping and winner selection.
    fitness = np.where(passed, 0, 10002.0 - code)
    fitness[~numeric] = 1e12
    return dict(passed=passed, numeric=numeric, joint=joint, fitness=fitness)


class PolicyTests(unittest.TestCase):
  def run_case(self, joint0, scorer, *, factory=GenerationOptimizer):
    raw, initial, posterior = arrays()
    passed = np.ones(64, dtype=bool)
    passed[1] = False
    joints = np.zeros(64)
    joints[1] = joint0
    decisions = []
    with patch.object(nr, "decode", identity_decode):
      out, mask, stats = nr.search_batch(
        None,
        raw,
        initial,
        posterior,
        passed,
        scorer,
        seed=2028,
        global_start=1024,
        config=nr.SearchConfig(inference_batch_size=64),
        original_metrics=dict(passed=passed, joint=joints),
        decisions=decisions,
        optimizer_factory=factory,
        log=None,
      )
    verify_decisions(
      dict(
        stats=stats,
        decisions=sorted(decisions, key=lambda d: d["original_row"]),
        global_row_start=1024,
      )
    )
    self.assertTrue(np.array_equal(out[passed], raw.numpy()[passed]))
    self.assertEqual(stats["attempted"] + stats["skipped_original_inserted"], 1)
    return out, mask, stats, decisions[0]

  def test_above_042_skips_all_search_and_keeps_original_invalid(self):
    class Forbidden:
      def __init__(self, **kwargs):
        raise AssertionError("a skipped row must not allocate an optimizer")

    out, mask, s, d = self.run_case(np.deg2rad(0.420001), None, factory=Forbidden)
    self.assertTrue((out == 0).all())
    self.assertFalse(mask[1])
    self.assertEqual(
      (s["attempted"], s["skipped_original_inserted"], s["search_decode_calls"]),
      (0, 1, 0),
    )
    self.assertTrue(d["selected_original"])

  def test_exact_042_still_enters_search(self):
    _, _, s, _ = self.run_case(np.deg2rad(0.42), GateScorer(lambda g: np.zeros(len(g))))
    self.assertEqual(
      (s["attempted"], s["skipped_original_inserted"], s["exhausted"]), (1, 0, 1)
    )

  def test_under_15_percent_stops_at10_with_best_combined_fitness(self):
    out, _, s, d = self.run_case(0.006, GateScorer(lambda g: np.full(len(g), 0.0054)))
    self.assertEqual(
      (s["early_stopped"], s["generations"], s["proposals"]), (1, 10, 320)
    )
    self.assertEqual(s["joint_improvement_checked"], 1)
    self.assertAlmostEqual(d["improvement_ratio"], 0.1)
    self.assertGreater(out[1, 0, 0], 0.009)
    self.assertFalse(d["selected_original"])

  def test_exact_15_percent_continues_to30(self):
    _, _, s, d = self.run_case(
      0.006, GateScorer(lambda g: np.full(len(g), 0.006 * 0.85))
    )
    self.assertEqual((s["early_stopped"], s["exhausted"], s["generations"]), (0, 1, 30))
    self.assertEqual(s["joint_improvement_continued"], 1)
    self.assertAlmostEqual(d["improvement_ratio"], 0.15)

  def test_gate_uses_best_across_first10_not_only_generation10(self):
    out, _, s, d = self.run_case(
      0.006, GateScorer(lambda g: np.where(g == 1, 0.004, 0.0059))
    )
    self.assertEqual(s["generations"], 30)
    self.assertAlmostEqual(d["best_joint_first10_rad"], 0.004)
    self.assertGreater(out[1, 0, 0], 0.029)

  def test_gate_is_not_repeated_after_generation10(self):
    _, _, s, d = self.run_case(
      0.006, GateScorer(lambda g: np.where(g <= 10, 0.004, 0.00599))
    )
    self.assertEqual((s["joint_improvement_checked"], s["generations"]), (1, 30))
    self.assertEqual(d["reason"], "generation_limit")

  def test_joint_valid_rows_are_exempt_even_at_positive_tolerance(self):
    for j in (0.0, 1e-5):
      with self.subTest(j=j):
        _, _, s, d = self.run_case(j, GateScorer(lambda g: np.full(len(g), 0.02)))
        self.assertEqual(
          (s["joint_gate_exempt"], s["joint_improvement_checked"], s["generations"]),
          (1, 0, 30),
        )
        self.assertFalse(d["gate_applied"])

  def test_physical_success_has_priority_at_generation10(self):
    _, mask, s, d = self.run_case(
      0.006, GateScorer(lambda g: np.full(len(g), 0.0059), success_at=10)
    )
    self.assertTrue(mask.all())
    self.assertEqual(
      (s["succeeded"], s["generations"], s["joint_improvement_checked"]), (1, 10, 0)
    )
    self.assertEqual(d["reason"], "physical_success")

  def test_nonnumeric_low_joint_candidate_cannot_improve_gate(self):
    _, _, s, _ = self.run_case(
      0.006, GateScorer(lambda g: np.full(len(g), 0.0059), invalid_first=True)
    )
    self.assertEqual(s["early_stopped"], 1)

  def test_radius_rejected_candidate_cannot_improve_gate(self):
    ScriptedOptimizer.mode = "radius"
    _, _, s, _ = self.run_case(
      0.006, GateScorer(lambda g: np.full(len(g), 0.0059)), factory=ScriptedOptimizer
    )
    self.assertEqual(s["early_stopped"], 1)
    self.assertEqual(s["radius_rejected"], 160)

  def test_bad_decision_audit_is_rejected(self):
    _, _, s, d = self.run_case(0.006, GateScorer(lambda g: np.full(len(g), 0.0054)))
    d["generation"] = 9
    with self.assertRaises(AssertionError):
      verify_decisions(dict(stats=s, decisions=[d], global_row_start=1024))


if __name__ == "__main__":
  unittest.main(verbosity=2)
