"""Execute the actual completion-check block with controlled admission ledgers."""

import ast
import copy
import json
import unittest
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent


def actual_checks(info):
  tree = ast.parse((HERE / "verify_completion.py").read_text(encoding="utf-8"))
  main = next(
    node
    for node in tree.body
    if isinstance(node, ast.FunctionDef) and node.name == "main"
  )
  active, nodes = False, []
  for node in main.body:
    if isinstance(node, ast.Assign):
      names = [n.id for n in node.targets if isinstance(n, ast.Name)]
      if "noise" in names:
        active = True
      if active and "common" in names:
        break
    if active:
      nodes.append(node)
  assert nodes
  rows = []
  scope = dict(
    info=info,
    totals=dict(qualified=3000),
    expected_insertions=4096,
    spec=json.loads((HERE / "EXPERIMENT.json").read_text()),
    report=SimpleNamespace(check=lambda name, ok: rows.append((name, bool(ok)))),
  )
  exec(
    compile(
      ast.Module(body=nodes, type_ignores=[]), "actual_completion_checks", "exec"
    ),
    scope,
  )
  return rows


def ledger():
  return dict(
    failed_admission="original unchanged after exhausted search",
    noise_rescue_config=dict(
      rms_limit=0.15,
      max_generations=30,
      population_size=32,
      sigma0=0.05,
      inference_batch_size=1024,
    ),
    noise_rescue_totals=dict(
      original_rows=4096,
      inserted=4096,
      direct_inserted=3000,
      attempted=1096,
      succeeded=200,
      exhausted=896,
      original_fallback_inserted=896,
      original_fallback_exact_verified=896,
      discarded=0,
      generations=896 * 30 + 200,
      success_generation_sum=200,
      proposals=(896 * 30 + 200) * 32,
      max_selected_rms=0.1,
      replay_max_abs=0,
    ),
  )


class CompletionPolicyTests(unittest.TestCase):
  def test_exact_original_fallback_is_accepted_even_with_failed_states_in_pool(self):
    rows = actual_checks(ledger())
    self.assertEqual(len(rows), 3)
    self.assertTrue(all(ok for _, ok in rows), rows)

  def test_old_best_candidate_policy_or_unverified_fallback_is_rejected(self):
    for edits in (
      dict(original_fallback_inserted=0),
      dict(original_fallback_exact_verified=895),
      dict(discarded=1),
      dict(exhausted=895),
    ):
      info = ledger()
      info["noise_rescue_totals"].update(edits)
      self.assertFalse(all(ok for _, ok in actual_checks(info)))
    info = ledger()
    info["failed_admission"] = "best search candidate only; original is not a fallback"
    self.assertFalse(all(ok for _, ok in actual_checks(info)))

  def test_exhausted_searches_cannot_skip_the_full_30_generation_budget(self):
    info = copy.deepcopy(ledger())
    info["noise_rescue_totals"]["generations"] -= 1
    info["noise_rescue_totals"]["proposals"] -= 32
    self.assertFalse(all(ok for _, ok in actual_checks(info)))


if __name__ == "__main__":
  unittest.main(verbosity=2)
