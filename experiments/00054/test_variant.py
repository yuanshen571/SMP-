"""00054 changes only failed-search admission relative to the pinned 00053."""

import ast
import json
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
REFERENCE = HERE.parent / "00053"


class VariantTests(unittest.TestCase):
  def test_training_settings_match_except_failed_admission_and_provenance(self):
    old = json.loads((REFERENCE / "EXPERIMENT.json").read_text(encoding="utf-8"))
    new = json.loads((HERE / "EXPERIMENT.json").read_text(encoding="utf-8"))
    self.assertEqual(old["seed"], new["seed"])
    self.assertEqual(new["seed"], 2028)
    self.assertEqual(old["control"], new["control"])
    self.assertEqual(
      new["noise_rescue"]["failed_admission"],
      "original unchanged after exhausted search",
    )
    self.assertEqual(new["baseline_branch"], "00053")
    self.assertEqual(new["baseline_commit"], "9cecb1244c3d83eb1e9c159682ea2cc591b3fd55")
    for item in (old, new):
      for key in (
        "experiment",
        "branch",
        "baseline_branch",
        "baseline_commit",
        "parent_experiment",
        "implementation_reference",
        "only_training_change",
      ):
        item.pop(key)
      item["noise_rescue"].pop("failed_admission")
    self.assertEqual(old, new)

  def test_native_sampler_decoder_scoring_and_search_config_unchanged(self):
    def definitions(path):
      tree = ast.parse(path.read_text(encoding="utf-8").replace("00054", "00053"))
      return {
        node.name: ast.dump(node, include_attributes=False)
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
      }

    old = definitions(REFERENCE / "noise_rescue.py")
    new = definitions(HERE / "noise_rescue.py")
    for name in ("decode", "draw_recorded", "PhysicalScorer", "SearchConfig"):
      self.assertEqual(old[name], new[name], name)

  def test_filter_reset_fifo_rewards_metrics_and_initialization_unchanged(self):
    for name in (
      "BASELINE_FILES.json",
      "events_adapter.py",
      "passive_metrics.py",
      "initialization.py",
      "test_reward_formula.py",
      "accept_sampler.py",
      "vendor/_sepcma.py",
      "vendor/LICENSE",
    ):
      old = (REFERENCE / name).read_text(encoding="utf-8").replace("00053", "00054")
      self.assertEqual((HERE / name).read_text(encoding="utf-8"), old, name)


if __name__ == "__main__":
  unittest.main(verbosity=2)
