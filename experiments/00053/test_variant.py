"""Verify that 00053 preserves the 00063 method apart from seed and budget."""

import json
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
REFERENCE = HERE.parent / "00063"


class VariantTests(unittest.TestCase):
  def test_training_settings_match_reference_except_explicit_changes(self):
    old = json.loads((REFERENCE / "EXPERIMENT.json").read_text(encoding="utf-8"))
    new = json.loads((HERE / "EXPERIMENT.json").read_text(encoding="utf-8"))
    self.assertEqual((old["seed"], new["seed"]), (2029, 2028))
    self.assertEqual(
      (old["noise_rescue"]["max_generations"], new["noise_rescue"]["max_generations"]),
      (60, 30),
    )
    self.assertEqual(new["control"]["experiment"], "0005")
    self.assertEqual(new["control"]["seed"], 2028)
    self.assertEqual(
      new["control"]["initial_models_sha256"],
      "03f0e7c818c5203ccb1945d0fdb0369e108ece9522f853f99d4057ed999c84c8",
    )
    for item in (old, new):
      for key in (
        "experiment",
        "branch",
        "baseline_branch",
        "baseline_commit",
        "seed",
        "only_training_change",
        "initialization",
        "control",
        "parent_experiment",
        "implementation_reference",
      ):
        item.pop(key, None)
      item["noise_rescue"].pop("max_generations")
      item["noise_rescue"].pop("seed_formula")
    self.assertEqual(old, new)

  def test_noise_engine_changes_only_namespace_and_generation_budget(self):
    old = (REFERENCE / "noise_rescue.py").read_text(encoding="utf-8")
    expected = old.replace("00063", "00053")
    expected = expected.replace(
      "max_generations: int = 60", "max_generations: int = 30"
    )
    expected = expected.replace(
      "assert self.max_generations == 60", "assert self.max_generations == 30"
    )
    self.assertEqual((HERE / "noise_rescue.py").read_text(encoding="utf-8"), expected)

  def test_filter_reset_fifo_and_reward_source_are_identical(self):
    for name in (
      "BASELINE_FILES.json",
      "events_adapter.py",
      "passive_metrics.py",
      "test_reward_formula.py",
      "vendor/_sepcma.py",
      "vendor/LICENSE",
    ):
      old = (REFERENCE / name).read_text(encoding="utf-8").replace("00063", "00053")
      self.assertEqual((HERE / name).read_text(encoding="utf-8"), old, name)


if __name__ == "__main__":
  unittest.main(verbosity=2)
