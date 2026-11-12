"""00033 changes only the training seed and its paired source, versus 00053."""

import json
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
REFERENCE = HERE.parent / "00053"


class VariantTests(unittest.TestCase):
  def test_training_settings_only_seed_changes(self):
    old = json.loads((REFERENCE / "EXPERIMENT.json").read_text())
    new = json.loads((HERE / "EXPERIMENT.json").read_text())
    self.assertEqual((old["seed"], new["seed"]), (2028, 2027))
    self.assertEqual(
      old["noise_rescue"]["max_generations"], new["noise_rescue"]["max_generations"]
    )
    self.assertEqual(new["noise_rescue"]["max_generations"], 30)
    self.assertEqual(new["control"]["experiment"], "0002")
    self.assertEqual(
      new["control"]["initial_models_sha256"],
      "774d69745290cdcce1522fe5875d48022cc4789f1ab47abb386bb45a19fec4eb",
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
      item["noise_rescue"].pop("seed_formula")
    self.assertEqual(old, new)

  def test_training_method_source_is_identical(self):
    for name in (
      "noise_rescue.py",
      "runtime.py",
      "events_adapter.py",
      "passive_metrics.py",
      "test_runtime.py",
      "test_scoring.py",
      "BASELINE_FILES.json",
      "test_reward_formula.py",
      "vendor/_sepcma.py",
      "vendor/LICENSE",
    ):
      expected = (
        (REFERENCE / name)
        .read_text()
        .replace("00053", "00033")
        .replace("00052", "0003")
        .replace("2028", "2027")
      )
      self.assertEqual((HERE / name).read_text(), expected, name)


if __name__ == "__main__":
  unittest.main(verbosity=2)
