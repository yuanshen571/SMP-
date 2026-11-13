"""00083 repeats fixed 00073 with seed 2026 and its corresponding baseline."""

import json
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
REFERENCE = HERE.parent / "00073"


def translated(text):
  return text.replace("00073", "00083").replace("0007", "0008").replace("2030", "2026")


class VariantTests(unittest.TestCase):
  def test_configuration_only_changes_seed_and_common_source(self):
    old = json.loads((REFERENCE / "EXPERIMENT.json").read_text())
    new = json.loads((HERE / "EXPERIMENT.json").read_text())
    self.assertEqual((old["seed"], new["seed"]), (2030, 2026))
    self.assertEqual(new["control"]["experiment"], "0008")
    self.assertEqual(new["control"]["seed"], 2026)
    self.assertNotIn("initial_models_sha256", new["control"])
    for item in (old, new):
      for key in (
        "experiment",
        "branch",
        "seed",
        "only_training_change",
        "initialization",
        "control",
        "implementation_reference",
      ):
        item.pop(key, None)
      item["noise_rescue"].pop("seed_formula")
    self.assertEqual(old, new)

  def test_fixed_startup_and_training_sources_only_change_namespace_and_seed(self):
    for name in (
      "noise_rescue.py",
      "runtime.py",
      "events_adapter.py",
      "passive_metrics.py",
      "initialization.py",
      "test_startup.py",
      "test_runtime.py",
      "test_scoring.py",
      "BASELINE_FILES.json",
      "test_reward_formula.py",
      "vendor/_sepcma.py",
      "vendor/LICENSE",
    ):
      self.assertEqual(
        (HERE / name).read_text(), translated((REFERENCE / name).read_text()), name
      )
    self.assertEqual(
      (HERE / "run_00083.py").read_text(),
      translated((REFERENCE / "run_00073.py").read_text()),
    )


if __name__ == "__main__":
  unittest.main(verbosity=2)
