"""00073 repeats 00033 with seed 2030 and its new paired baseline source."""

import json
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
REFERENCE = HERE.parent / "00033"


class VariantTests(unittest.TestCase):
  def test_training_configuration_only_changes_seed_and_paired_source(self):
    old = json.loads((REFERENCE / "EXPERIMENT.json").read_text())
    new = json.loads((HERE / "EXPERIMENT.json").read_text())
    self.assertEqual((old["seed"], new["seed"]), (2027, 2030))
    self.assertEqual(new["control"]["experiment"], "0007")
    self.assertEqual(new["control"]["seed"], 2030)
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

  def test_actual_search_filter_fifo_and_reward_sources_are_identical(self):
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
        (REFERENCE / name).read_text().replace("00033", "00073").replace("2027", "2030")
      )
      self.assertEqual((HERE / name).read_text(), expected, name)
    expected = (
      (REFERENCE / "run_00033.py")
      .read_text()
      .replace("00033", "00073")
      .replace("2027", "2030")
      .replace("2030 0002", "2030 0007")
    )
    self.assertEqual((HERE / "run_00073.py").read_text(), expected)


if __name__ == "__main__":
  unittest.main(verbosity=2)
