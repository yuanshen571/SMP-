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
    expected = expected.replace(
      "from initialization import load_common_initial, validate_control",
      "from initialization import load_common_initial, validate_startup_control",
    )
    expected = expected.replace(
      "  control_record = validate_control(args.initial_checkpoint, args.control_info)\n",
      "",
    )
    old_check = (
      "  assert (\n"
      '    control_record["initial_models_sha256"] == expected_control["initial_models_sha256"]\n'
      '  ), "must reuse the same untrained seed-2030 models as 0003"\n'
    )
    new_check = "  validate_startup_control(args.initial_checkpoint, args.control_info, expected_control)\n"
    self.assertIn(old_check, expected)
    expected = expected.replace(old_check, new_check)
    self.assertEqual((HERE / "run_00073.py").read_text(), expected)


if __name__ == "__main__":
  unittest.main(verbosity=2)
