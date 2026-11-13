"""Keep the original additive baseline; only seed and experiment identity change."""

import json
import subprocess
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REFERENCE = "f03baed604b639d292986fa1b5f43c9ef82f87ae"


def reference(name):
  return subprocess.check_output(
    ["git", "show", f"{REFERENCE}:experiments/0006/{name}"], cwd=ROOT, text=True
  )


class VariantTests(unittest.TestCase):
  def test_training_config_keeps_original_method(self):
    old = json.loads(reference("EXPERIMENT.json"))
    new = json.loads((HERE / "EXPERIMENT.json").read_text())
    self.assertEqual(new["seed"], 2030)
    self.assertFalse(new["physical_filter_enabled"])
    self.assertEqual(new["repeats_experiment"], "0002")
    for item in (old, new):
      for key in (
        "experiment",
        "branch",
        "baseline_branch",
        "baseline_commit",
        "seed",
        "runtime_validation",
        "initialization",
        "only_training_change",
        "seed2028_reference",
        "implementation_reference",
      ):
        item.pop(key, None)
    self.assertEqual(old, new)

  def test_actual_runner_and_logging_only_change_seed_namespace_and_provenance(self):
    for old, new in [
      ("run_0006.py", "run_0007.py"),
      ("passive_metrics.py", "passive_metrics.py"),
      ("accept_0006.py", "accept_0007.py"),
      ("verify_completion.py", "verify_completion.py"),
      ("BASELINE_FILES.json", "BASELINE_FILES.json"),
    ]:
      expected = reference(old).replace("0006", "0007").replace("2029", "2030")
      if new == "run_0007.py":
        expected = expected.replace(
          "a0b4c8fdefbed56aba59efcdce3b83425ee6578e",
          "0a4ceff351b1f4885b61b6bfddee80ec49cb1874",
        )
      actual = (HERE / new).read_text()
      if new == "verify_completion.py":
        # Generated report instructions now point to the new paired queue;
        # every executable verification statement remains byte-identical.
        def omit_report_entry(text):
          start = text.index("\n## 运行入口")
          return text[:start] + text[text.index('\n"""', start) :]

        actual, expected = omit_report_entry(actual), omit_report_entry(expected)
      self.assertEqual(actual, expected, new)


if __name__ == "__main__":
  unittest.main(verbosity=2)
