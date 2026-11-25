"""Verify 00097 has unchanged 00096 method code, data and training settings."""

import ast
import hashlib
import json
import subprocess
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = HERE.parent / "00096"


class RepeatContract(unittest.TestCase):
  def test_unchanged_method_settings(self):
    new = json.loads((HERE / "EXPERIMENT.json").read_text())
    old = json.loads((OLD / "EXPERIMENT.json").read_text())
    changed = {
      "experiment",
      "branch",
      "baseline_branch",
      "parent_experiment",
      "repeats_experiment",
      "only_training_change",
      "noise_engine_sha256",
      "repeat_protocol",
    }
    self.assertEqual(
      {k: v for k, v in new.items() if k not in changed},
      {k: v for k, v in old.items() if k not in changed},
    )
    self.assertEqual(new["parent_experiment"], "00096")
    self.assertEqual(new["repeats_experiment"], "00096")
    self.assertEqual(
      new["noise_engine_sha256"],
      hashlib.sha256((HERE / "noise_rescue.py").read_bytes()).hexdigest(),
    )

  def test_algorithm_sources_equal_except_identity(self):
    for name in (
      "noise_rescue.py",
      "paired_replay.py",
      "initialization.py",
      "archive_format.py",
      "verify_source_archive.py",
      "gpu_filter.py",
      "gpu_scorer.py",
      "gsi_physics.py",
      "passive_metrics.py",
      "accept_sampler.py",
      "validate_source.py",
    ):
      self.assertEqual(
        (HERE / name).read_text(),
        (OLD / name).read_text().replace("00096", "00097"),
        name,
      )
    for path in (OLD / "vendor").rglob("*.py"):
      self.assertEqual(
        path.read_bytes(),
        (HERE / "vendor" / path.relative_to(OLD / "vendor")).read_bytes(),
      )

  def test_runner_training_calls_and_defaults_unchanged(self):
    def tree(path):
      return ast.parse(path.read_text().replace("00097", "00096"))

    a, b = tree(HERE / "run_00097.py"), tree(OLD / "run_00096.py")

    def calls(t):
      return [
        ast.dump(n)
        for n in ast.walk(t)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr in ("learn", "save", "finish")
      ]

    self.assertEqual(calls(a), calls(b))

    def args(t):
      return [
        ast.dump(n)
        for n in ast.walk(t)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "add_argument"
        and isinstance(n.args[0], ast.Constant)
        and n.args[0].value != "--repeat-info"
      ]

    self.assertEqual(args(a), args(b))
    text = (HERE / "run_00097.py").read_text()
    self.assertIn('assert binding == repeat["paired_source"]', text)
    for bad in (
      "use_deterministic_algorithms",
      "cudnn.deterministic",
      "set_num_threads",
      ".resume(",
    ):
      self.assertNotIn(bad, text)

  def test_independent_standalone_entry(self):
    text = (HERE / "execute.sh").read_text()
    self.assertIn('"$(git branch --show-current)" == "00097"', text)
    self.assertIn(".00097_registered", text)
    self.assertIn('--repeat-info "${REPEAT_INFO}"', text)
    self.assertIn("--max-iterations 20000 --stage formal --seed 2026", text)
    self.assertNotIn("accept_gpu.py", text)
    self.assertNotIn("test_gpu_cpu_reference", text)
    self.assertNotIn("execute_queue.sh", text)
    self.assertNotIn("run_00095.py", text)
    self.assertEqual(
      (ROOT / "00097_EXECUTION.md").read_bytes(), (HERE / "EXECUTION.md").read_bytes()
    )

  def test_original_sources_not_changed(self):
    result = subprocess.check_output(
      [
        "git",
        "diff",
        "6600ec541f2aea431b961a86c4323c63147e5333",
        "--",
        "src",
        "uv.lock",
        "pyproject.toml",
        "experiments/00095",
        "experiments/00096",
      ],
      cwd=ROOT,
      text=True,
    )
    self.assertEqual(result, "")


if __name__ == "__main__":
  unittest.main(verbosity=2)
