"""Prove the baseline method stays unchanged when reusing the existing source."""

import ast
import json
import subprocess
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = HERE.parent / "00095"


class VariantTests(unittest.TestCase):
  def test_same_training_and_method_settings(self):
    new, old = [json.loads((p / "EXPERIMENT.json").read_text()) for p in (HERE, OLD)]
    for key in [
      "task",
      "seed",
      "num_envs",
      "num_steps_per_env",
      "iterations",
      "samples",
      "reward",
      "prior_sha256",
      "uv_lock_sha256",
      "physical_filter_enabled",
      "mixed_cache_enabled",
      "clustering_enabled",
      "gsi",
      "rng_protocol",
      "noise_rescue_enabled",
    ]:
      self.assertEqual(new[key], old[key], key)
    self.assertFalse(new["gsi_recording"]["enabled"])
    self.assertTrue(new["gsi_replay"]["enabled"])
    self.assertEqual(new["gsi_replay"]["source_experiment"], "00095")
    self.assertEqual(new["repeats_experiment"], "00095")

  def test_same_metrics_initialization_and_ppo_call(self):
    self.assertEqual(
      (HERE / "passive_metrics.py").read_text(),
      (OLD / "passive_metrics.py").read_text().replace("00095", "00098"),
    )
    self.assertEqual(
      (HERE / "initialization.py").read_bytes(),
      (OLD / "initialization.py").read_bytes(),
    )

    def calls(path):
      t = ast.parse(path.read_text())
      return [
        ast.dump(n)
        for n in ast.walk(t)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "learn"
      ]

    self.assertEqual(calls(HERE / "run_00098.py"), calls(OLD / "run_00095.py"))
    text = (HERE / "run_00098.py").read_text()
    self.assertNotIn("ArchiveRecorder", text)
    self.assertNotIn("search_batch", text)
    self.assertNotIn('physical_filter_enabled": True', text)
    self.assertIn(
      "load_common_initial(runner, initial_checkpoint, args.control_info)", text
    )
    self.assertIn('assert not hasattr(env, "_smp_physics")', text)
    for bad in ["use_deterministic_algorithms", "set_num_threads", "resume("]:
      self.assertNotIn(bad, text)

  def test_standalone_launcher(self):
    text = (HERE / "execute.sh").read_text()
    self.assertIn(".00098_registered", text)
    self.assertIn('"$(git branch --show-current)" == "00098"', text)
    self.assertIn('--control-info "${CONTROL_INFO}"', text)
    self.assertIn("--max-iterations 20000 --stage formal --seed 2026", text)
    self.assertNotIn("execute_queue.sh", text)
    self.assertNotIn("run_00095.py", text)
    self.assertNotIn("accept_gpu.py", text)
    self.assertNotIn("rmtree", text)
    self.assertEqual(
      (ROOT / "00098_EXECUTION.md").read_bytes(), (HERE / "EXECUTION.md").read_bytes()
    )

  def test_original_source_unchanged(self):
    value = subprocess.check_output(
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
    self.assertEqual(value, "")


if __name__ == "__main__":
  unittest.main(verbosity=2)
