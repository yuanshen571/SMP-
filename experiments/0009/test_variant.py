"""Keep the 0002 method unchanged; audit only the declared pairing adapter."""

import ast
import json
import subprocess
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
METHOD = "0a4ceff351b1f4885b61b6bfddee80ec49cb1874"
HARNESS = "17a5bdad6f4951c6c0363b459cd37a72cbf2fd6d"


def reference(commit, name):
  return subprocess.check_output(
    ["git", "show", f"{commit}:{name}"], cwd=ROOT, text=True
  )


class VariantTests(unittest.TestCase):
  def test_core_source_and_old_experiments_unchanged(self):
    self.assertEqual(
      subprocess.check_output(
        [
          "git",
          "diff",
          METHOD,
          "--",
          "src",
          "uv.lock",
          "pyproject.toml",
          ".python-version",
        ],
        cwd=ROOT,
        text=True,
      ),
      "",
    )
    self.assertEqual(
      subprocess.check_output(
        ["git", "diff", HARNESS, "--", "experiments/0002", "experiments/0008"],
        cwd=ROOT,
        text=True,
      ),
      "",
    )

  def test_original_task_reward_and_budget(self):
    old = json.loads(reference(METHOD, "experiments/0002/EXPERIMENT.json"))
    new = json.loads((HERE / "EXPERIMENT.json").read_text())
    for key in (
      "task",
      "seed",
      "num_envs",
      "num_steps_per_env",
      "reward",
      "prior_sha256",
      "uv_lock_sha256",
    ):
      self.assertEqual(new[key], old[key], key)
    self.assertEqual(new["seed"], 2027)
    self.assertEqual(new["iterations"], 20000)
    self.assertEqual(new["samples"], 1024 * 24 * 20000)
    self.assertTrue(
      all(
        new[k] is False
        for k in (
          "physical_filter_enabled",
          "noise_rescue_enabled",
          "mixed_cache_enabled",
          "clustering_enabled",
        )
      )
    )
    self.assertEqual(new["rng_protocol"]["gsi_seed"], 102027)
    self.assertEqual(new["gsi_recording"]["original_rows"], 4096 + 200 * 1024)
    self.assertEqual(new["gsi_recording"]["shards"], 4 + 200)

  def test_metrics_body_unchanged(self):
    expected = (
      reference(HARNESS, "experiments/0008/passive_metrics.py")
      .replace("0008", "0009")
      .replace("10000", "20000")
    )
    self.assertEqual((HERE / "passive_metrics.py").read_text(), expected)

  def test_capture_installed_before_environment_and_finished_before_close(self):
    source = (HERE / "run_0009.py").read_text()
    self.assertLess(
      source.index("recorder.install(native_events)"),
      source.index("env = ManagerBasedRlEnv("),
    )
    self.assertLess(
      source.index("recorder.finish(env"), source.index("wrapped.close()")
    )
    self.assertIn(
      "runner.learn(\n      num_learning_iterations=args.max_iterations, init_at_random_ep_len=True",
      source,
    )
    self.assertNotIn("runner.load(", source)
    tree = ast.parse(source)
    defaults = {
      node.args[0].value: node
      for node in ast.walk(tree)
      if isinstance(node, ast.Call)
      and isinstance(node.func, ast.Attribute)
      and node.func.attr == "add_argument"
      and node.args
      and isinstance(node.args[0], ast.Constant)
    }
    maximum = next(
      k.value.value for k in defaults["--max-iterations"].keywords if k.arg == "default"
    )
    self.assertEqual(maximum, 20000)

  def test_single_group_execution_and_archive_checks(self):
    source = (HERE / "execute.sh").read_text()
    self.assertNotIn("execute_pair.sh", source)
    self.assertNotIn("--allow-incomplete", source)
    self.assertIn("--max-iterations 2 --stage smoke --seed 2027", source)
    self.assertIn("--max-iterations 20000 --stage formal --seed 2027", source)
    self.assertIn('"${PREFIX}/verify_archive.py" --iterations 20000', source)
    self.assertNotIn("run_00091.py", source)


if __name__ == "__main__":
  unittest.main(verbosity=2)
