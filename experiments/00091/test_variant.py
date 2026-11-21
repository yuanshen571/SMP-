"""Check old source preservation and the exact paired-rescue method adapter."""

import ast
import hashlib
import json
import subprocess
import unittest
from pathlib import Path

E = Path(__file__).resolve().parent
R = E.parents[1]
BASE = "c5843c8fa1c67879e0c1f4e66c49d81ce82d8e1f"
OLD = "0a4ceff351b1f4885b61b6bfddee80ec49cb1874"


class VariantTests(unittest.TestCase):
  def test_original_source_and_0009_unchanged(self):
    for ref, paths in [
      (OLD, ["src", "uv.lock", "pyproject.toml"]),
      (BASE, ["experiments/0009", "0009_EXECUTION.md"]),
    ]:
      self.assertEqual(
        subprocess.check_output(["git", "diff", ref, "--", *paths], cwd=R, text=True),
        "",
      )

  def test_fixed_parameters(self):
    s = json.loads((E / "EXPERIMENT.json").read_text())
    b = json.loads((E.parent / "0009/EXPERIMENT.json").read_text())
    for key in [
      "seed",
      "num_envs",
      "num_steps_per_env",
      "iterations",
      "samples",
      "reward",
      "prior_sha256",
      "uv_lock_sha256",
      "rng_protocol",
    ]:
      self.assertEqual(s[key], b[key], key)
    self.assertEqual(
      s["control"],
      {
        "experiment": "0009",
        "seed": 2027,
        "selection": "explicit completed 20000-iteration formal run_info; runtime fingerprints, no static initial SHA",
      },
    )
    self.assertFalse(s["physical_filter"]["repair_small_joints"])
    self.assertEqual(s["noise_rescue"]["max_generations"], 30)
    self.assertEqual(s["paired_archive"]["source_rows"], 208896)
    self.assertEqual(
      hashlib.sha256((E / "gsi_physics.py").read_bytes()).hexdigest(),
      s["physical_source_sha256"],
    )

  def test_native_fifo_and_paired_before_env(self):
    s = (E / "run_00091.py").read_text()
    self.assertLess(
      s.index("consumer.install(native_events)"), s.index("env = ManagerBasedRlEnv(")
    )
    self.assertLess(s.index("load_common_initial("), s.index("runner.learn("))
    self.assertNotIn('expected_control["initial_models_sha256"]', s)
    self.assertNotIn("recorder.", s)
    self.assertNotIn("configure(env_cfg)", s)
    s = (E / "paired_replay.py").read_text()
    tree = ast.parse(s)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    self.assertFalse(
      any(isinstance(n.func, ast.Name) and n.func.id == "draw_recorded" for n in calls)
    )
    self.assertIn("events._ddpm_sample = self.sample", s)
    self.assertNotIn("env._smp_gsi_pool[", s)

  def test_launcher_only_00091_and_no_source_delete(self):
    s = (E / "execute.sh").read_text()
    self.assertIn("--max-iterations 2 --stage smoke", s)
    self.assertIn("--max-iterations 20000 --stage formal", s)
    self.assertIn("validate_source.py", s)
    self.assertNotIn("rm ", s)
    self.assertNotIn("rmtree", s)
    self.assertNotIn("run_0009.py", s)
    self.assertEqual(
      (R / "00091_EXECUTION.md").read_bytes(), (E / "EXECUTION.md").read_bytes()
    )


if __name__ == "__main__":
  unittest.main(verbosity=2)
