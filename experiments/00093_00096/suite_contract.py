"""Identity and parent-method checks for the four paired experiments."""

import ast
import hashlib
import json
import re
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DELIVERY_PARENT = "863350bae241daea8cc20082e04bc053de93b6d8"
CASES = {
  "00093": (2028, "0009", None),
  "00094": (2028, "00092", "00093"),
  "00095": (2026, "0009", None),
  "00096": (2026, "00092", "00095"),
}


def renamed(text, mapping):
  pattern = "|".join(sorted(map(re.escape, mapping), key=len, reverse=True))
  return re.sub(pattern, lambda match: mapping[match.group()], text)


def spec(exp):
  return json.loads((ROOT / "experiments" / exp / "EXPERIMENT.json").read_text())


def check(exp):
  seed, parent, control = CASES[exp]
  here = ROOT / "experiments" / exp
  current, old = spec(exp), spec(parent)
  assert current["experiment"] == current["branch"] == exp
  assert current["parent_experiment"] == current["repeats_experiment"] == parent
  assert current["seed"] == seed and current["iterations"] == 20000
  assert current["rng_protocol"]["training_seed"] == seed
  assert current["rng_protocol"]["gsi_seed"] == 100000 + seed
  for key in (
    "task",
    "num_envs",
    "num_steps_per_env",
    "samples",
    "reward",
    "prior_sha256",
    "uv_lock_sha256",
    "gsi",
    "physical_filter_enabled",
    "noise_rescue_enabled",
    "mixed_cache_enabled",
    "clustering_enabled",
  ):
    assert current[key] == old[key], key
  assert current["seed_reference_experiment"] == ("0005" if seed == 2028 else "0008")
  mapping = {parent: exp, "2027": str(seed)}
  if control:
    mapping["0009"] = control
    mapping["00091"] = "00091"
    for key in ("physical_filter", "physical_manifest_sha256", "gpu_module"):
      assert current[key] == old[key], key
    a, b = dict(current["noise_rescue"]), dict(old["noise_rescue"])
    assert a.pop("seed_formula") == b.pop("seed_formula").replace("2027", str(seed))
    assert a == b
    assert current["control"]["experiment"] == control
    assert current["control"]["seed"] == seed
    assert "initial_models_sha256" not in current["control"]
    assert current["paired_archive"]["source_experiment"] == control
    assert (
      current["paired_archive"]["root"]
      == f"/root/gpufree-share/smp_gsi_paired_{control}/formal"
    )
    assert current["gpu_module"]["pretraining_equivalence_rerun"] is False
    files = [
      "paired_replay.py",
      "initialization.py",
      "archive_format.py",
      "verify_source_archive.py",
      "verify_pairing.py",
      "noise_rescue.py",
      "passive_metrics.py",
    ]
    for name in ("gsi_physics.py", "gpu_filter.py", "gpu_scorer.py"):
      assert (here / name).read_bytes() == (here.parent / parent / name).read_bytes()
  else:
    mapping["00091"] = "00094" if exp == "00093" else "00096"
    a, b = dict(current["gsi_recording"]), dict(old["gsi_recording"])
    assert a.pop("root") == f"/root/gpufree-share/smp_gsi_paired_{exp}"
    b.pop("root")
    assert a == b
    files = [
      "initialization.py",
      "gsi_archive.py",
      "verify_archive.py",
      "passive_metrics.py",
    ]
  # No seed variants are allowed to silently change the sampling/search algorithm.
  for name in files:
    assert (here / name).read_text() == renamed(
      (here.parent / parent / name).read_text(), mapping
    ), name
  run_name = f"run_{exp}.py"
  expected = renamed((here.parent / parent / f"run_{parent}.py").read_text(), mapping)
  expected = (
    expected.replace('"repeats_experiment": "00091"', '"repeats_experiment": "00092"')
    if control
    else expected.replace(
      '"repeats_experiment": "0002"', '"repeats_experiment": "0009"'
    )
  )
  if control:
    expected = expected.replace(
      '"parent_experiment": "00091"', '"parent_experiment": "00092"'
    )
  assert (here / run_name).read_text() == expected, "unexpected runner change"
  tree = ast.parse(expected)
  defaults = {}
  for node in ast.walk(tree):
    if (
      isinstance(node, ast.Call)
      and isinstance(node.func, ast.Attribute)
      and node.func.attr == "add_argument"
      and node.args
      and isinstance(node.args[0], ast.Constant)
    ):
      defaults[node.args[0].value] = next(
        (
          x.value.value
          for x in node.keywords
          if x.arg == "default" and isinstance(x.value, ast.Constant)
        ),
        None,
      )
  assert defaults["--seed"] == seed and defaults["--max-iterations"] == 20000
  launcher = (here / "execute.sh").read_text()
  assert f"--max-iterations 20000 --stage formal --seed {seed}" in launcher
  assert f"--max-iterations 2 --stage smoke --seed {seed}" in launcher
  assert "accept_gpu.py" not in launcher and "test_gpu_cpu_reference" not in launcher
  assert "SMP_PAIRED_QUEUE_DIRECTORY" in launcher and "result.json" in launcher
  assert (ROOT / f"{exp}_EXECUTION.md").read_bytes() == (
    here / "EXECUTION.md"
  ).read_bytes()
  if control:
    assert f"experiments/{control}/verify_completion.py" in launcher
    assert f"run_{control}.py" not in launcher
    assert (
      current["noise_engine_sha256"]
      == hashlib.sha256((here / "noise_rescue.py").read_bytes()).hexdigest()
    )
  assert (
    subprocess.check_output(
      [
        "git",
        "diff",
        DELIVERY_PARENT,
        "--",
        "src",
        "uv.lock",
        "pyproject.toml",
        "experiments/0009",
        "experiments/00091",
        "experiments/00092",
      ],
      cwd=ROOT,
      text=True,
    )
    == ""
  )


def run(exp):
  class VariantTest(unittest.TestCase):
    def test_identity_method_and_pairing(self):
      check(exp)

  result = unittest.TextTestRunner(verbosity=2).run(
    unittest.defaultTestLoader.loadTestsFromTestCase(VariantTest)
  )
  raise SystemExit(0 if result.wasSuccessful() else 1)
