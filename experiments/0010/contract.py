"""Check exact baseline source, the new experiment identity and shared SDK."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parent


def check_files(*, quiet=False):
  spec = json.loads((EXP / "EXPERIMENT.json").read_text())
  expected = json.loads((EXP / "BASELINE_FILES.json").read_text())
  for name, digest in expected.items():
    actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    assert actual == digest, f"baseline file changed: {name}"
  tracked = subprocess.check_output(["git", "ls-files", "src"], cwd=ROOT, text=True)
  assert set(tracked.splitlines()) == {p for p in expected if p.startswith("src/")}
  assert spec["experiment"] == EXP.name and spec["seed"] == 2028
  assert spec["server"] == "B" and spec["repeats_experiment"] == "0005"
  assert spec["gsi_recording"]["root"] == "/home/yang/rollout/smp_gsi_paired_0010"
  assert spec["num_envs"] == 1024 and spec["num_steps_per_env"] == 24
  assert spec["iterations"] == 20000 and spec["samples"] == 491520000
  assert spec["physical_filter_enabled"] is False
  assert spec["mixed_cache_enabled"] is spec["clustering_enabled"] is False
  assert spec["noise_rescue_enabled"] is False
  assert spec["gsi_recording"]["dtype"] == "float32"
  assert spec["gsi_recording"]["original_rows"] == 208896
  assert spec["rng_protocol"]["training_seed"] == 2028
  assert spec["rng_protocol"]["gsi_seed"] == 102028
  if not quiet:
    print(
      f"BASELINE_SOURCE_CHECK=PASS ({len(expected)} files, {spec['repeats_experiment']}, seed 2028)"
    )
  return spec


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--environment", action="store_true")
  parser.add_argument("--out", type=Path)
  args = parser.parse_args()
  spec = check_files()
  result = {
    "experiment": spec["experiment"],
    "seed": 2028,
    "baseline_commit": spec["baseline_commit"],
  }
  if args.environment:
    from server_b_runtime import check_environment

    result.update(check_environment(ROOT))
    assert result["archive_free_bytes"] >= 35_000_000_000, (
      "reserve 35 GB for recorded GSI data"
    )
    print("SHARED_ENVIRONMENT_CHECK=PASS")
  if args.out:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
  main()
