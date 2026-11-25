"""Read-only provenance checks for an exact-input, same-seed 00096 repeat.

No simulation, sampling, network forward or random-number calls occur here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

EXP = Path(__file__).resolve().parent


def sha256(path):
  digest = hashlib.sha256()
  with Path(path).open("rb") as stream:
    for block in iter(lambda: stream.read(1 << 20), b""):
      digest.update(block)
  return digest.hexdigest()


def validate_repeat(path):
  path = Path(path).resolve()
  old = json.loads(path.read_text(encoding="utf-8"))
  spec = json.loads((EXP / "EXPERIMENT.json").read_text(encoding="utf-8"))
  assert old["experiment"] == "00096" and old["stage"] == "formal", (
    "select the original formal 00096, not 00095/smoke"
  )
  assert old["seed"] == old["initialization_seed"] == 2026
  assert old["max_iterations"] == old["completed_iterations"] == 20000
  assert old["total_samples"] == 491520000
  assert old["num_envs"] == 1024 and old["num_steps_per_env"] == 24
  assert old["git_commit"] == spec["repeat_protocol"]["reference_code_commit"], (
    "original 00096 code differs from the frozen repeat parent; inspect before running"
  )
  assert old["git_dirty"] == "", (
    "original run used local edits; cannot silently claim exact repeat"
  )
  assert old["prior_sha256"] == spec["prior_sha256"]
  assert old["uv_lock_sha256"] == spec["uv_lock_sha256"]
  assert old["gpu_module"] == spec["gpu_module"]
  assert old["physical_manifest_sha256"] == spec["physical_manifest_sha256"]
  assert old["physical_filter_enabled"] and old["noise_rescue_enabled"]
  assert old["gpu_physical_filter_enabled"] and old["gsi_replay_enabled"]
  assert (
    not old["gsi_recording_enabled"]
    and not old["mixed_cache_enabled"]
    and not old["clustering_enabled"]
  )
  assert old["paired_replay_status"] == "complete"
  assert old["initialization_mode"] == "load_00095_models_only"
  cfg = old["noise_rescue_config"]
  required = dict(
    population_size=32,
    max_generations=30,
    sigma0=0.05,
    rms_limit=0.15,
    coordinate_bound=1.0,
    inference_batch_size=1024,
    skip_joint_excess_deg=None,
    improvement_generation=10,
    minimum_joint_improvement=0.15,
    joint_tolerance_rad=1e-5,
  )
  assert cfg == required, "original 00096 rescue settings differ"
  exits = dict(
    line.split("=", 1)
    for line in (path.parent / "exit_code.txt").read_text().splitlines()
    if "=" in line
  )
  assert exits.get("python_exit_code") == exits.get("tee_exit_code") == "0", (
    "original 00096 did not exit successfully"
  )
  assert (path.parent / "COMPLETION.md").is_file(), (
    "original 00096 completion report required"
  )
  source = old["paired_source"]
  control = Path(old["control_info"]).resolve()
  assert control == Path(source["control_info"]).resolve()
  assert sha256(control) == source["control_info_sha256"], (
    "original 00095 record changed"
  )
  ci = json.loads(control.read_text(encoding="utf-8"))
  assert ci["experiment"] == "00095" and ci["stage"] == "formal" and ci["seed"] == 2026
  assert ci["run_id"] == source["source_run_id"]
  assert (
    ci["initial_models_sha256"]
    == old["initial_models_sha256"]
    == old["control_initial_models_sha256"]
    == source["initial_models_sha256"]
  )
  initial = Path(source["initial_checkpoint"]).resolve()
  assert initial == Path(ci["ckpt_dir"]).resolve() / "initial_untrained.pt"
  assert (
    sha256(initial)
    == source["initial_checkpoint_sha256"]
    == old["control_initial_checkpoint_sha256"]
    == ci["saved_initial_checkpoint_sha256"]
  )
  assert Path(ci["gsi_archive_dir"]).resolve() == Path(source["archive_dir"]).resolve()
  assert (
    sha256(Path(source["archive_dir"]) / "manifest.json") == source["manifest_sha256"]
  )
  return dict(
    status="PASS",
    reference_experiment="00096",
    reference_run_id=old["run_id"],
    reference_run_info=str(path),
    reference_run_info_sha256=sha256(path),
    reference_code_commit=old["git_commit"],
    control_info=str(control),
    paired_source=source,
    seed=2026,
    iterations=20000,
    comparison_policy="same inputs and method; no requirement for bitwise-identical training outputs",
  )


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--repeat-info", type=Path, required=True)
  p.add_argument("--print-control", action="store_true")
  p.add_argument("--out", type=Path)
  args = p.parse_args()
  result = validate_repeat(args.repeat_info)
  if args.out:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
      json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
  if args.print_control:
    print(result["control_info"])
  else:
    print("REPEAT_SOURCE_CHECK=PASS")


if __name__ == "__main__":
  main()
