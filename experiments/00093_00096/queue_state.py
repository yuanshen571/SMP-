"""Persist exact verified stage bindings; never choose a 'latest' run by glob."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ORDER = ("00093", "00094", "00095", "00096")
SEEDS = {"00093": 2028, "00094": 2028, "00095": 2026, "00096": 2026}
CONTROLS = {"00094": "00093", "00096": "00095"}


def read(path):
  return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, obj):
  path = Path(path)
  tmp = path.with_suffix(".partial")
  tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
  tmp.replace(path)


def fields(path):
  return dict(
    line.split("=", 1) for line in Path(path).read_text().splitlines() if "=" in line
  )


def sha(path):
  result = hashlib.sha256()
  with Path(path).open("rb") as stream:
    for chunk in iter(lambda: stream.read(1 << 20), b""):
      result.update(chunk)
  return result.hexdigest()


def complete(queue, root, exp):
  state = read(queue / "queue_progress.json")
  index = ORDER.index(exp)
  assert state["active"] == exp
  assert state["completed"] == list(ORDER[:index])
  # Written by this experiment's launcher, not guessed from run timestamps.
  registration = fields(root / "logs" / f".{exp}_registered")
  execution = Path(registration["execution"]).resolve()
  assert execution.parent == (root / "logs").resolve()
  assert execution.name.startswith(exp + "_execution_")
  assert fields(execution / "exit_code.txt")["execution_exit_code"] == "0"
  assert (execution / "completed.txt").read_text() == f"{exp}=PASS\n"
  result = read(execution / "result.json")
  assert result["status"] == "PASS" and result["experiment"] == exp
  assert result["seed"] == SEEDS[exp]
  assert Path(result["execution_directory"]).resolve() == execution
  info_path = Path(result["run_info"]).resolve()
  assert info_path.parent.parent == (root / "logs").resolve()
  info = read(info_path)
  assert info["run_id"] == result["run_id"] == info_path.parent.name
  assert info["experiment"] == exp and info["seed"] == SEEDS[exp]
  assert info["stage"] == "formal" and info["max_iterations"] == 20000
  assert info["completed_iterations"] == 20000 and info["total_samples"] == 491520000
  assert info["num_envs"] == 1024 and info["num_steps_per_env"] == 24
  assert info["initialization_seed"] == SEEDS[exp] and not info["git_dirty"]
  assert info["git_commit"] == registration["commit"]
  assert Path(info["ckpt_dir"]).resolve() == Path(result["ckpt_dir"]).resolve()
  assert (info_path.parent / "COMPLETION.md").is_file()
  assert all(
    fields(info_path.parent / "exit_code.txt")[k] == "0"
    for k in ("python_exit_code", "tee_exit_code")
  )
  if exp in CONTROLS:
    previous = read(queue / f"{CONTROLS[exp]}_result.json")
    previous_info = read(previous["run_info"])
    assert Path(info["control_info"]).resolve() == Path(previous["run_info"]).resolve()
    assert (
      Path(registration["control_info"]).resolve()
      == Path(previous["run_info"]).resolve()
    )
    assert info["initial_models_sha256"] == previous_info["initial_models_sha256"]
    assert (
      info["control_initial_checkpoint_sha256"]
      == previous_info["saved_initial_checkpoint_sha256"]
    )
    assert info["paired_source"]["control_info_sha256"] == previous["run_info_sha256"]
    assert info["gpu_physical_filter_enabled"] is True
    assert info["parent_experiment"] == "00092"
    assert info["gpu_module"]["pretraining_equivalence_rerun"] is False
    pair = read(execution / "pairing.json")
    assert pair["status"] == "PASS" and pair["rows"] == 208896 and pair["shards"] == 204
    assert (
      Path(pair["source"]["control_info"]).resolve()
      == Path(previous["run_info"]).resolve()
    )
  else:
    assert info["initialization_mode"] == "fresh_seed"
    assert info["repeats_experiment"] == "0009"
    assert info["gsi_archive_status"] == "complete" and info["gsi_recording_enabled"]
    assert info["gsi_generation_rng_seed"] == 100000 + SEEDS[exp]
    assert info["gsi_recorded_rows"] == 208896 and info["gsi_recorded_shards"] == 204
    archive = Path(info["gsi_archive_dir"]).resolve()
    assert (
      archive.parent
      == Path(f"/root/gpufree-share/smp_gsi_paired_{exp}/formal").resolve()
    )
    assert archive.name == info["run_id"]
    audit = read(execution / "archive_check.json")
    assert audit["status"] == "PASS" and Path(audit["directory"]).resolve() == archive
  result["run_info_sha256"] = sha(info_path)
  write(queue / f"{exp}_result.json", result)
  state["completed"].append(exp)
  state.update(active=None, status="running")
  write(queue / "queue_progress.json", state)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "action", choices=("preflight", "running", "completed", "control", "finish")
  )
  parser.add_argument("--queue", type=Path, required=True)
  parser.add_argument("--root", type=Path)
  parser.add_argument("--experiment", choices=ORDER)
  args = parser.parse_args()
  queue = args.queue.resolve()
  path = queue / "queue_progress.json"
  if args.action == "preflight":
    assert not path.exists()
    mount = Path("/root/gpufree-share")
    assert mount.is_dir(), (
      "use the existing shared storage; do not substitute a system-disk directory"
    )
    assert shutil.disk_usage(mount).free >= 70_000_000_000, (
      "reserve 70 GB for both lossless archives and temporary smoke data"
    )
    for exp in ORDER:
      assert not (args.root / "logs" / f".{exp}_registered").exists(), (
        f"{exp} already registered"
      )
    write(path, dict(order=list(ORDER), completed=[], active=None, status="prepared"))
  elif args.action == "running":
    state = read(path)
    assert args.experiment == ORDER[len(state["completed"])]
    assert state["active"] is None
    state.update(active=args.experiment, status="running")
    write(path, state)
  elif args.action == "completed":
    complete(queue, args.root.resolve(), args.experiment)
  elif args.action == "control":
    assert args.experiment in ("00093", "00095")
    assert args.experiment in read(path)["completed"]
    result = read(queue / f"{args.experiment}_result.json")
    assert (
      result["status"] == "PASS"
      and sha(result["run_info"]) == result["run_info_sha256"]
    )
    print(result["run_info"])
  else:
    state = read(path)
    assert state["completed"] == list(ORDER) and state["active"] is None
    state["status"] = "complete"
    write(path, state)
    lines = [
      "# 00093–00096 队列完成摘要",
      "",
      "四组均完成20000轮、491520000样本并通过各自核验。",
      "",
      "| 实验 | 种子 | 正式run_id |",
      "|---|---:|---|",
    ]
    for exp in ORDER:
      result = read(queue / f"{exp}_result.json")
      lines.append(f"| {exp} | {result['seed']} | {result['run_id']} |")
    (queue / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
  main()
