#!/usr/bin/env python3
"""Standalone, stdlib-only handoff. Run OUTSIDE the live shared checkout."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

PREDECESSOR_REVISION = "9cecb1244c3d83eb1e9c159682ea2cc591b3fd55"
LOCK_NAME = ".0003_00031_queue.lock"


def require(condition, message):
  if not condition:
    raise RuntimeError(message)


def read_pairs(path):
  lines = path.read_text(encoding="utf-8").splitlines()
  require(all("=" in line for line in lines if line), f"incomplete record: {path}")
  pairs = [line.split("=", 1) for line in lines if line]
  require(len({key for key, _ in pairs}) == len(pairs), f"duplicate fields: {path}")
  return dict(pairs)


def git(repo, *args):
  return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def clean_source(repo, revision, branch):
  require(
    git(repo, "rev-parse", "HEAD") == revision,
    "shared HEAD changed; preserve it and stop",
  )
  require(
    git(repo, "branch", "--show-current") == branch, "shared branch changed; stop"
  )
  require(
    not git(repo, "status", "--porcelain"),
    "shared workspace is dirty; preserve changes and stop",
  )


def validate_predecessor(root, pipeline):
  require(
    pipeline.parent == root / "logs"
    and re.fullmatch(r"00053_pipeline_[A-Za-z0-9_]+", pipeline.name),
    "provide the exact original 00053 pipeline directory",
  )
  revision = read_pairs(pipeline / "revision.txt")
  require(
    revision == dict(branch="00053", commit=PREDECESSOR_REVISION),
    "unexpected 00053 source revision",
  )
  require(
    read_pairs(pipeline / "exit_code.txt")
    == dict(pipeline_exit_code="0", last_stage="complete"),
    "00053 pipeline did not finish successfully",
  )
  require(
    read_pairs(pipeline / "progress.txt") == dict(experiment="00053", stage="complete"),
    "00053 pipeline is not complete",
  )
  finished = read_pairs(pipeline / "COMPLETED.txt")
  require(finished.get("00053") == "PASS", "00053 completion marker missing PASS")
  run = finished.get("run", "")
  require(
    re.fullmatch(r"00053_location_seed2028_[A-Za-z0-9_]+", run),
    "invalid 00053 formal run",
  )
  require(
    read_pairs(pipeline / "runs.txt") == {"00053": run}, "pipeline/formal run mismatch"
  )
  log, ckpt = root / "logs" / run, root / "ckpt" / run
  require(
    Path(finished.get("completion", "")).resolve() == log / "COMPLETION.md",
    "wrong completion report",
  )
  require((log / "COMPLETION.md").is_file(), "00053 completion report missing")
  for folder in (log, pipeline / "formal_verify"):
    require(
      read_pairs(folder / "exit_code.txt")
      == dict(python_exit_code="0", tee_exit_code="0"),
      f"unsuccessful stage: {folder}",
    )
  info = json.loads((log / "run_info.json").read_text(encoding="utf-8"))
  expected = dict(
    experiment="00053",
    seed=2028,
    completed_iterations=10000,
    max_iterations=10000,
    num_envs=1024,
    num_steps_per_env=24,
    total_samples=245760000,
    git_commit=PREDECESSOR_REVISION,
    git_branch="00053",
    git_dirty="",
  )
  require(
    all(info.get(key) == value for key, value in expected.items()),
    "00053 formal metadata mismatch",
  )
  require(
    Path(info["ckpt_dir"]).resolve() == ckpt, "00053 checkpoint directory mismatch"
  )
  require(Path(info["log_dir"]).resolve() == log, "00053 log directory mismatch")
  cfg = info.get("noise_rescue_config", {})
  require(
    cfg.get("max_generations") == 30 and cfg.get("rms_limit") == 0.15,
    "not the requested 00053 configuration",
  )
  return log, ckpt


def save_state(path, phase, **fields):
  fields.update(
    phase=phase, updated_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
  )
  temporary = path.with_suffix(".tmp")
  temporary.write_text(
    json.dumps(fields, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
  )
  temporary.replace(path)
  print(f"HANDOFF_PHASE={phase}", flush=True)


def acquire_shared(root):
  handle = (root / LOCK_NAME).open("a")
  try:
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
  except BlockingIOError:
    handle.close()
    return None
  return handle


def switch_target(repo, revision):
  # Network and checkout changes occur only AFTER predecessor verification.
  subprocess.run(
    ["git", "fetch", "origin", "refs/heads/00033:refs/remotes/origin/00033"],
    cwd=repo,
    check=True,
  )
  require(
    git(repo, "rev-parse", "origin/00033") == revision,
    "remote 00033 changed since arming; stop",
  )
  subprocess.run(
    ["git", "merge-base", "--is-ancestor", PREDECESSOR_REVISION, revision],
    cwd=repo,
    check=True,
  )
  if (
    subprocess.run(
      ["git", "show-ref", "--verify", "--quiet", "refs/heads/00033"], cwd=repo
    ).returncode
    == 0
  ):
    subprocess.run(
      ["git", "merge-base", "--is-ancestor", "00033", revision], cwd=repo, check=True
    )
    subprocess.run(["git", "switch", "00033"], cwd=repo, check=True)
    subprocess.run(["git", "merge", "--ff-only", revision], cwd=repo, check=True)
  else:
    subprocess.run(
      ["git", "switch", "-c", "00033", "--track", "origin/00033"], cwd=repo, check=True
    )
  clean_source(repo, revision, "00033")


def run_successor(repo, lock, out):
  environment = dict(os.environ, SMP_QUEUE_LOCK_FD=str(lock.fileno()))
  with (out / "00033_pipeline_console.log").open("w", encoding="utf-8") as log:
    result = subprocess.run(
      ["bash", "experiments/00033/execute_00033.sh"],
      cwd=repo,
      env=environment,
      pass_fds=(lock.fileno(),),
      stdout=log,
      stderr=subprocess.STDOUT,
    )
  (out / "00033_exit_code.txt").write_text(
    f"pipeline_exit_code={result.returncode}\n", encoding="utf-8"
  )
  require(
    result.returncode == 0,
    f"00033 stopped with exit {result.returncode}; no automatic retry",
  )


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--root", type=Path, default=Path("/root/gpufree-data/smp-g1"))
  parser.add_argument(
    "--pipeline",
    type=Path,
    required=True,
    help="exact active 00053_pipeline_* directory",
  )
  parser.add_argument(
    "--target-revision", required=True, help="full immutable 00033 commit SHA"
  )
  parser.add_argument("--poll-seconds", type=float, default=30)
  args = parser.parse_args(argv)
  root, pipeline = args.root.resolve(), args.pipeline.resolve()
  repo = root / "repo"
  require(
    re.fullmatch(r"[0-9a-f]{40}", args.target_revision), "full target SHA required"
  )
  require(1 <= args.poll_seconds <= 60, "poll interval must be 1..60 seconds")
  require(
    pipeline.parent == root / "logs"
    and re.fullmatch(r"00053_pipeline_[A-Za-z0-9_]+", pipeline.name),
    "wrong pipeline path",
  )
  require(
    not Path(__file__).resolve().is_relative_to(repo),
    "copy this watcher outside the shared checkout before arming",
  )
  require((root / "venv/bin/python").is_file(), "shared venv missing")
  require(
    read_pairs(pipeline / "revision.txt")
    == dict(branch="00053", commit=PREDECESSOR_REVISION),
    "wrong predecessor revision",
  )
  out = root / "logs" / f"handoff_00053_to_00033_{pipeline.name}"
  out.mkdir(exist_ok=True)
  state = out / "state.json"
  with (out / "watcher.lock").open("a") as watcher_lock:
    try:
      fcntl.flock(watcher_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
      raise RuntimeError("this predecessor already has an active watcher") from exc
    identity = dict(
      predecessor_pipeline=str(pipeline), target_revision=args.target_revision
    )
    if state.exists():
      previous = json.loads(state.read_text(encoding="utf-8"))
      require(
        previous.get("phase") == "waiting"
        and all(previous.get(k) == v for k, v in identity.items()),
        "handoff already attempted; do not launch twice",
      )
    clean_source(repo, PREDECESSOR_REVISION, "00053")
    save_state(state, "waiting", **identity)
    try:
      while True:
        shared = acquire_shared(root)
        if shared is None:
          time.sleep(args.poll_seconds)
          continue
        with shared:
          # Hold this SAME open-file lock through validation, branch switch,
          # and the complete successor pipeline (also inherited by its shell).
          clean_source(repo, PREDECESSOR_REVISION, "00053")
          log, ckpt = validate_predecessor(root, pipeline)
          python = str(root / "venv/bin/python")
          with (out / "00053_recheck_console.log").open(
            "w", encoding="utf-8"
          ) as output:
            subprocess.run(
              [
                python,
                "experiments/00053/verify_completion.py",
                "--log-dir",
                str(log),
                "--ckpt-dir",
                str(ckpt),
                "--out",
                str(out / "00053_RECHECK.md"),
              ],
              cwd=repo,
              check=True,
              stdout=output,
              stderr=subprocess.STDOUT,
            )
          # Never automatically repeat after this durable launch intention.
          save_state(state, "launch_committed", **identity)
          switch_target(repo, args.target_revision)
          committed = subprocess.check_output(
            [
              "git",
              "show",
              f"{args.target_revision}:experiments/00033/chain_after_00053.py",
            ],
            cwd=repo,
          )
          require(
            hashlib.sha256(committed).digest()
            == hashlib.sha256(Path(__file__).read_bytes()).digest(),
            "watcher does not match the pinned target commit",
          )
          save_state(state, "running_00033", **identity)
          run_successor(repo, shared, out)
          save_state(state, "complete", **identity)
          print(f"HANDOFF_COMPLETE={out}", flush=True)
          return 0
    except BaseException as exc:
      save_state(state, "stopped", error=str(exc), **identity)
      raise


if __name__ == "__main__":
  try:
    raise SystemExit(main())
  except (
    RuntimeError,
    OSError,
    subprocess.CalledProcessError,
    KeyError,
    ValueError,
  ) as exc:
    print(f"HANDOFF_STOPPED: {exc}", file=sys.stderr, flush=True)
    raise SystemExit(2) from exc
