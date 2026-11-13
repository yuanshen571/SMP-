#!/usr/bin/env python3
"""Outside-checkout watcher: completed 00073 -> seed-2026 pair, no simulator while waiting."""

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

LOCK_NAME = ".0003_00031_queue.lock"
WATCHER_PATH = "experiments/seed2026/chain_after_00073.py"


def require(condition, message):
  if not condition:
    raise RuntimeError(message)


def git(repo, *args):
  return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def clean_source(repo, revision):
  require(
    git(repo, "branch", "--show-current") == "00073", "shared branch changed; stop"
  )
  require(git(repo, "rev-parse", "HEAD") == revision, "shared HEAD changed; stop")
  require(
    not git(repo, "status", "--porcelain"), "shared source dirty; preserve it and stop"
  )


def read_pairs(path):
  rows = [
    line.split("=", 1) for line in path.read_text(encoding="utf-8").splitlines() if line
  ]
  require(all(len(row) == 2 for row in rows), f"incomplete record: {path}")
  require(len({row[0] for row in rows}) == len(rows), f"duplicate fields: {path}")
  return dict(rows)


def process_record(pid):
  """Linux start ticks prevent a reused PID being mistaken for the old process."""
  try:
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    if fields[0] in ("Z", "X"):
      return None
    return dict(pid=pid, start_ticks=fields[19])
  except FileNotFoundError:
    return None


def require_matching_learner(processes, log, repo):
  for process in processes:
    try:
      proc = Path(f"/proc/{process['pid']}")
      arguments = (
        proc.joinpath("cmdline").read_bytes().decode().rstrip("\0").split("\0")
      )
      if (
        "--run" not in arguments or arguments[arguments.index("--run") + 1] != log.name
      ):
        continue
      cwd = proc.joinpath("cwd").resolve()
      if any(
        (cwd / arg).resolve() == repo / "experiments/00073/run_00073.py"
        for arg in arguments
        if arg.endswith("run_00073.py")
      ):
        return
    except (FileNotFoundError, IndexError):
      continue
  raise RuntimeError("wait PIDs do not include this exact 00073 learner")


def predecessor_info(root, log, revision):
  require(
    log.parent == root / "logs", "provide an exact formal run directly under logs"
  )
  require(
    re.fullmatch(r"00073_location_seed2030_[A-Za-z0-9_]+", log.name),
    "not a formal 00073 run",
  )
  info = json.loads((log / "run_info.json").read_text(encoding="utf-8"))
  expected = dict(
    experiment="00073",
    seed=2030,
    max_iterations=10000,
    num_envs=1024,
    num_steps_per_env=24,
    git_commit=revision,
    git_branch="00073",
    git_dirty="",
  )
  require(
    all(info.get(k) == v for k, v in expected.items()), "00073 formal identity mismatch"
  )
  require(Path(info["log_dir"]).resolve() == log, "recorded log directory differs")
  require(
    Path(info["ckpt_dir"]).resolve() == root / "ckpt" / log.name,
    "wrong checkpoint directory",
  )
  config = info.get("noise_rescue_config", {})
  require(
    config.get("max_generations") == 30 and config.get("rms_limit") == 0.15,
    "wrong 00073 rescue configuration",
  )
  return info


def validate_finished(root, log, revision):
  info = predecessor_info(root, log, revision)
  require(
    info.get("completed_iterations") == 10000
    and info.get("total_samples") == 245760000,
    "00073 did not complete 10000 iterations",
  )
  require(
    read_pairs(log / "exit_code.txt") == dict(python_exit_code="0", tee_exit_code="0"),
    "00073 training/tee exit was not successful",
  )
  return Path(info["ckpt_dir"]).resolve()


def save_state(path, phase, **fields):
  fields.update(
    phase=phase, updated_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
  )
  temp = path.with_suffix(".tmp")
  temp.write_text(
    json.dumps(fields, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
  )
  temp.replace(path)
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
  subprocess.run(
    ["git", "fetch", "origin", "refs/heads/0008:refs/remotes/origin/0008"],
    cwd=repo,
    check=True,
  )
  require(
    git(repo, "rev-parse", "origin/0008") == revision,
    "0008 differs from pinned delivery",
  )
  committed = subprocess.check_output(
    ["git", "show", f"{revision}:{WATCHER_PATH}"], cwd=repo
  )
  require(
    hashlib.sha256(committed).digest()
    == hashlib.sha256(Path(__file__).read_bytes()).digest(),
    "watcher differs from pinned source",
  )
  exists = (
    subprocess.run(
      ["git", "show-ref", "--verify", "--quiet", "refs/heads/0008"], cwd=repo
    ).returncode
    == 0
  )
  if exists:
    subprocess.run(
      ["git", "merge-base", "--is-ancestor", "0008", revision], cwd=repo, check=True
    )
    subprocess.run(["git", "switch", "0008"], cwd=repo, check=True)
    subprocess.run(["git", "merge", "--ff-only", revision], cwd=repo, check=True)
  else:
    subprocess.run(
      ["git", "switch", "-c", "0008", "--track", "origin/0008"], cwd=repo, check=True
    )
  require(
    git(repo, "rev-parse", "HEAD") == revision
    and not git(repo, "status", "--porcelain"),
    "0008 checkout mismatch",
  )


def run_successor(repo, shared, out, rescue_revision):
  environment = dict(
    os.environ,
    SMP_QUEUE_LOCK_FD=str(shared.fileno()),
    PYTHONUTF8="1",
    PYTHONIOENCODING="utf-8",
  )
  with (out / "seed2026_queue_console.log").open("w", encoding="utf-8") as output:
    result = subprocess.run(
      ["bash", "experiments/seed2026/execute_pair.sh", rescue_revision],
      cwd=repo,
      env=environment,
      pass_fds=(shared.fileno(),),
      stdout=output,
      stderr=subprocess.STDOUT,
    )
  (out / "seed2026_exit_code.txt").write_text(
    f"queue_exit_code={result.returncode}\n", encoding="utf-8"
  )
  require(
    result.returncode == 0,
    f"seed-2026 queue exited {result.returncode}; no automatic retry",
  )


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--root", type=Path, default=Path("/root/gpufree-data/smp-g1"))
  parser.add_argument(
    "--run", type=Path, required=True, help="exact running 00073 formal log directory"
  )
  parser.add_argument(
    "--wait-pid",
    type=int,
    action="append",
    required=True,
    help="learner PID and any finite outer coordinator PID; repeatable",
  )
  parser.add_argument(
    "--target-revision", required=True, help="delivered full 0008 SHA"
  )
  parser.add_argument(
    "--rescue-revision", required=True, help="delivered full 00083 SHA"
  )
  parser.add_argument("--poll-seconds", type=float, default=30)
  parser.add_argument("--check-only", action="store_true")
  args = parser.parse_args(argv)
  root, log = args.root.resolve(), args.run.resolve()
  repo = root / "repo"
  for revision in (args.target_revision, args.rescue_revision):
    require(re.fullmatch(r"[0-9a-f]{40}", revision), "full immutable revision required")
  require(1 <= args.poll_seconds <= 60, "poll interval must be 1..60 seconds")
  require(
    not Path(__file__).resolve().is_relative_to(repo),
    "run this watcher outside the live checkout",
  )
  require((root / "venv/bin/python").is_file(), "shared venv missing")
  require(
    not (root / "logs/.seed2026_0008_00083_registered").exists(),
    "seed-2026 pair already registered; do not launch twice",
  )
  revision = git(repo, "rev-parse", "HEAD")
  clean_source(repo, revision)
  predecessor_info(root, log, revision)
  require(all(pid > 1 for pid in args.wait_pid), "invalid wait PID")
  processes = [process_record(pid) for pid in sorted(set(args.wait_pid))]
  require(
    all(processes),
    "one of the supplied processes is no longer active; identify the live run again",
  )
  require_matching_learner(processes, log, repo)
  boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
  identity = dict(
    predecessor_run=str(log),
    predecessor_revision=revision,
    processes=processes,
    boot_id=boot,
    target_revision=args.target_revision,
    rescue_revision=args.rescue_revision,
  )
  if args.check_only:
    print("HANDOFF_PRECHECK=PASS", flush=True)
    return 0
  out = root / "logs/handoff_00073_to_seed2026"
  out.mkdir(exist_ok=True)
  state = out / "state.json"
  with (out / "watcher.lock").open("a") as local_lock:
    try:
      fcntl.flock(local_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
      raise RuntimeError("handoff watcher already active") from exc
    if state.exists():
      raise RuntimeError(
        "handoff already registered; inspect its state, do not register again"
      )
    save_state(state, "waiting", **identity)
    try:
      while True:
        require(
          Path("/proc/sys/kernel/random/boot_id").read_text().strip() == boot,
          "server rebooted; stop",
        )
        clean_source(repo, revision)
        if any(process_record(p["pid"]) == p for p in processes):
          time.sleep(args.poll_seconds)
          continue
        shared = acquire_shared(root)
        if shared is None:
          time.sleep(args.poll_seconds)
          continue
        with shared:
          # SAME open-file lock is retained through verification, checkout and both experiments.
          clean_source(repo, revision)
          ckpt = validate_finished(root, log, revision)
          environment = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES="",
            PYTHONUTF8="1",
            PYTHONIOENCODING="utf-8",
          )
          with (out / "00073_recheck_console.log").open(
            "w", encoding="utf-8"
          ) as output:
            subprocess.run(
              [
                str(root / "venv/bin/python"),
                "experiments/00073/verify_completion.py",
                "--log-dir",
                str(log),
                "--ckpt-dir",
                str(ckpt),
                "--out",
                str(out / "00073_RECHECK.md"),
              ],
              cwd=repo,
              env=environment,
              check=True,
              stdout=output,
              stderr=subprocess.STDOUT,
            )
          save_state(state, "launch_committed", **identity)
          switch_target(repo, args.target_revision)
          save_state(state, "running_seed2026_pair", **identity)
          run_successor(repo, shared, out, args.rescue_revision)
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
