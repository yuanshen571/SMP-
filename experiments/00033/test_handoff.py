"""No GPU: real file-lock handover, real Git switch and failed/partial predecessor checks."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import chain_after_00053 as chain


def put(path, text):
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(text, encoding="utf-8")


def fixture(root):
  pipeline = root / "logs/00053_pipeline_fixture"
  run = "00053_location_seed2028_fixture"
  log, ckpt = root / "logs" / run, root / "ckpt" / run
  log.mkdir(parents=True)
  ckpt.mkdir(parents=True)
  put(pipeline / "revision.txt", f"branch=00053\ncommit={chain.PREDECESSOR_REVISION}\n")
  put(pipeline / "exit_code.txt", "pipeline_exit_code=0\nlast_stage=complete\n")
  put(pipeline / "progress.txt", "experiment=00053\nstage=complete\n")
  put(
    pipeline / "COMPLETED.txt",
    f"00053=PASS\nrun={run}\ncompletion={log}/COMPLETION.md\n",
  )
  put(pipeline / "runs.txt", f"00053={run}\n")
  put(log / "COMPLETION.md", "fixture only\n")
  for folder in (log, pipeline / "formal_verify"):
    put(folder / "exit_code.txt", "python_exit_code=0\ntee_exit_code=0\n")
  info = dict(
    experiment="00053",
    seed=2028,
    completed_iterations=10000,
    max_iterations=10000,
    num_envs=1024,
    num_steps_per_env=24,
    total_samples=245760000,
    git_commit=chain.PREDECESSOR_REVISION,
    git_branch="00053",
    git_dirty="",
    log_dir=str(log),
    ckpt_dir=str(ckpt),
    noise_rescue_config=dict(max_generations=30, rms_limit=0.15),
  )
  put(log / "run_info.json", json.dumps(info))
  return pipeline, log, ckpt


class HandoffTests(unittest.TestCase):
  def test_complete_watcher_waits_then_launches_once_or_stops_on_failure(self):
    for successful in (True, False):
      with self.subTest(successful=successful), tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        repo = root / "repo"
        repo.mkdir()

        def run(*args, repo=repo):
          return subprocess.check_output(
            ["git", *args], cwd=repo, text=True, stderr=subprocess.DEVNULL
          ).strip()

        run("init", "-b", "00053")
        run("config", "user.email", "test@example.invalid")
        run("config", "user.name", "Protocol test")
        put(
          repo / "experiments/00053/verify_completion.py",
          "import pathlib,sys\n"
          "pathlib.Path(sys.argv[sys.argv.index('--out')+1]).write_text('CPU fixture passed')\n",
        )
        run("add", ".")
        run("commit", "-m", "predecessor fixture")
        previous = run("rev-parse", "HEAD")
        with patch.object(chain, "PREDECESSOR_REVISION", previous):
          pipeline, _, _ = fixture(root)
        if not successful:
          put(
            pipeline / "exit_code.txt",
            "pipeline_exit_code=1\nlast_stage=formal_training\n",
          )
        run("switch", "-c", "00033")
        watcher_source = (
          Path(chain.__file__).read_text().replace(chain.PREDECESSOR_REVISION, previous)
        )
        put(repo / "experiments/00033/chain_after_00053.py", watcher_source)
        put(
          repo / "experiments/00033/execute_00033.sh",
          '#!/bin/bash\nset -eu\n[[ "$(git branch --show-current)" == 00033 ]]\n'
          'printf "one launch\\n" >> "$(dirname "$PWD")/launched.txt"\n',
        )
        run("add", ".")
        run("commit", "-m", "successor fixture")
        target = run("rev-parse", "HEAD")
        run("remote", "add", "origin", str(repo))
        run("switch", "00053")
        (root / "venv/bin").mkdir(parents=True)
        (root / "venv/bin/python").symlink_to(sys.executable)
        watcher = root / "tools/watcher.py"
        put(watcher, watcher_source)
        args = [
          sys.executable,
          str(watcher),
          "--root",
          str(root),
          "--pipeline",
          str(pipeline),
          "--target-revision",
          target,
          "--poll-seconds",
          "1",
        ]
        original_lock = chain.acquire_shared(root)
        proc = subprocess.Popen(
          args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        try:
          state = (
            root / "logs" / f"handoff_00053_to_00033_{pipeline.name}" / "state.json"
          )
          deadline = time.monotonic() + 5
          while not state.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
          self.assertTrue(state.exists())
          self.assertEqual(json.loads(state.read_text())["phase"], "waiting")
          self.assertEqual(run("branch", "--show-current"), "00053")
          self.assertFalse((root / "launched.txt").exists())
          original_lock.close()
          output = proc.communicate(timeout=10)[0]
          self.assertEqual(proc.returncode, 0 if successful else 2, output)
          self.assertEqual(
            json.loads(state.read_text())["phase"],
            "complete" if successful else "stopped",
          )
          if successful:
            self.assertEqual((root / "launched.txt").read_text(), "one launch\n")
            repeated = subprocess.run(args, capture_output=True, text=True)
            self.assertEqual(repeated.returncode, 2)
            self.assertIn("already attempted", repeated.stderr)
            self.assertEqual((root / "launched.txt").read_text(), "one launch\n")
          else:
            self.assertFalse((root / "launched.txt").exists())
            self.assertEqual(run("branch", "--show-current"), "00053")
        finally:
          original_lock.close()
          if proc.poll() is None:
            proc.kill()
            proc.communicate()

  def test_predecessor_rejects_failure_partial_or_wrong_run(self):
    with tempfile.TemporaryDirectory() as temp:
      root = Path(temp)
      pipeline, log, ckpt = fixture(root)
      self.assertEqual(chain.validate_predecessor(root, pipeline), (log, ckpt))
      edits = [
        (
          pipeline / "exit_code.txt",
          "pipeline_exit_code=1\nlast_stage=formal_training\n",
        ),
        (pipeline / "COMPLETED.txt", "00053=PASS\n"),
        (
          pipeline / "formal_verify/exit_code.txt",
          "python_exit_code=1\ntee_exit_code=0\n",
        ),
        (pipeline / "runs.txt", "00053=another_run\n"),
        (pipeline / "progress.txt", "experiment=00053\nstage=formal_training\n"),
      ]
      for path, contents in edits:
        original = path.read_text()
        put(path, contents)
        with self.assertRaises(RuntimeError):
          chain.validate_predecessor(root, pipeline)
        put(path, original)
      meta = json.loads((log / "run_info.json").read_text())
      for key, val in (
        ("completed_iterations", 9999),
        ("seed", 2027),
        ("git_dirty", "M src/file.py"),
      ):
        put(log / "run_info.json", json.dumps(dict(meta, **{key: val})))
        with self.assertRaises(RuntimeError):
          chain.validate_predecessor(root, pipeline)
      put(log / "run_info.json", json.dumps(meta))

  def test_live_predecessor_keeps_lock_and_source_then_hands_over(self):
    with tempfile.TemporaryDirectory() as temp:
      root = Path(temp)
      repo = root / "repo"
      repo.mkdir()

      def run(*args):
        return subprocess.run(
          ["git", *args],
          cwd=repo,
          check=True,
          text=True,
          stdout=subprocess.PIPE,
          stderr=subprocess.PIPE,
        ).stdout.strip()

      run("init", "-b", "00053")
      run("config", "user.email", "test@example.invalid")
      run("config", "user.name", "Protocol test")
      put(repo / "original.txt", "original\n")
      run("add", ".")
      run("commit", "-m", "predecessor")
      previous = run("rev-parse", "HEAD")
      run("switch", "-c", "00033")
      # Exercise the actual lock-inheritance block from execute_00033.sh,
      # then a CPU-only body instead of any simulator/training.
      actual = Path(__file__).with_name("execute_00033.sh").read_text()
      lock_code = actual[
        actual.index('if [[ -n "${SMP_QUEUE_LOCK_FD:-}"') : actual.index(
          "source experiments/00033/activate_00033.sh"
        )
      ]
      venv = root / "venv/bin"
      venv.mkdir(parents=True)
      (venv / "python").symlink_to(__import__("sys").executable)
      body = (
        '#!/usr/bin/env bash\nset -euo pipefail\nSMP_G1_ROOT="$(dirname "$PWD")"\n'
        + lock_code
        + '\nprintf "started\\n" > "$SMP_G1_ROOT/started.txt"\n'
      )
      put(repo / "experiments/00033/execute_00033.sh", body)
      run("add", ".")
      run("commit", "-m", "successor")
      target = run("rev-parse", "HEAD")
      run("remote", "add", "origin", str(repo))
      run("switch", "00053")
      original = chain.acquire_shared(root)
      self.assertIsNotNone(original)
      self.assertIsNone(chain.acquire_shared(root))
      self.assertEqual(run("branch", "--show-current"), "00053")
      self.assertFalse((root / "started.txt").exists())
      original.close()
      shared = chain.acquire_shared(root)
      with shared, patch.object(chain, "PREDECESSOR_REVISION", previous):
        chain.clean_source(repo, previous, "00053")
        chain.switch_target(repo, target)
        chain.run_successor(repo, shared, root)
        self.assertEqual((root / "started.txt").read_text(), "started\n")
        # Child completion did NOT release the parent's shared lock.
        self.assertIsNone(chain.acquire_shared(root))
      released = chain.acquire_shared(root)
      self.assertIsNotNone(released)
      released.close()

  def test_failed_successor_is_not_reported_complete(self):
    with tempfile.TemporaryDirectory() as temp:
      root = Path(temp)
      repo = root / "repo"
      repo.mkdir()
      put(repo / "experiments/00033/execute_00033.sh", "#!/bin/bash\nexit 7\n")
      with chain.acquire_shared(root) as lock:
        with self.assertRaises(RuntimeError):
          chain.run_successor(repo, lock, root)
      self.assertEqual(
        (root / "00033_exit_code.txt").read_text(), "pipeline_exit_code=7\n"
      )

  def test_record_parser_rejects_duplicate_and_partial_status(self):
    with tempfile.TemporaryDirectory() as temp:
      p = Path(temp) / "status.txt"
      for contents in ("status=0\nstatus=1\n", "status\n"):
        p.write_text(contents)
        with self.assertRaises(RuntimeError):
          chain.read_pairs(p)


if __name__ == "__main__":
  unittest.main(verbosity=2)
