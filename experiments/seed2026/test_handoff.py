"""CPU-only: real process identities, Git branches and flock; training is a fixture."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import chain_after_00073 as chain
from test_queue import fixture, put


def predecessor(root, log, revision):
  info = dict(
    experiment="00073",
    seed=2030,
    max_iterations=10000,
    num_envs=1024,
    num_steps_per_env=24,
    git_commit=revision,
    git_branch="00073",
    git_dirty="",
    log_dir=str(log),
    ckpt_dir=str(root / "ckpt" / log.name),
    noise_rescue_config=dict(max_generations=30, rms_limit=0.15),
    completed_iterations=10000,
    total_samples=245760000,
  )
  put(log / "run_info.json", json.dumps(info))
  put(log / "exit_code.txt", "python_exit_code=0\ntee_exit_code=0\n")
  return info


def wait_for(predicate, timeout=8):
  deadline = time.monotonic() + timeout
  while time.monotonic() < deadline:
    if predicate():
      return
    time.sleep(0.02)
  raise AssertionError("CPU fixture timed out")


class HandoffTests(unittest.TestCase):
  def test_live_process_and_shared_lock_both_gate_real_pair(self):
    for fail_verify in (False, True):
      with self.subTest(fail_verify=fail_verify), tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        repo, rescue, git = fixture(root)
        source = Path(chain.__file__).read_text(encoding="utf-8")
        put(repo / chain.WATCHER_PATH, source)
        git("add", ".")
        git("commit", "-m", "handoff fixture")
        baseline = git("rev-parse", "HEAD")
        git("switch", "-c", "00073")
        put(
          repo / "experiments/00073/verify_completion.py",
          "import sys\nraise SystemExit(" + ("7" if fail_verify else "0") + ")\n",
        )
        git("add", ".")
        git("commit", "-m", "predecessor fixture")
        previous = git("rev-parse", "HEAD")
        log = root / "logs/00073_location_seed2030_fixture"
        predecessor(root, log, previous)
        watcher = root / "tools/watcher.py"
        put(watcher, source)
        learner = subprocess.Popen(
          [
            sys.executable,
            "-c",
            "import time;time.sleep(60)",
            str(repo / "experiments/00073/run_00073.py"),
            "--run",
            log.name,
          ]
        )
        args = [
          sys.executable,
          str(watcher),
          "--root",
          str(root),
          "--run",
          str(log),
          "--wait-pid",
          str(learner.pid),
          "--target-revision",
          baseline,
          "--rescue-revision",
          rescue,
          "--poll-seconds",
          "1",
        ]
        state = root / "logs/handoff_00073_to_seed2026/state.json"
        proc = subprocess.Popen(
          args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        shared = None
        try:
          wait_for(state.exists)
          self.assertEqual(json.loads(state.read_text())["phase"], "waiting")
          time.sleep(1.1)
          self.assertEqual(git("branch", "--show-current"), "00073")
          self.assertFalse((root / "trace.txt").exists())
          shared = chain.acquire_shared(root)
          self.assertIsNotNone(shared)
          learner.terminate()
          learner.wait(timeout=5)
          time.sleep(1.1)
          self.assertEqual(git("branch", "--show-current"), "00073")
          self.assertFalse((root / "trace.txt").exists())
          shared.close()
          shared = None
          output = proc.communicate(timeout=30)[0]
          self.assertEqual(proc.returncode, 2 if fail_verify else 0, output)
          self.assertEqual(
            json.loads(state.read_text())["phase"],
            "stopped" if fail_verify else "complete",
          )
          if fail_verify:
            self.assertFalse((root / "trace.txt").exists())
            self.assertEqual(git("branch", "--show-current"), "00073")
          else:
            trace = (root / "trace.txt").read_text().splitlines()
            self.assertLess(
              trace.index("0008:formal_verify"), trace.index("00083:contract.py")
            )
            self.assertEqual(
              len(list((root / "logs").glob("seed2026_0008_00083_*/completed.txt"))), 1
            )
            repeated = subprocess.run(args, capture_output=True, text=True, timeout=10)
            self.assertNotEqual(repeated.returncode, 0)
            self.assertEqual((root / "trace.txt").read_text().splitlines(), trace)
        finally:
          if shared is not None:
            shared.close()
          for process in (learner, proc):
            if process.poll() is None:
              process.kill()
            process.communicate(timeout=5)

  def test_failed_incomplete_and_wrong_run_rejected(self):
    with tempfile.TemporaryDirectory() as temp:
      root = Path(temp)
      log = root / "logs/00073_location_seed2030_fixture"
      original = predecessor(root, log, "a" * 40)
      self.assertEqual(
        chain.validate_finished(root, log, "a" * 40), root / "ckpt" / log.name
      )
      for edit in (
        dict(seed=2026),
        dict(completed_iterations=9999),
        dict(git_dirty="M src/file.py"),
        dict(total_samples=1),
      ):
        put(log / "run_info.json", json.dumps(dict(original, **edit)))
        with self.assertRaises(RuntimeError):
          chain.validate_finished(root, log, "a" * 40)
      put(log / "run_info.json", json.dumps(original))
      for code in (
        "python_exit_code=1\ntee_exit_code=0\n",
        "python_exit_code=0\ntee_exit_code=1\n",
        "python_exit_code=0\n",
        "python_exit_code=0\npython_exit_code=1\ntee_exit_code=0\n",
      ):
        put(log / "exit_code.txt", code)
        with self.assertRaises(RuntimeError):
          chain.validate_finished(root, log, "a" * 40)

  def test_pid_reuse_or_zombie_is_not_original_active_process(self):
    proc = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(60)"])
    try:
      original = chain.process_record(proc.pid)
      self.assertIsNotNone(original)
      self.assertEqual(chain.process_record(proc.pid), original)
      self.assertNotEqual(
        dict(original, start_ticks="different"), chain.process_record(proc.pid)
      )
      proc.terminate()
      proc.wait(timeout=5)
      self.assertIsNone(chain.process_record(proc.pid))
    finally:
      if proc.poll() is None:
        proc.kill()
      proc.wait(timeout=5)
    with patch.object(chain.Path, "read_text", return_value="1 (dummy) Z " + "0 " * 25):
      self.assertIsNone(chain.process_record(1))

  def test_wrong_process_does_not_arm(self):
    with self.assertRaisesRegex(RuntimeError, "exact 00073 learner"):
      chain.require_matching_learner(
        [dict(pid=os.getpid())], Path("/wrong/run"), Path("/wrong/repo")
      )

  def test_inherited_lock_is_continuous_and_wrong_file_rejected(self):
    for correct in (True, False):
      with self.subTest(correct=correct), tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        repo, rescue, _ = fixture(root)
        shared = chain.acquire_shared(root)
        self.assertIsNotNone(shared)
        wrong = (root / "other.lock").open("a")
        passed = shared if correct else wrong
        try:
          result = subprocess.run(
            ["bash", "experiments/seed2026/execute_pair.sh", rescue],
            cwd=repo,
            env=dict(os.environ, SMP_QUEUE_LOCK_FD=str(passed.fileno())),
            pass_fds=(passed.fileno(),),
            capture_output=True,
            text=True,
            timeout=30,
          )
          if correct:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
          else:
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("wrong inherited queue lock", result.stderr)
            self.assertFalse((root / "trace.txt").exists())
          self.assertIsNone(chain.acquire_shared(root))
        finally:
          wrong.close()
          shared.close()


if __name__ == "__main__":
  unittest.main(verbosity=2)
