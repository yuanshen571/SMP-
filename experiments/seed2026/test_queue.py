"""CPU orchestration fixtures: real Git/locks; worker programs never simulate or train."""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

QUEUE = Path(__file__).with_name("execute_pair.sh").read_text(encoding="utf-8")
WORKER = r"""import hashlib,json,os,subprocess,sys
from pathlib import Path
root=Path.cwd().parent
experiment=Path(sys.argv[0]).parent.name
name=Path(sys.argv[0]).name
args=sys.argv[1:]
def arg(key,default=None):
    return args[args.index(key)+1] if key in args else default
stage=name
if name.startswith('run_'): stage='formal_train' if arg('--max-iterations')=='10000' else 'smoke_train'
if name=='verify_completion.py': stage='smoke_verify' if arg('--iterations')=='2' else 'formal_verify'
with (root/'trace.txt').open('a') as f: f.write(experiment+':'+stage+'\n')
if os.environ.get('SMP_TEST_FAIL')==experiment+':'+stage: raise SystemExit(11)
if name.startswith('run_'):
    n=int(arg('--max-iterations')); run=arg('--run')
    log=Path(arg('--log-root'))/run; ckpt=Path(arg('--ckpt-root'))/run
    log.mkdir(parents=True,exist_ok=True); ckpt.mkdir(parents=True,exist_ok=True)
    initial=ckpt/'initial_untrained.pt';initial.write_bytes(b'fixture untrained seed2026')
    h=hashlib.sha256(initial.read_bytes()).hexdigest()
    info=dict(experiment=experiment,seed=2026,completed_iterations=n,max_iterations=n,
              total_samples=n*1024*24,log_dir=str(log),ckpt_dir=str(ckpt),
              initial_models_sha256='same_fixture_model',saved_initial_checkpoint_sha256=h,
              physical_filter_enabled=experiment=='00083')
    if experiment=='00083':
        source=Path(arg('--control-info'));control=json.loads(source.read_text())
        assert control['experiment']=='0008' and control['completed_iterations']==10000
        assert source==root/'logs'/next(p.name for p in (root/'logs').glob('0008_location_*'))/'run_info.json'
        assert Path(arg('--initial-checkpoint'))==Path(control['ckpt_dir'])/'initial_untrained.pt'
        info.update(control_info=str(source),initial_checkpoint_sha256=h,noise_rescue_enabled=True,
                    noise_rescue_config=dict(max_generations=30,rms_limit=0.15))
    (log/'run_info.json').write_text(json.dumps(info))
if name=='contract.py' and arg('--out'): Path(arg('--out')).write_text('{}')
if name=='verify_completion.py' and arg('--out'): Path(arg('--out')).write_text('CPU fixture PASS')
"""


def put(path, text):
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(text, encoding="utf-8")


def fixture(root):
  repo = root / "repo"
  repo.mkdir()

  def git(*args):
    return subprocess.check_output(
      ["git", *args], cwd=repo, text=True, stderr=subprocess.DEVNULL
    ).strip()

  git("init", "-b", "0008")
  git("config", "user.email", "test@example.invalid")
  git("config", "user.name", "Queue protocol")
  put(repo / "experiments/seed2026/execute_pair.sh", QUEUE)
  files = [
    "contract.py",
    "test_variant.py",
    "test_startup.py",
    "test_preflight_cli.py",
    "preflight_control.py",
    "test_reward_formula.py",
    "test_passive_metrics.py",
    "test_initialization.py",
    "test_physics.py",
    "test_runtime.py",
    "test_scoring.py",
    "accept_sampler.py",
    "verify_completion.py",
  ]
  for exp in ("0008", "00083"):
    for name in (*files, f"run_{exp}.py", f"accept_{exp}.py"):
      put(repo / f"experiments/{exp}/{name}", WORKER)
    put(
      repo / f"experiments/{exp}/activate_{exp}.sh",
      "export PYTHONUTF8=1 PYTHONIOENCODING=utf-8\n",
    )
  spec = dict(
    experiment="00083",
    seed=2026,
    control=dict(experiment="0008", seed=2026),
    noise_rescue=dict(max_generations=30, rms_limit=0.15),
    physical_filter=dict(repair_small_joints=False),
    mixed_cache_enabled=False,
    clustering_enabled=False,
  )
  put(repo / "experiments/00083/EXPERIMENT.json", json.dumps(spec))
  git("add", ".")
  git("commit", "-m", "queue fixture baseline")
  git("switch", "-c", "00083")
  put(repo / "target.txt", "target\n")
  git("add", ".")
  git("commit", "-m", "queue fixture target")
  target = git("rev-parse", "HEAD")
  git("switch", "0008")
  git("remote", "add", "origin", str(repo))
  (root / "venv/bin").mkdir(parents=True)
  (root / "venv/bin/python").symlink_to(sys.executable)
  return repo, target, git


class QueueTests(unittest.TestCase):
  def test_serial_order_common_initial_and_failure_stop(self):
    for failure in (
      "",
      "0008:formal_train",
      "0008:formal_verify",
      "00083:formal_train",
      "00083:preflight_control.py",
    ):
      with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        repo, target, git = fixture(root)
        result = subprocess.run(
          ["bash", "experiments/seed2026/execute_pair.sh", target],
          cwd=repo,
          env=dict(os.environ, SMP_TEST_FAIL=failure),
          capture_output=True,
          text=True,
          timeout=30,
        )
        trace = (root / "trace.txt").read_text().splitlines()
        if failure:
          self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
          self.assertEqual(trace[-1], failure)
          if failure.startswith("0008:"):
            self.assertFalse(any(row.startswith("00083:") for row in trace))
            self.assertEqual(git("branch", "--show-current"), "0008")
        else:
          self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
          self.assertLess(
            trace.index("0008:formal_verify"), trace.index("00083:contract.py")
          )
          self.assertLess(
            trace.index("00083:smoke_verify"), trace.index("00083:formal_train")
          )
          self.assertLess(
            trace.index("00083:preflight_control.py"),
            trace.index("00083:accept_sampler.py"),
          )
          complete = list((root / "logs").glob("seed2026_0008_00083_*/completed.txt"))
          self.assertEqual(len(complete), 1)
          self.assertEqual(complete[0].read_text(), "0008=PASS\n00083=PASS\n")
          git("switch", "0008")
          repeated = subprocess.run(
            ["bash", "experiments/seed2026/execute_pair.sh", target],
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=10,
          )
          self.assertNotEqual(repeated.returncode, 0)
          self.assertIn("already registered", repeated.stderr)
          self.assertEqual((root / "trace.txt").read_text().splitlines(), trace)

  def test_shared_lock_blocks_another_queue(self):
    with tempfile.TemporaryDirectory() as temp:
      root = Path(temp)
      repo, target, _ = fixture(root)
      with (root / ".0003_00031_queue.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = subprocess.run(
          ["bash", "experiments/seed2026/execute_pair.sh", target],
          cwd=repo,
          capture_output=True,
          text=True,
          timeout=10,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("another SMP workflow", result.stderr)
        self.assertFalse((root / "trace.txt").exists())

  def test_wrong_target_pin_stops_before_workers(self):
    with tempfile.TemporaryDirectory() as temp:
      root = Path(temp)
      repo, _, _ = fixture(root)
      result = subprocess.run(
        ["bash", "experiments/seed2026/execute_pair.sh", "0" * 40],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=10,
      )
      self.assertNotEqual(result.returncode, 0)
      self.assertIn("differs from pinned delivery", result.stderr)
      self.assertFalse((root / "trace.txt").exists())


if __name__ == "__main__":
  unittest.main(verbosity=2)
