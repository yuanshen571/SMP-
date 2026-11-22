"""Run real shell queue/children with explicitly fake GPU/training workers.

No real archive, shared-storage write, simulator or policy training is performed.
"""

import fcntl
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[2]
WORKER = r"""
import hashlib,json,os,runpy,subprocess,sys
from pathlib import Path
args=sys.argv[1:]
def val(name): return args[args.index(name)+1]
def save(path,obj):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,indent=2)+'\n')
if args[0]=='-c':
 os.execv(os.environ['REAL_PYTHON'],[os.environ['REAL_PYTHON'],*args])
if args[0]=='-':
 code=sys.stdin.read()
 if 'shutil.rmtree' in code:
  # This is only the verified-smoke cleanup body. No actual tensors exist.
  execution=Path(args[2]);save(execution/'smoke_archive_manifest.json',{'fake':True})
  (execution/'smoke_archive_cleanup.txt').write_text('FAKE CPU harness: no real archive existed\n')
  sys.exit(0)
 sys.argv=['-',*args[1:]];exec(compile(code,'<real-launcher-result-body>','exec'));sys.exit(0)
script=Path(args[0]);name=script.name
if name=='queue_state.py':
 if args[1]=='preflight':
  # Only storage/GPU preflight is fake; provenance and stage checks below are real.
  save(Path(val('--queue'))/'queue_progress.json',dict(order=['00093','00094','00095','00096'],completed=[],active=None,status='prepared'))
  sys.exit(0)
 sys.argv=args;sys.path.insert(0,str(script.resolve().parent));runpy.run_path(str(script),run_name='__main__');sys.exit(0)
exp=script.parent.name
with open(os.environ['CALLS'],'a') as f:f.write(json.dumps(args)+'\n')
isrun=name.startswith('run_')
stage=val('--stage') if '--stage' in args else ''
if isrun and os.environ.get('FAIL_STAGE')==exp+':'+stage:sys.exit(17)
if name=='verify_archive.py' and os.environ.get('FAIL_STAGE')==exp+':archive' and val('--iterations')=='20000':sys.exit(17)
if isrun:
 seed=int(val('--seed'));iters=int(val('--max-iterations'));run=val('--run')
 log=Path(val('--log-root'))/run;ckpt=Path(val('--ckpt-root'))/run;ckpt.mkdir()
 commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
 info=dict(experiment=exp,seed=seed,initialization_seed=seed,stage=stage,run_id=run,
  log_dir=str(log),ckpt_dir=str(ckpt),max_iterations=iters,completed_iterations=iters,
  total_samples=iters*1024*24,num_envs=1024,num_steps_per_env=24,git_dirty='',git_commit=commit,
  initial_models_sha256='models-'+str(seed),saved_initial_checkpoint_sha256='file-'+str(seed))
 if exp in ('00093','00095'):
  info.update(initialization_mode='fresh_seed',repeats_experiment='0009',gsi_archive_status='complete',
   gsi_recording_enabled=True,gsi_generation_rng_seed=100000+seed,
   gsi_recorded_rows=208896 if stage=='formal' else 4096,gsi_recorded_shards=204 if stage=='formal' else 4,
   gsi_archive_dir=f'/root/gpufree-share/smp_gsi_paired_{exp}/{stage}/{run}')
 else:
  control=Path(val('--control-info')).resolve();previous=json.loads(control.read_text())
  expected='00093' if exp=='00094' else '00095'
  assert previous['experiment']==expected and previous['seed']==seed
  info.update(control_info=str(control),control_initial_checkpoint_sha256=previous['saved_initial_checkpoint_sha256'],
   paired_source=dict(control_info_sha256=hashlib.sha256(control.read_bytes()).hexdigest()),
   gpu_physical_filter_enabled=True,parent_experiment='00092',gpu_module=dict(pretraining_equivalence_rerun=False))
 if os.environ.get('CORRUPT_STAGE')==exp+':'+stage:info['seed']=2027
 save(log/'run_info.json',info)
if '--out' in args:
 out=Path(val('--out'));out.parent.mkdir(parents=True,exist_ok=True)
 if name=='verify_archive.py':
  info=json.loads((Path(val('--log-dir'))/'run_info.json').read_text())
  save(out,dict(status='PASS',directory=info['gsi_archive_dir']))
 elif name=='verify_pairing.py':
  info=json.loads((Path(val('--log-dir'))/'run_info.json').read_text())
  save(out,dict(status='PASS',rows=208896,shards=204,source=dict(control_info=info['control_info'])))
 else:out.write_text('FAKE CPU worker result\n')
"""


class LauncherTests(unittest.TestCase):
  def run_fixture(self, *, failure=None, corrupt=None):
    with tempfile.TemporaryDirectory(prefix="paired-queue-shell-") as folder:
      root = Path(folder)
      repo = root / "repo"
      for exp in ("00093", "00094", "00095", "00096", "00093_00096"):
        shutil.copytree(
          SOURCE / "experiments" / exp,
          repo / "experiments" / exp,
          ignore=shutil.ignore_patterns("__pycache__"),
        )
      (root / "venv/bin").mkdir(parents=True)
      (root / "env.sh").write_text("# Explicit CPU harness, not a server environment\n")
      worker = root / "venv/bin/python"
      worker.write_text("#!" + sys.executable + "\n" + WORKER)
      worker.chmod(0o755)

      def git(*args):
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)

      git("init", "-b", "00093-00096")
      git("config", "user.name", "CPU Harness")
      git("config", "user.email", "harness@example.invalid")
      git("add", ".")
      git("commit", "-m", "explicit fake-worker test fixture")
      env = dict(
        os.environ, CALLS=str(root / "calls.jsonl"), REAL_PYTHON=sys.executable
      )
      if failure:
        env["FAIL_STAGE"] = failure
      if corrupt:
        env["CORRUPT_STAGE"] = corrupt
      command = ["bash", "experiments/00093_00096/execute_queue.sh"]
      output = subprocess.run(
        command, cwd=repo, env=env, capture_output=True, text=True, timeout=90
      )
      calls = [
        json.loads(line) for line in (root / "calls.jsonl").read_text().splitlines()
      ]
      runs = [c for c in calls if Path(c[0]).name.startswith("run_")]
      formal = [c for c in runs if c[c.index("--stage") + 1] == "formal"]
      queue = next((root / "logs").glob("00093_00096_queue_*"))
      if failure or corrupt:
        self.assertNotEqual(output.returncode, 0, output.stdout + output.stderr)
        self.assertFalse((queue / "completed.txt").exists())
        failed_exp = (failure or corrupt).split(":")[0]
        for later in ("00093", "00094", "00095", "00096")[
          ("00093", "00094", "00095", "00096").index(failed_exp) + 1 :
        ]:
          self.assertFalse(any(Path(c[0]).parent.name == later for c in runs))
        return
      self.assertEqual(output.returncode, 0, output.stdout + output.stderr)
      self.assertEqual(
        [Path(c[0]).parent.name for c in formal], ["00093", "00094", "00095", "00096"]
      )
      self.assertEqual(
        [c[c.index("--seed") + 1] for c in formal], ["2028", "2028", "2026", "2026"]
      )
      self.assertEqual(
        [c[c.index("--max-iterations") + 1] for c in runs], ["2", "20000"] * 4
      )
      self.assertFalse(
        any(
          Path(c[0]).name
          in (
            "accept_gpu.py",
            "test_gpu_cpu_reference.py",
            "gpu_validate.py",
            "gpu_score_validate.py",
          )
          for c in calls
        )
      )
      for base, rescue in ((formal[0], formal[1]), (formal[2], formal[3])):
        base_info = str(
          Path(base[base.index("--log-root") + 1])
          / base[base.index("--run") + 1]
          / "run_info.json"
        )
        self.assertEqual(rescue[rescue.index("--control-info") + 1], base_info)
      self.assertEqual(
        json.loads((queue / "queue_progress.json").read_text())["status"], "complete"
      )
      self.assertTrue((queue / "SUMMARY.md").is_file())
      previous = (root / "calls.jsonl").read_bytes()
      again = subprocess.run(command, cwd=repo, env=env, capture_output=True, text=True)
      self.assertEqual(again.returncode, 2)
      self.assertIn("already registered", again.stderr)
      self.assertEqual((root / "calls.jsonl").read_bytes(), previous)
      # A competing workflow must also be blocked before any worker starts.
      with (root / ".0003_00031_queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        blocked = subprocess.run(
          command, cwd=repo, env=env, capture_output=True, text=True
        )
        self.assertEqual(blocked.returncode, 2)
        self.assertIn("shared lock", blocked.stderr)

  def test_full_queue_pairing_inherited_lock_and_no_equivalence_rerun(self):
    self.run_fixture()

  def test_each_stage_failure_stops_all_later_stages(self):
    for exp in ("00093", "00094", "00095", "00096"):
      with self.subTest(experiment=exp):
        self.run_fixture(failure=exp + ":formal")

  def test_archive_verification_failure_blocks_rescue(self):
    self.run_fixture(failure="00093:archive")

  def test_corrupt_seed_is_rejected_before_next_stage(self):
    self.run_fixture(corrupt="00094:formal")


if __name__ == "__main__":
  unittest.main(verbosity=2)
