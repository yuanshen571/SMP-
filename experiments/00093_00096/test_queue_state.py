"""Exercise queue provenance checks with small synthetic metadata, no training."""

import copy
import tempfile
import unittest
from pathlib import Path

import queue_state as qs


class QueueTests(unittest.TestCase):
  def setUp(self):
    self.tmp = tempfile.TemporaryDirectory(prefix="paired-queue-metadata-")
    self.root = Path(self.tmp.name)
    self.queue = self.root / "queue"
    self.queue.mkdir()
    (self.root / "logs").mkdir()
    qs.write(self.queue / "queue_progress.json", dict(completed=[], active=None))

  def tearDown(self):
    self.tmp.cleanup()

  def fixture(self, exp):
    index = qs.ORDER.index(exp)
    qs.write(
      self.queue / "queue_progress.json",
      dict(completed=list(qs.ORDER[:index]), active=exp),
    )
    execution = self.root / "logs" / (exp + "_execution_fixture")
    execution.mkdir()
    info_path = self.root / "logs" / (exp + "_formal") / "run_info.json"
    info_path.parent.mkdir()
    (info_path.parent / "COMPLETION.md").write_text("fixture PASS\n")
    (info_path.parent / "exit_code.txt").write_text(
      "python_exit_code=0\ntee_exit_code=0\n"
    )
    (execution / "exit_code.txt").write_text("execution_exit_code=0\n")
    (execution / "completed.txt").write_text(exp + "=PASS\n")
    result = dict(
      status="PASS",
      experiment=exp,
      seed=qs.SEEDS[exp],
      run_info=str(info_path),
      ckpt_dir=str(self.root / "ckpt" / exp),
      run_id=info_path.parent.name,
      execution_directory=str(execution),
    )
    info = dict(
      result,
      stage="formal",
      max_iterations=20000,
      completed_iterations=20000,
      total_samples=491520000,
      num_envs=1024,
      num_steps_per_env=24,
      initialization_seed=qs.SEEDS[exp],
      git_dirty="",
      git_commit="fixture-commit",
      initial_models_sha256="model-" + str(qs.SEEDS[exp]),
      saved_initial_checkpoint_sha256="initial-file-" + str(qs.SEEDS[exp]),
    )
    registration = dict(execution=str(execution), commit=info["git_commit"])
    if exp in qs.CONTROLS:
      previous = qs.read(self.queue / f"{qs.CONTROLS[exp]}_result.json")
      control = qs.read(previous["run_info"])
      info.update(
        control_info=previous["run_info"],
        control_initial_checkpoint_sha256=control["saved_initial_checkpoint_sha256"],
        paired_source=dict(control_info_sha256=previous["run_info_sha256"]),
        gpu_physical_filter_enabled=True,
        parent_experiment="00092",
        gpu_module=dict(pretraining_equivalence_rerun=False),
      )
      registration["control_info"] = previous["run_info"]
      qs.write(
        execution / "pairing.json",
        dict(
          status="PASS",
          rows=208896,
          shards=204,
          source=dict(control_info=previous["run_info"]),
        ),
      )
    else:
      archive = f"/root/gpufree-share/smp_gsi_paired_{exp}/formal/{info['run_id']}"
      info.update(
        initialization_mode="fresh_seed",
        repeats_experiment="0009",
        gsi_archive_status="complete",
        gsi_recording_enabled=True,
        gsi_generation_rng_seed=100000 + qs.SEEDS[exp],
        gsi_recorded_rows=208896,
        gsi_recorded_shards=204,
        gsi_archive_dir=archive,
      )
      qs.write(execution / "archive_check.json", dict(status="PASS", directory=archive))
    qs.write(info_path, info)
    qs.write(execution / "result.json", result)
    (self.root / "logs" / f".{exp}_registered").write_text(
      "".join(f"{k}={v}\n" for k, v in registration.items())
    )
    return info_path

  def test_all_four_exact_control_bindings(self):
    for exp in qs.ORDER:
      self.fixture(exp)
      qs.complete(self.queue, self.root, exp)
    self.assertEqual(
      qs.read(self.queue / "queue_progress.json")["completed"], list(qs.ORDER)
    )
    self.assertNotEqual(
      qs.read(self.queue / "00093_result.json")["run_info"],
      qs.read(self.queue / "00095_result.json")["run_info"],
    )

  def test_wrong_seed_fingerprint_and_control_rejected(self):
    self.fixture("00093")
    qs.complete(self.queue, self.root, "00093")
    path = self.fixture("00094")
    original = qs.read(path)
    for key, value in [
      ("seed", 2026),
      ("initial_models_sha256", "wrong"),
      ("control_info", str(self.root / "other_run_info.json")),
      ("parent_experiment", "00091"),
    ]:
      bad = copy.deepcopy(original)
      bad[key] = value
      qs.write(path, bad)
      with self.assertRaises((AssertionError, FileNotFoundError)):
        qs.complete(self.queue, self.root, "00094")
    qs.write(path, original)
    qs.complete(self.queue, self.root, "00094")

  def test_wrong_archive_or_incomplete_baseline_rejected(self):
    path = self.fixture("00093")
    original = qs.read(path)
    for key, value in [
      ("gsi_archive_dir", "/root/gpufree-share/smp_gsi_paired_0009/formal/old"),
      ("completed_iterations", 10000),
      ("stage", "smoke"),
      ("gsi_generation_rng_seed", 102027),
    ]:
      bad = dict(original, **{key: value})
      qs.write(path, bad)
      with self.assertRaises(AssertionError):
        qs.complete(self.queue, self.root, "00093")

  def test_missing_completion_or_bad_exit_rejected(self):
    path = self.fixture("00093")
    (path.parent / "COMPLETION.md").unlink()
    with self.assertRaises(AssertionError):
      qs.complete(self.queue, self.root, "00093")
    (path.parent / "COMPLETION.md").write_text("fixture")
    (path.parent / "exit_code.txt").write_text("python_exit_code=1\ntee_exit_code=0\n")
    with self.assertRaises(AssertionError):
      qs.complete(self.queue, self.root, "00093")

  def test_out_of_order_rejected(self):
    self.fixture("00093")
    qs.write(
      self.queue / "queue_progress.json", dict(completed=["00095"], active="00093")
    )
    with self.assertRaises(AssertionError):
      qs.complete(self.queue, self.root, "00093")


if __name__ == "__main__":
  unittest.main(verbosity=2)
