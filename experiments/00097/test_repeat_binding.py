"""Reject mismatched or incomplete repeat inputs before simulation starts."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

import repeat_binding as repeat

HERE = Path(__file__).resolve().parent


class BindingTest(unittest.TestCase):
  def setUp(self):
    self.tmp = tempfile.TemporaryDirectory()
    self.addCleanup(self.tmp.cleanup)
    self.root = Path(self.tmp.name)
    self.spec = json.loads((HERE / "EXPERIMENT.json").read_text())
    self.ckpt = self.root / "00095_ckpt"
    self.ckpt.mkdir()
    initial = self.ckpt / "initial_untrained.pt"
    initial.write_bytes(b"untrained-model-placeholder-for-read-only-provenance-test")
    archive = self.root / "archive"
    archive.mkdir()
    (archive / "manifest.json").write_text("{}")
    control = self.root / "00095.json"
    ci = dict(
      experiment="00095",
      stage="formal",
      seed=2026,
      run_id="original_00095",
      initial_models_sha256="tensor-fingerprint",
      ckpt_dir=str(self.ckpt),
      saved_initial_checkpoint_sha256=repeat.sha256(initial),
      gsi_archive_dir=str(archive),
    )
    control.write_text(json.dumps(ci))
    source = dict(
      control_info=str(control),
      control_info_sha256=repeat.sha256(control),
      source_run_id="original_00095",
      initial_models_sha256="tensor-fingerprint",
      initial_checkpoint=str(initial),
      initial_checkpoint_sha256=repeat.sha256(initial),
      archive_dir=str(archive),
      manifest_sha256=repeat.sha256(archive / "manifest.json"),
    )
    self.info = dict(
      experiment="00096",
      stage="formal",
      seed=2026,
      initialization_seed=2026,
      max_iterations=20000,
      completed_iterations=20000,
      total_samples=491520000,
      num_envs=1024,
      num_steps_per_env=24,
      git_commit=self.spec["repeat_protocol"]["reference_code_commit"],
      git_dirty="",
      prior_sha256=self.spec["prior_sha256"],
      uv_lock_sha256=self.spec["uv_lock_sha256"],
      gpu_module=self.spec["gpu_module"],
      physical_manifest_sha256=self.spec["physical_manifest_sha256"],
      physical_filter_enabled=True,
      noise_rescue_enabled=True,
      gpu_physical_filter_enabled=True,
      gsi_replay_enabled=True,
      gsi_recording_enabled=False,
      mixed_cache_enabled=False,
      clustering_enabled=False,
      paired_replay_status="complete",
      initialization_mode="load_00095_models_only",
      noise_rescue_config=dict(
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
      ),
      paired_source=source,
      control_info=str(control),
      initial_models_sha256="tensor-fingerprint",
      control_initial_models_sha256="tensor-fingerprint",
      control_initial_checkpoint_sha256=repeat.sha256(initial),
      run_id="original_00096",
    )
    (self.root / "exit_code.txt").write_text("python_exit_code=0\ntee_exit_code=0\n")
    (self.root / "COMPLETION.md").write_text("original report fixture")
    self.path = self.root / "run_info.json"

  def validate(self, info=None):
    self.path.write_text(json.dumps(self.info if info is None else info))
    return repeat.validate_repeat(self.path)

  def test_bind_exact_existing_inputs(self):
    result = self.validate()
    self.assertEqual(result["control_info"], self.info["control_info"])
    self.assertEqual(result["paired_source"], self.info["paired_source"])
    self.assertEqual(result, self.validate())

  def test_reject_wrong_seed_stage_identity_and_settings(self):
    for key, value in [
      ("seed", 2028),
      ("stage", "smoke"),
      ("experiment", "00095"),
      ("completed_iterations", 19999),
      ("git_commit", "other"),
      ("git_dirty", "M src/file"),
    ]:
      with self.subTest(key=key):
        bad = dict(self.info, **{key: value})
        with self.assertRaises(AssertionError):
          self.validate(bad)
    bad = copy.deepcopy(self.info)
    bad["noise_rescue_config"]["skip_joint_excess_deg"] = 0.42
    with self.assertRaises(AssertionError):
      self.validate(bad)

  def test_reject_changed_source(self):
    self.validate()
    Path(self.info["paired_source"]["archive_dir"], "manifest.json").write_text(
      '{"changed":true}'
    )
    with self.assertRaises(AssertionError):
      self.validate()

  def test_reject_failed_original(self):
    (self.root / "exit_code.txt").write_text("python_exit_code=1\ntee_exit_code=0\n")
    with self.assertRaises(AssertionError):
      self.validate()

  def test_reject_replaced_initial_model(self):
    Path(self.info["paired_source"]["initial_checkpoint"]).write_bytes(b"replaced")
    with self.assertRaises(AssertionError):
      self.validate()


if __name__ == "__main__":
  unittest.main(verbosity=2)
