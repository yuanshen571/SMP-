"""Full-shape read-only source/FIFO tests; explicit fake physics/search, no GPU."""

from __future__ import annotations

import copy
import random
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R / "src"))
import paired_replay as pr  # noqa: E402
from archive_format import (  # noqa: E402
  ArchiveRecorder,
  atomic_json,
  file_sha256,
)
from initialization import initial_models_digest  # noqa: E402

from smp.pretrain.scheduler import DDPMScheduler  # noqa: E402

torch.set_num_threads(1)


class Model(torch.nn.Module):
  def forward(self, x, t):
    return x * 0.01 + t[:, None, None] * 0.00001


class AllPass:
  def process(self, x, repair=False):
    assert repair is False
    return x.copy(), np.ones(len(x), dtype=bool), {}

  def evaluate(self, x):
    return dict(passed=np.ones(len(x), dtype=bool), joint=np.zeros(len(x)))


def fake_initialize(consumer, env):
  env._smp_physics = AllPass()
  env._smp_scorer = env._smp_physics
  env._smp_noise_config = pr.SearchConfig()
  env._smp_noise_totals = dict(
    original_rows=0,
    direct_inserted=0,
    inserted=0,
    batches=0,
    attempted=0,
    succeeded=0,
    exhausted=0,
    skipped_original_inserted=0,
    early_stopped=0,
    original_fallback_inserted=0,
    discarded=0,
    archive_replay_max_abs=0.0,
  )
  env._smp_gsi_original_pass = torch.zeros(4096, dtype=torch.bool)
  env._smp_gsi_physical_pass = torch.zeros(4096, dtype=torch.bool)


class PairTests(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.tmp = tempfile.TemporaryDirectory(prefix="00094-pair-cpu-")
    cls.root = Path(cls.tmp.name)
    cls.directory = cls.root / "formal" / "source"
    cls.bundle = (
      Model(),
      DDPMScheduler(50),
      torch.tensor(-1.0),
      torch.tensor(1.0),
      59,
      10,
    )
    env = SimpleNamespace(device="cpu", _smp_bundle=cls.bundle, common_step_counter=0)
    rec = ArchiveRecorder(cls.directory, seed=102028, device="cpu", provenance={})
    env._smp_gsi_pool = torch.cat([rec.sample(env, 1024) for _ in range(4)])
    env.common_step_counter = 2400
    env._smp_gsi_pool[:1024] = rec.sample(env, 1024)
    env._smp_gsi_head = 1024
    rec.finish(env, completed_iterations=100, steps_per_env=24)
    cls.binding = {
      "archive_dir": str(cls.directory),
      "manifest_sha256": file_sha256(cls.directory / "manifest.json"),
    }
    cls.source_pool = env._smp_gsi_pool.clone()

  @classmethod
  def tearDownClass(cls):
    cls.tmp.cleanup()

  def consumer(self, name):
    return pr.PairedConsumer(self.binding, self.root / f"{name}.json", iterations=2)

  def env(self):
    return SimpleNamespace(device="cpu", _smp_bundle=self.bundle, common_step_counter=0)

  def test_fullshape_order_native_fifo_source_unchanged_and_rng_neutral(self):
    consumer = self.consumer("allpass")
    env = self.env()
    cpu = torch.get_rng_state().clone()
    py = random.getstate()
    npstate = np.random.get_state()
    before = [
      file_sha256(self.directory / e["file"]) for e in consumer.source["batches"]
    ]
    with patch.object(pr.PairedConsumer, "initialize_physics", fake_initialize):
      env._smp_gsi_pool = torch.cat([consumer.sample(env, 1024) for _ in range(4)])
      env.common_step_counter = 2400
      env._smp_gsi_pool[:1024] = consumer.sample(env, 1024)
      env._smp_gsi_head = 1024
    self.assertTrue(torch.equal(env._smp_gsi_pool, self.source_pool))
    self.assertEqual(consumer.position, 5)
    self.assertEqual(consumer.audit["rows"], 5120)
    self.assertTrue(torch.equal(cpu, torch.get_rng_state()))
    self.assertEqual(py, random.getstate())
    self.assertTrue(np.array_equal(npstate[1], np.random.get_state()[1]))
    self.assertEqual(
      before,
      [file_sha256(self.directory / e["file"]) for e in consumer.source["batches"]],
    )
    self.assertTrue(env._smp_gsi_original_pass.all())
    self.assertTrue(env._smp_gsi_physical_pass.all())

  def test_timing_and_head_mismatch_rejected(self):
    consumer = self.consumer("bad-time")
    env = self.env()
    env.common_step_counter = 1
    with self.assertRaises(AssertionError):
      consumer.load_batch(env, 1024)
    consumer.position = 4
    env.common_step_counter = 2400
    env._smp_gsi_head = 1024
    with self.assertRaises(AssertionError):
      consumer.load_batch(env, 1024)

  def test_mixed_skip_and_early_stop_reach_native_pool_in_source_order(self):
    import noise_rescue as nr
    from test_search import identity_decode
    from verify_pairing import verify_decisions

    class MixedScorer:
      def __init__(self):
        self.calls = 0

      def evaluate(self, x):
        self.calls += 1
        if self.calls == 1:
          assert len(x) == 1024
          passed = np.ones(1024, dtype=bool)
          passed[:2] = False
          joint = np.zeros(1024)
          joint[:2] = [0.01, 0.006]
        else:
          passed = np.zeros(len(x), dtype=bool)
          joint = np.full(len(x), 0.0059)
        return dict(
          passed=passed,
          joint=joint,
          numeric=np.ones(len(x), dtype=bool),
          fitness=10001 + np.abs(x[:, 0, 0]),
        )

    def initialize(consumer, env):
      fake_initialize(consumer, env)
      env._smp_scorer = MixedScorer()

    consumer = self.consumer("mixed")
    env = self.env()
    rng = torch.get_rng_state().clone()
    with (
      patch.object(pr.PairedConsumer, "initialize_physics", initialize),
      patch.object(nr, "decode", identity_decode),
    ):
      result = consumer.sample(env, 1024)
    entry = consumer.audit["batches"][0]
    verify_decisions(entry)
    self.assertEqual(entry["stats"]["skipped_original_inserted"], 1)
    self.assertEqual(entry["stats"]["early_stopped"], 1)
    self.assertEqual(entry["stats"]["generations"], 10)
    self.assertEqual(entry["original_passed"], 1022)
    raw = torch.load(self.directory / "batch_0000.pt", weights_only=False)["windows"]
    self.assertTrue(torch.equal(result[0], raw[0]))
    self.assertTrue(torch.equal(result[2:], raw[2:]))
    self.assertFalse(torch.equal(result[1], raw[1]))
    self.assertTrue(torch.equal(rng, torch.get_rng_state()))
    self.assertEqual(consumer.position, 1)

  def test_hash_mismatch_rejected_without_advancing(self):
    c = self.consumer("bad-sha")
    c.source["batches"][0]["sha256"] = "0" * 64
    with self.assertRaises(AssertionError):
      c.load_batch(self.env(), 1024)
    self.assertEqual(c.position, 0)

  def test_rng_restored_on_exception_and_source_manifest_overwrite_refused(self):
    cpu = torch.get_rng_state().clone()
    py = random.getstate()
    npstate = np.random.get_state()
    with self.assertRaises(RuntimeError):
      with pr.protect_training_rng("cpu"):
        torch.rand(7)
        np.random.rand()
        random.random()
        raise RuntimeError("fixture")
    self.assertTrue(torch.equal(cpu, torch.get_rng_state()))
    self.assertEqual(py, random.getstate())
    self.assertTrue(np.array_equal(npstate[1], np.random.get_state()[1]))
    self.consumer("repeat")
    with self.assertRaises(AssertionError):
      self.consumer("repeat")

  def test_native_sampler_patch_uninstalled(self):
    c = self.consumer("patch")

    def _ddpm_sample(env, n):
      return "original"

    module = SimpleNamespace(_ddpm_sample=_ddpm_sample)
    c.install(module)
    self.assertEqual(module._ddpm_sample, c.sample)
    c.uninstall()
    self.assertIs(module._ddpm_sample, _ddpm_sample)

  def test_source_binding_requires_runtime_fingerprint_and_completed_formal(self):
    root = self.root / "binding"
    directory = root / "formal" / "control"
    directory.mkdir(parents=True)
    logs = root / "logs"
    ckpt = root / "ckpt"
    logs.mkdir()
    ckpt.mkdir()
    info_path = logs / "run_info.json"
    initial = ckpt / "initial_untrained.pt"
    state = {"weight": torch.ones(2), "obs_normalizer.count": torch.tensor(0.0)}
    payload = dict(
      actor_state_dict=state,
      critic_state_dict=copy.deepcopy(state),
      optimizer_state_dict={"state": {}},
      infos={
        "stage": "initial_untrained",
        "experiment": "00093",
        "initialization_seed": 2028,
      },
    )
    torch.save(payload, initial)
    info = dict(
      experiment="00093",
      seed=2028,
      initialization_seed=2028,
      initialization_mode="fresh_seed",
      completed_iterations=20000,
      max_iterations=20000,
      num_envs=1024,
      num_steps_per_env=24,
      total_samples=491520000,
      ckpt_dir=str(ckpt),
      saved_initial_checkpoint_sha256=file_sha256(initial),
      initial_models_sha256=initial_models_digest(payload),
      stage="formal",
      gsi_archive_status="complete",
      gsi_generation_rng_isolated=True,
      gsi_generation_rng_seed=102028,
      gsi_recording_enabled=True,
      noise_rescue_enabled=False,
      physical_filter_enabled=False,
      mixed_cache_enabled=False,
      gsi_archive_dir=str(directory),
      run_id="control",
      prior_sha256="prior",
      uv_lock_sha256="lock",
      git_commit="commit",
    )
    (logs / "exit_code.txt").write_text("python_exit_code=0\ntee_exit_code=0\n")
    manifest = dict(
      schema=pr.SCHEMA,
      experiment="00093",
      status="complete",
      completed_iterations=20000,
      rows=208896,
      batches=[{}] * 204,
      gsi_seed=102028,
      pool_size=4096,
      batch_size=1024,
      refresh_control_steps=2400,
      provenance={
        k: info[k]
        for k in [
          "run_id",
          "stage",
          "prior_sha256",
          "uv_lock_sha256",
          "initial_models_sha256",
          "saved_initial_checkpoint_sha256",
          "git_commit",
        ]
      },
    )
    manifest["provenance"].update(
      initial_checkpoint=str(initial), run_info=str(info_path)
    )
    atomic_json(info_path, info)
    atomic_json(directory / "manifest.json", manifest)
    with patch.object(pr, "SOURCE_ROOT", root / "formal"):
      binding = pr.source_binding(info_path)
      self.assertEqual(binding["initial_models_sha256"], initial_models_digest(payload))
      bad = copy.deepcopy(info)
      bad["stage"] = "smoke"
      atomic_json(info_path, bad)
      with self.assertRaises(AssertionError):
        pr.source_binding(info_path)
      bad = copy.deepcopy(info)
      bad["initial_models_sha256"] = "wrong"
      atomic_json(info_path, bad)
      with self.assertRaises(AssertionError):
        pr.source_binding(info_path)


if __name__ == "__main__":
  unittest.main(verbosity=2)
