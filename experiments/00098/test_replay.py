"""Actual-shape baseline replay tests: unchanged windows, FIFO and global RNG."""

import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import paired_replay as pr
import torch
from archive_format import SCHEMA, atomic_json, file_sha256, tensor_sha256


class ReplayTests(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    torch.set_num_threads(1)
    cls.tmp = tempfile.TemporaryDirectory()
    cls.directory = Path(cls.tmp.name)
    cls.control = cls.directory / "control.json"
    cls.control.write_text("{}")
    entries = []
    for i in range(5):
      meta = dict(
        batch_index=i,
        global_row_start=i * 1024,
        rows=1024,
        decoder_batch_size=1024,
        control_step=0 if i < 4 else (i - 3) * 2400,
        fifo_head_before=i * 1024 if i < 4 else ((i - 4) * 1024) % 4096,
        transaction_index=0 if i < 4 else i - 3,
        offset_in_transaction=i * 1024 if i < 4 else 0,
        phase="startup" if i < 4 else "refresh",
      )
      windows = torch.full((1024, 10, 59), float(i))
      payload = dict(
        schema=SCHEMA,
        metadata=meta,
        windows=windows,
        initial=windows,
        posterior=torch.tensor(0.0).expand(49, 1024, 10, 59),
      )
      name = f"batch_{i:04d}.pt"
      path = cls.directory / name
      torch.save(payload, path)
      entries.append(
        dict(
          meta,
          file=name,
          bytes=path.stat().st_size,
          sha256=file_sha256(path),
          windows_sha256=tensor_sha256(windows),
        )
      )
    manifest = dict(schema=SCHEMA, batches=entries)
    atomic_json(cls.directory / "manifest.json", manifest)
    cls.binding = dict(
      archive_dir=str(cls.directory),
      manifest_sha256=file_sha256(cls.directory / "manifest.json"),
      control_info=str(cls.control),
      control_info_sha256=file_sha256(cls.control),
    )

  @classmethod
  def tearDownClass(cls):
    cls.tmp.cleanup()

  def consumer(self):
    output = tempfile.TemporaryDirectory()
    self.addCleanup(output.cleanup)
    return pr.BaselineConsumer(
      self.binding, Path(output.name) / "audit.json", iterations=2
    )

  def env(self):
    return SimpleNamespace(
      device="cpu",
      common_step_counter=0,
      _smp_bundle=(None, SimpleNamespace(num_timesteps=50), None, None, 59, 10),
    )

  @staticmethod
  def fake_decode(bundle, initial, posterior):
    # Simulate compilation/library consumption: the sampler must protect all streams.
    torch.rand(3)
    random.random()
    np.random.rand(3)
    return initial.clone()

  def test_exact_windows_and_rng_preservation(self):
    c, env = self.consumer(), self.env()
    torch.manual_seed(2026)
    random.seed(2026)
    np.random.seed(2026)
    before = (torch.get_rng_state().clone(), random.getstate(), np.random.get_state())
    chunks = []
    with patch.object(pr, "decode", self.fake_decode):
      for i in range(4):
        value = c.sample(env, 1024)
        self.assertTrue(torch.equal(value, torch.full_like(value, float(i))))
        self.assertEqual(
          c.audit["batches"][-1]["output_sha256"],
          c.source["batches"][i]["windows_sha256"],
        )
        chunks.append(value)
    self.assertTrue(torch.equal(before[0], torch.get_rng_state()))
    self.assertEqual(before[1], random.getstate())
    self.assertTrue(np.array_equal(before[2][1], np.random.get_state()[1]))
    env._smp_gsi_pool = torch.cat(chunks)
    env.common_step_counter = 48
    c.finish(env, completed_iterations=2, steps_per_env=24)
    self.assertEqual(c.audit["status"], "complete")
    self.assertEqual(c.audit["new_candidates_generated"], 0)
    self.assertFalse(hasattr(env, "_smp_physics"))
    # Round-trip the completion audit; reject changed output identities.
    import verify_replay

    source = dict(
      self.binding,
      initial_models_sha256="same-model",
      initial_checkpoint_sha256="same-file",
      source_run_id="original_00095",
    )
    c.audit["source"] = source
    atomic_json(c.audit_path, c.audit)
    info = dict(
      experiment="00098",
      repeats_experiment="00095",
      stage="smoke",
      seed=2026,
      initialization_mode="load_00095_models_only",
      physical_filter_enabled=False,
      noise_rescue_enabled=False,
      gsi_recording_enabled=False,
      gsi_replay_enabled=True,
      control_info=source["control_info"],
      paired_source=source,
      initial_models_sha256="same-model",
      control_initial_models_sha256="same-model",
      control_initial_checkpoint_sha256="same-file",
      paired_replay_audit=str(c.audit_path),
      paired_replayed_shards=4,
      paired_replayed_rows=4096,
      paired_replay_status="complete",
    )
    atomic_json(c.audit_path.parent / "run_info.json", info)
    with patch.object(verify_replay, "source_binding", return_value=source):
      self.assertEqual(
        verify_replay.verify(c.audit_path.parent, iterations=2)["status"], "PASS"
      )
      c.audit["batches"][0]["output_sha256"] = "tampered"
      atomic_json(c.audit_path, c.audit)
      with self.assertRaises(AssertionError):
        verify_replay.verify(c.audit_path.parent, iterations=2)

  def test_refresh_timing_and_fifo(self):
    c, env = self.consumer(), self.env()
    c.position = 4
    with patch.object(pr, "decode", self.fake_decode):
      with self.assertRaises(AssertionError):
        c.sample(env, 1024)
      env.common_step_counter, env._smp_gsi_head = 2400, 1024
      with self.assertRaises(AssertionError):
        c.sample(env, 1024)
      env._smp_gsi_head = 0
      value = c.sample(env, 1024)
    self.assertTrue(torch.equal(value, torch.full_like(value, 4.0)))
    self.assertTrue(torch.equal(c.expected_pool[:1024], value))

  def test_reject_decoded_mismatch(self):
    c, env = self.consumer(), self.env()
    with patch.object(pr, "decode", lambda bundle, initial, posterior: initial + 0.01):
      with self.assertRaises(AssertionError):
        c.sample(env, 1024)
    self.assertEqual(c.position, 0)

  def test_reject_tampered_batch_or_manifest(self):
    c, env = self.consumer(), self.env()
    c.source["batches"][0]["sha256"] = "0" * 64
    with self.assertRaises(AssertionError):
      c.load_batch(env, 1024)
    with self.assertRaises(AssertionError):
      pr.BaselineConsumer(
        dict(self.binding, manifest_sha256="bad"),
        self.directory / "unused.json",
        iterations=2,
      )

  def test_fixed_noise_decoder_matches_parent(self):
    import ast

    here = Path(__file__).resolve().parent

    def body(path):
      tree = ast.parse(path.read_text())
      return ast.dump(
        next(
          n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "decode"
        )
      )

    self.assertEqual(
      body(here / "archive_format.py"), body(here.parent / "00095/gsi_archive.py")
    )


if __name__ == "__main__":
  unittest.main(verbosity=2)
