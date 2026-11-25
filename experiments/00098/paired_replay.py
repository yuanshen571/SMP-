"""Replay saved 00095 originals without physical processing or new draws."""

from __future__ import annotations

import json
import random
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from archive_format import SCHEMA, atomic_json, decode, file_sha256, tensor_sha256
from initialization import validate_control

SOURCE_ROOT = Path("/root/gpufree-share/smp_gsi_paired_00095/formal")


def source_binding(control_info):
  info_path = Path(control_info).resolve()
  info = json.loads(info_path.read_text(encoding="utf-8"))
  initial = Path(info["ckpt_dir"]).resolve() / "initial_untrained.pt"
  validate_control(initial, info_path)
  assert info["stage"] == "formal" and info["gsi_archive_status"] == "complete"
  assert (
    info["gsi_generation_rng_isolated"] and info["gsi_generation_rng_seed"] == 102026
  )
  assert info["gsi_recording_enabled"] and not info["noise_rescue_enabled"]
  assert not info["physical_filter_enabled"] and not info["mixed_cache_enabled"]
  directory = Path(info["gsi_archive_dir"]).resolve()
  root = SOURCE_ROOT.resolve()
  assert directory.parent == root and directory.name == info["run_id"]
  manifest_path = directory / "manifest.json"
  manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
  assert manifest["schema"] == SCHEMA and manifest["experiment"] == "00095"
  assert manifest["status"] == "complete" and manifest["completed_iterations"] == 20000
  assert manifest["rows"] == 208896 and len(manifest["batches"]) == 204
  assert manifest["gsi_seed"] == 102026
  assert (
    manifest["pool_size"],
    manifest["batch_size"],
    manifest["refresh_control_steps"],
  ) == (4096, 1024, 2400)
  for key in [
    "run_id",
    "stage",
    "prior_sha256",
    "uv_lock_sha256",
    "initial_models_sha256",
    "saved_initial_checkpoint_sha256",
    "git_commit",
  ]:
    assert manifest["provenance"][key] == info[key], f"source binding mismatch: {key}"
  assert Path(manifest["provenance"]["initial_checkpoint"]).resolve() == initial
  assert Path(manifest["provenance"]["run_info"]).resolve() == info_path
  return dict(
    control_info=str(info_path),
    initial_checkpoint=str(initial),
    archive_dir=str(directory),
    manifest_sha256=file_sha256(manifest_path),
    control_info_sha256=file_sha256(info_path),
    initial_models_sha256=info["initial_models_sha256"],
    initial_checkpoint_sha256=info["saved_initial_checkpoint_sha256"],
    prior_sha256=info["prior_sha256"],
    uv_lock_sha256=info["uv_lock_sha256"],
    source_run_id=info["run_id"],
  )


@contextmanager
def protect_training_rng(device):
  """Even lazy compilation/search/diagnostics cannot advance the caller RNG."""
  device = torch.device(device)
  devices = (
    []
    if device.type == "cpu"
    else [torch.cuda.current_device() if device.index is None else device.index]
  )
  python_state, numpy_state = random.getstate(), np.random.get_state()
  try:
    with torch.random.fork_rng(devices=devices):
      yield
  finally:
    random.setstate(python_state)
    np.random.set_state(numpy_state)


class BaselineConsumer:
  def __init__(self, binding, audit_path, *, iterations):
    self.binding = dict(binding)
    self.directory = Path(binding["archive_dir"])
    self.manifest_path = self.directory / "manifest.json"
    assert file_sha256(self.manifest_path) == binding["manifest_sha256"]
    self.source = json.loads(self.manifest_path.read_text())
    self.audit_path = Path(audit_path)
    assert not self.audit_path.exists(), "refuse existing replay audit"
    assert iterations in (2, 20000)
    self.iterations = iterations
    self.position = 0
    self.events = self.original = None
    self.expected_pool = torch.empty((4096, 10, 59), dtype=torch.float32)
    self.audit = dict(
      schema="smp-g1-baseline-repeat-v1",
      experiment="00098",
      status="running",
      source=binding,
      iterations=iterations,
      rows=0,
      batches=[],
    )
    atomic_json(self.audit_path, self.audit)

  def install(self, events):
    assert self.events is None and events._ddpm_sample.__name__ == "_ddpm_sample"
    self.events, self.original = events, events._ddpm_sample
    events._ddpm_sample = self.sample

  def uninstall(self):
    if self.events is not None:
      self.events._ddpm_sample = self.original
      self.events = None

  def load_batch(self, env, n):
    assert n == 1024 and self.position < len(self.source["batches"])
    i = self.position
    entry = self.source["batches"][i]
    step = int(getattr(env, "common_step_counter", 0))
    expected_step = 0 if i < 4 else (i - 3) * 2400
    head = i * 1024 if i < 4 else ((i - 4) * 1024) % 4096
    assert entry["batch_index"] == i and entry["global_row_start"] == i * 1024
    assert entry["file"] == f"batch_{i:04d}.pt"
    assert entry["rows"] == entry["decoder_batch_size"] == 1024
    assert entry["control_step"] == expected_step == step, "refresh timing mismatch"
    assert entry["fifo_head_before"] == head
    assert entry["transaction_index"] == (0 if i < 4 else i - 3)
    assert entry["offset_in_transaction"] == (i * 1024 if i < 4 else 0)
    assert entry["phase"] == ("startup" if i < 4 else "refresh")
    if i >= 4:
      assert int(getattr(env, "_smp_gsi_head", 0)) == head, "FIFO position mismatch"
    path = self.directory / entry["file"]
    assert (
      path.stat().st_size == entry["bytes"] and file_sha256(path) == entry["sha256"]
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["schema"] == SCHEMA
    for key, value in payload["metadata"].items():
      assert entry[key] == value
    for key, shape in [
      ("windows", (1024, 10, 59)),
      ("initial", (1024, 10, 59)),
      ("posterior", (49, 1024, 10, 59)),
    ]:
      value = payload[key]
      assert value.shape == shape and value.dtype == torch.float32
      assert bool(torch.isfinite(value).all()), f"non-finite source: {key}"
    assert tensor_sha256(payload["windows"]) == entry["windows_sha256"]
    return entry, payload

  @torch.no_grad()
  def sample(self, env, n):
    with protect_training_rng(env.device):
      entry, payload = self.load_batch(env, n)
      assert env._smp_bundle[1].num_timesteps == 50
      assert tuple(env._smp_bundle[4:]) == (59, 10)
      initial = payload["initial"].to(env.device)
      posterior = payload["posterior"].to(env.device)
      # Preserve the native prior-forward path/first compilation, using no new noise.
      decoded = decode(env._smp_bundle, initial, posterior)
      result = payload["windows"].to(env.device)
      error = float((decoded - result).abs().max())
      assert bool(torch.isfinite(decoded).all()) and error <= 1e-5, (
        "fixed-noise replay mismatch"
      )
      digest = tensor_sha256(result)
      assert digest == entry["windows_sha256"], (
        "baseline must insert exact original windows"
      )
      head = entry["fifo_head_before"]
      self.expected_pool[head : head + 1024].copy_(payload["windows"])
      self.audit["batches"].append(
        dict(
          batch_index=self.position,
          global_row_start=entry["global_row_start"],
          control_step=entry["control_step"],
          fifo_head_before=head,
          source_sha256=entry["sha256"],
          raw_windows_sha256=entry["windows_sha256"],
          output_sha256=digest,
          archive_replay_max_abs=error,
          inserted=n,
        )
      )
      self.position += 1
      self.audit["rows"] += n
      atomic_json(self.audit_path, self.audit)
    print(
      f"[00098 replay] batch={self.position} original_start={entry['global_row_start']} step={entry['control_step']} exact_original_inserted={n}",
      flush=True,
    )
    return result

  def finish(self, env, *, completed_iterations, steps_per_env):
    assert completed_iterations == self.iterations and steps_per_env == 24
    assert int(env.common_step_counter) == completed_iterations * steps_per_env
    expected = 4 + completed_iterations * steps_per_env // 2400
    assert self.position == expected and self.audit["rows"] == expected * 1024
    assert file_sha256(self.manifest_path) == self.binding["manifest_sha256"]
    assert (
      file_sha256(self.binding["control_info"]) == self.binding["control_info_sha256"]
    )
    actual_digest = tensor_sha256(env._smp_gsi_pool)
    assert actual_digest == tensor_sha256(self.expected_pool)
    head = ((expected - 4) * 1024) % 4096
    assert int(getattr(env, "_smp_gsi_head", 0)) == head
    if self.iterations == 20000:
      assert actual_digest == self.source["final_pool_sha256"]
    assert not hasattr(env, "_smp_physics") and not hasattr(env, "_smp_noise_totals")
    self.audit.update(
      status="complete",
      final_pool_sha256=actual_digest,
      final_fifo_head=head,
      new_candidates_generated=0,
      physically_processed_rows=0,
      noise_search_attempted=0,
    )
    atomic_json(self.audit_path, self.audit)
