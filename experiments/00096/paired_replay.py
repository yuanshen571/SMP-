"""Read-only 00095 source replay and 00033 rescue at native sampler call sites."""

from __future__ import annotations

import json
import random
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from archive_format import SCHEMA, atomic_json, file_sha256, tensor_sha256
from initialization import validate_control
from noise_rescue import SearchConfig, decode, search_batch

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


class PairedConsumer:
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
    self.audit = dict(
      schema="smp-g1-paired-rescue-v1",
      experiment="00096",
      status="running",
      source=binding,
      rows=0,
      batches=[],
      iterations=iterations,
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
    assert n == 1024
    assert self.position < len(self.source["batches"]), (
      "no more archived raw candidates"
    )
    entry = self.source["batches"][self.position]
    i = self.position
    step = int(getattr(env, "common_step_counter", 0))
    expected_step = 0 if i < 4 else (i - 3) * 2400
    expected_head = i * 1024 if i < 4 else ((i - 4) * 1024) % 4096
    assert entry["batch_index"] == i and entry["global_row_start"] == i * 1024
    assert entry["file"] == f"batch_{i:04d}.pt"
    assert entry["control_step"] == expected_step == step, "refresh timing mismatch"
    assert entry["fifo_head_before"] == expected_head
    assert entry["transaction_index"] == (0 if i < 4 else i - 3)
    assert entry["offset_in_transaction"] == (i * 1024 if i < 4 else 0)
    assert entry["phase"] == ("startup" if i < 4 else "refresh")
    if i >= 4:
      assert int(getattr(env, "_smp_gsi_head", 0)) == expected_head
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
      t = payload[key]
      assert t.shape == shape and t.dtype == torch.float32 and torch.isfinite(t).all()
    assert tensor_sha256(payload["windows"]) == entry["windows_sha256"]
    return entry, payload

  def initialize_physics(self, env):
    from gpu_filter import GpuPhysicalFilter
    from gpu_scorer import GpuScorer
    from gsi_physics import FilterConfig

    spec = json.loads(Path(__file__).with_name("EXPERIMENT.json").read_text())
    assert torch.device(env.device).type == "cuda", (
      "00096 requires the CUDA fp64 filter"
    )
    env._smp_physics = GpuPhysicalFilter(
      env.sim.mj_model,
      FilterConfig(
        repair=False,
        control_dt=float(env.cfg.sim.mujoco.timestep) * int(env.cfg.decimation),
      ),
      device=str(env.device),
    )
    env._smp_scorer = GpuScorer(env._smp_physics)
    assert env._smp_physics.manifest["sha256"] == spec["physical_manifest_sha256"]
    assert len(env._smp_physics.pairs) == 469
    env._smp_noise_config = SearchConfig(inference_batch_size=1024)
    env._smp_noise_config.validate()
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
      joint_gate_exempt=0,
      joint_improvement_checked=0,
      joint_improvement_continued=0,
      proposals=0,
      effective_candidates=0,
      radius_rejected=0,
      search_decode_calls=0,
      search_decoded_slots=0,
      replay_decode_calls=0,
      replay_decoded_slots=0,
      generations=0,
      success_generation_sum=0,
      success_generation_max=0,
      max_selected_rms=0.0,
      replay_max_abs=0.0,
      search_seconds=0.0,
      archive_replay_max_abs=0.0,
      original_fallback_inserted=0,
      discarded=0,
    )
    env._smp_gsi_original_pass = torch.zeros(4096, dtype=torch.bool, device=env.device)
    env._smp_gsi_physical_pass = torch.zeros(4096, dtype=torch.bool, device=env.device)

  @torch.no_grad()
  def sample(self, env, n):
    assert self.audit["status"] == "running"
    # All stochastic search is scoped, including any first-call compilation.
    with protect_training_rng(env.device):
      return self._sample(env, n)

  def _sample(self, env, n):
    entry, payload = self.load_batch(env, n)
    if self.position == 0:
      self.initialize_physics(env)
    assert (
      tuple(env._smp_bundle[4:]) == (59, 10) and env._smp_bundle[1].num_timesteps == 50
    )
    raw = payload["windows"].to(env.device)
    initial = payload["initial"].to(env.device)
    posterior = payload["posterior"].to(env.device)
    replay = decode(env._smp_bundle, initial, posterior)
    error = float((replay - raw).abs().max())
    assert error <= 1e-5, f"archived source/prior replay differs: {error}"
    original_array = raw.cpu().numpy()
    original_metrics = env._smp_scorer.evaluate(original_array)
    original_passed = original_metrics["passed"]
    decisions = []
    output, final_passed, stats = search_batch(
      env._smp_bundle,
      raw,
      initial,
      posterior,
      original_passed,
      env._smp_scorer,
      seed=2026,
      global_start=entry["global_row_start"],
      config=env._smp_noise_config,
      original_metrics=original_metrics,
      decisions=decisions,
    )
    assert np.isfinite(output).all() and output.shape == (1024, 10, 59)
    assert np.array_equal(output[original_passed], original_array[original_passed])
    skipped = (~original_passed) & (original_metrics["joint"] > np.deg2rad(0.42))
    assert np.array_equal(output[skipped], original_array[skipped])
    result = torch.as_tensor(output.copy(), device=env.device, dtype=torch.float32)
    # The native caller still owns actual GSI FIFO writes and reset sampling.
    head = entry["fifo_head_before"]
    env._smp_gsi_original_pass[head : head + n] = torch.as_tensor(
      original_passed, device=env.device
    )
    env._smp_gsi_physical_pass[head : head + n] = torch.as_tensor(
      final_passed, device=env.device
    )
    totals = env._smp_noise_totals
    for key, value in stats.items():
      totals[key] = (
        max(totals.get(key, 0), value)
        if key in ("max_selected_rms", "replay_max_abs", "success_generation_max")
        else totals.get(key, 0) + value
      )
    totals["original_rows"] += n
    totals["inserted"] += n
    totals["direct_inserted"] += int(original_passed.sum())
    totals["batches"] += 1
    totals["archive_replay_max_abs"] = max(totals["archive_replay_max_abs"], error)
    assert (
      totals["direct_inserted"]
      + totals["skipped_original_inserted"]
      + totals["attempted"]
      == totals["inserted"]
    )
    assert (
      totals["succeeded"] + totals["early_stopped"] + totals["exhausted"]
      == totals["attempted"]
    )
    self.audit["batches"].append(
      dict(
        batch_index=self.position,
        global_row_start=entry["global_row_start"],
        control_step=entry["control_step"],
        fifo_head_before=head,
        source_sha256=entry["sha256"],
        raw_windows_sha256=entry["windows_sha256"],
        output_sha256=tensor_sha256(result),
        original_passed=int(original_passed.sum()),
        final_passed=int(final_passed.sum()),
        stats=stats,
        decisions=sorted(decisions, key=lambda d: d["original_row"]),
        archive_replay_max_abs=error,
      )
    )
    self.position += 1
    self.audit["rows"] += n
    atomic_json(self.audit_path, self.audit)
    print(
      f"[00096 paired] batch={self.position} step={entry['control_step']} original_start={entry['global_row_start']} direct={int(original_passed.sum())} skipped_original={stats['skipped_original_inserted']} rescued={stats['succeeded']} early_stopped_best_inserted={stats['early_stopped']} exhausted_best_inserted={stats['exhausted']}",
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
    self.audit.update(
      status="complete",
      final_pool_sha256=tensor_sha256(env._smp_gsi_pool),
      final_fifo_head=int(getattr(env, "_smp_gsi_head", 0)),
      noise_totals=dict(env._smp_noise_totals),
    )
    atomic_json(self.audit_path, self.audit)
