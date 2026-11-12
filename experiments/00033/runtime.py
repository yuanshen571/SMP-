"""All-original-row admission for 00033; atomic batches, no rejection refill."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from noise_rescue import PhysicalScorer, SearchConfig, draw_recorded, search_batch


def initialize(env, *, batch_size):
  from smp.rl.gsi_physics_runtime import initialize as initialize_physics

  initialize_physics(env, repair=False, batch_size=batch_size)
  spec = json.loads(
    Path(__file__).with_name("EXPERIMENT.json").read_text(encoding="utf-8")
  )
  assert env._smp_physics.manifest["sha256"] == spec["physical_manifest_sha256"], (
    "G1 geometry/limits differ from the pinned 0003 model"
  )
  env._smp_noise_config = SearchConfig(inference_batch_size=batch_size)
  env._smp_noise_config.validate()
  env._smp_noise_totals = dict(
    original_rows=0,
    direct_inserted=0,
    attempted=0,
    succeeded=0,
    exhausted=0,
    inserted=0,
    proposals=0,
    effective_candidates=0,
    radius_rejected=0,
    search_decode_calls=0,
    search_decoded_slots=0,
    replay_decode_calls=0,
    replay_decoded_slots=0,
    raw_decode_calls=0,
    generations=0,
    success_generation_sum=0,
    success_generation_max=0,
    max_selected_rms=0.0,
    replay_max_abs=0.0,
    search_seconds=0.0,
    generation_seconds=0.0,
    collection_seconds=0.0,
    original_fallback_inserted=0,
    discarded=0,
  )


def collect(env, count):
  config = env._smp_noise_config
  B = config.inference_batch_size
  if count <= 0 or count % B:
    raise ValueError("00033 uses whole fixed decoder batches; no surplus raw rows")
  started = time.perf_counter()
  stats = env._smp_physics_totals
  totals = env._smp_noise_totals
  chunks, original_flags, final_flags = [], [], []
  scorer = PhysicalScorer(env._smp_physics)
  for _ in range(count // B):
    before = time.perf_counter()
    raw, initial, posterior = draw_recorded(env._smp_bundle, B, env.device)
    raw_np = raw.detach().cpu().numpy()
    totals["generation_seconds"] += time.perf_counter() - before
    _, original_passed, original_stats = env._smp_physics.process(raw_np, repair=False)
    for key, value in original_stats.items():
      if key in stats:
        stats[key] = max(stats[key], value) if "_max_" in key else stats[key] + value
    output, final_passed, search_stats = search_batch(
      env._smp_bundle,
      raw,
      initial,
      posterior,
      original_passed,
      scorer,
      seed=int(env.cfg.seed),
      global_start=int(totals["original_rows"]),
      config=config,
    )
    if not np.isfinite(output).all():
      raise FloatingPointError(
        "non-finite GSI state; original finite-value checks remain enabled"
      )
    assert np.array_equal(output[original_passed], raw_np[original_passed])
    for key, value in search_stats.items():
      totals[key] = (
        max(totals[key], value)
        if key in ("max_selected_rms", "replay_max_abs", "success_generation_max")
        else totals[key] + value
      )
    totals["original_rows"] += B
    totals["direct_inserted"] += int(original_passed.sum())
    totals["raw_decode_calls"] += 1
    chunks.append(torch.as_tensor(output.copy(), device=raw.device, dtype=raw.dtype))
    original_flags.append(torch.as_tensor(original_passed, device=raw.device))
    final_flags.append(torch.as_tensor(final_passed, device=raw.device))
  result = torch.cat(chunks)
  assert result.shape == (count, 10, 59)
  env._smp_noise_last_original_pass = torch.cat(original_flags)
  env._smp_noise_last_final_pass = torch.cat(final_flags)
  env._smp_physics_last_repaired = torch.zeros(
    count, dtype=torch.bool, device=result.device
  )
  stats["updates"] += 1
  stats["inserted"] += count
  totals["inserted"] += count
  totals["collection_seconds"] += time.perf_counter() - started
  assert totals["original_rows"] == totals["inserted"]
  assert totals["direct_inserted"] + totals["attempted"] == totals["inserted"]
  assert totals["succeeded"] + totals["exhausted"] == totals["attempted"]
  print(
    f"[00033 collection] step={int(env.common_step_counter)} count={count} "
    f"all_inserted={count} direct={int(env._smp_noise_last_original_pass.sum())} "
    f"physical_pass_after={int(env._smp_noise_last_final_pass.sum())} "
    f"failed_search_results_also_inserted={int((~env._smp_noise_last_final_pass).sum())}",
    flush=True,
  )
  return result
