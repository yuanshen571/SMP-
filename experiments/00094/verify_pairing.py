"""Audit one representative per original row, native timing, unchanged source."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from archive_format import atomic_json
from paired_replay import source_binding


def verify_decisions(entry):
  stats = entry["stats"]
  decisions = entry["decisions"]
  assert len(decisions) == stats["attempted"] + stats["skipped_original_inserted"]
  rows = [d["original_row"] for d in decisions]
  assert rows == sorted(set(rows))
  assert all(
    entry["global_row_start"] <= r < entry["global_row_start"] + 1024 for r in rows
  )
  counts = {
    key: 0
    for key in (
      "physical_success",
      "generation_limit",
      "joint_improvement_below_15_percent",
      "joint_above_0.42_deg",
    )
  }
  checked = continued = exempt = 0
  for d in decisions:
    reason, gen, joint0 = d["reason"], d["generation"], d["original_joint_rad"]
    assert reason in counts
    counts[reason] += 1
    if reason == "joint_above_0.42_deg":
      assert joint0 > np.deg2rad(0.42) and gen == 0
      assert d["selected_original"] and not d["final_passed"] and not d["gate_applied"]
      continue
    assert joint0 is None or joint0 <= np.deg2rad(0.42)
    assert 1 <= gen <= 30 and not d["selected_original"]
    assert d["selected_rms"] <= 0.15 and math.isfinite(d["selected_fitness"])
    exempt += int(joint0 is None or joint0 <= 1e-5)
    assert d["final_passed"] == (reason == "physical_success")
    if reason == "generation_limit":
      assert gen == 30
    if reason == "joint_improvement_below_15_percent":
      assert gen == 10 and d["gate_applied"]
    if d["gate_applied"]:
      checked += 1
      assert joint0 > 1e-5 and gen >= 10
      best10 = d["best_joint_first10_rad"]
      poor = best10 is None or best10 > joint0 * 0.85
      continued += int(not poor)
      if best10 is not None:
        assert math.isclose(
          d["improvement_ratio"], (joint0 - best10) / joint0, abs_tol=1e-12
        )
      assert poor == (reason == "joint_improvement_below_15_percent")
    else:
      assert reason != "joint_improvement_below_15_percent"
      assert (
        joint0 is None or joint0 <= 1e-5 or (reason == "physical_success" and gen <= 10)
      )
  assert stats["generations"] == sum(d["generation"] for d in decisions)
  assert stats["succeeded"] == counts["physical_success"]
  assert stats["exhausted"] == counts["generation_limit"]
  assert stats["early_stopped"] == counts["joint_improvement_below_15_percent"]
  assert stats["skipped_original_inserted"] == counts["joint_above_0.42_deg"]
  assert stats["joint_gate_exempt"] == exempt
  assert stats["joint_improvement_checked"] == checked
  assert stats["joint_improvement_continued"] == continued


def verify(log_dir, *, iterations):
  log_dir = Path(log_dir)
  info = json.loads((log_dir / "run_info.json").read_text())
  assert info["experiment"] == "00094" and info["seed"] == 2028
  assert info["stage"] == ("formal" if iterations == 20000 else "smoke")
  binding = source_binding(info["control_info"])
  assert info["paired_source"] == binding
  assert info["initial_models_sha256"] == binding["initial_models_sha256"]
  assert (
    info["control_initial_checkpoint_sha256"] == binding["initial_checkpoint_sha256"]
  )
  assert info["initialization_mode"] == "load_00093_models_only"
  audit = json.loads(Path(info["paired_replay_audit"]).read_text())
  assert (
    audit["source"] == binding
    and audit["status"] == "complete"
    and audit["iterations"] == iterations
  )
  source = json.loads((Path(binding["archive_dir"]) / "manifest.json").read_text())
  count = 4 + iterations * 24 // 2400
  assert info["paired_replay_status"] == "complete"
  assert info["paired_replayed_rows"] == audit["rows"] == count * 1024
  assert info["paired_replayed_shards"] == len(audit["batches"]) == count
  accumulated = {}
  for i, entry in enumerate(audit["batches"]):
    raw = source["batches"][i]
    for key in ["batch_index", "global_row_start", "control_step", "fifo_head_before"]:
      assert entry[key] == raw[key]
    assert (
      entry["source_sha256"] == raw["sha256"]
      and entry["raw_windows_sha256"] == raw["windows_sha256"]
    )
    assert entry["archive_replay_max_abs"] <= 1e-5
    stats = entry["stats"]
    assert (
      entry["original_passed"] + stats["skipped_original_inserted"] + stats["attempted"]
      == 1024
    )
    assert (
      stats["succeeded"] + stats["early_stopped"] + stats["exhausted"]
      == stats["attempted"]
    )
    verify_decisions(entry)
    assert entry["final_passed"] == entry["original_passed"] + stats["succeeded"]
    assert (
      stats["proposals"] == 32 * stats["generations"] <= stats["attempted"] * 32 * 30
    )
    assert (
      stats["effective_candidates"] + stats["radius_rejected"] == stats["proposals"]
    )
    assert stats["max_selected_rms"] <= 0.15 and stats["success_generation_max"] <= 30
    assert stats["replay_max_abs"] <= 1e-5
    for k, v in stats.items():
      assert math.isfinite(v)
      accumulated[k] = (
        max(accumulated.get(k, 0), v)
        if k in ["max_selected_rms", "replay_max_abs", "success_generation_max"]
        else accumulated.get(k, 0) + v
      )
    assert len(entry["output_sha256"]) == 64
  totals = info["noise_rescue_totals"]
  assert totals == audit["noise_totals"]
  for k, v in accumulated.items():
    assert math.isclose(totals[k], v, rel_tol=1e-10, abs_tol=1e-10), k
  assert totals["original_rows"] == totals["inserted"] == count * 1024
  assert totals["direct_inserted"] == sum(
    e["original_passed"] for e in audit["batches"]
  )
  assert totals["attempted"] > 0
  assert totals["original_fallback_inserted"] == totals["discarded"] == 0
  assert audit["final_fifo_head"] == (count - 4) * 1024 % 4096
  assert len(audit["final_pool_sha256"]) == 64
  return dict(
    status="PASS",
    rows=count * 1024,
    shards=count,
    source=binding,
    final_pool_sha256=audit["final_pool_sha256"],
    noise_rescue_totals=totals,
  )


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--log-dir", type=Path, required=True)
  p.add_argument("--iterations", type=int, required=True)
  p.add_argument("--out", type=Path, required=True)
  a = p.parse_args()
  result = verify(a.log_dir, iterations=a.iterations)
  atomic_json(a.out, result)
  print("PAIRING_AUDIT=PASS", json.dumps(result))


if __name__ == "__main__":
  main()
