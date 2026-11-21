"""Audit one representative per original row, native timing, unchanged source."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from archive_format import atomic_json
from paired_replay import source_binding


def verify(log_dir, *, iterations):
  log_dir = Path(log_dir)
  info = json.loads((log_dir / "run_info.json").read_text())
  assert info["experiment"] == "00091" and info["seed"] == 2027
  assert info["stage"] == ("formal" if iterations == 20000 else "smoke")
  binding = source_binding(info["control_info"])
  assert info["paired_source"] == binding
  assert info["initial_models_sha256"] == binding["initial_models_sha256"]
  assert (
    info["control_initial_checkpoint_sha256"] == binding["initial_checkpoint_sha256"]
  )
  assert info["initialization_mode"] == "load_0009_models_only"
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
    assert entry["original_passed"] + stats["attempted"] == 1024
    assert stats["succeeded"] + stats["exhausted"] == stats["attempted"]
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
