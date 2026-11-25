"""Verify exact baseline windows/order and unchanged original-source bindings."""

import argparse
import json
from pathlib import Path

from archive_format import atomic_json
from paired_replay import source_binding


def verify(log_dir, *, iterations):
  log_dir = Path(log_dir)
  info = json.loads((log_dir / "run_info.json").read_text())
  assert info["experiment"] == "00098" and info["repeats_experiment"] == "00095"
  assert info["stage"] == ("formal" if iterations == 20000 else "smoke")
  assert (
    info["seed"] == 2026 and info["initialization_mode"] == "load_00095_models_only"
  )
  assert info["physical_filter_enabled"] is info["noise_rescue_enabled"] is False
  assert info["gsi_recording_enabled"] is False and info["gsi_replay_enabled"] is True
  source = source_binding(info["control_info"])
  assert source == info["paired_source"]
  assert (
    info["initial_models_sha256"]
    == info["control_initial_models_sha256"]
    == source["initial_models_sha256"]
  )
  assert (
    info["control_initial_checkpoint_sha256"] == source["initial_checkpoint_sha256"]
  )
  manifest = json.loads((Path(source["archive_dir"]) / "manifest.json").read_text())
  audit = json.loads(Path(info["paired_replay_audit"]).read_text())
  assert audit["schema"] == "smp-g1-baseline-repeat-v1"
  assert audit["source"] == source and audit["status"] == "complete"
  assert audit["iterations"] == iterations and audit["experiment"] == "00098"
  count = 4 + iterations * 24 // 2400
  assert len(audit["batches"]) == info["paired_replayed_shards"] == count
  assert audit["rows"] == info["paired_replayed_rows"] == count * 1024
  assert info["paired_replay_status"] == "complete"
  for index, (entry, raw) in enumerate(
    zip(audit["batches"], manifest["batches"], strict=False)
  ):
    assert entry["batch_index"] == index
    for key in ["global_row_start", "control_step", "fifo_head_before"]:
      assert entry[key] == raw[key]
    assert entry["source_sha256"] == raw["sha256"]
    assert (
      entry["output_sha256"] == entry["raw_windows_sha256"] == raw["windows_sha256"]
    )
    assert entry["inserted"] == 1024 and 0 <= entry["archive_replay_max_abs"] <= 1e-5
  assert audit["final_fifo_head"] == ((count - 4) * 1024) % 4096
  if iterations == 20000:
    assert audit["final_pool_sha256"] == manifest["final_pool_sha256"]
  assert (
    audit["new_candidates_generated"]
    == audit["physically_processed_rows"]
    == audit["noise_search_attempted"]
    == 0
  )
  return dict(
    status="PASS",
    source_run_id=source["source_run_id"],
    same_original_windows=True,
    initial_models_sha256=source["initial_models_sha256"],
    rows=audit["rows"],
    batches=count,
    final_pool_sha256=audit["final_pool_sha256"],
  )


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--log-dir", type=Path, required=True)
  p.add_argument("--iterations", type=int, required=True)
  p.add_argument("--out", type=Path, required=True)
  args = p.parse_args()
  result = verify(args.log_dir, iterations=args.iterations)
  atomic_json(args.out, result)
  print("BASELINE_REPLAY_AUDIT=PASS")


if __name__ == "__main__":
  main()
