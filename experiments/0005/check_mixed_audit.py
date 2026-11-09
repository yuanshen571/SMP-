"""Offline verification of provenance/mean/lookahead/alternating FIFO metadata."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from smp.rl.mixed_cache import META_COLUMNS, digest  # noqa: E402


def verify_audit(audit, cfg, iterations, *, steps_per_iteration=24, num_envs=1024):
  boundaries = list(
    range(cfg.activate_after_iteration, iterations + 1, cfg.update_interval)
  )
  rows = audit.get("transactions", [])
  assert len(rows) == len(boundaries), (len(rows), len(boundaries))
  if rows:
    assert audit["meta_columns"] == META_COLUMNS
  pool = np.empty((cfg.capacity, len(META_COLUMNS)), dtype=np.int64)
  first_half, all_ids = None, set()
  for number, (row, boundary) in enumerate(zip(rows, boundaries, strict=True)):
    expected_start = (
      cfg.collect_after_iteration if number == 0 else boundary - cfg.update_interval
    ) * steps_per_iteration + 1
    expected_end = boundary * steps_per_iteration
    assert row["completed_iteration"] == boundary
    assert row["collection_start_step"] == expected_start
    assert row["collection_end_step"] == expected_end
    count, total = row["complete_episodes"], row["sum_complete_episode_steps"]
    histogram = {int(k): v for k, v in row["episode_histogram"].items()}
    candidates = {int(k): v for k, v in row["candidate_counts"].items()}
    assert (
      count > 0
      and total > 0
      and all(0 < k <= 1000 and v > 0 for k, v in histogram.items())
    )
    assert sum(histogram.values()) == count
    assert sum(k * v for k, v in histogram.items()) == total
    assert row["mean_episode_steps"] == total / count
    assert all(
      k in histogram and 0 < v <= histogram[k] * max(0, k - 59)
      for k, v in candidates.items()
    )
    assert sum(candidates.values()) == row["candidate_windows"]
    assert (
      sum(v for k, v in candidates.items() if k * count > total)
      == row["eligible_windows"]
    )
    quota = cfg.capacity if number == 0 else cfg.replace_count
    assert row["selected_windows"] == quota and row["eligible_windows"] >= quota
    slots = np.asarray(row["replaced_slots"], dtype=np.int64)
    meta = np.asarray(row["selected_meta"], dtype=np.int64)
    assert (
      len(set(slots.tolist())) == quota
      and ((0 <= slots) & (slots < cfg.capacity)).all()
    )
    assert meta.shape == (quota, len(META_COLUMNS))
    assert ((0 <= meta[:, 0]) & (meta[:, 0] < num_envs)).all()
    assert (meta[:, 1] >= 0).all()
    assert (meta[:, 3] - meta[:, 2] == meta[:, 4]).all()
    assert (meta[:, 4] * count > total).all()
    assert ((meta[:, 3] >= expected_start) & (meta[:, 3] <= expected_end)).all()
    assert (meta[:, 5] == expected_start).all() and (meta[:, 6] == expected_end).all()
    assert (meta[:, 8] - meta[:, 7] + 1 == 10).all()
    assert (meta[:, 7] > meta[:, 2]).all() and (meta[:, 7] >= expected_start).all()
    assert (meta[:, 9] == 50).all() and np.isin(meta[:, 10], [0, 1]).all()
    assert (meta[:, 8] + 50 <= meta[:, 3] - meta[:, 10]).all()
    for r in meta:
      key = tuple(r[[0, 1, 8]].tolist())
      assert key not in all_ids, ("duplicate source", key)
      all_ids.add(key)
    if number == 0:
      assert row["initial"] and set(slots.tolist()) == set(range(cfg.capacity))
    else:
      assert not row["initial"]
      if number == 1:
        first_half = set(slots.tolist())
      expected_slots = (
        first_half if number % 2 else set(range(cfg.capacity)) - first_half
      )
      assert set(slots.tolist()) == expected_slots
    pool[slots] = meta
    assert digest(pool) == row["pool_metadata_sha256"]
    assert len(row["pool_payload_sha256"]) == 64
  return {
    "transactions": len(rows),
    "inserted": len(all_ids),
    "last_update_iteration": boundaries[-1] if boundaries else 0,
    "last_payload_sha256": rows[-1]["pool_payload_sha256"] if rows else None,
  }
