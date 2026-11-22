"""Stream the exact recorded batches, verify integrity and reconstruct FIFO.

This does not reject/relabel generated motion or apply any physical threshold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from archive_format import SCHEMA, atomic_json, file_sha256, tensor_sha256


def verify(directory, *, iterations, steps_per_env=24, run_info=None):
  directory = Path(directory).resolve()
  manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
  assert manifest["schema"] == SCHEMA and manifest["status"] == "complete"
  assert manifest["experiment"] == "00093"
  assert manifest["gsi_seed"] == 102028
  assert manifest["completed_iterations"] == iterations
  assert manifest["dtype"] == "float32" and manifest["feature_shape"] == [10, 59]
  assert manifest["diffusion_steps"] == 50
  capacity, batch_size, interval = (
    manifest[k] for k in ("pool_size", "batch_size", "refresh_control_steps")
  )
  assert (capacity, batch_size, interval) == (4096, 1024, 2400)
  refreshes = iterations * steps_per_env // interval
  assert manifest["control_steps"] == iterations * steps_per_env
  assert (
    manifest["refreshes"] == refreshes and manifest["transactions"] == refreshes + 1
  )
  assert manifest["rows"] == capacity + refreshes * batch_size
  assert len(manifest["batches"]) == capacity // batch_size + refreshes
  pool = torch.empty((capacity, 10, 59), dtype=torch.float32)
  rows, head, previous_rng = 0, 0, None
  for index, entry in enumerate(manifest["batches"]):
    assert entry["batch_index"] == index and entry["global_row_start"] == rows
    name = f"batch_{index:04d}.pt"
    assert entry["file"] == name
    path = directory / name
    assert path.stat().st_size == entry["bytes"]
    assert file_sha256(path) == entry["sha256"], f"corrupt shard: {name}"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["schema"] == SCHEMA
    for key, value in payload["metadata"].items():
      assert entry[key] == value, f"{name} metadata differs: {key}"
    for key, shape in [
      ("windows", (1024, 10, 59)),
      ("initial", (1024, 10, 59)),
      ("posterior", (49, 1024, 10, 59)),
    ]:
      value = payload[key]
      assert tuple(value.shape) == shape and value.dtype == torch.float32
      assert bool(torch.isfinite(value).all()), f"non-finite {name}/{key}"
    assert tensor_sha256(payload["windows"]) == entry["windows_sha256"]
    if previous_rng is not None:
      for key in ("cpu", "cuda"):
        a, b = previous_rng[key], payload["gsi_rng_before"][key]
        assert (a is None and b is None) or (
          a is not None and b is not None and torch.equal(a, b)
        ), "GSI RNG discontinuity"
    previous_rng = payload["gsi_rng_after"]
    if index < 4:
      assert entry["phase"] == "startup" and entry["control_step"] == 0
      assert entry["transaction_index"] == 0
      assert entry["offset_in_transaction"] == index * batch_size
      assert entry["fifo_head_before"] == index * batch_size
      pool[index * batch_size : (index + 1) * batch_size] = payload["windows"]
    else:
      transaction = index - 3
      assert entry["phase"] == "refresh"
      assert entry["control_step"] == transaction * interval
      assert (
        entry["transaction_index"] == transaction
        and entry["offset_in_transaction"] == 0
      )
      assert entry["fifo_head_before"] == head
      pool[head : head + batch_size] = payload["windows"]
      head = (head + batch_size) % capacity
    rows += batch_size
    del payload
  assert rows == manifest["rows"] and head == manifest["final_fifo_head"]
  assert tensor_sha256(pool) == manifest["final_pool_sha256"], (
    "recorded data differs from actual final GSI pool"
  )
  assert not list(directory.glob("*.partial"))
  assert {p.name for p in directory.glob("batch_*.pt")} == {
    e["file"] for e in manifest["batches"]
  }
  if run_info is not None:
    info = json.loads(Path(run_info).read_text(encoding="utf-8"))
    assert info["experiment"] == "00093" and info["seed"] == 2028
    assert (
      info["completed_iterations"] == iterations
      and info["num_steps_per_env"] == steps_per_env
    )
    assert Path(info["gsi_archive_dir"]).resolve() == directory
    assert info["gsi_archive_status"] == "complete"
    assert info["gsi_recorded_rows"] == rows and info["gsi_recorded_shards"] == len(
      manifest["batches"]
    )
    assert (
      info["gsi_generation_rng_isolated"] is True
      and info["gsi_generation_rng_seed"] == 102028
    )
    for key in (
      "run_id",
      "stage",
      "prior_sha256",
      "uv_lock_sha256",
      "initial_models_sha256",
      "saved_initial_checkpoint_sha256",
      "git_commit",
    ):
      assert info[key] == manifest["provenance"][key]
  return dict(
    status="PASS",
    schema=SCHEMA,
    directory=str(directory),
    rows=rows,
    shards=len(manifest["batches"]),
    transactions=refreshes + 1,
    final_pool_sha256=manifest["final_pool_sha256"],
  )


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--log-dir", type=Path, required=True)
  parser.add_argument("--iterations", type=int, required=True)
  parser.add_argument("--out", type=Path, required=True)
  args = parser.parse_args()
  torch.set_num_threads(min(8, torch.get_num_threads()))
  info_path = args.log_dir / "run_info.json"
  info = json.loads(info_path.read_text(encoding="utf-8"))
  result = verify(
    info["gsi_archive_dir"], iterations=args.iterations, run_info=info_path
  )
  atomic_json(args.out, result)
  print("GSI_ARCHIVE_CHECK=PASS", json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
  main()
