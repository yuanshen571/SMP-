"""Generate only final qualified unique windows; never step or train a policy."""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from types import SimpleNamespace

import numpy as np
from experiments.ts2vec_g1.common import (
  HERE,
  ROOT,
  choose_unique,
  config,
  fit_normalizer,
  git_commit,
  now,
  read_json,
  seed_all,
  sha256,
  validate_windows,
  window_digest,
  write_json,
)


def generate(run, reference_path, device):
  from mjlab.scene import Scene
  from mjlab.tasks.registry import load_env_cfg

  import smp.rl.tasks  # noqa: F401
  from smp.rl.events import _ddpm_sample, _maybe_compile
  from smp.rl.gsi_physics import EE_NAMES, JOINT_NAMES, FilterConfig, PhysicalFilter
  from smp.rl.utils import load_denoiser

  cfg = config()
  out = run / "dataset"
  if (out / "manifest.json").exists():
    verify_dataset(out)
    return
  if out.exists():
    raise RuntimeError(f"unfinished dataset exists: {out}; stop, do not overwrite")
  out.mkdir(parents=True)
  started = time.monotonic()
  reference = read_json(reference_path)
  if reference.get("experiment") != "00031":
    raise ValueError("reference must be the existing 00031 run_info.json")
  if reference.get("completed_iterations") != 10000:
    raise ValueError("reference 00031 run must be complete (10000 iterations)")
  if reference["prior_sha256"] != cfg["prior_sha256"]:
    raise ValueError("00031 prior differs from frozen prior")
  if reference["gsi_physics_config"] != cfg["physical_filter"]:
    raise ValueError("00031 physical conditions differ")

  env_cfg = load_env_cfg(cfg["task"], play=False)
  dt = env_cfg.sim.mujoco.timestep * env_cfg.decimation
  if abs(dt - cfg["control_dt"]) > 1e-12:
    raise ValueError("G1 control sampling interval changed")
  # Compile the same scene as 00031, with no GPU environments, managers or steps.
  # Friction/COM randomization does not alter the private CPU collision geometry.
  scene = Scene(env_cfg.scene, device="cpu")
  model = scene.compile()
  physics = PhysicalFilter(model, FilterConfig(**cfg["physical_filter"]))
  if physics.manifest != reference["gsi_model_manifest"]:
    # JSON converts tuples into lists; compare the serialized representation.
    if json.loads(json.dumps(physics.manifest)) != reference["gsi_model_manifest"]:
      raise ValueError("scene geometry/limits differ from actual 00031 manifest")

  prior = ROOT / cfg["prior_path"]
  if sha256(prior) != cfg["prior_sha256"]:
    raise ValueError("frozen prior SHA256 mismatch")
  seed_all(cfg["seed"])
  bundle = list(load_denoiser(str(prior), device))
  params = env_cfg.events["init_smp_state"].params
  if not params["compile_model"] or params["gsi_batch_size"] != cfg["raw_batch_size"]:
    raise ValueError("00031 GSI sampling configuration changed")
  bundle[0] = _maybe_compile(bundle[0], True, params.get("compile_mode"))
  if bundle[4:] != [59, 10] or bundle[1].num_timesteps != 50:
    raise ValueError("prior must have 10x59 features and 50 diffusion steps")
  holder = SimpleNamespace(_smp_bundle=tuple(bundle), device=device)
  count, raw_batch = cfg["unique_windows"], cfg["raw_batch_size"]
  features = np.lib.format.open_memmap(
    out / "features.partial.npy", mode="w+", dtype="<f4", shape=(count, 10, 59)
  )
  hashes = np.empty((count, 32), dtype=np.uint8)
  source = np.empty((count, 2), dtype=np.int64)
  repaired_flags = np.empty(count, dtype=np.bool_)
  seen, inserted, batch_id = set(), 0, 0
  totals = dict(generated=0, qualified=0, duplicates=0, surplus=0, inserted=0)
  while inserted < count:
    if totals["generated"] + raw_batch > cfg["max_generated"]:
      raise RuntimeError("generation emergency bound reached; no relaxed fallback")
    raw = _ddpm_sample(holder, raw_batch).detach().cpu().numpy()
    if raw.dtype != np.float32 or raw.shape != (raw_batch, 10, 59):
      raise ValueError("DDPM returned an unexpected shape/type")
    corrected, passed, stats = physics.process(raw)
    ids, digests, duplicate, surplus = choose_unique(
      corrected, passed, seen, count - inserted
    )
    for name, value in stats.items():
      totals[name] = (
        max(totals.get(name, 0), value)
        if "_max_" in name
        else totals.get(name, 0) + value
      )
    if ids:
      finish = inserted + len(ids)
      features[inserted:finish] = corrected[ids]
      hashes[inserted:finish] = np.stack(digests)
      source[inserted:finish, 0] = batch_id
      source[inserted:finish, 1] = ids
      angles = raw[ids, :, 9:38].astype(np.float64)
      repaired_flags[inserted:finish] = (
        np.maximum(0, np.maximum(physics.lo - angles, angles - physics.hi)).max(
          axis=(1, 2)
        )
        > cfg["physical_filter"]["joint_tolerance_rad"]
      )
      inserted = finish
    totals["duplicates"] += duplicate
    totals["surplus"] += surplus
    totals["inserted"] = inserted
    batch_id += 1
    if batch_id % 10 == 0 or inserted == count:
      features.flush()
      write_json(out / "progress.json", dict(stage="generation", **totals))
      print(f"[dataset] batch={batch_id} unique={inserted}/{count}", flush=True)
    # Rejected, duplicate and quota-surplus raw/corrected arrays are only in RAM.
    del raw, corrected
  features.flush()
  del features
  (out / "features.partial.npy").rename(out / "features.npy")
  np.save(out / "sample_sha256.npy", hashes, allow_pickle=False)
  np.save(out / "source_batch_row.npy", source, allow_pickle=False)
  np.save(out / "repaired.npy", repaired_flags, allow_pickle=False)

  x = np.load(out / "features.npy", mmap_mode="r", allow_pickle=False)
  # Reload and strictly re-check final windows, without another correction.
  for start in range(0, count, 512):
    chunk = np.array(x[start : start + 512], copy=True)
    rechecked, ok, _ = physics.process(chunk, repair=False)
    if not ok.all() or not np.array_equal(chunk, rechecked):
      raise AssertionError("saved dataset fails strict post-correction checks")
  np.savez(
    out / "normalizer.npz",
    **fit_normalizer(x, cfg["normalizer"]["constant_std_threshold"]),
  )
  write_json(
    out / "feature_schema.json",
    dict(
      shape=[10, 59],
      dtype="little-endian float32",
      dt=dt,
      frame_order="oldest_to_last",
      root_position=dict(
        slice=[0, 3],
        units="m",
        xy="last-root-position relative, last-yaw frame",
        z="world height",
      ),
      root_rotation=dict(
        slice=[3, 9], representation="rotation columns 0 and 2 in last-yaw frame"
      ),
      joint_angles=dict(slice=[9, 38], units="rad", names=JOINT_NAMES),
      end_effectors=dict(
        slice=[38, 53],
        units="m",
        names=EE_NAMES,
        reference="each-frame root relative, last-yaw frame; body origins",
      ),
      root_linear_velocity=dict(slice=[53, 56], units="m/s", frame="last-yaw"),
      root_angular_velocity=dict(slice=[56, 59], units="rad/s", frame="last-yaw"),
      joint_velocity="not included; initialization derives it from adjacent corrected angles",
    ),
  )
  write_json(out / "physical_manifest.json", physics.manifest)
  files = [
    "features.npy",
    "sample_sha256.npy",
    "source_batch_row.npy",
    "repaired.npy",
    "normalizer.npz",
    "feature_schema.json",
    "physical_manifest.json",
  ]
  write_json(
    out / "manifest.json",
    dict(
      status="PASS",
      created_at=now(),
      git_commit=git_commit(),
      config=cfg,
      config_sha256=sha256(HERE / "config.json"),
      unique_windows=count,
      deduplication="SHA256 of full final contiguous little-endian float32 10x59 bytes; first qualified occurrence",
      totals=totals,
      repaired_inserted=int(repaired_flags.sum()),
      reference_00031_path=str(reference_path),
      reference_00031_sha256=sha256(reference_path),
      prior_sha256=sha256(prior),
      physical_config=dataclasses.asdict(physics.config),
      model_manifest_sha256=physics.manifest["sha256"],
      seconds=time.monotonic() - started,
      files={name: sha256(out / name) for name in files},
    ),
  )
  verify_dataset(out)


def verify_dataset(out):
  cfg = config()
  manifest = read_json(out / "manifest.json")
  if manifest["status"] != "PASS" or manifest["config"] != cfg:
    raise AssertionError("dataset was not built under the current fixed configuration")
  for name, digest in manifest["files"].items():
    if sha256(out / name) != digest:
      raise AssertionError(f"dataset file changed: {name}")
  x = np.load(out / "features.npy", mmap_mode="r", allow_pickle=False)
  validate_windows(x)
  if len(x) != cfg["unique_windows"]:
    raise AssertionError("dataset quota differs")
  stored = np.load(out / "sample_sha256.npy", allow_pickle=False)
  digests = [window_digest(window) for window in x]
  if len(set(digests)) != len(x) or not np.array_equal(
    np.stack([np.frombuffer(d, dtype=np.uint8) for d in digests]), stored
  ):
    raise AssertionError("dataset uniqueness or per-window hashes differ")
  totals = manifest["totals"]
  if totals["qualified"] != len(x) + totals["duplicates"] + totals["surplus"]:
    raise AssertionError("qualified/duplicate/surplus accounting differs")
  source = np.load(out / "source_batch_row.npy", allow_pickle=False)
  repaired = np.load(out / "repaired.npy", allow_pickle=False)
  if source.shape != (len(x), 2) or source.dtype != np.int64:
    raise AssertionError("invalid source index")
  ordered = source[:, 0] * cfg["raw_batch_size"] + source[:, 1]
  if (
    (source < 0).any()
    or (source[:, 1] >= cfg["raw_batch_size"]).any()
    or (np.diff(ordered) <= 0).any()
    or int(ordered[-1]) >= totals["generated"]
  ):
    raise AssertionError("source order/index differs")
  if repaired.dtype != np.bool_ or repaired.shape != (len(x),):
    raise AssertionError("invalid correction flags")
  if int(repaired.sum()) != manifest["repaired_inserted"]:
    raise AssertionError("correction flag count differs")
  expected = fit_normalizer(x, cfg["normalizer"]["constant_std_threshold"])
  with np.load(out / "normalizer.npz", allow_pickle=False) as saved:
    for key in expected:
      np.testing.assert_array_equal(saved[key], expected[key])
  return manifest


if __name__ == "__main__":
  from pathlib import Path

  parser = argparse.ArgumentParser()
  parser.add_argument("--run", type=Path, required=True)
  parser.add_argument("--reference-00031", type=Path, required=True)
  parser.add_argument("--device", default="cuda:0")
  args = parser.parse_args()
  generate(args.run.resolve(), args.reference_00031.resolve(), args.device)
