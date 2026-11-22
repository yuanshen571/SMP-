"""Actual task-model CPU/GPU filter and score equivalence; no policy training."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from archive_format import atomic_json
from contract import check_files
from paired_replay import protect_training_rng, source_binding


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--control-info", type=Path, required=True)
  p.add_argument("--out", type=Path, required=True)
  a = p.parse_args()
  spec = check_files(quiet=True)
  binding = source_binding(a.control_info)
  assert torch.cuda.is_available()
  from gpu_filter import GpuPhysicalFilter
  from gpu_scorer import GpuScorer
  from gsi_physics import FilterConfig, PhysicalFilter
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg
  from noise_rescue import PhysicalScorer

  import smp.rl.tasks  # noqa: F401

  cfg = load_env_cfg("Smp-Location-G1", play=False)
  cfg.scene.num_envs = 1
  for k in list(cfg.events):
    if "smp" in k or "gsi" in k:
      cfg.events.pop(k)
  with protect_training_rng("cuda:0"):
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode=None)
    try:
      options = FilterConfig(repair=False, control_dt=0.02)
      cpu = PhysicalFilter(env.sim.mj_model, options)
      gpu = GpuPhysicalFilter(env.sim.mj_model, options, device="cuda:0")
      assert gpu.device.type == "cuda" and gpu._rad.dtype == torch.float64
      assert (
        cpu.manifest["sha256"]
        == gpu.manifest["sha256"]
        == spec["physical_manifest_sha256"]
      )
      cpu_scorer, gpu_scorer = PhysicalScorer(cpu), GpuScorer(gpu)
      manifest = json.loads(
        (Path(binding["archive_dir"]) / "manifest.json").read_text()
      )
      rows = []
      for entry in manifest["batches"][:4]:
        payload = torch.load(
          Path(binding["archive_dir"]) / entry["file"],
          map_location="cpu",
          weights_only=False,
        )
        rows.append(payload["windows"].numpy())
      raw = np.concatenate(rows)
      # Extra numerical/threshold probes derived from actual G1 windows.
      edges = raw[:16].copy()
      for k in range(len(edges)):
        j = k % 29
        edges[k, k % 10, 9 + j] = np.float32(cpu.hi[j] + (k % 4 - 1) * 1e-5)
      edges[0, 0, 9] = np.nan
      results = []
      for name, windows in [("startup4096", raw), ("joint_boundary16", edges)]:
        torch.cuda.synchronize()
        t = time.perf_counter()
        co, cm, cd = cpu.process(windows, repair=False)
        reference = cpu_scorer.evaluate(windows)
        cpu_seconds = time.perf_counter() - t
        torch.cuda.synchronize()
        t = time.perf_counter()
        go, gm, gd = gpu.process(windows, repair=False)
        candidate = gpu_scorer.evaluate(windows)
        torch.cuda.synchronize()
        gpu_seconds = time.perf_counter() - t
        assert np.array_equal(co, go, equal_nan=True), name
        assert np.array_equal(cm, gm), name
        assert cd == gd, name
        for key in ("passed", "numeric", "joint", "ground", "overlap", "fitness"):
          assert np.array_equal(reference[key], candidate[key], equal_nan=True), (
            name,
            key,
          )
        assert np.array_equal(candidate["passed"], cm), name
        results.append(
          dict(
            name=name,
            rows=len(windows),
            exact_equal=True,
            cpu_reference_seconds=cpu_seconds,
            gpu_seconds=gpu_seconds,
          )
        )
      atomic_json(
        a.out,
        dict(
          status="PASS",
          training_updates=0,
          manifest_sha256=gpu.manifest["sha256"],
          gpu_module=spec["gpu_module"],
          comparisons=results,
          note="timings include filter plus scorer and are acceptance measurements, not an isolated performance benchmark",
        ),
      )
    finally:
      env.close()
  print("GPU_FILTER_AND_SCORER_ACCEPTANCE=PASS")


if __name__ == "__main__":
  main()
