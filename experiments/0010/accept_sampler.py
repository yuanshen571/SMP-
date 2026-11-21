"""GPU parity/replay acceptance on the real frozen prior and native batch shape."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import torch
from contract import check_files
from gsi_archive import PrivateTorchStream, atomic_json, decode, draw_recorded


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--out", type=Path, required=True)
  args = parser.parse_args()
  check_files(quiet=True)
  assert torch.cuda.is_available()
  import smp.rl.events as events
  from smp.rl.utils import load_denoiser

  device = torch.device("cuda:0")
  bundle = load_denoiser("datasets/pretrain_ckpt/pretrained_lafan_run.pt", device)
  model, scheduler, low, high, feature_dim, window_size = bundle
  assert (scheduler.num_timesteps, feature_dim, window_size) == (50, 59, 10)
  model = events._maybe_compile(model, True, "max-autotune")
  bundle = (model, scheduler, low, high, feature_dim, window_size)
  with torch.no_grad():
    model(
      torch.zeros((1024, 10, 59), device=device),
      torch.zeros(1024, dtype=torch.long, device=device),
    )
  env = SimpleNamespace(_smp_bundle=bundle, device=device)
  with torch.random.fork_rng(devices=[0]):
    torch.manual_seed(102028)
    native = events._ddpm_sample(env, 1024)
  stream = PrivateTorchStream(102028, device)
  cpu_before, cuda_before = (
    torch.get_rng_state().clone(),
    torch.cuda.get_rng_state(0).clone(),
  )
  with stream.use():
    raw, initial, posterior = draw_recorded(bundle, 1024, device)
  assert torch.equal(cpu_before, torch.get_rng_state())
  assert torch.equal(cuda_before, torch.cuda.get_rng_state(0))
  replay = decode(bundle, initial, posterior)
  native_error = float((raw - native).abs().max())
  replay_error = float((raw - replay).abs().max())
  assert bool(torch.isfinite(raw).all())
  assert native_error <= 1e-5 and replay_error <= 1e-5
  result = dict(
    status="PASS",
    actual_prior=True,
    compiled=True,
    batch_size=1024,
    feature_shape=[10, 59],
    dtype=str(raw.dtype),
    posterior_count=49,
    native_max_abs=native_error,
    replay_max_abs=replay_error,
    gpu=torch.cuda.get_device_name(0),
    torch_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
    torch_peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
    caller_cpu_rng_unchanged=True,
    caller_cuda_rng_unchanged=True,
  )
  args.out.parent.mkdir(parents=True, exist_ok=True)
  atomic_json(args.out, result)
  print("GSI_REAL_PRIOR_REPLAY_CHECK=PASS", result)


if __name__ == "__main__":
  main()
