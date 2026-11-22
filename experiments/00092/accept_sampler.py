"""GPU preflight: archived 1024-row decoder replay, duplicate layout and RNG."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from archive_format import atomic_json
from noise_rescue import decode
from paired_replay import protect_training_rng, source_binding


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--control-info", type=Path, required=True)
  p.add_argument("--out", type=Path, required=True)
  args = p.parse_args()
  assert torch.cuda.is_available()
  from smp.rl.events import _maybe_compile
  from smp.rl.utils import load_denoiser

  binding = source_binding(args.control_info)
  directory = Path(binding["archive_dir"])
  manifest = json.loads((directory / "manifest.json").read_text())
  first = manifest["batches"][0]
  from archive_format import file_sha256

  assert file_sha256(directory / first["file"]) == first["sha256"]
  x = torch.load(directory / first["file"], map_location="cpu", weights_only=False)
  before = torch.get_rng_state().clone()
  cuda_before = torch.cuda.get_rng_state().clone()
  with protect_training_rng("cuda:0"), torch.no_grad():
    bundle = list(
      load_denoiser("datasets/pretrain_ckpt/pretrained_lafan_run.pt", "cuda:0")
    )
    assert bundle[-2:] == [59, 10] and bundle[1].num_timesteps == 50
    assert all(not p.requires_grad for p in bundle[0].parameters())
    bundle[0] = _maybe_compile(bundle[0], True, "max-autotune")
    initial = x["initial"].cuda()
    posterior = x["posterior"].cuda()
    raw = x["windows"].cuda()
    replay = decode(bundle, initial, posterior)
    error = float((raw - replay).abs().max())
    src = torch.arange(1024, device="cuda:0") // 32
    repeat = decode(bundle, initial[src], posterior[:, src])
    repeat_error = float((repeat - raw[src]).abs().max())
    assert error <= 1e-5 and repeat_error <= 1e-5
    assert torch.isfinite(replay).all() and torch.isfinite(repeat).all()
    assert torch.equal(initial.cpu(), x["initial"]) and torch.equal(
      posterior.cpu(), x["posterior"]
    )
  assert torch.equal(before, torch.get_rng_state())
  assert torch.equal(cuda_before, torch.cuda.get_rng_state())
  atomic_json(
    args.out,
    dict(
      status="PASS",
      source=binding,
      batch_size=1024,
      archived_replay_max_abs=error,
      duplicate_layout_max_abs=repeat_error,
      caller_rng_unchanged=True,
      posterior_unchanged=True,
    ),
  )
  print("PAIRED_GPU_REPLAY_CHECK=PASS")


if __name__ == "__main__":
  main()
