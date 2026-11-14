"""Server-only frozen-prior/GPU preflight at the actual 1024-row layout."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from noise_rescue import SearchConfig, decode, draw_recorded  # noqa: E402

from smp.rl.events import _ddpm_sample, _maybe_compile  # noqa: E402
from smp.rl.utils import load_denoiser  # noqa: E402


@torch.no_grad()
def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--out", type=Path, required=True)
  args = parser.parse_args()
  assert torch.cuda.is_available()
  device = "cuda:0"
  bundle = list(
    load_denoiser(str(ROOT / "datasets/pretrain_ckpt/pretrained_lafan_run.pt"), device)
  )
  assert bundle[-2:] == [59, 10]
  assert bundle[1].num_timesteps == 50
  assert all(not p.requires_grad for p in bundle[0].parameters())
  bundle[0] = _maybe_compile(bundle[0], True, "max-autotune")
  B = SearchConfig().inference_batch_size
  # Compile once before the RNG parity check, at the one actual layout.
  bundle[0](
    torch.zeros(B, 10, 59, device=device),
    torch.zeros(B, dtype=torch.long, device=device),
  )
  env = SimpleNamespace(device=device, _smp_bundle=bundle)
  torch.manual_seed(2028)
  native = _ddpm_sample(env, B)
  native_rng = torch.cuda.get_rng_state().clone()
  torch.manual_seed(2028)
  raw, initial, posterior = draw_recorded(bundle, B, device)
  result = dict(
    batch_size=B,
    ddpm_steps=50,
    shape=list(raw.shape),
    native_max_abs=float((raw - native).abs().max()),
    native_rng_equal=bool(torch.equal(native_rng, torch.cuda.get_rng_state())),
  )
  before = posterior.clone()
  rng_before = torch.cuda.get_rng_state().clone()
  replay = decode(bundle, initial, posterior)
  result["zero_delta_max_abs"] = float((replay - raw).abs().max())
  # Search duplicates each original row into 32 contiguous candidate slots.
  sources = torch.arange(B, device=device) // 32
  repeat = decode(bundle, initial[sources], posterior[:, sources])
  result["repeated_rows_max_abs"] = float((repeat - raw[sources]).abs().max())
  result["posterior_unchanged"] = bool(torch.equal(before, posterior))
  result["search_replay_rng_neutral"] = bool(
    torch.equal(rng_before, torch.cuda.get_rng_state())
  )
  result["finite"] = bool(torch.isfinite(raw).all() and torch.isfinite(repeat).all())
  ok = (
    result["native_max_abs"] <= 1e-5
    and result["zero_delta_max_abs"] <= 1e-5
    and result["repeated_rows_max_abs"] <= 1e-5
    and all(
      result[k]
      for k in (
        "native_rng_equal",
        "posterior_unchanged",
        "search_replay_rng_neutral",
        "finite",
      )
    )
  )
  result["passed"] = ok
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
  print(json.dumps(result, indent=2), flush=True)
  if not ok:
    raise SystemExit("00054 sampler preflight failed; no training started")
  print("00054_SAMPLER_CHECK=PASS")


if __name__ == "__main__":
  main()
