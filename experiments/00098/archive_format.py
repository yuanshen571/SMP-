"""Read-only archive helpers and native fixed-noise DDPM decoding."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch

SCHEMA = "smp-g1-paired-gsi-v1"


def file_sha256(path):
  digest = hashlib.sha256()
  with Path(path).open("rb") as stream:
    for chunk in iter(lambda: stream.read(1 << 20), b""):
      digest.update(chunk)
  return digest.hexdigest()


def tensor_sha256(tensor):
  value = tensor.detach().cpu().contiguous()
  digest = hashlib.sha256(f"{value.dtype}:{tuple(value.shape)}".encode())
  digest.update(value.numpy().tobytes())
  return digest.hexdigest()


def atomic_json(path, value):
  path = Path(path)
  temporary = path.with_suffix(".json.partial")
  with temporary.open("w", encoding="utf-8") as stream:
    json.dump(value, stream, indent=2, ensure_ascii=False)
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())
  os.replace(temporary, path)


@torch.no_grad()
def decode(bundle, initial, posterior):
  """Read-only replay, t=49..1 noise at posterior[49-t], t=0 has no noise."""
  model, scheduler, low, high, feature_dim, window_size = bundle
  count = int(scheduler.num_timesteps)
  assert initial.shape[1:] == (window_size, feature_dim)
  assert posterior.shape == (count - 1, *initial.shape)
  x = initial
  for t in reversed(range(count)):
    times = torch.full((len(x),), t, dtype=torch.long, device=x.device)
    eps = model(x, times)
    x0 = scheduler.sqrt_recip_alphas_cumprod[t] * x - (
      scheduler.sqrt_recipm1_alphas_cumprod[t] * eps
    )
    mean = (
      scheduler.posterior_mean_coef1[t] * x0 + scheduler.posterior_mean_coef2[t] * x
    )
    x = (
      mean
      if t == 0
      else mean + torch.sqrt(scheduler.posterior_variance[t]) * posterior[count - 1 - t]
    )
  return (x + 1.0) / 2.0 * (high - low) + low
