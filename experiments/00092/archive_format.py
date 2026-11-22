"""Lossless unfiltered GSI recording with a private persistent Torch RNG stream.

No PPO, reward, reset sampling or FIFO implementation is replaced. Only the
module-level DDPM sampler is adapted; its native algebra/draw order is retained.
"""

from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
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


class PrivateTorchStream:
  """Swaps private CPU/CUDA states in only while sampling; restores caller RNG."""

  def __init__(self, seed, device):
    self.seed = int(seed)
    self.device = torch.device(device)
    self.cpu = torch.Generator(device="cpu").manual_seed(self.seed).get_state()
    self.cuda = None
    self.devices = []
    if self.device.type == "cuda":
      index = self.device.index
      self.devices = [torch.cuda.current_device() if index is None else index]
      self.cuda = torch.Generator(device=self.device).manual_seed(self.seed).get_state()
    elif self.device.type != "cpu":
      raise ValueError("only native CPU/CUDA samplers are supported")
    self.active = False

  @contextmanager
  def use(self):
    if self.active:
      raise RuntimeError("nested private sampler stream")
    self.active = True
    try:
      with torch.random.fork_rng(devices=self.devices):
        torch.set_rng_state(self.cpu)
        if self.cuda is not None:
          torch.cuda.set_rng_state(self.cuda, self.devices[0])
        try:
          yield
        finally:
          self.cpu = torch.get_rng_state().clone()
          if self.cuda is not None:
            self.cuda = torch.cuda.get_rng_state(self.devices[0]).clone()
    finally:
      self.active = False

  def snapshot(self):
    return {
      "cpu": self.cpu.clone(),
      "cuda": None if self.cuda is None else self.cuda.clone(),
    }


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


@torch.no_grad()
def draw_recorded(bundle, batch_size, device):
  """The native sampler's exact operation/draw order, with all noises retained."""
  model, scheduler, low, high, feature_dim, window_size = bundle
  count = int(scheduler.num_timesteps)
  initial = torch.randn(batch_size, window_size, feature_dim, device=device)
  posterior = torch.empty(
    (count - 1, batch_size, window_size, feature_dim), device=device
  )
  x = initial
  for t in reversed(range(count)):
    times = torch.full((batch_size,), t, dtype=torch.long, device=device)
    eps = model(x, times)
    x0 = scheduler.sqrt_recip_alphas_cumprod[t] * x - (
      scheduler.sqrt_recipm1_alphas_cumprod[t] * eps
    )
    mean = (
      scheduler.posterior_mean_coef1[t] * x0 + scheduler.posterior_mean_coef2[t] * x
    )
    if t:
      noise = torch.randn_like(x)
      posterior[count - 1 - t].copy_(noise)
      x = mean + torch.sqrt(scheduler.posterior_variance[t]) * noise
    else:
      x = mean
  raw = (x + 1.0) / 2.0 * (high - low) + low
  assert raw.dtype == initial.dtype == posterior.dtype == torch.float32
  return raw, initial, posterior


class ArchiveRecorder:
  def __init__(
    self,
    directory,
    *,
    seed,
    device,
    provenance,
    pool_size=4096,
    batch_size=1024,
    refresh_interval=2400,
  ):
    self.directory = Path(directory).resolve()
    self.directory.mkdir(parents=True, exist_ok=False)
    self.stream = PrivateTorchStream(seed, device)
    self.pool_size = pool_size
    self.batch_size = batch_size
    self.interval = refresh_interval
    self.manifest = {
      "schema": SCHEMA,
      "status": "recording",
      "experiment": "0009",
      "gsi_seed": int(seed),
      "dtype": "float32",
      "feature_shape": [10, 59],
      "noise_order": "initial x_T then posterior[t=49,48,...,1]; no t=0 noise",
      "diffusion_steps": 50,
      "pool_size": pool_size,
      "batch_size": batch_size,
      "refresh_control_steps": refresh_interval,
      "rows": 0,
      "batches": [],
      "provenance": provenance,
    }
    atomic_json(self.directory / "manifest.json", self.manifest)
    (self.directory / "README.md").write_text(
      "# 0009 原始GSI配对数据\n\n"
      "所有候选原样保存，不去重、不做物理筛选或修正。每个batch文件保存float32 "
      "windows[N,10,59]、initial[N,10,59]及posterior[49,N,10,59]。"
      "posterior按t=49至1排列；每条噪声须与同一文件内同一行对应。"
      "四个启动批次合成4096条；之后每批对应一次1024条FIFO刷新。"
      "使用manifest中的实际control_step与fifo_head_before重放，不用墙钟时间。"
      "只有status=complete并通过归档核验的正式训练数据可用于00092。\n",
      encoding="utf-8",
    )
    self.events = None
    self.original = None

  def install(self, events):
    assert self.events is None
    original = events._ddpm_sample
    assert (
      original.__module__ == events.__name__ and original.__name__ == "_ddpm_sample"
    )
    self.events, self.original = events, original
    events._ddpm_sample = self.sample

  def uninstall(self):
    if self.events is not None:
      self.events._ddpm_sample = self.original
      self.events = None

  @torch.no_grad()
  def sample(self, env, n):
    if self.manifest["status"] != "recording":
      raise RuntimeError("archive is already sealed")
    before = int(self.manifest["rows"])
    step = int(getattr(env, "common_step_counter", 0))
    startup = before < self.pool_size
    assert n == self.batch_size
    if startup:
      assert step == 0 and before + n <= self.pool_size
      transaction, offset, head = 0, before, before
    else:
      transaction = 1 + (before - self.pool_size) // self.batch_size
      assert step == transaction * self.interval
      offset = 0
      head = int(getattr(env, "_smp_gsi_head", 0))
      assert head == (before - self.pool_size) % self.pool_size
    assert env._smp_bundle[1].num_timesteps == 50
    assert tuple(env._smp_bundle[4:]) == (59, 10)
    caller_cpu = torch.get_rng_state().clone()
    caller_cuda = [torch.cuda.get_rng_state(d).clone() for d in self.stream.devices]
    private_before = self.stream.snapshot()
    with self.stream.use():
      windows, initial, posterior = draw_recorded(env._smp_bundle, n, env.device)
    assert torch.equal(caller_cpu, torch.get_rng_state()), (
      "GSI changed training CPU RNG"
    )
    for state, device in zip(caller_cuda, self.stream.devices, strict=True):
      assert torch.equal(state, torch.cuda.get_rng_state(device)), (
        "GSI changed training CUDA RNG"
      )
    metadata = {
      "batch_index": len(self.manifest["batches"]),
      "transaction_index": transaction,
      "phase": "startup" if startup else "refresh",
      "control_step": step,
      "offset_in_transaction": offset,
      "fifo_head_before": head,
      "global_row_start": before,
      "rows": n,
      "decoder_batch_size": n,
    }
    payload = {
      "schema": SCHEMA,
      "metadata": metadata,
      "windows": windows.detach().cpu().contiguous(),
      "initial": initial.detach().cpu().contiguous(),
      "posterior": posterior.detach().cpu().contiguous(),
      "gsi_rng_before": private_before,
      "gsi_rng_after": self.stream.snapshot(),
    }
    name = f"batch_{metadata['batch_index']:04d}.pt"
    target = self.directory / name
    temporary = target.with_suffix(".pt.partial")
    if target.exists():
      raise FileExistsError(target)
    with temporary.open("xb") as stream:
      torch.save(payload, stream)
      stream.flush()
      os.fsync(stream.fileno())
    os.replace(temporary, target)
    entry = dict(
      metadata,
      file=name,
      bytes=target.stat().st_size,
      sha256=file_sha256(target),
      windows_sha256=tensor_sha256(windows),
    )
    self.manifest["rows"] = before + n
    self.manifest["batches"].append(entry)
    atomic_json(self.directory / "manifest.json", self.manifest)
    return windows

  def finish(self, env, *, completed_iterations, steps_per_env):
    control_steps = int(completed_iterations) * int(steps_per_env)
    assert int(env.common_step_counter) == control_steps
    refreshes = control_steps // self.interval
    expected_rows = self.pool_size + refreshes * self.batch_size
    assert self.manifest["rows"] == expected_rows
    assert (
      len(self.manifest["batches"]) == self.pool_size // self.batch_size + refreshes
    )
    self.manifest.update(
      status="complete",
      completed_iterations=int(completed_iterations),
      control_steps=control_steps,
      refreshes=refreshes,
      transactions=1 + refreshes,
      final_fifo_head=int(getattr(env, "_smp_gsi_head", 0)),
      final_pool_sha256=tensor_sha256(env._smp_gsi_pool),
    )
    atomic_json(self.directory / "manifest.json", self.manifest)
