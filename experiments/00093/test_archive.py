"""Real native DDPM algebra and scheduler; fake deterministic denoiser, no simulator."""

from __future__ import annotations

import ast
import importlib.util
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
from gsi_archive import (
  ArchiveRecorder,
  PrivateTorchStream,
  decode,
  draw_recorded,
  tensor_sha256,
)

ROOT = Path(__file__).resolve().parents[2]
torch.set_num_threads(2)
spec = importlib.util.spec_from_file_location(
  "native_scheduler", ROOT / "src/smp/pretrain/scheduler.py"
)
scheduler_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scheduler_module)
tree = ast.parse((ROOT / "src/smp/rl/events.py").read_text())
fn = next(
  n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_ddpm_sample"
)
module = ast.parse("from __future__ import annotations\n")
module.body.append(fn)
namespace = {"torch": torch}
exec(
  compile(ast.fix_missing_locations(module), "actual_native_ddpm_sampler", "exec"),
  namespace,
)
native = namespace["_ddpm_sample"]


def model(x, times):
  return x * 0.1 + times[:, None, None] * 0.0001


bundle = (
  model,
  scheduler_module.DDPMScheduler(50),
  torch.full((59,), -1.0),
  torch.ones(59),
  59,
  10,
)
env = SimpleNamespace(
  _smp_bundle=bundle, device="cpu", common_step_counter=0, _smp_gsi_head=0
)
torch.manual_seed(102028)
expected = native(env, 2)
torch.manual_seed(12345)
training_state = torch.get_rng_state().clone()
stream = PrivateTorchStream(102028, "cpu")
with stream.use():
  raw, initial, posterior = draw_recorded(bundle, 2, "cpu")
assert torch.equal(raw, expected), "altered native DDPM algebra or random draw order"
assert torch.equal(training_state, torch.get_rng_state())
assert torch.equal(raw, decode(bundle, initial, posterior))
assert torch.equal(training_state, torch.get_rng_state())
initial_saved, posterior_saved = initial.clone(), posterior.clone()
decode(bundle, initial, posterior)
assert torch.equal(initial, initial_saved) and torch.equal(posterior, posterior_saved)
with stream.use():
  raw_second, _, _ = draw_recorded(bundle, 2, "cpu")
assert not torch.equal(raw, raw_second), "private RNG did not advance"
assert torch.equal(training_state, torch.get_rng_state())
try:
  with stream.use():
    torch.randn(10)
    raise RuntimeError("intentional test")
except RuntimeError:
  pass
assert torch.equal(training_state, torch.get_rng_state()), (
  "exception leaked private RNG"
)

with tempfile.TemporaryDirectory(prefix="00093-archive-test-") as temporary:
  directory = Path(temporary) / "archive"
  recorder = ArchiveRecorder(
    directory,
    seed=102028,
    device="cpu",
    provenance={"test_only": True},
    pool_size=4,
    batch_size=2,
    refresh_interval=24,
  )
  first, second = recorder.sample(env, 2), recorder.sample(env, 2)
  env._smp_gsi_pool = torch.cat([first, second])
  env.common_step_counter = 24
  refresh = recorder.sample(env, 2)
  env._smp_gsi_pool[:2] = refresh
  env._smp_gsi_head = 2
  recorder.finish(env, completed_iterations=1, steps_per_env=24)
  manifest = json.loads((directory / "manifest.json").read_text())
  assert manifest["status"] == "complete" and manifest["rows"] == 6
  assert manifest["final_pool_sha256"] == tensor_sha256(env._smp_gsi_pool)
  previous = None
  for entry in manifest["batches"]:
    data = torch.load(directory / entry["file"], weights_only=False)
    assert torch.equal(
      data["windows"], decode(bundle, data["initial"], data["posterior"])
    )
    if previous is not None:
      assert torch.equal(previous["cpu"], data["gsi_rng_before"]["cpu"])
    previous = data["gsi_rng_after"]
  try:
    ArchiveRecorder(directory, seed=102028, device="cpu", provenance={})
  except FileExistsError:
    pass
  else:
    raise AssertionError("overwrote existing archive")
  try:
    recorder.sample(env, 2)
  except RuntimeError:
    pass
  else:
    raise AssertionError("appended after seal")
print(
  "ARCHIVE_PROTOCOL_TESTS=PASS (CPU, exact native sampler parity, replay, RNG isolation, FIFO, no overwrite)"
)
