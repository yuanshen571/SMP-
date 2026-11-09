"""Synthetic lifetime/TensorBoard verifier roundtrip; no policy or simulator.

Temporary baseline metadata is a protocol fixture, never a training input.
"""

import json
import sys
import tempfile
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

root = Path(__file__).resolve().parents[2]
exp = root / "experiments/0005"
sys.path[:0] = [str(exp), str(root / "src")]
import verify_completion as verify  # noqa: E402
from preflight import sha256  # noqa: E402
from test_mixed import FakeEnv  # noqa: E402

from smp.rl.mixed_cache import MixedConfig, install  # noqa: E402

torch.set_num_threads(2)
with tempfile.TemporaryDirectory(prefix="0005-verifier-protocol-") as folder:
  out = Path(folder)
  env = FakeEnv()
  cache = install(env, config=MixedConfig(), audit_path=out / "mixed_cache_audit.json")
  writer = SummaryWriter(str(out))
  count = length = 0
  for step in range(1, 48001):
    env.step(None)
    for period in (80, 100, 120):
      if step >= 2 * period and step % period == 0:
        count += 1
        length += period
    if step % 24 == 0:
      it = step // 24
      for key, value in cache.stats.items():
        writer.add_scalar("0005/Mixed/" + key, value, it)
      writer.add_scalar("0005/Mixed/active", int(cache.active), it)
      writer.add_scalar(
        "0005/Mixed/size", 0 if cache.pool is None else len(cache.pool), it
      )
      writer.add_scalar("0005/Mixed/policy_probability", 0.1, it)
      writer.add_scalar("0005/Count/complete_episode", count, it)
      writer.add_scalar("0005/Sum/complete_episode_length_steps", length, it)
      count = length = 0
  writer.close()
  baseline = {
    "experiment": "00031",
    "completed_iterations": 10000,
    "initial_models_sha256": "synthetic_protocol_only",
    "gsi_model_manifest": {},
    "gsi_physics_config": {},
  }
  path = out / "baseline_fixture.json"
  path.write_text(json.dumps(baseline))
  spec = json.loads((exp / "EXPERIMENT.json").read_text())
  info = {
    "mixed_cache": spec["mixed_cache"],
    "mixed_cache_summary": cache.summary(),
    "baseline_info": str(path),
    "baseline_info_sha256": sha256(path),
    "initial_models_sha256": baseline["initial_models_sha256"],
    "gsi_model_manifest": {},
    "gsi_physics_config": {},
  }

  class ProtocolReport(verify.Report):
    def check(self, name, ok, detail=""):
      return super().check("synthetic protocol: " + name, ok, detail)

  report = ProtocolReport()
  verify.check_mixed(info, spec, verify.load_scalars(out), out, 2000, 3, 24, report)
  assert report.failed == 0, report.markdown()
print(
  "MIXED_COMPLETION_VERIFIER_PROTOCOL=PASS (synthetic lifecycle/TensorBoard; no real policy or simulator)"
)
