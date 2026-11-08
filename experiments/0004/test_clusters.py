"""CPU contract/FIFO/sampling tests, with optional synthetic bundle roundtrip.

Temporary artificial module files test loading, not trained-model quality.
No real GSI data, policy learning, simulator steps or persistent weights.
"""

from __future__ import annotations
import __future__

import argparse
import ast
import importlib
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from smp.rl import gsi_clusters as c  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument(
  "--code-dir", type=Path, default=Path("/root/gpufree-data/ts2vec_g1_code")
)
args = parser.parse_args()


class ProbeModule:
  def __init__(self):
    self.centers = np.zeros((16, 64), dtype=np.float32)
    self.centers[:, 0] = np.arange(16)
    self.fail = False

  def encode(self, x):
    if self.fail:
      raise ValueError("injected encoding failure")
    random.random(), np.random.rand(), torch.rand(2)
    out = np.zeros((len(x), 64), dtype=np.float32)
    out[:, 0] = x[:, 0, 0]
    x[:] = -999  # An external encoder cannot mutate the actual pool input.
    return out


class Gate:
  lo, hi = np.full(29, -1e9), np.full(29, 1e9)
  config = SimpleNamespace(joint_tolerance_rad=1e-5)

  def process(self, x):
    return (
      x.copy(),
      np.ones(len(x), dtype=bool),
      dict(generated=len(x), qualified=len(x)),
    )


def environment():
  pool = torch.zeros(8, 10, 59)
  pool[:, 0, 0] = torch.arange(8)
  env = SimpleNamespace(
    device="cpu",
    _smp_gsi_pool=pool,
    _smp_gsi_repaired_rows=torch.ones(8, dtype=torch.bool),
    _smp_cluster_module=ProbeModule(),
    _smp_cluster_manifest={"output_dims": 64},
    _smp_cluster_stats=dict(
      classification_updates=0, classified_windows=0, reset_calls=0, reset_draws=0
    ),
    _smp_cluster_draw_counts=torch.zeros(16, dtype=torch.long),
    _smp_physics=Gate(),
    _smp_physics_batch=4,
    _smp_physics_totals=dict(
      updates=0, generated=0, qualified=0, inserted=0, surplus=0, repair_inserted=0
    ),
    common_step_counter=2400,
  )
  c.commit(env, c._plan(env, c.classify(env, pool)), 8)
  return env


def actual_event(name, draw=None, prime=None):
  tree = ast.parse((ROOT / "src/smp/rl/events.py").read_text(encoding="utf-8"))
  node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
  node.decorator_list = []
  namespace = {"torch": torch, "_ddpm_sample": draw, "_prime_sim_and_buffer": prime}
  exec(
    compile(
      ast.Module(body=[node], type_ignores=[]),
      str(ROOT / "src/smp/rl/events.py"),
      "exec",
      flags=__future__.annotations.compiler_flag,
    ),
    namespace,
  )
  return namespace[name]


def draw(env, count):
  result = torch.zeros(count, 10, 59)
  result[:, 0, 0] = torch.arange(count) + 10
  return result


class ClusterTests(unittest.TestCase):
  def test_per_item_cap_empty_clusters_and_empirical_sampling(self):
    labels = np.repeat(np.arange(3), [3072, 768, 256])
    d = c.pool_distribution(labels)
    np.testing.assert_allclose(d["mass"][:3], [0.6, 0.3, 0.1])
    np.testing.assert_array_equal(d["weights"][:3], [1, 2, 2])
    self.assertTrue((d["mass"][3:] == 0).all())
    np.testing.assert_allclose(
      c.pool_distribution(np.array([5, 5]))["probability"], [0.5, 0.5]
    )
    with self.assertRaises(ValueError):
      c.pool_distribution(np.array([], dtype=int))
    env = environment()
    env._smp_gsi_pool = torch.zeros(len(labels), 10, 59)
    c.commit(env, c._plan(env, labels), len(labels))
    torch.manual_seed(2027)
    indices = c.sample_indices(env, 200000)
    empirical = np.bincount(labels[indices.numpy()], minlength=16) / len(indices)
    np.testing.assert_allclose(empirical, d["mass"], atol=0.004, rtol=0)
    self.assertEqual(c.summary(env)["reset_draws"], 200000)

  def test_encoding_preserves_random_streams_flags_and_pool(self):
    env = environment()
    original = env._smp_gsi_pool.clone()
    for precision in ("highest", "high", "medium"):
      torch.set_float32_matmul_precision(precision)
      flags = (
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.allow_tf32,
        torch.backends.cuda.matmul.allow_tf32,
      )
      t, n, p = torch.get_rng_state().clone(), np.random.get_state(), random.getstate()
      c.classify(env, original)
      self.assertTrue(torch.equal(t, torch.get_rng_state()))
      self.assertTrue(np.array_equal(n[1], np.random.get_state()[1]))
      self.assertEqual(n[2:], np.random.get_state()[2:])
      self.assertEqual(p, random.getstate())
      self.assertEqual(torch.get_float32_matmul_precision(), precision)
      self.assertEqual(
        flags,
        (
          torch.backends.cudnn.benchmark,
          torch.backends.cudnn.deterministic,
          torch.backends.cudnn.allow_tf32,
          torch.backends.cuda.matmul.allow_tf32,
        ),
      )
    self.assertTrue(torch.equal(original, env._smp_gsi_pool))

  def test_actual_fifo_refresh_relabels_wrap_and_keeps_untouched_rows(self):
    env = environment()
    old_pool = env._smp_gsi_pool.clone()
    env._smp_gsi_head = 6
    actual_event("gsi_refresh", draw=draw)(env, num_samples=4)
    changed = [6, 7, 0, 1]
    self.assertEqual(env._smp_cluster_labels[changed].tolist(), [10, 11, 12, 13])
    self.assertEqual(env._smp_cluster_labels[2:6].tolist(), [2, 3, 4, 5])
    self.assertTrue(torch.equal(old_pool[2:6], env._smp_gsi_pool[2:6]))
    self.assertEqual(env._smp_gsi_head, 2)
    self.assertEqual(c.summary(env)["classification_updates"], 2)
    np.testing.assert_allclose(
      env._smp_cluster_probability.numpy(),
      c.pool_distribution(env._smp_cluster_labels.numpy())["probability"],
    )

  def test_encoding_failure_does_not_partially_replace_pool_or_labels(self):
    env = environment()
    env._smp_cluster_module.fail = True
    before = [
      env._smp_gsi_pool.clone(),
      env._smp_cluster_labels.clone(),
      env._smp_cluster_probability.clone(),
    ]
    with self.assertRaises(ValueError):
      actual_event("gsi_refresh", draw=draw)(env, num_samples=4)
    for a, b in zip(
      before,
      [env._smp_gsi_pool, env._smp_cluster_labels, env._smp_cluster_probability],
      strict=True,
    ):
      self.assertTrue(torch.equal(a, b))
    self.assertFalse(hasattr(env, "_smp_gsi_head"))

  def test_actual_reset_uses_chosen_window_and_partial_env_ids(self):
    env = environment()
    calls = []

    def prime(env, ids, windows):
      calls.append((ids.clone(), windows.clone()))

    actual_event("gsi_reset", prime=prime)(env, torch.tensor([1, 4, 7]))
    self.assertEqual(c.summary(env)["reset_draws"], 3)
    self.assertEqual(calls[0][0].tolist(), [1, 4, 7])
    self.assertEqual(tuple(calls[0][1].shape), (3, 10, 59))
    for window in calls[0][1]:
      self.assertTrue(any(torch.equal(window, row) for row in env._smp_gsi_pool))
    actual_event("gsi_reset", prime=prime)(env, torch.tensor([], dtype=torch.long))
    self.assertEqual(len(calls), 1)
    del env._smp_cluster_module
    torch.manual_seed(3)
    state = torch.get_rng_state()
    expected = torch.randint(0, 8, (3,))
    torch.set_rng_state(state)
    actual_event("gsi_reset", prime=prime)(env, torch.tensor([1, 4, 7]))
    self.assertTrue(torch.equal(calls[-1][1], env._smp_gsi_pool[expected]))

  def test_actual_64_and_320_bundle_loading(self):
    if not args.code_dir.is_dir():
      self.fail(f"external code directory required for bundle tests: {args.code_dir}")
    sys.path.insert(0, str(args.code_dir.resolve()))
    external = importlib.import_module("experiments.ts2vec_g1.common")
    with tempfile.TemporaryDirectory(prefix="g1-cluster-contract-") as temporary:
      for dim in (64, 320):
        out = Path(temporary) / f"dim{dim}"
        out.mkdir()
        model = external.official()(
          input_dims=59, output_dims=dim, hidden_dims=64, depth=10, device="cpu"
        )
        # Synthetic file-format fixture ONLY. No claim of 31240 training updates.
        model.net.n_averaged.fill_(31241)
        model.save(str(out / "encoder_final.pt"))
        rng = np.random.default_rng(2027)
        np.save(out / "centers.npy", rng.normal(size=(16, dim)).astype(np.float32))
        np.savez(
          out / "normalizer.npz",
          mean=np.zeros(59),
          std=np.ones(59),
          scale=np.ones(59),
          threshold=np.array(1e-6),
        )
        files = {
          name: c.sha256(out / name)
          for name in ["encoder_final.pt", "centers.npy", "normalizer.npz"]
        }
        bundle = dict(
          input_shape=[10, 59],
          output_dims=dim,
          hidden_dims=64,
          depth=10,
          mask="all_true",
          pooling="full_series_max",
          encode_batch_size=64,
          source_ts2vec_commit=c.OFFICIAL_COMMIT,
          config=external.config(),
          files=files,
          normalizer_sha256=files["normalizer.npz"],
          dataset_sha256="0" * 64,
        )
        (out / "bundle.json").write_text(json.dumps(bundle), encoding="utf-8")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_before = torch.get_rng_state().clone()
        cuda_before = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
        frozen, manifest = c.load_module(args.code_dir, out, dim, device)
        env = environment()
        env.device = device
        env._smp_gsi_pool = env._smp_gsi_pool.to(device)
        env._smp_cluster_module, env._smp_cluster_manifest = frozen, manifest
        labels = c.classify(env, env._smp_gsi_pool)
        self.assertTrue(torch.equal(torch_before, torch.get_rng_state()))
        if cuda_before is not None:
          self.assertTrue(torch.equal(cuda_before, torch.cuda.get_rng_state()))
        self.assertEqual(labels.shape, (8,))
        c.check_frozen(env)
        frozen.centers[0, 0] += 1
        with self.assertRaises(AssertionError):
          c.check_frozen(env)


torch.set_num_threads(2)
result = unittest.TextTestRunner(verbosity=2).run(
  unittest.defaultTestLoader.loadTestsFromTestCase(ClusterTests)
)
if not result.wasSuccessful():
  raise SystemExit(1)
print("GSI_CLUSTER_TESTS=PASS")
