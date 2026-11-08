"""Synthetic acceptance only; no saved GSI data, physics steps or PPO updates."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from experiments.ts2vec_g1.common import (
  choose_unique,
  config,
  fit_normalizer,
  nearest_centers,
  normalize,
  official,
  sampling_probabilities,
  sha256,
  window_digest,
  write_json,
)
from experiments.ts2vec_g1.module import G1ClusterModule
from experiments.ts2vec_g1.train import fit_encoder
from sklearn.cluster import KMeans


class ModuleTests(unittest.TestCase):
  def test_exact_final_deduplication_and_quota(self):
    x = np.zeros((5, 10, 59), dtype=np.float32)
    x[2, 4, 12] = np.nextafter(np.float32(0), np.float32(1))
    x[3, 0, 0] = np.nan  # Numerical reject must not poison accepted candidates.
    x[4, 4, 12] = 1
    mask = np.array([True, True, True, False, True])
    seen = set()
    ids, hashes, duplicate, surplus = choose_unique(x, mask, seen, 2)
    self.assertEqual(ids, [0, 2])
    self.assertEqual((duplicate, surplus, len(seen)), (1, 1, 2))
    np.testing.assert_array_equal(
      hashes[1], np.frombuffer(window_digest(x[2]), dtype=np.uint8)
    )
    again = choose_unique(x, mask, seen, 1)
    self.assertEqual(again[0], [4])
    self.assertEqual(again[2], 3)

  def test_channel_normalization_and_constant_channels(self):
    x = np.random.default_rng(7).normal(size=(96, 10, 59)).astype(np.float32)
    x[:, :, 3] = 4
    x[:, :, 7] *= 1e-9
    p = fit_normalizer(x)
    np.testing.assert_allclose(
      p["mean"], x.mean(axis=(0, 1), dtype=np.float64), atol=1e-14
    )
    np.testing.assert_allclose(
      p["std"], x.std(axis=(0, 1), dtype=np.float64), atol=1e-14
    )
    self.assertEqual(p["scale"][3], 1)
    self.assertEqual(p["scale"][7], 1)
    z = normalize(x, p)
    self.assertEqual(z.dtype, np.float32)
    self.assertTrue(np.isfinite(z).all())
    np.testing.assert_allclose(z[:, :, 3], 0)

  def test_current_pool_per_item_weights(self):
    labels = np.repeat(np.arange(3), [3072, 768, 256])
    p = sampling_probabilities(labels)
    np.testing.assert_allclose(np.bincount(labels, weights=p), [0.6, 0.3, 0.1])
    np.testing.assert_allclose(sampling_probabilities(np.array([7, 7])), [0.5, 0.5])
    np.testing.assert_allclose(
      sampling_probabilities(np.arange(16)), np.full(16, 1 / 16)
    )
    with self.assertRaises(ValueError):
      sampling_probabilities(np.array([], dtype=int))
    with self.assertRaises(ValueError):
      sampling_probabilities(np.array([16]))

  def test_official_files_and_actual_two_dimension_roundtrip(self):
    self.assertEqual(official().__module__, "ts2vec")
    rng = np.random.default_rng(2027)
    x = rng.normal(size=(96, 10, 59)).astype(np.float32)
    parameters = fit_normalizer(x)
    z = normalize(x, parameters)
    settings = dict(config()["ts2vec"], epochs=2)
    with tempfile.TemporaryDirectory(prefix="g1-ts2vec-test-") as temporary:
      for dimension in (64, 320):
        out = Path(temporary) / f"dim{dimension}"
        out.mkdir()
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model, epochs = fit_encoder(
          z, dimension, device, settings, 2027, out / "progress.json"
        )
        self.assertEqual((model.n_epochs, model.n_iters, len(epochs)), (2, 2, 2))
        model.save(str(out / "encoder_final.pt"))
        embedding = model.encode(
          z, mask="all_true", encoding_window="full_series", batch_size=64
        )
        self.assertEqual(embedding.shape, (96, dimension))
        centers = (
          KMeans(n_clusters=16, n_init=2, max_iter=30, random_state=2027)
          .fit(embedding)
          .cluster_centers_
        )
        np.save(out / "centers.npy", centers)
        np.savez(out / "normalizer.npz", **parameters)
        write_json(
          out / "bundle.json",
          dict(
            output_dims=dimension,
            hidden_dims=64,
            depth=10,
            encode_batch_size=64,
            files={
              name: sha256(out / name)
              for name in ["encoder_final.pt", "centers.npy", "normalizer.npz"]
            },
          ),
        )
        rng_before = torch.get_rng_state().clone()
        frozen = G1ClusterModule(out, device=device)
        again = frozen.encode(x, batch_size=37)
        self.assertTrue(torch.equal(rng_before, torch.get_rng_state()))
        np.testing.assert_allclose(embedding, again, atol=1e-5, rtol=1e-5)
        np.testing.assert_array_equal(
          frozen.predict(x), nearest_centers(embedding, centers)
        )
        np.testing.assert_array_equal(frozen.encode(x), frozen.encode(x))
        self.assertEqual(frozen.encode(x[:0]).shape, (0, dimension))
        # Inference must not change model weights or statistics.
        original_hash = sha256(out / "encoder_final.pt")
        frozen.predict(x[:8])
        self.assertEqual(sha256(out / "encoder_final.pt"), original_hash)


if __name__ == "__main__":
  torch.set_num_threads(2)
  unittest.main(verbosity=2)
