"""One final averaged encoder and K-means fit per dimension, same saved data."""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from experiments.ts2vec_g1.common import (
  config,
  nearest_centers,
  normalize,
  now,
  official,
  seed_all,
  sha256,
  write_json,
)
from experiments.ts2vec_g1.generate import verify_dataset
from experiments.ts2vec_g1.module import G1ClusterModule
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits


def fit_encoder(z, dimension, device, settings, seed, progress):
  """Single upstream fit() call. Also used by the synthetic local acceptance."""
  seed_all(seed)
  model = official()(
    input_dims=59,
    output_dims=dimension,
    hidden_dims=settings["hidden_dims"],
    depth=settings["depth"],
    device=device,
    lr=settings["lr"],
    batch_size=settings["batch_size"],
    max_train_length=None,
    temporal_unit=0,
  )
  # Isolate data-loader/crop random streams from dimension-dependent initialization.
  # GPU dropout still depends on tensor shape; no claim of identical trajectories.
  seed_all(seed)
  epoch_metrics, last_epoch = [], time.monotonic()

  def after_iter(instance, loss):
    if not np.isfinite(loss):
      raise FloatingPointError("non-finite TS2Vec loss")

  def after_epoch(instance, loss):
    nonlocal last_epoch
    if not np.isfinite(loss):
      raise FloatingPointError("non-finite epoch loss")
    elapsed = time.monotonic() - last_epoch
    last_epoch = time.monotonic()
    epoch_metrics.append(
      dict(
        epoch=instance.n_epochs,
        updates=instance.n_iters,
        loss=float(loss),
        seconds=elapsed,
      )
    )
    write_json(
      progress, dict(stage="training", dimension=dimension, **epoch_metrics[-1])
    )
    print(
      f"[dim{dimension}] epoch={instance.n_epochs} updates={instance.n_iters} loss={loss:.6g}",
      flush=True,
    )

  model.after_iter_callback, model.after_epoch_callback = after_iter, after_epoch
  losses = model.fit(z, n_epochs=settings["epochs"], n_iters=None, verbose=False)
  expected = len(z) // settings["batch_size"] * settings["epochs"]
  if (model.n_epochs, model.n_iters, len(losses)) != (
    settings["epochs"],
    expected,
    settings["epochs"],
  ):
    raise AssertionError("upstream training counters differ from fixed budget")
  model.net.eval()
  model.net.requires_grad_(False)
  return model, epoch_metrics


def embedding_diagnostics(embedding, seed):
  std = embedding.std(axis=0, dtype=np.float64)
  if not np.isfinite(embedding).all() or np.all(std < 1e-8):
    raise ValueError("non-finite or fully collapsed representation")
  rng = np.random.default_rng(seed)
  n = len(embedding)
  first = rng.integers(0, n, 100000)
  second = (first + rng.integers(1, n, 100000)) % n
  distances = np.empty(len(first), dtype=np.float64)
  for start in range(0, len(first), 4096):
    a, b = first[start : start + 4096], second[start : start + 4096]
    distances[start : start + 4096] = np.linalg.norm(
      embedding[a].astype(np.float64) - embedding[b].astype(np.float64), axis=1
    )
  return dict(
    per_dimension_std=std.tolist(),
    near_zero_dimensions=np.flatnonzero(std < 1e-8).tolist(),
    pair_count=len(first),
    pair_sampling="seeded random pairs of different dataset indices",
    pair_distance_quantiles=dict(
      zip(
        ["min", "p1", "p5", "p50", "p95", "p99", "max"],
        np.quantile(distances, [0, 0.01, 0.05, 0.5, 0.95, 0.99, 1]).tolist(),
        strict=True,
      )
    ),
  )


def train(run, dimension, device):
  from experiments.ts2vec_g1.verify import verify_model

  cfg = config()
  if dimension not in cfg["ts2vec"]["output_dims"]:
    raise ValueError("only the agreed 64 and 320 dimensions are allowed")
  dataset = run / "dataset"
  manifest = verify_dataset(dataset)
  out = run / "models" / f"dim{dimension}"
  if (out / "result.json").exists():
    verify_model(run, dimension)
    return
  if out.exists():
    raise RuntimeError(f"unfinished model exists: {out}; stop, do not overwrite")
  out.mkdir(parents=True)
  started = time.monotonic()
  x = np.load(dataset / "features.npy", mmap_mode="r", allow_pickle=False)
  with np.load(dataset / "normalizer.npz", allow_pickle=False) as p:
    normalizer = {key: p[key] for key in p.files}
  z = normalize(x, normalizer)  # RAM only; do not save a second full dataset.
  settings = cfg["ts2vec"]
  model, epochs = fit_encoder(
    z, dimension, device, settings, cfg["seed"], out / "progress.json"
  )
  torch.cuda.synchronize() if device.startswith("cuda") else None
  train_seconds = time.monotonic() - started
  model.save(str(out / "encoder_final.pt"))
  before = time.monotonic()
  embedding = model.encode(
    z,
    mask="all_true",
    encoding_window="full_series",
    batch_size=settings["encode_batch_size"],
  )
  if embedding.shape != (len(x), dimension) or embedding.dtype != np.float32:
    raise ValueError("invalid full-series embedding shape/type")
  diagnostics = embedding_diagnostics(embedding, cfg["seed"])
  encode_seconds = time.monotonic() - before
  before = time.monotonic()
  with threadpool_limits(limits=8):
    kmeans = KMeans(**cfg["kmeans"]).fit(embedding)
    sklearn_labels = kmeans.predict(embedding)
  labels = nearest_centers(embedding, kmeans.cluster_centers_)
  different = np.flatnonzero(labels != sklearn_labels)
  # float32 sklearn and stable float64 distances can resolve a near tie
  # differently. Reject a substantive difference, record numerical ties.
  if len(different):
    a = embedding[different].astype(np.float64)
    centers64 = kmeans.cluster_centers_.astype(np.float64)
    assigned = np.square(a - centers64[sklearn_labels[different]]).sum(axis=1)
    nearest = np.square(a - centers64[labels[different]]).sum(axis=1)
    np.testing.assert_allclose(assigned, nearest, atol=1e-5, rtol=1e-5)
  counts = np.bincount(labels, minlength=16)
  if (counts == 0).any() or len(np.unique(kmeans.cluster_centers_, axis=0)) != 16:
    raise ValueError("empty or duplicate clusters; stop without parameter changes")
  np.save(out / "centers.npy", kmeans.cluster_centers_, allow_pickle=False)
  np.save(out / "dataset_labels.npy", labels.astype(np.int16), allow_pickle=False)
  shutil.copyfile(dataset / "normalizer.npz", out / "normalizer.npz")
  cluster_seconds = time.monotonic() - before
  bundle_files = ["encoder_final.pt", "centers.npy", "normalizer.npz"]
  write_json(
    out / "bundle.json",
    dict(
      format_version=1,
      input_shape=[10, 59],
      output_dims=dimension,
      hidden_dims=settings["hidden_dims"],
      depth=settings["depth"],
      encode_batch_size=settings["encode_batch_size"],
      mask="all_true",
      pooling="full_series_max",
      normalizer_sha256=manifest["files"]["normalizer.npz"],
      dataset_sha256=manifest["files"]["features.npy"],
      source_ts2vec_commit=cfg["official_ts2vec_commit"],
      config=cfg,
      files={name: sha256(out / name) for name in bundle_files},
    ),
  )
  # Re-load the saved average encoder; compare ALL examples and frozen labels.
  before = time.monotonic()
  frozen = G1ClusterModule(out, device=device)
  again = frozen.encode(x)
  np.testing.assert_allclose(again, embedding, atol=1e-5, rtol=1e-5)
  np.testing.assert_array_equal(nearest_centers(again, frozen.centers), labels)
  probe = frozen.encode(np.array(x[:131], copy=True), batch_size=37)
  np.testing.assert_allclose(probe, embedding[:131], atol=1e-5, rtol=1e-5)
  reload_error = float(np.max(np.abs(again - embedding)))
  batching_error = float(np.max(np.abs(probe - embedding[:131])))
  verify_seconds = time.monotonic() - before
  diagnostics.update(
    cluster_counts=counts.tolist(),
    cluster_proportions=(counts / len(x)).tolist(),
    inertia=float(kmeans.inertia_),
    iterations=int(kmeans.n_iter_),
    kmeans=cfg["kmeans"],
    sklearn_near_tie_label_count=int(len(different)),
    representation_transform="raw final full-series embeddings; no PCA/L2/extra scaling",
    reload_max_abs_error=reload_error,
    batching_max_abs_error=batching_error,
  )
  write_json(out / "diagnostics.json", diagnostics)
  artifacts = bundle_files + ["dataset_labels.npy", "bundle.json", "diagnostics.json"]
  write_json(
    out / "result.json",
    dict(
      status="PASS",
      finished_at=now(),
      dimension=dimension,
      epochs=settings["epochs"],
      updates=model.n_iters,
      per_epoch_updates=len(x) // 64,
      dropped_per_epoch=len(x) % 64,
      epoch_metrics=epochs,
      dataset_sha256=manifest["files"]["features.npy"],
      normalizer_sha256=manifest["files"]["normalizer.npz"],
      timings=dict(
        training=train_seconds,
        encoding=encode_seconds,
        kmeans=cluster_seconds,
        verification=verify_seconds,
        total=time.monotonic() - started,
      ),
      files={name: sha256(out / name) for name in artifacts},
    ),
  )
  verify_model(run, dimension)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--run", type=Path, required=True)
  parser.add_argument("--dimension", type=int, choices=(64, 320), required=True)
  parser.add_argument("--device", default="cuda:0")
  args = parser.parse_args()
  torch.set_num_threads(8)
  train(args.run.resolve(), args.dimension, args.device)
