"""Completion checks from saved data, averaged encoders and epoch summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from experiments.ts2vec_g1.common import config, now, read_json, sha256, write_json
from experiments.ts2vec_g1.generate import verify_dataset
from experiments.ts2vec_g1.module import G1ClusterModule


def verify_model(run, dimension):
  cfg = config()
  out = run / "models" / f"dim{dimension}"
  result = read_json(out / "result.json")
  manifest = read_json(run / "dataset/manifest.json")
  bundle = read_json(out / "bundle.json")
  if result["status"] != "PASS" or bundle["config"] != cfg:
    raise AssertionError("model result/configuration mismatch")
  for name, digest in result["files"].items():
    if sha256(out / name) != digest:
      raise AssertionError(f"model artifact changed: {name}")
  if result["dataset_sha256"] != manifest["files"]["features.npy"]:
    raise AssertionError("model used another dataset")
  normalizer = manifest["files"]["normalizer.npz"]
  if (
    result["normalizer_sha256"] != normalizer
    or sha256(out / "normalizer.npz") != normalizer
  ):
    raise AssertionError("model used another input normalizer")
  updates = (
    cfg["unique_windows"] // cfg["ts2vec"]["batch_size"] * cfg["ts2vec"]["epochs"]
  )
  if (
    result["updates"] != updates
    or result["epochs"] != 20
    or result["dimension"] != dimension
  ):
    raise AssertionError("training budget/dimension mismatch")
  epochs = result["epoch_metrics"]
  if [entry["epoch"] for entry in epochs] != list(range(1, 21)):
    raise AssertionError("epoch records missing")
  if [entry["updates"] for entry in epochs] != [1562 * i for i in range(1, 21)]:
    raise AssertionError("per-epoch update counts differ")
  if not np.isfinite([entry["loss"] for entry in epochs]).all():
    raise AssertionError("non-finite epoch loss")
  state = torch.load(out / "encoder_final.pt", map_location="cpu", weights_only=True)
  if int(state["n_averaged"]) != updates + 1:
    raise AssertionError("saved file is not the final upstream averaged encoder")
  if not all(bool(torch.isfinite(value).all()) for value in state.values()):
    raise AssertionError("non-finite final encoder weights")
  if list(out.glob("**/events.out.tfevents.*")):
    raise AssertionError("TensorBoard files must not be retained")
  centers = np.load(out / "centers.npy", allow_pickle=False)
  labels = np.load(out / "dataset_labels.npy", allow_pickle=False)
  if centers.shape != (16, dimension) or not np.isfinite(centers).all():
    raise AssertionError("invalid centers")
  if labels.shape != (100000,) or labels.min() < 0 or labels.max() > 15:
    raise AssertionError("invalid labels")
  diag = read_json(out / "diagnostics.json")
  if diag["cluster_counts"] != np.bincount(labels, minlength=16).tolist():
    raise AssertionError("cluster counts mismatch")
  module = G1ClusterModule(out, device="cpu")
  x = np.load(run / "dataset/features.npy", mmap_mode="r", allow_pickle=False)
  probe_ids = np.random.default_rng(2027).choice(len(x), size=256, replace=False)
  predicted = module.predict(np.array(x[probe_ids], copy=True))
  # CPU/GPU numeric variation may change a near-tied boundary. Require the stored
  # assigned center to be numerically nearest, rather than claim bitwise equality.
  if not np.array_equal(predicted, labels[probe_ids]):
    encoded = module.encode(np.array(x[probe_ids], copy=True))
    distances = ((encoded[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    assigned = distances[np.arange(len(probe_ids)), labels[probe_ids]]
    np.testing.assert_allclose(assigned, distances.min(axis=1), atol=1e-5, rtol=1e-5)
  probability = module.current_pool_probabilities(labels[:4096])
  if not np.isfinite(probability).all() or not np.isclose(probability.sum(), 1):
    raise AssertionError("invalid future pool weighting")
  return result


def verify_all(run):
  import importlib.metadata as metadata

  preflight = read_json(run / "preflight.json")
  if any(
    metadata.version(name) != value for name, value in preflight["versions"].items()
  ):
    raise AssertionError("shared environment versions changed during this pipeline")
  if list(run.glob("**/events.out.tfevents.*")):
    raise AssertionError("TensorBoard files must not be retained")
  manifest = verify_dataset(run / "dataset")
  results = [verify_model(run, dimension) for dimension in (64, 320)]
  if len({r["dataset_sha256"] for r in results}) != 1:
    raise AssertionError("the two models used different data")
  if len({r["normalizer_sha256"] for r in results}) != 1:
    raise AssertionError("the two models used different normalization")
  write_json(
    run / "result.json",
    dict(
      status="PASS",
      finished_at=now(),
      dataset_sha256=manifest["files"]["features.npy"],
      unique_windows=100000,
      input_shape=[10, 59],
      models=results,
      scope="data generation, two TS2Vec models and K-means; no PPO/policy evaluation",
    ),
  )
  rows = [
    "# G1 TS2Vec＋K-means 完成报告",
    "",
    "数据：100,000 条去重后的最终合格片段，完整十帧、每帧59维、50 Hz。",
    f"数据 SHA256：`{manifest['files']['features.npy']}`",
    "",
    "| 编码维度 | epoch | 更新次数 | 16簇数量 |",
    "|---|---:|---:|---|",
  ]
  for result in results:
    dim = result["dimension"]
    counts = read_json(run / f"models/dim{dim}/diagnostics.json")["cluster_counts"]
    rows.append(f"| {dim} | 20 | {result['updates']} | {counts} |")
  rows += [
    "",
    "两版输入数据与归一化参数完全相同。模块完成不等于已证明策略训练收益。",
    "本轮未训练 PPO，未生成策略评估、视频或完整仿真轨迹。",
    "最终数据在 dataset/，两版独立模块在 models/dim64/ 和 models/dim320/。",
  ]
  (run / "REPORT.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
  print("G1_TS2VEC_KMEANS_CHECK=PASS", flush=True)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--run", type=Path, required=True)
  args = parser.parse_args()
  verify_all(args.run.resolve())
