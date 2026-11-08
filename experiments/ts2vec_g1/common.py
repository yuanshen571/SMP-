"""Small shared helpers; no simulator import or side effect at import time."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
VENDOR = ROOT / "third_party/ts2vec"


def now():
  return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def read_json(path):
  return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_name(path.name + ".tmp")
  temporary.write_text(
    json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
    encoding="utf-8",
  )
  os.replace(temporary, path)


def sha256(path):
  h = hashlib.sha256()
  with Path(path).open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
      h.update(block)
  return h.hexdigest()


def config():
  return read_json(HERE / "config.json")


def git_commit():
  return subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
  ).strip()


def official():
  """Load exact upstream files; refuse a pre-existing conflicting top-level import."""
  manifest = read_json(VENDOR / "UPSTREAM.json")
  for name, digest in manifest["files"].items():
    if sha256(VENDOR / name) != digest:
      raise RuntimeError(f"vendored TS2Vec file changed: {name}")
  if manifest["commit"] != config()["official_ts2vec_commit"]:
    raise RuntimeError("official TS2Vec revision does not match configuration")
  for name in ("models", "utils", "ts2vec"):
    module = sys.modules.get(name)
    if module is not None:
      origin = Path(module.__file__).resolve()
      if VENDOR.resolve() not in origin.parents:
        raise RuntimeError(f"conflicting import {name}: {origin}; use a fresh process")
  if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))
  return importlib.import_module("ts2vec").TS2Vec


def seed_all(seed):
  import torch

  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True
  torch.backends.cuda.matmul.allow_tf32 = False
  torch.backends.cudnn.allow_tf32 = False


def validate_windows(x):
  if x.dtype != np.float32 or x.ndim != 3 or x.shape[1:] != (10, 59):
    raise ValueError(f"expected float32 [N,10,59], got {x.dtype} {x.shape}")
  if not np.isfinite(x).all():
    raise ValueError("non-finite G1 input")


def fit_normalizer(x, threshold=1e-6):
  validate_windows(x)
  if len(x) == 0:
    raise ValueError("empty dataset")
  mean = np.mean(x, axis=(0, 1), dtype=np.float64)
  squared = np.zeros(59, dtype=np.float64)
  for start in range(0, len(x), 4096):
    residual = x[start : start + 4096].astype(np.float64) - mean
    squared += np.sum(residual * residual, axis=(0, 1))
  std = np.sqrt(squared / (len(x) * 10))
  scale = np.where(std < threshold, 1.0, std)
  return dict(mean=mean, std=std, scale=scale, threshold=np.array(threshold))


def normalize(x, parameters):
  validate_windows(x)
  z = ((x.astype(np.float64) - parameters["mean"]) / parameters["scale"]).astype(
    np.float32
  )
  if not np.isfinite(z).all():
    raise FloatingPointError("normalization produced non-finite features")
  return z


def window_digest(window):
  return hashlib.sha256(np.ascontiguousarray(window, dtype="<f4").tobytes()).digest()


def choose_unique(corrected, passed, seen, remaining):
  """First qualified occurrence in generation order; exact final float32 bytes."""
  if (
    corrected.dtype != np.float32
    or corrected.ndim != 3
    or corrected.shape[1:] != (10, 59)
  ):
    raise ValueError("expected float32 [N,10,59]")
  if passed.shape != (len(corrected),) or passed.dtype != np.bool_:
    raise ValueError("invalid qualification mask")
  if not np.isfinite(corrected[passed]).all():
    raise ValueError("qualified windows must be finite")
  ids, hashes, duplicates, surplus = [], [], 0, 0
  for row in np.flatnonzero(passed):
    digest = window_digest(corrected[row])
    if digest in seen:
      duplicates += 1
    elif len(ids) < remaining:
      seen.add(digest)
      ids.append(int(row))
      hashes.append(np.frombuffer(digest, dtype=np.uint8))
    else:
      surplus += 1
  return ids, hashes, duplicates, surplus


def nearest_centers(embeddings, centers):
  labels = np.empty(len(embeddings), dtype=np.int64)
  for start in range(0, len(embeddings), 512):
    difference = embeddings[start : start + 512, None, :].astype(np.float64) - centers[
      None, :, :
    ].astype(np.float64)
    labels[start : start + 512] = np.square(difference).sum(axis=2).argmin(axis=1)
  return labels


def sampling_probabilities(labels, *, cap=2.0, clusters=16):
  labels = np.asarray(labels)
  if (
    labels.ndim != 1
    or not len(labels)
    or labels.dtype.kind not in "iu"
    or labels.min() < 0
    or labels.max() >= clusters
    or cap < 1
  ):
    raise ValueError("need a nonempty valid current pool and cap >= 1")
  counts = np.bincount(labels, minlength=clusters)
  weights = np.zeros(clusters, dtype=np.float64)
  occupied = counts > 0
  weights[occupied] = np.minimum(cap, np.sqrt(counts.max() / counts[occupied]))
  probability = weights[labels]
  return probability / probability.sum()
