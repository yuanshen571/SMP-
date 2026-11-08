"""Frozen external TS2Vec labels and capped per-item GSI reset probabilities.

Only admitted, corrected float32 windows are classified. No model fitting,
policy observation changes, reference-state fallback or cluster quotas.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import random
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

OFFICIAL_COMMIT = "b0088e14a99706c05451316dc6db8d3da9351163"


def sha256(path):
  h = hashlib.sha256()
  with Path(path).open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
      h.update(block)
  return h.hexdigest()


def read_json(path):
  return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_artifacts(code_dir, module_dir, output_dims):
  """Read-only validation. A moved bundle needs no original full training data."""
  code_dir, module_dir = Path(code_dir).resolve(), Path(module_dir).resolve()
  if output_dims not in (64, 320):
    raise ValueError("expected a 64D or 320D G1 module")
  bundle = read_json(module_dir / "bundle.json")
  if (
    bundle.get("input_shape") != [10, 59]
    or bundle.get("output_dims") != output_dims
    or bundle.get("hidden_dims") != 64
    or bundle.get("depth") != 10
    or bundle.get("mask") != "all_true"
    or bundle.get("pooling") != "full_series_max"
    or bundle.get("source_ts2vec_commit") != OFFICIAL_COMMIT
  ):
    raise ValueError("incompatible G1 bundle schema/architecture/pooling")
  cfg = bundle["config"]
  expected_physics = dict(
    repair=True,
    joint_tolerance_rad=1e-5,
    repair_limit_deg=1.0,
    ground_limit_mm=19.72,
    control_dt=0.02,
  )
  if (
    cfg.get("physical_filter") != expected_physics
    or cfg.get("unique_windows") != 100000
    or cfg.get("input_shape") != [10, 59]
    or cfg.get("control_dt") != 0.02
    or cfg.get("prior_sha256")
    != "74787104ff124191136eeed2268f79448da3ac2e30e710d42aebfc5a60e58cb3"
  ):
    raise ValueError("module was not trained on the agreed G1 physical conditions")
  training = cfg["ts2vec"]
  if training["epochs"] != 20 or training["batch_size"] != 64:
    raise ValueError("unexpected TS2Vec training budget")
  files = {}
  for name in ("encoder_final.pt", "normalizer.npz", "centers.npy"):
    digest = sha256(module_dir / name)
    if digest != bundle["files"].get(name):
      raise ValueError(f"module SHA256 mismatch: {name}")
    files[name] = digest
  if files["normalizer.npz"] != bundle["normalizer_sha256"]:
    raise ValueError("normalizer does not match training bundle")
  if len(bundle.get("dataset_sha256", "")) != 64:
    raise ValueError("missing common training dataset fingerprint")
  centers = np.load(module_dir / "centers.npy", allow_pickle=False)
  if centers.shape != (16, output_dims) or not np.isfinite(centers).all():
    raise ValueError("invalid centers")
  if len(np.unique(centers, axis=0)) != 16:
    raise ValueError("duplicate centers")
  with np.load(module_dir / "normalizer.npz", allow_pickle=False) as normalizer:
    for name in ("mean", "std", "scale"):
      if normalizer[name].shape != (59,) or not np.isfinite(normalizer[name]).all():
        raise ValueError(f"invalid 59-channel normalizer {name}")
    if (normalizer["scale"] <= 0).any():
      raise ValueError("nonpositive normalization scale")
  state = torch.load(
    module_dir / "encoder_final.pt", map_location="cpu", weights_only=True
  )
  if int(state["n_averaged"]) != 31241:
    raise ValueError("expected final average after 31240 TS2Vec updates")
  if not all(bool(torch.isfinite(value).all()) for value in state.values()):
    raise ValueError("non-finite encoder weights")
  source = read_json(code_dir / "third_party/ts2vec/UPSTREAM.json")
  if source["commit"] != OFFICIAL_COMMIT:
    raise ValueError("external official TS2Vec revision differs")
  code_files = {}
  for name, digest in source["files"].items():
    relative = str(Path("third_party/ts2vec") / name)
    if sha256(code_dir / relative) != digest:
      raise ValueError(f"official TS2Vec source changed: {relative}")
    code_files[relative] = digest
  for name in ("common.py", "module.py", "config.json"):
    relative = f"experiments/ts2vec_g1/{name}"
    code_files[relative] = sha256(code_dir / relative)
  files["bundle.json"] = sha256(module_dir / "bundle.json")
  return dict(
    code_dir=str(code_dir),
    module_dir=str(module_dir),
    output_dims=output_dims,
    input_shape=[10, 59],
    clusters=16,
    cap=2.0,
    dataset_sha256=bundle["dataset_sha256"],
    normalizer_sha256=files["normalizer.npz"],
    files=files,
    code_files=code_files,
    official_commit=OFFICIAL_COMMIT,
  )


@contextmanager
def frozen_inference(device):
  """Reproduce offline inference flags; restore PPO/DDPM flags and RNG streams."""
  device = torch.device(device)
  devices = (
    [device.index if device.index is not None else torch.cuda.current_device()]
    if device.type == "cuda"
    else []
  )
  python_state, numpy_state = random.getstate(), np.random.get_state()
  precision = torch.get_float32_matmul_precision()
  flags = (
    torch.backends.cudnn.benchmark,
    torch.backends.cudnn.deterministic,
    torch.backends.cudnn.allow_tf32,
    torch.backends.cuda.matmul.allow_tf32,
  )
  try:
    with (
      torch.random.fork_rng(devices=devices),
      torch.no_grad(),
      torch.autocast(device_type=device.type, enabled=False),
    ):
      torch.backends.cudnn.benchmark = False
      torch.backends.cudnn.deterministic = True
      torch.backends.cudnn.allow_tf32 = False
      torch.backends.cuda.matmul.allow_tf32 = False
      yield
  finally:
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = flags[:2]
    torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = flags[2:]
    torch.set_float32_matmul_precision(precision)


def state_digest(network):
  h = hashlib.sha256()
  for name, tensor in sorted(network.state_dict().items()):
    value = tensor.detach().cpu().contiguous()
    h.update(f"{name}:{value.dtype}:{tuple(value.shape)}".encode())
    h.update(value.numpy().tobytes())
  return h.hexdigest()


def representation_digest(module):
  h = hashlib.sha256()
  arrays = {"centers": module.centers, **module.normalizer}
  for name, value in sorted(arrays.items()):
    array = np.ascontiguousarray(value)
    h.update(f"{name}:{array.dtype}:{array.shape}".encode())
    h.update(array.tobytes())
  return h.hexdigest()


def load_module(code_dir, module_dir, output_dims, device):
  manifest = validate_artifacts(code_dir, module_dir, output_dims)
  code = Path(manifest["code_dir"])
  sys.path.insert(0, str(code)) if str(code) not in sys.path else None
  importlib.invalidate_caches()
  module = importlib.import_module("experiments.ts2vec_g1.module")
  if Path(module.__file__).resolve() != code / "experiments/ts2vec_g1/module.py":
    raise RuntimeError(
      "another TS2Vec package is already imported; use a fresh process"
    )
  with frozen_inference(device):
    encoder = module.G1ClusterModule(module_dir, device=str(device))
    encoder.encoder.net.eval().requires_grad_(False)
    encoder.encoder._net.eval().requires_grad_(False)
  manifest["encoder_state_sha256"] = state_digest(encoder.encoder.net)
  manifest["representation_state_sha256"] = representation_digest(encoder)
  return encoder, manifest


def pool_distribution(labels, cap=2.0):
  """Per-state weights; cluster mass is n_k*w_k, not just w_k."""
  labels = np.asarray(labels)
  if labels.ndim != 1 or labels.dtype.kind not in "iu" or not len(labels):
    raise ValueError("need nonempty integer current-pool labels")
  if labels.min() < 0 or labels.max() >= 16 or cap != 2.0:
    raise ValueError("0004/00041 require 16 clusters and cap=2")
  counts = np.bincount(labels, minlength=16)
  weights = np.zeros(16, dtype=np.float64)
  nonempty = counts > 0
  weights[nonempty] = np.minimum(cap, np.sqrt(counts.max() / counts[nonempty]))
  probability = weights[labels]
  probability /= probability.sum()
  mass = np.bincount(labels, weights=probability, minlength=16)
  return dict(counts=counts, weights=weights, probability=probability, mass=mass)


def classify(env, windows):
  if (
    windows.dtype != torch.float32 or windows.ndim != 3 or windows.shape[1:] != (10, 59)
  ):
    raise ValueError("classify final float32 [N,10,59] windows only")
  x = windows.detach().cpu().numpy().copy()
  with frozen_inference(env.device):
    embedding = env._smp_cluster_module.encode(x)
  dimension = env._smp_cluster_manifest["output_dims"]
  if embedding.shape != (len(x), dimension) or not np.isfinite(embedding).all():
    raise ValueError("invalid TS2Vec embedding")
  centers = env._smp_cluster_module.centers.astype(np.float64)
  labels = np.empty(len(x), dtype=np.int64)
  for start in range(0, len(x), 512):
    difference = (
      embedding[start : start + 512, None, :].astype(np.float64) - centers[None, :, :]
    )
    labels[start : start + 512] = np.square(difference).sum(axis=2).argmin(axis=1)
  return labels


def _plan(env, labels):
  distribution = pool_distribution(labels)
  return dict(
    labels=torch.as_tensor(labels.copy(), device=env.device, dtype=torch.long),
    probability=torch.as_tensor(
      distribution["probability"], device=env.device, dtype=torch.float64
    ),
    counts=distribution["counts"].tolist(),
    weights=distribution["weights"].tolist(),
    mass=distribution["mass"].tolist(),
  )


def initialize(env, *, code_dir, module_dir, output_dims, cap):
  if cap != 2.0:
    raise ValueError("fixed sampling cap is 2")
  env._smp_cluster_module, env._smp_cluster_manifest = load_module(
    code_dir, module_dir, output_dims, env.device
  )
  env._smp_cluster_stats = dict(
    classification_updates=0, classified_windows=0, reset_calls=0, reset_draws=0
  )
  env._smp_cluster_draw_counts = torch.zeros(16, device=env.device, dtype=torch.long)
  labels = classify(env, env._smp_gsi_pool)
  commit(env, _plan(env, labels), len(labels))


def prepare_update(env, new_windows, head):
  """Prepare before any pool mutation, including replacement across its end."""
  labels = env._smp_cluster_labels.detach().cpu().numpy().copy()
  if len(new_windows) > len(labels) or not 0 <= head < len(labels):
    raise ValueError("invalid FIFO replacement")
  new_labels = classify(env, new_windows)
  indices = (np.arange(len(new_windows)) + head) % len(labels)
  labels[indices] = new_labels
  return _plan(env, labels)


def commit(env, plan, added):
  env._smp_cluster_labels = plan["labels"]
  env._smp_cluster_probability = plan["probability"]
  env._smp_cluster_counts = plan["counts"]
  env._smp_cluster_weights = plan["weights"]
  env._smp_cluster_mass = plan["mass"]
  env._smp_cluster_stats["classification_updates"] += 1
  env._smp_cluster_stats["classified_windows"] += added


def sample_indices(env, count):
  probability = env._smp_cluster_probability
  if probability.shape != (len(env._smp_gsi_pool),) or count <= 0:
    raise ValueError("invalid effective-pool probability/count")
  indices = torch.multinomial(probability, count, replacement=True)
  env._smp_cluster_stats["reset_calls"] += 1
  env._smp_cluster_stats["reset_draws"] += count
  env._smp_cluster_draw_counts += torch.bincount(
    env._smp_cluster_labels[indices], minlength=16
  )
  return indices


def summary(env):
  return dict(
    **env._smp_cluster_stats,
    cap=2.0,
    counts=list(env._smp_cluster_counts),
    weights=list(env._smp_cluster_weights),
    mass=list(env._smp_cluster_mass),
    reset_draw_counts=env._smp_cluster_draw_counts.detach().cpu().tolist(),
  )


def check_frozen(env):
  manifest = env._smp_cluster_manifest
  for name, digest in manifest["files"].items():
    if sha256(Path(manifest["module_dir"]) / name) != digest:
      raise AssertionError(f"frozen module artifact changed: {name}")
  for name, digest in manifest["code_files"].items():
    if sha256(Path(manifest["code_dir"]) / name) != digest:
      raise AssertionError(f"external module code changed during training: {name}")
  network = env._smp_cluster_module.encoder.net
  if network.training or any(p.requires_grad for p in network.parameters()):
    raise AssertionError("TS2Vec encoder must remain frozen in eval mode")
  if state_digest(network) != manifest["encoder_state_sha256"]:
    raise AssertionError("in-memory encoder changed")
  if (
    representation_digest(env._smp_cluster_module)
    != manifest["representation_state_sha256"]
  ):
    raise AssertionError("in-memory centers or normalization changed")
