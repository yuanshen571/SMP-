"""Frozen inference/weighting interface. No PPO or simulator integration here."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from experiments.ts2vec_g1.common import (
  nearest_centers,
  normalize,
  official,
  read_json,
  sampling_probabilities,
  sha256,
)


class G1ClusterModule:
  def __init__(self, directory, device="cpu"):
    import torch

    self.directory = Path(directory)
    self.config = read_json(self.directory / "bundle.json")
    for name, digest in self.config["files"].items():
      if sha256(self.directory / name) != digest:
        raise ValueError(f"module artifact changed: {name}")
    with np.load(self.directory / "normalizer.npz", allow_pickle=False) as p:
      self.normalizer = {key: p[key].copy() for key in p.files}
    self.centers = np.load(self.directory / "centers.npy", allow_pickle=False)
    # Upstream initializes on CPU before moving weights. Loading a frozen module
    # must not disturb the caller's state-generation/sampling random stream.
    with torch.random.fork_rng(devices=[]):
      self.encoder = official()(
        input_dims=59,
        output_dims=self.config["output_dims"],
        hidden_dims=self.config["hidden_dims"],
        depth=self.config["depth"],
        device=device,
        batch_size=self.config["encode_batch_size"],
      )
    self.encoder.load(str(self.directory / "encoder_final.pt"))
    self.encoder.net.eval()
    self.encoder.net.requires_grad_(False)
    if self.centers.shape != (16, self.config["output_dims"]):
      raise ValueError("invalid frozen centers")

  def encode(self, windows, *, batch_size=None):
    import torch

    if isinstance(windows, torch.Tensor):
      windows = windows.detach().cpu().numpy()
    z = normalize(windows, self.normalizer)
    if not len(z):
      return np.empty((0, self.config["output_dims"]), dtype=np.float32)
    # Even an unshuffled upstream DataLoader draws a CPU iterator base seed.
    with torch.random.fork_rng(devices=[]):
      return self.encoder.encode(
        z, mask="all_true", encoding_window="full_series", batch_size=batch_size
      )

  def predict(self, windows, *, batch_size=None):
    return nearest_centers(self.encode(windows, batch_size=batch_size), self.centers)

  def current_pool_probabilities(self, current_pool_labels):
    return sampling_probabilities(current_pool_labels, cap=2.0, clusters=16)
