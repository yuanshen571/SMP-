"""Require a completed, validated 00095 formal run before any 00097 training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from archive_format import atomic_json
from paired_replay import source_binding
from verify_source_archive import verify


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--control-info", type=Path, required=True)
  p.add_argument("--out", type=Path, required=True)
  args = p.parse_args()
  import torch

  torch.set_num_threads(min(8, torch.get_num_threads()))
  binding = source_binding(args.control_info)
  result = verify(binding["archive_dir"], iterations=20000, run_info=args.control_info)
  assert result["status"] == "PASS"
  atomic_json(args.out, dict(status="PASS", binding=binding, archive=result))
  print("PAIRED_SOURCE_CHECK=PASS", json.dumps(binding))


if __name__ == "__main__":
  main()
