"""CPU-only validation of the actual completed baseline before GSI generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from initialization import sha256, validate_startup_control


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--initial-checkpoint", type=Path, required=True)
  parser.add_argument("--control-info", type=Path, required=True)
  parser.add_argument("--out", type=Path, required=True)
  args = parser.parse_args(argv)
  expected = json.loads(
    Path(__file__).with_name("EXPERIMENT.json").read_text(encoding="utf-8")
  )["control"]
  record = validate_startup_control(
    args.initial_checkpoint, args.control_info, expected
  )
  result = dict(
    experiment=record["experiment"],
    seed=record["seed"],
    initial_checkpoint=str(args.initial_checkpoint.resolve()),
    control_info=str(args.control_info.resolve()),
    initial_checkpoint_sha256=sha256(args.initial_checkpoint),
    initial_models_sha256=record["initial_models_sha256"],
    completed_iterations=record["completed_iterations"],
    static_models_pin_present="initial_models_sha256" in expected,
    static_file_pin_present="initial_checkpoint_sha256" in expected,
  )
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(
    json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
  )
  print("CONTROL_STARTUP_CHECK=PASS (CPU, no simulation or GSI generation)")


if __name__ == "__main__":
  main()
