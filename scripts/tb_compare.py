#!/usr/bin/env python3
"""Overlay TensorBoard across experiments by stripping per-run tag prefixes.

Each experiment writes its scalars under its own namespace (`0001/...`,
`0002/...`), so TensorBoard renders them as separate charts even when several
runs are selected. This daemon mirrors every run it finds into
``<out>/<run name>/`` with the leading numeric namespace removed, which makes
the tag names identical across runs and lets TensorBoard draw them on the same
chart.

Properties:

* read-only towards the experiment directories (never writes there);
* append-only event files, so TensorBoard reloads them like a live run;
* discovers new run directories on every poll, so later experiments show up
  without restarting anything;
* viewer-only artefacts: the mirror is disposable and is never part of an
  experiment's recorded output.

Usage:
  tb_compare.py --roots logs --out /tmp/tb_compare --interval 20
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import time
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

# `0001/Reward/task_raw_mean` -> `Reward/task_raw_mean`
NAMESPACE = re.compile(r"^\d{3,}/")


def clean_tag(tag: str) -> str:
  return NAMESPACE.sub("", tag, count=1)


def discover(roots: list[Path], exclude: tuple[str, ...]) -> dict[str, Path]:
  """Map run name -> directory for every directory holding event files."""
  found: dict[str, Path] = {}
  for root in roots:
    if not root.exists():
      continue
    for entry in sorted(root.iterdir()):
      if not entry.is_dir():
        continue
      name = entry.name
      if name.startswith("_") or any(token in name for token in exclude):
        continue
      if not any(entry.glob("events.out.tfevents.*")):
        continue
      found.setdefault(name, entry)
  return found


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--roots", required=True,
                      help="comma separated directories holding run subdirectories")
  parser.add_argument("--out", required=True, help="mirror directory (disposable)")
  parser.add_argument("--interval", type=float, default=20.0)
  parser.add_argument("--max-idle", type=float, default=1800.0,
                      help="stop after this many seconds without new scalars")
  parser.add_argument("--exclude", default="accept,smoke,setup,_tensorboard",
                      help="comma separated substrings of run names to skip")
  args = parser.parse_args()

  roots = [Path(p).expanduser().resolve() for p in args.roots.split(",") if p.strip()]
  exclude = tuple(x.strip() for x in args.exclude.split(",") if x.strip())
  out_root = Path(args.out).expanduser().resolve()
  out_root.mkdir(parents=True, exist_ok=True)

  accs: dict[str, EventAccumulator] = {}
  writers: dict[str, SummaryWriter] = {}
  seen: dict[str, dict[str, int]] = {}
  labels: dict[str, Path] = {}
  last_progress = time.time()

  def ensure(name: str, src: Path) -> None:
    if name in writers:
      return
    # `0002_location_seed2027_...` -> `0002`, so the run list reads like the
    # experiment ids; a second run of the same id keeps its full directory name.
    match = re.match(r"^(\d{3,4})(?:_|$)", name)
    label = match.group(1) if match else name
    if label in labels and labels[label] != src:
      label = name
    labels[label] = src
    accs[name] = EventAccumulator(str(src), size_guidance={"scalars": 0})
    writers[name] = SummaryWriter(log_dir=str(out_root / label))
    seen[name] = {}
    print(f"[+] mirroring {name} as {label} from {src}", flush=True)

  try:
    while True:
      progressed = False
      for name, src in discover(roots, exclude).items():
        ensure(name, src)
        acc = accs[name]
        try:
          acc.Reload()
        except Exception as exc:  # a live run may be mid-write
          print(f"[warn] {name}: {exc}", flush=True)
          continue
        for tag in acc.Tags()["scalars"]:
          last = seen[name].get(tag, -1)
          new = [e for e in acc.Scalars(tag) if e.step > last]
          if not new:
            continue
          clean = clean_tag(tag)
          for event in new:
            writers[name].add_scalar(clean, event.value, event.step,
                                     walltime=event.wall_time)
          seen[name][tag] = new[-1].step
          progressed = True
      for writer in writers.values():
        writer.flush()

      now = time.time()
      if progressed:
        last_progress = now
        total = sum(len(v) for v in seen.values())
        print(f"[{time.strftime('%H:%M:%S')}] runs={len(writers)} tags={total}",
              flush=True)
      elif now - last_progress > args.max_idle:
        print("[info] sources idle; stopping mirror", flush=True)
        break
      time.sleep(args.interval)
  except KeyboardInterrupt:
    pass
  finally:
    for writer in writers.values():
      writer.close()
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
