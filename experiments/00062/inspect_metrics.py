"""Inspect the ``00062/`` scalars of a run directory (acceptance helper)."""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load(log_dir: Path) -> dict[str, list[tuple[int, float]]]:
  acc = EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
  acc.Reload()
  out: dict[str, list[tuple[int, float]]] = {}
  for tag in acc.Tags().get("scalars", []):
    out[tag] = [(e.step, float(e.value)) for e in acc.Scalars(tag)]
  return out


def main(argv: list[str]) -> int:
  log_dir = Path(argv[1])
  tags = load(log_dir)
  prefixed = {k: v for k, v in tags.items() if k.startswith("00062/")}
  if len(argv) > 2 and argv[2] == "--tags":
    for tag in sorted(prefixed):
      print(tag)
    return 0

  print(f"total scalar tags: {len(tags)}, 00062/* tags: {len(prefixed)}")
  missing = [
    "Reward/task_raw_mean",
    "Reward/smp_raw_mean",
    "Reward/additive_raw_mean",
    "Reward/env_reward_mean",
    "Task/target_distance_mean_m",
    "Motion/horizontal_speed_mean_mps",
    "Count/ended_episode",
    "Count/failure_episode",
    "Count/timeout_only_episode",
    "Count/complete_episode",
    "Count/complete_failure_episode",
    "Count/complete_timeout_only_episode",
    "Count/startup_incomplete_episode",
    "Sum/complete_episode_length_steps",
    "Sum/complete_failure_duration_s",
    "Sum/complete_task_return_raw",
    "Sum/complete_training_return",
    "Progress/completed_iterations",
    "Progress/training_samples",
    "Time/iteration_s",
    "Time/training_elapsed_s",
  ]
  for name in missing:
    if f"00062/{name}" not in prefixed:
      print(f"MISSING required tag: 00062/{name}")

  steps: defaultdict[str, set[int]] = defaultdict(set)
  for tag, values in prefixed.items():
    steps[tag] = {s for s, _ in values}
  all_steps = sorted({s for v in steps.values() for s in v})
  print(f"steps range: {all_steps[0]}..{all_steps[-1]} ({len(all_steps)} iterations)")

  def series(name: str) -> dict[int, float]:
    return dict(prefixed.get(f"00062/{name}", []))

  ended = series("Count/ended_episode")
  failure = series("Count/failure_episode")
  timeout = series("Count/timeout_only_episode")
  complete = series("Count/complete_episode")
  c_fail = series("Count/complete_failure_episode")
  c_timeout = series("Count/complete_timeout_only_episode")
  startup = series("Count/startup_incomplete_episode")
  samples = series("Progress/training_samples")
  iters = series("Progress/completed_iterations")

  bad = 0
  for step in sorted(ended):
    if failure.get(step, 0) + timeout.get(step, 0) != ended[step]:
      print(f"IDENTITY FAIL step {step}: failure+timeout != ended")
      bad += 1
    if complete.get(step, 0) + startup.get(step, 0) != ended[step]:
      print(f"IDENTITY FAIL step {step}: complete+startup != ended")
      bad += 1
    if c_fail.get(step, 0) + c_timeout.get(step, 0) != complete.get(step, 0):
      print(f"IDENTITY FAIL step {step}: complete_failure+timeout != complete")
      bad += 1
  print(f"count identity failures: {bad}")

  for step in sorted(ended)[:6]:
    print(
      f"it={step} ended={ended[step]:.0f} fail={failure.get(step, 0):.0f} "
      f"timeout={timeout.get(step, 0):.0f} complete={complete.get(step, 0):.0f} "
      f"startup={startup.get(step, 0):.0f} "
      f"task={series('Reward/task_raw_mean')[step]:.5f} "
      f"smp={series('Reward/smp_raw_mean')[step]:.5f} "
      f"additive={series('Reward/additive_raw_mean')[step]:.5f} "
      f"env={series('Reward/env_reward_mean')[step]:.5f} "
      f"dist={series('Task/target_distance_mean_m')[step]:.3f} "
      f"speed={series('Motion/horizontal_speed_mean_mps')[step]:.3f}"
    )
  last = sorted(ended)[-1]
  print(
    f"last it={last}: samples={samples[last]:.0f} iters={iters[last]:.0f} "
    f"ratio env/additive={series('Reward/env_reward_mean')[last] / max(series('Reward/additive_raw_mean')[last], 1e-12):.4f}"
  )
  opt_tags = sorted(
    t for t in prefixed if t.startswith(("00062/Episode/", "00062/Failure/"))
  )
  for tag in opt_tags:
    values = prefixed[tag]
    print(
      f"optional {tag}: n={len(values)} last={values[-1][1]:.4f} step={values[-1][0]}"
    )
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv))
