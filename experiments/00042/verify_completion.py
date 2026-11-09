#!/usr/bin/env python3
"""Stage D verification for experiment 00042 and COMPLETION.md generation.

Reads the artefacts written by ``run_00042.py`` (console log, run_info.json,
checkpoint index, TensorBoard events, checkpoints) and checks the completion
criteria of ``experiments/00042/EXECUTION.md`` §10 D:

* python exit code 0 and the run reported exactly the requested iterations,
* 10000 iterations / 245_760_000 training samples,
* every fixed TensorBoard scalar has 10000 unique, self-consistent steps,
* count identities hold and the conditional means are recomputable from the
  corresponding sums/counts,
* the final checkpoint loads, matches the initial state schema, differs from ``initial_untrained.pt``
  (the policy really learned) and the prior file is unchanged,
* no forbidden per-iteration CSV/JSONL/trajectory/GSI/video/W&B artefacts.

Usage:
  verify_completion.py --log-dir logs/<run> --ckpt-dir ckpt/<run> \
      --out COMPLETION.md [--allow-incomplete]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

NS = "00042"

FIXED_TAGS = [
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

OPTIONAL_TAGS = [
  "Episode/complete_task_return_raw_mean",
  "Episode/complete_training_return_mean",
  "Episode/complete_length_steps_mean",
  "Episode/complete_survival_s_mean",
  "Failure/complete_rate",
  "Failure/complete_duration_s_mean",
  "Failure/all_ended_rate",
]

FORBIDDEN_SUFFIXES = (
  ".csv",
  ".jsonl",
  ".mp4",
  ".avi",
  ".png",
  ".jpg",
  ".jpeg",
  ".gif",
  ".onnx",
  ".npz",
  ".npy",
)
PRIOR_SHA256 = "74787104ff124191136eeed2268f79448da3ac2e30e710d42aebfc5a60e58cb3"


class Report:
  def __init__(self) -> None:
    self.rows: list[tuple[str, bool, str]] = []

  def check(self, name: str, ok: bool, detail: str = "") -> bool:
    self.rows.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    return bool(ok)

  @property
  def failed(self) -> int:
    return sum(1 for _, ok, _ in self.rows if not ok)

  def markdown(self) -> str:
    lines = ["| 检查项 | 结果 | 说明 |", "| --- | --- | --- |"]
    for name, ok, detail in self.rows:
      lines.append(f"| {name} | {'PASS' if ok else 'FAIL'} | {detail} |")
    return "\n".join(lines)


def sha256_file(path: Path) -> str:
  h = hashlib.sha256()
  with path.open("rb") as fh:
    for chunk in iter(lambda: fh.read(1 << 20), b""):
      h.update(chunk)
  return h.hexdigest()


def load_scalars(log_dir: Path) -> dict[str, dict[int, float]]:
  from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

  ea = EventAccumulator(str(log_dir), size_guidance={"scalars": 0, "tensors": 0})
  ea.Reload()
  out: dict[str, dict[int, float]] = {}
  for tag in ea.Tags().get("scalars", []):
    series: dict[int, float] = {}
    for ev in ea.Scalars(tag):
      value = ev.value
      if value is not None and not math.isfinite(value):
        value = float("nan")
      series[int(ev.step)] = value
    out[tag] = series
  return out


def last_iteration(console: Path) -> int:
  text = console.read_text(errors="replace") if console.exists() else ""
  best = -1
  marker = "Learning iteration "
  idx = text.rfind(marker)
  if idx >= 0:
    tail = text[idx + len(marker) :]
    num = ""
    for ch in tail:
      if ch.isdigit():
        num += ch
      else:
        break
    if num:
      best = int(num)
  return best


def check_artifacts(root: Path) -> list[str]:
  bad: list[str] = []
  for path in sorted(root.rglob("*")):
    if not path.is_file():
      continue
    if path.suffix.lower() in FORBIDDEN_SUFFIXES:
      bad.append(str(path))
    if "wandb" in path.parts:
      bad.append(str(path))
  return bad


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--log-dir", required=True)
  parser.add_argument("--ckpt-dir", required=True)
  parser.add_argument("--iterations", type=int, default=10000)
  parser.add_argument("--num-envs", type=int, default=1024)
  parser.add_argument("--steps-per-env", type=int, default=24)
  parser.add_argument("--out", default="")
  parser.add_argument("--allow-incomplete", action="store_true")
  args = parser.parse_args(argv)

  log_dir = Path(args.log_dir).resolve()
  ckpt_dir = Path(args.ckpt_dir).resolve()
  out = Path(args.out).resolve() if args.out else log_dir / "COMPLETION.md"
  expected_samples = args.iterations * args.num_envs * args.steps_per_env
  report = Report()

  # ------------------------------------------------------------ process
  exit_file = log_dir / "exit_code.txt"
  exit_code = None
  if exit_file.exists():
    for token in exit_file.read_text().split():
      if token.startswith("python_exit_code="):
        exit_code = int(token.split("=", 1)[1])
  report.check("python exit code == 0", exit_code == 0, f"exit_code={exit_code}")

  tee_ok = exit_file.exists() and "tee_exit_code=0" in exit_file.read_text()
  report.check("console tee exit code == 0", tee_ok, str(exit_file))

  console = log_dir / "console.log"
  finished = console.read_text(errors="replace") if console.exists() else ""
  last_it = last_iteration(console)
  finished_ok = f"iterations={args.iterations} samples={expected_samples}" in finished
  report.check(
    f"run reported {args.iterations} iterations / {expected_samples} samples",
    finished_ok,
    "console tail matched" if finished_ok else f"last iteration={last_it}",
  )

  info = json.loads((log_dir / "run_info.json").read_text())
  physics = info.get("gsi_physics", {})
  totals = info.get("gsi_physics_totals", {})
  spec = json.loads(
    Path(__file__).with_name("EXPERIMENT.json").read_text(encoding="utf-8")
  )
  expected_updates = 1 + (args.iterations * args.steps_per_env) // 2400
  expected_insertions = 4096 + (expected_updates - 1) * 1024
  import sys

  sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
  from smp.rl.gsi_clusters import sha256, validate_artifacts

  cluster = info.get("gsi_cluster_totals", {})
  cluster_spec = spec["cluster_sampling"]
  manifest = info.get("gsi_cluster_manifest", {})
  report.check(
    "frozen cluster configuration and correct dimension",
    info.get("cluster_sampling") == cluster_spec
    and manifest.get("output_dims") == cluster_spec["output_dims"]
    and manifest.get("cap") == cluster.get("cap") == cluster_spec["cap"] == 1.5
    and info.get("gsi_cluster_frozen_check") is True,
  )
  report.check(
    "cluster updates exactly match actual GSI insertions",
    cluster.get("classification_updates") == expected_updates
    and cluster.get("classified_windows") == expected_insertions,
    str(cluster),
  )
  counts, weights, mass = [
    cluster.get(key, []) for key in ("counts", "weights", "mass")
  ]
  valid_distribution = (
    len(counts) == len(weights) == len(mass) == 16 and sum(counts) == 4096
  )
  if valid_distribution:
    maximum = max(counts)
    expected_weights = [
      min(cluster_spec["cap"], math.sqrt(maximum / n)) if n else 0 for n in counts
    ]
    denominator = sum(n * w for n, w in zip(counts, expected_weights, strict=True))
    valid_distribution &= all(
      math.isclose(w, expected, abs_tol=1e-12, rel_tol=1e-12)
      and math.isclose(p, n * expected / denominator, abs_tol=1e-12, rel_tol=1e-12)
      for n, w, p, expected in zip(counts, weights, mass, expected_weights, strict=True)
    )
  report.check(
    "final probabilities use current pool counts and per-item cap=1.5",
    valid_distribution,
  )
  draws = cluster.get("reset_draw_counts", [])
  report.check(
    "weighted reset path actually used with complete aggregate counters",
    cluster.get("reset_draws", 0) > 0
    and cluster.get("reset_calls", 0) > 0
    and len(draws) == 16
    and sum(draws) == cluster.get("reset_draws"),
  )
  try:
    current = validate_artifacts(
      cluster_spec["code_dir"],
      cluster_spec["module_dir"],
      cluster_spec["output_dims"],
      cap=cluster_spec["cap"],
    )
    unchanged = all(current[key] == manifest.get(key) for key in current)
    report.check("frozen external code/module artifacts unchanged", unchanged)
  except Exception as exc:
    report.check("frozen external code/module artifacts unchanged", False, repr(exc))
  baseline_path = Path(info.get("baseline_info", "missing"))
  baseline_ok = baseline_path.is_file() and sha256(baseline_path) == info.get(
    "baseline_info_sha256"
  )
  if baseline_ok:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_ok &= (
      baseline["experiment"] == "00031" and baseline["completed_iterations"] == 10000
    )
    baseline_ok &= info.get("initial_models_sha256") == baseline.get(
      "initial_models_sha256"
    )
    baseline_ok &= info.get("gsi_model_manifest") == baseline.get("gsi_model_manifest")
    baseline_ok &= info.get("gsi_physics_config") == baseline.get("gsi_physics_config")
  report.check(
    "same untrained models and physical model as completed 00031", baseline_ok
  )
  report.check(
    "full ten-frame physics configuration",
    physics.get("ground_limit_mm") == 19.72
    and physics.get("ground_frames")
    == physics.get("joint_frames")
    == physics.get("self_frames")
    == 10
    and physics.get("self_geom_pairs") == 469
    and physics.get("self_tolerance_mm") == 0
    and physics.get("joint_tolerance_rad") == 1e-5
    and physics.get("repair_small_joints")
    == spec["physical_filter"]["repair_small_joints"],
  )
  report.check(
    "fixed GSI quotas and refresh count",
    totals.get("updates") == expected_updates
    and totals.get("inserted") == expected_insertions,
    str(totals),
  )
  report.check(
    "qualified insertion accounting",
    totals.get("qualified", -1) == totals.get("inserted", 0) + totals.get("surplus", 0)
    and totals.get("generated", -1) >= totals.get("qualified", 0),
  )
  report.check(
    "00031 joint correction remains enabled",
    totals.get("repair_inserted", 0) > 0
    if spec["physical_filter"]["repair_small_joints"]
    else totals.get("repair_attempted", -1) == 0,
  )
  common = Path(info.get("initial_checkpoint", "missing"))
  report.check(
    "common initial source preserved",
    common.is_file() and sha256_file(common) == info.get("initial_checkpoint_sha256"),
  )
  report.check(
    "run_info completed_iterations",
    info.get("completed_iterations") == args.iterations,
    f"{info.get('completed_iterations')}",
  )
  report.check(
    "run_info total_samples",
    info.get("total_samples") == expected_samples,
    f"{info.get('total_samples')}",
  )
  report.check(
    "run_info fixed parameters (1024 envs / 24 steps / seed 2027)",
    info.get("num_envs") == args.num_envs
    and info.get("num_steps_per_env") == args.steps_per_env
    and info.get("seed") == 2027,
    f"{info.get('num_envs')}/{info.get('num_steps_per_env')}/{info.get('seed')}",
  )
  report.check(
    "config deviations recorded",
    info.get("decimation") == 4
    and info.get("episode_length_s") == 20.0
    and info.get("scale_rewards_by_dt") is True
    and info.get("timestep") == 0.005,
    f"dec={info.get('decimation')} horizon={info.get('episode_length_s')} "
    f"dt={info.get('timestep')} scaled={info.get('scale_rewards_by_dt')}",
  )
  report.check(
    "metrics layer enabled",
    info.get("metrics_enabled") is True,
    f"{info.get('metrics_enabled')}",
  )

  prior = Path("datasets/pretrain_ckpt/pretrained_lafan_run.pt")
  if prior.exists():
    report.check(
      "prior file unchanged (sha256)",
      sha256_file(prior) == PRIOR_SHA256,
      sha256_file(prior)[:16],
    )

  lock = Path("uv.lock")
  expected_lock = "ce041c7774b1d4abf42e868fa3b4af2973954cec61d35c3b4e58a7e27464d9ce"
  report.check(
    "uv.lock unchanged", lock.is_file() and sha256_file(lock) == expected_lock
  )
  report.check("prior exists", prior.is_file(), str(prior))

  # ------------------------------------------------------- checkpoints
  index_path = ckpt_dir / "index.json"
  index = json.loads(index_path.read_text()) if index_path.exists() else []
  report.check("checkpoint index present", bool(index), f"{len(index)} entries")
  expected_iters = set(range(0, args.iterations, 500))
  present = {Path(e["file"]).name for e in index}
  missing = [
    f"model_{it}.pt" for it in sorted(expected_iters) if f"model_{it}.pt" not in present
  ]
  final_name = f"model_{args.iterations - 1}.pt"
  if final_name not in present:
    missing.append(final_name)
  if "initial_untrained.pt" not in present:
    missing.append("initial_untrained.pt")
  report.check(
    "interval + final + initial checkpoints present",
    not missing,
    f"missing={missing[:5]}" if missing else f"{len(present)} files",
  )

  bad_hash = []
  for entry in index:
    path = ckpt_dir / entry["file"]
    if not path.exists() or sha256_file(path) != entry["sha256"]:
      bad_hash.append(entry["file"])
  report.check(
    "every indexed checkpoint matches its sha256",
    not bad_hash,
    f"{len(index)} verified" if not bad_hash else f"mismatch={bad_hash[:3]}",
  )
  mapping_ok = all(
    e["native_iter"]
    == (
      0
      if e["file"] == "initial_untrained.pt"
      else int(e["file"].split("_")[1].split(".")[0])
    )
    for e in index
  )
  mapping_ok = mapping_ok and all(
    e["completed_iterations"] == e["native_iter"] + 1
    for e in index
    if e["file"] != "initial_untrained.pt"
  )
  report.check(
    "native iter -> completed iterations mapping (i+1)",
    mapping_ok,
    f"{len(index)} entries",
  )

  import torch

  final_ckpt = ckpt_dir / final_name
  initial_ckpt = ckpt_dir / "initial_untrained.pt"
  learned = None
  loaded = False
  ckpt_notes = ""
  if final_ckpt.exists() and initial_ckpt.exists():
    fin = torch.load(final_ckpt, map_location="cpu", weights_only=False)
    ini = torch.load(initial_ckpt, map_location="cpu", weights_only=False)
    from importlib.util import module_from_spec, spec_from_file_location

    run_spec = spec_from_file_location(
      "experiment_runner", Path(__file__).with_name(f"run_{NS}.py")
    )
    run_mod = module_from_spec(run_spec)
    run_spec.loader.exec_module(run_mod)
    common_state = torch.load(common, map_location="cpu", weights_only=False)
    report.check(
      "initial model tensors exactly match common untrained source",
      run_mod.initial_models_digest(ini)
      == run_mod.initial_models_digest(common_state)
      == info.get("initial_models_sha256"),
    )
    report.check("initial optimizer is fresh", not ini["optimizer_state_dict"]["state"])
    keys_ok = all(k in fin for k in ("actor_state_dict", "critic_state_dict", "iter"))
    loaded = bool(keys_ok)
    report.check(
      "final checkpoint loads with actor/critic/iter",
      loaded,
      f"keys={sorted(k for k in fin)[:6]}",
    )
    for component in ("actor_state_dict", "critic_state_dict"):
      final_state = fin.get(component, {})
      initial_state = ini.get(component, {})
      schema_ok = bool(final_state) and final_state.keys() == initial_state.keys()
      schema_ok = schema_ok and all(
        final_state[k].shape == initial_state[k].shape for k in final_state
      )
      finite_ok = bool(final_state) and all(
        bool(torch.isfinite(v).all()) for v in final_state.values()
      )
      report.check(
        f"{component}: same schema as initial and all tensors finite",
        schema_ok and finite_ok,
        f"{len(final_state)} tensors",
      )
    num_params = sum(v.numel() for v in fin["actor_state_dict"].values())
    # Only the MLP parameters are policy weights; obs_normalizer.count is the
    # transition counter (== total samples) and _mean/_var are running stats.
    deltas = [
      float((tensor.float() - ini["actor_state_dict"][key].float()).abs().max())
      for key, tensor in fin["actor_state_dict"].items()
      if key.startswith("mlp.") and key in ini["actor_state_dict"]
    ]
    max_delta = max(deltas) if deltas else 0.0
    learned = max_delta > 1e-4
    report.check(
      "policy actually learned (actor MLP weights changed vs untrained)",
      learned,
      f"max|d|={max_delta:.4e} actor_params={num_params}",
    )
    norm_count = fin["actor_state_dict"].get("obs_normalizer.count")
    if norm_count is not None:
      report.check(
        "obs normalizer count == training samples",
        int(norm_count.item()) == expected_samples,
        f"{int(norm_count.item())}",
      )
    std_param = fin["actor_state_dict"].get("distribution.std_param")
    if std_param is not None:
      report.check(
        "action std stays fixed at 0.30 (learn_std=False)",
        abs(float(std_param.float().max()) - 0.30) < 1e-6
        and abs(float(std_param.float().min()) - 0.30) < 1e-6,
        f"{float(std_param.float().max()):.4f}",
      )
    ckpt_notes = f"final={final_name} actor_params={num_params} max|d|={max_delta:.4e}"
    del fin, ini
  else:
    report.check("final checkpoint present", False, str(final_ckpt))

  # --------------------------------------------------------- tensorboard
  try:
    scalars = load_scalars(log_dir)
  except Exception as exc:  # pragma: no cover - only on broken event files
    report.check("TensorBoard events readable", False, repr(exc))
    scalars = {}
  tags = {t.replace(f"{NS}/", "", 1) for t in scalars}
  missing_tags = [t for t in FIXED_TAGS if t not in tags]
  report.check(
    "all 21 fixed TensorBoard scalars present",
    not missing_tags,
    f"{21 - len(missing_tags)}/21" if missing_tags else "21/21",
  )
  optional_present = [t for t in OPTIONAL_TAGS if t in tags]
  print(f"[info] optional scalars present: {len(optional_present)}/7")
  gsi_tags_ok = True
  for name in (
    "updates",
    "generated",
    "qualified",
    "inserted",
    "surplus",
    "repair_attempted",
    "repair_inserted",
  ):
    values = scalars.get(f"{NS}/GSI/{name}", {})
    expected_steps = list(range(1, args.iterations + 1))
    ordered = [values[i] for i in expected_steps if i in values]
    gsi_tags_ok &= sorted(values) == expected_steps and all(
      math.isfinite(v) for v in ordered
    )
    gsi_tags_ok &= all(a <= b for a, b in zip(ordered, ordered[1:], strict=False))
    gsi_tags_ok &= values.get(args.iterations) == totals.get(name)
  report.check(
    "GSI TensorBoard cumulative counters complete and match metadata", gsi_tags_ok
  )
  cluster_tags_ok = True
  for name in (
    "classification_updates",
    "classified_windows",
    "reset_calls",
    "reset_draws",
    "weight_cap",
  ):
    values = scalars.get(f"{NS}/Cluster/{name}", {})
    expected_final = cluster_spec["cap"] if name == "weight_cap" else cluster.get(name)
    cluster_tags_ok &= sorted(values) == list(range(1, args.iterations + 1))
    cluster_tags_ok &= all(math.isfinite(v) for v in values.values())
    cluster_tags_ok &= values.get(args.iterations) == expected_final
    if name == "weight_cap":
      cluster_tags_ok &= all(v == 1.5 for v in values.values())
  for k in range(16):
    for name, expected in [
      (f"count_{k:02d}", counts[k] if len(counts) == 16 else -1),
      (f"probability_mass_{k:02d}", mass[k] if len(mass) == 16 else -1),
    ]:
      values = scalars.get(f"{NS}/Cluster/{name}", {})
      cluster_tags_ok &= sorted(values) == list(range(1, args.iterations + 1))
      cluster_tags_ok &= math.isclose(
        values.get(args.iterations, -999), expected, abs_tol=1e-6, rel_tol=1e-6
      )
  report.check(
    "cluster TensorBoard counters and distribution complete", cluster_tags_ok
  )

  steps = {
    t: sorted(scalars[f"{NS}/{t}"]) for t in FIXED_TAGS if f"{NS}/{t}" in scalars
  }
  short = {
    t: (len(v), v[:1] + v[-1:])
    for t, v in steps.items()
    if v != list(range(1, args.iterations + 1))
  }
  report.check(
    f"each fixed scalar has {args.iterations} unique steps",
    not short,
    f"short={list(short.items())[:2]}" if short else f"{len(steps)} tags",
  )
  progress = scalars.get(f"{NS}/Progress/completed_iterations", {})
  report.check(
    "progress axis is 1..10000 with step == value",
    bool(progress)
    and min(progress) == 1
    and max(progress) == args.iterations
    and all(int(s) == int(v) for s, v in progress.items()),
    f"{min(progress)}..{max(progress)}" if progress else "missing",
  )
  samples = scalars.get(f"{NS}/Progress/training_samples", {})
  report.check(
    "training samples final == 245760000",
    bool(samples) and int(samples[max(samples)]) == expected_samples,
    f"{int(samples[max(samples)])}" if samples else "missing",
  )

  ended = scalars.get(f"{NS}/Count/ended_episode", {})
  failure = scalars.get(f"{NS}/Count/failure_episode", {})
  timeout = scalars.get(f"{NS}/Count/timeout_only_episode", {})
  complete = scalars.get(f"{NS}/Count/complete_episode", {})
  startup = scalars.get(f"{NS}/Count/startup_incomplete_episode", {})
  cf = scalars.get(f"{NS}/Count/complete_failure_episode", {})
  ct = scalars.get(f"{NS}/Count/complete_timeout_only_episode", {})
  common = set(ended) & set(failure) & set(timeout) & set(complete) & set(startup)
  bad_id = 0
  bad_cf = 0
  for s in common:
    if int(failure[s] + timeout[s]) != int(ended[s]):
      bad_id += 1
    if int(complete[s] + startup[s]) != int(ended[s]):
      bad_id += 1
    if int(cf[s] + ct[s]) != int(complete[s]):
      bad_cf += 1
  report.check(
    "count identities hold on every iteration",
    bad_id == 0 and bad_cf == 0 and len(common) == args.iterations,
    f"violations={bad_id + bad_cf} iters={len(common)}",
  )

  s_len = scalars.get(f"{NS}/Sum/complete_episode_length_steps", {})
  s_fail = scalars.get(f"{NS}/Sum/complete_failure_duration_s", {})
  mean_fail = scalars.get(f"{NS}/Failure/complete_duration_s_mean", {})
  bad_mean = 0
  for s, value in mean_fail.items():
    cnt = cf.get(s, 0)
    if cnt > 0 and abs(s_fail[s] / cnt - value) > 1e-4:
      bad_mean += 1
  report.check(
    "failure duration mean recomputable from sum/count",
    bad_mean == 0 and bool(mean_fail),
    f"checked={len(mean_fail)} bad={bad_mean}",
  )
  mean_len = scalars.get(f"{NS}/Episode/complete_length_steps_mean", {})
  bad_len = 0
  for s, value in mean_len.items():
    cnt = complete.get(s, 0)
    if cnt > 0 and abs(s_len[s] / cnt - value) > 1e-3:
      bad_len += 1
  report.check(
    "episode length mean recomputable from sum/count",
    bad_len == 0 and bool(mean_len),
    f"checked={len(mean_len)} bad={bad_len}",
  )

  env_rew = scalars.get(f"{NS}/Reward/env_reward_mean", {})
  additive = scalars.get(f"{NS}/Reward/additive_raw_mean", {})
  task_rew = scalars.get(f"{NS}/Reward/task_raw_mean", {})
  smp_rew = scalars.get(f"{NS}/Reward/smp_raw_mean", {})
  reward_steps = set(env_rew) & set(additive) & set(task_rew) & set(smp_rew)
  bad_dt = [
    s
    for s in reward_steps
    if not math.isclose(env_rew[s], 0.02 * additive[s], rel_tol=2e-5, abs_tol=1e-7)
  ]
  bad_add = [
    s
    for s in reward_steps
    if not math.isclose(
      additive[s], 0.5 * task_rew[s] + 0.5 * smp_rew[s], rel_tol=2e-5, abs_tol=1e-7
    )
  ]
  report.check(
    "env reward == 0.02 x r_additive on every iteration",
    len(reward_steps) == args.iterations and not bad_dt,
    f"checked={len(reward_steps)} bad={len(bad_dt)}",
  )
  report.check(
    "r_additive == 0.5*r_task + 0.5*r_smp on every iteration",
    len(reward_steps) == args.iterations and not bad_add,
    f"checked={len(reward_steps)} bad={len(bad_add)}",
  )
  r = info.get("reward", {})
  report.check(
    "original SMP outer and Location internal weights recorded",
    r.get("task_reward_weight") == r.get("smp_reward_weight") == 0.5
    and r.get("position_weight") == 1.0
    and r.get("velocity_weight") == r.get("facing_weight") == 0.0
    and r.get("position_kernel") == "exp(-0.25 * horizontal_distance_m)"
    and r.get("smp_ws") == 4
    and r.get("reward_manager_weight") == 1.0,
    str(r),
  )

  finite_bad = 0
  for key in (
    f"{NS}/Reward/task_raw_mean",
    f"{NS}/Reward/smp_raw_mean",
    f"{NS}/Reward/additive_raw_mean",
    f"{NS}/Reward/env_reward_mean",
    f"{NS}/Time/iteration_s",
  ):
    for value in scalars.get(key, {}).values():
      if value is None or not math.isfinite(value):
        finite_bad += 1
  report.check(
    "key reward/time scalars finite",
    finite_bad == 0,
    f"non-finite={finite_bad}",
  )

  iter_times = sorted(scalars.get(f"{NS}/Time/iteration_s", {}).values())
  if iter_times:
    median = iter_times[len(iter_times) // 2]
    print(
      f"[info] iteration time median={median:.3f}s min={iter_times[0]:.3f}s "
      f"max={iter_times[-1]:.3f}s"
    )

  # ------------------------------------------------------------ artefacts
  bad_files = check_artifacts(log_dir) + check_artifacts(ckpt_dir)
  report.check(
    "no forbidden per-iteration/video/GSI artefacts",
    not bad_files,
    f"{len(bad_files)} files" if bad_files else "clean",
  )

  elapsed = sorted(scalars.get(f"{NS}/Time/training_elapsed_s", {}))
  elapsed_h = (
    scalars[f"{NS}/Time/training_elapsed_s"][elapsed[-1]] / 3600.0
    if elapsed
    else float("nan")
  )
  print(f"[info] training elapsed ~{elapsed_h:.2f} h")

  code = 0 if report.failed == 0 else 1
  if args.allow_incomplete and code != 0:
    code = 0

  if args.out:
    final_sha = sha256_file(final_ckpt) if final_ckpt.exists() else "n/a"

    def window_mean(tag: str, lo: int, hi: int) -> float:
      series = scalars.get(f"{NS}/{tag}", {})
      vals = [series[s] for s in series if lo <= s <= hi]
      return sum(vals) / len(vals) if vals else float("nan")

    trend = [
      ("Reward/task_raw_mean", "任务奖励（原始 r_task）"),
      ("Reward/smp_raw_mean", "SMP 奖励（原始 r_smp）"),
      ("Reward/additive_raw_mean", "加权和奖励 r_additive"),
      ("Reward/env_reward_mean", "环境返回奖励（0.02×r_additive）"),
      ("Task/target_distance_mean_m", "目标距离均值 m"),
      ("Episode/complete_length_steps_mean", "完整回合长度（步）"),
      ("Failure/complete_rate", "完整回合失败率"),
    ]
    trend_rows = "\n".join(
      f"| {label} | {window_mean(tag, 1, 100):.4f} | {window_mean(tag, args.iterations - 99, args.iterations):.4f} |"
      for tag, label in trend
    )

    md = f"""# 实验 00042 完成摘要（G1 Location additive reward）

运行标识：`{info.get("run_id")}`
生成时间：`{info.get("finished_at", "n/a")}`（Asia/Shanghai）

## 结果

| 项目 | 数值 |
| --- | --- |
| Python 退出码 | `{exit_code}` |
| 完成迭代 | `{info.get("completed_iterations")}` / {args.iterations} |
| 训练样本 | `{info.get("total_samples")}` / {expected_samples} |
| 训练耗时 | {elapsed_h:.2f} h（`Time/training_elapsed_s`） |
| 环境数 / 每轮步数 / 种子 | {args.num_envs} / {args.steps_per_env} / 2027 |
| TensorBoard 固定标量 | {21 - len(missing_tags)}/21 |
| TensorBoard 条件标量 | {len(optional_present)}/7 |
| 最终检查点 | `{final_name}` |
| 最终检查点 SHA256 | `{final_sha}` |
| {ckpt_notes} | |

## 训练趋势（首 100 轮 vs 末 100 轮，TensorBoard 原值窗口均值）

| 指标 | 首 100 轮 | 末 100 轮 |
| --- | --- | --- |
{trend_rows}

## 产出位置

- 日志 / TensorBoard：`{log_dir}`
- 检查点 / 索引：`{ckpt_dir}`
- 先验：`datasets/pretrain_ckpt/pretrained_lafan_run.pt`（sha256 `{info.get("prior_sha256")}`，未修改）
- 代码提交：`{info.get("git_commit")}`（分支 `{info.get("git_branch")}`）；启动时工作区状态见 `run_info.json` 的 `git_dirty`

## 验收明细

{report.markdown()}

## 结论

{"全部检查通过（" + str(len(report.rows)) + " 项）。" if report.failed == 0 else str(report.failed) + " 项检查未通过，见上表；本文件不伪造 PASS。"}

## 运行入口（以下训练命令会新建另一次运行）

```bash
cd /root/gpufree-data/smp-g1/repo && source experiments/00042/activate_00042.sh
experiments/00042/launch_00042.sh            # 正式训练（阶段 C）
experiments/00042/accept_00042.py --num-envs 64 --steps 2400 --baseline-info "$SMP_00031_RUN_INFO" --out logs/_accept/b_stage
experiments/00042/verify_completion.py --log-dir logs/<run> --ckpt-dir ckpt/<run> --out COMPLETION.md
```

TensorBoard：`127.0.0.1:6010`（`experiments/00042/launch_tensorboard.sh 6010`）。
"""
    out.write_text(md, encoding="utf-8")
    print(f"[info] wrote {out}")

  print(f"{len(report.rows) - report.failed}/{len(report.rows)} checks passed")
  return code


if __name__ == "__main__":
  raise SystemExit(main())
