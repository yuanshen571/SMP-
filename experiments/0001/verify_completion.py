#!/usr/bin/env python3
"""Stage D verification for experiment 0001 and COMPLETION.md generation.

Reads the artefacts written by ``run_0001.py`` (console log, run_info.json,
checkpoint index, TensorBoard events, checkpoints) and checks the completion
criteria of ``experiments/0001/EXECUTION.md`` §10 D:

* python exit code 0 and the run reported exactly the requested iterations,
* 10000 iterations / 245_760_000 training samples,
* every fixed TensorBoard scalar has 10000 unique, self-consistent steps,
* count identities hold and the conditional means are recomputable from the
  corresponding sums/counts,
* the final checkpoint loads strictly, differs from ``initial_untrained.pt``
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
import sys
from pathlib import Path

NS = "0001"

FIXED_TAGS = [
  "Reward/task_raw_mean",
  "Reward/smp_raw_mean",
  "Reward/product_raw_mean",
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
  report.check(
    "python exit code == 0", exit_code == 0, f"exit_code={exit_code}"
  )

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
    e["native_iter"] == (0 if e["file"] == "initial_untrained.pt" else int(
      e["file"].split("_")[1].split(".")[0]
    ))
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
    keys_ok = all(k in fin for k in ("actor_state_dict", "critic_state_dict", "iter"))
    loaded = bool(keys_ok)
    report.check(
      "final checkpoint loads with actor/critic/iter",
      loaded,
      f"keys={sorted(k for k in fin)[:6]}",
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

  steps = {t: sorted(scalars[f"{NS}/{t}"]) for t in FIXED_TAGS if f"{NS}/{t}" in scalars}
  short = {t: (len(v), v[:1] + v[-1:]) for t, v in steps.items() if len(v) != args.iterations}
  report.check(
    f"each fixed scalar has {args.iterations} unique steps",
    not short,
    f"short={list(short.items())[:2]}" if short else f"{len(steps)} tags",
  )
  progress = scalars.get(f"{NS}/Progress/completed_iterations", {})
  report.check(
    "progress axis is 1..10000 with step == value",
    bool(progress) and min(progress) == 1 and max(progress) == args.iterations
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
  prod = scalars.get(f"{NS}/Reward/product_raw_mean", {})
  ratios = [
    env_rew[s] / prod[s] for s in (set(env_rew) & set(prod)) if abs(prod[s]) > 1e-6
  ]
  ratio_ok = bool(ratios) and abs(sum(ratios) / len(ratios) - 0.02) < 1e-3
  report.check(
    "env reward == 0.02 x r_product (dt scaling)",
    ratio_ok,
    f"mean ratio={sum(ratios) / len(ratios):.6f}" if ratios else "missing",
  )

  finite_bad = 0
  for key in (
    f"{NS}/Reward/task_raw_mean",
    f"{NS}/Reward/smp_raw_mean",
    f"{NS}/Reward/product_raw_mean",
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
    print(f"[info] iteration time median={median:.3f}s min={iter_times[0]:.3f}s "
          f"max={iter_times[-1]:.3f}s")

  # ------------------------------------------------------------ artefacts
  bad_files = check_artifacts(log_dir) + check_artifacts(ckpt_dir)
  report.check(
    "no forbidden per-iteration/video/GSI artefacts",
    not bad_files,
    f"{len(bad_files)} files" if bad_files else "clean",
  )

  elapsed = sorted(scalars.get(f"{NS}/Time/training_elapsed_s", {}))
  elapsed_h = (
    scalars[f"{NS}/Time/training_elapsed_s"][elapsed[-1]] / 3600.0 if elapsed else float("nan")
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
      ("Reward/product_raw_mean", "乘积奖励 r_product"),
      ("Reward/env_reward_mean", "环境返回奖励（0.02×r_product）"),
      ("Task/target_distance_mean_m", "目标距离均值 m"),
      ("Episode/complete_length_steps_mean", "完整回合长度（步）"),
      ("Failure/complete_rate", "完整回合失败率"),
    ]
    trend_rows = "\n".join(
      f"| {label} | {window_mean(tag, 1, 100):.4f} | {window_mean(tag, args.iterations - 99, args.iterations):.4f} |"
      for tag, label in trend
    )

    md = f"""# 实验 0001 完成摘要（G1 Location baseline）

运行标识：`{info.get('run_id')}`
生成时间：`{info.get('finished_at', 'n/a')}`（Asia/Shanghai）

## 结果

| 项目 | 数值 |
| --- | --- |
| Python 退出码 | `{exit_code}` |
| 完成迭代 | `{info.get('completed_iterations')}` / {args.iterations} |
| 训练样本 | `{info.get('total_samples')}` / {expected_samples} |
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
- 先验：`datasets/pretrain_ckpt/pretrained_lafan_run.pt`（sha256 `{info.get('prior_sha256')}`，未修改）
- 代码提交：`{info.get('git_commit')}`（分支 `{info.get('git_branch')}`）；启动时工作区包含本实验新增的 `experiments/0001/` 支撑文件（见 `run_info.json` 的 `git_dirty`）

## 验收明细

{report.markdown()}

## 结论

{'全部检查通过（' + str(len(report.rows)) + ' 项）。' if report.failed == 0 else str(report.failed) + ' 项检查未通过，见上表；本文件不伪造 PASS。'}

## 复现命令

```bash
cd /root/gpufree-data/smp-g1 && source env.sh && cd repo
experiments/0001/launch_0001.sh            # 正式训练（阶段 C）
experiments/0001/accept_0001.py --num-envs 64 --steps 2400 --out logs/_accept/b_stage  # 阶段 B
experiments/0001/verify_completion.py --log-dir logs/<run> --ckpt-dir ckpt/<run> --out COMPLETION.md
```

TensorBoard：`127.0.0.1:6009`（`experiments/0001/launch_tensorboard.sh 6009`）。
"""
    out.write_text(md, encoding="utf-8")
    print(f"[info] wrote {out}")

  print(f"{len(report.rows) - report.failed}/{len(report.rows)} checks passed")
  return code


if __name__ == "__main__":
  raise SystemExit(main())
