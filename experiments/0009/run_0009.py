#!/usr/bin/env python
"""Experiment 0009: G1 Location with 0.5 task + 0.5 SMP rewards.

Experiment 0002 method with 20000 iterations and paired GSI/noise recording.
Only DDPM generation uses a private RNG; FIFO/reset/reward/PPO logic is unchanged.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

import importlib.util  # noqa: E402

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import MjlabOnPolicyRunner  # noqa: E402
from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg  # noqa: E402
from mjlab.utils.os import dump_yaml  # noqa: E402

import smp.rl.tasks  # noqa: E402,F401  (registers Smp-Location-G1)

_spec = importlib.util.spec_from_file_location(
  "smp_experiment_0009_metrics", Path(__file__).with_name("passive_metrics.py")
)
assert _spec is not None and _spec.loader is not None
_metrics_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_metrics_mod)
PassiveMetrics = _metrics_mod.PassiveMetrics
build_logged_task_smp_additive = _metrics_mod.build_logged_task_smp_additive
install_metrics = _metrics_mod.install

from gsi_archive import ArchiveRecorder, atomic_json  # noqa: E402
from initialization import assert_untrained, initial_models_digest  # noqa: E402

TASK_ID = "Smp-Location-G1"
EXPERIMENT = "0009"
TZ = ZoneInfo("Asia/Shanghai")

PRIOR = Path("datasets/pretrain_ckpt/pretrained_lafan_run.pt")
PRIOR_SHA256 = "74787104ff124191136eeed2268f79448da3ac2e30e710d42aebfc5a60e58cb3"
UV_LOCK = Path("uv.lock")
UV_LOCK_SHA256 = "ce041c7774b1d4abf42e868fa3b4af2973954cec61d35c3b4e58a7e27464d9ce"

METRICS_DEFINITION = """\
# 0009 metric definitions (TensorBoard namespace `0009/`)

All values are raw (never smoothed in the writer) and are logged with
`global_step` = number of completed PPO iterations (1-based, last = 20000).
Every quantity is cached from tensors the environment already produced during
that step; the logging path does not consume random numbers and does not run
extra simulation, resets or forwards.

## Reward
- `Reward/task_raw_mean` — mean over all envs x steps of the raw task reward
  `r_task = exp(-0.25 * horizontal_distance_m)` (before `weight`/`dt` scaling).
- `Reward/smp_raw_mean` — mean of `r_smp = exp(-mean_mse * ws)` from the single
  `smp_guidance_reward()` call per step (`ws = 4` for Location).
- `Reward/additive_raw_mean` — `mean(0.5*r_task + 0.5*r_smp)` over the
  same env-step samples. It must equal 0.5*mean(task) + 0.5*mean(smp).
- `Reward/env_reward_mean` — mean of the reward buffer actually returned to the
  trainer, i.e. `weight * r_additive * step_dt` with `step_dt = 0.02 s`
  (`scale_rewards_by_dt=True`, weight 1.0).

## Task / motion
- `Task/target_distance_mean_m` — mean horizontal distance between robot root
  and the current location target, sampled at the same step.
- `Motion/horizontal_speed_mean_mps` — mean horizontal root speed.

## Episode counts (per iteration, over the num_envs x 24 step slots)
- `Count/ended_episode` — envs that ended in this iteration.
- `Count/failure_episode` — terminated for any reason other than pure timeout
  (includes simultaneous termination+timeout, matching §8).
- `Count/timeout_only_episode` — ended only because the 20 s horizon expired.
- Identity: `failure_episode + timeout_only_episode == ended_episode`.
- `Count/complete_episode` / `_failure_episode` / `_timeout_only_episode` —
  same, restricted to episodes that started from a clean reset (the first,
  randomly offset episode of every training run is excluded and counted in
  `Count/startup_incomplete_episode`).
- `Count/startup_incomplete_episode` — offset first episodes ending here.

## Episode sums (nothing is censored)
- `Sum/complete_episode_length_steps` — total lasted steps of complete episodes.
- `Sum/complete_failure_duration_s` — total duration of complete failed episodes.
- `Sum/complete_task_return_raw` — sum of undiscounted `r_task` returns.
- `Sum/complete_training_return` — sum of undiscounted `weight*dt*r_additive`
  returns (the reward the learner sees).

## Means (written only when the denominator is non-zero)
- `Episode/complete_task_return_raw_mean`
- `Episode/complete_training_return_mean`
- `Episode/complete_length_steps_mean`
- `Episode/complete_survival_s_mean` = steps * 0.02 s
- `Failure/complete_rate` = complete_failure / complete
- `Failure/complete_duration_s_mean` = sum_dur / complete_failure
- `Failure/all_ended_rate` = failure / ended

## Progress / time
- `Progress/completed_iterations` — 1..20000.
- `Progress/training_samples` = iterations * 24 * num_envs (must reach
  491520000 at iteration 20000 with 1024 envs).
- `Time/iteration_s` — collection + learning wall time of this iteration.
- `Time/training_elapsed_s` — accumulated `iteration_s`.
"""


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1 << 20), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _git(*args: str) -> str:
  try:
    return subprocess.check_output(
      ["git", *args], text=True, stderr=subprocess.DEVNULL
    ).strip()
  except (OSError, subprocess.CalledProcessError):
    return "unknown"


def _now() -> datetime:
  return datetime.now(TZ)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="experiment 0009 launcher")
  parser.add_argument("--num-envs", type=int, default=1024)
  parser.add_argument("--max-iterations", type=int, default=20000)
  parser.add_argument("--seed", type=int, default=2027)
  parser.add_argument("--device", type=str, default="cuda:0")
  parser.add_argument(
    "--log-root", type=str, default=str(Path("logs").resolve()), help="run logs root"
  )
  parser.add_argument(
    "--ckpt-root", type=str, default=str(Path("ckpt").resolve()), help="checkpoints"
  )
  parser.add_argument("--run", type=str, default="", help="run id (default: timestamp)")
  parser.add_argument(
    "--no-metrics",
    action="store_true",
    help="disable the passive metrics layer entirely (A/B determinism check)",
  )
  parser.add_argument("--stage", choices=("smoke", "formal"), default="formal")
  parser.add_argument("--archive-root", type=Path, default=Path("/root/gpufree-share/smp_gsi_paired_0009"))
  return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
  args = parse_args(argv)
  assert args.seed == 2027, "this experiment is fixed to seed 2027"
  assert args.num_envs == 1024
  assert args.max_iterations == (20000 if args.stage == "formal" else 2)
  assert not args.no_metrics, "both paired experiments use the same passive metrics"
  assert Path("/root/gpufree-share").is_dir(), "requested /root/gpufree-share must exist; do not silently use another disk"
  assert args.archive_root.is_absolute()
  assert args.archive_root.resolve().is_relative_to(Path("/root/gpufree-share").resolve())
  if args.stage == "formal":
    assert shutil.disk_usage("/root/gpufree-share").free >= 35_000_000_000, "reserve at least 35 GB before formal recording"

  from contract import check_files

  check_files(quiet=True)

  env_cfg = load_env_cfg(TASK_ID, play=False)
  agent_cfg = load_rl_cfg(TASK_ID)

  # ---------------------------------------------------------- overrides §6
  env_cfg.scene.num_envs = args.num_envs
  env_cfg.seed = args.seed
  agent_cfg.seed = args.seed
  agent_cfg.max_iterations = args.max_iterations
  agent_cfg.experiment_name = f"{EXPERIMENT}_location"
  agent_cfg.logger = "tensorboard"
  agent_cfg.upload_model = False
  agent_cfg.save_interval = 500  # explicit; matches the task-sheet default
  agent_cfg.run_name = agent_cfg.run_name or f"seed{args.seed}"

  steps_per_env = int(agent_cfg.num_steps_per_env)
  # Fail before creating the environment if this branch/config was mixed up.
  assert set(env_cfg.rewards) == {"task_smp_additive"}
  reward_term = env_cfg.rewards["task_smp_additive"]
  rp = reward_term.params
  assert reward_term.weight == 1.0
  assert rp["task_reward_weight"] == rp["smp_reward_weight"] == 0.5
  assert rp["ws"] == 4
  assert len(rp["task_terms"]) == 1
  pos_func, pos_weight, pos_kwargs = rp["task_terms"][0]
  assert pos_func.__name__ == "location_position" and pos_weight == 1.0
  assert pos_kwargs == {"command_name": "location", "pos_err_scale": 0.25}
  reward_cfg_before = dataclasses.asdict(reward_term)

  # ------------------------------------------------------------- run paths
  stamp = _now()
  run_id = args.run or f"{EXPERIMENT}_location_seed{args.seed}_{stamp:%Y%m%d_%H%M%S}"
  log_dir = Path(args.log_root).resolve() / run_id
  ckpt_dir = Path(args.ckpt_root).resolve() / run_id
  for path in (log_dir, ckpt_dir):
    # The stage-C launcher pre-creates logs/<run> so that `tee` can open
    # console.log before python starts.  Refuse only when the directory
    # already holds run artefacts, never silently overwrite a previous run.
    if path.exists():
      leftovers = [
        p
        for p in path.iterdir()
        if not (path == log_dir and p.name == "console.log" and p.stat().st_size == 0)
      ]
      if leftovers:
        sys.exit(f"refusing to overwrite existing run directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
  params_dir = log_dir / "params"
  if params_dir.exists() and any(params_dir.iterdir()):
    sys.exit(f"refusing to overwrite existing run directory: {log_dir}")
  params_dir.mkdir(exist_ok=True)

  print(f"[0009] run_id           = {run_id}")
  print(f"[0009] log_dir          = {log_dir}")
  print(f"[0009] ckpt_dir         = {ckpt_dir}")
  print(
    f"[0009] num_envs={args.num_envs} steps/env={steps_per_env} "
    f"iters={args.max_iterations} seed={args.seed} metrics={not args.no_metrics}"
  )

  # ------------------------------------------------------- passive metrics
  term = build_logged_task_smp_additive()
  if not args.no_metrics:
    env_cfg.rewards["task_smp_additive"].func = term
  reward_cfg_after = dataclasses.asdict(env_cfg.rewards["task_smp_additive"])
  assert reward_cfg_after["weight"] == reward_cfg_before["weight"] and repr(
    reward_cfg_after["params"]
  ) == repr(reward_cfg_before["params"]), (
    "logging must preserve the configured 0009 reward weights/params"
  )
  print(
    "[0009] reward term: weight=%.3f params=%s"
    % (reward_cfg_after["weight"], reward_cfg_after["params"])
  )

  # ---------------------------------------------------------------- record
  prior_sha = _sha256(PRIOR) if PRIOR.exists() else "missing"
  lock_sha = _sha256(UV_LOCK) if UV_LOCK.exists() else "missing"
  if prior_sha != PRIOR_SHA256:
    sys.exit(f"prior checkpoint sha256 mismatch: {prior_sha} != {PRIOR_SHA256}")
  if lock_sha != UV_LOCK_SHA256:
    sys.exit(f"uv.lock sha256 mismatch: {lock_sha} != {UV_LOCK_SHA256}")

  env_dict = dataclasses.asdict(env_cfg)
  agent_dict = dataclasses.asdict(agent_cfg)
  dump_yaml(log_dir / "params" / "env.yaml", env_dict)
  dump_yaml(log_dir / "params" / "agent.yaml", agent_dict)
  (log_dir / "metrics_definition.md").write_text(METRICS_DEFINITION, encoding="utf-8")

  gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
  run_info = {
    "run_id": run_id,
    "started_at": stamp.isoformat(),
    "task_id": TASK_ID,
    "experiment": EXPERIMENT,
    "repeats_experiment": "0002",
    "initialization_seed": 2027,
    "initialization_mode": "fresh_seed",
    "baseline_files_verified": True,
    "physical_filter_enabled": False,
    "mixed_cache_enabled": False,
    "clustering_enabled": False,
    "baseline_commit": "0a4ceff351b1f4885b61b6bfddee80ec49cb1874",
    "method_origin_commit": "0a4ceff351b1f4885b61b6bfddee80ec49cb1874",
    "reward": {
      "formula": "0.5 * r_task + 0.5 * r_smp",
      "task_reward_weight": 0.5,
      "smp_reward_weight": 0.5,
      "position_weight": 1.0,
      "velocity_weight": 0.0,
      "facing_weight": 0.0,
      "position_kernel": "exp(-0.25 * horizontal_distance_m)",
      "smp_ws": 4,
      "reward_manager_weight": 1.0,
    },
    "log_dir": str(log_dir),
    "ckpt_dir": str(ckpt_dir),
    "num_envs": args.num_envs,
    "num_steps_per_env": steps_per_env,
    "max_iterations": args.max_iterations,
    "expected_samples": args.num_envs * steps_per_env * args.max_iterations,
    "seed": args.seed,
    "device": args.device,
    "gpu": gpu,
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "python": sys.version.split()[0],
    "platform": platform.platform(),
    "hostname": platform.node(),
    "argv": sys.argv,
    "git_commit": _git("rev-parse", "HEAD"),
    "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
    "git_dirty": _git("status", "--porcelain"),
    "prior_path": str(PRIOR),
    "prior_sha256": prior_sha,
    "uv_lock_sha256": lock_sha,
    "metrics_enabled": not args.no_metrics,
    "stage": args.stage,
    "gsi_generation_rng_seed": 102027,
    "gsi_generation_rng_isolated": True,
    "gsi_recording_enabled": True,
    "noise_rescue_enabled": False,
    "episode_length_s": env_dict.get("episode_length_s"),
    "decimation": env_dict.get("decimation"),
    "scale_rewards_by_dt": env_dict.get("scale_rewards_by_dt"),
    "timestep": (env_dict.get("sim") or {}).get("mujoco", {}).get("timestep"),
    "commands": {
      name: {"resampling_time_range": c.get("resampling_time_range")}
      for name, c in (env_dict.get("commands") or {}).items()
    },
  }
  (log_dir / "run_info.json").write_text(
    json.dumps(run_info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
  )
  print(f"[0009] prior sha256     = {prior_sha}")
  print(f"[0009] uv.lock sha256   = {lock_sha}")
  print(
    f"[0009] git              = {run_info['git_commit'][:12]} "
    f"({run_info['git_branch']})"
  )

  # Pairing archive is created before the actual environment startup GSI.
  archive_dir = args.archive_root.resolve() / args.stage / run_id
  archive_dir.parent.mkdir(parents=True, exist_ok=True)
  recorder = ArchiveRecorder(archive_dir, seed=102027, device=args.device,
                            provenance={"run_id": run_id, "stage": args.stage,
                                        "training_seed": 2027,
                                        "git_commit": run_info["git_commit"],
                                        "prior_sha256": prior_sha,
                                        "uv_lock_sha256": lock_sha})
  import smp.rl.events as native_events
  recorder.install(native_events)
  run_info["gsi_archive_dir"] = str(archive_dir)
  run_info["gsi_archive_manifest"] = str(archive_dir / "manifest.json")
  (log_dir / "run_info.json").write_text(
    json.dumps(run_info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
  )
  # ------------------------------------------------------------------- env
  env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode=None)
  print(
    f"[0009] env created: num_envs={env.num_envs} step_dt={env.step_dt} "
    f"max_episode_length={env.max_episode_length}"
  )

  metrics: PassiveMetrics | None = None
  if not args.no_metrics:
    metrics = PassiveMetrics(env, steps_per_env=steps_per_env)
    term.metrics = metrics
    install_metrics(env, metrics)

  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner = MjlabOnPolicyRunner(wrapped, agent_dict, str(log_dir), device=args.device)
  assert not hasattr(env, "_smp_physics"), "0009 must have no physical filter"
  assert_untrained(runner.alg.save())
  run_info["initial_models_sha256"] = initial_models_digest(runner.alg.save())
  runner.add_git_repo_to_log(__file__)

  # ------------------------------------------------ checkpoint redirection
  index_path = ckpt_dir / "index.json"
  index: list[dict] = []
  original_save = runner.save

  def save(path: str, infos: dict | None = None) -> None:
    target = ckpt_dir / Path(path).name
    payload = dict(infos or {})
    payload.setdefault("run_id", run_id)
    payload.setdefault("experiment", EXPERIMENT)
    payload.setdefault("initialization_seed", 2027)
    original_save(str(target), payload)
    initial = target.name == "initial_untrained.pt"
    iteration = int(runner.current_learning_iteration)
    completed = 0 if initial else iteration + 1
    index.append(
      {
        "file": target.name,
        "native_iter": iteration,
        "completed_iterations": completed,
        "samples_on_save": completed * steps_per_env * int(env.num_envs),
        "size_bytes": target.stat().st_size,
        "sha256": _sha256(target),
      }
    )
    index_path.write_text(
      json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

  runner.save = save  # type: ignore[method-assign]

  # §9: real untrained checkpoint, stored before the first gradient update,
  # written through runner.save so it carries iter/common_step_counter and
  # consumes no random numbers.
  save(str(ckpt_dir / "initial_untrained.pt"), {"stage": "initial_untrained"})
  print(f"[0009] wrote {ckpt_dir / 'initial_untrained.pt'}")
  run_info["saved_initial_checkpoint_sha256"] = _sha256(
    ckpt_dir / "initial_untrained.pt"
  )
  (log_dir / "run_info.json").write_text(
    json.dumps(run_info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
  )

  recorder.manifest["provenance"].update(
    initial_models_sha256=run_info["initial_models_sha256"],
    saved_initial_checkpoint_sha256=run_info["saved_initial_checkpoint_sha256"],
    initial_checkpoint=str(ckpt_dir / "initial_untrained.pt"),
    run_info=str(log_dir / "run_info.json"),
  )
  atomic_json(archive_dir / "manifest.json", recorder.manifest)
  if metrics is not None:
    metrics.attach(runner)

  try:
    runner.learn(
      num_learning_iterations=args.max_iterations, init_at_random_ep_len=True
    )
    assert int(runner.current_learning_iteration) + 1 == args.max_iterations
    recorder.finish(env, completed_iterations=args.max_iterations, steps_per_env=steps_per_env)
    run_info["gsi_recorded_rows"] = recorder.manifest["rows"]
    run_info["gsi_recorded_shards"] = len(recorder.manifest["batches"])
    run_info["gsi_archive_status"] = recorder.manifest["status"]
  finally:
    if metrics is not None:
      metrics.finalize(runner)
    wrapped.close()
    recorder.uninstall()

  completed = int(runner.current_learning_iteration) + 1
  run_info["finished_at"] = _now().isoformat()
  run_info["completed_iterations"] = completed
  run_info["total_samples"] = completed * steps_per_env * int(args.num_envs)
  (log_dir / "run_info.json").write_text(
    json.dumps(run_info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
  )
  print(f"[0009] finished: iterations={completed} samples={run_info['total_samples']}")
  return 0


if __name__ == "__main__":
  os.environ.setdefault("WANDB_MODE", "disabled")
  raise SystemExit(main())
