"""Scoped startup/refresh adapters; baseline reset and ten-frame priming reused."""

from __future__ import annotations

import torch


def configure(env_cfg):
  startup = env_cfg.events["init_smp_state"]
  startup.func = init_smp_state
  startup.params.update(physical_filter=True, repair_small_joints=False)
  assert startup.params["gsi_batch_size"] == 1024
  assert startup.params["gsi_buffer_size"] == 4096
  refresh = env_cfg.events["gsi_refresh"]
  refresh.func = gsi_refresh
  assert refresh.params.get("num_samples", 1024) == 1024
  assert refresh.params.get("step_interval", 2400) == 2400


def init_smp_state(
  env,
  env_ids=None,
  ckpt_path="",
  gsi_buffer_size=4096,
  gsi_batch_size=1024,
  compile_model=True,
  compile_mode=None,
  physical_filter=True,
  repair_small_joints=False,
):
  from runtime import collect, initialize

  from smp.rl.events import NUM_JOINTS, _maybe_compile, gsi_reset
  from smp.rl.utils import DiffNormalizer, MotionFeatureBuffer, load_denoiser
  from smp.sampling.feature_to_state import EE_BODY_NAMES, NUM_EE

  del env_ids
  assert physical_filter and not repair_small_joints
  if not ckpt_path:
    raise ValueError("frozen prior checkpoint is required")
  model, scheduler, low, high, F, W = load_denoiser(ckpt_path, env.device)
  model = _maybe_compile(model, compile_model, compile_mode)
  env._smp_bundle = (model, scheduler, low, high, F, W)
  robot = env.scene["robot"]
  env._smp_ee_indexes = torch.tensor(
    robot.find_bodies(list(EE_BODY_NAMES), preserve_order=True)[0],
    dtype=torch.long,
    device=env.device,
  )
  env._smp_buffer = MotionFeatureBuffer(
    num_envs=env.num_envs,
    window_size=W,
    num_joints=NUM_JOINTS,
    num_ee=NUM_EE,
    device=env.device,
  )
  env._smp_normalizer = DiffNormalizer(scheduler.num_timesteps, env.device)
  initialize(env, batch_size=gsi_batch_size)
  env._smp_gsi_pool = collect(env, gsi_buffer_size)
  env._smp_gsi_repaired_rows = env._smp_physics_last_repaired.clone()
  env._smp_gsi_original_pass = env._smp_noise_last_original_pass.clone()
  env._smp_gsi_physical_pass = env._smp_noise_last_final_pass.clone()
  if compile_model and env.num_envs != gsi_batch_size:
    with torch.no_grad():
      dummy = torch.randn(env.num_envs, W, F, device=env.device)
      times = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
      model(dummy, times)
  gsi_reset(env)


@torch.no_grad()
def gsi_refresh(env, env_ids=None, num_samples=1024, step_interval=2400):
  from runtime import collect

  del env_ids
  cur = int(env.common_step_counter)
  if cur == 0 or cur % step_interval:
    return
  pool = env._smp_gsi_pool
  if num_samples > len(pool):
    raise ValueError("refresh count exceeds GSI capacity")
  new = collect(env, num_samples)
  head = int(getattr(env, "_smp_gsi_head", 0))
  # The index expression consumes no RNG and matches the baseline FIFO split.
  ids = (torch.arange(num_samples, device=pool.device) + head) % len(pool)
  pool[ids] = new
  env._smp_gsi_repaired_rows[ids] = env._smp_physics_last_repaired
  env._smp_gsi_original_pass[ids] = env._smp_noise_last_original_pass
  env._smp_gsi_physical_pass[ids] = env._smp_noise_last_final_pass
  env._smp_gsi_head = (head + num_samples) % len(pool)
