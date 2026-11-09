"""Independent real-GPU mixed-cache integration check; zero PPO updates.

Uses capacity 8 and a 100+100 iteration clock ONLY in this acceptance process.
Formal EXPERIMENT.json and all 50Hz/10-frame/1-second rules remain unchanged.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import torch
from check_mixed_audit import verify_audit
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

import smp.rl.tasks  # noqa: F401
from smp.rl.mixed_cache import MixedConfig, digest, install, restore


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--baseline-info", type=Path, required=True)
  p.add_argument("--out", type=Path, required=True)
  args = p.parse_args()
  if args.out.exists():
    raise FileExistsError(args.out)
  args.out.mkdir(parents=True)
  baseline = json.loads(args.baseline_info.read_text(encoding="utf-8"))
  cfg = load_env_cfg("Smp-Location-G1", play=False)
  cfg.scene.num_envs, cfg.seed = 64, 2027
  cfg.events["init_smp_state"].params.update(
    physical_filter=True, repair_small_joints=True
  )
  torch.manual_seed(2027)
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode=None)
  assert dataclasses.asdict(env._smp_physics.config) == baseline["gsi_physics_config"]
  assert (
    json.loads(json.dumps(env._smp_physics.manifest)) == baseline["gsi_model_manifest"]
  )
  env.reset()
  small = MixedConfig(
    capacity=8,
    replace_count=4,
    collect_after_iteration=0,
    activate_after_iteration=100,
    update_interval=100,
  )
  cache = install(env, config=small, audit_path=args.out / "mixed_cache_audit.json")
  generator = torch.Generator(device=env.device).manual_seed(2027007)
  try:
    for step in range(4800):
      action = (
        torch.randn(
          (env.num_envs, env.action_manager.total_action_dim),
          generator=generator,
          device=env.device,
        )
        * 0.3
      )
      obs, reward, _terminated, _timeouts, _extra = env.step(action)
      assert all(bool(torch.isfinite(value).all()) for value in obs.values())
      assert bool(torch.isfinite(reward).all())
      if step < 2400:
        assert cache.stats["policy_reset_draws"] == 0
    audit = json.loads((args.out / "mixed_cache_audit.json").read_text())
    verified = verify_audit(audit, small, 200, num_envs=64)
    assert verified["transactions"] == 2 and verified["inserted"] == 12
    assert cache.active and cache.stats["policy_reset_draws"] > 0
    assert cache.stats["pre_activation_policy_draws"] == 0
    assert cache.stats["first_policy_iteration"] >= 101
    assert env._smp_physics_totals["updates"] == 3
    assert env._smp_physics_totals["inserted"] == 6144
    ids = torch.tensor([0, 2, 4], device=env.device)
    other = torch.tensor([1, 3, 5], device=env.device)
    robot = env.scene["robot"]
    saved = robot.data.joint_pos[other].clone(), robot.data.joint_vel[other].clone()
    pool_hash = digest(env._smp_gsi_pool.detach().cpu().numpy())
    windows = cache.pool[:3].clone()
    restore(env, ids, windows)
    env.scene.write_data_to_sim()
    env.sim.forward()  # Acceptance only, to inspect values just written to sim.
    assert torch.allclose(
      robot.data.joint_pos[ids], windows[:, -1, 13:42], atol=1e-6, rtol=0
    )
    assert torch.allclose(
      robot.data.joint_vel[ids], windows[:, -1, 42:71], atol=1e-6, rtol=0
    )
    assert torch.equal(robot.data.joint_pos[other], saved[0])
    assert torch.equal(robot.data.joint_vel[other], saved[1])
    buf = env._smp_buffer
    assert torch.equal(buf.joint_pos[ids], windows[:, :, 13:42])
    assert torch.equal(buf.joint_vel[ids], windows[:, :, 42:71])
    assert torch.equal(buf.root_quat_w[ids], windows[:, :, 3:7])
    assert torch.allclose(
      robot.data.root_link_pos_w[ids, :2] - env.scene.env_origins[ids, :2],
      robot.data.default_root_state[ids, :2],
      atol=1e-5,
      rtol=0,
    )
    assert torch.allclose(
      robot.data.root_link_quat_w[ids], windows[:, -1, 3:7], atol=1e-5, rtol=0
    )
    assert torch.allclose(
      robot.data.root_link_lin_vel_w[ids], windows[:, -1, 7:10], atol=1e-5, rtol=1e-5
    )
    assert torch.allclose(
      robot.data.root_link_ang_vel_w[ids], windows[:, -1, 10:13], atol=1e-5, rtol=1e-5
    )
    # Saved EE positions came from post-forward frames. FK must agree after translation.
    assert torch.allclose(
      robot.data.body_link_pos_w[ids][:, env._smp_ee_indexes]
      - env.scene.env_origins[ids, None],
      buf.ee_pos_w[ids, -1],
      atol=1e-4,
      rtol=1e-5,
    )
    assert digest(env._smp_gsi_pool.detach().cpu().numpy()) == pool_hash
    assert np.isfinite(cache.pool.cpu().numpy()).all()
    result = dict(
      status="PASS",
      training_updates=0,
      formal_schedule_modified=False,
      test_config=dataclasses.asdict(small),
      verified_audit=verified,
      summary=cache.summary(),
      partial_restore=True,
      true_velocities_restored=True,
      ten_real_frames_restored=True,
      gsi_unchanged_by_policy_restore=True,
    )
    (args.out / "result.json").write_text(
      json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print("MIXED_ENVIRONMENT_CHECK=PASS")
  finally:
    env.close()


if __name__ == "__main__":
  main()
