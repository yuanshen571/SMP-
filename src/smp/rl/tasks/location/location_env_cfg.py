"""G1 location task with SMP guidance.

Each env gets a periodically-resampled world-frame xy goal; reward is
position tracking plus SMP guidance with the original SMP outer weights.
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from smp.rl.env_cfg import g1_smp_env_cfg
from smp.rl.rewards import task_smp_additive
from smp.rl.tasks.location import mdp


def g1_location_smp_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Build the G1 location env cfg with SMP guidance."""
  cfg = g1_smp_env_cfg(play=play)

  # --- Commands ------------------------------------------------------------
  cfg.commands["location"] = mdp.LocationCommandCfg(
    entity_name="robot",
    resampling_time_range=(5.0, 10.0),
    tar_dist_min=1.0,
    tar_dist_max=10.0,
    debug_vis=True,
  )

  # --- Observations --------------------------------------------------------
  command_obs = ObservationTermCfg(
    func=mdp.generated_commands,
    params={"command_name": "location"},
  )
  cfg.observations["actor"].terms["command"] = command_obs
  cfg.observations["critic"].terms["command"] = command_obs

  # --- Rewards -------------------------------------------------------------
  # Original Location internal weights: position=1, velocity=0, facing=0.
  # Zero-weight terms stay absent, as in 0001; keep the G1 position kernel.
  # Original SMP outer weights: 0.5 task + 0.5 SMP.
  cfg.rewards["task_smp_additive"] = RewardTermCfg(
    func=task_smp_additive,
    weight=1.0,
    params={
      "task_terms": (
        (
          mdp.location_position,
          1.0,
          {"command_name": "location", "pos_err_scale": 0.25},
        ),
      ),
      "ws": 4,
      "task_reward_weight": 0.5,
      "smp_reward_weight": 0.5,
    },
  )

  # --- Events --------------------------------------------------------------
  cfg.events["init_smp_state"].params["ckpt_path"] = (
    "datasets/pretrain_ckpt/pretrained_lafan_run.pt"
  )

  # --- Terminations --------------------------------------------------------
  cfg.terminations["base_too_low"] = TerminationTermCfg(
    func=mdp.root_height_below_minimum,
    params={
      "minimum_height": 0.3,
      "asset_cfg": SceneEntityCfg("robot"),
    },
  )

  return cfg
