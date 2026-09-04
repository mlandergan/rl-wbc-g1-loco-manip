# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""At each env's BEST-achieved vertical alignment moment during GRASP, report
how close the arm joints are to their hard mechanical limits.

Answers: is the ~5cm vertical-error plateau a hard reach-margin constraint
(arm pinned near its limits even at its closest approach) or a habit the
policy could still unlearn (real headroom left, just not using it)? Same
technique already used this session for the finger-knuckle and wrist-twist
findings, applied to the vertical axis.

Usage:
    CUDA_VISIBLE_DEVICES=0 python -u \\
        move_garbage/scripts/measure_vert_joint_margin.py --headless \\
        +checkpoint=<path_to_checkpoint.pt> ++num_envs=128 ++opt.use_wandb=False \\
        'hydra.searchpath=[file://<repo>/move_garbage/config]'
"""

import logging
import os
import sys

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from loguru import logger
from omegaconf import OmegaConf
from pathlib import Path

from gr00t.rl.utils.config_utils import register_rl_resolvers

register_rl_resolvers()

_CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "gr00t/rl/config")

ARM_JOINT_NAMES = [
    "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
    "wrist_roll", "wrist_pitch", "wrist_yaw",
]


@hydra.main(config_path=_CONFIG_DIR, config_name="base_eval")
def main(override_config: OmegaConf):
    hydra_log_path = os.path.join(HydraConfig.get().runtime.output_dir, "eval.log")
    logger.remove()
    logger.add(hydra_log_path, level="DEBUG")
    logger.add(sys.stdout, level="INFO", colorize=True)

    from gr00t.rl.utils.logging import HydraLoggerBridge

    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger().addHandler(HydraLoggerBridge())
    os.chdir(hydra.utils.get_original_cwd())

    checkpoint = Path(override_config.checkpoint)
    config_path = checkpoint.parent / "config.yaml"
    with open(config_path) as f:
        train_config = OmegaConf.load(f)
    if train_config.eval_overrides is not None:
        train_config = OmegaConf.merge(train_config, train_config.eval_overrides)
    config = OmegaConf.merge(train_config, override_config)
    config.experiment_dir = checkpoint.parent

    simulator_type = config.simulator["_target_"].split(".")[-1]
    if simulator_type == "IsaacSim":
        from isaaclab.app import AppLauncher
        import argparse

        parser = argparse.ArgumentParser()
        AppLauncher.add_app_launcher_args(parser)
        args_cli, hydra_args = parser.parse_known_args()
        sys.argv = [sys.argv[0]] + hydra_args
        args_cli.num_envs = config.num_envs
        args_cli.seed = config.seed
        args_cli.env_spacing = config.env.config.env_spacing
        args_cli.output_dir = config.output_dir
        args_cli.enable_cameras = False
        args_cli.headless = config.headless
        app_launcher = AppLauncher(args_cli)
        simulation_app = app_launcher.app

    from accelerate import Accelerator
    from transformers import HfArgumentParser
    from trl import ModelConfig, PPOConfig, ScriptArguments

    from gr00t.rl.trl.utils.common import custom_instantiate
    from gr00t.rl.utils.common import seeding
    from gr00t.rl.utils.helpers import pre_process_config

    os.chdir(hydra.utils.get_original_cwd())
    pre_process_config(config)

    config.env.config.save_rendering_dir = str(checkpoint.parent / "renderings" / "vert_margin_tmp")
    config.env.config.ckpt_dir = str(checkpoint.parent)
    config.algo.trl.output_dir = str(Path(config.experiment_dir))
    parser = HfArgumentParser((ScriptArguments, PPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_dict(config.algo.trl)

    accelerator = Accelerator()
    device = str(accelerator.device)
    if device == "cuda":
        device = "cuda:0"
    seeding(config.seed)

    env = instantiate(config=config.env, device=device)

    from gr00t.rl.agents.modules.ppo_modules import PPOStateActor, PPOCritic

    module_dim_dict = getattr(config.algo.config, "module_dim", {})
    from hydra.utils import instantiate as inst

    policy = inst(
        config.algo.config.actor, env_config=env.config, algo_config=config.algo.config,
        module_dim_dict=module_dim_dict, _recursive_=False,
    ).to(device)
    value_model = inst(
        config.algo.config.critic, env_config=env.config, algo_config=config.algo.config,
        module_dim_dict=module_dim_dict, _recursive_=False,
    ).to(device)

    accelerator.wait_for_everyone()

    trainer = custom_instantiate(
        config.trainer, args=training_args, config=config.algo.config, env=env,
        model=policy, ref_model=None, use_ref_model=False, value_model=value_model,
        train_dataset=None, eval_dataset=None, callbacks=[],
        checkpoint=config.checkpoint, local_seed=config.seed, accelerator=accelerator,
    )

    import torch
    import numpy as np
    from gr00t.rl.trl.modules.homie_modules import init_actor_critic_dict as _iacd

    play_steps = int(config.get("play_steps", 400))

    env_u = env
    device_t = env_u.device
    model = trainer.model
    policy_model = model.policy
    homie_walk_model = model.homie_walk_model
    homie_stand_model = model.homie_stand_model
    homie_switch_threshold = getattr(trainer, "homie_switch_threshold", 0.5)
    homie_one_step_obs = _iacd["num_one_step_obs"]

    trainer._eval_mode()
    policy_model.init_rollout()

    obs_dict = env_u.reset_all()
    for k in obs_dict:
        obs_dict[k] = obs_dict[k].to(device_t)

    left_dof_idx = env_u._left_arm_dof_idx
    right_dof_idx = env_u._right_arm_dof_idx
    hard_limits = env_u.simulator.hard_dof_pos_limits  # (num_dof, 2), radians

    stage_hist, left_vert_hist, right_vert_hist = [], [], []
    left_arm_pos_hist, right_arm_pos_hist = [], []

    with torch.no_grad():
        for step in range(play_steps):
            policy_model.rollout(obs_dict=obs_dict)
            action_mean = policy_model.action_mean.detach()
            homie_obs = obs_dict["homie_obs"]
            walk_out = homie_walk_model(homie_obs)
            stand_out = homie_stand_model(homie_obs)
            commands = homie_obs[..., -homie_one_step_obs:-(homie_one_step_obs - 3)]
            walk_mask = torch.norm(commands, dim=-1, keepdim=True) > homie_switch_threshold
            m = walk_mask
            while m.dim() < walk_out["actions"].dim():
                m = m.unsqueeze(-1)
            homie_actions = torch.where(m, walk_out["action_mean"], stand_out["action_mean"])
            step_actions = torch.cat([action_mean, homie_actions], dim=-1)

            obs_dict, rewards, dones, infos = env_u.step({"actions": step_actions})
            for k in obs_dict:
                obs_dict[k] = obs_dict[k].to(device_t)

            left_palm, right_palm = env_u._palm_grasp_points()
            left_target = env_u._left_handle_target()
            right_target = env_u._right_handle_target()
            left_vert = (left_target - left_palm)[:, 2]
            right_vert = (right_target - right_palm)[:, 2]

            stage_hist.append(env_u.stage_buf.detach().cpu().numpy().copy())
            left_vert_hist.append(left_vert.detach().cpu().numpy())
            right_vert_hist.append(right_vert.detach().cpu().numpy())
            left_arm_pos_hist.append(env_u.simulator.dof_pos[:, left_dof_idx].detach().cpu().numpy())
            right_arm_pos_hist.append(env_u.simulator.dof_pos[:, right_dof_idx].detach().cpu().numpy())

            if step % 50 == 0:
                print(f"[vert_margin] step={step}", flush=True)

    stage = np.stack(stage_hist)                  # (T, N)
    left_vert = np.stack(left_vert_hist)           # (T, N)
    right_vert = np.stack(right_vert_hist)         # (T, N)
    left_arm_pos = np.stack(left_arm_pos_hist)     # (T, N, 7)
    right_arm_pos = np.stack(right_arm_pos_hist)   # (T, N, 7)
    lo = hard_limits[:, 0].detach().cpu().numpy()
    hi = hard_limits[:, 1].detach().cpu().numpy()
    left_lo, left_hi = lo[left_dof_idx.cpu().numpy()], hi[left_dof_idx.cpu().numpy()]
    right_lo, right_hi = lo[right_dof_idx.cpu().numpy()], hi[right_dof_idx.cpu().numpy()]

    grasp_mask = stage == 2  # (T, N)
    T, N = stage.shape

    print()
    print("==================== VERTICAL-ERROR JOINT-MARGIN CHECK ====================")
    print(f"num_envs={N}  play_steps={T}  checkpoint={checkpoint.name}")
    print(f"GRASP-stage (step,env) samples: {grasp_mask.sum()}")
    print()

    for side, vert, arm_pos, jlo, jhi in [
        ("LEFT", left_vert, left_arm_pos, left_lo, left_hi),
        ("RIGHT", right_vert, right_arm_pos, right_lo, right_hi),
    ]:
        abs_vert_grasp = np.where(grasp_mask, np.abs(vert), np.inf)  # (T, N)
        envs_with_grasp = np.where(grasp_mask.any(axis=0))[0]
        if len(envs_with_grasp) == 0:
            print(f"{side}: no env reached GRASP stage, skipping.")
            continue
        best_step_per_env = abs_vert_grasp[:, envs_with_grasp].argmin(axis=0)  # (n_envs,)
        best_vert_per_env = abs_vert_grasp[best_step_per_env, envs_with_grasp]
        best_joint_pos = arm_pos[best_step_per_env, envs_with_grasp, :]  # (n_envs, 7)

        margin_lo = best_joint_pos - jlo[None, :]   # (n_envs, 7)
        margin_hi = jhi[None, :] - best_joint_pos
        margin = np.minimum(margin_lo, margin_hi)   # (n_envs, 7) -- distance to nearer limit

        print(f"--- {side} hand: at each env's BEST vertical-alignment moment in GRASP ---")
        print(f"  n_envs_with_grasp_samples={len(envs_with_grasp)}  "
              f"mean_best_abs_vert={best_vert_per_env.mean()*100:.2f}cm  "
              f"median_best_abs_vert={np.median(best_vert_per_env)*100:.2f}cm  "
              f"frac_under_3cm_even_at_best={ (best_vert_per_env < 0.03).mean():.4f}")
        print(f"  {'joint':14s}  {'mean_angle_deg':>15s}  {'range_deg':>18s}  "
              f"{'mean_margin_deg':>16s}  {'min_margin_deg':>15s}  {'frac_within_5deg_of_limit':>26s}")
        for j, name in enumerate(ARM_JOINT_NAMES):
            ang = np.degrees(best_joint_pos[:, j])
            rng = f"[{np.degrees(jlo[j]):.0f},{np.degrees(jhi[j]):.0f}]"
            m_deg = np.degrees(margin[:, j])
            near_limit = (m_deg < 5.0).mean()
            print(f"  {name:14s}  {ang.mean():15.1f}  {rng:>18s}  {m_deg.mean():16.1f}  "
                  f"{m_deg.min():15.1f}  {near_limit:26.4f}")
        print()

    print("================================================================")
    os._exit(0)


if __name__ == "__main__":
    main()
