# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure pelvis standoff distance averaged across ALL envs (not just env 0).

play_bin_push.py only ever logs env-0's trajectory regardless of num_envs, so it
can't answer "what's the average stop distance across envs". This runs a policy
rollout at a larger num_envs and reports per-step and steady-state (tail) mean
signed standoff, split by stage, across the whole batch.

Usage:
    CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n isaacsim5.1 python -u \\
        move_garbage/scripts/measure_standoff.py --headless \\
        +checkpoint=<path_to_checkpoint.pt> ++num_envs=256 ++opt.use_wandb=False \\
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


@hydra.main(
    config_path=_CONFIG_DIR,
    config_name="base_eval",
)
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
        import isaaclab

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

    unresolved_conf = OmegaConf.to_container(config, resolve=False)
    os.chdir(hydra.utils.get_original_cwd())
    pre_process_config(config)

    config.env.config.save_rendering_dir = str(checkpoint.parent / "renderings" / "measure_tmp")
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
        config.algo.config.actor,
        env_config=env.config,
        algo_config=config.algo.config,
        module_dim_dict=module_dim_dict,
        _recursive_=False,
    ).to(device)
    value_model = inst(
        config.algo.config.critic,
        env_config=env.config,
        algo_config=config.algo.config,
        module_dim_dict=module_dim_dict,
        _recursive_=False,
    ).to(device)

    accelerator.wait_for_everyone()

    callbacks = []
    trainer = custom_instantiate(
        config.trainer,
        args=training_args,
        config=config.algo.config,
        env=env,
        model=policy,
        ref_model=None,
        use_ref_model=False,
        value_model=value_model,
        train_dataset=None,
        eval_dataset=None,
        callbacks=callbacks,
        checkpoint=config.checkpoint,
        local_seed=config.seed,
        accelerator=accelerator,
    )

    import torch
    from gr00t.rl.trl.modules.homie_modules import init_actor_critic_dict as _iacd

    play_steps = int(config.get("play_steps", 300))

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

    standoff_hist = []
    stage_hist = []
    left_dist_hist = []
    right_dist_hist = []
    left_horiz_hist = []
    right_horiz_hist = []
    left_vert_hist = []
    right_vert_hist = []
    left_align_hist = []
    right_align_hist = []
    left_twist_hist = []
    right_twist_hist = []
    grasp_valid_hist = []

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

            signed_standoff, _, _ = env_u._grasp_standoff_state()
            left_palm, right_palm = env_u._palm_grasp_points()
            left_target = env_u._left_handle_target()
            right_target = env_u._right_handle_target()
            left_delta = left_target - left_palm
            right_delta = right_target - right_palm
            left_dist = left_delta.norm(dim=-1)
            right_dist = right_delta.norm(dim=-1)
            left_horiz = left_delta[:, :2].norm(dim=-1)
            right_horiz = right_delta[:, :2].norm(dim=-1)
            left_vert = left_delta[:, 2]
            right_vert = right_delta[:, 2]
            left_align, right_align = env_u._palm_down_alignments()
            left_twist, right_twist = env_u._palm_twist_errors()
            grasp_valid = env_u._grasp_pose_valid_mask()

            standoff_hist.append(signed_standoff.detach().cpu().numpy())
            stage_hist.append(env_u.stage_buf.detach().cpu().numpy().copy())
            left_dist_hist.append(left_dist.detach().cpu().numpy())
            right_dist_hist.append(right_dist.detach().cpu().numpy())
            left_horiz_hist.append(left_horiz.detach().cpu().numpy())
            right_horiz_hist.append(right_horiz.detach().cpu().numpy())
            left_vert_hist.append(left_vert.detach().cpu().numpy())
            right_vert_hist.append(right_vert.detach().cpu().numpy())
            left_align_hist.append(left_align.detach().cpu().numpy())
            right_align_hist.append(right_align.detach().cpu().numpy())
            left_twist_hist.append(left_twist.detach().cpu().numpy())
            right_twist_hist.append(right_twist.detach().cpu().numpy())
            grasp_valid_hist.append(grasp_valid.detach().cpu().numpy())

            if step % 50 == 0:
                print(f"[measure] step={step} mean_standoff={signed_standoff.mean().item():.4f} "
                      f"stage_counts={dict(zip(*__import__('numpy').unique(env_u.stage_buf.cpu().numpy(), return_counts=True)))}")

    import numpy as np
    standoff = np.stack(standoff_hist)   # (T, N)
    stage = np.stack(stage_hist)         # (T, N)
    left_dist = np.stack(left_dist_hist)
    right_dist = np.stack(right_dist_hist)
    left_horiz = np.stack(left_horiz_hist)
    right_horiz = np.stack(right_horiz_hist)
    left_vert = np.stack(left_vert_hist)
    right_vert = np.stack(right_vert_hist)
    left_align = np.stack(left_align_hist)
    right_align = np.stack(right_align_hist)
    left_twist = np.stack(left_twist_hist)
    right_twist = np.stack(right_twist_hist)
    grasp_valid = np.stack(grasp_valid_hist)

    tail = slice(max(0, play_steps - 100), play_steps)

    print()
    print("==================== STANDOFF MEASUREMENT ====================")
    print(f"num_envs={config.num_envs}  play_steps={play_steps}  checkpoint={checkpoint.name}")
    print(f"stop_standoff config target = {env_u.config.get('stop_standoff', 'N/A')}")
    print(f"min_grasp_standoff config floor = {env_u.config.get('min_grasp_standoff', 'N/A')}")
    print()
    print(f"OVERALL mean standoff (all steps, all envs): {standoff.mean():.4f} m  (std={standoff.std():.4f})")
    print(f"TAIL (last 100 steps) mean standoff, all envs: {standoff[tail].mean():.4f} m (std={standoff[tail].std():.4f})")
    print()
    for s in range(4):
        mask = stage == s
        if mask.sum() == 0:
            print(f"stage {s}: no samples")
            continue
        print(f"stage {s}: n={mask.sum()}  mean_standoff={standoff[mask].mean():.4f}  "
              f"mean_left_dist={left_dist[mask].mean():.4f}  mean_right_dist={right_dist[mask].mean():.4f}  "
              f"grasp_valid_frac={grasp_valid[mask].mean():.4f}")
        print(f"          left:  horiz={left_horiz[mask].mean():.4f}  vert={left_vert[mask].mean():+.4f}  "
              f"align={left_align[mask].mean():+.4f}  twist_rad={left_twist[mask].mean():.4f}  "
              f"frac_close(<0.10m)={(left_dist[mask]<0.10).mean():.4f}  "
              f"frac_aligned(>0.5)={(left_align[mask]>0.5).mean():.4f}")
        print(f"          right: horiz={right_horiz[mask].mean():.4f}  vert={right_vert[mask].mean():+.4f}  "
              f"align={right_align[mask].mean():+.4f}  twist_rad={right_twist[mask].mean():.4f}  "
              f"frac_close(<0.10m)={(right_dist[mask]<0.10).mean():.4f}  "
              f"frac_aligned(>0.5)={(right_align[mask]>0.5).mean():.4f}")
    print()
    print(f"Fraction of (step,env) with stage>=1 (REACH+): {(stage>=1).mean():.4f}")
    print(f"Fraction of (step,env) with stage>=2 (GRASP+): {(stage>=2).mean():.4f}")
    print(f"Fraction of (step,env) with grasp_valid=True: {grasp_valid.mean():.4f}")
    print()
    print("Per-hand close+aligned breakdown (stage>=2 only):")
    s2 = stage >= 2
    if s2.sum() > 0:
        l_close = left_dist[s2] < 0.10
        r_close = right_dist[s2] < 0.10
        l_aligned = left_align[s2] > 0.5
        r_aligned = right_align[s2] > 0.5
        print(f"  left  close&aligned: {(l_close & l_aligned).mean():.4f}   close_only: {(l_close & ~l_aligned).mean():.4f}   aligned_only: {(~l_close & l_aligned).mean():.4f}   neither: {(~l_close & ~l_aligned).mean():.4f}")
        print(f"  right close&aligned: {(r_close & r_aligned).mean():.4f}   close_only: {(r_close & ~r_aligned).mean():.4f}   aligned_only: {(~r_close & r_aligned).mean():.4f}   neither: {(~r_close & ~r_aligned).mean():.4f}")
        print()
        print("Per-hand VERTICAL-hovering breakdown (stage>=2 only) — diagnoses")
        print("'closing fingers above the true target instead of on it':")
        l_horiz_ok = left_horiz[s2] < 0.04
        r_horiz_ok = right_horiz[s2] < 0.04
        l_vert_ok = np.abs(left_vert[s2]) < 0.03
        r_vert_ok = np.abs(right_vert[s2]) < 0.03
        l_hover = l_horiz_ok & ~l_vert_ok & (left_vert[s2] > 0)
        r_hover = r_horiz_ok & ~r_vert_ok & (right_vert[s2] > 0)
        print(f"  left  horiz_ok(<4cm)={l_horiz_ok.mean():.4f}  vert_ok(<3cm)={l_vert_ok.mean():.4f}  "
              f"HOVERING_ABOVE(horiz_ok&NOT vert_ok&above)={l_hover.mean():.4f}  mean_vert_when_horiz_ok={left_vert[s2][l_horiz_ok].mean() if l_horiz_ok.any() else float('nan'):+.4f}")
        print(f"  right horiz_ok(<4cm)={r_horiz_ok.mean():.4f}  vert_ok(<3cm)={r_vert_ok.mean():.4f}  "
              f"HOVERING_ABOVE(horiz_ok&NOT vert_ok&above)={r_hover.mean():.4f}  mean_vert_when_horiz_ok={right_vert[s2][r_horiz_ok].mean() if r_horiz_ok.any() else float('nan'):+.4f}")

    print()
    print("==================== TWIST/POSITION TRADE-OFF (GRASP stage only) ====================")
    stage2 = stage == 2
    n2 = int(stage2.sum())
    print(f"GRASP-stage samples (step,env pairs): n={n2}")
    if n2 >= 2:
        l_twist2 = left_twist[stage2]
        r_twist2 = right_twist[stage2]
        l_horiz2 = left_horiz[stage2]
        r_horiz2 = right_horiz[stage2]
        l_vert2 = np.abs(left_vert[stage2])
        r_vert2 = np.abs(right_vert[stage2])

        def _corr(a, b):
            if np.std(a) < 1e-9 or np.std(b) < 1e-9:
                return float("nan")
            return float(np.corrcoef(a, b)[0, 1])

        print(f"  LEFT  corr(twist_rad, horiz)={_corr(l_twist2, l_horiz2):+.4f}   "
              f"corr(twist_rad, vert)={_corr(l_twist2, l_vert2):+.4f}")
        print(f"  RIGHT corr(twist_rad, horiz)={_corr(r_twist2, r_horiz2):+.4f}   "
              f"corr(twist_rad, vert)={_corr(r_twist2, r_vert2):+.4f}")

        # Bucket by horiz-distance quartile and report mean twist per bucket —
        # a monotonic trend (closer horiz => lower or higher twist) is direct
        # evidence for/against a kinematic position/orientation trade-off.
        for side, twist2, horiz2 in [("LEFT", l_twist2, l_horiz2), ("RIGHT", r_twist2, r_horiz2)]:
            order = np.argsort(horiz2)
            twist_sorted = twist2[order]
            horiz_sorted = horiz2[order]
            nq = 4
            edges = np.array_split(np.arange(len(horiz_sorted)), nq)
            print(f"  {side} horiz-quartile breakdown (closest -> farthest):")
            for qi, idx in enumerate(edges):
                if len(idx) == 0:
                    continue
                print(f"    Q{qi+1}: n={len(idx)}  mean_horiz={horiz_sorted[idx].mean():.4f}m  "
                      f"mean_twist={np.degrees(twist_sorted[idx].mean()):.1f}deg")
    else:
        print("  Not enough GRASP-stage samples to compute correlation.")
    print("================================================================")

    os._exit(0)


if __name__ == "__main__":
    main()
