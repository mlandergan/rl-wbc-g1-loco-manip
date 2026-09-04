# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Evaluation script for trained RL agents (teacher or student).

Loads a checkpoint and its associated training config, sets up the environment
and model, optionally exports the policy as ONNX, then runs evaluation episodes.

Usage:
    python groot/rl/eval_agent_trl.py +checkpoint=<path_to_checkpoint.pt> [+num_envs=1]
"""

import logging
import os
import shutil
import sys
from pathlib import Path

import hydra
import yaml
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from loguru import logger
from omegaconf import OmegaConf

from gr00t.rl.utils.config_utils import register_rl_resolvers

register_rl_resolvers()

_CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "gr00t/rl/config")


def process_output_dim_in_config(config):
    """Process and adapt output dimensions for actor and teacher_actor backbones.

    When output_dim is set to -1 in the config, this function auto-calculates
    the correct dimension based on the homie command keys.
    """

    def calculate_homie_output_dim():
        output_dim = 0
        for key in config.obs["homie_command_keys"].keys():
            output_dim += len(config.obs["homie_command_default"][key])
        return output_dim

    def adapt_backbone_output_dim(backbone_config, config_name=""):
        try:
            if hasattr(backbone_config, "module_config_dict"):
                if backbone_config.module_config_dict.output_dim[0] == -1:
                    output_dim = calculate_homie_output_dim()
                    backbone_config.module_config_dict.output_dim = [output_dim]
                    return True
            elif hasattr(backbone_config, "mlp_module") and hasattr(
                backbone_config.mlp_module, "module_config_dict"
            ):
                if backbone_config.mlp_module.module_config_dict.output_dim[0] == -1:
                    output_dim = calculate_homie_output_dim()
                    backbone_config.mlp_module.module_config_dict.output_dim = [output_dim]
                    return True
        except (AttributeError, IndexError) as e:
            logger.warning(f"Could not adapt {config_name} backbone output_dim: {e}")
        return False

    if (
        config.algo.config.get("use_new_actor_critic", False)
        and hasattr(config.algo.config, "actor")
        and hasattr(config.algo.config.actor, "backbone")
    ):
        adapt_backbone_output_dim(config.algo.config.actor.backbone, "actor")

        if (
            getattr(config.algo.config, "use_dagger", False)
            and hasattr(config.algo.config, "teacher_actor")
            and hasattr(config.algo.config.teacher_actor, "backbone")
        ):
            adapt_backbone_output_dim(config.algo.config.teacher_actor.backbone, "teacher_actor")


@hydra.main(
    config_path=_CONFIG_DIR,
    config_name="base_eval",
)
def main(override_config: OmegaConf):
    # --- Logging setup ---
    hydra_log_path = os.path.join(HydraConfig.get().runtime.output_dir, "eval.log")
    logger.remove()
    logger.add(hydra_log_path, level="DEBUG")
    console_log_level = os.environ.get("LOGURU_LEVEL", "INFO").upper()
    logger.add(sys.stdout, level=console_log_level, colorize=True)

    from gr00t.rl.utils.logging import HydraLoggerBridge

    logging.basicConfig(level=logging.INFO)
    # TorchAudio probes optional FFmpeg ABIs at DEBUG level. Forwarding those
    # exception records through Loguru recursively can produce unbounded logs.
    logging.getLogger("torio").setLevel(logging.CRITICAL)
    logging.getLogger().addHandler(HydraLoggerBridge())
    os.chdir(hydra.utils.get_original_cwd())

    # --- Load and merge training config from checkpoint directory ---
    if override_config.checkpoint is not None:
        has_config = True
        checkpoint = Path(override_config.checkpoint)
        config_path = checkpoint.parent / "config.yaml"
        if not config_path.exists():
            config_path = checkpoint.parent.parent / "config.yaml"
            if not config_path.exists():
                has_config = False
                logger.error(f"Could not find config path: {config_path}")

        if has_config:
            logger.info(f"Loading training config file from {config_path}")
            with open(config_path) as file:
                train_config = OmegaConf.load(file)

            if train_config.eval_overrides is not None:
                train_config = OmegaConf.merge(train_config, train_config.eval_overrides)

            config = OmegaConf.merge(train_config, override_config)
        else:
            config = override_config

        config.experiment_dir = checkpoint.parent
    else:
        if override_config.eval_overrides is not None:
            config = override_config.copy()
            eval_overrides = OmegaConf.to_container(config.eval_overrides, resolve=True)
            for arg in sys.argv[1:]:
                if not arg.startswith("+"):
                    key = arg.split("=")[0]
                    if key in eval_overrides:
                        del eval_overrides[key]
            config.eval_overrides = OmegaConf.create(eval_overrides)
            config = OmegaConf.merge(config, eval_overrides)
        else:
            config = override_config

    # Resume wandb run if meta.yaml exists
    meta_path = Path(config.experiment_dir) / "meta.yaml"
    if meta_path.exists():
        meta = yaml.safe_load(open(meta_path, "r"))
        config.wandb.wandb_id = meta["wandb_run"]
        print(f"resume wandb from run: {config.wandb.wandb_id}")

    # --- Setup Isaac Sim ---
    simulator_type = config.simulator["_target_"].split(".")[-1]
    if simulator_type == "IsaacSim":
        try:
            with open("./rl/simulator/isaacsim/.isaacsim_version", "r", encoding="utf-8") as f:
                DEFAULT_ISAACSIM_VERSION = f.read().strip()
        except FileNotFoundError:
            DEFAULT_ISAACSIM_VERSION = "4.5"

        if DEFAULT_ISAACSIM_VERSION == "4.5":
            from isaaclab.app import AppLauncher
        elif DEFAULT_ISAACSIM_VERSION == "4.2":
            logger.warning("Using IsaacSim 4.2, replacing isaaclab with omni.isaac.lab")
            from omni.isaac.lab.app import AppLauncher

        import argparse

        import isaaclab

        parser = argparse.ArgumentParser(description="Evaluate an RL agent.")
        AppLauncher.add_app_launcher_args(parser)

        args_cli, hydra_args = parser.parse_known_args()
        sys.argv = [sys.argv[0]] + hydra_args
        args_cli.num_envs = config.num_envs
        args_cli.seed = config.seed
        args_cli.env_spacing = config.env.config.env_spacing
        args_cli.output_dir = config.output_dir
        args_cli.enable_cameras = (
            config.simulator.config.cameras.enable_cameras
            or config.simulator.config.get("render_results", False)
        )
        args_cli.headless = config.headless

        app_launcher = AppLauncher(args_cli)
        simulation_app = app_launcher.app

    # --- Imports that must come after Isaac Sim initialization ---
    from accelerate import Accelerator
    from transformers import HfArgumentParser
    from trl import ModelConfig, PPOConfig, ScriptArguments

    from gr00t.rl.agents.modules.ppo_modules import (
        PPOCritic,
        PPOStateActor,
        PPOStateActorFixSigma,
    )
    from gr00t.rl.trl.utils.common import custom_instantiate
    from gr00t.rl.utils.common import seeding
    from gr00t.rl.utils.helpers import pre_process_config

    # --- Config processing ---
    unresolved_conf = OmegaConf.to_container(config, resolve=False)
    os.chdir(hydra.utils.get_original_cwd())
    pre_process_config(config)

    # Set rendering output directory
    ckpt_num = config.checkpoint.split("/")[-1].split("_")[-1].split(".")[0]
    if config.env.config.get("save_rendering_dir", None) is None:
        if config.simulator.config.get("render_results", False):
            from datetime import datetime

            parent_dir = checkpoint.parent.name
            config.env.config.save_rendering_dir = str(
                Path("logs_rl") / "renderings" / datetime.now().strftime("%Y-%m-%d") / parent_dir
            )
        else:
            config.env.config.save_rendering_dir = str(
                checkpoint.parent / "renderings" / f"ckpt_{ckpt_num}"
            )
    config.env.config.ckpt_dir = str(checkpoint.parent)

    # --- Setup TRL training args (reused for eval) ---
    config.algo.trl.output_dir = str(Path(config.experiment_dir))
    parser = HfArgumentParser((ScriptArguments, PPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_dict(config.algo.trl)
    if "eval_output_dir" in config:
        training_args.eval_output_dir = config.eval_output_dir

    ref_model = None
    value_model = None

    # --- Accelerator and seeding ---
    accelerator = Accelerator()
    device = str(accelerator.device)
    if device == "cuda":
        device = "cuda:0"
    config.multi_gpu = accelerator.num_processes > 1
    if config.multi_gpu:
        config.global_rank = accelerator.process_index
        config.seed += accelerator.process_index
        config.algo.config.global_rank = accelerator.process_index
        config.algo.config.world_size = accelerator.num_processes
    seeding(config.seed)

    # --- Create environment ---
    env = instantiate(config=config.env, device=device)

    # --- Build policy and value models ---
    process_output_dim_in_config(config)

    if config.algo.config.get("use_new_actor_critic", False):
        module_dim_dict = getattr(config.algo.config, "module_dim", {})
        policy = instantiate(
            config.algo.config.actor,
            env_config=env.config,
            algo_config=config.algo.config,
            module_dim_dict=module_dim_dict,
            _recursive_=False,
        ).to(device)
        if getattr(config.algo.config, "use_dagger", False):
            ref_model = instantiate(
                config.algo.config.teacher_actor,
                env_config=env.config,
                algo_config=config.algo.config,
                module_dim_dict=module_dim_dict,
                _recursive_=False,
                input_key="teacher_obs",
            ).to(device)
        if not getattr(config.algo.config, "distill_only", False) and hasattr(
            config.algo.config, "critic"
        ):
            value_model = instantiate(
                config.algo.config.critic,
                env_config=env.config,
                algo_config=config.algo.config,
                module_dim_dict=module_dim_dict,
                _recursive_=False,
            ).to(device)
    else:
        if getattr(config.algo.config, "use_dagger", False):
            module_dim_dict = getattr(config.algo.config, "module_dim", {})
            policy = PPOStateActorFixSigma(
                obs_dim_dict=env.config.robot.algo_obs_dim_dict,
                module_config_dict=config.algo.config.module_dict.actor,
                num_actions=env.config.robot.actions_dim,
                module_dim_dict=module_dim_dict,
            ).to(device)
            ref_model = PPOStateActorFixSigma(
                obs_dim_dict=env.config.robot.algo_obs_dim_dict,
                module_config_dict=config.algo.config.module_dict.teacher_actor,
                num_actions=env.config.robot.actions_dim,
                module_dim_dict=module_dim_dict,
            ).to(device)
        else:
            policy = PPOStateActor(
                obs_dim_dict=env.config.robot.algo_obs_dim_dict,
                module_config_dict=config.algo.config.module_dict.actor,
                num_actions=env.config.robot.actions_dim,
                input_key="actor_obs",
                init_noise_std=config.algo.config.init_noise_std,
            ).to(device)
            value_model = PPOCritic(
                env.config.robot.algo_obs_dim_dict,
                config.algo.config.module_dict.critic,
            ).to(device)

    accelerator.wait_for_everyone()

    # --- Callbacks ---
    callbacks = []
    for callback in config.callbacks.values():
        callbacks.append(instantiate(callback))

    # --- Build trainer (loads checkpoint weights) ---
    trainer = custom_instantiate(
        config.trainer,
        args=training_args,
        config=config.algo.config,
        env=env,
        model=policy,
        ref_model=ref_model,
        use_ref_model=getattr(config.algo.config, "use_dagger", False),
        value_model=value_model,
        train_dataset=None,
        eval_dataset=None,
        callbacks=callbacks,
        checkpoint=config.checkpoint,
        local_seed=config.seed,
        accelerator=accelerator,
    )

    # --- Optional ONNX export (only with single env) ---
    EXPORT_ONNX = config.num_envs == 1
    checkpoint_path = str(checkpoint)

    exported_policy_path = os.path.join(config.experiment_dir, "exported")
    os.makedirs(exported_policy_path, exist_ok=True)
    exported_onnx_name = f"model_step_{trainer.state.global_step:06d}.onnx"
    new_cp_path = (
        f"{os.path.dirname(config.checkpoint)}/model_step_{trainer.state.global_step:06d}.pt"
    )

    if not os.path.exists(new_cp_path):
        shutil.copy(checkpoint_path, new_cp_path)

    if EXPORT_ONNX:
        assert config.num_envs == 1, "num_envs must be 1 for exporting onnx"
        from gr00t.rl.utils.inference_helpers import (
            export_policy_as_onnx,
            export_policy_CNN_as_onnx,
            export_policy_CNN_as_onnx_with_obj_pred,
            export_policy_z_as_onnx,
        )

        algo = trainer
        example_obs_dict = algo.get_example_obs()

        if config.env.config.get("use_z", False):
            export_policy_z_as_onnx(
                algo.inference_model,
                env,
                exported_policy_path,
                exported_onnx_name,
                example_obs_dict,
            )
        elif config.simulator.config.cameras.enable_cameras:
            if hasattr(config.algo.config.actor.backbone, "obj_pred_mlp"):
                export_policy_CNN_as_onnx_with_obj_pred(
                    algo.inference_model,
                    env,
                    exported_policy_path,
                    exported_onnx_name,
                    example_obs_dict,
                )
            else:
                export_policy_CNN_as_onnx(
                    algo.inference_model,
                    env,
                    exported_policy_path,
                    exported_onnx_name,
                    example_obs_dict,
                )
        else:
            export_policy_as_onnx(
                algo.inference_model, exported_policy_path, exported_onnx_name, example_obs_dict
            )
        logger.info(
            f"Exported policy as onnx to: {os.path.join(exported_policy_path, exported_onnx_name)}"
        )

    if config.get("export_onnx_only", False):
        exit()

    # --- PASS 1: policy rollout, LOG per-step world poses (no rendering) ---
    # In-env camera capture is a confirmed dead-end in this build (TiledCamera
    # ignores pose; raw cam needs orchestrator which deadlocks). So here we
    # just run the trained policy and record the world pose of every G1 body +
    # every bin body each step to a .npz. A separate replay script
    # (move_garbage/scripts/replay_traj.py) then reconstructs the scene
    # kinematically and records the video with the proven vidtest_d camera.
    import numpy as np
    from gr00t.rl.trl.modules.homie_modules import init_actor_critic_dict as _iacd
    import torch as _torch

    # NOTE: 400 steps was ending before the G1 reached the bin. Two factors:
    #   (1) not enough steps for the full approach->contact->push arc, and
    #   (2) this loop ignores `dones` — if the episode hits its episode_length_s
    #       cap mid-rollout the env auto-resets and the robot teleports to start.
    # Default bumped to 800; raise episode_length_s (or pass ++play_steps=N) if
    # the tail still gets cut. See the step-loop below for the dones diagnostic.
    play_steps = int(config.get("play_steps", 800))
    traj_path = config.get("traj_out", "/tmp/bin_traj.npz")
    logger.info(f"[pass1] policy rollout for {play_steps} steps -> {traj_path}")

    env_u = env
    device_t = env_u.device

    model = trainer.model
    policy_model = model.policy
    homie_walk_model = model.homie_walk_model
    homie_stand_model = model.homie_stand_model
    homie_switch_threshold = getattr(trainer, "homie_switch_threshold", 0.5)
    homie_one_step_obs = _iacd["num_one_step_obs"]

    # Freeze running statistics and initialize recurrent state before inference.
    # Without this, a one-env rollout updates RunningMeanStd with an unbiased
    # variance of NaN and corrupts every action after the first step.
    trainer._eval_mode()
    policy_model.init_rollout()

    obs_dict = env_u.reset_all()
    for k in obs_dict:
        obs_dict[k] = obs_dict[k].to(device_t)

    robot = env_u.simulator._robot
    bin_art = env_u.simulator._task["bin"]
    robot_body_names = list(robot.data.body_names)
    bin_body_names = list(bin_art.data.body_names)
    logger.info(f"[pass1] robot bodies={len(robot_body_names)} bin bodies={len(bin_body_names)}")

    robot_pos_frames, robot_quat_frames = [], []
    bin_pos_frames, bin_quat_frames = [], []
    # Also log root pose + joint positions for clean kinematic articulation replay.
    robot_root_frames, robot_jpos_frames = [], []
    bin_root_frames, bin_jpos_frames = [], []
    stage_frames, grasp_valid_frames, done_frames = [], [], []
    palm_anchor_dist_frames, palm_down_frames, hand_contact_frames = [], [], []
    bin_progress_frames = []
    robot_joint_names = list(robot.data.joint_names)
    bin_joint_names = list(bin_art.data.joint_names)

    with _torch.no_grad():
        for step in range(play_steps):
            policy_model.rollout(obs_dict=obs_dict)
            action_mean = policy_model.action_mean.detach()
            homie_obs = obs_dict["homie_obs"]
            walk_out = homie_walk_model(homie_obs)
            stand_out = homie_stand_model(homie_obs)
            commands = homie_obs[..., -homie_one_step_obs:-(homie_one_step_obs - 3)]
            walk_mask = _torch.norm(commands, dim=-1, keepdim=True) > homie_switch_threshold
            m = walk_mask
            while m.dim() < walk_out["actions"].dim():
                m = m.unsqueeze(-1)
            homie_actions = _torch.where(m, walk_out["action_mean"], stand_out["action_mean"])
            step_actions = _torch.cat([action_mean, homie_actions], dim=-1)

            obs_dict, rewards, dones, infos = env_u.step({"actions": step_actions})
            for k in obs_dict:
                obs_dict[k] = obs_dict[k].to(device_t)

            # Log env-0 world poses (wxyz quats as Isaac Lab stores them).
            robot_pos_frames.append(robot.data.body_pos_w[0].detach().cpu().numpy())
            robot_quat_frames.append(robot.data.body_quat_w[0].detach().cpu().numpy())
            bin_pos_frames.append(bin_art.data.body_pos_w[0].detach().cpu().numpy())
            bin_quat_frames.append(bin_art.data.body_quat_w[0].detach().cpu().numpy())
            # Root pose (pos + wxyz quat) + joint positions for kinematic replay.
            robot_root_frames.append(robot.data.root_state_w[0, :7].detach().cpu().numpy())
            robot_jpos_frames.append(robot.data.joint_pos[0].detach().cpu().numpy())
            bin_root_frames.append(bin_art.data.root_state_w[0, :7].detach().cpu().numpy())
            bin_jpos_frames.append(bin_art.data.joint_pos[0].detach().cpu().numpy())

            left_anchor, right_anchor = env_u._palm_grasp_points()
            left_align, right_align = env_u._palm_down_alignments()
            # 2026-08-29: renamed from _hand_contact_counts() (removed) --
            # now returns per-hand palm-to-bar-target DISTANCES, not PhysX
            # contact-link counts. See bin_push_homie.py's
            # _hand_bar_distances docstring for the full history.
            left_bar_dist, right_bar_dist = env_u._hand_bar_distances()
            anchor_dist = _torch.stack(
                [
                    (env_u._left_handle_target() - left_anchor).norm(dim=-1),
                    (env_u._right_handle_target() - right_anchor).norm(dim=-1),
                ],
                dim=-1,
            )
            stage_frames.append(int(env_u.stage_buf[0].item()))
            done_frames.append(bool(dones[0].item()))
            grasp_valid_frames.append(bool(env_u._grasp_pose_valid_mask()[0].item()))
            palm_anchor_dist_frames.append(anchor_dist[0].detach().cpu().numpy())
            palm_down_frames.append(
                _torch.stack([left_align[0], right_align[0]]).detach().cpu().numpy()
            )
            hand_contact_frames.append(
                _torch.stack([left_bar_dist[0], right_bar_dist[0]]).detach().cpu().numpy()
            )
            bin_progress_frames.append(float(env_u._bin_push_progress()[0].item()))

            # Surface env-0 resets: a done mid-rollout means the episode ended
            # (length cap / termination) and the robot teleported to start, so
            # the logged tail is a *new* episode, not continued approach->push.
            if bool(dones[0].item()):
                print(f"[pass1] !! env-0 DONE at step={step} — trajectory reset "
                      f"here (raise episode_length_s to log a full approach->push)")

            if step % 50 == 0:
                bp = bin_art.data.body_pos_w[0, 0].tolist()
                rz = robot.data.body_pos_w[0, 0, 2].item()
                print(f"[pass1] step={step} bin=({bp[0]:.2f},{bp[1]:.2f},{bp[2]:.2f}) robot_z={rz:.2f}")

    np.savez_compressed(
        traj_path,
        robot_pos=np.stack(robot_pos_frames),      # (T, nB_robot, 3)
        robot_quat=np.stack(robot_quat_frames),    # (T, nB_robot, 4) wxyz
        bin_pos=np.stack(bin_pos_frames),          # (T, nB_bin, 3)
        bin_quat=np.stack(bin_quat_frames),        # (T, nB_bin, 4) wxyz
        robot_body_names=np.array(robot_body_names),
        bin_body_names=np.array(bin_body_names),
        robot_root=np.stack(robot_root_frames),    # (T, 7) pos+wxyz
        robot_jpos=np.stack(robot_jpos_frames),    # (T, nJ_robot)
        bin_root=np.stack(bin_root_frames),        # (T, 7)
        bin_jpos=np.stack(bin_jpos_frames),        # (T, nJ_bin)
        robot_joint_names=np.array(robot_joint_names),
        bin_joint_names=np.array(bin_joint_names),
        stage=np.asarray(stage_frames, dtype=np.int64),
        done=np.asarray(done_frames, dtype=np.bool_),
        grasp_valid=np.asarray(grasp_valid_frames, dtype=np.bool_),
        palm_anchor_dist=np.stack(palm_anchor_dist_frames),
        palm_down_alignment=np.stack(palm_down_frames),
        # 2026-08-29: renamed from hand_contact_count -- now per-hand
        # palm-to-bar-target distance (meters), not a PhysX contact-link
        # count. See _hand_bar_distances in bin_push_homie.py.
        hand_bar_dist=np.stack(hand_contact_frames),
        bin_push_progress=np.asarray(bin_progress_frames, dtype=np.float32),
    )
    stage_np = np.asarray(stage_frames)
    valid_np = np.asarray(grasp_valid_frames)
    dist_np = np.stack(palm_anchor_dist_frames)
    align_np = np.stack(palm_down_frames)
    print(f"[pass1] wrote {traj_path}  ({len(robot_pos_frames)} frames)")
    print(
        "[pass1] diagnostics: "
        f"stage_counts={np.bincount(stage_np, minlength=4).tolist()} "
        f"valid_grasp_frames={int(valid_np.sum())} "
        f"best_anchor_dist=({dist_np[:, 0].min():.4f}, {dist_np[:, 1].min():.4f}) "
        f"mean_palm_down=({align_np[:, 0].mean():.3f}, {align_np[:, 1].mean():.3f}) "
        f"max_push_progress={max(bin_progress_frames):.4f}"
    )
    logger.info("Finished pass-1 trajectory logging")
    os._exit(0)


if __name__ == "__main__":
    main()
