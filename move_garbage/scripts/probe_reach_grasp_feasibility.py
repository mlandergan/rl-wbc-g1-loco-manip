# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Settle the real robot into REACH, drive both hands through the pregrasp arc
and final grasp pose with iterative IK, and log the real BinPushHomie reward
terms + true joint-limit status at each keyframe -- a direct test of whether
the authored twist/grasp targets are kinematically reachable at all, decoupled
from whether the RL policy has learned to reach them.

Usage (mirrors measure_standoff.py / play_bin_push.py):
    CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n isaacsim5.1 python -u \\
        move_garbage/scripts/probe_reach_grasp_feasibility.py --headless \\
        +checkpoint="$PWD/move_garbage/checkpoints/model_step_012900.pt" \\
        ++num_envs=1 ++opt.use_wandb=False \\
        'hydra.searchpath=[file://<repo>/move_garbage/config]'

The checkpoint is used ONLY to load config + the frozen HOMIE walk/stand
models (see README: "frozen HOMIE policies provide lower-body walking and
standing control") -- the checkpoint's own upper-body arm policy is never
queried; every arm action this script issues comes from IK against the
env's own authored targets instead. HOMIE still runs every physics step
through the env's real step() pipeline, exactly as in a trained rollout,
so the robot is genuinely balanced/standing throughout, not just kinematically
teleported.

Render the captured trajectory at 1x with (see README "Replay at simulation
speed"):
    python -u move_garbage/scripts/replay_traj.py --headless --enable_cameras \\
        --traj /tmp/probe_traj.npz --out /tmp/probe_rollout_1x.mp4 \\
        --fps 25 --stride 2
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np


# ============================================================================
# Pure numpy helpers -- zero isaaclab/torch/hydra dependency on purpose, so
# they can be imported and unit-tested without Isaac Sim (verified standalone
# against a set of closed-form assertions: hump(0)=hump(blend_dist)=0,
# hump(peak_frac*blend_dist)=1, waypoint frac=0/1 endpoints, grasp-point/
# palm-pose round trip, delta-action round trip + clip). Isaac-Sim-side code
# below calls these on plain .cpu().numpy() arrays per env/hand.
# ============================================================================

def _quat_apply_wxyz(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate 3-vector v by quaternion q (wxyz). Verbatim port of the helper
    already validated in replay_traj.py (same formula, reused rather than
    re-derived to avoid a second, possibly-inconsistent implementation)."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    qvec = np.array([x, y, z])
    cross1 = np.cross(qvec, v)
    cross2 = np.cross(qvec, cross1)
    return v + 2.0 * w * cross1 + 2.0 * cross2


def _quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product q1*q2, both wxyz. Verbatim port from replay_traj.py."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def _slerp_wxyz(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical-linear-interpolate two wxyz quaternions, shortest path."""
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        result = q0 + t * (q1 - q0)
        return result / np.linalg.norm(result)
    theta_0 = math.acos(max(-1.0, min(1.0, dot)))
    theta = theta_0 * t
    q2 = q1 - q0 * dot
    q2 = q2 / np.linalg.norm(q2)
    return q0 * math.cos(theta) + q2 * math.sin(theta)


def _arc_hump(horiz: float, blend_dist: float, peak_frac: float) -> float:
    """Smooth (half-cosine) hump in horiz-space: 0 at horiz=0 and
    horiz>=blend_dist, 1 at horiz=peak_frac*blend_dist, zero slope at both
    ends and at the peak. EXACT port of _one_hand_reach's Run 20 pregrasp arc
    (move_garbage/envs/bin_push_homie.py) -- also already ported once before
    as replay_traj.py's `_arc_hump`; this is a third, independently-checked
    copy of the same formula, not a re-derivation from scratch."""
    s = np.clip(horiz / blend_dist, 0.0, 1.0)
    rising_phase = np.clip(s / peak_frac, 0.0, 1.0)
    falling_phase = np.clip((1.0 - s) / (1.0 - peak_frac), 0.0, 1.0)
    phase = np.where(s < peak_frac, rising_phase, falling_phase)
    return 0.5 * (1.0 - np.cos(np.pi * phase))


def _pregrasp_waypoint_pose(start_pos, start_quat, target_pos, target_quat,
                             frac, blend_dist, peak_frac, arc_height):
    """One point along the synthetic pregrasp arc, frac in [0,1] (0=start
    pose, 1=target). XY (and orientation) lerp/slerp linearly start->target;
    Z follows the same triangular hump _one_hand_reach rewards, using
    horiz(frac) = start_horiz*(1-frac) since XY moves linearly and the target
    itself has horiz=0 -- mirrors replay_traj.py's --show_full_arc
    `_sweep_points`, generalized to also carry an orientation slerp (needed
    here to command IK, not just to draw a debug trail)."""
    pos = start_pos + (target_pos - start_pos) * frac
    start_horiz = float(np.linalg.norm(start_pos[:2] - target_pos[:2]))
    horiz = start_horiz * (1.0 - frac)
    hump = _arc_hump(horiz, blend_dist, peak_frac)
    pos = pos.copy()
    pos[2] = target_pos[2] + arc_height * hump
    quat = _slerp_wxyz(start_quat, target_quat, frac)
    return pos, quat


def _grasp_point_to_palm_pose(grasp_point_pos, grasp_quat_wxyz, axis_xyz, offset):
    """Invert _palm_grasp_points()'s virtual-grasp-point offset to recover the
    PALM-LINK pose IK must actually achieve. Mirrors show_gripper_targets.py's
    `palm_position = target_position - rotate(target_quaternion, axis*offset)`.
    Palm orientation target == grasp orientation target directly (no extra
    rotation): confirmed against _palm_twist_errors, which compares
    quat_error_magnitude(palm_rot, handle_target_rot) with no offset applied."""
    palm_pos = grasp_point_pos - offset * _quat_apply_wxyz(grasp_quat_wxyz, axis_xyz)
    return palm_pos, grasp_quat_wxyz


def _delta_action_for_target(q_target, q_default, action_scale, clip):
    """Convert an absolute joint target into this env's accumulated
    delta-action value. FingerPrimitiveBase._apply_force_in_physics_step
    computes combined_target = accumulated_delta * action_scale +
    default_dof_pos, so accumulated_delta = (q_target - default_dof_pos) /
    action_scale is the exact (not approximate) value that makes the low-level
    PD target equal q_target. Writing this straight into env._delta_actions
    (see _set_integrated_finger_action in bin_push_homie.py for the same
    idiom used by the env itself) and passing a raw per-step action of 0 at
    that column leaves it there unperturbed by DeltaActionBase's `+=`."""
    return np.clip((q_target - q_default) / action_scale, -clip, clip)


# ============================================================================
# Isaac Sim / hydra entry point -- imports below this point are intentionally
# deferred into main() (mirroring measure_standoff.py), so the pure helpers
# above stay importable without Isaac Sim / hydra / torch installed.
# ============================================================================

import hydra  # noqa: E402
from hydra.core.hydra_config import HydraConfig  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from gr00t.rl.utils.config_utils import register_rl_resolvers  # noqa: E402

register_rl_resolvers()

_CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "gr00t/rl/config")

# Reward terms this probe reports, in keyframe-relevant order. Pulled from
# reward_bin_push_homie.yaml's `reward_scales` keys (the source of truth the
# task asked for) rather than hardcoded a second time; any term without a
# matching `_reward_<name>` method on the env (e.g. terms handled by base
# classes under a different name) is skipped with a warning at runtime.
_REACH_GRASP_REWARD_TERMS = [
    "stop_at_bin", "maintain_standoff", "penalty_too_close_to_bin",
    "reach_left", "reach_right", "reach_finger_dof_pos_l1",
    "penalty_not_standing_still",
    "palm_down_alignment", "palm_twist_alignment", "palm_twist_progress",
    "grasp_finger_dof_pos_l1", "grasp_target_distance",
    "grasp_target_anisotropic", "grasp_vert_progress", "grasp_contact",
    "penalty_wrong_twist_contact",
    "penalty_humanly_dof_limit",
]


@hydra.main(config_path=_CONFIG_DIR, config_name="base_eval")
def main(override_config: OmegaConf):
    hydra_log_path = os.path.join(HydraConfig.get().runtime.output_dir, "probe.log")
    from loguru import logger

    logger.remove()
    logger.add(hydra_log_path, level="DEBUG")
    logger.add(sys.stdout, level="INFO", colorize=True)

    from gr00t.rl.utils.logging import HydraLoggerBridge
    import logging

    logging.basicConfig(level=logging.INFO)
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
        simulation_app = app_launcher.app  # noqa: F841

    # ---- imports that require Isaac Sim / torch to already be initialized ----
    import torch
    from accelerate import Accelerator
    from hydra.utils import instantiate
    from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
    from isaaclab.utils.math import subtract_frame_transforms
    from transformers import HfArgumentParser
    from trl import ModelConfig, PPOConfig, ScriptArguments

    from gr00t.rl.trl.utils.common import custom_instantiate
    from gr00t.rl.utils.common import seeding
    from gr00t.rl.utils.helpers import pre_process_config
    from move_garbage.envs.bin_push_homie import BinPushHomie

    unresolved_conf = OmegaConf.to_container(config, resolve=False)  # noqa: F841
    os.chdir(hydra.utils.get_original_cwd())
    pre_process_config(config)

    config.env.config.save_rendering_dir = str(checkpoint.parent / "renderings" / "probe_tmp")
    config.env.config.ckpt_dir = str(checkpoint.parent)
    config.algo.trl.output_dir = str(Path(config.experiment_dir))
    parser = HfArgumentParser((ScriptArguments, PPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_dict(config.algo.trl)  # noqa: F841

    accelerator = Accelerator()
    device = str(accelerator.device)
    if device == "cuda":
        device = "cuda:0"
    seeding(config.seed)

    env = instantiate(config=config.env, device=device)
    env_u: BinPushHomie = env

    module_dim_dict = getattr(config.algo.config, "module_dim", {})
    policy = hydra.utils.instantiate(
        config.algo.config.actor, env_config=env.config, algo_config=config.algo.config,
        module_dim_dict=module_dim_dict, _recursive_=False,
    ).to(device)
    value_model = hydra.utils.instantiate(
        config.algo.config.critic, env_config=env.config, algo_config=config.algo.config,
        module_dim_dict=module_dim_dict, _recursive_=False,
    ).to(device)
    accelerator.wait_for_everyone()

    # custom_instantiate loads the checkpoint AND the two frozen HOMIE
    # locomotion checkpoints (config.algo.config.homie_walk_model_path /
    # homie_stand_model_path -- see ppo_trainer_homie_api.py) into
    # trainer.model.homie_walk_model / homie_stand_model. This is the ONLY
    # reason this script needs a --checkpoint at all: to get a correctly
    # merged config and these two frozen leg controllers. `policy`'s own
    # weights are loaded too but its action_mean is never read below.
    trainer = custom_instantiate(
        config.trainer, args=training_args, config=config.algo.config, env=env,
        model=policy, ref_model=None, use_ref_model=False, value_model=value_model,
        train_dataset=None, eval_dataset=None, callbacks=[], checkpoint=config.checkpoint,
        local_seed=config.seed, accelerator=accelerator,
    )

    from gr00t.rl.trl.modules.homie_modules import init_actor_critic_dict as _iacd

    model = trainer.model
    homie_walk_model = model.homie_walk_model
    homie_stand_model = model.homie_stand_model
    homie_switch_threshold = getattr(trainer, "homie_switch_threshold", 0.5)
    homie_one_step_obs = _iacd["num_one_step_obs"]

    trainer._eval_mode()

    def homie_leg_actions(obs_dict):
        """Exactly measure_standoff.py / play_bin_push.py's walk/stand blend
        -- reused verbatim (not reimplemented) so HOMIE is driven identically
        to a real rollout. With this probe's locomotion command pinned at
        zero throughout (see below), `walk_mask` will read False every step
        and this always selects `homie_stand_model`, but the SAME code path
        used in training is what selects it, not a special-cased shortcut."""
        homie_obs = obs_dict["homie_obs"]
        walk_out = homie_walk_model(homie_obs)
        stand_out = homie_stand_model(homie_obs)
        commands = homie_obs[..., -homie_one_step_obs:-(homie_one_step_obs - 3)]
        walk_mask = torch.norm(commands, dim=-1, keepdim=True) > homie_switch_threshold
        m = walk_mask
        while m.dim() < walk_out["actions"].dim():
            m = m.unsqueeze(-1)
        return torch.where(m, walk_out["action_mean"], stand_out["action_mean"])

    num_envs = env_u.num_envs
    if num_envs > 1:
        logger.warning(
            "[probe] this script derives IK targets from env 0 only and "
            "broadcasts them to every env (their bins may sit at different "
            "poses) -- recommend ++num_envs=1 for a meaningful result"
        )
    settle_steps = int(config.get("probe_settle_steps", 30))
    n_arc_waypoints = int(config.get("probe_arc_waypoints", 6))
    steps_per_waypoint = int(config.get("probe_steps_per_waypoint", 12))
    final_hold_steps = int(config.get("probe_final_hold_steps", 20))
    traj_path = config.get("probe_out", "/tmp/probe_traj.npz")
    ik_damping = float(config.get("probe_ik_damping", 0.05))

    action_scale = float(env_u.config.robot.control.action_scale)
    delta_action_clip = float(env_u.config.get("delta_action_clip", 15.0))
    num_lower = env_u._num_lower_dof  # HomieBase: legs(12)+waist(3) = 15

    def delta_idx_for_dof(dof_idx: int) -> int:
        """Positional correspondence established in HomieBase.step() /
        FingerPrimitiveBase._apply_force_in_physics_step: whole_body_actions
        column j in [num_lower, num_lower+14) is added straight onto
        default_dof_pos[:, j] with no separate remap tensor, and that same
        column range is filled from processed_commands[:, num_homie_commands:]
        which in turn is action-index [3:17) of our 19-dim action vector
        (config.delta_action_indices is literally range(19)). So
        delta-action-index = 3 + (physical_dof_idx - num_lower). Derived from
        reading the actual step() chain across DeltaActionBase / HomieBase /
        FingerPrimitiveBase in this repo, not assumed from a naming
        convention -- see this script's accompanying report for the full
        trace."""
        return 3 + (dof_idx - num_lower)

    # ---- reset, then teleport straight to the APPROACH stop pose ----
    obs_dict = env_u.reset_all()
    for k in obs_dict:
        obs_dict[k] = obs_dict[k].to(device)

    env_ids = torch.arange(num_envs, device=device)
    stop_xy = env_u._stop_target_xy()  # (num_envs, 2), grasp-relative stop target
    root_state = env_u.base_init_state.clone().reshape(1, -1).repeat(num_envs, 1)
    root_state[:, :3] = root_state[:, :3] + env_u.env_origins
    root_state[:, 0:2] = stop_xy
    # Face the bin explicitly (identity quat, wxyz) rather than trust
    # base_init_state's own stored orientation: _reset_root_states always
    # OVERWRITES the quat every reset via quat_from_euler_xyz(0, 0,
    # random_yaw) -- push_axis_world is hardcoded (1,0,0), and the robot
    # spawns facing +X toward the bin with only a +-15deg yaw jitter this
    # probe drops for a repeatable pose -- so base_init_state's own quat
    # field may never actually be read elsewhere and isn't a safe thing to
    # inherit unedited.
    root_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
    root_state[:, 7:13] = 0.0  # zero all velocities

    robot = env_u.simulator._robot
    dof_pos = env_u.default_dof_pos.repeat(num_envs, 1).clone()
    dof_vel = torch.zeros_like(dof_pos)
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:13])
    robot.write_joint_state_to_sim(dof_pos, dof_vel)
    robot.write_data_to_sim()
    env_u._refresh_sim_tensors()

    stage_reach = torch.full((num_envs,), BinPushHomie.STAGE_REACH, dtype=torch.long, device=device)
    env_u.set_to_stage(env_ids, stage_reach)
    env_u._delta_actions.zero_()
    env_u._last_delta_actions.zero_()
    # ROOT CAUSE of the "teleport doesn't stick" bug: reset_buf is
    # torch.ones(...) at construction time (base_task.py) and is NEVER
    # cleared by reset_all()/set_to_stage() -- only a real step()'s
    # _post_physics_step() clears it (legged_robot_base.py: `self.reset_buf[:]
    # = 0` then re-derives it from this step's own termination check). So the
    # very first real_env_step() call after a manual teleport sees this
    # stale all-ones flag, includes env 0 in `reset_buf.nonzero()`, and
    # unconditionally calls reset_envs_idx() on it -- discarding the teleport
    # via a real (effectively random) respawn before our IK loop ever runs.
    # Confirmed empirically: root_pos jumped from the teleported
    # [-0.82, -0.0, 0.75] to [-1.87, 0.10, 0.75] on settle step 0 with this
    # flag left uncleared. Fix: clear it explicitly, same as a real step()
    # would once it has actually reset something.
    env_u.reset_buf.zero_()
    if hasattr(env_u, "time_out_buf"):
        env_u.time_out_buf.zero_()
    # SECOND, independent stale-flag bug (reset_buf alone was NOT enough --
    # confirmed empirically, teleport still got clobbered after clearing only
    # reset_buf): `need_to_refresh_envs` (also torch.ones(...) at construction,
    # legged_robot_base.py:242) drives a DEFERRED write in _post_physics_step
    # -- `target_robot_root_states`/`target_robot_dof_state` (whatever
    # reset_all() staged there, e.g. a random/default spawn) get written
    # BACK onto the sim via set_actor_root_state_tensor/set_dof_state_tensor
    # for any env still flagged here, discarding our direct
    # write_root_pose_to_sim/write_joint_state_to_sim teleport. Also keep
    # target_robot_root_states/target_robot_dof_state themselves consistent
    # with what we just wrote, in case anything else reads them before this
    # flag would next be set (defensive, not required to fix the bug above).
    if hasattr(env_u, "target_robot_root_states"):
        env_u.target_robot_root_states[env_ids] = root_state
    if hasattr(env_u, "target_robot_dof_state"):
        env_u.target_robot_dof_state[env_ids, :, 0] = dof_pos
        env_u.target_robot_dof_state[env_ids, :, 1] = dof_vel
    env_u.need_to_refresh_envs.zero_()
    # THIRD stale-buffer bug, same pattern as the two above: time_in_stage_buf
    # (staged_task_base.py:271) drives `is_overtime = time_in_stage_buf >=
    # max_stage_time[stage_buf]` -> reset_buf |= is_overtime every step
    # (staged_task_base.py:343-352). It zero-inits at construction, but
    # reset_all() does not necessarily zero it again for envs it resets (same
    # gap as reset_buf/need_to_refresh_envs above) -- if it's still carrying
    # time accrued before our forced set_to_stage(REACH) call, REACH's
    # 100-step budget (max_stage_time[1]=100) can get silently exceeded
    # mid-probe, firing a real reset partway through a run (confirmed
    # empirically: base_root_pos jumped ~1m between two consecutive
    # real_env_step() calls late in the final_grasp hold, with no other
    # explanation fitting -- IK/arm state alone can't move the ROOT that far
    # in one physics step).
    if hasattr(env_u, "time_in_stage_buf"):
        env_u.time_in_stage_buf.zero_()
    if hasattr(env_u, "actual_time_in_stage_buf"):
        env_u.actual_time_in_stage_buf.zero_()

    # DIAGNOSTIC (temporary -- root-cause the "spawn doesn't stick" bug):
    # log every buffer that could plausibly still hold a pre-teleport value
    # and get silently re-applied to the sim by the first real env_u.step().
    logger.info(f"[probe][DIAG] post-teleport robot.data.root_state_w[:3]={robot.data.root_state_w[0, :3].tolist()}")
    for attr in ("episode_length_buf", "reset_buf", "time_in_stage", "actual_time_in_stage",
                 "stage_buf", "_target_robot_dof_state", "_target_robot_root_state"):
        if hasattr(env_u, attr):
            v = getattr(env_u, attr)
            try:
                v0 = v[0] if hasattr(v, "__getitem__") else v
                logger.info(f"[probe][DIAG] env_u.{attr}[0] = {v0}")
            except Exception as e:
                logger.info(f"[probe][DIAG] env_u.{attr} -- could not index: {e}")

    logger.info(
        f"[probe] teleported to stop pose xy={stop_xy[0].tolist()}, forced "
        f"stage_buf={env_u.stage_buf[0].item()} (STAGE_REACH={BinPushHomie.STAGE_REACH})"
    )

    # ---- IK controllers, one per hand (Isaac Lab's differential IK,
    # damped-least-squares) ----
    # UNVERIFIED against this repo's installed Isaac Lab 2.3.2.post1: this
    # class/cfg is not used anywhere else in the repo (grepped clean), so
    # this constructor call and .compute()/.set_command() signature are from
    # general Isaac Lab API knowledge only. Needs a real run to confirm.
    ik_cfg = DifferentialIKControllerCfg(
        command_type="pose", use_relative_mode=False, ik_method="dls",
        ik_params={"lambda_val": ik_damping},
    )
    ik_left = DifferentialIKController(ik_cfg, num_envs=num_envs, device=device)
    ik_right = DifferentialIKController(ik_cfg, num_envs=num_envs, device=device)

    left_dof_idx = env_u._left_arm_dof_idx
    right_dof_idx = env_u._right_arm_dof_idx
    left_palm_body = env_u.left_palm_idx
    right_palm_body = env_u.right_palm_idx
    # Floating-base Jacobian body-index offset-by-one convention copied from
    # Isaac Lab's DifferentialIKController tutorial (source jacobian excludes
    # the free root link for a floating-base articulation). UNVERIFIED here
    # -- confirm on the VM by checking jacobian.shape[1] == num_bodies-1.
    left_jacobi_idx = left_palm_body - 1
    right_jacobi_idx = right_palm_body - 1

    hard_limits = env_u.simulator.hard_dof_pos_limits  # (num_dof, 2), radians, true mechanical limits

    def ik_step(controller, jacobi_idx, dof_idx, target_pos_w, target_quat_wxyz):
        """One damped-least-squares IK update toward (target_pos_w,
        target_quat_wxyz) (world frame, (num_envs,3)/(num_envs,4)); returns
        the desired absolute joint targets for `dof_idx` and the resulting
        world-frame position error norm."""
        # body_pos_w / body_quat_w (not the combined body_state_w) are the
        # field names actually confirmed in this repo (play_bin_push.py /
        # replay_traj.py both read robot.data.body_pos_w / body_quat_w) --
        # used here for the same reason, root_state_w is likewise confirmed.
        root_pos_w = robot.data.root_state_w[:, 0:3]
        root_quat_w = robot.data.root_state_w[:, 3:7]
        ee_pos_w = robot.data.body_pos_w[:, jacobi_idx + 1, :]
        ee_quat_w = robot.data.body_quat_w[:, jacobi_idx + 1, :]
        ee_pos_b, ee_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
        target_pos_b, target_quat_b = subtract_frame_transforms(
            root_pos_w, root_quat_w, target_pos_w, target_quat_wxyz
        )
        controller.set_command(torch.cat([target_pos_b, target_quat_b], dim=-1))
        jacobian = robot.root_physx_view.get_jacobians()[:, jacobi_idx, :, dof_idx]
        joint_pos_curr = env_u.simulator.dof_pos[:, dof_idx]
        joint_pos_des = controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos_curr)
        pos_err = torch.linalg.norm(target_pos_w - ee_pos_w, dim=-1)
        return joint_pos_des, pos_err

    def apply_arm_targets(q_left_target, q_right_target):
        """Clip to the true hard joint limits, record whether clipping fired
        (== IK asked for something outside the physical range), and write the
        resulting delta-actions directly into the env's integrator buffer."""
        hit_limit = torch.zeros(num_envs, 2, dtype=torch.bool, device=device)  # (env, hand)
        for side, dof_idx, q_target in (("left", left_dof_idx, q_left_target),
                                         ("right", right_dof_idx, q_right_target)):
            lo = hard_limits[dof_idx, 0]
            hi = hard_limits[dof_idx, 1]
            clipped = torch.clamp(q_target, lo, hi)
            over = (q_target < lo) | (q_target > hi)
            if side == "left":
                hit_limit[:, 0] = over.any(dim=-1)
            else:
                hit_limit[:, 1] = over.any(dim=-1)
            for local_i, d in enumerate(dof_idx.tolist()):
                delta = _delta_action_for_target(
                    clipped[:, local_i].cpu().numpy(),
                    env_u.default_dof_pos[0, d].cpu().numpy(),
                    action_scale, delta_action_clip,
                )
                env_u._delta_actions[:, delta_idx_for_dof(d)] = torch.as_tensor(
                    delta, device=device, dtype=torch.float32
                )
        return hit_limit

    def real_env_step():
        """The ONE place this script advances physics -- always through
        env_u.step({"actions": ...}), i.e. the exact same DeltaActionBase ->
        WarpedActionBase -> HomieBase step() chain a real rollout uses.
        HOMIE leg control (`homie_leg_actions`) is computed from the live
        `obs_dict["homie_obs"]` every call, so the frozen standing policy
        genuinely runs every physics step of this probe -- this script never
        writes joint targets to the simulator directly outside of the
        one-time teleport above, and never calls a lower-level physics-step
        API bypassing env_u.step()."""
        nonlocal obs_dict
        action19 = torch.zeros(num_envs, 19, device=device)  # cols 3:17 ignored: the
        # arm targets are already injected into env_u._delta_actions above and
        # DeltaActionBase does `_delta_actions += action*delta_action_scale`,
        # so a raw 0 here leaves our injected value untouched this step.
        homie_actions = homie_leg_actions(obs_dict)
        step_actions = torch.cat([action19, homie_actions], dim=-1)
        obs_dict, rewards, dones, infos = env_u.step({"actions": step_actions})
        for k in obs_dict:
            obs_dict[k] = obs_dict[k].to(device)
        return rewards, dones, infos

    # ---- trajectory + diagnostic logging (play_bin_push.py-compatible npz
    # schema, so replay_traj.py can render it unmodified) ----
    bin_art = env_u.simulator._task["bin"]
    robot_body_names = list(robot.data.body_names)
    bin_body_names = list(bin_art.data.body_names)
    robot_joint_names = list(robot.data.joint_names)
    bin_joint_names = list(bin_art.data.joint_names)

    log = dict(
        robot_pos=[], robot_quat=[], bin_pos=[], bin_quat=[],
        robot_root=[], robot_jpos=[], bin_root=[], bin_jpos=[],
        stage=[], done=[], grasp_valid=[], palm_anchor_dist=[],
        palm_down_alignment=[], hand_bar_dist=[], bin_push_progress=[],
        keyframe=[], ik_hit_limit_left=[], ik_hit_limit_right=[],
        ik_pos_err_left=[], ik_pos_err_right=[],
    )
    reward_log = {name: [] for name in _REACH_GRASP_REWARD_TERMS}
    available_reward_terms = []
    for name in _REACH_GRASP_REWARD_TERMS:
        if hasattr(env_u, f"_reward_{name}"):
            available_reward_terms.append(name)
        else:
            logger.warning(f"[probe] no _reward_{name} method on env -- skipping")

    def log_step(keyframe_label, hit_limit=None, pos_err_left=None, pos_err_right=None):
        left_anchor, right_anchor = env_u._palm_grasp_points()
        left_align, right_align = env_u._palm_down_alignments()
        # 2026-08-29: renamed from _hand_contact_counts() (removed) -- now
        # returns per-hand palm-to-bar-target DISTANCES, not PhysX
        # contact-link counts. See bin_push_homie.py's _hand_bar_distances.
        left_contacts, right_contacts = env_u._hand_bar_distances()
        anchor_dist = torch.stack([
            (env_u._left_handle_target() - left_anchor).norm(dim=-1),
            (env_u._right_handle_target() - right_anchor).norm(dim=-1),
        ], dim=-1)
        log["robot_pos"].append(robot.data.body_pos_w[0].detach().cpu().numpy())
        log["robot_quat"].append(robot.data.body_quat_w[0].detach().cpu().numpy())
        log["bin_pos"].append(bin_art.data.body_pos_w[0].detach().cpu().numpy())
        log["bin_quat"].append(bin_art.data.body_quat_w[0].detach().cpu().numpy())
        log["robot_root"].append(robot.data.root_state_w[0, :7].detach().cpu().numpy())
        log["robot_jpos"].append(robot.data.joint_pos[0].detach().cpu().numpy())
        log["bin_root"].append(bin_art.data.root_state_w[0, :7].detach().cpu().numpy())
        log["bin_jpos"].append(bin_art.data.joint_pos[0].detach().cpu().numpy())
        log["stage"].append(int(env_u.stage_buf[0].item()))
        log["done"].append(False)
        log["grasp_valid"].append(bool(env_u._grasp_pose_valid_mask()[0].item()))
        log["palm_anchor_dist"].append(anchor_dist[0].detach().cpu().numpy())
        log["palm_down_alignment"].append(
            torch.stack([left_align[0], right_align[0]]).detach().cpu().numpy()
        )
        log["hand_bar_dist"].append(
            torch.stack([left_contacts[0], right_contacts[0]]).detach().cpu().numpy()
        )
        log["bin_push_progress"].append(float(env_u._bin_push_progress()[0].item()))
        log["keyframe"].append(keyframe_label)
        log["ik_hit_limit_left"].append(bool(hit_limit[0, 0].item()) if hit_limit is not None else False)
        log["ik_hit_limit_right"].append(bool(hit_limit[0, 1].item()) if hit_limit is not None else False)
        log["ik_pos_err_left"].append(float(pos_err_left[0].item()) if pos_err_left is not None else float("nan"))
        log["ik_pos_err_right"].append(float(pos_err_right[0].item()) if pos_err_right is not None else float("nan"))
        for name in available_reward_terms:
            val = float(getattr(env_u, f"_reward_{name}")().mean().item())
            reward_log[name].append(val)

    # ---- settle: hold the teleported pose under the REAL homie-stand
    # controller before starting the IK trajectory ----
    for _settle_i in range(settle_steps):
        env_u._delta_actions[:, 3:17] = 0.0  # hold default_dof_pos (arms at rest)
        rewards, dones, infos = real_env_step()
        if _settle_i < 3:
            logger.info(
                f"[probe][DIAG] settle step {_settle_i}: "
                f"root_pos={robot.data.root_state_w[0, :3].tolist()} "
                f"dones={dones[0].item() if hasattr(dones, '__getitem__') else dones} "
                f"stage_buf={env_u.stage_buf[0].item()}"
            )
        log_step("settle")

    # ---- build per-hand target sequences: pregrasp arc -> final grasp ----
    def hand_targets(handle_target_fn, handle_rot_fn, palm_body_idx, axis, offset_cfg_key):
        target_grasp_pt = handle_target_fn()[0].detach().cpu().numpy()
        target_quat = handle_rot_fn()[0].detach().cpu().numpy()  # wxyz -- see note below
        offset = float(env_u.config.get(offset_cfg_key, 0.0464))
        target_palm_pos, target_palm_quat = _grasp_point_to_palm_pose(
            target_grasp_pt, target_quat, axis, offset
        )
        start_pos = robot.data.body_pos_w[0, palm_body_idx, :].detach().cpu().numpy()
        start_quat = robot.data.body_quat_w[0, palm_body_idx, :].detach().cpu().numpy()
        blend_dist = float(env_u.config.get("pregrasp_blend_dist", 0.30))
        peak_frac = float(env_u.config.get("pregrasp_arc_peak_frac", 0.75))
        arc_h = float(env_u.config.get("pregrasp_arc_height", 0.10))
        waypoints = []
        for i in range(1, n_arc_waypoints + 1):
            frac = i / n_arc_waypoints
            pos, quat = _pregrasp_waypoint_pose(
                start_pos, start_quat, target_palm_pos, target_palm_quat,
                frac, blend_dist, peak_frac, arc_h,
            )
            waypoints.append((pos, quat))
        return waypoints

    left_axis, right_axis = env_u._palm_grasp_axes()
    left_axis_np = left_axis[0].detach().cpu().numpy()
    right_axis_np = right_axis[0].detach().cpu().numpy()

    # NOTE on wxyz/xyzw: bin_push_homie.py's _left_handle_target_rot() returns
    # xyzw (it converts via wxyz_to_xyzw for the reward's quat_error_magnitude
    # call, which itself expects wxyz -- see _palm_twist_errors doing
    # xyzw_to_wxyz(self._left_handle_target_rot())). This script needs wxyz
    # throughout (Isaac Lab body_pos_w/body_quat_w convention and the IK
    # controller), so it converts explicitly via _wrap_wxyz below rather than
    # assuming a convention.
    # DIAGNOSTIC (temporary): dump every quantity feeding the LEFT-hand target
    # so a divergent/degrading arc (errors growing waypoint over waypoint
    # instead of shrinking) can be traced to a specific wrong number.
    _dbg_grasp_pt = env_u._left_handle_target()[0].detach().cpu().numpy()
    _dbg_target_quat_xyzw = env_u._left_handle_target_rot()[0].detach().cpu().numpy()
    _dbg_target_quat_wxyz = _wrap_wxyz(env_u._left_handle_target_rot())[0].detach().cpu().numpy()
    _dbg_start_pos = robot.data.body_pos_w[0, left_palm_body, :].detach().cpu().numpy()
    _dbg_start_quat = robot.data.body_quat_w[0, left_palm_body, :].detach().cpu().numpy()
    _dbg_palm_pos, _dbg_palm_quat = _grasp_point_to_palm_pose(
        _dbg_grasp_pt, _dbg_target_quat_wxyz, left_axis_np,
        float(env_u.config.get("palm_grasp_point_offset", 0.0464)),
    )
    logger.info(f"[probe][DIAG] LEFT grasp_pt(world)={_dbg_grasp_pt.tolist()}")
    logger.info(f"[probe][DIAG] LEFT target_quat_xyzw(raw from env)={_dbg_target_quat_xyzw.tolist()}")
    logger.info(f"[probe][DIAG] LEFT target_quat_wxyz(after _wrap_wxyz)={_dbg_target_quat_wxyz.tolist()}")
    logger.info(f"[probe][DIAG] LEFT palm axis={left_axis_np.tolist()}")
    logger.info(f"[probe][DIAG] LEFT derived target_palm_pos={_dbg_palm_pos.tolist()} target_palm_quat={_dbg_palm_quat.tolist()}")
    logger.info(f"[probe][DIAG] LEFT current palm start_pos={_dbg_start_pos.tolist()} start_quat(wxyz, from body_quat_w)={_dbg_start_quat.tolist()}")
    logger.info(f"[probe][DIAG] LEFT start->target pos delta norm={float(np.linalg.norm(_dbg_palm_pos - _dbg_start_pos)):.4f}m")
    _dbg_quat_dot = float(np.dot(_dbg_start_quat, _dbg_palm_quat))
    logger.info(f"[probe][DIAG] LEFT start_quat . target_quat dot={_dbg_quat_dot:.4f} (near +-1 = similar orientation, near 0 = ~90deg apart)")

    left_waypoints = hand_targets(
        env_u._left_handle_target,
        lambda: _wrap_wxyz(env_u._left_handle_target_rot()),
        left_palm_body, left_axis_np, "palm_grasp_point_offset",
    )
    right_waypoints = hand_targets(
        env_u._right_handle_target,
        lambda: _wrap_wxyz(env_u._right_handle_target_rot()),
        right_palm_body, right_axis_np, "palm_grasp_point_offset",
    )

    # ---- drive both hands through their waypoint sequence, one IK update
    # per real physics step ----
    for wp_i in range(n_arc_waypoints):
        left_pos_np, left_quat_np = left_waypoints[wp_i]
        right_pos_np, right_quat_np = right_waypoints[wp_i]
        left_pos = torch.as_tensor(left_pos_np, device=device, dtype=torch.float32).repeat(num_envs, 1)
        left_quat = torch.as_tensor(left_quat_np, device=device, dtype=torch.float32).repeat(num_envs, 1)
        right_pos = torch.as_tensor(right_pos_np, device=device, dtype=torch.float32).repeat(num_envs, 1)
        right_quat = torch.as_tensor(right_quat_np, device=device, dtype=torch.float32).repeat(num_envs, 1)
        keyframe = "final_grasp" if wp_i == n_arc_waypoints - 1 else f"arc_{wp_i}"
        hold = final_hold_steps if wp_i == n_arc_waypoints - 1 else steps_per_waypoint
        for _hold_i in range(hold):
            q_left, err_left = ik_step(ik_left, left_jacobi_idx, left_dof_idx, left_pos, left_quat)
            q_right, err_right = ik_step(ik_right, right_jacobi_idx, right_dof_idx, right_pos, right_quat)
            hit_limit = apply_arm_targets(q_left, q_right)
            real_env_step()
            log_step(keyframe, hit_limit, err_left, err_right)
            if _hold_i == 0 or _hold_i == hold - 1:
                logger.info(
                    f"[probe][DIAG] {keyframe} hold_i={_hold_i}: "
                    f"base_root_pos={robot.data.root_state_w[0, :3].tolist()} "
                    f"base_height={robot.data.root_state_w[0, 2].item():.3f} "
                    f"left_ee_pos={robot.data.body_pos_w[0, left_palm_body, :].tolist()} "
                    f"left_target_pos={left_pos[0].tolist()} err_left={float(err_left[0]):.4f}"
                )

    # ---- save trajectory + diagnostics ----
    np.savez_compressed(
        traj_path,
        robot_pos=np.stack(log["robot_pos"]), robot_quat=np.stack(log["robot_quat"]),
        bin_pos=np.stack(log["bin_pos"]), bin_quat=np.stack(log["bin_quat"]),
        robot_body_names=np.array(robot_body_names), bin_body_names=np.array(bin_body_names),
        robot_root=np.stack(log["robot_root"]), robot_jpos=np.stack(log["robot_jpos"]),
        bin_root=np.stack(log["bin_root"]), bin_jpos=np.stack(log["bin_jpos"]),
        robot_joint_names=np.array(robot_joint_names), bin_joint_names=np.array(bin_joint_names),
        stage=np.asarray(log["stage"], dtype=np.int64),
        done=np.asarray(log["done"], dtype=np.bool_),
        grasp_valid=np.asarray(log["grasp_valid"], dtype=np.bool_),
        palm_anchor_dist=np.stack(log["palm_anchor_dist"]),
        palm_down_alignment=np.stack(log["palm_down_alignment"]),
        hand_bar_dist=np.stack(log["hand_bar_dist"]),
        bin_push_progress=np.asarray(log["bin_push_progress"], dtype=np.float32),
        # -- probe-specific extras (ignored by replay_traj.py, safe to carry) --
        keyframe=np.array(log["keyframe"]),
        ik_hit_limit_left=np.asarray(log["ik_hit_limit_left"], dtype=np.bool_),
        ik_hit_limit_right=np.asarray(log["ik_hit_limit_right"], dtype=np.bool_),
        ik_pos_err_left=np.asarray(log["ik_pos_err_left"], dtype=np.float32),
        ik_pos_err_right=np.asarray(log["ik_pos_err_right"], dtype=np.float32),
        reward_term_names=np.array(available_reward_terms),
        **{f"reward_{k}": np.asarray(v, dtype=np.float32) for k, v in reward_log.items()},
    )
    logger.info(f"[probe] wrote {traj_path} ({len(log['stage'])} frames)")

    # ---- summary ----
    keyframes = log["keyframe"]
    print("\n==================== IK FEASIBILITY ====================")
    for wp_i in range(n_arc_waypoints):
        label = "final_grasp" if wp_i == n_arc_waypoints - 1 else f"arc_{wp_i}"
        idxs = [i for i, k in enumerate(keyframes) if k == label]
        last = idxs[-1]
        l_hit = log["ik_hit_limit_left"][last]
        r_hit = log["ik_hit_limit_right"][last]
        l_err = log["ik_pos_err_left"][last]
        r_err = log["ik_pos_err_right"][last]
        l_status = "HIT_LIMIT" if l_hit else ("CONVERGED" if l_err < 0.02 else "NOT_CONVERGED")
        r_status = "HIT_LIMIT" if r_hit else ("CONVERGED" if r_err < 0.02 else "NOT_CONVERGED")
        print(f"  {label:12s}  LEFT: {l_status:14s} err={l_err:.4f}m   "
              f"RIGHT: {r_status:14s} err={r_err:.4f}m")

    print("\n==================== REWARD SANITY (final_grasp keyframe) ====================")
    final_idxs = [i for i, k in enumerate(keyframes) if k == "final_grasp"]
    tail = final_idxs[-min(5, len(final_idxs)):]
    for name in available_reward_terms:
        vals = [reward_log[name][i] for i in tail]
        mean_val = sum(vals) / len(vals)
        flag = "" if mean_val > 0.7 else "  <-- FAR FROM MAX (expect ~1.0 for a textbook grasp)"
        print(f"  {name:32s} mean={mean_val:+.4f}{flag}")
    print("================================================================\n")

    print(f"Next: render with\n"
          f"  python -u move_garbage/scripts/replay_traj.py --headless --enable_cameras \\\n"
          f"    --traj {traj_path} --out /tmp/probe_rollout_1x.mp4 --fps 25 --stride 2")

    os._exit(0)


def _wrap_wxyz(xyzw_tensor):
    """_left_handle_target_rot()/_right_handle_target_rot() return xyzw
    (bin_push_homie.py wraps them in wxyz_to_xyzw before returning -- see
    scene_creation_callback / _palm_twist_errors). Convert to wxyz here,
    once, at the single call site that needs it, rather than assuming."""
    import torch

    x, y, z, w = xyzw_tensor[0, 0], xyzw_tensor[0, 1], xyzw_tensor[0, 2], xyzw_tensor[0, 3]
    return torch.stack([w, x, y, z]).unsqueeze(0)


if __name__ == "__main__":
    main()
