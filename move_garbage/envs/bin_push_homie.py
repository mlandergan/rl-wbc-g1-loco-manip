# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bin-push task: G1 bimanually pushes a 2-wheeled recycling bin.

Mirrors ``gr00t/rl/envs/cabinet/drawer_pull_homie.py`` with bimanual reach,
grasp, and push rewards.
"""

import math
import os
from datetime import datetime

from loguru import logger

try:
    import omni.usd  # type: ignore
    from pxr import Usd  # type: ignore

    _HAS_OMNI = True
except Exception:  # pragma: no cover - import-only smoke tests run without Isaac
    _HAS_OMNI = False

try:
    import imageio  # type: ignore
except Exception:  # pragma: no cover
    imageio = None

import torch
import torch.nn.functional as F
from isaaclab.sensors import (
    ContactSensor,
    ContactSensorCfg,
    FrameTransformer,
    FrameTransformerCfg,
    OffsetCfg,
)
from isaaclab.utils.math import (
    euler_xyz_from_quat,
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    subtract_frame_transforms,
)
from typing_extensions import override

from gr00t.rl.envs.base_task.delta_action_base import DeltaActionBase
from gr00t.rl.envs.base_task.finger_primitive_base import FingerPrimitiveBase
from gr00t.rl.envs.base_task.homie_base import HomieBase
from gr00t.rl.envs.base_task.staged_task_base import StagedTaskBase
from gr00t.rl.envs.base_task.warped_action_base import WarpedActionBase
from gr00t.rl.isaac_utils.rotations import quat_to_tan_norm, wxyz_to_xyzw, xyzw_to_wxyz
from gr00t.rl.utils.torch_utils import torch_rand_float


# Fallback bin metadata used when the dynamic USD hasn't been prepared yet.
_FALLBACK_BIN_METADATA = {
    "binWidth": 0.50,
    "binHeight": 0.95,
    "handleHeight": 1.05,
    "wheelRadius": 0.10,
    "pushAxisWorld": (1.0, 0.0, 0.0),
}

# 2026-09-29 overhand: bar capsule radius (handle_collider) -- used by the
# overhand wrap score (surface distance = distance to bar axis - R).
_BAR_RADIUS = 0.0165


def _effective_in_named_stages(*names):
    """2026-09-29 overhand: like ``StagedTaskBase.effective_in_stage`` but the
    stages are given by NAME ("GRASP", "TILT", "PUSH", ...) and resolved on
    the instance (``self.STAGE_<NAME>``) at call time. Needed because the
    stage indices now depend on config: with ``enable_tilt_stage`` TILT=3 and
    PUSH=4, without it TILT=-1 (never matches) and PUSH=3 (old 4-stage
    layout, so old experiments keep working unchanged)."""

    def decorator(func):
        def wrapper(self, *args, **kwargs):
            reward = func(self, *args, **kwargs)
            assert isinstance(reward, torch.Tensor)
            assert reward.shape == (self.num_envs,)
            return torch.where(
                self._named_stage_mask(names), reward, torch.zeros_like(reward)
            )

        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper

    return decorator


class BinPushHomie(
    StagedTaskBase,
    DeltaActionBase,
    WarpedActionBase,
    HomieBase,
    FingerPrimitiveBase,
):
    """Bimanual bin-push task with a 4-stage curriculum (5 with enable_tilt_stage)."""

    STAGE_APPROACH = 0
    STAGE_REACH = 1
    STAGE_GRASP = 2
    STAGE_PUSH = 3
    # 2026-09-29 overhand: optional TILT stage between GRASP and PUSH. These
    # class values are the OLD 4-stage layout (TILT disabled = -1); __init__
    # overrides them per instance when env.config.enable_tilt_stage is True
    # (TILT=3, PUSH=4). Rewards that involve TILT/PUSH therefore use
    # _effective_in_named_stages (resolved at call time), never the class
    # constants.
    STAGE_TILT = -1

    # ================================================================
    # SETUP
    # ================================================================

    def __init__(self, config, device):
        # 2026-09-29 overhand: stage layout must be known BEFORE the base
        # constructors run (they compute observations / logs on reset).
        self._tilt_enabled = bool(config.get("enable_tilt_stage", False))
        expected_stages = 5 if self._tilt_enabled else 4
        assert len(config.max_stage_time) == expected_stages, (
            f"enable_tilt_stage={self._tilt_enabled} needs max_stage_time with "
            f"{expected_stages} entries, got {len(config.max_stage_time)}"
        )
        if self._tilt_enabled:
            self.STAGE_TILT = 3
            self.STAGE_PUSH = 4
        # 2026-09-29 overhand: bin asset selectable from config
        # (env.config.bin_asset_path, relative to the repo root). The scenario
        # module (move_garbage/scenario_cfg/isaacsim.py) reads BIN_ASSET_PATH
        # when the simulator builds the scene inside super().__init__, so it
        # is exported here. An explicitly set BIN_ASSET_PATH env var still
        # wins (old launchers unchanged).
        bin_asset_path = config.get("bin_asset_path", None)
        if bin_asset_path and not os.environ.get("BIN_ASSET_PATH"):
            from gr00t.rl.utils.paths import resolve_repo_path

            resolved = str(resolve_repo_path(str(bin_asset_path)))
            assert os.path.isfile(resolved), f"bin_asset_path not found: {resolved}"
            os.environ["BIN_ASSET_PATH"] = resolved
            logger.info(f"bin_asset_path -> BIN_ASSET_PATH={resolved}")
        super().__init__(config, device)

        # finger primitive related — mirror drawer_pull_homie
        self._left_p0 = torch.tensor(
            self.config.robot.finger_primitive.primitive_action_map.left.pos_0,
            device=self.device, requires_grad=False,
        )
        self._left_p1 = torch.tensor(
            self.config.robot.finger_primitive.primitive_action_map.left.pos_1,
            device=self.device, requires_grad=False,
        )
        self._right_p0 = torch.tensor(
            self.config.robot.finger_primitive.primitive_action_map.right.pos_0,
            device=self.device, requires_grad=False,
        )
        self._right_p1 = torch.tensor(
            self.config.robot.finger_primitive.primitive_action_map.right.pos_1,
            device=self.device, requires_grad=False,
        )
        self._left_hand_dof_idx = [
            self.dof_names.index(name)
            for name in self.config.robot.finger_primitive.primitive_action_map.left.dof_names
        ]
        self._right_hand_dof_idx = [
            self.dof_names.index(name)
            for name in self.config.robot.finger_primitive.primitive_action_map.right.dof_names
        ]
        self._upper_non_finger_dof_idx = [
            i for i in self.upper_dof_indices
            if i not in self._left_hand_dof_idx and i not in self._right_hand_dof_idx
        ]

        self._left_finger_action_idx = int(
            self.config.get("left_finger_delta_action_index", 17)
        )
        self._right_finger_action_idx = int(
            self.config.get("right_finger_delta_action_index", 18)
        )
        left_match = torch.where(
            self._delta_action_indices == self._left_finger_action_idx
        )[0]
        right_match = torch.where(
            self._delta_action_indices == self._right_finger_action_idx
        )[0]
        assert left_match.numel() == 1 and right_match.numel() == 1
        self._left_finger_delta_idx = int(left_match.item())
        self._right_finger_delta_idx = int(right_match.item())
        primitive_names = [cfg["action_name"] for cfg in self._primitive_action_map]
        self._left_whole_body_finger_action_idx = (
            self._num_non_primitive_dof + primitive_names.index("left")
        )
        self._right_whole_body_finger_action_idx = (
            self._num_non_primitive_dof + primitive_names.index("right")
        )
        action_scale = float(self.config.robot.control.action_scale)
        self._finger_open_action = -1.0 / action_scale
        self._finger_closed_action = 1.0 / action_scale

        # Body indices — bimanual, so we read both palms plus per-hand link lists.
        self.left_palm_idx = self.simulator.body_names.index("left_hand_palm_link")
        self.right_palm_idx = self.simulator.body_names.index("right_hand_palm_link")
        self.root_idx = self.simulator.body_names.index("pelvis")
        self.left_hand_indices = [
            self.simulator.body_names.index(link)
            for link in self.simulator.robot_config.left_hand_body_names
        ]
        self.right_hand_indices = [
            self.simulator.body_names.index(link)
            for link in self.simulator.robot_config.right_hand_body_names
        ]

        # DOF indices — mirror drawer.
        finger_dof_names = [dof for dof in self.simulator.dof_names if "hand" in dof]
        self.finger_dof_idx = torch.tensor(
            [self.simulator.dof_names.index(dof) for dof in finger_dof_names],
            dtype=torch.long, device=self.device,
        )
        self.wrist_dof_idx = torch.tensor(
            [self.simulator.dof_names.index(dof)
             for dof in self.simulator.dof_names if "wrist" in dof],
            dtype=torch.long, device=self.device,
        )
        self.dof_pos_humanly_lower_limit = torch.tensor(
            self.simulator.robot_config.dof_pos_humanly_lower_limit_list, device=self.device
        )[None, :]
        self.dof_pos_humanly_upper_limit = torch.tensor(
            self.simulator.robot_config.dof_pos_humanly_upper_limit_list, device=self.device
        )[None, :]

        self._left_arm_dof_idx = torch.tensor(self.left_arm_dof_indices, device=self.device)
        self._right_arm_dof_idx = torch.tensor(self.right_arm_dof_indices, device=self.device)

        # Register the bin as a task-state to track. StagedTaskBase relies on this to
        # snapshot object state for staged reset.
        self._register_task_state_to_track(
            self.simulator.scene.articulations["bin"], "bin"
        )
        # Register delta_actions buffer so staged reset restores the delta stream too.
        self._register_buffer_to_track(
            "delta_actions",
            self._get_delta_actions_buffer_shape(),
            self._store_delta_actions_buffer,
            self._load_delta_actions_buffer,
            dtype=torch.float32,
        )

        self.resting_dof_pos = torch.tensor([self.config.resting_dof_pos], device=self.device)

        # ----- Read bin customData metadata (with graceful fallback) -----
        # If the dynamic USD hasn't been prepared yet (Agent A pipeline), or if the
        # prim/metadata is missing, we log a warning and use fallbacks so the smoke
        # test can still exercise the rest of the wiring.
        self.bin_width = torch.zeros(self.num_envs, device=self.device)
        self.bin_height = torch.zeros(self.num_envs, device=self.device)
        self.handle_height = torch.zeros(self.num_envs, device=self.device)
        self.wheel_radius = torch.zeros(self.num_envs, device=self.device)
        self.push_axis_world = torch.zeros(self.num_envs, 3, device=self.device)
        # Config-level push target distance in meters. Duplicated per-env for privileged obs.
        self.target_bin_push_distance = float(
            self.config.get("target_bin_push_distance", 1.0)
        )
        self._read_bin_metadata()

        # ----- Privileged obs scratch buffers (populated on reset) -----
        # [mass, ground_friction, wheel_stiffness, push_target_distance, push_axis_x, push_axis_y]
        self._privileged_bin_mass = torch.full(
            (self.num_envs,), 15.0, device=self.device
        )
        self._privileged_ground_friction = torch.full(
            (self.num_envs,), 0.9, device=self.device
        )
        self._privileged_wheel_stiffness = torch.zeros(
            self.num_envs, device=self.device
        )

        # Cache the initial bin XY (world frame, per-env) for push-progress obs.
        self._initial_bin_pos_xy = torch.zeros(self.num_envs, 2, device=self.device)
        # 2026-10-03 push2: handle-bar midpoint XY at the PUSH handover (for
        # push_progress_use_bar_min, so backward tipping cannot fake +x progress).
        self._initial_bar_pos_xy = torch.zeros(self.num_envs, 2, device=self.device)
        self._register_buffer_to_track(
            "initial_bin_pos_xy",
            (self.num_envs, 2),
            self._store_initial_bin_pos_xy,
            self._load_initial_bin_pos_xy,
            dtype=torch.float32,
        )

        # Sustained-reach: per-env count of consecutive steps BOTH palms held near
        # their handles. Rewards settle-and-hold over swipe-through (Run 17).
        self._reach_hold_steps = torch.zeros(self.num_envs, device=self.device)

    def _read_bin_metadata(self):
        """Read per-env bin customData from the USD stage; fall back if missing."""
        default = dict(_FALLBACK_BIN_METADATA)
        # Let the config override the pushAxisWorld fallback if provided.
        cfg_axis = self.config.get("push_axis_world", None)
        if cfg_axis is not None:
            default["pushAxisWorld"] = tuple(cfg_axis)
        push_axis_default = torch.tensor(
            default["pushAxisWorld"], device=self.device
        )

        if not _HAS_OMNI:
            logger.warning(
                "bin USD not yet prepared (omni.usd unavailable at import time), "
                "using fallback metadata"
            )
            self.bin_width[:] = default["binWidth"]
            self.bin_height[:] = default["binHeight"]
            self.handle_height[:] = default["handleHeight"]
            self.wheel_radius[:] = default["wheelRadius"]
            self.push_axis_world[:] = push_axis_default[None, :]
            return

        try:
            stage: "Usd.Stage" = omni.usd.get_context().get_stage()
        except Exception as e:  # pragma: no cover
            logger.warning(
                f"bin USD not yet prepared (stage not available: {e}), "
                "using fallback metadata"
            )
            self.bin_width[:] = default["binWidth"]
            self.bin_height[:] = default["binHeight"]
            self.handle_height[:] = default["handleHeight"]
            self.wheel_radius[:] = default["wheelRadius"]
            self.push_axis_world[:] = push_axis_default[None, :]
            return

        any_env_failed = False
        for env_id in range(self.num_envs):
            bin_prim_path = f"/World/envs/env_{env_id}/bin"
            bin_prim = stage.GetPrimAtPath(bin_prim_path)
            if not bin_prim or not bin_prim.IsValid():
                any_env_failed = True
                self.bin_width[env_id] = default["binWidth"]
                self.bin_height[env_id] = default["binHeight"]
                self.handle_height[env_id] = default["handleHeight"]
                self.wheel_radius[env_id] = default["wheelRadius"]
                self.push_axis_world[env_id] = push_axis_default
                continue
            metadata = bin_prim.GetPrim().GetMetadata("customData") or {}
            self.bin_width[env_id] = metadata.get("binWidth", default["binWidth"])
            self.bin_height[env_id] = metadata.get("binHeight", default["binHeight"])
            self.handle_height[env_id] = metadata.get(
                "handleHeight", default["handleHeight"]
            )
            self.wheel_radius[env_id] = metadata.get(
                "wheelRadius", default["wheelRadius"]
            )
            axis = metadata.get("pushAxisWorld", default["pushAxisWorld"])
            self.push_axis_world[env_id] = torch.tensor(
                tuple(axis), device=self.device, dtype=torch.float32
            )

        if any_env_failed:
            logger.warning(
                "bin USD not yet prepared for at least one env, using fallback metadata"
            )

    def _init_buffers(self):
        super()._init_buffers()
        self.relative_bin_pos_buf = torch.zeros(self.num_envs, 3, device=self.device)
        self.relative_bin_rot_buf = torch.zeros(self.num_envs, 4, device=self.device)
        # bin root state cache: (pos_xyz, rot_xyzw, lin_vel_xyz, ang_vel_xyz) = 13
        self.bin_root_state_buf = torch.zeros(self.num_envs, 13, device=self.device)
        self._grasp_valid_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._push_invalid_grasp_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._reach_transition_ready_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._approach_transition_ready_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._left_finger_close_latched = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._right_finger_close_latched = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        # Run 24 (2026-08-11): previous-step twist error, for the progress
        # (potential-based) twist reward below. Initial value is irrelevant —
        # _reward_palm_twist_progress masks out the first step of every
        # episode (episode_length_buf <= 1) before this has a meaningful
        # "previous" value to compare against.
        self._prev_left_twist = torch.full(
            (self.num_envs,), math.pi, device=self.device
        )
        self._prev_right_twist = torch.full(
            (self.num_envs,), math.pi, device=self.device
        )
        # Run 25 (2026-08-13): previous-step vertical palm-to-handle error, for
        # the progress (potential-based) vertical reward below. Same masking
        # as _prev_left_twist above — initial value is irrelevant.
        self._prev_left_vert_error = torch.full(
            (self.num_envs,), 1.0, device=self.device
        )
        self._prev_right_vert_error = torch.full(
            (self.num_envs,), 1.0, device=self.device
        )
        # 2026-08-29: previous-step RAW palm-to-bar distance (not the offset
        # hover target), for the progress (potential-based) bar-distance
        # reward below. Same masking pattern as _prev_left_twist above.
        # Added because grasp_target_distance/anisotropic only ever pull
        # toward the offset hover target -- with no continuous reward
        # toward the true bar, the policy plateaued at its offset-target
        # equilibrium (~0.10-0.11m) well short of grasp_bar_valid_dist
        # (0.03m), confirmed via a 150-iteration flat trend. Mirrors
        # _reward_grasp_vert_progress, added earlier for an analogous
        # "dense reward alone isn't enough" plateau on vertical error.
        self._prev_left_bar_dist = torch.full(
            (self.num_envs,), 1.0, device=self.device
        )
        self._prev_right_bar_dist = torch.full(
            (self.num_envs,), 1.0, device=self.device
        )
        # 2026-09-29 overhand: consecutive TILT steps with the bin inside the
        # tilt band while the grip is maintained (TILT->PUSH gate), and the
        # previous-step tilt potential for _reward_bin_tilt_progress (same
        # first-step masking pattern as _prev_left_twist above).
        self._tilt_hold_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._prev_tilt_potential = torch.zeros(self.num_envs, device=self.device)
        # 2026-09-30 overhand grasp_init: True for envs whose current episode
        # started from the canned grasped state (directly in TILT). Cleared on
        # every reset in _reset_tasks_callback, set in
        # _reset_robot_states_callback (stage-0 resets only).
        self._grasp_init_mask = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        # 2026-10-01 success metrics (logging only, see _record_episode_outcomes):
        # stage each episode started in (set_to_stage) and a ring buffer of
        # the last `success_metrics_window` ended episodes.
        self._ep_start_stage = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        w = int(self.config.get("success_metrics_window", 512))
        self._succ_win = w
        self._succ_ptr = 0
        self._succ_count = 0
        # columns: completed, entered_grasp, grasp_reached_tilt, entered_tilt,
        # time_at_completion_s, started_at_or_before_grasp
        # 2026-10-03 push1: + reached_push, bin_body_mass_kg (logging only;
        # the buffer is not part of the env state dict, so old checkpoints load).
        # 2026-10-06 full3: + started_approach, reached_reach (columns 8, 9).
        self._succ_buf = torch.zeros(w, 10, device=self.device)
        # 2026-10-01 tilt3: bin body-mass curriculum (bin_mass_curriculum,
        # default False = the USD's fixed 15 kg body). See
        # _apply_bin_mass_curriculum / _update_bin_mass_curriculum. The level
        # is also written to log_dict["bin_mass_level_kg"]; legged_robot_base
        # saves log_dict as the env state dict and load_env_state_dict restores
        # attributes of the same name, so a resume keeps the level.
        self._bin_mass_cur = bool(self.config.get("bin_mass_curriculum", False))
        self.bin_mass_level_kg = torch.tensor(
            float(self.config.get("bin_mass_start_kg", 15.0)), device=self.device
        )
        self._bin_mass_ep_level = torch.full((self.num_envs,), -1.0, device=self.device)
        self._bin_mass_env = torch.full((self.num_envs,), 15.0, device=self.device)
        self._bin_mass_lvl_eps = 0
        self._bin_mass_lvl_succ = 0
        # 2026-10-03 tilt7: steady-hold TILT->PUSH gate (tilt_advance_max_rate_dps,
        # default None = no rate condition = old behaviour). The hold counter and
        # the gate step also require |d tilt/dt| (_bin_tilt_rate_dps) <= the
        # episode's threshold _tilt_rate_thr_env, copied from
        # tilt_gate_max_rate_dps at every reset. With tilt_rate_curriculum_start_dps
        # the threshold starts there and _update_tilt_rate_curriculum tightens it
        # down to tilt_advance_max_rate_dps; without it, it is fixed at
        # tilt_advance_max_rate_dps. tilt_gate_max_rate_dps is also written to
        # log_dict (= the env state dict), so a resume restores it (same
        # mechanism as bin_mass_level_kg; re-clamped in load_env_state_dict).
        rate_floor = self.config.get("tilt_advance_max_rate_dps", None)
        rate_start = self.config.get("tilt_rate_curriculum_start_dps", None)
        self._tilt_rate_gate = rate_floor is not None
        self._tilt_rate_cur = self._tilt_rate_gate and rate_start is not None
        if self._tilt_rate_cur:
            thr0 = max(float(rate_start), float(rate_floor))
        elif self._tilt_rate_gate:
            thr0 = float(rate_floor)
        else:
            thr0 = float("inf")
        self.tilt_gate_max_rate_dps = torch.tensor(thr0, device=self.device)
        self._tilt_rate_thr_env = torch.full((self.num_envs,), thr0, device=self.device)
        self._tilt_rate_lvl_eps = 0
        self._tilt_rate_lvl_succ = 0
        # 2026-10-03 tilt7 (logging only): ring buffer of the last
        # tilt_gate_log_window (1024) TILT->PUSH handovers: tilt rate (deg/s,
        # signed) and tilt (deg) at the handover step. Not in the env state dict.
        gw = int(self.config.get("tilt_gate_log_window", 1024))
        self._gate_win = gw
        self._gate_ptr = 0
        self._gate_count = 0
        self._gate_buf = torch.zeros(gw, 2, device=self.device)
        # 2026-10-03 push2: PUSH target distance curriculum
        # (push_distance_curriculum, default False = fixed target_bin_push_distance).
        # Saved in log_dict["target_bin_push_distance_m"] (= env state dict) so
        # resumes keep the current level; privileged_bin_info obs keeps reading
        # self.target_bin_push_distance (1.0) so actor obs never shifts.
        self._push_dist_cur = bool(self.config.get("push_distance_curriculum", False))
        pdist0 = (
            float(self.config.get("push_distance_start_m", 0.10))
            if self._push_dist_cur
            else float(self.config.get("target_bin_push_distance", 1.0))
        )
        self.target_bin_push_distance_m = torch.tensor(pdist0, device=self.device)
        self._push_dist_ep_level = torch.full((self.num_envs,), pdist0, device=self.device)
        self._push_dist_lvl_eps = 0
        self._push_dist_lvl_succ = 0
        # 2026-10-06 full3: polar STAGE_APPROACH spawn DR & curriculum
        # (spawn_polar_dr, default False = rectangular spawn_x_range / spawn_y_range).
        # Saved in log_dict["spawn_dr_dist_level_m"] (= env state dict) so resumes
        # keep the current level.
        self._spawn_polar_dr = bool(self.config.get("spawn_polar_dr", False))
        self._spawn_polar_cur = (
            bool(self.config.get("spawn_polar_curriculum", False)) and self._spawn_polar_dr
        )
        sdist0 = (
            float(self.config.get("spawn_polar_dist_start_m", 0.30))
            if self._spawn_polar_cur
            else float(self.config.get("spawn_polar_dist_max_m", 1.10))
        )
        self.spawn_dr_dist_level_m = torch.tensor(sdist0, device=self.device)
        self._spawn_dr_ep_level = torch.full((self.num_envs,), sdist0, device=self.device)
        self._spawn_dr_lvl_eps = 0
        self._spawn_dr_lvl_reach = 0
        self._spawn_dr_lvl_succ = 0
        self._spawn_dr_init_dist = torch.zeros(self.num_envs, device=self.device)
        self._spawn_dr_init_phi_deg = torch.zeros(self.num_envs, device=self.device)
        self._spawn_dr_init_yaw_deg = torch.zeros(self.num_envs, device=self.device)
        self._spawn_bin_yaw_rad = torch.zeros(self.num_envs, device=self.device)

    # 2026-09-29 overhand: stage-mask helpers (see _effective_in_named_stages).
    def _named_stage_mask(self, names):
        mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for name in names:
            stage = getattr(self, f"STAGE_{name}")
            if stage >= 0:
                mask |= self.stage_buf == stage
        return mask

    def _holding_stage_mask(self, stage_buf=None):
        """True in the stages where the bar must stay grasped (TILT, PUSH).
        == STAGE_PUSH only when the TILT stage is disabled (STAGE_TILT=-1)."""
        stage_buf = self.stage_buf if stage_buf is None else stage_buf
        return (stage_buf == self.STAGE_PUSH) | (stage_buf == self.STAGE_TILT)

    def _push_motion_anti_stall_weight(self):
        """2026-10-04 push6 (push_anti_stall_enable, default False): in
        STAGE_PUSH, after a brief settle grace window (push_stall_grace_steps,
        default 25 = 0.50 s), ramp in (over push_stall_ramp_steps, default 15 =
        0.30 s) a forward-velocity gate based on min(pelvis_fwd_vel,
        bin_root_fwd_vel, bar_fwd_vel). Standing still or pushing only with the
        arms (pelvis_fwd_vel <= push_stall_min_vel) scales static holding
        rewards down to push_stall_hold_floor (0.15); walking forward with the
        bin at >= push_stall_full_vel (0.12 m/s) earns 1.0."""
        if not bool(self.config.get("push_anti_stall_enable", False)):
            return torch.ones(self.num_envs, device=self.device)
        root_vel = self.simulator.robot_root_states[:, 7:10].clone()
        root_vel[:, 2] = 0.0
        axis_unit = F.normalize(self.push_axis_world, dim=-1)
        pelvis_fwd = (root_vel * axis_unit).sum(dim=-1)
        bin_vel = self.simulator.get_task_root_state("bin")[:, 7:10]
        bin_fwd = torch.minimum((bin_vel * axis_unit).sum(dim=-1), self._bar_fwd_vel())
        v_eff = torch.minimum(pelvis_fwd, bin_fwd)
        v_min = float(self.config.get("push_stall_min_vel", 0.02))
        v_full = max(float(self.config.get("push_stall_full_vel", 0.12)), v_min + 1e-3)
        floor = float(self.config.get("push_stall_hold_floor", 0.15))
        w_vel = floor + (1.0 - floor) * ((v_eff - v_min) / (v_full - v_min)).clamp(0.0, 1.0)
        grace = float(self.config.get("push_stall_grace_steps", 25.0))
        ramp = max(float(self.config.get("push_stall_ramp_steps", 15.0)), 1e-3)
        alpha = ((self.time_in_stage_buf.float() - grace) / ramp).clamp(0.0, 1.0)
        w_push = 1.0 - alpha * (1.0 - w_vel)
        return torch.where(
            self.stage_buf == self.STAGE_PUSH, w_push, torch.ones_like(w_push)
        )

    def _grasp_shape_stage_scale(self):
        """2026-10-03 push4: optional multiplier (grasp_shape_in_tilt_push_scale,
        default 1.0 = unchanged) applied to the static grasp-acquisition shaping
        terms once the episode is in TILT or PUSH. Scaling them down (e.g. 0.15)
        in TILT/PUSH removes the +0.74/step static survival-farming baseline in
        TILT and frees wrist compliance during tilting/pushing, while
        maintain_grasp_while_pushing continues to reward keeping both hands on
        the bar."""
        scale = float(self.config.get("grasp_shape_in_tilt_push_scale", 1.0))
        if scale == 1.0:
            return 1.0
        return torch.where(
            self._holding_stage_mask(),
            torch.full((self.num_envs,), scale, device=self.device),
            torch.ones(self.num_envs, device=self.device),
        )

    # ================================================================
    # OBSERVATIONS
    # ================================================================

    def _pre_compute_observations_callback(self, env_ids=None):
        super()._pre_compute_observations_callback(env_ids)
        env_ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids

        current_root_pos = self.simulator.robot_root_states[env_ids, :3].clone()
        current_root_rot = self.simulator.robot_root_states[env_ids, 3:7].clone()
        current_root_rot_wxyz = xyzw_to_wxyz(current_root_rot)

        bin_root_pos = self.simulator.get_task_root_state("bin")[env_ids, :3].clone()
        bin_root_pos[:, 2] = current_root_pos[:, 2]
        bin_root_rot_wxyz = self.simulator.get_task_root_state("bin")[env_ids, 3:7].clone()

        relative_pos, relative_rot = subtract_frame_transforms(
            current_root_pos, current_root_rot_wxyz, bin_root_pos, bin_root_rot_wxyz
        )
        self.relative_bin_pos_buf[env_ids] = relative_pos
        self.relative_bin_rot_buf[env_ids] = wxyz_to_xyzw(relative_rot)

        self._update_grasp_validity_counters(env_ids)
        self._update_custom_logs()

    def _update_custom_logs(self):
        """Track the sub-conditions that gate reach and grasp transitions."""
        # 2026-08-27: _finger_control_masks() no longer drives any action
        # (see _postprocess_delta_actions's docstring) -- called here purely
        # for its side effect of updating _left/right_finger_close_latched,
        # which grasp_left/right_close_latched_frac_s2 below still logs.
        # Without this call the latches would freeze at their reset value
        # (False) forever, since this was their only remaining updater.
        self._finger_control_masks()
        # 2026-09-29 overhand: num_stages (5 with enable_tilt_stage).
        for stage in range(self.num_stages):
            self.log_dict[f"stage_{stage}_frac"] = (self.stage_buf == stage).float().mean()
        # 2026-09-30 overhand grasp_init: fraction of envs currently running an
        # episode that started from the canned grasped state.
        self.log_dict["grasp_init_frac"] = self._grasp_init_mask.float().mean()
        for dbg_step in self.config.get("grasp_init_debug_steps", None) or []:
            dbg_ids = torch.where(
                self._grasp_init_mask & (self.episode_length_buf == int(dbg_step))
            )[0]
            if len(dbg_ids) > 0:
                self._grasp_init_debug_report(int(dbg_step), dbg_ids)
        if self.config.get("grasp_init_debug_csv", None):
            self._grasp_init_debug_csv_row()

        left_palm, right_palm = self._palm_grasp_points()
        ref_left, ref_right = self._grasp_reference_targets()
        left_delta = ref_left - left_palm
        right_delta = ref_right - right_palm
        left_dist = left_delta.norm(dim=-1)
        right_dist = right_delta.norm(dim=-1)
        both_close = (left_dist < 0.15) & (right_dist < 0.15)

        left_finger_dev = torch.abs(
            self.simulator.dof_pos[:, self._left_hand_dof_idx] - self._left_p0
        ).mean(dim=-1)
        right_finger_dev = torch.abs(
            self.simulator.dof_pos[:, self._right_hand_dof_idx] - self._right_p0
        ).mean(dim=-1)
        fingers_open = (left_finger_dev < 0.174533) & (right_finger_dev < 0.174533)
        homie_still = torch.norm(self.get_physical_homie_commands()[:, :3], dim=1) <= 0.1

        signed_standoff, standoff_error, lateral_error = self._grasp_standoff_state()
        stage0 = self.stage_buf == self.STAGE_APPROACH
        approach_mask = stage0 if stage0.any() else torch.ones_like(stage0)
        root_speed = self.simulator.robot_root_states[:, 7:10].norm(dim=-1)
        upper_deviation = torch.abs(
            self.simulator.dof_pos[:, self._upper_non_finger_dof_idx]
            - self.resting_dof_pos[:, self._upper_non_finger_dof_idx]
        ).max(dim=-1).values
        approach_stance_ready = self._standoff_ready_mask()
        approach_yaw_ready = self._approach_yaw_ready_mask()
        approach_yaw_err_deg = torch.rad2deg(self._approach_yaw_error_rad().abs())
        approach_speed_ready = root_speed < float(
            self.config.get("stage_advance_vel", 0.2)
        )
        arm_tol = float(self.config.get("stage_advance_arm_tolerance", 0.3))
        approach_arms_ready = upper_deviation < arm_tol
        approach_min_standoff = float(
            self.config.get("approach_min_standoff", 0.40)
        )
        self.log_dict["approach_standoff_s0"] = signed_standoff[
            approach_mask
        ].mean()
        self.log_dict["approach_standoff_error_s0"] = standoff_error[
            approach_mask
        ].mean()
        self.log_dict["approach_lateral_error_s0"] = lateral_error[
            approach_mask
        ].mean()
        self.log_dict["approach_yaw_error_deg_s0"] = approach_yaw_err_deg[
            approach_mask
        ].mean()
        self.log_dict["approach_yaw_ready_frac_s0"] = approach_yaw_ready[
            approach_mask
        ].float().mean()
        self.log_dict["approach_homie_cmd_norm_s0"] = (
            self.get_physical_homie_commands()[approach_mask, :3].norm(dim=-1).mean()
        )
        self.log_dict["approach_too_close_frac_s0"] = (
            signed_standoff < approach_min_standoff
        )[approach_mask].float().mean()
        self.log_dict["approach_stance_ready_frac_s0"] = approach_stance_ready[
            approach_mask
        ].float().mean()
        self.log_dict["approach_speed_ready_frac_s0"] = approach_speed_ready[
            approach_mask
        ].float().mean()
        self.log_dict["approach_arms_ready_frac_s0"] = approach_arms_ready[
            approach_mask
        ].float().mean()
        self.log_dict["approach_max_arm_dev_s0"] = upper_deviation[
            approach_mask
        ].mean()
        self.log_dict["approach_transition_ready_frac_s0"] = (
            approach_stance_ready
            & approach_yaw_ready
            & approach_speed_ready
            & approach_arms_ready
        )[approach_mask].float().mean()
        if getattr(self, "_spawn_polar_dr", False):
            _, d_lvl, phi_lvl, dpsi_lvl = self._spawn_polar_active_bounds()
            self.log_dict["spawn_dr_dist_level_m"] = self.spawn_dr_dist_level_m.clone()
            self.log_dict["spawn_dr_phi_level_deg"] = torch.tensor(
                phi_lvl, device=self.device
            )
            self.log_dict["spawn_dr_fov_yaw_level_deg"] = torch.tensor(
                dpsi_lvl, device=self.device
            )
            self.log_dict["spawn_dr_init_dist_mean_m"] = self._spawn_dr_init_dist.mean()
            self.log_dict["spawn_dr_init_phi_abs_deg"] = self._spawn_dr_init_phi_deg.abs().mean()
            self.log_dict["spawn_dr_init_yaw_abs_deg"] = self._spawn_dr_init_yaw_deg.abs().mean()
            self.log_dict["spawn_dr_bin_yaw_abs_deg"] = torch.rad2deg(
                self._spawn_bin_yaw_rad.abs()
            ).mean()

        stage1 = self.stage_buf == self.STAGE_REACH
        metric_mask = stage1 if stage1.any() else torch.ones_like(stage1)
        self.log_dict["reach_left_dist_s1"] = left_dist[metric_mask].mean()
        self.log_dict["reach_right_dist_s1"] = right_dist[metric_mask].mean()
        self.log_dict["reach_left_horiz_s1"] = left_delta[metric_mask, :2].norm(dim=-1).mean()
        self.log_dict["reach_right_horiz_s1"] = right_delta[metric_mask, :2].norm(dim=-1).mean()
        self.log_dict["reach_left_vert_s1"] = left_delta[metric_mask, 2].abs().mean()
        self.log_dict["reach_right_vert_s1"] = right_delta[metric_mask, 2].abs().mean()
        self.log_dict["reach_both_close_frac_s1"] = both_close[metric_mask].float().mean()
        self.log_dict["reach_fingers_open_frac_s1"] = fingers_open[metric_mask].float().mean()
        self.log_dict["reach_homie_still_frac_s1"] = homie_still[metric_mask].float().mean()
        reach_ready = self._reach_transition_ready_mask()
        self.log_dict["reach_transition_ready_frac_s1"] = reach_ready[
            metric_mask
        ].float().mean()
        self.log_dict["reach_transition_held_frac_s1"] = (
            self._reach_transition_ready_steps
            >= int(self.config.get("reach_transition_hold_steps", 5))
        )[metric_mask].float().mean()
        left_alignment, right_alignment = self._palm_down_alignments()
        reach_alignment_threshold = float(
            self.config.get("reach_transition_alignment_threshold", 0.0)
        )
        self.log_dict["reach_left_palm_down_s1"] = left_alignment[
            metric_mask
        ].mean()
        self.log_dict["reach_right_palm_down_s1"] = right_alignment[
            metric_mask
        ].mean()
        self.log_dict["reach_both_aligned_frac_s1"] = (
            (left_alignment > reach_alignment_threshold)
            & (right_alignment > reach_alignment_threshold)
        )[metric_mask].float().mean()
        left_twist, right_twist = self._palm_twist_errors()
        self.log_dict["reach_left_twist_rad_s1"] = left_twist[metric_mask].mean()
        self.log_dict["reach_right_twist_rad_s1"] = right_twist[metric_mask].mean()
        min_standoff = float(self.config.get("min_grasp_standoff", 0.34))
        self.log_dict["reach_standoff_s1"] = signed_standoff[metric_mask].mean()
        self.log_dict["reach_standoff_error_s1"] = standoff_error[
            metric_mask
        ].mean()
        self.log_dict["reach_lateral_error_s1"] = lateral_error[
            metric_mask
        ].mean()
        self.log_dict["reach_too_close_frac_s1"] = (
            signed_standoff < min_standoff
        )[metric_mask].float().mean()

        (
            left_contact,
            right_contact,
            left_grasp_close,
            right_grasp_close,
            left_closed,
            right_closed,
        ) = self._grasp_transition_components()
        stage2 = self.stage_buf == self.STAGE_GRASP
        contact_mask = stage2 if stage2.any() else torch.ones_like(stage2)
        close_horiz = float(
            self.config.get("grasp_finger_close_horiz_dist", 0.04)
        )
        close_vert = float(
            self.config.get("grasp_finger_close_vert_dist", 0.03)
        )
        left_horiz = left_delta[:, :2].norm(dim=-1)
        right_horiz = right_delta[:, :2].norm(dim=-1)
        left_abs_vert = left_delta[:, 2].abs()
        right_abs_vert = right_delta[:, 2].abs()
        left_horiz_ok = left_horiz < close_horiz
        right_horiz_ok = right_horiz < close_horiz
        left_vert_ok = left_abs_vert < close_vert
        right_vert_ok = right_abs_vert < close_vert
        self.log_dict["grasp_left_horiz_s2"] = left_horiz[contact_mask].mean()
        self.log_dict["grasp_right_horiz_s2"] = right_horiz[contact_mask].mean()
        self.log_dict["grasp_left_abs_vert_s2"] = left_abs_vert[contact_mask].mean()
        self.log_dict["grasp_right_abs_vert_s2"] = right_abs_vert[contact_mask].mean()
        self.log_dict["grasp_left_horiz_ok_frac_s2"] = left_horiz_ok[
            contact_mask
        ].float().mean()
        self.log_dict["grasp_right_horiz_ok_frac_s2"] = right_horiz_ok[
            contact_mask
        ].float().mean()
        self.log_dict["grasp_left_vert_ok_frac_s2"] = left_vert_ok[
            contact_mask
        ].float().mean()
        self.log_dict["grasp_right_vert_ok_frac_s2"] = right_vert_ok[
            contact_mask
        ].float().mean()
        self.log_dict["grasp_both_position_ready_frac_s2"] = (
            left_horiz_ok & right_horiz_ok & left_vert_ok & right_vert_ok
        )[contact_mask].float().mean()
        # 2026-08-29: renamed from grasp_left/right_contacts_s2 -- now logs
        # raw palm-to-bar distance (meters), not a PhysX contact-link count.
        # See _hand_bar_distances docstring.
        left_bar_dists, right_bar_dists = self._hand_bar_distances()
        self.log_dict["grasp_left_bar_dist_s2"] = left_bar_dists[contact_mask].mean()
        self.log_dict["grasp_right_bar_dist_s2"] = right_bar_dists[contact_mask].mean()

        # --- SINGLE SOURCE OF TRUTH: every hand-to-handle distance, side by side
        # (2026-09-12). Added because the project spent ~5 runs tuning against
        # two silently different reference points (see _grasp_reference_targets).
        # Every metric below is named <from>_to_<to>, so which point a number
        # refers to is never ambiguous again:
        #   grasppoint = _palm_grasp_points() (palm link + palm_grasp_point_offset
        #                along the finger-cavity axis) -- what rewards/gates score
        #   palmlink   = raw palm_link origin -- what the old obs reported
        #   bar        = _left/right_bar_raw_pos() -- the true bar, the thing to grasp
        #   hover      = _left/right_handle_target() -- REACH waypoint, bar + offset
        # `dist_hover_to_bar` is a geometric constant (grasp_target_standoff +
        # height_offset, ~0.0412) and exists purely as a sanity check: if it ever
        # drifts, a transformer offset changed underneath us.
        left_hover, right_hover = self._left_handle_target(), self._right_handle_target()
        left_bar, right_bar = self._left_bar_raw_pos(), self._right_bar_raw_pos()
        left_palm_link = self.simulator._rigid_body_pos[:, self.left_palm_idx, :]
        right_palm_link = self.simulator._rigid_body_pos[:, self.right_palm_idx, :]
        for side, gp, plink, hover, bar in (
            ("left", left_palm, left_palm_link, left_hover, left_bar),
            ("right", right_palm, right_palm_link, right_hover, right_bar),
        ):
            gp_to_bar = bar - gp
            self.log_dict[f"dist_{side}_grasppoint_to_bar_s2"] = (
                gp_to_bar.norm(dim=-1)[contact_mask].mean()
            )
            self.log_dict[f"dist_{side}_grasppoint_to_bar_horiz_s2"] = (
                gp_to_bar[:, :2].norm(dim=-1)[contact_mask].mean()
            )
            self.log_dict[f"dist_{side}_grasppoint_to_bar_vert_s2"] = (
                gp_to_bar[:, 2].abs()[contact_mask].mean()
            )
            self.log_dict[f"dist_{side}_grasppoint_to_hover_s2"] = (
                (hover - gp).norm(dim=-1)[contact_mask].mean()
            )
            # 2026-09-30: per-hand distance to the grasp-point target (where
            # the grasp point sits with the palm link at the handle target
            # pose; scored by grasp_bar_distance/progress when
            # grasp_bar_reward_target="grasp_target"), and that target's own
            # offset from the bar (geometric sanity check, expect ~0.0446).
            gpt = self._grasp_point_targets()[0 if side == "left" else 1]
            self.log_dict[f"grasp_target_dist_{side}_s2"] = (
                (gpt - gp).norm(dim=-1)[contact_mask].mean()
            )
            self.log_dict[f"dist_{side}_grasptarget_to_bar_s2"] = (
                (bar - gpt).norm(dim=-1)[contact_mask].mean()
            )
            self.log_dict[f"dist_{side}_palmlink_to_bar_s2"] = (
                (bar - plink).norm(dim=-1)[contact_mask].mean()
            )
            self.log_dict[f"dist_{side}_hover_to_bar_s2"] = (
                (bar - hover).norm(dim=-1)[contact_mask].mean()
            )

        # Distribution of the GRADED distance (worse hand), so grasp_bar_valid_dist
        # can be retuned from data instead of a guess -- the previous 0.03 -> 0.08
        # retune was measured against the mis-referenced target and mostly encoded
        # the size of the offset bug rather than what the arm can actually reach.
        worst_bar_dist = torch.maximum(left_bar_dists, right_bar_dists)[contact_mask]
        self.log_dict["dist_worst_grasppoint_to_bar_min_s2"] = worst_bar_dist.min()
        for q in (0.1, 0.25, 0.5):
            self.log_dict[f"dist_worst_grasppoint_to_bar_p{int(q * 100)}_s2"] = (
                torch.quantile(worst_bar_dist, q)
            )
        valid_dist = float(self.config.get("grasp_bar_valid_dist", 0.05))
        for probe in (0.03, 0.04, 0.05, 0.06, 0.08, 0.10):
            self.log_dict[f"dist_worst_under_{int(probe * 100):02d}cm_frac_s2"] = (
                (worst_bar_dist < probe).float().mean()
            )
        self.log_dict["dist_worst_under_gate_frac_s2"] = (
            (worst_bar_dist < valid_dist).float().mean()
        )
        # 2026-08-30: per GRASP_FLOW_SUGGESTIONS_2026-08-30.md hypothesis #3's
        # "concrete next check" -- logging _pregrasp_curl_fraction() alongside
        # bar_dist above, so it's directly visible whether the
        # grasp_bar_distance 6.0->9.0 bump actually moves bar_dist (and
        # therefore raises this fraction's ceiling, since
        # frac=(pregrasp_curl_start_dist-bar_dist)/pregrasp_curl_start_dist)
        # or whether bar_dist stays flat despite the bump -- which would
        # point at a different fix (shrinking grasp_target_standoff/height_
        # offset so the competing offset-hover-target pull converges with the
        # raw-bar pull, rather than a further weight-race).
        left_curl_frac, right_curl_frac = self._pregrasp_curl_fraction()
        self.log_dict["grasp_left_curl_frac_s2"] = left_curl_frac[contact_mask].mean()
        self.log_dict["grasp_right_curl_frac_s2"] = right_curl_frac[contact_mask].mean()
        # 2026-09-28 handlefix5e: direct posture + measured-closure logging.
        _jn = self.simulator.dof_names
        _q = self.simulator.dof_pos
        _deg = 57.29578
        for _key, _joint in (
            ("waist_pitch_deg_s2", "waist_pitch_joint"),
            ("left_shoulder_yaw_deg_s2", "left_shoulder_yaw_joint"),
            ("left_wrist_roll_deg_s2", "left_wrist_roll_joint"),
            ("right_shoulder_yaw_deg_s2", "right_shoulder_yaw_joint"),
            ("right_wrist_roll_deg_s2", "right_wrist_roll_joint"),
        ):
            self.log_dict[_key] = _q[contact_mask, _jn.index(_joint)].mean() * _deg
        _reach = self.stage_buf == self.STAGE_REACH
        if _reach.any():
            self.log_dict["left_shoulder_yaw_deg_s1"] = (
                _q[_reach, _jn.index("left_shoulder_yaw_joint")].mean() * _deg
            )

        def _closure(idx, p0, p1):
            span = (p1 - p0)[None, :]
            return ((_q[:, idx] - p0[None, :]) * span).sum(-1) / span.square().sum(-1).clamp(min=1e-6)

        _lc = _closure(self._left_hand_dof_idx, self._left_p0, self._left_p1)
        _rc = _closure(self._right_hand_dof_idx, self._right_p0, self._right_p1)
        self.log_dict["grasp_left_closure_s2"] = _lc[contact_mask].mean()
        self.log_dict["grasp_right_closure_s2"] = _rc[contact_mask].mean()
        _gate = float(self.config.get("grasp_bar_valid_dist", 0.05))
        self.log_dict["grasp_left_close_outside_gate_frac_s2"] = (
            (_lc > 0.35) & (left_bar_dists > _gate)
        )[contact_mask].float().mean()
        self.log_dict["grasp_right_close_outside_gate_frac_s2"] = (
            (_rc > 0.35) & (right_bar_dists > _gate)
        )[contact_mask].float().mean()
        self.log_dict["grasp_both_required_contacts_frac_s2"] = (
            left_contact & right_contact
        )[contact_mask].float().mean()
        self.log_dict["grasp_both_close_frac_s2"] = (
            left_grasp_close & right_grasp_close
        )[contact_mask].float().mean()
        self.log_dict["grasp_both_closed_frac_s2"] = (
            left_closed & right_closed
        )[contact_mask].float().mean()
        self.log_dict["grasp_transition_ready_frac_s2"] = (
            left_contact
            & right_contact
            & left_grasp_close
            & right_grasp_close
            & left_closed
            & right_closed
        )[contact_mask].float().mean()
        alignment_threshold = float(
            self.config.get("grasp_palm_alignment_threshold", 0.5)
        )
        contact_alignment_threshold = float(
            self.config.get("grasp_contact_alignment_threshold", 0.0)
        )
        self.log_dict["grasp_left_palm_down_s2"] = left_alignment[
            contact_mask
        ].mean()
        self.log_dict["grasp_right_palm_down_s2"] = right_alignment[
            contact_mask
        ].mean()
        self.log_dict["grasp_both_aligned_frac_s2"] = (
            (left_alignment > alignment_threshold)
            & (right_alignment > alignment_threshold)
        )[contact_mask].float().mean()
        max_twist = float(self.config.get("grasp_finger_close_max_twist_rad", 1.0472))
        self.log_dict["grasp_left_twist_rad_s2"] = left_twist[contact_mask].mean()
        self.log_dict["grasp_right_twist_rad_s2"] = right_twist[contact_mask].mean()
        self.log_dict["grasp_both_twist_ok_frac_s2"] = (
            (left_twist < max_twist) & (right_twist < max_twist)
        )[contact_mask].float().mean()
        self.log_dict["grasp_left_wrong_orientation_contact_frac_s2"] = (
            left_contact & (left_alignment <= contact_alignment_threshold)
        )[contact_mask].float().mean()
        self.log_dict["grasp_right_wrong_orientation_contact_frac_s2"] = (
            right_contact & (right_alignment <= contact_alignment_threshold)
        )[contact_mask].float().mean()
        self.log_dict["grasp_any_wrong_orientation_contact_frac_s2"] = (
            (left_contact & (left_alignment <= contact_alignment_threshold))
            | (right_contact & (right_alignment <= contact_alignment_threshold))
        )[contact_mask].float().mean()
        left_close_ready, right_close_ready = self._finger_close_ready()
        self.log_dict["grasp_left_finger_close_ready_frac_s2"] = left_close_ready[
            contact_mask
        ].float().mean()
        self.log_dict["grasp_right_finger_close_ready_frac_s2"] = right_close_ready[
            contact_mask
        ].float().mean()
        self.log_dict["grasp_both_finger_close_ready_frac_s2"] = (
            left_close_ready & right_close_ready
        )[contact_mask].float().mean()
        self.log_dict["grasp_left_close_latched_frac_s2"] = (
            self._left_finger_close_latched[contact_mask].float().mean()
        )
        self.log_dict["grasp_right_close_latched_frac_s2"] = (
            self._right_finger_close_latched[contact_mask].float().mean()
        )
        self.log_dict["grasp_full_pose_valid_frac_s2"] = self._grasp_pose_valid_mask()[
            contact_mask
        ].float().mean()
        # 2026-10-01: fraction of GRASP-stage envs passing the
        # grasp_gate_min_closure part of the gate (both hands); only when set.
        _gate_min_closure = self.config.get("grasp_gate_min_closure", None)
        if _gate_min_closure is not None:
            _glc, _grc = self._finger_closure_fractions()
            self.log_dict["grasp_gate_closure_ok_frac_s2"] = (
                (_glc >= float(_gate_min_closure)) & (_grc >= float(_gate_min_closure))
            )[contact_mask].float().mean()
        signed_standoff, _, _ = self._grasp_standoff_state()
        min_standoff = float(self.config.get("min_grasp_standoff", 0.34))
        self.log_dict["grasp_standoff_s2"] = signed_standoff[contact_mask].mean()
        self.log_dict["grasp_too_close_frac_s2"] = (
            signed_standoff < min_standoff
        )[contact_mask].float().mean()
        hold_steps = int(self.config.get("grasp_transition_hold_steps", 8))
        self.log_dict["grasp_sustained_ready_frac_s2"] = (
            self._grasp_valid_steps >= hold_steps
        )[contact_mask].float().mean()
        # 2026-08-30: per GRASP_FLOW_SUGGESTIONS_2026-08-30.md hypothesis #4's
        # "concrete next experiment" -- bucket the existing _grasp_valid_steps
        # counter (already computed above, just not previously logged as a
        # distribution) to check whether a substantial mass of envs are
        # reaching 4-7 consecutive under-threshold steps and then resetting
        # (evidence the 8-consecutive-step hold requirement itself is
        # discarding near-successes to bar_dist noise) vs. reaching 0-2 and
        # clearly not close (evidence the hold length isn't the bottleneck).
        # near_miss = "at least half the required streak, but hasn't reached
        # it" -- a real mass here alongside a low grasp_sustained_ready_frac_
        # s2 above is the signal to try grasp_transition_hold_steps: 8 -> 4
        # as an isolated change, per that hypothesis's own methodology.
        self.log_dict["grasp_valid_steps_near_miss_frac_s2"] = (
            (self._grasp_valid_steps >= hold_steps // 2)
            & (self._grasp_valid_steps < hold_steps)
        )[contact_mask].float().mean()
        self.log_dict["grasp_valid_steps_mean_s2"] = self._grasp_valid_steps[
            contact_mask
        ].float().mean()

        stage3 = self.stage_buf == self.STAGE_PUSH
        push_mask = stage3 if stage3.any() else torch.ones_like(stage3)
        self.log_dict["push_valid_grasp_frac_s3"] = self._grasp_pose_valid_mask()[
            push_mask
        ].float().mean()
        # 2026-09-25: PUSH observability (logging only). How far / fast the bin
        # moves while pushing, and how it tilts (positive tilt_back = top leaning
        # toward the robot = the human "tip onto the wheels" posture).
        bin_state = self.simulator.get_task_root_state("bin")
        progress = self._bin_push_progress()
        axis_xy = F.normalize(self.push_axis_world[:, :2], dim=-1)
        vel_along = (bin_state[:, 7:9] * axis_xy).sum(dim=-1)
        q = bin_state[:, 3:7]  # wxyz (see _palm_down_alignments)
        qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        up_z = (1.0 - 2.0 * (qx * qx + qy * qy)).clamp(-1.0, 1.0)
        up_xy = torch.stack(
            [2.0 * (qx * qz + qw * qy), 2.0 * (qy * qz - qw * qx)], dim=-1
        )
        tilt_deg = torch.rad2deg(torch.acos(up_z))
        tilt_back_deg = torch.rad2deg(
            torch.asin((-(up_xy * axis_xy).sum(dim=-1)).clamp(-1.0, 1.0))
        )
        self.log_dict["push_progress_m_s3"] = progress[push_mask].mean()
        self.log_dict["push_progress_max_m_s3"] = progress[push_mask].max()
        self.log_dict["push_vel_s3"] = vel_along[push_mask].mean()
        self.log_dict["bin_tilt_deg_s3"] = tilt_deg[push_mask].mean()
        self.log_dict["bin_tilt_back_deg_s3"] = tilt_back_deg[push_mask].mean()
        self.log_dict["bin_tilt_back_max_deg_s3"] = tilt_back_deg[push_mask].max()
        # 2026-09-25 handlefix3: wrap / lid-contact observability (GRASP+PUSH).
        # 2026-09-29 overhand: + TILT (the "_s23" suffix is kept for dashboard
        # continuity; it means GRASP + the grasp-holding stages).
        s23 = (self.stage_buf == self.STAGE_GRASP) | self._holding_stage_mask()
        s23 = s23 if s23.any() else torch.ones_like(s23)
        (wl, pl), (wr, pr) = self._finger_wrap_scores()
        lid_l, lid_r = self._lid_contact_per_hand()
        self.log_dict["wrap_worst_left_s23"] = wl.min(dim=-1).values[s23].mean()
        self.log_dict["wrap_worst_right_s23"] = wr.min(dim=-1).values[s23].mean()
        self.log_dict["wrap_mean_left_s23"] = wl.mean(dim=-1)[s23].mean()
        self.log_dict["wrap_mean_right_s23"] = wr.mean(dim=-1)[s23].mean()
        self.log_dict["tip_near_left_s23"] = pl.mean(dim=-1)[s23].mean()
        self.log_dict["tip_near_right_s23"] = pr.mean(dim=-1)[s23].mean()
        self.log_dict["lid_contact_left_s23"] = lid_l[s23].mean()
        self.log_dict["lid_contact_right_s23"] = lid_r[s23].mean()
        self.log_dict["lid_contact_any_frac_s23"] = ((lid_l + lid_r) > 0)[s23].float().mean()
        # 2026-09-30 reach_grasp1 fix: palm grasp point height above the bar
        # axis (bin frame, cm) and the fraction below grasp_palm_min_dz (0
        # when unset), REACH + GRASP.
        s12 = (self.stage_buf == self.STAGE_REACH) | (self.stage_buf == self.STAGE_GRASP)
        s12 = s12 if s12.any() else torch.ones_like(s12)
        min_dz = self._palm_min_dz() or 0.0
        for side, d in zip(("left", "right"), self._palm_bar_deltas_bin()):
            self.log_dict[f"palm_dz_{side}_s12"] = 100.0 * d[:, 2][s12].mean()
            self.log_dict[f"palm_below_frac_{side}_s12"] = (d[:, 2] <= min_dz)[s12].float().mean()
        self._update_overhand_tilt_logs(s23)

    def _update_overhand_tilt_logs(self, s23):
        """2026-09-29 overhand: overhand wrap score / per-finger bar distance,
        TILT-stage and grip-maintenance observability. Logging only."""
        # --- overhand wrap (GRASP + holding stages), per hand / per finger ---
        if self._overhand_mode():
            ovh = self._overhand_finger_terms()
            names = ("index", "middle", "thumb")
            for side, (score, dsurf_tip, dsurf_seg, hand_score, wrapped) in ovh.items():
                self.log_dict[f"ovh_score_{side}_s23"] = hand_score[s23].mean()
                self.log_dict[f"ovh_wrapped_frac_{side}_s23"] = wrapped[s23].float().mean()
                for k, fname in enumerate(names):
                    self.log_dict[f"ovh_{side}_{fname}_score_s23"] = score[:, k][s23].mean()
                    self.log_dict[f"ovh_{side}_{fname}_dsurf_cm_s23"] = (
                        100.0 * dsurf_seg[:, k][s23].mean()
                    )
            self.log_dict["ovh_both_wrapped_frac_s23"] = (
                ovh["left"][4] & ovh["right"][4]
            )[s23].float().mean()
        # --- closure on the p0 -> p1 axis ---
        lc, rc = self._finger_closure_fractions()
        self.log_dict["closure_left_s23"] = lc[s23].mean()
        self.log_dict["closure_right_s23"] = rc[s23].mean()
        # --- grip maintained (loose check, I3) per holding stage ---
        grip = self._grip_maintained_mask()
        theta = self._bin_tilt_back_deg()
        w_band = self._tilt_band_weight(theta)
        self.log_dict["bin_tilt_back_deg_all"] = theta.mean()
        lo = float(self.config.get("bin_topple_min_deg", -8.0))
        hi = float(self.config.get("bin_topple_max_deg", 30.0))
        self.log_dict["bin_topple_frac_all"] = ((theta < lo) | (theta > hi)).float().mean()
        if self._tilt_enabled:
            tilt = self.stage_buf == self.STAGE_TILT
            tmask = tilt if tilt.any() else torch.ones_like(tilt)
            band_lo = float(self.config.get("tilt_advance_min_deg", 10.0))
            band_hi = float(self.config.get("tilt_advance_max_deg", 22.0))
            self.log_dict["tilt_back_deg_tilt"] = theta[tmask].mean()
            self.log_dict["tilt_back_max_deg_tilt"] = theta[tmask].max()
            self.log_dict["tilt_in_band_frac_tilt"] = (
                (theta >= band_lo) & (theta <= band_hi)
            )[tmask].float().mean()
            self.log_dict["tilt_hold_steps_mean_tilt"] = self._tilt_hold_steps[
                tmask
            ].float().mean()
            # 2026-10-03 tilt7 (logging only): |tilt rate| of the TILT envs
            # inside the band, and the fraction of them slow enough for the
            # steady-hold gate (1.0 without tilt_advance_max_rate_dps). Kept
            # from the previous step when no TILT env is in the band.
            rate = self._bin_tilt_rate_dps()
            band = tilt & (theta >= band_lo) & (theta <= band_hi)
            if band.any():
                rb = rate[band].abs()
                self.log_dict["tilt_rate_abs_dps_band_tilt"] = rb.mean()
                self.log_dict["tilt_steady_frac_band_tilt"] = (
                    rb <= self._tilt_rate_thr_env[band]
                ).float().mean()
                # 2026-10-03 push2: steady-hold weight and forward vx command
                # inside the TILT band (tracks tilt_walk_ramp_in_band).
                self.log_dict["tilt_steady_weight_band_tilt"] = (
                    self._tilt_steady_weight()[band].mean()
                )
                self.log_dict["homie_vx_cmd_band_tilt"] = (
                    self.get_physical_homie_commands()[band, 0].mean()
                )
            if getattr(self, "_tilt_rate_gate", False):
                # also the env-state-dict copy restored on resume
                self.log_dict["tilt_gate_max_rate_dps"] = self.tilt_gate_max_rate_dps.clone()
            self.log_dict["grip_maintained_frac_tilt"] = grip[tmask].float().mean()
            # 2026-10-01: closure in TILT only (closure_*_s23 mixes GRASP +
            # TILT); reach_grasp2 loosened its grip after entering TILT.
            self.log_dict["closure_left_tilt"] = lc[tmask].mean()
            self.log_dict["closure_right_tilt"] = rc[tmask].mean()
            # 2026-10-01 tilt2: hand moment about the wheel contact line (+ =
            # tips back), its clamped ratio to bin_tilt_moment_ref_Nm (NOT
            # grip-gated, unlike the reward) and total hand-on-bin force.
            m_tilt, f_tilt = self._bin_tilt_moment_terms()
            ref_tilt = float(self.config.get("bin_tilt_moment_ref_Nm", self.BIN_TILT_MOMENT_REF_NM))
            self.log_dict["tilt_moment_Nm_tilt"] = m_tilt[tmask].mean()
            self.log_dict["tilt_moment_frac_tilt"] = (
                (m_tilt / max(ref_tilt, 1e-3)).clamp(0.0, 1.0)[tmask].mean()
            )
            self.log_dict["hand_force_N_tilt"] = f_tilt[tmask].mean()
            # 2026-10-01 tilt3: with bin_tilt_moment_friction the three logs
            # above include friction; keep the normal-only moment (what tilt1/
            # tilt2 logged and paid for) for comparison.
            if bool(self.config.get("bin_tilt_moment_friction", False)):
                f_nrm = self.BIN_TILT_MOMENT_FORCE_SIGN * self.simulator.object_to_hand_contact_forces[
                    :, 0, :, :
                ]
                self.log_dict["tilt_moment_normal_Nm_tilt"] = self._bin_hand_moment(f_nrm)[tmask].mean()
            # 2026-10-01 tilt2: body (non-hand/wrist/foot links) contact with
            # the bin in TILT: mean force and fraction above 50 N (chest brace).
            bf_tilt = self._body_bin_contact_force()
            self.log_dict["body_bin_contact_N_tilt"] = bf_tilt[tmask].mean()
            self.log_dict["body_bin_contact_frac_tilt"] = (bf_tilt > 50.0)[tmask].float().mean()
        push = self.stage_buf == self.STAGE_PUSH
        pmask = push if push.any() else torch.ones_like(push)
        self.log_dict["grip_maintained_frac_push"] = grip[pmask].float().mean()
        self.log_dict["w_tilt_push"] = w_band[pmask].mean()
        self.log_dict["push_force_nomoment_N_push"] = self._push_force_nomoment_raw()[
            pmask
        ].mean()
        self.log_dict["push_force_line_below_horiz_deg"] = torch.rad2deg(
            torch.asin((-self._push_line_dir[:, 2]).clamp(-1.0, 1.0))
        ).mean()
        self.log_dict["grip_invalid_steps_mean_hold"] = self._push_invalid_grasp_steps[
            s23
        ].float().mean()
        if getattr(self, "_push_dist_cur", False):
            # also the env-state-dict copy restored on resume
            self.log_dict["target_bin_push_distance_m"] = (
                self.target_bin_push_distance_m.clone()
            )
        # 2026-10-03 push1 (push_stage_logs, default False): PUSH-stage
        # observability over envs IN PUSH only (skipped while none are, so the
        # last value carries; the *_push logs above fall back to all envs).
        if bool(self.config.get("push_stage_logs", False)) and push.any():
            prog = self._bin_push_progress()
            bar_prog = self._bin_bar_push_progress()
            axis = F.normalize(self.push_axis_world[:, :2], dim=-1)
            root_v = (self.simulator.robot_root_states[:, 7:9] * axis).sum(dim=-1)
            bin_v = (self.simulator.get_task_root_state("bin")[:, 7:9] * axis).sum(dim=-1)
            bar_v = self._bar_fwd_vel()
            vx_cmd = self.get_physical_homie_commands()[:, 0]
            p_lo = float(self.config.get("push_complete_min_tilt_deg", 5.0))
            p_hi = float(self.config.get("push_complete_max_tilt_deg", 25.0))
            db = float(self.config.get("push_cmd_deadband", 0.1))
            self.log_dict["push_frac_envs"] = push.float().mean()
            self.log_dict["push_progress_m_push"] = prog[push].mean()
            self.log_dict["push_progress_max_m_push"] = prog[push].max()
            self.log_dict["push_bar_progress_m_push"] = bar_prog[push].mean()
            self.log_dict["tilt_back_deg_push"] = theta[push].mean()
            self.log_dict["tilt_in_band_frac_push"] = (
                (theta >= p_lo) & (theta <= p_hi)
            )[push].float().mean()
            self.log_dict["tilt_rate_abs_dps_push"] = (
                self._bin_tilt_rate_dps()[push].abs().mean()
            )
            self.log_dict["tilt_steady_weight_push"] = (
                self._tilt_steady_weight()[push].mean()
            )
            self.log_dict["homie_vx_cmd_push"] = vx_cmd[push].mean()
            self.log_dict["walk_cmd_frac_push"] = (vx_cmd > db)[push].float().mean()
            self.log_dict["pelvis_fwd_vel_push"] = root_v[push].mean()
            self.log_dict["bin_fwd_vel_push"] = bin_v[push].mean()
            self.log_dict["bar_fwd_vel_push"] = bar_v[push].mean()
            if bool(self.config.get("push_anti_stall_enable", False)):
                self.log_dict["push_anti_stall_weight_push"] = (
                    self._push_motion_anti_stall_weight()[push].mean()
                )
            self.log_dict["body_bin_contact_frac_push"] = (
                self._body_bin_contact_force() > 50.0
            )[push].float().mean()

    # --- observation getters ---

    def _get_obs_relative_to_bin(self):
        relative_rot_6d = quat_to_tan_norm(self.relative_bin_rot_buf, w_last=True)
        return torch.cat([self.relative_bin_pos_buf, relative_rot_6d], dim=-1)

    def _get_obs_hand_bin_handle_transform(self):
        left_hand_pos = self.simulator.left_hand_transform_pos[:, 0, :]
        left_hand_rot_wxyz = self.simulator.left_hand_transform_rot[:, 0, :]
        left_hand_rot_6d = quat_to_tan_norm(wxyz_to_xyzw(left_hand_rot_wxyz), w_last=True)
        right_hand_pos = self.simulator.right_hand_transform_pos[:, 0, :]
        right_hand_rot_wxyz = self.simulator.right_hand_transform_rot[:, 0, :]
        right_hand_rot_6d = quat_to_tan_norm(wxyz_to_xyzw(right_hand_rot_wxyz), w_last=True)
        return torch.cat(
            [left_hand_pos, left_hand_rot_6d, right_hand_pos, right_hand_rot_6d], dim=-1
        )

    def _get_obs_hand_bar_transform(self):
        """Per-hand pose of the RAW bar relative to that hand's virtual grasp
        point, in the palm-link frame: [err_xyz (3), bar_rot_6d (6)] x 2 = 18.

        2026-09-12: the teacher previously could not observe the quantity it is
        graded on. `hand_bin_handle_transform` reports the OFFSET hover target
        relative to the palm LINK origin, while the grasp gate scores
        `_palm_grasp_points` (4.64cm out along the finger cavity, per
        palm_grasp_point_offset) against `_left/right_bar_raw_pos`. Two
        different source points and two different targets, so the observed
        "zero error" and the rewarded "zero error" were ~4-9cm apart depending
        on palm orientation -- recoverable in principle for a privileged
        teacher, but there was no reason to make it infer a transform we can
        just hand it.

        err_xyz is exact, not approximated: `_palm_grasp_points` offsets the
        palm link origin by `offset * axis` expressed in the palm's own local
        frame, so subtracting that same vector from the bar transformer's
        palm-frame reading gives the grasp-point-to-bar error directly.
        """
        offset = float(self.config.get("palm_grasp_point_offset", 0.0464))
        left_axis, right_axis = self._palm_grasp_axes()
        left_sensor = self.simulator.scene.sensors["left_hand_bar_frame_transformer"]
        right_sensor = self.simulator.scene.sensors["right_hand_bar_frame_transformer"]
        left_err = left_sensor.data.target_pos_source[:, 0, :] - offset * left_axis
        right_err = right_sensor.data.target_pos_source[:, 0, :] - offset * right_axis
        left_rot_6d = quat_to_tan_norm(
            wxyz_to_xyzw(left_sensor.data.target_quat_source[:, 0, :]), w_last=True
        )
        right_rot_6d = quat_to_tan_norm(
            wxyz_to_xyzw(right_sensor.data.target_quat_source[:, 0, :]), w_last=True
        )
        return torch.cat(
            [left_err, left_rot_6d, right_err, right_rot_6d], dim=-1
        )

    def _get_obs_hand_force(self):
        left_hand_force = self.simulator.contact_forces[:, self.left_hand_indices, :]
        right_hand_force = self.simulator.contact_forces[:, self.right_hand_indices, :]
        return torch.cat([
            left_hand_force.reshape(left_hand_force.shape[0], -1),
            right_hand_force.reshape(right_hand_force.shape[0], -1),
        ], dim=-1)

    def _get_obs_privileged_bin_info(self):
        """6-D privileged info: [mass, ground_friction, wheel_stiffness,
        push_target_distance, push_axis_x, push_axis_y]."""
        if bool(self.config.get("privileged_bin_info_ego_push_axis", True)):
            push_axis_x = torch.ones_like(self._privileged_bin_mass)
            push_axis_y = torch.zeros_like(self._privileged_bin_mass)
        else:
            push_axis_x = self.push_axis_world[:, 0]
            push_axis_y = self.push_axis_world[:, 1]
        push_target = torch.full(
            (self.num_envs,), self.target_bin_push_distance, device=self.device
        )
        return torch.stack([
            self._privileged_bin_mass,
            self._privileged_ground_friction,
            self._privileged_wheel_stiffness,
            push_target,
            push_axis_x,
            push_axis_y,
        ], dim=1)

    def _get_obs_bin_pose_progress(self):
        """1-D: bin displacement from initial pose along the push axis."""
        current_xy = self.simulator.get_task_root_state("bin")[:, :2]
        delta_xy = current_xy - self._initial_bin_pos_xy
        axis_xy = self.push_axis_world[:, :2]
        axis_norm = axis_xy.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        axis_unit = axis_xy / axis_norm
        progress = (delta_xy * axis_unit).sum(dim=-1, keepdim=True)
        return progress

    def _get_obs_dof_pos_non_finger(self):
        return self.simulator.dof_pos[:, :-14]

    def _get_obs_dof_vel_non_finger(self):
        return self.simulator.dof_vel[:, :-14]

    def _get_obs_target_obj_pos(self):
        return (
            self.simulator.scene.sensors["head_target_frame_transformer"]
            .data.target_pos_source[:, 0, :]
            .clone()
        )

    # ================================================================
    # STAGE TRANSITIONS
    # ================================================================

    def _bin_center_xy(self):
        return self.simulator.get_task_root_state("bin")[:, :2].clone()

    def _left_handle_target(self):
        """World-frame target position for the left palm on the bin handle."""
        return (
            self.simulator.scene.sensors["left_hand_frame_transformer"]
            .data.target_pos_w[:, 0, :]
            .clone()
        )

    def _right_handle_target(self):
        """World-frame target position for the right palm on the bin handle."""
        return (
            self.simulator.scene.sensors["right_hand_frame_transformer"]
            .data.target_pos_w[:, 0, :]
            .clone()
        )

    def _left_bar_raw_pos(self):
        """World-frame RAW bar position for the left palm -- no standoff/
        height offset, unlike `_left_handle_target` above (a pre-contact
        hover point meant for REACH-stage guidance). 2026-08-29, used by
        `_hand_bar_distances` for the grasp-validity gate specifically,
        where "close to the bar" should mean close to the bar, not close to
        a 4.1cm-offset hover point. See scene_creation_callback's
        left_hand_bar_frame_transformer registration."""
        return (
            self.simulator.scene.sensors["left_hand_bar_frame_transformer"]
            .data.target_pos_w[:, 0, :]
            .clone()
        )

    def _right_bar_raw_pos(self):
        """World-frame RAW bar position for the right palm -- see
        `_left_bar_raw_pos` docstring."""
        return (
            self.simulator.scene.sensors["right_hand_bar_frame_transformer"]
            .data.target_pos_w[:, 0, :]
            .clone()
        )

    def _grasp_reference_targets(self):
        """Single source of truth for the point GRASP-stage geometry is measured
        against. Returns (left, right) world positions.

        2026-09-12: every GRASP-stage reward, hard gate and diagnostic log used
        `_left/right_handle_target` -- the pre-contact HOVER point, offset from the
        bar by grasp_target_standoff (4cm, along the bin's push axis) plus
        grasp_target_height_offset (1cm) -- while the validity gate
        (`_grasp_pose_valid_mask` via `_hand_bar_distances`) used
        `_left/right_bar_raw_pos`. Both measure FROM the same point
        (`_palm_grasp_points`), so the two disagreed by a fixed ~4.1cm that is
        almost entirely horizontal.

        That made the gate arithmetically unreachable, which is why
        `stage_3_frac` and `grasp_full_pose_valid_frac_s2` were identically 0.0
        for the whole project: at the end of the best run the policy had
        converged the palm grasp point to 5.8cm of the hover point
        (grasp_left_horiz_s2 0.047, grasp_left_abs_vert_s2 0.033) while sitting
        9.6cm from the bar (grasp_left_bar_dist_s2) against an 0.08 threshold.
        The 4cm residual is exactly grasp_target_standoff. It also explains why
        vertical converged (1cm offset) and horizontal did not (4cm offset), and
        why the isolated standoff ablation read as inert -- shrinking the offset
        alone left 15.0 of reward weight still pulling to the hover point
        against 21.0 pulling to the bar.

        REACH deliberately keeps the hover target: the offset is a genuine
        pre-contact waypoint there, and the pregrasp arc is built around it. The
        stage boundary is now a real 4cm "close the last gap onto the bar" step
        rather than two rewards disagreeing about where the bar is.
        """
        if bool(self.config.get("grasp_use_raw_bar_reference", True)):
            return self._left_bar_raw_pos(), self._right_bar_raw_pos()
        return self._left_handle_target(), self._right_handle_target()

    def _left_handle_target_rot(self):
        """World-frame target orientation (xyzw) for the left palm."""
        return wxyz_to_xyzw(
            self.simulator.scene.sensors["left_hand_frame_transformer"]
            .data.target_quat_w[:, 0, :]
            .clone()
        )

    def _right_handle_target_rot(self):
        """World-frame target orientation (xyzw) for the right palm."""
        return wxyz_to_xyzw(
            self.simulator.scene.sensors["right_hand_frame_transformer"]
            .data.target_quat_w[:, 0, :]
            .clone()
        )

    def _palm_contact_axes(self):
        """Anatomical palm-side axes in each mirrored palm-link frame."""
        left_axis = torch.tensor(
            self.config.get("left_palm_contact_axis", [0.0, -1.0, 0.0]),
            dtype=torch.float32,
            device=self.device,
        ).repeat(self.num_envs, 1)
        right_axis = torch.tensor(
            self.config.get("right_palm_contact_axis", [0.0, 1.0, 0.0]),
            dtype=torch.float32,
            device=self.device,
        ).repeat(self.num_envs, 1)
        return left_axis, right_axis

    def _palm_grasp_axes(self):
        """Anatomical axis (in each palm-link local frame) pointing INTO the
        finger-closing cavity — distinct from `_palm_contact_axes` (the palm-FACE
        normal used for down-alignment checks).

        Root cause of the thumb-overshoot bug (2026-08-01): `_palm_grasp_points`
        used to reuse `_palm_contact_axes` (local +-Y, the face normal) as the
        cavity direction too. Measured actual simulated link transforms (rest
        pose, from a real rollout frame): index_0/middle_0 sit at local
        (+0.078, ~0, +-0.029), thumb_0 sits at local (+0.026, ~0, 0) — i.e. the
        three finger mounts form a triangle almost entirely along local +X
        (forward off the wrist), with a small offset along Y.

        Follow-up (2026-08-01, user video review): a pure midpoint (50/50 vs
        thumb_0) under-weighted the thumb — biased toward it using thumb_1
        (the curled first-knuckle position, further along the thumb's
        actual closing sweep than the uncurled thumb_0 mount) at a 60/40
        blend against the index/middle mount average. This is where the
        measured, MIRRORED Y-component (~-0.024 left / +0.024 right,
        opposite sign per hand — the thumb curls toward different sides)
        stops being noise-level and needs its own axis per hand.
        """
        left_axis = torch.tensor(
            self.config.get("left_palm_grasp_axis", [0.9717, -0.2356, -0.0181]),
            dtype=torch.float32,
            device=self.device,
        ).repeat(self.num_envs, 1)
        right_axis = torch.tensor(
            self.config.get("right_palm_grasp_axis", [0.9717, 0.2359, 0.0155]),
            dtype=torch.float32,
            device=self.device,
        ).repeat(self.num_envs, 1)
        return left_axis, right_axis

    def _palm_grasp_points(self):
        """World points used to align each palm with its rim target.

        The palm-link origin is at the wrist-side center, not at the desired
        grasp location. The virtual point lies in the finger-closing cavity —
        a 40/60 blend between the top-two-finger mounts (index_0/middle_0) and
        the thumb's curled position (thumb_1), biased toward the thumb per
        user video review — avoiding both link-origin penetration and
        dorsal/back-of-hand grasps.
        """
        left_axis, right_axis = self._palm_grasp_axes()
        offset = float(self.config.get("palm_grasp_point_offset", 0.0464))
        left_point = self.simulator._rigid_body_pos[:, self.left_palm_idx, :] + quat_apply(
            xyzw_to_wxyz(
                self.simulator._rigid_body_rot[:, self.left_palm_idx, :]
            ),
            offset * left_axis,
        )
        right_point = self.simulator._rigid_body_pos[:, self.right_palm_idx, :] + quat_apply(
            xyzw_to_wxyz(
                self.simulator._rigid_body_rot[:, self.right_palm_idx, :]
            ),
            offset * right_axis,
        )
        return left_point, right_point

    def _palm_down_alignments(self):
        """Cosine alignment of each anatomical palm face with bin-down."""
        # get_task_root_state returns wxyz (same as L375); do NOT reorder.
        bin_quat_wxyz = self.simulator.get_task_root_state("bin")[:, 3:7]
        bin_up = quat_apply(
            bin_quat_wxyz,
            torch.tensor([[0.0, 0.0, 1.0]], device=self.device).repeat(
                self.num_envs, 1
            ),
        )
        left_axis, right_axis = self._palm_contact_axes()
        left_world = quat_apply(
            xyzw_to_wxyz(
                self.simulator._rigid_body_rot[:, self.left_palm_idx, :]
            ),
            left_axis,
        )
        right_world = quat_apply(
            xyzw_to_wxyz(
                self.simulator._rigid_body_rot[:, self.right_palm_idx, :]
            ),
            right_axis,
        )
        desired = -F.normalize(bin_up, dim=-1)
        if bool(self.config.get("grasp_palm_align_from_target", False)):
            # 2026-09-28 handlefix5: desired palm-face direction = the target
            # frame's palm axis (== -bin_up for the old palm-down target), so
            # every consumer follows the configured grasp orientation.
            left_des = quat_apply(xyzw_to_wxyz(self._left_handle_target_rot()), left_axis)
            right_des = quat_apply(xyzw_to_wxyz(self._right_handle_target_rot()), right_axis)
            return (
                (left_world * F.normalize(left_des, dim=-1)).sum(dim=-1).clamp(-1.0, 1.0),
                (right_world * F.normalize(right_des, dim=-1)).sum(dim=-1).clamp(-1.0, 1.0),
            )
        return (
            (left_world * desired).sum(dim=-1).clamp(min=-1.0, max=1.0),
            (right_world * desired).sum(dim=-1).clamp(min=-1.0, max=1.0),
        )

    def _palm_twist_errors(self):
        """Full angular error (radians) between each palm and its authored target
        orientation, including twist about the palm-down axis.

        `_palm_down_alignments` only checks the ONE axis that must point at
        bin-down (a palm rotated 90 degrees about that same axis reads identical
        alignment, but can't curl its fingers around the rim). The frame
        transformers already carry a correctly cross-assigned and mirrored
        target rotation (see scene_creation_callback and the July 27 fix to
        grasp_target_left's observed rotation) — this reads that full quaternion
        via quat_error_magnitude instead of inventing a second reference axis.
        """
        left_target_rot = xyzw_to_wxyz(self._left_handle_target_rot())
        right_target_rot = xyzw_to_wxyz(self._right_handle_target_rot())
        left_palm_rot = xyzw_to_wxyz(
            self.simulator._rigid_body_rot[:, self.left_palm_idx, :]
        )
        right_palm_rot = xyzw_to_wxyz(
            self.simulator._rigid_body_rot[:, self.right_palm_idx, :]
        )
        return (
            quat_error_magnitude(left_palm_rot, left_target_rot),
            quat_error_magnitude(right_palm_rot, right_target_rot),
        )

    def _stop_target_xy(self):
        """World-XY of the grasp-relative stop pose (num_envs, 2).

        The old stop goal was a 0.9 m circle around the bin CENTER — symmetric, so
        the robot could satisfy it standing anywhere around the bin, including
        spots where the handles are unreachable. This computes a target directly
        in front of the handles instead:

            stop_xy = handle_mid_xy + standoff * face_normal_xy

        where ``face_normal`` points from the bin center out through the handle
        midpoint (i.e. the grasp face). Derived from logged geometry: the robot
        naturally settles ~0.40 m out from the handle mid, and the handles sit on
        one face, so this couples "stopped" to "positioned to grasp". Rotates
        correctly with any bin yaw because it's built from live handle/center
        world positions.
        """
        handle_mid = 0.5 * (self._left_handle_target()[:, :2] + self._right_handle_target()[:, :2])
        bin_center = self._bin_center_xy()
        face_normal = handle_mid - bin_center
        face_normal = face_normal / face_normal.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        standoff = float(self.config.get("stop_standoff", 0.40))
        return handle_mid + standoff * face_normal

    def _grasp_standoff_state(self):
        """Return signed handle standoff, target error, and lateral error."""
        root_xy = self.simulator.robot_root_states[:, :2]
        handle_mid = 0.5 * (
            self._left_handle_target()[:, :2] + self._right_handle_target()[:, :2]
        )
        face_normal = handle_mid - self._bin_center_xy()
        face_normal = face_normal / face_normal.norm(
            dim=-1, keepdim=True
        ).clamp(min=1e-6)
        root_from_handles = root_xy - handle_mid
        signed_standoff = (root_from_handles * face_normal).sum(dim=-1)
        lateral = torch.stack([-face_normal[:, 1], face_normal[:, 0]], dim=-1)
        lateral_error = (root_from_handles * lateral).sum(dim=-1).abs()
        target_standoff = float(self.config.get("stop_standoff", 0.40))
        return signed_standoff, signed_standoff - target_standoff, lateral_error

    def _standoff_ready_mask(self):
        _, along_error, lateral_error = self._grasp_standoff_state()
        along_tolerance = float(
            self.config.get("stop_transition_along_tolerance", 0.08)
        )
        lateral_tolerance = float(
            self.config.get("stop_transition_lateral_tolerance", 0.12)
        )
        return (along_error.abs() < along_tolerance) & (
            lateral_error < lateral_tolerance
        )

    def _spawn_polar_active_bounds(self):
        """2026-10-06 full3: active (d_min_m, d_max_m, phi_max_deg, fov_yaw_max_deg)
        for Polar STAGE_APPROACH Spawn DR, interpolating phi and fov_yaw with
        spawn_dr_dist_level_m when spawn_polar_curriculum is enabled."""
        d_min = float(self.config.get("spawn_polar_dist_min_m", 0.02))
        d_start = float(self.config.get("spawn_polar_dist_start_m", 0.25))
        d_max_full = float(self.config.get("spawn_polar_dist_max_m", 1.10))
        phi_start = float(self.config.get("spawn_polar_phi_start_deg", 15.0))
        phi_max_full = float(self.config.get("spawn_polar_phi_max_deg", 45.0))
        yaw_start = float(self.config.get("spawn_polar_fov_yaw_start_deg", 10.0))
        yaw_max_full = float(self.config.get("spawn_polar_fov_yaw_max_deg", 20.0))
        if not getattr(self, "_spawn_polar_cur", False):
            return d_min, d_max_full, phi_max_full, yaw_max_full
        lvl = float(self.spawn_dr_dist_level_m)
        denom = max(d_max_full - d_start, 1e-6)
        alpha = min(max((lvl - d_start) / denom, 0.0), 1.0)
        phi_cur = phi_start + alpha * (phi_max_full - phi_start)
        yaw_cur = yaw_start + alpha * (yaw_max_full - yaw_start)
        return d_min, lvl, phi_cur, yaw_cur

    def _approach_yaw_error_rad(self):
        """2026-10-06 full3: signed yaw error (rad) in [-pi, pi] between robot
        heading and the grasp-face inward direction (-face_normal). When the
        bin is at its default yaw (+X), -face_normal = (+1, 0), so yaw error is 0
        when robot faces +X."""
        handle_mid = 0.5 * (
            self._left_handle_target()[:, :2] + self._right_handle_target()[:, :2]
        )
        inward = self._bin_center_xy() - handle_mid
        target_yaw = torch.atan2(inward[:, 1], inward[:, 0])
        _, _, robot_yaw = euler_xyz_from_quat(
            xyzw_to_wxyz(self.simulator.robot_root_states[:, 3:7])
        )
        diff = robot_yaw - target_yaw
        return torch.atan2(torch.sin(diff), torch.cos(diff))

    def _approach_yaw_ready_mask(self):
        """2026-10-06 full3: True if |approach_yaw_error| < stop_transition_yaw_tolerance_deg
        (or all-True when stop_transition_yaw_tolerance_deg is None)."""
        tol_deg = self.config.get("stop_transition_yaw_tolerance_deg", None)
        if tol_deg is None:
            return torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        tol_rad = math.radians(float(tol_deg))
        return self._approach_yaw_error_rad().abs() < tol_rad

    def _bar_mid_xy(self):
        """2026-10-03 push2: world XY of the handle-bar midpoint (N, 2)."""
        return 0.5 * (self._left_bar_raw_pos()[:, :2] + self._right_bar_raw_pos()[:, :2])

    def _bar_fwd_vel(self):
        """2026-10-03 push2: handle-bar midpoint linear velocity along pushAxisWorld (m/s)."""
        st = self.simulator.get_task_root_state("bin")
        bar_mid = 0.5 * (self._left_bar_raw_pos() + self._right_bar_raw_pos())
        bar_vel = st[:, 7:10] + torch.cross(st[:, 10:13], bar_mid - st[:, :3], dim=-1)
        axis_unit = F.normalize(self.push_axis_world, dim=-1)
        return (bar_vel * axis_unit).sum(dim=-1)

    def _bin_bar_push_progress(self):
        """2026-10-03 push2: handle-bar midpoint displacement along pushAxisWorld (m)."""
        delta_xy = self._bar_mid_xy() - self._initial_bar_pos_xy
        axis_unit = F.normalize(self.push_axis_world[:, :2], dim=-1)
        return (delta_xy * axis_unit).sum(dim=-1)

    def _target_bin_push_distance_m(self):
        """2026-10-03 push2: active PUSH target distance (m), curriculum-aware."""
        if getattr(self, "_push_dist_cur", False):
            return float(self.target_bin_push_distance_m)
        return float(self.config.get("target_bin_push_distance", 1.0))

    def _bin_push_progress(self):
        """Bin displacement along pushAxisWorld (scalar per env, meters)."""
        current_xy = self.simulator.get_task_root_state("bin")[:, :2]
        delta_xy = current_xy - self._initial_bin_pos_xy
        axis_xy = self.push_axis_world[:, :2]
        axis_norm = axis_xy.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        axis_unit = axis_xy / axis_norm
        root_prog = (delta_xy * axis_unit).sum(dim=-1)
        if bool(self.config.get("push_progress_use_bar_min", False)):
            # 2026-10-03 push2: tipping the bin back around the handle swings
            # the bottom root +0.16 m forward without any translation; min of
            # root and bar progress requires both wheels and handle to advance.
            return torch.minimum(root_prog, self._bin_bar_push_progress())
        return root_prog

    def _hand_bar_distances(self):
        """Per-hand palm-to-bar-target distance (num_envs,) x2, world frame.

        2026-08-29: REPLACES the PhysX-contact-count approach entirely --
        both the whole-bin-body version (pre-2026-08-27) and the bar-scoped-
        ContactSensor version (2026-08-27, this function was briefly named
        `_hand_contact_counts` and read `_hand_contact_forces()`, which read
        two dedicated ContactSensors against a small collision sphere
        authored on grasp_target_left/right). That geometry+sensor approach
        reliably hung `simulation_app.close()` at training shutdown -- a
        real, reproducible bug (confirmed 3 times: once burned ~22hrs of
        idle VM time before being caught, then reproduced twice more in a
        tightly-bounded retest even after fixing a purpose/visibility
        mismatch that looked like a promising root cause but wasn't the
        whole story). Not worth further VM time chasing a Kit-internal
        teardown bug. User's call: pure kinematic distance from each palm's
        virtual grasp point (`_palm_grasp_points`) to its own targeted bar
        location (`_left_handle_target`/`_right_handle_target`, already
        correctly CROSSED per the Run 16 CROSSING FIX) is a perfectly good
        proxy for "did the hand actually reach the bar" -- zero new USD
        geometry, zero new ContactSensors, zero shutdown-hang risk. The
        `bar_collider` USD geometry and its two ContactSensorCfg
        registrations in scene_creation_callback were reverted alongside
        this change (see that function's own history comment).

        Old whole-bin-body-sensor version (pre-2026-08-27):
            forces = self.simulator.object_to_hand_contact_forces[:, 0, :, :]
            g1_hand_links = [n for n in self.simulator.robot_config.body_names
                              if ("left_hand" in n or "right_hand" in n)]
            left_mask = [i for i, n in enumerate(g1_hand_links) if "left_hand" in n]
            right_mask = [i for i, n in enumerate(g1_hand_links) if "right_hand" in n]
            left_count = (forces[:, left_mask, :].norm(dim=-1) > 1.0).sum(dim=-1).float()
            right_count = (forces[:, right_mask, :].norm(dim=-1) > 1.0).sum(dim=-1).float()
            return left_count, right_count
        (the bar-scoped-sensor version was identical except `forces` came
        from the two dedicated sensors instead of the whole-body one.)

        NOTE for callers: this returns raw DISTANCES (lower = better, unlike
        the old raw COUNTS where higher = better) -- every call site was
        updated accordingly (`>= grasp_contact_links_required` became
        `< grasp_bar_valid_dist`, ratio rewards were inverted), not just a
        drop-in swap.

        2026-08-29 (same day, follow-up): switched from `_left_handle_target`/
        `_right_handle_target` to `_left_bar_raw_pos`/`_right_bar_raw_pos`.
        The former is a pre-contact HOVER point offset by grasp_target_
        standoff (4cm) + grasp_target_height_offset (1cm) ≈4.12cm from the
        true bar -- correct for REACH-stage approach guidance, but wrong
        here: DoorMan's contact-force gate only fires at near-zero physical
        separation, so the correct kinematic analog for grasp VALIDITY is
        distance to the bar itself, not to an offset hover waypoint.
        grasp_bar_valid_dist was retuned accordingly (was 0.04 matching the
        hover-target scale, now a tighter value measured against the raw
        bar -- see its config comment).
        """
        left_palm, right_palm = self._palm_grasp_points()
        left_dist = (left_palm - self._left_bar_raw_pos()).norm(dim=-1)
        right_dist = (right_palm - self._right_bar_raw_pos()).norm(dim=-1)
        return left_dist, right_dist

    def _grasp_transition_components(self):
        """Return per-hand contact, proximity, and finger-closure conditions."""
        # 2026-08-29: "contact" here is now bar-distance-based (see
        # _hand_bar_distances) -- grasp_bar_valid_dist is deliberately
        # tighter than grasp_transition_dist below (which governs left_close/
        # right_close, a separate, looser proximity check already used
        # elsewhere in this tuple).
        bar_valid_dist = float(self.config.get("grasp_bar_valid_dist", 0.03))
        transition_dist = float(self.config.get("grasp_transition_dist", 0.10))
        finger_threshold = float(
            self.config.get("grasp_finger_closed_threshold", 0.174533)
        )

        left_dist, right_dist = self._hand_bar_distances()
        left_palm, right_palm = self._palm_grasp_points()
        ref_left, ref_right = self._grasp_reference_targets()
        left_close = (left_palm - ref_left).norm(dim=-1) < transition_dist
        right_close = (right_palm - ref_right).norm(dim=-1) < transition_dist
        left_closed = (
            self.simulator.dof_pos[:, self._left_hand_dof_idx] - self._left_p1
        ).abs().mean(dim=-1) < finger_threshold
        right_closed = (
            self.simulator.dof_pos[:, self._right_hand_dof_idx] - self._right_p1
        ).abs().mean(dim=-1) < finger_threshold
        return (
            left_dist < bar_valid_dist,
            right_dist < bar_valid_dist,
            left_close,
            right_close,
            left_closed,
            right_closed,
        )

    def _grasp_readiness_score(self):
        """Continuous per-hand grasp-readiness score R in [0,1] -- alternative
        to _finger_close_ready's hard 5-condition AND-gate below.

        Prototype, 2026-08-27, off by default (enable_continuous_grasp_
        readiness) -- _finger_close_ready's original 5-condition AND-gate
        remains the live/default path unless this flag is set. UNTESTED
        against real Isaac Sim.

        Motivation: the AND-gate requires contact, horiz, vert, alignment,
        and twist to ALL cross their hard thresholds simultaneously -- a
        hand excellent on 4 axes and mediocre on the 5th gets exactly zero
        credit, same as a hand that's bad on all 5. Observed repeatedly this
        session: fixing one axis's convergence just exposes the next axis as
        the new bottleneck (twist -> vertical -> horizontal, same hand, same
        run, see the reward-tuning history and the 2026-08-27 ablation
        results) because the gate only ever responds to whichever single
        axis is currently worst. R replaces the AND of 5 booleans with a
        PRODUCT of 5 smooth (Gaussian/sigmoid) terms on the exact same
        underlying quantities the AND-gate already checks, so partial credit
        on every axis compounds into one combined score instead of being
        thrown away by whichever axis fails hardest. Sigma widths default to
        the old hard-threshold values, so the readiness curve is centered
        where the old gate used to cut off, not an arbitrary new scale.
        """
        close_dist_horiz = float(self.config.get("grasp_finger_close_horiz_dist", 0.04))
        close_dist_vert = float(self.config.get("grasp_finger_close_vert_dist", 0.03))
        min_alignment = float(self.config.get("grasp_finger_close_alignment", 0.2))
        max_twist_error = float(
            self.config.get("grasp_finger_close_max_twist_rad", 1.0472)
        )
        # 2026-08-29: bar_valid_dist replaces required_contacts (see
        # _hand_bar_distances) -- contact_term is now a Gaussian falloff on
        # distance, matching horiz_term/vert_term's own shape, instead of a
        # linear count/required ratio.
        bar_dist_sigma = float(self.config.get("grasp_readiness_bar_dist_sigma", 0.03))
        horiz_sigma = float(self.config.get("grasp_readiness_horiz_sigma", close_dist_horiz))
        vert_sigma = float(self.config.get("grasp_readiness_vert_sigma", close_dist_vert))
        twist_sigma = float(self.config.get("grasp_readiness_twist_sigma", max_twist_error))
        align_scale = float(self.config.get("grasp_readiness_align_scale", 0.2))

        left_bar_dist, right_bar_dist = self._hand_bar_distances()
        left_palm, right_palm = self._palm_grasp_points()
        left_alignment, right_alignment = self._palm_down_alignments()
        left_twist, right_twist = self._palm_twist_errors()
        ref_left, ref_right = self._grasp_reference_targets()
        left_delta = ref_left - left_palm
        right_delta = ref_right - right_palm

        def _score(delta, alignment, twist, bar_dist):
            horiz_err = delta[:, :2].norm(dim=-1)
            vert_err = delta[:, 2].abs()
            contact_term = torch.exp(-0.5 * (bar_dist / bar_dist_sigma) ** 2)
            horiz_term = torch.exp(-0.5 * (horiz_err / horiz_sigma) ** 2)
            vert_term = torch.exp(-0.5 * (vert_err / vert_sigma) ** 2)
            align_term = torch.sigmoid((alignment - min_alignment) / align_scale)
            twist_term = torch.exp(-0.5 * (twist / twist_sigma) ** 2)
            return contact_term * horiz_term * vert_term * align_term * twist_term

        left_r = _score(left_delta, left_alignment, left_twist, left_bar_dist)
        right_r = _score(right_delta, right_alignment, right_twist, right_bar_dist)
        return left_r, right_r

    def _finger_close_ready(self):
        """Per-hand readiness to switch the finger target from open to closed.

        Run 19 (2026-08-01): the old isotropic 3D `close_dist` sphere let a hand
        floating directly ABOVE the target pass the same as one truly centered on
        the rim (a purely vertical offset counts the same as a purely horizontal
        one). Checkpoint 6300 showed exactly this: left palm horizontally centered
        (<0.03m) but sitting +7cm above the true handle in GRASP for 100+ frames,
        closing fingers there instead of descending. Decomposed into separate
        horizontal and vertical tolerances (both must pass) so closing requires
        genuine descent onto the rim, not just isotropic proximity. Also added a
        twist-angle check alongside the existing single-axis palm_alignment, so a
        hand that points down but is rotated wrong about that axis (can't curl
        fingers around the rim) doesn't get marked ready either.

        2026-08-27: gated behind enable_continuous_grasp_readiness (default
        False, i.e. the 5-condition AND-gate below stays the LIVE/DEFAULT
        path). When the flag is True, uses _grasp_readiness_score()'s single
        combined smooth score crossing grasp_readiness_threshold instead of
        ANDing 5 independent hard conditions -- see that method's docstring.
        The original 5-condition logic is preserved below, not deleted, and
        keeps running by default for any existing/in-progress experiment.
        """
        if self.config.get("enable_continuous_grasp_readiness", False):
            threshold = float(self.config.get("grasp_readiness_threshold", 0.5))
            left_r, right_r = self._grasp_readiness_score()
            left_above, right_above = self._palm_above_bar_masks()  # 2026-09-30
            return (left_r > threshold) & left_above, (right_r > threshold) & right_above

        # --- Original 5-condition AND-gate (LIVE DEFAULT as of 2026-08-27) ---
        # 2026-08-29: "contact" condition is now bar-distance-based, see
        # _hand_bar_distances.
        # 2026-09-24: finger-close proximity decoupled from the GRASP->PUSH
        # success gate. With open fingers the hands rest flat on the lid and
        # cannot get within grasp_bar_valid_dist (5cm) of the bar, but fingers
        # only close once inside it -- a deadlock (diag rollout of step 4200:
        # best worst-hand dist 6.4cm, palms ~5cm above the bar). A looser
        # close radius lets the fingers curl over the lid lip toward the bar.
        # Falls back to grasp_bar_valid_dist when unset (old behaviour).
        bar_valid_dist = float(
            self.config.get(
                "grasp_finger_close_bar_dist",
                self.config.get("grasp_bar_valid_dist", 0.03),
            )
        )
        close_dist_horiz = float(
            self.config.get("grasp_finger_close_horiz_dist", 0.04)
        )
        close_dist_vert = float(
            self.config.get("grasp_finger_close_vert_dist", 0.03)
        )
        min_alignment = float(
            self.config.get("grasp_finger_close_alignment", 0.2)
        )
        max_twist_error = float(
            self.config.get("grasp_finger_close_max_twist_rad", 1.0472)  # 60 deg
        )
        left_bar_dist, right_bar_dist = self._hand_bar_distances()
        left_palm, right_palm = self._palm_grasp_points()
        left_alignment, right_alignment = self._palm_down_alignments()
        left_twist, right_twist = self._palm_twist_errors()

        ref_left, ref_right = self._grasp_reference_targets()
        left_delta = ref_left - left_palm
        right_delta = ref_right - right_palm
        left_horiz_ok = left_delta[:, :2].norm(dim=-1) < close_dist_horiz
        left_vert_ok = left_delta[:, 2].abs() < close_dist_vert
        right_horiz_ok = right_delta[:, :2].norm(dim=-1) < close_dist_horiz
        right_vert_ok = right_delta[:, 2].abs() < close_dist_vert

        left_ready = (
            (left_bar_dist < bar_valid_dist)
            & left_horiz_ok
            & left_vert_ok
            & (left_alignment > min_alignment)
            & (left_twist < max_twist_error)
        )
        right_ready = (
            (right_bar_dist < bar_valid_dist)
            & right_horiz_ok
            & right_vert_ok
            & (right_alignment > min_alignment)
            & (right_twist < max_twist_error)
        )
        # 2026-09-30: palm above the bar (grasp_palm_min_dz, default off).
        left_above, right_above = self._palm_above_bar_masks()
        return left_ready & left_above, right_ready & right_above

    def _palm_bar_deltas_bin(self):
        """2026-09-30: (palm grasp point - that hand's raw bar point) in the
        BIN frame (x along the bar, y horizontal across it, z up), per hand."""
        left_palm, right_palm = self._palm_grasp_points()
        q_inv = quat_inv(self.simulator.get_task_root_state("bin")[:, 3:7])  # wxyz
        return (
            quat_apply(q_inv, left_palm - self._left_bar_raw_pos()),
            quat_apply(q_inv, right_palm - self._right_bar_raw_pos()),
        )

    def _palm_min_dz(self):
        v = self.config.get("grasp_palm_min_dz", None)
        return None if v is None else float(v)

    def _palm_above_bar_masks(self):
        """2026-09-30: per-hand bool, palm grasp point height above the bar
        axis (bin frame) > grasp_palm_min_dz; all True when the key is unset."""
        min_dz = self._palm_min_dz()
        if min_dz is None:
            ones = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            return ones, ones
        ld, rd = self._palm_bar_deltas_bin()
        return ld[:, 2] > min_dz, rd[:, 2] > min_dz

    def _reach_transition_ready_mask(self):
        """Instantaneous REACH readiness before applying the hold duration."""
        left_palm, right_palm = self._palm_grasp_points()
        left_close = (
            left_palm - self._left_handle_target()
        ).norm(dim=-1) < 0.15
        right_close = (
            right_palm - self._right_handle_target()
        ).norm(dim=-1) < 0.15
        # 2026-09-24: "open" now means "at the pre-shape pose" (p0 when
        # finger_preshape_frac=0) -- see _finger_preshape_poses.
        left_pre, right_pre = self._finger_preshape_poses()
        left_open = (
            self.simulator.dof_pos[:, self._left_hand_dof_idx] - left_pre
        ).abs().mean(dim=-1) < 0.174533
        right_open = (
            self.simulator.dof_pos[:, self._right_hand_dof_idx] - right_pre
        ).abs().mean(dim=-1) < 0.174533
        homie_still = (
            torch.norm(self.get_physical_homie_commands()[:, :3], dim=1) <= 0.1
        )
        left_alignment, right_alignment = self._palm_down_alignments()
        alignment_threshold = float(
            self.config.get("reach_transition_alignment_threshold", 0.0)
        )
        return (
            left_close
            & right_close
            & left_open
            & right_open
            & homie_still
            & self._standoff_ready_mask()
            & (left_alignment > alignment_threshold)
            & (right_alignment > alignment_threshold)
        )

    def _finger_control_masks(self):
        in_reach = self.stage_buf == self.STAGE_REACH
        in_grasp = self.stage_buf == self.STAGE_GRASP
        # 2026-09-29 overhand: TILT holds the grasp exactly like PUSH.
        in_push = self._holding_stage_mask()
        left_ready, right_ready = self._finger_close_ready()
        self._left_finger_close_latched &= in_grasp
        self._right_finger_close_latched &= in_grasp
        self._left_finger_close_latched |= in_grasp & left_ready
        self._right_finger_close_latched |= in_grasp & right_ready
        return (
            in_reach | (in_grasp & ~self._left_finger_close_latched),
            in_reach | (in_grasp & ~self._right_finger_close_latched),
            in_push | (in_grasp & self._left_finger_close_latched),
            in_push | (in_grasp & self._right_finger_close_latched),
        )

    def _set_integrated_finger_action(
        self, actor_state, env_mask, action_idx, delta_idx, target
    ):
        self._delta_actions[env_mask, delta_idx] = target
        self._last_delta_actions[env_mask, delta_idx] = 0.0
        actor_state["actions"][env_mask, action_idx] = target

    def _finger_preshape_poses(self):
        """2026-09-24: partially-curled "hook" pre-shape finger pose per hand,
        p_pre = p0 + finger_preshape_frac * (p1 - p0). Used as the REACH-stage
        finger target (reward, reach-position finger gate, and REACH->GRASP
        transition) instead of the fully open p0, so the hand ARRIVES
        pre-curled like a human reaching for a handle rather than landing flat
        on the lid and trying to curl too late (video review of step 4200/4700:
        fingers flat on the lid, curling blocked by the lid surface).
        finger_preshape_frac defaults to 0.0 == p0 == previous behaviour."""
        alpha = float(self.config.get("finger_preshape_frac", 0.0))
        # 2026-09-24 (#1): distance-graded, per env. Fingers stay OPEN in
        # APPROACH; in REACH the curl ramps 0 -> alpha as the grasp point
        # closes from finger_preshape_start_dist to finger_preshape_full_dist
        # of the hover (pre-grasp) target. Returns (num_envs, n_finger_dof).
        d_start = float(self.config.get("finger_preshape_start_dist", 0.25))
        d_full = float(self.config.get("finger_preshape_full_dist", 0.05))
        left_gp, right_gp = self._palm_grasp_points()
        left_d = (left_gp - self._left_handle_target()).norm(dim=-1)
        right_d = (right_gp - self._right_handle_target()).norm(dim=-1)
        span = max(d_start - d_full, 1e-6)
        in_reach = (self.stage_buf == self.STAGE_REACH).float()
        left_a = alpha * in_reach * ((d_start - left_d) / span).clamp(0.0, 1.0)
        right_a = alpha * in_reach * ((d_start - right_d) / span).clamp(0.0, 1.0)
        left = self._left_p0[None, :] + left_a[:, None] * (self._left_p1 - self._left_p0)[None, :]
        right = self._right_p0[None, :] + right_a[:, None] * (self._right_p1 - self._right_p0)[None, :]
        return left, right

    def _pregrasp_curl_fraction(self):
        """Distance-graded finger-primitive curl fraction, in [0, 1] per
        hand: 0 = fully open (p0), 1 = fully closed (p1).

        2026-08-29: REPURPOSED. Originally written 2026-08-27 as an ACTION
        OVERRIDE input (capped at pregrasp_curl_max_fraction, e.g. 0.3 --
        deliberately short of full closure, meant to hand off to a separate
        control-time "close trigger" for the final approach). That override
        pathway was removed the same day it was written (see
        _postprocess_delta_actions's docstring -- fingers are under full
        continuous policy control now, matching the DoorMan reference),
        leaving this function fully disconnected dead code for ~2 days.

        Now reconnected as a REWARD TARGET (not an action write) for
        _reward_grasp_finger_dof_pos_l1, per user request after ~5 runs of
        video review showed fingers landing flat on the bin's lid -- that
        reward's previous unconditional p1 (fully closed) target for the
        ENTIRE GRASP+PUSH span was diagnosed as part of the problem
        (rewards "closed" without regard to whether closing is physically
        achievable/correct at the current position, see that function's own
        prior docstring). This version therefore removes the old
        max_fraction CAP -- the target now genuinely reaches p1 (fraction=1)
        as distance shrinks to 0, not just a partial pre-curl handed off to
        something else, since nothing else now exists to finish the job.

        Also switched from _left/right_handle_target (the offset hover
        point, ~4-11cm from the true bar depending on convergence) to
        _hand_bar_distances (the RAW bar distance, added 2026-08-29 for the
        same "measuring the wrong point" reason) -- consistent with this
        session's broader shift toward raw-bar-referenced measurements.

        Deliberately still a UNIFORM blend across all 7 finger DOF (the
        primitive mechanism's only supported interpolation) -- NOT a true
        differential "C" curl that keeps the base knuckle less curled than
        the fingertip, which would need direct per-joint targets bypassing
        the primitive scalar entirely (a larger change, not attempted here).

        UNTESTED against real Isaac Sim in this new (reward-target) role --
        prepared alongside grasp_fingertip_curl/penalty_grasp_flat_press,
        same "watch it before trusting it" caveat applies.
        """
        start_dist = float(self.config.get("pregrasp_curl_start_dist", 0.15))
        left_dist, right_dist = self._hand_bar_distances()
        left_frac = ((start_dist - left_dist) / start_dist).clamp(0.0, 1.0)
        right_frac = ((start_dist - right_dist) / start_dist).clamp(0.0, 1.0)
        return left_frac, right_frac

    @override
    def _postprocess_delta_actions(self, actor_state):
        """Superseded 2026-08-27 -- removed to let the policy control fingers
        continuously for its entire episode, modeled on
        NVlabs/GR00T-VisualSim2Real (doorman branch)
        envs/door/door_open_homie.py, which has no equivalent override at
        all: _left_p0/_left_p1 exist there only as dense reward-shaping
        targets (see _reward_grasp_finger_dof_pos_l1), never used to
        override the policy's own action. The base class's own
        _postprocess_delta_actions is a no-op (delta_action_base.py),
        confirming this task-specific override was fighting the base
        class's generic no-FSM lerp-based primitive mapping
        (FingerPrimitiveBase._apply_force_in_physics_step), not filling a
        gap in it. Old logic preserved below, commented, not deleted:

        left_open, right_open, left_close, right_close = (
            self._finger_control_masks()
        )
        in_grasp = self.stage_buf == self.STAGE_GRASP
        left_curl_frac, right_curl_frac = self._pregrasp_curl_fraction()
        action_span = self._finger_closed_action - self._finger_open_action
        left_open_target = torch.where(
            in_grasp,
            self._finger_open_action + left_curl_frac * action_span,
            torch.full_like(left_curl_frac, self._finger_open_action),
        )
        right_open_target = torch.where(
            in_grasp,
            self._finger_open_action + right_curl_frac * action_span,
            torch.full_like(right_curl_frac, self._finger_open_action),
        )
        self._set_integrated_finger_action(
            actor_state, left_open, self._left_finger_action_idx,
            self._left_finger_delta_idx, left_open_target[left_open],
        )
        self._set_integrated_finger_action(
            actor_state, right_open, self._right_finger_action_idx,
            self._right_finger_delta_idx, right_open_target[right_open],
        )
        self._set_integrated_finger_action(
            actor_state, left_close, self._left_finger_action_idx,
            self._left_finger_delta_idx, self._finger_closed_action,
        )
        self._set_integrated_finger_action(
            actor_state, right_close, self._right_finger_action_idx,
            self._right_finger_delta_idx, self._finger_closed_action,
        )

        Known, expected side effect of removing this: _delta_actions[...,
        17/18] (the finger dims) stop being re-pinned every step and become
        a real accumulator like every other DOF, clamped only by the global
        delta_action_clip. reward_limits_primitive_action /
        primitive_action_over_limit_buf -- currently silently dead-zero for
        fingers, since the override always wrote exactly the clamp boundary
        -- becomes live for fingers for the first time. Not a bug; watch it
        in early logs rather than mistaking a dip there for a regression.
        """
        decay = float(self.config.get("approach_arm_delta_decay", 0.0))
        if decay > 0.0:
            in_app = self.stage_buf == self.STAGE_APPROACH
            if in_app.any():
                st = self._resting_action_state()
                u_cols = st["upper_delta_cols"]
                rest_u = st["delta"][u_cols].unsqueeze(0)
                cur_u = self._delta_actions[:, u_cols]
                blended = (1.0 - decay) * cur_u + decay * rest_u
                self._delta_actions[:, u_cols] = torch.where(
                    in_app.unsqueeze(-1), blended, cur_u
                )
                act_cols = self._delta_action_indices[u_cols]
                actor_state["actions"][:, act_cols] = self._delta_actions[:, u_cols]

    def _enforce_whole_body_finger_actions(self, actions):
        """Superseded 2026-08-27 -- see _postprocess_delta_actions's
        docstring for the full rationale (this is the pathway that actually
        determines the applied physics action, called from
        _pre_physics_step; the two functions were mirrors of each other by
        necessity, not by choice -- _left_finger_action_idx and
        _left_whole_body_finger_action_idx resolve to the same underlying
        action column, just addressed via two different index spaces at two
        different pipeline stages). Old logic preserved below, commented:

        in_reach = self.stage_buf == self.STAGE_REACH
        in_grasp = self.stage_buf == self.STAGE_GRASP
        left_open = in_reach | (in_grasp & ~self._left_finger_close_latched)
        right_open = in_reach | (in_grasp & ~self._right_finger_close_latched)
        left_close = (self.stage_buf == self.STAGE_PUSH) | (
            (self.stage_buf == self.STAGE_GRASP) & self._left_finger_close_latched
        )
        right_close = (self.stage_buf == self.STAGE_PUSH) | (
            (self.stage_buf == self.STAGE_GRASP) & self._right_finger_close_latched
        )
        left_curl_frac, right_curl_frac = self._pregrasp_curl_fraction()
        action_span = self._finger_closed_action - self._finger_open_action
        left_open_target = torch.where(
            in_grasp,
            self._finger_open_action + left_curl_frac * action_span,
            torch.full_like(left_curl_frac, self._finger_open_action),
        )
        right_open_target = torch.where(
            in_grasp,
            self._finger_open_action + right_curl_frac * action_span,
            torch.full_like(right_curl_frac, self._finger_open_action),
        )
        actions[left_open, self._left_whole_body_finger_action_idx] = (
            left_open_target[left_open]
        )
        actions[right_open, self._right_whole_body_finger_action_idx] = (
            right_open_target[right_open]
        )
        actions[left_close, self._left_whole_body_finger_action_idx] = (
            self._finger_closed_action
        )
        actions[right_close, self._right_whole_body_finger_action_idx] = (
            self._finger_closed_action
        )
        """
        pass

    @override
    def _pre_physics_step(self, actions):
        super()._pre_physics_step(actions)
        self._enforce_whole_body_finger_actions(self.actions)
        self._enforce_whole_body_finger_actions(self.actions_after_delay)
        if hasattr(self, "action_queue"):
            for queue_idx in range(self.action_queue.shape[1]):
                self._enforce_whole_body_finger_actions(
                    self.action_queue[:, queue_idx, :]
                )

    @override
    def _load_delta_actions_buffer(self, env_ids, data):
        super()._load_delta_actions_buffer(env_ids, data)
        # 2026-09-29 overhand: TILT keeps the (closed) finger delta like PUSH.
        pre_push = ~self._holding_stage_mask(self.stage_buf[env_ids])
        open_env_ids = env_ids[pre_push]
        self._delta_actions[open_env_ids, self._left_finger_delta_idx] = (
            self._finger_open_action
        )
        self._delta_actions[open_env_ids, self._right_finger_delta_idx] = (
            self._finger_open_action
        )

    def _grasp_pose_valid_mask(self):
        """Bimanual bar-proximity only -- matches NVlabs/GR00T-VisualSim2Real
        (doorman branch) envs/door/door_open_homie.py's
        _stage_2_to_complete_condition in SPIRIT (a single per-hand
        condition, no geometric decomposition at the success gate itself --
        precision is enforced once, earlier, at the REACH-equivalent
        transition, not re-checked here), though DoorMan's own condition is
        contact-force-based, not distance-based -- see _hand_bar_distances
        docstring for why this project's version diverges (a bar-scoped
        PhysX ContactSensor reliably hung `simulation_app.close()` at
        training shutdown, confirmed 3 times; pure kinematic distance from
        the palm's virtual grasp point to the bar target sidesteps that
        entirely, 2026-08-29).

        2026-08-27: previously ANDed together ~10 conditions (contact +
        proximity + finger-closed-DOF for both hands from
        _grasp_transition_components(), PLUS palm alignment + twist for both
        hands, added 2026-08-24 -- see the commented block below for that
        full history). This function has SIX consumers, not just the
        GRASP->PUSH transition: _stage_3_reward_condition (feeds
        _reward_stage, active the whole PUSH stage), _stage_3_to_complete_
        condition (episode success), _reward_push_bin_progress,
        _reward_step_forward_while_pushing (the two main PUSH-stage
        rewards), and the push_valid_grasp_frac_s3 log. Simplifying loosens
        all of these at once, not just the transition gate -- a hand could
        in principle lose fine alignment/twist mid-push while still
        registering contact, and keep collecting push reward through a
        degraded grip. Accepted per the reference design; watch
        push_valid_grasp_frac_s3 once tested rather than assuming this is
        free. Old logic preserved below, commented, not deleted:

        valid = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        for condition in self._grasp_transition_components():
            valid &= condition
        left_alignment, right_alignment = self._palm_down_alignments()
        left_twist, right_twist = self._palm_twist_errors()
        alignment_threshold = float(self.config.get("grasp_palm_alignment_threshold", 0.5))
        max_twist_error = float(self.config.get("grasp_finger_close_max_twist_rad", 1.0472))
        return (
            valid
            & (left_alignment > alignment_threshold)
            & (right_alignment > alignment_threshold)
            & (left_twist < max_twist_error)
            & (right_twist < max_twist_error)
        )
        """
        bar_valid_dist = float(self.config.get("grasp_bar_valid_dist", 0.03))
        left_dist, right_dist = self._hand_bar_distances()
        valid = (left_dist < bar_valid_dist) & (right_dist < bar_valid_dist)
        # 2026-09-30 reach_grasp1 fix: the distance gate is direction-blind --
        # the left palm settled UNDER the bar (dz -4..-5 cm) at the same 5 cm.
        # grasp_palm_min_dz (default None = off) requires each palm grasp
        # point to sit above the bar axis (bin frame).
        left_above, right_above = self._palm_above_bar_masks()
        valid = valid & left_above & right_above
        # 2026-10-01 reach_grasp2 finding: the right hand entered TILT at
        # closure 0.35-0.39 (gate is geometry-only) and then loosened.
        # grasp_gate_min_closure (default None = off) additionally requires
        # BOTH hands' p0->p1 closure >= the value. Placed before the overhand
        # branch so it covers both gate paths. Not applied to finger_close_ready.
        min_closure = self.config.get("grasp_gate_min_closure", None)
        if min_closure is not None:
            lc, rc = self._finger_closure_fractions()
            valid = valid & (lc >= float(min_closure)) & (rc >= float(min_closure))
        min_wrap = float(self.config.get("grasp_gate_min_wrap", 0.0))
        if self._overhand_mode():
            # 2026-09-29 overhand: hand counts as wrapped only if its overhand
            # score > grasp_gate_min_wrap AND every finger (index, middle,
            # thumb) is within grasp_finger_max_bar_dist of the bar surface.
            ovh = self._overhand_finger_terms()
            return valid & ovh["left"][4] & ovh["right"][4]
        if min_wrap > 0.0:
            # 2026-09-28 handlefix5: distance alone passed fingers-on-the-lid.
            lw, rw = self._wrap_hand_scores()
            valid = valid & (lw > min_wrap) & (rw > min_wrap)
        return valid

    def _grasp_force_sustained_mask(self):
        """PUSH-stage grip-QUALITY check: _grasp_pose_valid_mask() (pure
        kinematic proximity, no force/contact-quality component at all) AND
        real sustained fingertip force on BOTH hands.

        2026-08-30, added per GRASP_FLOW_SUGGESTIONS_2026-08-30.md hypothesis
        #1 ("high confidence" root-cause finding): _grasp_pose_valid_mask()
        alone gates FOUR consumers (GRASP->PUSH entry, PUSH-stage reward,
        PUSH->COMPLETE, and the PUSH-stage grasp-loss termination clock) with
        nothing checking whether the hand is actually gripping vs. merely
        near the bar with a single light fingertip touch -- this decoupled
        "success" from "grip quality" project-wide (matches the doorman-
        style run's own observation: 98.8% valid-grasp occupancy, ~1.0
        contact-link touches per hand, rew_complete=0.0000 the entire run).

        USED ONLY for the PUSH-stage branch of _update_grasp_validity_counters
        (the grasp-loss grace-period termination clock, push_grasp_loss_grace_
        steps) -- deliberately NOT substituted into GRASP->PUSH entry
        (_stage_2_to_3_advance_condition), _stage_3_reward_condition, or
        _stage_3_to_complete_condition, which all stay on the original pure-
        proximity _grasp_pose_valid_mask(). Tightening ENTRY conditions is the
        exact lever that caused this project's earlier 0%-transition catch-22
        (the original 5-condition AND-gate, and _grasp_pose_valid_mask before
        the 2026-08-27 doorman-style simplification) -- this term targets
        SURVIVING push once already inside it, not getting in.

        grasp_force_sustained_floor grounded in measure_fingertip_force.py's
        real distribution (histogrammed fingertip force during PUSH-stage
        steps the proximity-only gate already calls "valid"), not a guess --
        see that script's output/the value's own config comment for the
        reasoning. UNTESTED against real Isaac Sim -- watch push_valid_grasp_
        frac_s3 alongside a NEW force-based analog once retested, and confirm
        via video that episodes surviving longer under this gate correspond
        to genuinely better-held grasps, not just a stricter number.
        """
        force_floor = float(self.config.get("grasp_force_sustained_floor", 1.0))
        left_tip_force, right_tip_force = self._fingertip_forces()
        return (
            self._grasp_pose_valid_mask()
            & (left_tip_force > force_floor)
            & (right_tip_force > force_floor)
        )

    def _update_grasp_validity_counters(self, env_ids):
        active = ~self.just_resetted_buf[env_ids]
        approach_ready = self._approach_transition_ready_mask()[env_ids]
        approach_steps = self._approach_transition_ready_steps[env_ids]
        self._approach_transition_ready_steps[env_ids] = torch.where(
            active
            & (self.stage_buf[env_ids] == self.STAGE_APPROACH)
            & approach_ready,
            approach_steps + 1,
            torch.zeros_like(approach_steps),
        )

        reach_ready = self._reach_transition_ready_mask()[env_ids]
        reach_steps = self._reach_transition_ready_steps[env_ids]
        self._reach_transition_ready_steps[env_ids] = torch.where(
            active
            & (self.stage_buf[env_ids] == self.STAGE_REACH)
            & reach_ready,
            reach_steps + 1,
            torch.zeros_like(reach_steps),
        )

        valid = self._grasp_pose_valid_mask()[env_ids]
        grasp_steps = self._grasp_valid_steps[env_ids]
        self._grasp_valid_steps[env_ids] = torch.where(
            active
            & (self.stage_buf[env_ids] == self.STAGE_GRASP)
            & valid,
            grasp_steps + 1,
            torch.zeros_like(grasp_steps),
        )
        # 2026-08-30: PUSH-stage survival now requires _grasp_force_sustained_
        # mask (proximity AND real fingertip force), not the plain proximity-
        # only `valid` used above for GRASP-stage entry counting -- see that
        # method's docstring. Losing real grip force, not just drifting a few
        # cm, is what should start the grasp-loss grace-period clock.
        # 2026-09-29 overhand (I3): counted in every grasp-holding stage
        # (TILT + PUSH); with push_grip_loose_check the check is the loose
        # _grip_maintained_mask (palm < 8cm AND closure > 60%) instead.
        if bool(self.config.get("push_grip_loose_check", False)):
            grip_ok = self._grip_maintained_mask()[env_ids]
        else:
            grip_ok = self._grasp_force_sustained_mask()[env_ids]
        invalid_push_steps = self._push_invalid_grasp_steps[env_ids]
        self._push_invalid_grasp_steps[env_ids] = torch.where(
            active
            & self._holding_stage_mask(self.stage_buf[env_ids])
            & ~grip_ok,
            invalid_push_steps + 1,
            torch.zeros_like(invalid_push_steps),
        )
        # 2026-09-29 overhand: TILT->PUSH hold counter (bin inside the tilt
        # band while the grip is maintained).
        if self._tilt_enabled:
            theta = self._bin_tilt_back_deg()[env_ids]
            lo = float(self.config.get("tilt_advance_min_deg", 10.0))
            hi = float(self.config.get("tilt_advance_max_deg", 22.0))
            in_band = (theta >= lo) & (theta <= hi)
            tilt_steps = self._tilt_hold_steps[env_ids]
            hold_ok = (
                active
                & (self.stage_buf[env_ids] == self.STAGE_TILT)
                & in_band
                & self._grip_maintained_mask()[env_ids]
            )
            # 2026-10-03 tilt7 (tilt_advance_max_rate_dps): steady-hold gate --
            # a step only counts while |d tilt/dt| <= the episode's threshold
            # (tilt5 / push1 handed over mid-swing at ~+31 deg/s: the 10-step
            # hold was met by swinging through the band).
            if getattr(self, "_tilt_rate_gate", False):
                hold_ok = hold_ok & (
                    self._bin_tilt_rate_dps()[env_ids].abs()
                    <= self._tilt_rate_thr_env[env_ids]
                )
            self._tilt_hold_steps[env_ids] = torch.where(
                hold_ok,
                tilt_steps + 1,
                torch.zeros_like(tilt_steps),
            )
            if bool(self.config.get("reanchor_bar_on_push_settle", False)):
                # 2026-10-03 push4: if the bin settles slightly deeper inside
                # [lo, hi] during the first push_settle_reanchor_steps (15 =
                # 0.3 s) of PUSH without rolling backward, re-anchor
                # _initial_bar_pos_xy so the small backward handle arc from
                # settling (e.g. 11 -> 14 deg = -4.5 cm) does not create a
                # negative bar-progress debt before forward walking begins.
                settle_steps = int(self.config.get("push_settle_reanchor_steps", 15))
                in_push_settle = (
                    active
                    & (self.stage_buf[env_ids] == self.STAGE_PUSH)
                    & (self.time_in_stage_buf[env_ids] <= settle_steps)
                    & in_band
                    & (self._bin_bar_push_progress()[env_ids] < 0.0)
                )
                if in_push_settle.any():
                    s_ids = env_ids[in_push_settle]
                    self._initial_bar_pos_xy[s_ids] = self._bar_mid_xy()[s_ids]

    # ----------------------------------------------------------------
    # 2026-09-29 overhand: shared helpers for the overhand claw grasp and the
    # TILT / PUSH stages.
    # ----------------------------------------------------------------
    def _finger_closure_fractions(self):
        """Per-hand finger closure on the p0 -> p1 primitive axis (N,) x2:
        projection of (q - p0) onto (p1 - p0), 0 = p0, 1 = p1. Same measure as
        the grasp_*_closure_s2 logs (handlefix5e)."""
        q = self.simulator.dof_pos

        def _closure(idx, p0, p1):
            span = (p1 - p0)[None, :]
            return ((q[:, idx] - p0[None, :]) * span).sum(-1) / span.square().sum(
                -1
            ).clamp(min=1e-6)

        return (
            _closure(self._left_hand_dof_idx, self._left_p0, self._left_p1),
            _closure(self._right_hand_dof_idx, self._right_p0, self._right_p1),
        )

    def _grip_maintained_mask(self):
        """'Grip maintained' for TILT/PUSH (I3). With push_grip_loose_check:
        both palms' grasp points within push_grip_max_bar_dist (0.08) of the
        bar AND both closures (p0->p1 axis) above push_grip_min_closure (0.6).
        Without it: the strict _grasp_pose_valid_mask (old behaviour).

        2026-10-03 push4: optional push_stage_grip_max_bar_dist /
        push_stage_grip_min_closure override the thresholds in STAGE_PUSH only
        (default = the TILT thresholds), allowing slight palm/wrist compliance
        during the dynamic TILT->PUSH stepping handover."""
        if not bool(self.config.get("push_grip_loose_check", False)):
            return self._grasp_pose_valid_mask()
        max_d = float(self.config.get("push_grip_max_bar_dist", 0.08))
        min_c = float(self.config.get("push_grip_min_closure", 0.6))
        push_max_d = float(self.config.get("push_stage_grip_max_bar_dist", max_d))
        push_min_c = float(self.config.get("push_stage_grip_min_closure", min_c))
        ld, rd = self._hand_bar_distances()
        lc, rc = self._finger_closure_fractions()
        if push_max_d != max_d or push_min_c != min_c:
            in_push = self.stage_buf == self.STAGE_PUSH
            max_d = torch.where(
                in_push,
                torch.full_like(ld, push_max_d),
                torch.full_like(ld, max_d),
            )
            min_c = torch.where(
                in_push,
                torch.full_like(lc, push_min_c),
                torch.full_like(lc, min_c),
            )
        return (ld < max_d) & (rd < max_d) & (lc > min_c) & (rc > min_c)

    def _tilt_band_weight(self, theta=None):
        """w_tilt = clamp((th-lo)/lo_ramp,0,1) * clamp((hi-th)/hi_ramp,0,1):
        0 for a flat bin, 1 between 10 and 22 deg, 0 again by 25 deg
        (defaults lo=5, lo_ramp=5, hi=25, hi_ramp=3)."""
        theta = self._bin_tilt_back_deg() if theta is None else theta
        lo = float(self.config.get("tilt_band_lo_deg", 5.0))
        lo_ramp = float(self.config.get("tilt_band_lo_ramp_deg", 5.0))
        hi = float(self.config.get("tilt_band_hi_deg", 25.0))
        hi_ramp = float(self.config.get("tilt_band_hi_ramp_deg", 3.0))
        return ((theta - lo) / lo_ramp).clamp(0.0, 1.0) * (
            (hi - theta) / hi_ramp
        ).clamp(0.0, 1.0)

    def _overhand_mode(self):
        return str(self.config.get("grasp_wrap_mode", "hook")) == "overhand"

    def _overhand_finger_terms(self):
        """Overhand claw wrap terms per hand (grasp_wrap_mode: overhand).

        Bin frame (see the handlefix3 block above _init_hand_link_indices):
        +x along the bar, +y from the lid toward the robot, +z up, measured
        from each hand's raw bar point (bar centre). Uses the REAL fingertip
        points (distal link origin + _overhand_tip_offsets).
          * index / middle: in-slot = smooth indicator of
            dy in [grasp_ovh_slot_dy_min, -R_BAR] (slot between bar and lid)
            AND dz < 0 (below the bar centre), times exp(-(d_surf/sigma)^2)
            with d_surf = hypot(dy, dz) - R_BAR at the tip.
          * thumb: under / in front of the bar: dz < 0 AND
            dy > grasp_ovh_thumb_dy_min (-1cm), times the same proximity.
          * hand score = (1-w) mean + w min over the three fingers
            (w = grasp_wrap_worst_weight, same semantics as the hook score).
          * wrapped = hand score > grasp_gate_min_wrap AND EVERY finger's
            distal phalanx (segment link-origin -> tip, in the bar's
            cross-section plane) within grasp_finger_max_bar_dist of the bar
            surface.
        Returns {side: (score (N,3), dsurf_tip (N,3), dsurf_seg (N,3),
        hand_score (N,), wrapped (N,) bool)}, fingers ordered
        (index, middle, thumb).
        """
        self._init_hand_link_indices()
        k = float(self.config.get("grasp_ovh_edge_width", 0.003))
        sigma = float(self.config.get("grasp_ovh_sigma", 0.01))
        dy_min = float(self.config.get("grasp_ovh_slot_dy_min", -0.042))
        dy_max = float(self.config.get("grasp_ovh_slot_dy_max", -_BAR_RADIUS))
        thumb_dy_min = float(self.config.get("grasp_ovh_thumb_dy_min", -0.01))
        max_dist = float(self.config.get("grasp_finger_max_bar_dist", 0.01))
        w = float(self.config.get("grasp_wrap_worst_weight", 0.5))
        min_score = float(self.config.get("grasp_gate_min_wrap", 0.5))
        q_inv = quat_inv(self.simulator.get_task_root_state("bin")[:, 3:7])  # wxyz
        out = {}
        self._ovh_last_sub = {}
        self._ovh_last_variants = {}
        self._ovh_last_tip = {}
        for side, bar in (
            ("left", self._left_bar_raw_pos()),
            ("right", self._right_bar_raw_pos()),
        ):
            idx = self._overhand_link_idx(side)  # (3,) index, middle, thumb
            n_k = idx.numel()
            origin_w = self.simulator._rigid_body_pos[:, idx, :]
            rot_w = xyzw_to_wxyz(self.simulator._rigid_body_rot[:, idx, :].reshape(-1, 4))
            tip_w = origin_w + quat_apply(
                rot_w,
                self._overhand_tip_offsets(side).expand(self.num_envs, -1, -1).reshape(-1, 3),
            ).reshape(self.num_envs, n_k, 3)
            qi = q_inv[:, None, :].expand(-1, n_k, 4).reshape(-1, 4)
            tip = quat_apply(qi, (tip_w - bar[:, None, :]).reshape(-1, 3)).reshape(
                self.num_envs, n_k, 3
            )
            org = quat_apply(qi, (origin_w - bar[:, None, :]).reshape(-1, 3)).reshape(
                self.num_envs, n_k, 3
            )
            dy, dz = tip[..., 1], tip[..., 2]
            dsurf_tip = torch.sqrt(dy * dy + dz * dz + 1e-10) - _BAR_RADIUS
            is_thumb = torch.zeros(n_k, dtype=torch.bool, device=self.device)
            is_thumb[-1] = True
            # Distal phalanx segment distance to the bar axis (cross-section).
            p0 = org[..., 1:3]
            seg = tip[..., 1:3] - p0
            t = (-(p0 * seg).sum(-1) / seg.square().sum(-1).clamp(min=1e-10)).clamp(0.0, 1.0)
            closest = p0 + t[..., None] * seg
            dsurf_seg = closest.norm(dim=-1) - _BAR_RADIUS
            # 2026-09-30 (grasp-init acceptance): at the validated claw pose the
            # index/middle TIP points sit ~2 cm off the bar surface (contact is
            # mid-phalanx) and their centreline passes ~1.03 cm from it (finger
            # half-thickness), so the tip-based prox kills the score (0.15).
            # Optional `grasp_ovh_prox_mode` (default "tip" = unchanged):
            #   "segment_prox": prox from the phalanx segment distance minus the
            #       finger radius (grasp_ovh_finger_radius, index/middle/thumb),
            #       slot/below/front indicators still at the tip;
            #   "segment": prox AND indicators at the closest segment point.
            radius = torch.tensor(
                self.config.get("grasp_ovh_finger_radius", [0.010, 0.010, 0.003]),
                device=self.device, dtype=torch.float32,
            )[None, :]

            def _score(d_eff, py, pz):
                prox_ = torch.exp(-torch.square(d_eff.clamp(min=0.0) / sigma))
                below_ = torch.sigmoid(-pz / k)
                slot_ = torch.sigmoid((py - dy_min) / k) * torch.sigmoid((dy_max - py) / k)
                front_ = torch.sigmoid((py - thumb_dy_min) / k)
                sc = torch.where(is_thumb[None, :], front_, slot_) * below_ * prox_
                hs = (1.0 - w) * sc.mean(dim=-1) + w * sc.min(dim=-1).values
                return sc, hs, (slot_, below_, front_, prox_)

            variants = {
                "tip": _score(dsurf_tip, dy, dz),
                "segment_prox": _score(dsurf_seg - radius, dy, dz),
                "segment": _score(dsurf_seg - radius, closest[..., 0], closest[..., 1]),
            }
            mode = str(self.config.get("grasp_ovh_prox_mode", "tip"))
            score, hand_score, (slot, below, thumb_front, prox) = variants[mode]
            wrapped = (hand_score > min_score) & (dsurf_seg < max_dist).all(dim=-1)
            out[side] = (score, dsurf_tip, dsurf_seg, hand_score, wrapped)
            # 2026-09-30: sub-terms kept for _grasp_init_debug_report only.
            self._ovh_last_sub[side] = (slot, below, thumb_front, prox, dy, dz)
            self._ovh_last_tip[side] = tip  # (N, 3 fingers, xyz) bin frame rel. bar
            self._ovh_last_variants[side] = {m: v[1] for m, v in variants.items()}
        return out

    def _overhand_link_idx(self, side):
        """Simulator body indices of (index_1, middle_1, thumb_2) for a hand."""
        if not hasattr(self, "_ovh_link_idx"):
            sim_names = list(self.simulator.body_names)
            self._ovh_link_idx = {
                s: torch.tensor(
                    [
                        sim_names.index(f"{s}_hand_{f}_link")
                        for f in ("index_1", "middle_1", "thumb_2")
                    ],
                    device=self.device, dtype=torch.long,
                )
                for s in ("left", "right")
            }
        return self._ovh_link_idx[side]

    def _overhand_tip_offsets(self, side):
        """(1,3,3) link-local fingertip points for (index_1, middle_1,
        thumb_2). Index/middle = far point of the distal hull (same as
        _wrap_tip_offsets); thumb = tip of the thumb_2 hull along its long
        axis (left hull y in [-0.052, 0.007], right hull mirrored y in
        [-0.007, 0.052], checked in g1_43dof_s3.usda 2026-09-29)."""
        if not hasattr(self, "_ovh_tip_off"):
            default = {
                "left": [[0.0519, 0.0038, -0.0022], [0.0519, 0.0038, -0.0022], [0.0, -0.047, 0.0]],
                "right": [[0.0519, -0.0038, 0.0022], [0.0519, -0.0038, 0.0022], [0.0, 0.047, 0.0]],
            }
            self._ovh_tip_off = {
                s: torch.tensor(
                    self.config.get(f"grasp_ovh_tip_offsets_{s}", default[s]),
                    device=self.device, dtype=torch.float32,
                )[None]
                for s in ("left", "right")
            }
        return self._ovh_tip_off[side]

    def _push_force_nomoment_raw(self):
        """Hand force on the bin (N) projected on the line from the handle to
        the wheel-ground contact (B4). A force along that line produces no
        tipping moment about the wheel contact, so it rolls the tilted bin
        instead of toppling it forward. Line = (mean wheel centre - wheel
        radius * z) - bar midpoint when the wheel bodies are found on the bin
        articulation, else a fixed push_force_line_angle_deg (60) below
        horizontal along the push axis. Clamped to [0, push_force_nomoment_max]
        (60 N)."""
        forces = self.simulator.object_to_hand_contact_forces[:, 0, :, :].sum(dim=1)  # (N,3)
        bar_mid = 0.5 * (self._left_bar_raw_pos() + self._right_bar_raw_pos())
        direction = None
        if not hasattr(self, "_bin_wheel_body_idx"):
            self._bin_wheel_body_idx = None
            try:
                names = list(self.simulator.scene.articulations["bin"].body_names)
                idx = [names.index(n) for n in ("wheel_left", "wheel_right")]
                self._bin_wheel_body_idx = torch.tensor(idx, device=self.device)
            except Exception as e:  # pragma: no cover - asset-dependent
                logger.warning(f"push_force_nomoment: wheel bodies not found ({e}); fixed angle")
        if self._bin_wheel_body_idx is not None:
            wheel_pos = self.simulator.scene.articulations["bin"].data.body_pos_w[
                :, self._bin_wheel_body_idx, :
            ].mean(dim=1)
            contact = wheel_pos.clone()
            contact[:, 2] = contact[:, 2] - self.wheel_radius
            direction = F.normalize(contact - bar_mid, dim=-1)
        if direction is None:
            ang = math.radians(float(self.config.get("push_force_line_angle_deg", 60.0)))
            axis_xy = F.normalize(self.push_axis_world[:, :2], dim=-1)
            direction = torch.cat(
                [math.cos(ang) * axis_xy, torch.full_like(axis_xy[:, :1], -math.sin(ang))],
                dim=-1,
            )
        max_f = float(self.config.get("push_force_nomoment_max", 60.0))
        self._push_line_dir = direction  # for the push_force_line_deg log
        return (forces * direction).sum(dim=-1).clamp(min=0.0, max=max_f)

    # ================================================================
    # RENDERING (eval video)
    # ================================================================

    def _setup_raw_eval_camera(self):
        """Create a raw UsdGeom.Camera + omni.replicator render product for
        eval video capture (env 0 only). We do NOT use the simulator's
        TiledCamera (`eval_camera`): in this Isaac Sim build its rendered
        viewpoint does not follow `set_world_poses_from_view` (the XformPrimView
        accepts the pose but the tiled render product still renders from the
        config offset), producing an unusable close-up. The raw camera + a
        replicator "rgb" annotator DOES honor a Gf.Matrix4d world transform —
        this is the technique validated by the camera fan-out (see
        move_garbage/scripts/vidtest_d.py + TRAINING_LOG 2026-07-12)."""
        import omni.replicator.core as rep
        from pxr import Gf, UsdGeom

        stage = omni.usd.get_context().get_stage()
        cam_path = "/World/eval_raw_camera"
        cam = UsdGeom.Camera.Define(stage, cam_path)
        cam.CreateFocalLengthAttr().Set(24.0)
        cam.CreateHorizontalApertureAttr().Set(20.955)
        cam.CreateClippingRangeAttr().Set(Gf.Vec2f(0.05, 1000.0))
        xf = UsdGeom.Xformable(cam.GetPrim())
        xf.ClearXformOpOrder()
        self._eval_cam_op = xf.AddTransformOp()
        self._eval_render_product = rep.create.render_product(cam_path, resolution=(1280, 720))
        self._eval_rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
        self._eval_rgb_annot.attach([self._eval_render_product])

    @staticmethod
    def _eval_look_at(eye, target, up=(0.0, 0.0, 1.0)):
        from pxr import Gf
        eye = Gf.Vec3d(float(eye[0]), float(eye[1]), float(eye[2]))
        target = Gf.Vec3d(float(target[0]), float(target[1]), float(target[2]))
        up = Gf.Vec3d(*up)
        fn = (target - eye).GetNormalized()
        r = Gf.Cross(fn, up).GetNormalized()
        u = Gf.Cross(r, fn).GetNormalized()
        m = Gf.Matrix4d(1.0)
        m.SetRow(0, Gf.Vec4d(r[0], r[1], r[2], 0.0))
        m.SetRow(1, Gf.Vec4d(u[0], u[1], u[2], 0.0))
        m.SetRow(2, Gf.Vec4d(-fn[0], -fn[1], -fn[2], 0.0))
        m.SetRow(3, Gf.Vec4d(eye[0], eye[1], eye[2], 1.0))
        return m

    @override
    def render_results(self):
        """Record a 3rd-person eval video framing BOTH the robot and the bin,
        using a raw UsdGeom.Camera + replicator capture (env 0 only)."""
        if not (
            self.config.simulator.config.get("render_results", False)
            and self.config.simulator.config.name == "isaacsim"
        ):
            return

        if self.debug_viz:
            self._draw_debug_vis()

        import numpy as np
        from pxr import UsdGeom

        # Frame env 0: look at the midpoint of robot + bin, backed off by their
        # separation so both stay in view.
        root_pos = self.simulator._rigid_body_pos[0, 0]           # env0 robot root
        bin_pos = self.simulator.get_task_root_state("bin")[0, :3]
        target = 0.5 * (root_pos + bin_pos)
        target = target + torch.tensor([0.0, 0.0, 0.4], device=self.device)
        sep = (root_pos - bin_pos).norm().item()
        standoff = min(max(2.5 * sep, 4.0), 9.0)
        view_dir = torch.tensor([-1.0, 1.0, 0.6], device=self.device)
        view_dir = view_dir / view_dir.norm()
        eye = target + standoff * view_dir

        # Author the eval_camera PRIM's transform directly, with resetXformStack
        # so it ignores its parent per-env Xform. ROOT CAUSE (found 2026-07-13):
        # eval_camera lives at /World/envs/env_0/eval_camera — a CHILD of the
        # per-env Xform (at env_0 origin = where the bin is). `set_world_poses`
        # updated the view but the TiledCamera re-derives pose from
        # parent_xform * local_offset each sim.render(), snapping it back onto
        # the bin. Writing the prim's own world matrix + resetXformStack makes
        # the render honor our pose. The env's internal sim.render()
        # (isaacsim.py:2166, cameras enabled) then populates the RGB buffer —
        # NO orchestrator.step() (which deadlocks) and NO extra sim.render().
        if "_eval_cam_prim_op" not in self.__dict__:
            stage = omni.usd.get_context().get_stage()
            cam_prim = stage.GetPrimAtPath("/World/envs/env_0/eval_camera")
            xf = UsdGeom.Xformable(cam_prim)
            xf.ClearXformOpOrder()
            self._eval_cam_prim_op = xf.AddTransformOp()
            xf.SetResetXformStack(True)  # ignore parent env Xform

        self._eval_cam_prim_op.Set(self._eval_look_at(eye.tolist(), target.tolist()))

        rgb = self.simulator.eval_camera.data.output["rgb"][0].detach().cpu().numpy()
        if rgb.ndim == 3 and rgb.shape[-1] == 4:
            rgb = rgb[..., :3]

        if "writers_rendering_results" not in self.__dict__:
            os.makedirs(self.config.save_rendering_dir, exist_ok=True)
            time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            file_name = f"{self.config.save_rendering_dir}/{time_str}_viewer_0.mp4"
            print(f"Saving rendering to {file_name}")
            self.writers_rendering_results = [
                imageio.get_writer(file_name, fps=int(1 / self.dt))
            ]
        self.writers_rendering_results[0].append_data(rgb)

    def _stage_0_reward_condition(self):
        return torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

    def _approach_transition_ready_mask(self):
        # Pelvis near the grasp-relative stop pose (XY), robot mostly stopped,
        # upper body near resting pose. Must match _reward_stop_at_bin's target so
        # the reward and the stage transition agree on WHERE to stop.
        cond = self._standoff_ready_mask()

        # 2026-10-06 full3: optional yaw squaring-up gate (stop_transition_yaw_tolerance_deg,
        # default None = off) so diagonal approaches from phi != 0 turn to face the bin
        # handle bar before entering STAGE_REACH.
        cond &= self._approach_yaw_ready_mask()

        # Require a genuine stop, not a cruise-through. Run 6 showed the robot
        # hovering the stop zone at ~0.27 m/s — just under the old 0.3 gate — so
        # it flickered into REACH without settling. Tightened to 0.2.
        root_vel = self.simulator.robot_root_states[:, 7:10].norm(dim=-1)
        cond &= root_vel < float(self.config.get("stage_advance_vel", 0.2))

        ang_vel_max = self.config.get("stage_advance_ang_vel", None)
        if ang_vel_max is not None:
            yaw_rate = self.simulator.robot_root_states[:, 12].abs()
            cond &= yaw_rate < float(ang_vel_max)

        if self.config.get("stage_advance_require_resting_arms", True):
            max_deviation = (
                torch.abs(
                    self.simulator.dof_pos[:, self._upper_non_finger_dof_idx]
                    - self.resting_dof_pos[:, self._upper_non_finger_dof_idx]
                ).max(dim=-1).values
            )
            arm_tolerance = float(
                self.config.get("stage_advance_arm_tolerance", 0.3)
            )
            cond &= max_deviation < arm_tolerance
        return cond

    def _stage_0_to_1_advance_condition(self):
        hold_steps = int(self.config.get("approach_transition_hold_steps", 1))
        cond = self._approach_transition_ready_mask()
        if hold_steps > 1:
            cond = cond & (self._approach_transition_ready_steps >= hold_steps)
        return cond

    def _stage_0_to_1_advance_callback(self, env_ids):
        # 2026-10-06 full3: StagedTaskBase.save_stage_states records
        # time_in_stage_buf = max_stage_time[0] into STAGE_REACH snapshots, and
        # check_stage_progression does not zero time_in_stage_buf unless
        # reset_stage_time_on_progression is True globally. Zeroing it here when
        # approach_reset_stage_timer_on_advance is True prevents a larger
        # max_stage_time[0] (e.g. 200 steps for 1.1 m walking) from giving
        # downstream stages an unintended 200-step budget.
        if bool(self.config.get("approach_reset_stage_timer_on_advance", False)):
            self.time_in_stage_buf[env_ids] = 0

    def _stage_1_reward_condition(self):
        # BUGFIX (Run 7): previously this aliased _stage_0_to_1_advance_condition(),
        # which requires the upper body to be NEAR RESTING (max_deviation < 0.3).
        # That made the REACH stage bonus vanish the instant the robot raised its
        # arms to reach the handles — i.e. it was PUNISHED for reaching. The robot
        # got stuck cruising the stop zone with arms down (reach plateaued ~0.04).
        # Fix: keep the REACH bonus alive on being at the stop pose + stopped
        # (small HOMIE command), WITHOUT the arms-down constraint — mirrors the
        # door task's _stage_1_reward_condition. Arm raising is now free to be
        # shaped by reach_target_distance (which is @effective_in_stage(REACH)).
        cond = self._standoff_ready_mask()
        cond &= torch.norm(self.get_physical_homie_commands()[:, :3], dim=1) <= 0.1
        return cond

    def _stage_1_to_2_advance_condition(self):
        hold_steps = int(self.config.get("reach_transition_hold_steps", 5))
        return self._reach_transition_ready_mask() & (
            self._reach_transition_ready_steps >= hold_steps
        )

    def _stage_2_reward_condition(self):
        still = torch.norm(self.get_physical_homie_commands()[:, :3], dim=1) <= 0.1
        return still & self._standoff_ready_mask()

    def _stage_2_to_3_advance_condition(self):
        hold_steps = int(self.config.get("grasp_transition_hold_steps", 8))
        return self._grasp_pose_valid_mask() & (
            self._grasp_valid_steps >= hold_steps
        )

    def _stage_2_to_3_advance_callback(self, env_ids):
        # Define push progress from the live bin pose at the moment the grasp is
        # validated. This also makes older staged snapshots, which predate the
        # tracked baseline buffer, safe to resume from.
        # 2026-09-29 overhand: with the TILT stage the baseline is taken at
        # TILT->PUSH instead (_stage_3_to_4_advance_callback), so the bin
        # motion during tilting does not count as push progress.
        if self._tilt_enabled:
            return
        self._initial_bin_pos_xy[env_ids] = self.simulator.get_task_root_state("bin")[
            env_ids, :2
        ].clone()
        self._initial_bar_pos_xy[env_ids] = self._bar_mid_xy()[env_ids]

    def _stage_3_reward_condition(self):
        # 2026-09-29 overhand: stage 3 is TILT when enable_tilt_stage, else PUSH.
        if self._tilt_enabled:
            return self._grip_maintained_mask()
        return self._grasp_pose_valid_mask()

    def _stage_3_to_complete_condition(self):
        # Old 4-stage layout only (enable_tilt_stage False); see
        # _stage_4_to_complete_condition for the 5-stage layout.
        target = float(self.config.get("target_bin_push_distance", 1.0))
        return self._grasp_pose_valid_mask() & (self._bin_push_progress() > target)

    # --- 2026-09-29 overhand: 5-stage layout (enable_tilt_stage) ---------
    # 0 APPROACH, 1 REACH, 2 GRASP, 3 TILT, 4 PUSH. StagedTaskBase only looks
    # these up when len(max_stage_time) == 5.
    def _stage_3_to_4_advance_condition(self):
        """TILT -> PUSH: back-tilt inside [tilt_advance_min_deg,
        tilt_advance_max_deg] (10..22) held for tilt_advance_hold_steps (10)
        consecutive steps with the grip maintained (_grip_maintained_mask).

        2026-10-03 tilt7: with tilt_advance_max_rate_dps (default None) the
        bin must also be steady -- |d tilt/dt| <= the episode's threshold on
        every hold step (counter) and on the gate step itself."""
        hold = int(self.config.get("tilt_advance_hold_steps", 10))
        theta = self._bin_tilt_back_deg()
        lo = float(self.config.get("tilt_advance_min_deg", 10.0))
        hi = float(self.config.get("tilt_advance_max_deg", 22.0))
        cond = (
            (theta >= lo)
            & (theta <= hi)
            & self._grip_maintained_mask()
            & (self._tilt_hold_steps >= hold)
        )
        if getattr(self, "_tilt_rate_gate", False):
            cond = cond & (self._bin_tilt_rate_dps().abs() <= self._tilt_rate_thr_env)
        return cond

    def _stage_3_to_4_advance_callback(self, env_ids):
        # Push-progress baseline = live bin pose when the tilted bin is handed
        # over to PUSH (was the 2->3 callback in the 4-stage layout).
        self._initial_bin_pos_xy[env_ids] = self.simulator.get_task_root_state("bin")[
            env_ids, :2
        ].clone()
        self._initial_bar_pos_xy[env_ids] = self._bar_mid_xy()[env_ids]
        # 2026-10-03 tilt7 (logging only): tilt rate / tilt at the handover
        # (same physics state as the gate check), over the last
        # tilt_gate_log_window handovers: mean signed rate, p90 of |rate|,
        # mean tilt. tilt5 5100 / push1: ~+31 deg/s (p90 41-46) at ~15 deg.
        if hasattr(self, "_gate_buf"):
            rows = torch.stack(
                [self._bin_tilt_rate_dps()[env_ids], self._bin_tilt_back_deg()[env_ids]],
                dim=-1,
            )[-self._gate_win:]
            k = rows.shape[0]
            idx = (torch.arange(k, device=self.device) + self._gate_ptr) % self._gate_win
            self._gate_buf[idx] = rows
            self._gate_ptr = (self._gate_ptr + k) % self._gate_win
            self._gate_count = min(self._gate_count + k, self._gate_win)
            b = self._gate_buf[: self._gate_count]
            self.log_dict["tilt_gate_rate_dps"] = b[:, 0].mean()
            self.log_dict["tilt_gate_rate_p90_dps"] = torch.quantile(b[:, 0].abs(), 0.9)
            self.log_dict["tilt_gate_tilt_deg"] = b[:, 1].mean()

    def _stage_4_reward_condition(self):
        return self._grip_maintained_mask()

    def _stage_4_to_complete_condition(self):
        """PUSH complete: progress >= target_bin_push_distance AND back-tilt
        in [push_complete_min_tilt_deg, push_complete_max_tilt_deg] (5..25)
        AND grip maintained.

        2026-09-29 overhand grasp_tilt: with `tilt_is_final` (default False)
        reaching PUSH is itself the success -- i.e. the TILT->PUSH gate
        (tilt 10..22 deg held tilt_advance_hold_steps with grip maintained)
        passed. Combine with reset_on_complete: True so the episode ends as a
        timeout (bootstrap) with the `complete` bonus."""
        # 2026-10-06 approach_dr: `approach_is_final` (default False) --
        # passing STAGE_APPROACH -> STAGE_REACH (held approach_transition_hold_steps
        # inside the stop+yaw+velocity gate) is itself the episode completion.
        if bool(self.config.get("approach_is_final", False)):
            return self.stage_buf >= self.STAGE_REACH
        if bool(self.config.get("tilt_is_final", False)):
            return self.stage_buf == self.STAGE_PUSH
        # 2026-09-30 overhand reach_grasp: `grasp_is_final` (default False)
        # -- entering TILT (GRASP->TILT gate: overhand wrap + finger gate +
        # palms within grasp_bar_valid_dist, held grasp_transition_hold_steps)
        # is the success. Same mechanism as tilt_is_final.
        if bool(self.config.get("grasp_is_final", False)):
            return self.stage_buf >= self.STAGE_TILT
        target = self._target_bin_push_distance_m()
        theta = self._bin_tilt_back_deg()
        lo = float(self.config.get("push_complete_min_tilt_deg", 5.0))
        hi = float(self.config.get("push_complete_max_tilt_deg", 25.0))
        return (
            (self.stage_buf == self.STAGE_PUSH)  # base class does not stage-mask this
            & self._grip_maintained_mask()
            & (self._bin_push_progress() >= target)
            & (theta >= lo)
            & (theta <= hi)
        )

    # ================================================================
    # REWARDS  (ported per-term from drawer_pull_homie, bimanual)
    # ================================================================

    @StagedTaskBase.effective_in_stage(STAGE_APPROACH)
    def _reward_walk_to_bin(self):
        """Reward pelvis velocity toward the bin, with a TIGHT close-range taper.

        Run 8 showed the robot stops correctly early (~0.55m) then creeps inward
        over training to 0.33m (past the 0.48 stop target) — walk_to_bin keeps
        paying approach velocity with no distance cutoff, overpowering stop_at_bin
        at close range. Fix: taper walk reward to zero ONLY in the last stretch,
        measured to the stop target.

        Run 4 tried a taper too and it BACKFIRED — but that band was broad
        (0.5-1.2m to the target, and the robot spawns ~1.8m out) so it damped the
        dense approach signal across most of the trajectory. This taper is TIGHT
        and CLOSE: full reward until `walk_taper_far` (0.6m from stop target),
        linearly to 0 by `walk_taper_near` (0.4m). The robot spawns ~2m out, so
        walk reward is FULL for essentially the entire approach and only yields in
        the final ~0.2m where stop_at_bin should take over — exactly the zone
        where the inward creep happens.
        """
        root_pos = self.simulator.robot_root_states[:, :3].clone()
        stop_target = root_pos.clone()
        stop_target[:, :2] = self._stop_target_xy()
        target_dir = F.normalize(stop_target - root_pos, dim=-1)
        # Fix (2026-08-24): target_vel's z-component is 0 by construction
        # (stop_target only overwrites XY, so target_dir has no vertical
        # component) but root_vel previously kept the real vertical velocity
        # from normal walking gait bob -- inflating the tracking error (and
        # suppressing this reward, scale 5.0, the largest APPROACH term) for
        # reasons unrelated to actually walking toward the bin. Mirrors
        # _reward_step_forward_while_pushing's existing root_vel[:, 2] = 0.0.
        root_vel = self.simulator.robot_root_states[:, 7:10].clone()
        root_vel[:, 2] = 0.0
        v_t = float(self.config.get("target_root_vel", 0.3))
        target_vel = v_t * target_dir
        std = float(self.config.get("walk_to_bin_vel_std", 0.15))
        vel_reward = self._tracking_reward_util(
            torch.linalg.norm(root_vel - target_vel, dim=-1),
            std=std, target=0.0, scale=1.0, offset=0.0,
        )
        # 2026-10-06 full3: optional baseline subtraction so standing still
        # (root_vel = 0) yields 0.0 instead of exp(-0.5 * (v_t / std)^2) = 0.135.
        if bool(self.config.get("walk_to_bin_zero_at_rest", False)):
            rest_val = math.exp(-0.5 * (v_t / std) ** 2)
            vel_reward = ((vel_reward - rest_val) / max(1.0 - rest_val, 1e-6)).clamp(min=0.0)
        # Tight taper measured to the stop target (not bin center): full beyond
        # walk_taper_far, linearly to 0 at walk_taper_near.
        stop_xy = self._stop_target_xy()
        dist = (root_pos[:, :2] - stop_xy).norm(dim=-1)
        near = float(self.config.get("walk_taper_near", 0.4))
        far = float(self.config.get("walk_taper_far", 0.6))
        taper = ((dist - near) / max(far - near, 1e-6)).clamp(0.0, 1.0)
        return vel_reward * taper

    @StagedTaskBase.effective_in_stage(STAGE_APPROACH)
    def _reward_stop_at_bin(self):
        """Reward settling at the grasp-relative stop pose (gates transition to REACH).

        Old design rewarded ``close_enough(<0.9m of bin center) * gaussian(vel, std=0.1)``.
        Trajectory analysis showed why it never fired: the pelvis DID enter the
        0.9 m circle, but (a) the circle is bin-centered so it doesn't imply the
        handles are reachable, and (b) the velocity std=0.1 needed <0.12 m/s while
        the robot cruised the zone at ~0.29 m/s, so the vel gate gave ~0.02.

        New design: a distance term toward the grasp-relative stop target
        (``_stop_target_xy``) instead of a hard circle, times a *loosened*
        velocity term (std=0.2). Being "stopped" now only pays off at the pose
        that sets up the grasp — coupling stop to grasp-readiness.
        """
        root_pos = self.simulator.robot_root_states[:, :3].clone()
        stop_xy = self._stop_target_xy()
        delta = root_pos[:, :2] - stop_xy  # (num_envs, 2) offset from stop target in XY

        # Run 14: decompose the stop offset into ALONG-approach (parallel to the
        # grasp-face normal) vs LATERAL (perpendicular), and penalize LATERAL much
        # harder. Isolation (ISO3) showed the robot squared up on the RIGHT handle
        # (pelvis y aligned with right handle, 0.23m off the left) — the isotropic
        # std=0.20 tolerated standing off-center, so the left arm couldn't reach and
        # min(L,R) stalled. A tight lateral std forces it to center on the handle
        # midpoint so BOTH arms are symmetric/equidistant.
        handle_mid = 0.5 * (self._left_handle_target()[:, :2] + self._right_handle_target()[:, :2])
        face_normal = handle_mid - self._bin_center_xy()
        face_normal = face_normal / face_normal.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        # lateral unit = face_normal rotated 90deg in-plane
        lateral = torch.stack([-face_normal[:, 1], face_normal[:, 0]], dim=-1)
        along_dist = (delta * face_normal).sum(dim=-1).abs()   # distance along approach
        lat_dist = (delta * lateral).sum(dim=-1).abs()         # lateral offset (must be ~0)

        along_std = float(self.config.get("stop_pos_std", 0.20))
        lat_std = float(self.config.get("stop_lat_std", 0.08))  # TIGHT: square up centered
        along_reward = self._tracking_reward_util(along_dist, std=along_std, target=0.0, scale=1.0, offset=0.0)
        lat_reward = self._tracking_reward_util(lat_dist, std=lat_std, target=0.0, scale=1.0, offset=0.0)
        vel_std = float(self.config.get("stop_vel_std", 0.2))
        root_vel = self.simulator.robot_root_states[:, 7:10].norm(dim=-1)
        vel_reward = self._tracking_reward_util(
            root_vel, std=vel_std, target=0.0, scale=1.0, offset=0.0
        )

        # 2026-10-06 full3: when stop_at_bin_include_yaw is True (default False),
        # include a coarse + fine position gradient, a yaw squaring-up kernel
        # gated on being near the stop target, and gate vel_reward by
        # (along_reward * lat_reward * yaw_reward) so standing still far away
        # earns zero stopping bonus.
        if bool(self.config.get("stop_at_bin_include_yaw", False)):
            coarse_std = float(self.config.get("stop_pos_coarse_std", 0.60))
            along_coarse = self._tracking_reward_util(
                along_dist, std=coarse_std, target=0.0, scale=1.0, offset=0.0
            )
            lat_coarse = self._tracking_reward_util(
                lat_dist, std=coarse_std, target=0.0, scale=1.0, offset=0.0
            )
            yaw_std_rad = math.radians(float(self.config.get("stop_yaw_std_deg", 15.0)))
            yaw_err = self._approach_yaw_error_rad().abs()
            yaw_reward = self._tracking_reward_util(
                yaw_err, std=yaw_std_rad, target=0.0, scale=1.0, offset=0.0
            )
            pos_gate = along_reward * lat_reward
            pose_gate = pos_gate * yaw_reward
            return (
                0.25 * (0.4 * along_coarse + 0.6 * along_reward)
                + 0.25 * (0.4 * lat_coarse + 0.6 * lat_reward)
                + 0.25 * (pos_gate * yaw_reward)
                + 0.25 * (pose_gate * vel_reward)
            )

        mode = self.config.get("stop_reward_mode", "product")
        if mode == "product":
            return along_reward * lat_reward * vel_reward
        if mode == "additive":
            weights = self.config.get("stop_reward_additive_weights", [0.4, 0.4, 0.2])
            if len(weights) != 3 or sum(weights) <= 0:
                raise ValueError(
                    "stop_reward_additive_weights must contain three positive-sum weights"
                )
            weights = [float(weight) / sum(weights) for weight in weights]
            return (
                weights[0] * along_reward
                + weights[1] * lat_reward
                + weights[2] * vel_reward
            )
        raise ValueError(f"Unknown stop_reward_mode: {mode}")

    @StagedTaskBase.effective_in_stage(STAGE_APPROACH)
    def _reward_approach_walk_command(self):
        """2026-10-06 full3: guide physical_homie_commands[:, :3] ([vx, vy, vyaw]
        in robot frame) during STAGE_APPROACH across HOMIE's 0.50 stand->walk
        threshold when outside the stop+yaw gate, and toward [0, 0, 0] once
        inside the stop+yaw gate so the robot settles into STAGE_REACH."""
        cmd = self.get_physical_homie_commands()[:, :3]
        root_xy = self.simulator.robot_root_states[:, :2]
        stop_xy = self._stop_target_xy()
        delta_w = stop_xy - root_xy
        dist = delta_w.norm(dim=-1)
        _, _, robot_yaw = euler_xyz_from_quat(
            xyzw_to_wxyz(self.simulator.robot_root_states[:, 3:7])
        )
        c_yaw = torch.cos(robot_yaw)
        s_yaw = torch.sin(robot_yaw)
        # Rotate world XY error into robot body frame (+x forward, +y left)
        dx_b = c_yaw * delta_w[:, 0] + s_yaw * delta_w[:, 1]
        dy_b = -s_yaw * delta_w[:, 0] + c_yaw * delta_w[:, 1]
        dir_b = F.normalize(torch.stack([dx_b, dy_b], dim=-1), dim=-1)

        # Target yaw error to square up with -face_normal (so desired vyaw = -yaw_err)
        yaw_err = self._approach_yaw_error_rad()
        target_vyaw = (-2.0 * yaw_err).clamp(-0.65, 0.65)

        # Speed scales down near stop_xy, but whenever outside the stop+yaw gate
        # keep target command norm >= 0.58 so HOMIE switches from stand_model to
        # walk_model (threshold in ppo_trainer_homie_api.py is 0.50).
        in_gate = self._standoff_ready_mask() & self._approach_yaw_ready_mask()
        lin_mag = (dist / 0.25).clamp(0.0, 1.0) * 0.55
        target_vxy = dir_b * lin_mag.unsqueeze(-1)
        target_cmd = torch.cat([target_vxy, target_vyaw.unsqueeze(-1)], dim=-1)
        cmd_norm = target_cmd.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        boosted_cmd = torch.where(
            cmd_norm < 0.58,
            target_cmd * (0.58 / cmd_norm),
            target_cmd,
        )
        # If both dist and yaw_err are tiny (<2 cm and <3 deg), fallback direction is +x
        tiny = (dist < 0.02) & (yaw_err.abs() < math.radians(3.0))
        fallback_cmd = torch.zeros_like(cmd)
        fallback_cmd[:, 0] = 0.58
        boosted_cmd = torch.where(tiny.unsqueeze(-1), fallback_cmd, boosted_cmd)

        desired_cmd = torch.where(
            in_gate.unsqueeze(-1),
            torch.zeros_like(cmd),
            boosted_cmd,
        )
        std = float(self.config.get("approach_walk_command_std", 0.25))
        return self._tracking_reward_util(
            torch.linalg.norm(cmd - desired_cmd, dim=-1),
            std=std, target=0.0, scale=1.0, offset=0.0,
        )

    @StagedTaskBase.effective_in_stage(STAGE_APPROACH)
    def _reward_approach_advance(self):
        """2026-10-06 full3: one-time milestone bonus on the step the episode
        passes the STAGE_APPROACH -> STAGE_REACH gate (mirrors grasp_advance
        and tilt_advance)."""
        return self._stage_0_to_1_advance_condition().float()

    @StagedTaskBase.effective_in_stage([STAGE_REACH, STAGE_GRASP])
    def _reward_maintain_standoff(self):
        """Hold the pelvis at the grasp-relative stance after APPROACH."""
        _, along_error, lateral_error = self._grasp_standoff_state()
        along_std = float(self.config.get("maintain_standoff_std", 0.08))
        lateral_std = float(
            self.config.get("maintain_standoff_lateral_std", 0.12)
        )
        along_reward = self._tracking_reward_util(
            along_error.abs(), std=along_std, target=0.0, scale=1.0, offset=0.0
        )
        lateral_reward = self._tracking_reward_util(
            lateral_error, std=lateral_std, target=0.0, scale=1.0, offset=0.0
        )
        return along_reward * lateral_reward

    @StagedTaskBase.effective_in_stage(
        [STAGE_APPROACH, STAGE_REACH, STAGE_GRASP]
    )
    def _reward_penalty_too_close_to_bin(self):
        """Stage-aware barrier that preserves clearance without penalizing distance."""
        signed_standoff, _, _ = self._grasp_standoff_state()
        in_approach = self.stage_buf == self.STAGE_APPROACH
        approach_min = float(self.config.get("approach_min_standoff", 0.40))
        grasp_min = float(self.config.get("min_grasp_standoff", 0.34))
        approach_margin = float(
            self.config.get("approach_too_close_penalty_margin", 0.08)
        )
        grasp_margin = float(self.config.get("too_close_penalty_margin", 0.10))
        min_standoff = torch.where(
            in_approach,
            torch.full_like(signed_standoff, approach_min),
            torch.full_like(signed_standoff, grasp_min),
        )
        margin = torch.where(
            in_approach,
            torch.full_like(signed_standoff, approach_margin),
            torch.full_like(signed_standoff, grasp_margin),
        )
        normalized_deficit = ((min_standoff - signed_standoff) / margin).clamp(min=0.0)
        # Quadratic near the boundary, then linear. This keeps a corrective
        # gradient after a large overshoot without allowing an unbounded square
        # to dominate every other reward.
        return torch.where(
            normalized_deficit <= 1.0,
            normalized_deficit.square(),
            2.0 * normalized_deficit - 1.0,
        )

    def _one_hand_reach(
        self, palm_idx, palm_pos, target, hand_dof_idx, p0, palm_alignment
    ):
        """Per-hand reach reward: split horizontal(xy)+vertical(z) position, plus
        velocity-toward-target, gated by finger-openness. Shared by the separate
        left/right reach reward terms.

        Run 12: split H+V so the policy can't farm the cheap vertical axis (lift)
        while leaving horizontal (extend/lean) at 0.5m. Widened std 0.12->0.25 so
        the Gaussian has a real slope from where the palm sits (0.12 was ~dead).

        Run 16: TOP-DOWN pre-grasp arc. Handles are a rim you grab from ABOVE, but
        the policy was approaching from BELOW and hitting the bottom rail. The
        vertical target is RAISED while the palm is still far horizontally, and
        blends down to the true handle as the palm closes in over it.

        Run 19 (2026-08-01, video review): checkpoint 6300 showed the left palm
        centered horizontally on the target (<0.03m) but floating +7cm above the
        true handle for 100+ consecutive frames in GRASP stage — closing fingers
        around the elevated waypoint, not the rim. Root cause: the OLD single
        v_std=0.20 vertical Gaussian gave ~94% reward at a 7cm vertical error, so
        there was no gradient pulling the palm down over the last few cm even
        though the arc's own target had already collapsed to ~0 there. Same
        coarse-vs-fine gap that horizontal already had (fixed Run 12/17) but
        vertical never got. Fixed with a coarse+fine vertical split mirroring the
        existing horizontal one, so the tolerance actually tightens once close.

        Run 20 (2026-08-01): the "arc" was a misnomer — Run 16/17's blend only
        ever DESCENDS as horiz shrinks (full lift at/beyond blend_dist, monotonic
        collapse to the true handle). It never asked the palm to rise above the
        rim; it just started elevated and came down. That let a policy satisfy
        the vertical target by approaching flat/from-below the whole time and
        only dropping the last few cm at the very end. Replaced with a genuine
        triangular HUMP in horiz-space: target height is the true handle height
        at both horiz=0 (centered, must be AT the rim to pass GRASP) and
        horiz>=blend_dist (still far out, no lift demanded yet — the huge
        vertical gap from the rest pose already dominates there), and peaks
        `pregrasp_arc_height` ABOVE the rim at horiz=pregrasp_arc_peak_frac *
        blend_dist. This forces an actual up-and-over swing during the middle
        of the approach, then back down onto the rim to finish — not just a
        one-way descent. `peak_frac` > 0.5 keeps the peak (and the steepest
        slope) positioned away from the palm's fine-tracking zone near the
        target, mirroring Run 17's lesson that a steep vertical goalpost near
        where the palm actually lingers causes jitter/swipe-through.
        """
        pregrasp_vel = self.config.get("pregrasp_target_vel", 0.3)
        h_std = float(self.config.get("reach_pos_std", 0.25))          # horizontal (xy)
        h_fine_std = float(self.config.get("reach_pos_fine_std", 0.08))
        h_fine_weight = float(self.config.get("reach_pos_fine_weight", 0.35))
        v_std = float(self.config.get("reach_pos_std_vert", 0.20))     # vertical (z), coarse
        v_fine_std = float(self.config.get("reach_pos_fine_std_vert", 0.04))
        v_fine_weight = float(self.config.get("reach_pos_fine_weight_vert", 0.5))
        horiz = torch.norm(target[:, :2] - palm_pos[:, :2], dim=-1)

        # Top-down pre-grasp ARC (Run 20): a smooth hump in horiz-space, not a
        # one-way blend. s=0 at the target, s=1 at/beyond blend_dist; hump=0 at
        # both ends, hump=1 at s=peak_frac. target_z sits `pregrasp_arc_height`
        # above the true handle at the peak, and exactly AT the handle at either
        # end — so the palm must swing up and over mid-approach, then descend
        # onto the rim to finish, instead of approaching flat or from below.
        # Built from two half-cosines (rising 0->peak_frac, falling peak_frac->1)
        # so the target height is continuous AND smooth (zero slope at both the
        # rim and the peak) — a real rounded arc, not a sharp-cornered "V".
        pregrasp_arc_h = float(self.config.get("pregrasp_arc_height", 0.10))
        peak_frac = float(self.config.get("pregrasp_arc_peak_frac", 0.75))
        blend_dist = float(self.config.get("pregrasp_blend_dist", 0.30))
        s = (horiz / blend_dist).clamp(0.0, 1.0)
        rising_phase = (s / peak_frac).clamp(0.0, 1.0)
        falling_phase = ((1.0 - s) / (1.0 - peak_frac)).clamp(0.0, 1.0)
        phase = torch.where(s < peak_frac, rising_phase, falling_phase)
        hump = 0.5 * (1.0 - torch.cos(math.pi * phase))
        target_z = target[:, 2] + pregrasp_arc_h * hump
        vert = torch.abs(target_z - palm_pos[:, 2])
        horiz_coarse = self._tracking_reward_util(
            horiz, std=h_std, target=0.0, scale=1.0, offset=0.0
        )
        horiz_fine = self._tracking_reward_util(
            horiz, std=h_fine_std, target=0.0, scale=1.0, offset=0.0
        )
        horiz_reward = (
            (1.0 - h_fine_weight) * horiz_coarse + h_fine_weight * horiz_fine
        )
        vert_coarse = self._tracking_reward_util(
            vert, std=v_std, target=0.0, scale=1.0, offset=0.0
        )
        vert_fine = self._tracking_reward_util(
            vert, std=v_fine_std, target=0.0, scale=1.0, offset=0.0
        )
        vert_reward = (1.0 - v_fine_weight) * vert_coarse + v_fine_weight * vert_fine
        # Run 18: WEIGHT HORIZONTAL HEAVILY. With an equal 0.5/0.5 split the policy
        # farmed the cheap vertical axis — palm parked at the right HEIGHT (vert_reward
        # ~0.87) while stuck 0.31 m short HORIZONTALLY (horiz_reward ~0.05), collecting
        # ~half the reach reward without ever extending toward the handle. Horizontal
        # extension is the actual objective, so weight it 0.8 and vertical 0.2: parking
        # at height now scores ~0.17 max, and the only way to raise reach reward is to
        # close the horizontal gap.
        h_weight = float(self.config.get("reach_horiz_weight", 0.8))
        pos_reward = h_weight * horiz_reward + (1.0 - h_weight) * vert_reward
        # velocity target points at the (elevated) pre-grasp point too, so the
        # approach velocity is shaped up-and-over, not straight through the rail.
        elevated = torch.stack([target[:, 0], target[:, 1], target_z], dim=-1)
        direction = F.normalize(elevated - palm_pos, dim=-1)
        palm_vel = self.simulator._rigid_body_vel[:, palm_idx, :]
        target_vel = pregrasp_vel * direction
        vel_reward = self._tracking_reward_util(
            torch.linalg.norm(palm_vel - target_vel, dim=-1),
            std=0.15, target=0.0, scale=1.0, offset=0.0,
        )
        finger_dev = torch.abs(self.simulator.dof_pos[:, hand_dof_idx] - p0).mean(dim=-1)
        finger_open = self._tracking_reward_util(finger_dev, std=0.3, target=0.0, scale=1.0, offset=0.0)
        # A sum followed by clamp let velocity tracking saturate this reward even
        # when the palm was far away, erasing the distance gradient. Keep velocity
        # as a small shaping term and let position dominate the optimum.
        velocity_weight = float(self.config.get("reach_velocity_weight", 0.15))
        finger_floor = float(self.config.get("reach_finger_gate_floor", 0.25))
        finger_gate = finger_floor + (1.0 - finger_floor) * finger_open
        orientation_floor = float(
            self.config.get("reach_orientation_gate_floor", 0.10)
        )
        orientation_score = (
            ((palm_alignment + 1.0) * 0.5).clamp(0.0, 1.0).square()
        )
        orientation_gate = orientation_floor + (1.0 - orientation_floor) * (
            orientation_score
        )
        return (
            ((1.0 - velocity_weight) * pos_reward + velocity_weight * vel_reward)
            * finger_gate
            * orientation_gate
        )

    @StagedTaskBase.effective_in_stage(STAGE_REACH)
    def _reward_reach_left(self):
        """LEFT palm toward the left handle. INDEPENDENT term (Run 15).

        History: 0.5*(L+R) let the policy farm the easy hand; min(L,R) gave the
        BETTER hand zero gradient (sparse-gradient local optimum) — isolation
        stalled with right palm at 0.10m but left stuck at 0.25m for 100s of iters.
        Separate per-hand terms give EACH hand its own persistent dense gradient
        toward its own handle, and each saturates at ~1.0 so the only way to raise
        total reward once one hand maxes is to bring the other in. Also self-
        diagnosing: left/right show as separate log lines.
        """
        left_palm, _ = self._palm_grasp_points()
        left_alignment, _ = self._palm_down_alignments()
        return self._one_hand_reach(
            self.left_palm_idx,
            left_palm,
            self._left_handle_target(),
            self._left_hand_dof_idx,
            # 2026-09-24: finger gate measured against the pre-shape pose.
            self._finger_preshape_poses()[0],
            left_alignment,
        )

    @StagedTaskBase.effective_in_stage(STAGE_REACH)
    def _reward_reach_right(self):
        """RIGHT palm toward the right handle. INDEPENDENT term (see _reward_reach_left)."""
        _, right_palm = self._palm_grasp_points()
        _, right_alignment = self._palm_down_alignments()
        return self._one_hand_reach(
            self.right_palm_idx,
            right_palm,
            self._right_handle_target(),
            self._right_hand_dof_idx,
            # 2026-09-24: finger gate measured against the pre-shape pose.
            self._finger_preshape_poses()[1],
            right_alignment,
        )

    @StagedTaskBase.effective_in_stage([STAGE_APPROACH, STAGE_REACH])
    def _reward_reach_finger_dof_pos_l1(self):
        """Both hands OPEN (fingers near *_p0). Averaged L+R Gaussian on dof deviation."""
        # 2026-09-24: target is the pre-shape pose (p0 when
        # finger_preshape_frac=0) -- see _finger_preshape_poses.
        left_pre, right_pre = self._finger_preshape_poses()
        left_diff = self.simulator.dof_pos[:, self._left_hand_dof_idx] - left_pre
        right_diff = self.simulator.dof_pos[:, self._right_hand_dof_idx] - right_pre
        left = self._tracking_reward_util(left_diff, std=0.3, target=0.0, scale=1.0, offset=0.0).mean(dim=-1)
        right = self._tracking_reward_util(right_diff, std=0.3, target=0.0, scale=1.0, offset=0.0).mean(dim=-1)
        return 0.5 * (left + right)

    @_effective_in_named_stages("GRASP", "TILT", "PUSH")  # 2026-09-29 overhand: + TILT
    def _reward_grasp_finger_dof_pos_l1(self):
        """Track a DISTANCE-GRADED finger target for both hands (open far
        from the bar, closed at it), gated by enable_pregrasp_curl -- falls
        back to the old unconditional p1 (fully closed) target when False.

        2026-08-27: originally switched target between p0/p1 based on a
        close-latch (the control-time override's own latch). Later that
        same day, changed to unconditional p1 the whole GRASP+PUSH span,
        matching NVlabs/GR00T-VisualSim2Real (doorman branch)
        envs/door/door_open_homie.py's _reward_grasp_finger_dof_pos_l1
        exactly, once fingers came under full continuous policy control (no
        override, see _postprocess_delta_actions).

        2026-08-29: changed again, per user request after ~5 runs of video
        review showed both hands landing with fingers flat, pressed on top
        of the bin's flat lid-lip surface rather than curled around it. The
        unconditional-p1 version rewards "closed" regardless of whether
        closing is physically sensible at the current position -- it can't
        distinguish "closing while genuinely at the bar" from "closing
        while resting flat on the lid, blocked from curling further". Now
        uses _pregrasp_curl_fraction (reconnected the same day from ~2 days
        of disconnected dead code -- see that method's docstring for the
        full history) to grade the target continuously from p0 (open, far
        from the bar) to p1 (closed, at the bar), so there's a real reward
        gradient toward closing PROGRESSIVELY as the hand approaches,
        instead of an all-or-nothing pull toward "closed" independent of
        position. grasp_precontact_open_reward_scale in the env yaml
        remains superseded/unread by either version -- left with a note,
        not deleted.

        UNTESTED against real Isaac Sim in this new graded-target form --
        same "watch it before trusting it" caveat as every other new term
        prepared alongside this change.
        """

        def _one_hand_graded(hand_dof_idx, p0, p1, fraction):
            target = p0[None, :] + fraction[:, None] * (p1[None, :] - p0[None, :])
            diff = self.simulator.dof_pos[:, hand_dof_idx] - target
            # 2026-09-25: blend in the WORST joint so one finger jammed on the
            # lid (seen in video: right middle / left index) costs reward.
            per_joint = self._tracking_reward_util(
                diff, std=0.3, target=0.0, scale=1.0, offset=0.0
            )
            w = float(self.config.get("grasp_finger_worst_weight", 0.0))
            return (1.0 - w) * per_joint.mean(dim=-1) + w * per_joint.min(dim=-1).values

        def _one_hand_unconditional(hand_dof_idx, p1):
            diff = self.simulator.dof_pos[:, hand_dof_idx] - p1[None, :]
            # 2026-09-25: blend in the WORST joint so one finger jammed on the
            # lid (seen in video: right middle / left index) costs reward.
            per_joint = self._tracking_reward_util(
                diff, std=0.3, target=0.0, scale=1.0, offset=0.0
            )
            w = float(self.config.get("grasp_finger_worst_weight", 0.0))
            return (1.0 - w) * per_joint.mean(dim=-1) + w * per_joint.min(dim=-1).values

        if self.config.get("enable_pregrasp_curl", False):
            left_frac, right_frac = self._pregrasp_curl_fraction()
            # 2026-09-24: never ask for LESS curl than the REACH pre-shape,
            # so GRASP ramps pre-shape -> closed instead of re-opening first.
            preshape = float(self.config.get("finger_preshape_frac", 0.0))
            left_frac = left_frac.clamp(min=preshape)
            right_frac = right_frac.clamp(min=preshape)
            left = _one_hand_graded(
                self._left_hand_dof_idx, self._left_p0, self._left_p1, left_frac
            )
            right = _one_hand_graded(
                self._right_hand_dof_idx, self._right_p0, self._right_p1, right_frac
            )
        else:
            # --- Original unconditional-p1 version (superseded 2026-08-29
            # by default, preserved for fallback via enable_pregrasp_curl) ---
            left = _one_hand_unconditional(self._left_hand_dof_idx, self._left_p1)
            right = _one_hand_unconditional(self._right_hand_dof_idx, self._right_p1)
        # 2026-09-25 handlefix3: closure only pays when the fingertips are
        # near the bar (mean tip prox, see _finger_wrap_scores). floor=1 = off.
        floor = float(self.config.get("grasp_finger_near_gate_floor", 1.0))
        if floor < 1.0:
            (_, lp), (_, rp) = self._finger_wrap_scores()
            left = left * (floor + (1.0 - floor) * lp.mean(dim=-1))
            right = right * (floor + (1.0 - floor) * rp.mean(dim=-1))
        return 0.5 * (left + right) * self._grasp_shape_stage_scale()

    @_effective_in_named_stages("GRASP", "TILT", "PUSH")  # 2026-09-29 overhand: + TILT
    def _reward_grasp_target_distance(self):
        """Both palms close to their handle targets. Averaged L+R Gaussian."""
        lp, rp = self._palm_grasp_points()
        ref_left, ref_right = self._grasp_reference_targets()
        ld = torch.norm(ref_left - lp, dim=-1)
        rd = torch.norm(ref_right - rp, dim=-1)
        left = self._tracking_reward_util(ld, std=0.1, target=0.0, scale=1.0, offset=0.0)
        right = self._tracking_reward_util(rd, std=0.1, target=0.0, scale=1.0, offset=0.0)
        return 0.5 * (left + right) * self._grasp_shape_stage_scale()

    @_effective_in_named_stages("GRASP", "TILT", "PUSH")  # 2026-09-29 overhand: + TILT
    def _reward_grasp_target_anisotropic(self):
        """Axis-aware grasp-point shaping with a wider approach region.

        The hard gate remains 4 cm horizontal and 3 cm vertical. Wider widths
        preserve useful signal at the current 8-11 cm operating point while
        making vertical error more costly than the isotropic reward does.
        """
        lp, rp = self._palm_grasp_points()
        ref_left, ref_right = self._grasp_reference_targets()
        ld = ref_left - lp
        rd = ref_right - rp
        horiz_std = float(
            self.config.get("grasp_target_anisotropic_horiz_std", 0.08)
        )
        vert_std = float(
            self.config.get("grasp_target_anisotropic_vert_std", 0.06)
        )

        def _one_hand(delta):
            horiz = delta[:, :2].norm(dim=-1)
            vert = delta[:, 2].abs()
            normalized_sq = (horiz / horiz_std) ** 2 + (vert / vert_std) ** 2
            return torch.exp(-0.5 * normalized_sq)

        return 0.5 * (_one_hand(ld) + _one_hand(rd)) * self._grasp_shape_stage_scale()

    @_effective_in_named_stages("GRASP", "TILT", "PUSH")  # 2026-09-29 overhand: + TILT
    def _reward_grasp_vert_progress(self):
        """Progress (potential-based) reward for REDUCING vertical
        palm-to-handle error step-over-step, mirroring
        _reward_palm_twist_progress but for the vertical axis specifically.

        Run 25 (2026-08-13): the fine proximity gate just added to
        _reward_grasp_contact correctly stopped rewarding a hand hovering
        ~10cm above the true handle (measure_standoff.py: vert_ok(<3cm) was
        ~0% for both hands despite reasonable horizontal centering). But the
        only remaining dense pull toward the fine target is
        grasp_target_distance's isotropic 3D Gaussian (std=0.1) - and
        removing ~0.65/step of grasp_contact reward all at once (5.0 scale x
        ~0.13 mean) measurably REGRESSED even horizontal centering within
        ~350 iterations (horiz_ok dropped 40%->16% left, 15%->9% right),
        while vertical stayed pinned at ~0%. Added a dedicated potential-based
        term for vertical error specifically (the diagnosed, non-moving
        bottleneck) so there's an explicit, non-vanishing gradient pulling
        the palm down onto the target even while grasp_contact stays
        unsatisfied. Masks the first step of every episode, same as
        _reward_palm_twist_progress.
        """
        lp, rp = self._palm_grasp_points()
        ref_left, ref_right = self._grasp_reference_targets()
        left_vert_error = (ref_left - lp)[:, 2].abs()
        right_vert_error = (ref_right - rp)[:, 2].abs()
        max_step_error = float(self.config.get("grasp_vert_progress_norm", 0.3))
        valid = (self.episode_length_buf > 1).float()
        left_progress = (
            (self._prev_left_vert_error - left_vert_error) / max_step_error * valid
        )
        right_progress = (
            (self._prev_right_vert_error - right_vert_error) / max_step_error * valid
        )
        self._prev_left_vert_error = left_vert_error.detach().clone()
        self._prev_right_vert_error = right_vert_error.detach().clone()
        return 0.5 * (left_progress + right_progress)

    @_effective_in_named_stages("GRASP", "TILT", "PUSH")  # 2026-09-29 overhand: + TILT
    def _reward_grasp_bar_distance(self):
        """Dense pull toward the RAW bar position (not the offset hover
        target) -- new 2026-08-29.

        grasp_target_distance/grasp_target_anisotropic (above) both target
        _left/right_handle_target, the pre-contact HOVER point offset by
        grasp_target_standoff+height_offset (~4.12cm from the true bar) --
        correct for REACH-stage approach guidance, deliberately left
        unchanged. But grasp_bar_valid_dist (the validity gate, see
        _hand_bar_distances) is a hard 0/1 threshold measured against the
        RAW bar with NO partial credit below it -- so once the policy
        settled near its hover-target equilibrium (~0.10-0.11m from the
        true bar), there was no further gradient pulling it the remaining
        ~7-8cm. Confirmed via a real 150-iteration trend: bar distance
        dropped fast early (adapting toward the new gate existing at all),
        then plateaued flat for 40+ iterations once it reached the
        hover-target's own equilibrium distance. This term gives that
        missing pull, same isotropic-Gaussian shape as grasp_target_distance.
        """
        # 2026-09-30: distance source selectable via grasp_bar_reward_target
        # (default "bar" = the raw-bar distance used here before).
        left_dist, right_dist = self._grasp_bar_reward_distances()
        left = self._tracking_reward_util(left_dist, std=0.1, target=0.0, scale=1.0, offset=0.0)
        right = self._tracking_reward_util(right_dist, std=0.1, target=0.0, scale=1.0, offset=0.0)
        return 0.5 * (left + right) * self._grasp_shape_stage_scale()

    def _grasp_bar_reward_distances(self):
        """2026-09-30 (grasp_bar_reward_target): per-hand distance used by
        _reward_grasp_bar_distance / _reward_grasp_bar_progress.

        "bar" (default, old behaviour): `_hand_bar_distances()`, palm grasp
        point -> RAW bar centre. Direction-blind: at reach_grasp1 step 2300
        the left palm parked 4-5cm UNDER the bar and still scored ~5cm.
        "grasp_target": palm grasp point -> `_grasp_point_targets()`, where
        the grasp point sits when the palm LINK is at the handle target pose
        (`_left/right_handle_target()` + target rotation). With the overhand
        offsets that is 2.95 horiz / 3.35 vert = 4.46cm above/in front of the
        bar. NOT the handle target itself: that is a palm-LINK pose 8.89cm
        from the bar (standoff 0.072 / height 0.0521, measured
        dist_*_hover_to_bar_s2), outside grasp_bar_valid_dist 0.07, so its
        optimum could never validate. Its minimum is above the rim, so
        pulling toward it from underneath drives the palm up into the rim
        (and penalty_palm_below_bar makes staying there clearly worse).
        """
        mode = str(self.config.get("grasp_bar_reward_target", "bar"))
        if mode == "bar":
            return self._hand_bar_distances()
        if mode == "grasp_target":
            lp, rp = self._palm_grasp_points()
            lt, rt = self._grasp_point_targets()
            return torch.norm(lt - lp, dim=-1), torch.norm(rt - rp, dim=-1)
        raise ValueError(
            f"grasp_bar_reward_target must be 'bar' or 'grasp_target', got {mode!r}"
        )

    def _grasp_point_targets(self):
        """2026-09-30: world position of each palm GRASP point when the palm
        link is exactly at its handle target pose -- same construction as
        `_palm_grasp_points` (link + R * offset * grasp_axis) but with the
        target pose. See `_grasp_bar_reward_distances`."""
        left_axis, right_axis = self._palm_grasp_axes()
        offset = float(self.config.get("palm_grasp_point_offset", 0.0464))
        left = self._left_handle_target() + quat_apply(
            xyzw_to_wxyz(self._left_handle_target_rot()), offset * left_axis
        )
        right = self._right_handle_target() + quat_apply(
            xyzw_to_wxyz(self._right_handle_target_rot()), offset * right_axis
        )
        return left, right

    @_effective_in_named_stages("GRASP", "TILT", "PUSH")  # 2026-09-29 overhand: + TILT
    def _reward_grasp_bar_progress(self):
        """Progress (potential-based) reward for REDUCING raw bar distance
        step-over-step -- new 2026-08-29, mirrors _reward_grasp_vert_progress
        above (added for an analogous plateau on vertical error: a dense
        Gaussian alone regressed under competing pressure once removed/
        insufficient, a dedicated non-vanishing progress term fixed it).
        See _reward_grasp_bar_distance docstring for the full motivation.
        Masks the first step of every episode, same as the other progress
        rewards in this file.
        """
        # 2026-09-30: same selectable distance as _reward_grasp_bar_distance
        # (grasp_bar_reward_target). The prev buffers always hold the distance
        # of the CURRENT mode; if the mode ever differs from the one that
        # filled them (first call, or a runtime config change) this step is
        # masked for all envs and the buffers are refilled.
        left_dist, right_dist = self._grasp_bar_reward_distances()
        mode = str(self.config.get("grasp_bar_reward_target", "bar"))
        max_step_error = float(self.config.get("grasp_bar_progress_norm", 0.3))
        valid = (self.episode_length_buf > 1).float()
        if getattr(self, "_prev_bar_dist_mode", mode) != mode:
            valid = torch.zeros_like(valid)
        self._prev_bar_dist_mode = mode
        left_progress = (
            (self._prev_left_bar_dist - left_dist) / max_step_error * valid
        )
        right_progress = (
            (self._prev_right_bar_dist - right_dist) / max_step_error * valid
        )
        self._prev_left_bar_dist = left_dist.detach().clone()
        self._prev_right_bar_dist = right_dist.detach().clone()
        return 0.5 * (left_progress + right_progress)

    @_effective_in_named_stages("GRASP", "TILT", "PUSH")  # 2026-09-29 overhand: + TILT
    def _reward_grasp_fingertip_curl(self):
        """Reward genuine FINGERTIP contact, smoothly gated by proximity to
        the true bar -- new 2026-08-29, prepared (not yet deployed) after
        direct video review showed both hands landing with fingers nearly
        flat, resting ON TOP of the bin's flat lid-lip surface rather than
        curled under/around it. The lid surface directly blocks further
        downward curl once the fingers are already pressed flat against it
        -- a plausible local optimum (hard-contact discontinuity), not
        something the existing rewards specifically detect: neither
        _reward_grasp_finger_dof_pos_l1 (a DOF-angle target, agnostic to
        whether closing is physically blocked) nor _reward_grasp_bar_distance
        (palm position only, see that method's docstring -- doesn't know or
        care about finger state) can tell a genuine wrap apart from a flat
        press with the palm merely nearby.

        Two ingredients, combined multiplicatively:
        1. PROXIMITY GATE (smooth, not hard-boolean, matching this file's
           established preference -- see _grasp_readiness_score's docstring
           for why a hard AND-gate was previously replaced with a smooth
           product): a Gaussian falloff on _hand_bar_distances (the same
           raw-bar distance _reward_grasp_bar_distance uses), so this term
           is inert far from the bar and ramps in near it -- doesn't reward
           premature closing during REACH-equivalent approach.
        2. FINGERTIP-SPECIFIC contact force magnitude: sums contact force at
           just the 3 distal/fingertip links per hand (index_1, middle_1,
           thumb_2 -- confirmed via direct inspection of
           gr00t/rl/data/robots/g1/g1_43dof_s3.usda, out of 8 total
           candidate links per hand: palm_link + index_0/1, middle_0/1,
           thumb_0/1/2). A flat press mostly loads the base/middle segments
           and the palm; a genuine hook-wrap specifically loads the
           fingertip on the far side of the bar. Reuses the whole-bin-body
           object_to_hand_contact_forces tensor (same one _reward_grasp_
           contact already reads, proven safe -- no new ContactSensor, no
           new collision geometry, none of the simulation_app.close() hang
           risk from the reverted bar-collider approach) -- safe here
           specifically BECAUSE it's multiplied by the proximity gate above,
           so a fingertip touching some unrelated far-away part of the bin
           while the palm is simultaneously near the true bar isn't
           physically possible; the combination is effectively localized
           even though the raw sensor isn't.

        UNTESTED against real Isaac Sim -- prepared per user request, not
        yet synced/deployed. Needs the same empirical validation as every
        other new term this session (watch its mean, confirm sign/magnitude
        make sense, before trusting it).
        """
        left_dist, right_dist = self._hand_bar_distances()
        near_std = float(self.config.get("grasp_curl_near_std", 0.05))
        left_near = torch.exp(-0.5 * (left_dist / near_std) ** 2)
        right_near = torch.exp(-0.5 * (right_dist / near_std) ** 2)

        left_tip_force, right_tip_force = self._fingertip_forces()
        max_force = float(self.config.get("grasp_fingertip_curl_force_max", 5.0))
        left_force_term = (left_tip_force / max_force).clamp(max=1.0)
        right_force_term = (right_tip_force / max_force).clamp(max=1.0)

        return 0.5 * (left_near * left_force_term + right_near * right_force_term) * self._grasp_shape_stage_scale()

    def _fingertip_forces(self):
        """Raw per-hand fingertip contact force magnitude (Newtons), summed
        across the 3 distal/fingertip links per hand (index_1, middle_1,
        thumb_2). Extracted 2026-08-30 from _reward_grasp_fingertip_curl's
        body -- PURE EXTRACTION, no math change -- so the raw force is
        reusable by _grasp_force_sustained_mask (new PUSH-stage grip-quality
        termination gate, see GRASP_FLOW_SUGGESTIONS_2026-08-30.md hypothesis
        #1) and by measure_fingertip_force.py without duplicating the lazy
        fingertip-link-mask construction. See _reward_grasp_fingertip_curl's
        own docstring for the full design rationale (why these 3 links, why
        this contact-force tensor is safe to reuse without new geometry)."""
        forces = self.simulator.object_to_hand_contact_forces[:, 0, :, :]
        if not hasattr(self, "_left_fingertip_mask"):
            g1_hand_links = [
                n for n in self.simulator.robot_config.body_names
                if ("left_hand" in n or "right_hand" in n)
            ]
            fingertip_substrings = ("index_1", "middle_1", "thumb_2")
            self._left_fingertip_mask = [
                i for i, n in enumerate(g1_hand_links)
                if "left_hand" in n and any(s in n for s in fingertip_substrings)
            ]
            self._right_fingertip_mask = [
                i for i, n in enumerate(g1_hand_links)
                if "right_hand" in n and any(s in n for s in fingertip_substrings)
            ]
        left_tip_force = forces[:, self._left_fingertip_mask, :].norm(dim=-1).sum(dim=-1)
        right_tip_force = forces[:, self._right_fingertip_mask, :].norm(dim=-1).sum(dim=-1)
        return left_tip_force, right_tip_force

    @_effective_in_named_stages("GRASP", "TILT", "PUSH")  # 2026-09-29 overhand: + TILT
    def _reward_penalty_grasp_flat_press(self):
        """Penalize non-fingertip contact force, proximity-gated -- the
        logical complement of _reward_grasp_fingertip_curl above.

        2026-08-29: added per user request after direct video review (see
        _reward_grasp_fingertip_curl docstring for the full incident) showed
        both hands consistently landing with fingers flat, pressed on top
        of the bin's flat lid-lip surface -- a light, explicit penalty for
        exactly that pattern, not just "fail to reward" it. No new geometry
        or sensors: there is no signal anywhere in this codebase (and
        deliberately none added -- see history below) that can detect
        contact specifically ON the flat lid region; a prior attempt to
        build one (a small collision sphere + dedicated ContactSensors on
        the bin asset) reliably hung simulation_app.close() at training
        shutdown and was reverted (see _hand_bar_distances docstring).
        Instead this reuses the exact same proximity gate and whole-body
        contact tensor as _reward_grasp_fingertip_curl, just summed over
        the COMPLEMENT link set (palm_link + the base/middle knuckle links:
        index_0, middle_0, thumb_0, thumb_1) instead of the 3 fingertip
        links. Physically: a flat press mostly loads the palm and base
        knuckles; a genuine wrap loads the fingertip specifically and much
        less of the base. Penalizing this complement while near the bar is
        a cheap, safe proxy for "penalize contact on the flat surface",
        without needing to geometrically distinguish lid from bar at all.

        UNTESTED against real Isaac Sim -- prepared alongside
        _reward_grasp_fingertip_curl's own untested status; watch its mean
        (should be small/negative, not swamping the positive fingertip
        term) before trusting the sign/magnitude.
        """
        left_dist, right_dist = self._hand_bar_distances()
        near_std = float(self.config.get("grasp_curl_near_std", 0.05))
        left_near = torch.exp(-0.5 * (left_dist / near_std) ** 2)
        right_near = torch.exp(-0.5 * (right_dist / near_std) ** 2)

        forces = self.simulator.object_to_hand_contact_forces[:, 0, :, :]
        if not hasattr(self, "_left_fingertip_mask"):
            g1_hand_links = [
                n for n in self.simulator.robot_config.body_names
                if ("left_hand" in n or "right_hand" in n)
            ]
            fingertip_substrings = ("index_1", "middle_1", "thumb_2")
            self._left_fingertip_mask = [
                i for i, n in enumerate(g1_hand_links)
                if "left_hand" in n and any(s in n for s in fingertip_substrings)
            ]
            self._right_fingertip_mask = [
                i for i, n in enumerate(g1_hand_links)
                if "right_hand" in n and any(s in n for s in fingertip_substrings)
            ]
        # 2026-09-28 handlefix5: in the front hook the palm SHOULD press the bar.
        exclude_palm = bool(self.config.get("grasp_flat_press_exclude_palm", False))
        if not hasattr(self, "_left_flat_press_mask"):
            g1_hand_links = [
                n for n in self.simulator.robot_config.body_names
                if ("left_hand" in n or "right_hand" in n)
            ]
            fingertip_substrings = ("index_1", "middle_1", "thumb_2")
            self._left_flat_press_mask = [
                i for i, n in enumerate(g1_hand_links)
                if "left_hand" in n and not any(s in n for s in fingertip_substrings)
                and not (exclude_palm and "palm" in n)
            ]
            self._right_flat_press_mask = [
                i for i, n in enumerate(g1_hand_links)
                if "right_hand" in n and not any(s in n for s in fingertip_substrings)
                and not (exclude_palm and "palm" in n)
            ]
        left_flat_force = forces[:, self._left_flat_press_mask, :].norm(dim=-1).sum(dim=-1)
        right_flat_force = forces[:, self._right_flat_press_mask, :].norm(dim=-1).sum(dim=-1)
        max_force = float(self.config.get("grasp_flat_press_force_max", 5.0))
        left_penalty_term = (left_flat_force / max_force).clamp(max=1.0)
        right_penalty_term = (right_flat_force / max_force).clamp(max=1.0)

        return 0.5 * (left_near * left_penalty_term + right_near * right_penalty_term)

    @_effective_in_named_stages("REACH", "GRASP", "TILT", "PUSH")  # 2026-09-29 overhand: + TILT
    def _reward_grasp_contact(self):
        """Continuous, physics-grounded force reward -- replaces the prior
        geometry-gated version (see the commented block below for its full
        "Run 23"/"Run 25" incident history: two prior real bugs where gating
        this reward on face-normal alignment, then decomposed proximity, got
        the policy reinforced for sitting in a physically-wrong wedge or
        hovering above the target without ever actually descending onto it
        -- the same "AND together too many conditions" failure category the
        rest of this 2026-08-27 change is fixing elsewhere in this file).

        Modeled on NVlabs/GR00T-VisualSim2Real (doorman branch)
        envs/door/door_open_homie.py's _reward_grasp: rotate real per-link
        contact forces into a palm-referenced frame, reward the force
        component pointing OUT of the palm (correct grip direction),
        penalize the perpendicular (lateral/shear) component, smoothly
        clamped -- no geometric gating at all. During REACH (this task's
        pregrasp-equivalent stage, before contact is wanted) inverted to a
        pure penalty for ANY force, matching doorman's pregrasp behavior.

        Adapted, not a blind port: rotates each per-link force into the
        PALM LINK's own body frame (matching how _palm_down_alignments /
        _palm_grasp_axes already anchor everything to the palm link, not
        each individual finger link), then takes the component along
        left_palm_contact_axis / right_palm_contact_axis (already a plain
        unit-axis vector in this file, reused from _palm_down_alignments)
        as "outward", and the L2 norm of the residual as "lateral" -- doorman
        instead does a second rotation into a quaternion-valued "palm side
        direction" we don't have a config equivalent for, so this uses a
        dot-product decomposition against our existing axis vector instead.

        UNTESTED against real Isaac Sim -- quat_inv's exact signature isn't
        verifiable in this local sandbox (IsaacLab isn't installed here),
        and the sign of left/right_palm_contact_axis as an "outward"
        direction needs empirical confirmation (log this reward's mean
        during a scripted open->close rollout on the VM, confirm it goes
        positive on real contact) before trusting it, per the approved plan.
        """
        left_axis, right_axis = self._palm_contact_axes()
        left_rot = xyzw_to_wxyz(self.simulator._rigid_body_rot[:, self.left_palm_idx, :])
        right_rot = xyzw_to_wxyz(self.simulator._rigid_body_rot[:, self.right_palm_idx, :])

        # 2026-08-29: reverted to the whole-bin-body object_to_hand_contact_
        # forces -- the bar-scoped ContactSensor pair this briefly read
        # (2026-08-27) was removed entirely (see _hand_bar_distances
        # docstring: reliably hung simulation_app.close() at shutdown).
        # This reward is about force DIRECTION (grip quality), which is
        # orthogonal to the bar-distance validity gate (_grasp_pose_valid_
        # mask) that actually needed the spatial scoping fix -- going back
        # to the whole-body sensor here doesn't reintroduce that problem.
        forces = self.simulator.object_to_hand_contact_forces[:, 0, :, :]
        if not hasattr(self, "_left_contact_mask"):
            g1_hand_links = [
                n for n in self.simulator.robot_config.body_names
                if ("left_hand" in n or "right_hand" in n)
            ]
            self._left_contact_mask = [i for i, n in enumerate(g1_hand_links) if "left_hand" in n]
            self._right_contact_mask = [i for i, n in enumerate(g1_hand_links) if "right_hand" in n]

        # 2026-09-12: NORMALIZED to [-1, 1] per hand, then averaged across hands
        # (was: raw Newtons per link, clamped +-10, MEAN over links, then SUMMED
        # over both hands -- so the term's value was in force units while every
        # other shaped reward in this file is a bounded [0,1] Gaussian/cosine).
        # That scale mismatch, not the 5.0 weight, is why this became the
        # dominant term in the whole reward stack: it earned a mean of 2.896
        # while palm_down_alignment earned 0.051 (57x) and the policy visibly
        # traded palm orientation away for contact force -- grasp_left_palm_
        # down_s2 moved BACKWARDS 0.125 -> 0.064 over the run while this term
        # climbed. Dividing by grasp_contact_force_norm puts a weight of N here
        # on the same footing as a weight of N anywhere else in the file.
        #
        # Deliberately NOT re-gated on alignment or proximity: the commented
        # block below is the history of two real bugs caused by exactly that
        # (hard AND-gates on face-normal, then on decomposed proximity, both
        # reinforced a physically-wrong wedge). Fixing the units is the
        # narrower change; the alignment signal is raised on its own weight
        # instead.
        force_norm = float(self.config.get("grasp_contact_force_norm", 10.0))

        def _hand_force_reward(contact_mask, rot, axis):
            link_forces = forces[:, contact_mask, :]  # (num_envs, num_links, 3)
            num_links = link_forces.shape[1]
            rot_expand = rot[:, None, :].expand(-1, num_links, -1).reshape(-1, 4)
            force_flat = link_forces.reshape(-1, 3)
            force_palm_frame = quat_apply(quat_inv(rot_expand), force_flat)
            axis_expand = axis[:, None, :].expand(-1, num_links, -1).reshape(-1, 3)
            outward = (force_palm_frame * axis_expand).sum(dim=-1)
            lateral = (force_palm_frame - outward[:, None] * axis_expand).norm(dim=-1)
            reward = ((outward - lateral) / force_norm).clamp(min=-1.0, max=1.0)
            return reward.reshape(self.num_envs, num_links).mean(dim=-1)

        left_reward = _hand_force_reward(self._left_contact_mask, left_rot, left_axis)
        right_reward = _hand_force_reward(self._right_contact_mask, right_rot, right_axis)
        reward = 0.5 * (left_reward + right_reward)

        in_reach = self.stage_buf == self.STAGE_REACH
        return torch.where(in_reach, -1.0 * torch.abs(reward), reward * self._grasp_shape_stage_scale())

        # --- Original geometry-gated version (superseded 2026-08-27) ---
        # left_count, right_count = self._hand_contact_counts()
        # required = float(self.config.get("grasp_contact_links_required", 1))
        # left_reward = (left_count / required).clamp(max=1.0)
        # right_reward = (right_count / required).clamp(max=1.0)
        #
        # lp, rp = self._palm_grasp_points()
        # close_horiz = float(self.config.get("grasp_finger_close_horiz_dist", 0.04))
        # close_vert = float(self.config.get("grasp_finger_close_vert_dist", 0.03))
        # left_delta = self._left_handle_target() - lp
        # right_delta = self._right_handle_target() - rp
        # left_close = (
        #     (left_delta[:, :2].norm(dim=-1) < close_horiz)
        #     & (left_delta[:, 2].abs() < close_vert)
        # ).float()
        # right_close = (
        #     (right_delta[:, :2].norm(dim=-1) < close_horiz)
        #     & (right_delta[:, 2].abs() < close_vert)
        # ).float()
        # left_alignment, right_alignment = self._palm_down_alignments()
        # alignment_threshold = float(
        #     self.config.get("grasp_contact_alignment_threshold", 0.0)
        # )
        # left_orientation = left_alignment.clamp(min=0.0, max=1.0)
        # right_orientation = right_alignment.clamp(min=0.0, max=1.0)
        # left_orientation *= (left_alignment > alignment_threshold).float()
        # right_orientation *= (right_alignment > alignment_threshold).float()
        #
        # left_twist, right_twist = self._palm_twist_errors()
        # twist_gate_std = float(self.config.get("grasp_contact_twist_std", 1.2))
        # left_twist_gate = torch.exp(-(left_twist ** 2) / (2.0 * twist_gate_std ** 2))
        # right_twist_gate = torch.exp(-(right_twist ** 2) / (2.0 * twist_gate_std ** 2))
        #
        # left_valid = left_reward * left_close * left_orientation * left_twist_gate
        # right_valid = right_reward * right_close * right_orientation * right_twist_gate
        # both_valid = left_valid * right_valid
        # return 0.25 * (left_valid + right_valid) + 0.5 * both_valid

    @_effective_in_named_stages("GRASP", "TILT", "PUSH")  # 2026-09-29 overhand: + TILT
    def _reward_penalty_wrong_twist_contact(self):
        """Penalize hand-bin contact while twist is still badly wrong.

        Run 23 (2026-08-11): companion to the _reward_grasp_contact twist
        gate above. Removing the "free" positive reward for wedged-wrong
        contact (done there) isn't the same as telling the policy that
        backing off to reorient is GOOD — every proximity/contact term in
        this reward function only rewards getting closer or staying in
        contact, so there was no direct pressure to reverse a bad wedge, only
        reduced pressure to stay in it. This adds an explicit negative signal
        specifically for the failure mode confirmed on a live rollout: palm
        touching the bin while twist exceeds the hard grasp gate
        (grasp_finger_close_max_twist_rad), the same threshold
        _finger_close_ready already uses to decide fingers shouldn't close
        yet.
        """
        left_contact, right_contact, _, _, _, _ = self._grasp_transition_components()
        left_twist, right_twist = self._palm_twist_errors()
        max_twist = float(self.config.get("grasp_finger_close_max_twist_rad", 1.0472))
        left_wrong = left_contact & (left_twist > max_twist)
        right_wrong = right_contact & (right_twist > max_twist)
        return left_wrong.float() + right_wrong.float()

    @_effective_in_named_stages("REACH", "GRASP", "TILT", "PUSH")  # 2026-09-29 overhand: + TILT
    def _reward_palm_down_alignment(self):
        """Align each mirrored palm's contact face with the bin's downward normal."""
        left_alignment, right_alignment = self._palm_down_alignments()
        # Keep the objective signed so dorsal-hand approaches are penalized
        # instead of receiving a smaller positive reward. The minimum term makes
        # the worse hand the bilateral bottleneck without removing each hand's
        # independent rotation signal.
        left_score = left_alignment.clamp(-1.0, 1.0)
        right_score = right_alignment.clamp(-1.0, 1.0)
        return (
            0.25 * (left_score + right_score)
            + 0.5 * torch.minimum(left_score, right_score)
        ) * self._grasp_shape_stage_scale()

    @_effective_in_named_stages("REACH", "GRASP", "TILT", "PUSH")  # 2026-09-29 overhand: + TILT
    def _reward_palm_twist_alignment(self):
        """Reward matching the authored target's twist about the palm-down axis.

        `_reward_palm_down_alignment` only rewards the palm FACING down; it can't
        distinguish a correctly-twisted palm from one rotated 90 degrees about
        that same down-axis (same face direction, wrong finger-curl orientation
        for the rim). This reads the full angular error to the already-correct,
        cross-assigned, mirrored target rotation (see `_palm_twist_errors`) as a
        dense Gaussian, so twist gets a real, independent gradient rather than
        only gating grasp transition (`_finger_close_ready`) after the fact.

        Run 21 (2026-08-04, measured at checkpoint 4500): twist error sat at
        115-138deg (2.0-2.4 rad) — 3-4 std beyond the old std=0.6 rad (34deg),
        where the Gaussian reward is ~0.0003-0.004 and its gradient w.r.t. error
        is nearly flat. Same dead-zone failure mode already diagnosed for
        reach_pos_std/reach_pos_std_vert: a too-tight std gives no pull from far
        away. Widened std 0.6->1.2 rad (69deg) so there's real gradient at the
        measured operating point (reward ~0.25-0.34, slope ~0.22-0.34 there vs.
        ~0.002-0.02 before), accepting looser final precision in exchange for
        actually being able to climb toward it.
        """
        left_twist, right_twist = self._palm_twist_errors()
        twist_std = float(self.config.get("palm_twist_std", 1.2))
        left_score = self._tracking_reward_util(
            left_twist, std=twist_std, target=0.0, scale=1.0, offset=0.0
        )
        right_score = self._tracking_reward_util(
            right_twist, std=twist_std, target=0.0, scale=1.0, offset=0.0
        )
        return 0.5 * (left_score + right_score) * self._grasp_shape_stage_scale()

    @_effective_in_named_stages("REACH", "GRASP", "TILT", "PUSH")  # 2026-09-29 overhand: + TILT
    def _reward_palm_twist_progress(self):
        """Potential-based reward for REDUCING twist error step-over-step.

        Run 24 (2026-08-11): even at std=1.2, `_reward_palm_twist_alignment`'s
        Gaussian has near-flat gradient at the measured 70-140deg operating
        point (e.g. d(reward)/d(twist) ~ -0.2 to -0.4 there vs. a max around
        -0.5 near the peak) — any fixed-width Gaussian's gradient vanishes as
        error grows, it's just a question of how far out. A literature check
        (dexterous pre-grasp manipulation reward shaping, e.g. arXiv:2307.16752
        eq 10/16 "hand rotation reward"/"object orient reward") uses a
        DIFFERENCE-of-error term instead: reward = (prev_error - cur_error),
        which has constant, non-vanishing gradient at every error magnitude
        including near 180deg, because it's linear in the change of error, not
        exponential in the absolute error. Added alongside (not replacing) the
        existing Gaussian, which still gives sharp pull once actually close.
        Masks out each episode's first step (comparing to a stale pre-reset
        value would inject a bogus one-step spike/penalty).
        """
        left_twist, right_twist = self._palm_twist_errors()
        max_step_error = float(self.config.get("palm_twist_progress_norm", math.pi))
        valid = (self.episode_length_buf > 1).float()
        left_progress = (self._prev_left_twist - left_twist) / max_step_error * valid
        right_progress = (self._prev_right_twist - right_twist) / max_step_error * valid
        self._prev_left_twist = left_twist.detach().clone()
        self._prev_right_twist = right_twist.detach().clone()
        return 0.5 * (left_progress + right_progress)

    @_effective_in_named_stages("TILT", "PUSH")  # 2026-09-29 overhand: + TILT
    def _reward_maintain_grasp_while_pushing(self):
        """Reward sustained bimanual bar proximity during the push.

        2026-08-29: bar-distance-based (see _hand_bar_distances) -- reward
        ramps 1.0 at dist=0 down to 0.0 at dist>=grasp_bar_valid_dist,
        inverted from the old count/required linear ramp (higher count was
        better; lower distance is better).
        """
        left_dist, right_dist = self._hand_bar_distances()
        bar_valid_dist = float(self.config.get("grasp_bar_valid_dist", 0.03))
        if bool(self.config.get("push_grip_loose_check", False)):
            # 2026-09-29 overhand (I3): loose version matching
            # _grip_maintained_mask -- 1.0 inside the grasp gate
            # (grasp_bar_valid_dist; the overhand target itself sits at
            # ~4.5cm), linear to 0 at push_grip_max_bar_dist (8cm), times a
            # closure ramp (p0->p1 axis) reaching 1 at push_grip_min_closure.
            max_d = float(self.config.get("push_grip_max_bar_dist", 0.08))
            min_c = float(self.config.get("push_grip_min_closure", 0.6))
            push_max_d = float(self.config.get("push_stage_grip_max_bar_dist", max_d))
            push_min_c = float(self.config.get("push_stage_grip_min_closure", min_c))
            if push_max_d != max_d or push_min_c != min_c:
                in_push = self.stage_buf == self.STAGE_PUSH
                max_d = torch.where(
                    in_push,
                    torch.full_like(left_dist, push_max_d),
                    torch.full_like(left_dist, max_d),
                )
                min_c = torch.where(
                    in_push,
                    torch.full_like(left_dist, push_min_c),
                    torch.full_like(left_dist, min_c),
                )
                span = (max_d - bar_valid_dist).clamp(min=1e-3)
            else:
                span = max(max_d - bar_valid_dist, 1e-3)
            lc, rc = self._finger_closure_fractions()
            left_reward = ((max_d - left_dist) / span).clamp(0.0, 1.0) * (
                lc / min_c
            ).clamp(0.0, 1.0)
            right_reward = ((max_d - right_dist) / span).clamp(0.0, 1.0) * (
                rc / min_c
            ).clamp(0.0, 1.0)
        else:
            left_reward = (1.0 - left_dist / bar_valid_dist).clamp(min=0.0, max=1.0)
            right_reward = (1.0 - right_dist / bar_valid_dist).clamp(min=0.0, max=1.0)
        return 0.25 * (left_reward + right_reward) + 0.5 * (
            left_reward * right_reward
        )

    @_effective_in_named_stages("PUSH")
    def _reward_push_bin_progress(self):
        """Blend normalized forward velocity with monotonic displacement.

        2026-08-29: dropped the `* self._grasp_pose_valid_mask().float()`
        gate this used to have. Found comparing against DoorMan's analogous
        OPEN-stage rewards (push_door_force/push_door_hinge/push_door_handle,
        all stage-gated only, never grasp-validity-gated). Multiplying by
        our strict 4cm-proximity validity mask risked zeroing this reward on
        exactly the steps where real pushing force deflects the grip
        slightly out of that tight window -- a plausible compounding
        contributor (alongside the missing force-reward gap, see
        _reward_push_bin_force) to rew_push_bin_progress plateauing near
        zero / rew_complete staying at exactly 0 across every run so far.
        Grip-maintenance incentive is NOT lost -- _reward_maintain_grasp_
        while_pushing is a separate, dedicated term for that.
        """
        bin_vel = self.simulator.get_task_root_state("bin")[:, 7:10].clone()  # world lin vel
        axis = self.push_axis_world  # unit-ish (1,0,0) in v1
        axis_unit = F.normalize(axis, dim=-1)
        vel_along = (bin_vel * axis_unit).sum(dim=-1)          # m/s along push axis
        if bool(self.config.get("push_progress_use_bar_min", False)):
            # 2026-10-03 push2: tipping the bin back around the handle swings
            # the bottom root forward (+0.5 m/s) while the handle moves back;
            # require both root and handle bar to move forward.
            vel_along = torch.minimum(vel_along, self._bar_fwd_vel())
        target_vel = float(self.config.get("push_target_vel", 0.30))
        vel_reward = (vel_along / target_vel).clamp(min=0.0, max=1.0)

        progress = self._bin_push_progress()                    # meters along axis
        target = self._target_bin_push_distance_m()
        pos_reward = progress.clamp(min=0.0, max=target) / target
        if bool(self.config.get("push_progress_require_positive_vel", False)):
            # 2026-10-04 push6: only pay the static position fraction while the
            # bin/bar is actively moving forward (vel_along > 0); a parked bin
            # at e.g. +0.05 m earns 0.0 instead of farming pos_reward for 8+ s.
            v_gate = max(float(self.config.get("push_progress_vel_gate_mps", 0.05)), 1e-3)
            pos_reward = pos_reward * (vel_along / v_gate).clamp(0.0, 1.0)
        velocity_weight = float(self.config.get("push_velocity_weight", 0.50))
        reward = velocity_weight * vel_reward + (1.0 - velocity_weight) * pos_reward
        # 2026-09-29 overhand (B4/I4): only pay progress with the bin tilted
        # back inside the band (a forward topple also moves the bin root).
        if bool(self.config.get("push_tilt_band_gate", False)):
            reward = reward * self._tilt_band_weight()
        return reward
        # --- Original grasp-gated version (superseded 2026-08-29) ---
        # return reward * self._grasp_pose_valid_mask().float()

    @_effective_in_named_stages("TILT", "PUSH")
    def _reward_step_forward_while_pushing(self):
        """Reward pelvis velocity along pushAxisWorld (walking WITH the bin).
        Mirror of drawer step_backward_while_pulling with the sign flipped.

        2026-08-29: dropped the grasp-validity gate, same reasoning as
        _reward_push_bin_progress above -- see that docstring.
        """
        root_vel = self.simulator.robot_root_states[:, 7:10].clone()
        root_vel[:, 2] = 0.0
        axis_unit = F.normalize(self.push_axis_world, dim=-1)
        fwd_vel = (root_vel * axis_unit).sum(dim=-1)
        target_v = float(self.config.get("step_forward_target_vel", 0.20))
        std_v = float(self.config.get("step_forward_vel_std", 0.15))
        reward = self._tracking_reward_util(
            fwd_vel, std=std_v, target=target_v, scale=1.0, offset=0.0
        )
        if bool(self.config.get("step_forward_zero_at_rest", False)):
            # 2026-10-04 push6: subtract the zero-velocity Gaussian baseline
            # exp(-(target/std)^2) (~0.169) so standing still (fwd_vel <= 0)
            # earns 0.0 instead of +0.169.
            b0 = math.exp(-((target_v / max(std_v, 1e-3)) ** 2))
            reward = ((reward - b0) / max(1.0 - b0, 1e-4)).clamp(0.0, 1.0)
        # 2026-09-29 overhand (B4): walk only once the bin is tilted.
        if bool(self.config.get("push_tilt_band_gate", False)):
            reward = reward * self._tilt_band_weight()
        if self._tilt_enabled:
            in_push = (self.stage_buf == self.STAGE_PUSH).float()
            if bool(self.config.get("tilt_walk_ramp_in_band", False)):
                # 2026-10-03 push2: smooth walk ramp in TILT once the bin is in
                # the tilt band and settling, bridging into PUSH.
                in_tilt = (self.stage_buf == self.STAGE_TILT).float()
                ramp_w = float(self.config.get("tilt_walk_ramp_scale", 0.5)) * self._tilt_steady_weight(
                    float(self.config.get("tilt_walk_ramp_rate_std_dps", 25.0))
                )
                reward = reward * (in_push + in_tilt * ramp_w)
            else:
                reward = reward * in_push
        stall_pen = float(self.config.get("step_forward_stall_penalty", 0.0))
        if stall_pen > 0.0:
            # 2026-10-04 push6: in STAGE_PUSH after the brief settle grace
            # window (push_stall_grace_steps, default 25 = 0.5 s), explicitly
            # penalize non-positive / sub-threshold forward velocity of the
            # coupled robot+bin system (min(pelvis_fwd_vel, bar_fwd_vel) <
            # step_forward_min_vel).
            grace = float(self.config.get("push_stall_grace_steps", 25.0))
            ramp = max(float(self.config.get("push_stall_ramp_steps", 15.0)), 1e-3)
            alpha = ((self.time_in_stage_buf.float() - grace) / ramp).clamp(0.0, 1.0)
            v_req = float(self.config.get("step_forward_min_vel", 0.05))
            v_eff = torch.minimum(fwd_vel, self._bar_fwd_vel())
            deficit = ((v_req - v_eff) / 0.10).clamp(0.0, 1.5)
            in_push_mask = (self.stage_buf == self.STAGE_PUSH).float()
            reward = reward - stall_pen * deficit * alpha * in_push_mask
        return reward
        # --- Original grasp-gated version (superseded 2026-08-29) ---
        # return reward * self._grasp_pose_valid_mask().float()

    @_effective_in_named_stages("PUSH")
    def _reward_push_bin_force(self):
        """Reward raw hand-to-bin contact force along the push direction,
        ungated by grasp validity or bin motion (stage mask only).

        2026-08-29: NEW, addresses a gap found comparing against DoorMan's
        `_reward_push_door_force` (scale 0.3, stage-gated only). Our only
        other PUSH-stage progress reward (_reward_push_bin_progress) is a
        pure OUTCOME measure -- it pays nothing until the bin is already
        moving/displaced, so with a ~15kg bin and real ground friction,
        there's zero reward gradient until PPO exploration noise happens to
        produce enough coordinated bimanual force to overcome static
        friction in the first place. A chicken-and-egg stall, plausibly
        explaining why rew_push_bin_progress plateaus near zero and
        rew_complete stays at exactly 0 across every run so far. This term
        gives a direct, always-available gradient toward "push harder in
        the right direction," independent of whether the bin has actually
        started moving yet.

        Sums signed per-link contact force (object_to_hand_contact_forces,
        whole-body sensor, all hand links both hands combined -- same
        tensor _reward_grasp_contact reads) projected onto push_axis_world,
        clamped to [0, push_force_reward_max] so backward/off-axis force
        earns nothing (not penalized either -- this is a pure bonus, not a
        replacement for the direction/lateral shaping already done by
        _reward_grasp_contact).
        """
        forces = self.simulator.object_to_hand_contact_forces[:, 0, :, :]
        axis_unit = F.normalize(self.push_axis_world, dim=-1)
        force_along = (forces * axis_unit[:, None, :]).sum(dim=-1)  # (num_envs, num_hand_links)
        total_force = force_along.sum(dim=-1)  # (num_envs,) -- both hands combined
        max_force = float(self.config.get("push_force_reward_max", 20.0))
        return total_force.clamp(min=0.0, max=max_force)

    @_effective_in_named_stages("PUSH")
    def _reward_push_force_nomoment(self):
        """2026-09-29 overhand (B4): replaces push_bin_force. Hand force on the
        bin projected on the handle -> wheel-contact line (no tipping moment,
        see _push_force_nomoment_raw), capped at push_force_nomoment_max
        (60 N), times the tilt-band weight w_tilt (a flat bin tips forward at
        ~44 N instead of rolling)."""
        return self._push_force_nomoment_raw() * self._tilt_band_weight()

    def _tilt_potential(self):
        """Gaussian tilt potential shared by bin_tilt_back / bin_tilt_progress."""
        t = float(self.config.get("bin_tilt_target_deg", 15.0))
        std = float(self.config.get("bin_tilt_back_std_deg", 10.0))
        return torch.exp(-torch.square((self._bin_tilt_back_deg() - t) / std))

    @_effective_in_named_stages("TILT")
    def _reward_bin_tilt_progress(self):
        """2026-09-29 overhand: potential-based tilt shaping in TILT: step
        change of the bin_tilt_back Gaussian, divided by
        bin_tilt_progress_norm (0.05 -> a typical 15deg-in-1.5s tilt earns ~0.3
        per step before the reward scale). First step of every episode masked
        (stale previous value), same pattern as _reward_palm_twist_progress.
        The previous potential is updated in every stage so the first TILT
        step already sees a meaningful delta."""
        pot = self._tilt_potential()
        norm = float(self.config.get("bin_tilt_progress_norm", 0.05))
        valid = (self.episode_length_buf > 1).float()
        progress = (pot - self._prev_tilt_potential) / norm * valid
        self._prev_tilt_potential = pot.detach().clone()
        clip_deg = self.config.get("bin_tilt_progress_clip_deg", None)
        if clip_deg is not None:
            # 2026-10-03 push2: once inside the tilt band (>= clip_deg, e.g. 14),
            # stop paying positive progress for accelerating further back while
            # keeping the negative penalty if the bin overshoots past 15 deg.
            progress = torch.where(
                self._bin_tilt_back_deg() >= float(clip_deg),
                progress.clamp(max=0.0),
                progress,
            )
        return progress

    # 2026-10-01 tilt2: force-based TILT shaping. tilt1 grasped (86 % reach
    # TILT, grip kept 82-92 %) but the bin never rotated (tilt_back_max <=
    # 0.09 deg): bin_tilt_back / bin_tilt_progress are outcome-only and the
    # wheeled bin needs a finite moment about its wheel-ground contact line
    # before it rotates at all, so there was no gradient below that moment.
    # +1: object_to_hand_contact_forces reads as the force ON THE BIN from each
    # hand link (tilt_feasibility A2, canned grasp: hands resting on the bar ->
    # F_z < 0, hands pulled down -> more negative, pulled back-and-up -> F_z
    # >= 0; hands pulled toward the robot -> F . push_axis < 0).
    BIN_TILT_MOMENT_FORCE_SIGN = 1.0
    # Default bin_tilt_moment_ref_Nm = measured tilt onset (tilt_feasibility
    # A1, external force at the bar midpoint on the 17 kg bin): 29.9-33.4 Nm
    # for horizontal / 30 deg-down pulls (the bin also slides/rolls toward the
    # puller), 28.4 Nm at 60 deg, 22.8 Nm straight down; static analytic
    # m*g*(COM 12.1 cm ahead of the wheel contact line) = 20.2 Nm.
    BIN_TILT_MOMENT_REF_NM = 30.0

    def _bin_wheel_contact_point(self):
        """(N,3) wheel-ground contact point = mean wheel-body centre - wheel
        radius * z (same construction as _push_force_nomoment_raw, sharing its
        lazy wheel-body lookup), or None when the wheel bodies are missing."""
        if not hasattr(self, "_bin_wheel_body_idx"):
            self._bin_wheel_body_idx = None
            try:
                names = list(self.simulator.scene.articulations["bin"].body_names)
                idx = [names.index(n) for n in ("wheel_left", "wheel_right")]
                self._bin_wheel_body_idx = torch.tensor(idx, device=self.device)
            except Exception as e:  # pragma: no cover - asset-dependent
                logger.warning(f"bin wheel bodies not found ({e})")
        if self._bin_wheel_body_idx is None:
            return None
        wheel_pos = self.simulator.scene.articulations["bin"].data.body_pos_w[
            :, self._bin_wheel_body_idx, :
        ].mean(dim=1)
        contact = wheel_pos.clone()
        contact[:, 2] = contact[:, 2] - self.wheel_radius
        return contact

    def _bin_tilt_moment_terms(self):
        """2026-10-01 tilt2: (M, |F|), each (N,).

        M (Nm) = hand-on-bin moment about the wheel-ground contact line,
        + = tips the top back toward the robot (same sign as
        _bin_tilt_back_deg):  M = sum_hands ((p_hand - c) x F_hand) . k,
        c = _bin_wheel_contact_point(), k = a x z, a = horizontal push axis
        (away from the robot). F_hand = BIN_TILT_MOMENT_FORCE_SIGN * sum of the
        hand's links in object_to_hand_contact_forces (sign verified offline
        with tilt_feasibility A1/A2: hands pulled toward the robot -> M > 0
        and the bin tilts back). Contact point p_hand
        (bin_tilt_moment_point): "bar" (default) = that hand's bar target
        point (_left/_right_bar_raw_pos); robust -- the distal-link origins
        sit 3-5 cm behind/above the bar while the tips are in the slot, and
        the force is transmitted through the bar either way. "link" = each
        link's origin (per-link force).
        |F| = total hand-on-bin force magnitude (sum over links).

        2026-10-01 tilt3: force_matrix_w (object_to_hand_contact_forces) is
        the NORMAL contact force only. With bin_tilt_moment_friction (default
        False) the friction part of the same contacts (_hand_friction_forces;
        needs task.target_obj_contact_track_friction) is added per link before
        the moment / |F|. Evidence: tilt3 offline test, tilt2 iter-2300
        policy on a 6 kg bin (A1m5-scaled onset ~10-14 Nm): normal-only M >
        10 Nm on 38 % of TILT steps, runs of up to 28 straight steps, and the
        bin never moved (max +0.03 deg off rest) -- a squeezing claw reads as
        a large + M while friction cancels it."""
        self._init_hand_link_indices()
        forces = self.BIN_TILT_MOMENT_FORCE_SIGN * self.simulator.object_to_hand_contact_forces[
            :, 0, :, :
        ]
        if bool(self.config.get("bin_tilt_moment_friction", False)):
            fr = self._hand_friction_forces()
            if fr is not None:
                forces = forces + fr
        fmag = forces.norm(dim=-1).sum(dim=-1)
        return self._bin_hand_moment(forces), fmag

    def _hand_friction_forces(self):
        """2026-10-01 tilt3: (N, L, 3) friction force ON THE BIN from each
        hand link, same link order as object_to_hand_contact_forces:
        bin_tilt_moment_friction_sign (default +1, i.e. the same convention
        as force_matrix_w; checked offline against the bin's static moment
        limit) * simulator.object_to_hand_friction_forces. None (warned once)
        when task.target_obj_contact_track_friction is off."""
        fr = getattr(self.simulator, "object_to_hand_friction_forces", None)
        if fr is None:
            if not getattr(self, "_warned_no_hand_friction", False):
                logger.warning(
                    "hand friction forces requested but task.target_obj_contact_track_friction "
                    "is off -- using normal contact forces only"
                )
                self._warned_no_hand_friction = True
            return None
        return float(self.config.get("bin_tilt_moment_friction_sign", 1.0)) * fr[:, 0, :, :]

    def _bin_hand_moment(self, forces):
        """(N,) moment (Nm, + tips back) about the wheel contact line of
        per-hand-link forces ON THE BIN, forces (N, L, 3) in the
        object_to_hand_contact_forces link order; contact point per
        bin_tilt_moment_point (see _bin_tilt_moment_terms). Zeros when the
        wheel bodies are missing."""
        self._init_hand_link_indices()
        c = self._bin_wheel_contact_point()
        if c is None:
            return torch.zeros(forces.shape[0], device=forces.device, dtype=forces.dtype)
        axis = F.normalize(self.push_axis_world[:, :2].float(), dim=-1)
        a = torch.cat([axis, torch.zeros_like(axis[:, :1])], dim=-1)
        z = torch.zeros_like(a)
        z[:, 2] = 1.0
        k = torch.cross(a, z, dim=-1)
        mode = str(self.config.get("bin_tilt_moment_point", "bar"))
        m = torch.zeros(forces.shape[0], device=forces.device, dtype=forces.dtype)
        for side, bar in (
            ("left", self._left_bar_raw_pos()),
            ("right", self._right_bar_raw_pos()),
        ):
            f_h = forces[:, self._hand_link_contact_idx[side], :]  # (N,K,3)
            if mode == "link":
                p = self.simulator._rigid_body_pos[:, self._hand_link_sim_idx[side], :]
                r = p - c[:, None, :]
                m = m + (torch.cross(r, f_h, dim=-1) * k[:, None, :]).sum(dim=-1).sum(dim=-1)
            else:
                m = m + (torch.cross(bar - c, f_h.sum(dim=1), dim=-1) * k).sum(dim=-1)
        return m

    def _wrist_bin_contact_terms(self):
        """2026-10-01 tilt3 diagnostic: (|F| N, M Nm) of the wrist links'
        net contact force (robot-wide sensor, NORMAL only, unfiltered -- in
        TILT the wrists only touch the bin), M = moment of the force ON THE BIN
        (= -force on the link) about the wheel contact line at the link
        origins, + tips back. Wrists are in neither the hand filter list nor
        _body_bin_contact_force."""
        if not hasattr(self, "_wrist_contact_idx"):
            names = list(self.simulator.body_names)
            self._wrist_contact_idx = torch.tensor(
                [i for i, n in enumerate(names) if "wrist" in n],
                device=self.device, dtype=torch.long,
            )
        f_link = self.simulator.contact_forces[:, self._wrist_contact_idx, :]
        fmag = f_link.norm(dim=-1).sum(-1)
        c = self._bin_wheel_contact_point()
        if c is None:
            return fmag, torch.zeros_like(fmag)
        axis = F.normalize(self.push_axis_world[:, :2].float(), dim=-1)
        a = torch.cat([axis, torch.zeros_like(axis[:, :1])], dim=-1)
        z = torch.zeros_like(a)
        z[:, 2] = 1.0
        k = torch.cross(a, z, dim=-1)
        r = self.simulator._rigid_body_pos[:, self._wrist_contact_idx, :] - c[:, None, :]
        m = (torch.cross(r, -f_link, dim=-1) * k[:, None, :]).sum(dim=-1).sum(dim=-1)
        return fmag, m

    @_effective_in_named_stages("TILT", "PUSH")
    def _reward_bin_tilt_moment(self):
        """2026-10-01 tilt2: clamp(M / bin_tilt_moment_ref_Nm, 0, 1) * grip,
        M from _bin_tilt_moment_terms (hand moment about the wheel contact
        line, + = tips back), ref = measured tilt-onset moment (A1), grip =
        _grip_maintained_mask (the grip_maintained_frac_tilt mask) so shoving
        the lid / pulling without holding does not pay. TILT only (unless
        bin_tilt_moment_brake_in_push is enabled). Gives the
        gradient toward "pull back harder" that the outcome-only tilt rewards
        lack below the rotation threshold. Scale key `bin_tilt_moment`
        (absent = off).

        2026-10-02 grasp_tilt3: the lower clamp is `bin_tilt_moment_min_ratio`
        (default 0.0 = the tilt2/tilt3 behaviour above)."""
        m, _ = self._bin_tilt_moment_terms()
        ref = float(self.config.get("bin_tilt_moment_ref_Nm", self.BIN_TILT_MOMENT_REF_NM))
        # 2026-10-02 grasp_tilt3: optional SIGNED moment. tilt3 (friction-
        # corrected M) kept a 100 % grip at ~82 N hand force but M ~ -1.8 Nm,
        # i.e. the hands pushed the bin toward tipping FORWARD; with the lower
        # clamp at 0 a negative M cost nothing and gave no gradient toward
        # pulling back. bin_tilt_moment_min_ratio < 0 (e.g. -1) makes the term
        # linear through zero down to that ratio; values > 0 are treated as 0.
        lo = min(float(self.config.get("bin_tilt_moment_min_ratio", 0.0)), 0.0)
        grip = self._grip_maintained_mask().float()
        r = (m / max(ref, 1e-3)).clamp(lo, 1.0) * grip
        # 2026-10-01 tilt2 topple guard (tilt_feasibility A7: a steep push tips
        # the bin past its ~23 deg balance point and it falls onto the robot):
        # with bin_tilt_moment_off_deg set, the reward fades linearly to 0 over
        # the last bin_tilt_moment_off_ramp_deg before that tilt. Holding 11-15
        # deg needs only ~7-10 Nm, which bin_tilt_back pays for. Unset = old
        # behaviour (no fade).
        theta = self._bin_tilt_back_deg()
        off = self.config.get("bin_tilt_moment_off_deg", None)
        if off is not None:
            ramp = max(float(self.config.get("bin_tilt_moment_off_ramp_deg", 4.0)), 1e-3)
            r = r * ((float(off) - theta) / ramp).clamp(0.0, 1.0)
        stall_steps = self.config.get("bin_tilt_moment_stall_decay_steps", None)
        if stall_steps is not None:
            # 2026-10-03 push4/push5: pulling below the gate band (theta <
            # bin_tilt_moment_stall_min_deg) only pays during the initial pull
            # initiation window (stall_steps). If the bin is still below
            # min_deg, linearly decay positive moment reward to 0 over
            # stall_span_steps so the policy cannot farm moment by holding below
            # the gate band. Negative M keeps its penalty unchanged.
            span = max(float(self.config.get("bin_tilt_moment_stall_span_steps", 35.0)), 1e-3)
            min_deg = float(self.config.get("bin_tilt_moment_stall_min_deg", 2.0))
            decay = (
                1.0 - (self.time_in_stage_buf.float() - float(stall_steps)) / span
            ).clamp(0.0, 1.0)
            stalled = theta < min_deg
            r = torch.where(stalled & (r > 0.0), r * decay, r)
        if bool(self.config.get("bin_tilt_moment_brake_in_push", False)):
            # 2026-10-03 push5: in delta-action space, the arm targets are wound
            # up pulling backward (~130 N, +10..15 Nm) at TILT->PUSH handover.
            # Penalize excess backward pull M > m_allow once tilted (>= 11 deg)
            # and rotating backward (rate > 2 deg/s) or overshooting (> 14 deg)
            # so the policy unwinds the backward arm yank on entering the band.
            rate = self._bin_tilt_rate_dps()
            brake_deg = float(self.config.get("bin_tilt_moment_brake_start_deg", 11.0))
            w_ang = ((theta - brake_deg) / 3.0).clamp(0.0, 1.0)
            w_rate = (
                ((rate - 2.0) / 13.0).clamp(0.0, 1.0)
                * ((theta - 8.0) / 3.0).clamp(0.0, 1.0)
            )
            m_steady = float(self.config.get("bin_tilt_moment_steady_hold_Nm", 7.0))
            m_allow = (
                m_steady
                * (1.0 - w_rate)
                * ((18.0 - theta) / 4.5).clamp(0.0, 1.0)
            )
            over_pull = (
                ((m - m_allow) / 15.0).clamp(0.0, 1.0)
                * torch.maximum(w_ang, w_rate)
                * grip
            )
            r = torch.where(
                self.stage_buf == self.STAGE_PUSH, -over_pull, r - over_pull
            )
        else:
            r = torch.where(
                self.stage_buf == self.STAGE_PUSH, torch.zeros_like(r), r
            )
        return r

    def _body_bin_contact_force(self):
        """2026-10-01 tilt2: summed net contact force (N) on robot links other
        than hands / wrists / feet (name filter body_bin_contact_exclude). From
        the robot-wide net contact sensor (simulator.contact_forces), so it is
        not filtered to the bin, but in TILT/PUSH these links only touch the bin
        (tilt_feasibility A6/A8: a pure hold logs ~0 N; a chest brace on the
        handle/lid 150-290 N on torso_link)."""
        if not hasattr(self, "_body_contact_idx"):
            excl = list(self.config.get(
                "body_bin_contact_exclude", ["hand", "wrist", "ankle", "foot", "toe"]))
            names = list(self.simulator.body_names)
            self._body_contact_idx = torch.tensor(
                [i for i, n in enumerate(names) if not any(x in n for x in excl)],
                device=self.device, dtype=torch.long,
            )
        return self.simulator.contact_forces[:, self._body_contact_idx, :].norm(dim=-1).sum(-1)

    @_effective_in_named_stages("TILT", "PUSH")
    def _reward_penalty_body_bin_contact(self):
        """2026-10-01 tilt2: discourage tipping / pushing the bin with the
        body (chest brace on the handle). Positive magnitude
        clamp((F - deadband) / scale, 0, max) with F = _body_bin_contact_force
        (deadband 10 N, scale 100 N, max 3), times a per-stage multiplier:
        body_bin_contact_tilt_scale in TILT (small: a brace is tolerated while
        tipping if it is the only way) and body_bin_contact_push_scale in PUSH
        (ramped up: the bin must be pushed with the hands, in front of the
        torso). Use a NEGATIVE reward scale `penalty_body_bin_contact` (absent
        = off)."""
        f = self._body_bin_contact_force()
        dead = float(self.config.get("body_bin_contact_deadband_N", 10.0))
        sc = max(float(self.config.get("body_bin_contact_scale_N", 100.0)), 1e-3)
        mx = float(self.config.get("body_bin_contact_max", 3.0))
        mag = ((f - dead) / sc).clamp(0.0, mx)
        mult = torch.where(
            self._named_stage_mask(("PUSH",)),
            torch.full_like(mag, float(self.config.get("body_bin_contact_push_scale", 1.0))),
            torch.full_like(mag, float(self.config.get("body_bin_contact_tilt_scale", 1.0))),
        )
        return mag * mult

    def _dof_idx(self, names):
        return torch.tensor(
            [self.simulator.dof_names.index(n) for n in names],
            device=self.device, dtype=torch.long,
        )

    @StagedTaskBase.effective_in_stage([STAGE_REACH, STAGE_GRASP])
    def _reward_torso_lean(self):
        """2026-09-28 handlefix5b: lean forward (waist pitch ~0.45 rad) so the
        LEFT arm can reach the hook (IK: unreachable upright, exact at ~30 deg)."""
        if not hasattr(self, "_waist_pitch_idx"):
            self._waist_pitch_idx = self.simulator.dof_names.index("waist_pitch_joint")
        q = self.simulator.dof_pos[:, self._waist_pitch_idx]
        tgt = float(self.config.get("torso_lean_target", 0.45))
        std = float(self.config.get("torso_lean_std", 0.2))
        return torch.exp(-torch.square((q - tgt) / std))

    @StagedTaskBase.effective_in_stage([STAGE_REACH, STAGE_GRASP])
    def _reward_left_arm_posture(self):
        """2026-09-28 handlefix5b: pull the LEFT arm toward the IK 'good branch'
        for the hook (shoulder_yaw +, wrist_roll ~0) and out of the twisted
        shoulder_yaw -150 / wrist_roll +112 pose it gets stuck in."""
        if not hasattr(self, "_left_arm_idx"):
            self._left_arm_idx = self._dof_idx([
                f"left_{j}_joint" for j in (
                    "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
                    "wrist_roll", "wrist_pitch", "wrist_yaw")
            ])
            self._left_arm_ref = torch.tensor(
                self.config.get("left_arm_ref_pose",
                                [-1.967, -0.077, 1.761, 0.930, 0.0, 0.162, 1.194]),
                device=self.device, dtype=torch.float32,
            )
        std = float(self.config.get("left_arm_ref_std", 0.6))
        err = self.simulator.dof_pos[:, self._left_arm_idx] - self._left_arm_ref[None]
        return torch.exp(-torch.square(err / std).mean(dim=-1))

    @StagedTaskBase.effective_in_stage([STAGE_REACH, STAGE_GRASP])
    def _reward_penalty_left_wrist_roll(self):
        """2026-09-28 handlefix5b: penalize the twisted left-forearm branch."""
        if not hasattr(self, "_left_wrist_roll_idx"):
            self._left_wrist_roll_idx = self.simulator.dof_names.index("left_wrist_roll_joint")
        lim = float(self.config.get("left_wrist_roll_soft_limit", 1.3))
        q = self.simulator.dof_pos[:, self._left_wrist_roll_idx]
        return (q.abs() - lim).clamp(min=0.0)

    def _arm_branch_q(self):
        if not hasattr(self, "_arm_branch_idx"):
            self._arm_branch_idx = self._dof_idx([
                "left_shoulder_yaw_joint", "left_wrist_roll_joint",
                "right_shoulder_yaw_joint", "right_wrist_roll_joint",
            ])
        return self.simulator.dof_pos[:, self._arm_branch_idx]

    def _overhand_arm_wrist_q(self):
        if not hasattr(self, "_ovh_arm_wrist_idx"):
            self._ovh_arm_wrist_idx = self._dof_idx([
                "left_shoulder_yaw_joint",
                "left_wrist_roll_joint",
                "left_wrist_pitch_joint",
                "left_wrist_yaw_joint",
                "right_shoulder_yaw_joint",
                "right_wrist_roll_joint",
                "right_wrist_pitch_joint",
                "right_wrist_yaw_joint",
            ])
        return self.simulator.dof_pos[:, self._ovh_arm_wrist_idx]

    @_effective_in_named_stages("APPROACH", "REACH", "GRASP", "TILT", "PUSH")
    def _reward_penalty_arm_branch_guard(self):
        """2026-09-28 handlefix5e: keep each arm on the IK branch that reaches the
        hook (left shoulder_yaw >= 0, right <= 0, |wrist_roll| small). One-sided:
        0 at the resting pose, so it does not fight the APPROACH arms-down term.

        2026-10-04 full2 (arm_branch_overhand_mode, default False): in overhand
        claw mode, the natural branch has left_shoulder_yaw in [-1.15, +0.10],
        right_shoulder_yaw in [-0.80, +0.35], left_wrist_roll in [+0.25, +1.65],
        right_wrist_roll in [-1.65, -0.25], |wrist_pitch| <= 0.95, and
        |wrist_yaw| <= 0.85 (0.0 penalty at resting_dof_pos, overhand_th10_a080,
        and reach_grasp2 1700, while penalizing chicken-wing shoulder_yaw and
        pinned +-92.5 deg wrist pitch/yaw joint stops across all 5 stages)."""
        if bool(self.config.get("arm_branch_overhand_mode", False)):
            q = self._overhand_arm_wrist_q()
            lsy, lwr, lwp, lwy, rsy, rwr, rwp, rwy = q.unbind(dim=-1)
            lsy_lo = float(self.config.get("arm_branch_left_syaw_min", -1.15))
            lsy_hi = float(self.config.get("arm_branch_left_syaw_max", 0.10))
            rsy_lo = float(self.config.get("arm_branch_right_syaw_min", -0.80))
            rsy_hi = float(self.config.get("arm_branch_right_syaw_max", 0.35))
            lwr_lo = float(self.config.get("arm_branch_left_wroll_min", 0.25))
            lwr_hi = float(self.config.get("arm_branch_left_wroll_max", 1.65))
            rwr_lo = float(self.config.get("arm_branch_right_wroll_min", -1.65))
            rwr_hi = float(self.config.get("arm_branch_right_wroll_max", -0.25))
            wp_lim = float(self.config.get("arm_branch_wpitch_limit", 0.95))
            wy_lim = float(self.config.get("arm_branch_wyaw_limit", 0.85))
            left = (
                (lsy_lo - lsy).clamp(min=0.0)
                + (lsy - lsy_hi).clamp(min=0.0)
                + (lwr_lo - lwr).clamp(min=0.0)
                + (lwr - lwr_hi).clamp(min=0.0)
                + (lwp.abs() - wp_lim).clamp(min=0.0)
                + (lwy.abs() - wy_lim).clamp(min=0.0)
            )
            right = (
                (rsy_lo - rsy).clamp(min=0.0)
                + (rsy - rsy_hi).clamp(min=0.0)
                + (rwr_lo - rwr).clamp(min=0.0)
                + (rwr - rwr_hi).clamp(min=0.0)
                + (rwp.abs() - wp_lim).clamp(min=0.0)
                + (rwy.abs() - wy_lim).clamp(min=0.0)
            )
            rw = float(self.config.get("arm_branch_right_weight", 1.0))
            out = left + rw * right
            if bool(self.config.get("arm_branch_skip_in_approach", True)):
                out = out * (self.stage_buf != self.STAGE_APPROACH).float()
            return out
        q = self._arm_branch_q()
        m = float(self.config.get("arm_branch_shoulder_yaw_margin", 0.0))
        wr = float(self.config.get("arm_branch_wrist_roll_limit", 0.5))
        rw = float(self.config.get("arm_branch_right_weight", 1.0))
        left = (-q[:, 0] - m).clamp(min=0.0) + (q[:, 1].abs() - wr).clamp(min=0.0)
        right = (q[:, 2] - m).clamp(min=0.0) + (q[:, 3].abs() - wr).clamp(min=0.0)
        pre_hold = (self.stage_buf <= self.STAGE_GRASP).float()
        return (left + rw * right) * pre_hold

    @StagedTaskBase.effective_in_stage([STAGE_REACH, STAGE_GRASP])
    def _reward_penalty_left_arm_posture_l1(self):
        """2026-09-28 handlefix5e: L1 pull of the 7 left-arm joints to the IK
        good-branch pose (constant gradient at any distance)."""
        if not hasattr(self, "_left_arm_idx"):
            self._left_arm_idx = self._dof_idx([
                f"left_{j}_joint" for j in (
                    "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
                    "wrist_roll", "wrist_pitch", "wrist_yaw")
            ])
            self._left_arm_ref = torch.tensor(
                self.config.get("left_arm_ref_pose",
                                [-1.967, -0.077, 1.761, 0.930, 0.0, 0.162, 1.194]),
                device=self.device, dtype=torch.float32,
            )
        err = self.simulator.dof_pos[:, self._left_arm_idx] - self._left_arm_ref[None]
        return err.abs().mean(dim=-1)

    @_effective_in_named_stages("TILT", "PUSH")
    def _reward_push_forward_command(self):
        """2026-09-28 handlefix5: HOMIE switches to the STAND model when
        |cmd[:3]| < 0.1 (homie_base obs deadband), and GRASP trains the policy
        to output 0 -- so in PUSH it never walks. Reward a forward command above
        the deadband (linear push_cmd_deadband -> push_cmd_target), weighted up
        when the bin is tipped back (force study: flat bin tips forward over its
        base instead of rolling; tipped back 10-20 deg it rolls with any force)."""
        vx = self.get_physical_homie_commands()[:, 0]
        db = float(self.config.get("push_cmd_deadband", 0.1))
        tgt = float(
            self.config.get(
                "push_cmd_target", self.config.get("push_cmd_target_vx", 0.3)
            )
        )
        pre_w = float(self.config.get("push_cmd_pre_deadband_weight", 0.0))
        if pre_w > 0.0 and db > 1e-4:
            # 2026-10-03 push4 / 2026-10-04 push6: vx = get_physical_homie_commands()[:, 0]
            # is 2.0 * _homie_commands[:, 0], and ppo_trainer_homie_api.py only
            # switches from homie_stand_model to homie_walk_model when
            # ||cmd[:3]|| > homie_switch_threshold = 0.50 (NOT 0.10).
            # A linear pre-ramp (0 -> pre_w over [0, db], then pre_w -> 1.0 over
            # [db, tgt]) gives immediate positive gradient from vx = 0.0 (and
            # from tilt_push5's +0.27) across the true 0.50 walk-switch threshold.
            pre_r = (vx / db).clamp(0.0, 1.0) * pre_w
            post_r = ((vx - db) / max(tgt - db, 1e-3)).clamp(0.0, 1.0) * (1.0 - pre_w)
            r = pre_r + post_r
        else:
            r = ((vx - db) / max(tgt - db, 1e-3)).clamp(0.0, 1.0)
        floor = float(self.config.get("push_cmd_tilt_floor", 0.25))
        tilt_w = (self._bin_tilt_back_deg() / 10.0).clamp(0.0, 1.0)
        if bool(self.config.get("push_tilt_band_gate", False)):
            # 2026-09-29 overhand (B4): band weight (0 flat, 1 in 10..22 deg,
            # 0 past 25 deg) instead of the one-sided ramp; with
            # push_cmd_tilt_floor 0 a flat bin earns nothing.
            tilt_w = self._tilt_band_weight()
        out = r * (floor + (1.0 - floor) * tilt_w)
        if self._tilt_enabled:
            in_push = (self.stage_buf == self.STAGE_PUSH).float()
            if bool(self.config.get("tilt_walk_ramp_in_band", False)):
                # 2026-10-03 push2: once the bin is tilted into the band and
                # settling in TILT, ramp in the forward walk-command reward.
                in_tilt = (self.stage_buf == self.STAGE_TILT).float()
                ramp_w = float(self.config.get("tilt_walk_ramp_scale", 0.5)) * self._tilt_steady_weight(
                    float(self.config.get("tilt_walk_ramp_rate_std_dps", 25.0))
                )
                out = out * (in_push + in_tilt * ramp_w)
            else:
                out = out * in_push
        sub_db_pen = float(self.config.get("push_cmd_sub_deadband_penalty", 0.0))
        if sub_db_pen > 0.0 and db > 1e-4:
            # 2026-10-04 push6: in STAGE_PUSH after the settle grace window,
            # penalize commanding vx < db (0.50 HOMIE stand->walk switch
            # threshold) so staying in HOMIE's stand model or commanding
            # backward vx <= 0 is actively penalized.
            grace = float(self.config.get("push_stall_grace_steps", 25.0))
            ramp = max(float(self.config.get("push_stall_ramp_steps", 15.0)), 1e-3)
            alpha = ((self.time_in_stage_buf.float() - grace) / ramp).clamp(0.0, 1.0)
            in_push_mask = (self.stage_buf == self.STAGE_PUSH).float() * alpha
            deficit = ((db - vx) / max(db, 1e-3)).clamp(0.0, 1.5)
            out = out - sub_db_pen * deficit * in_push_mask
        return out

    @StagedTaskBase.effective_in_stage([STAGE_APPROACH])
    def _reward_penalty_upper_body_non_finger_deviation_l1(self):
        """L1 deviation of non-finger upper-body DoFs from resting pose (arms down while walking)."""
        dev = torch.abs(
            self.simulator.dof_pos[:, self._upper_non_finger_dof_idx]
            - self.resting_dof_pos[:, self._upper_non_finger_dof_idx]
        ).sum(dim=-1)
        act_w = float(self.config.get("approach_arm_action_l1_weight", 0.0))
        if act_w > 0.0:
            arm_cols = self._resting_action_state()["arm_delta_cols"]
            dev = dev + act_w * self._last_delta_actions[:, arm_cols].abs().sum(dim=-1)
        return dev

    @StagedTaskBase.effective_in_stage([STAGE_APPROACH, STAGE_REACH, STAGE_GRASP])
    def _reward_penalty_face_bin(self):
        """Penalize the robot forward axis not pointing at the bin (XY plane).
        0 when facing the bin, up to 2 when facing away. Mirrors drawer penalty_face_handle."""
        root_pos = self.simulator.robot_root_states[:, :3].clone()
        root_rot = self.simulator.robot_root_states[:, 3:7].clone()
        forward_vec = torch.zeros(self.num_envs, 3, device=self.device)
        forward_vec[:, 0] = 1.0
        robot_forward = quat_apply(xyzw_to_wxyz(root_rot), forward_vec)
        robot_forward[:, 2] = 0.0
        robot_forward = F.normalize(robot_forward, dim=-1)

        bin_pos = self.simulator.get_task_root_state("bin")[:, :3].clone()
        to_bin = bin_pos - root_pos
        to_bin[:, 2] = 0.0
        to_bin = F.normalize(to_bin, dim=-1)
        dot = (robot_forward * to_bin).sum(dim=-1)
        return 1.0 - dot

    @_effective_in_named_stages("REACH", "GRASP", "TILT", "PUSH")  # 2026-09-29 overhand: + TILT; 2026-10-04 push6: + PUSH (gated)
    def _reward_penalty_not_standing_still(self):
        """Penalize nonzero HOMIE locomotion command while reaching / grasping."""
        cmd = self.get_physical_homie_commands()[:, :3]
        if bool(self.config.get("tilt_walk_ramp_in_band", False)) and self._tilt_enabled:
            # 2026-10-03 push2: relax the penalty on forward vx > 0 in TILT once
            # the bin is inside the tilt band and settling (lateral vy, vyaw,
            # and backward vx < 0 stay penalized).
            in_tilt = (self.stage_buf == self.STAGE_TILT).float()
            keep = (
                1.0
                - in_tilt
                * self._tilt_steady_weight(
                    float(self.config.get("tilt_walk_ramp_rate_std_dps", 25.0))
                )
            ).clamp(0.0, 1.0)
            vx = torch.where(cmd[:, 0] > 0.0, cmd[:, 0] * keep, cmd[:, 0])
            cmd = torch.stack([vx, cmd[:, 1], cmd[:, 2]], dim=-1)
        in_push = self.stage_buf == self.STAGE_PUSH
        if not bool(self.config.get("penalize_lateral_yaw_cmd_in_push", False)):
            if in_push.any():
                cmd = torch.where(in_push[:, None], torch.zeros_like(cmd), cmd)
        else:
            # 2026-10-04 push6: in PUSH, forward vx > 0 is free, while lateral
            # vy, yaw rate vyaw, and backward vx < 0 are penalized so the robot
            # walks straight forward with the bin without yaw drift.
            vx_push = cmd[:, 0].clamp(max=0.0)
            cmd_push = torch.stack([vx_push, cmd[:, 1], cmd[:, 2]], dim=-1)
            cmd = torch.where(in_push[:, None], cmd_push, cmd)
        return torch.norm(cmd, dim=1)

    @_effective_in_named_stages("TILT", "PUSH")  # 2026-09-29 overhand: + TILT
    def _reward_penalty_bin_wheel_kick(self):
        """Penalize foot contact with the bin's WHEEL bodies during PUSH.

        BUGFIX (Run 6): this term previously (a) had NO stage gating, so it was
        active during APPROACH/REACH and punished the robot for ANY contact with
        the bin — it learned to stop at the stop line and step sideways rather
        than close the last 0.3m to the handles; and (b) read the `/body` contact
        sensor (whole bin), not the wheels, despite its name. Now gated to PUSH
        only (kicking the wheels is only undesirable once we're driving the bin)
        and reads the dedicated wheel contact sensor.
        """
        # 2026-09-25: force_matrix_w = (N, wheels, filtered robot links, 3):
        # only robot-leg-on-wheel contact counts (floor contact excluded).
        # 2026-09-27: one sensor per wheel (see scene_creation_callback).
        total = torch.zeros(self.num_envs, device=self.device)
        for wheel in ("wheel_left", "wheel_right"):
            fm = self.simulator.scene.sensors[f"bin_{wheel}_contact_sensor"].data.force_matrix_w
            if fm is not None:
                total = total + fm.norm(dim=-1).sum(dim=(1, 2))
        return total

    # ------------------------------------------------------------------
    # 2026-09-25 handlefix3: finger WRAP around the bar + lid contact.
    # Video + wrap_probe on 5400/5600: index/middle tips go OVER the bar and
    # land on the lid (bin-frame dy -3..-6cm, dz +5cm vs bar centre); only
    # the thumb tip goes under. The 5cm grasp-point gate is satisfied by a
    # fist above the bar and the finger-closure reward pays for closing in
    # the air. Bin frame: +x along the bar, +y from lid toward the robot
    # (bar y 0.3425, lid panel y<0.312), +z up. Measured from each hand's
    # raw bar point (_left/_right_bar_raw_pos).
    # ------------------------------------------------------------------
    def _init_hand_link_indices(self):
        if hasattr(self, "_hand_link_sim_idx"):
            return
        # Same order as object_to_hand_contact_forces' filter list (see
        # _fingertip_forces); mapped to simulator body indices for positions.
        names = [
            n for n in self.simulator.robot_config.body_names
            if ("left_hand" in n or "right_hand" in n)
        ]
        sim_names = list(self.simulator.body_names)
        tips = ("index_1", "middle_1", "thumb_2")
        self._hand_link_contact_idx = {}
        self._hand_link_sim_idx = {}
        self._wrap_tip_sim_idx = {}
        for side in ("left", "right"):
            ci = [i for i, n in enumerate(names) if f"{side}_hand" in n]
            self._hand_link_contact_idx[side] = ci
            self._hand_link_sim_idx[side] = torch.tensor(
                [sim_names.index(names[i]) for i in ci],
                device=self.device, dtype=torch.long,
            )
            self._wrap_tip_sim_idx[side] = torch.tensor(
                [
                    sim_names.index(n) for n in names
                    if f"{side}_hand" in n and any(t in n for t in tips)
                ],
                device=self.device, dtype=torch.long,
            )

    def _bar_frame_deltas(self, link_idx, bar_world):
        """(N,K,3) offsets of robot links from a world bar point, in the bin frame."""
        q_inv = quat_inv(self.simulator.get_task_root_state("bin")[:, 3:7])  # wxyz
        d = self.simulator._rigid_body_pos[:, link_idx, :] - bar_world[:, None, :]
        n, k, _ = d.shape
        return quat_apply(
            q_inv[:, None, :].expand(n, k, 4).reshape(-1, 4), d.reshape(-1, 3)
        ).reshape(n, k, 3)

    def _finger_wrap_scores(self):
        """((wrap_l, prox_l), (wrap_r, prox_r)), each (N,3) for index_1/
        middle_1/thumb_2. angle = phi/pi, phi measured around the bar
        axis from the TOP (0 on top, 0.5 at the side, 1 underneath; side-
        agnostic). prox = 1 within grasp_wrap_rho0 of the bar axis, Gaussian
        falloff (grasp_wrap_rho_std) beyond. wrap = angle * prox."""
        self._init_hand_link_indices()
        if self._overhand_mode():
            # 2026-09-29 overhand: (score, prox) from the overhand claw terms,
            # fingers ordered (index, middle, thumb). prox = Gaussian on the
            # distal-phalanx surface distance (grasp_ovh_prox_std, wider than
            # the score's sigma) -- used by grasp_finger_near_gate_floor.
            prox_std = float(self.config.get("grasp_ovh_prox_std", 0.02))
            ovh = self._overhand_finger_terms()
            return tuple(
                (
                    ovh[s][0],
                    torch.exp(-torch.square(ovh[s][2].clamp(min=0.0) / prox_std)),
                )
                for s in ("left", "right")
            )
        r0 = float(self.config.get("grasp_wrap_rho0", 0.04))
        std = float(self.config.get("grasp_wrap_rho_std", 0.02))
        out = []
        for side, bar in (
            ("left", self._left_bar_raw_pos()),
            ("right", self._right_bar_raw_pos()),
        ):
            idx = self._wrap_tip_sim_idx[side]
            d = self._bar_frame_deltas(idx, bar)
            wrap, prox = self._wrap_terms(side, d, r0, std)
            if bool(self.config.get("grasp_wrap_use_tip", False)):
                # 2026-09-28 handlefix5: also score the REAL fingertip (link
                # origins are the PIP/IP joints, ~5.2cm short of the tip).
                tip_w = self.simulator._rigid_body_pos[:, idx, :] + quat_apply(
                    xyzw_to_wxyz(self.simulator._rigid_body_rot[:, idx, :].reshape(-1, 4)),
                    self._wrap_tip_offsets(side).expand(self.num_envs, -1, -1).reshape(-1, 3),
                ).reshape(self.num_envs, -1, 3)
                q_inv = quat_inv(self.simulator.get_task_root_state("bin")[:, 3:7])
                dt = tip_w - bar[:, None, :]
                k = dt.shape[1]
                dt = quat_apply(
                    q_inv[:, None, :].expand(-1, k, 4).reshape(-1, 4), dt.reshape(-1, 3)
                ).reshape(self.num_envs, k, 3)
                wrap_t, prox_t = self._wrap_terms(side, dt, r0, std)
                wrap, prox = 0.5 * (wrap + wrap_t), 0.5 * (prox + prox_t)
            out.append((wrap, prox))
        return out[0], out[1]

    def _wrap_is_thumb(self, side):
        if not hasattr(self, "_wrap_thumb_mask"):
            sim_names = list(self.simulator.body_names)
            self._wrap_thumb_mask = {
                s: torch.tensor(
                    ["thumb" in sim_names[i] for i in self._wrap_tip_sim_idx[s].tolist()],
                    device=self.device,
                )
                for s in ("left", "right")
            }
        return self._wrap_thumb_mask[side]

    def _wrap_tip_offsets(self, side):
        """(1,K,3) link-local fingertip offsets (far point of the distal
        collision hull, measured from g1_43dof_s3.usda by the hand probe)."""
        if not hasattr(self, "_wrap_tip_off"):
            finger = {"left": (0.0519, 0.0038, -0.0022), "right": (0.0519, -0.0038, 0.0022)}
            thumb = {"left": (-0.0038, -0.0519, 0.0022), "right": (-0.0038, 0.0519, -0.0022)}
            self._wrap_tip_off = {
                s: torch.tensor(
                    [thumb[s] if t else finger[s] for t in self._wrap_is_thumb(s).tolist()],
                    device=self.device, dtype=torch.float32,
                )[None]
                for s in ("left", "right")
            }
        return self._wrap_tip_off[side]

    def _wrap_terms(self, side, d, r0, std):
        """wrap = angle * prox from bin-frame bar offsets d (N,K,3). angle =
        phi/pi (0 on top, 1 underneath). grasp_wrap_thumb_over (handlefix5,
        front hook): the THUMB scores 1-angle (on top = good), the fingers
        keep angle (under = good) -- opposition, not 'everything under'."""
        dy, dz = d[..., 1], d[..., 2]
        rho = torch.sqrt(dy * dy + dz * dz + 1e-8)
        # 2026-09-27 handlefix4: linear in phi (0 top, 0.5 side, 1 under).
        # (1-cos phi)/2 had ~zero slope at the bar top, where the fingers sit.
        angle = torch.acos((dz / rho).clamp(-1.0, 1.0)) / math.pi
        if bool(self.config.get("grasp_wrap_thumb_over", False)):
            angle = torch.where(self._wrap_is_thumb(side)[None, :], 1.0 - angle, angle)
        prox = torch.exp(-torch.square((rho - r0).clamp(min=0.0) / std))
        return angle * prox, prox

    def _wrap_hand_scores(self):
        """Per-hand aggregated wrap score (N,), same blend as the reward."""
        (lw, _), (rw, _) = self._finger_wrap_scores()
        w = float(self.config.get("grasp_wrap_worst_weight", 0.5))
        agg = lambda s: (1.0 - w) * s.mean(dim=-1) + w * s.min(dim=-1).values  # noqa: E731
        return agg(lw), agg(rw)

    @_effective_in_named_stages("GRASP", "TILT", "PUSH")  # 2026-09-29 overhand: + TILT
    def _reward_grasp_finger_wrap(self):
        """2026-09-25: reward every fingertip wrapping AROUND the bar (see
        _finger_wrap_scores). Worst-finger blend (grasp_wrap_worst_weight) so
        the index/middle fingers must wrap too, not just the thumb."""
        (lw, _), (rw, _) = self._finger_wrap_scores()
        w = float(self.config.get("grasp_wrap_worst_weight", 0.5))

        def agg(s):
            return (1.0 - w) * s.mean(dim=-1) + w * s.min(dim=-1).values

        return 0.5 * (agg(lw) + agg(rw)) * self._grasp_shape_stage_scale()

    def _lid_contact_per_hand(self):
        """Per-hand sum over hand links of clamp(|F_bin|/Fmax, 1) for links
        that are on the LID side of the bar (bin dy < grasp_lid_dy_max) and
        above the bar centre (dz > grasp_lid_dz_min). Geometric region is
        needed because the lid and handle colliders are one rigid body (the
        contact filter can't separate them)."""
        self._init_hand_link_indices()
        forces = self.simulator.object_to_hand_contact_forces[:, 0, :, :].norm(dim=-1)
        fmax = float(self.config.get("grasp_lid_contact_force_max", 5.0))
        dy_max = float(self.config.get("grasp_lid_dy_max", -0.024))
        dz_min = float(self.config.get("grasp_lid_dz_min", 0.0))
        out = []
        for side, bar in (
            ("left", self._left_bar_raw_pos()),
            ("right", self._right_bar_raw_pos()),
        ):
            d = self._bar_frame_deltas(self._hand_link_sim_idx[side], bar)
            in_lid = ((d[..., 1] < dy_max) & (d[..., 2] > dz_min)).float()
            if bool(self.config.get("grasp_lid_exclude_thumb", False)):
                # 2026-09-28 handlefix5: hook thumb pad rests behind the bar.
                sim_names = list(self.simulator.body_names)
                non_thumb = torch.tensor(
                    ["thumb" not in sim_names[i] for i in self._hand_link_sim_idx[side].tolist()],
                    device=self.device, dtype=torch.float32,
                )
                in_lid = in_lid * non_thumb[None, :]
            f = (forces[:, self._hand_link_contact_idx[side]] / fmax).clamp(max=1.0)
            out.append((f * in_lid).sum(dim=-1))
        return out[0], out[1]

    @_effective_in_named_stages("REACH", "GRASP", "TILT", "PUSH")  # 2026-09-29 overhand: + TILT
    def _reward_penalty_lid_contact(self):
        """2026-09-25: penalize fingers/palm pressing on the lid past the bar
        (right outer finger seen hitting the lid top in 5400/5600 videos)."""
        left, right = self._lid_contact_per_hand()
        return 0.5 * (left + right)

    @_effective_in_named_stages("REACH", "GRASP")
    def _reward_penalty_palm_below_bar(self):
        """2026-09-30 reach_grasp1 fix: per hand, (bin-frame horizontal palm
        grasp point -> bar distance < 0.10 m) * clamp((grasp_palm_min_dz - dz)
        / 0.04, 0, 1), summed over hands (use a NEGATIVE scale). Ramps from
        0 at the min height to 1 at 4 cm below it; far-away hands (still
        approaching from below) are not penalised."""
        min_dz = self._palm_min_dz() or 0.0
        out = 0.0
        for d in self._palm_bar_deltas_bin():
            near = (d[:, :2].norm(dim=-1) < 0.10).float()
            out = out + near * ((min_dz - d[:, 2]) / 0.04).clamp(0.0, 1.0)
        return out

    def _bin_tilt_back_deg(self):
        """Signed bin tilt (deg) toward the robot (+ = top leans back toward the
        robot / handle down / front lifts -- the human tip-onto-wheels pose)."""
        q = self.simulator.get_task_root_state("bin")[:, 3:7]  # wxyz
        qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        up_xy = torch.stack(
            [2.0 * (qx * qz + qw * qy), 2.0 * (qy * qz - qw * qx)], dim=-1
        )
        axis_xy = F.normalize(self.push_axis_world[:, :2], dim=-1)
        return torch.rad2deg(
            torch.asin((-(up_xy * axis_xy).sum(dim=-1)).clamp(-1.0, 1.0))
        )

    def _bin_tilt_rate_dps(self):
        """2026-10-03 tilt7: d/dt of _bin_tilt_back_deg (deg/s, + = tipping
        further back). theta = asin(-up . a) with up the bin body's z axis and
        a the horizontal unit push axis; d up/dt = w x up (w = world angular
        velocity, root state [10:13]), so d theta/dt = -w . (up x a) /
        cos(theta). Small-angle check, a = +x: -w_y. Checkpoint stage-4
        snapshots (tilt5 5100 / push1): ~+31 deg/s at the handover."""
        st = self.simulator.get_task_root_state("bin")
        qw, qx, qy, qz = st[:, 3], st[:, 4], st[:, 5], st[:, 6]  # wxyz
        up = torch.stack(
            [
                2.0 * (qx * qz + qw * qy),
                2.0 * (qy * qz - qw * qx),
                1.0 - 2.0 * (qx * qx + qy * qy),
            ],
            dim=-1,
        )
        a = torch.zeros_like(up)
        a[:, :2] = F.normalize(self.push_axis_world[:, :2], dim=-1)
        s = (-(up * a).sum(dim=-1)).clamp(-1.0, 1.0)  # sin(theta)
        cos_t = (1.0 - s * s).clamp(min=1e-4).sqrt()
        w = st[:, 10:13]
        return torch.rad2deg(-(w * torch.cross(up, a, dim=-1)).sum(dim=-1) / cos_t)

    def _tilt_steady_weight(self, std_dps=None):
        """2026-10-03 push2: w_band(theta) * grip_maintained * exp(-(rate/std)^2).
        0 for a flat or toppling bin or when the grip is lost; 1 inside [10, 22]
        deg when the bin is held still (d theta/dt ~ 0 deg/s)."""
        if std_dps is None:
            std_dps = float(self.config.get("bin_tilt_steady_std_dps", 15.0))
        rate = self._bin_tilt_rate_dps()
        return (
            self._tilt_band_weight()
            * self._grip_maintained_mask().float()
            * torch.exp(-torch.square(rate / max(std_dps, 1e-3)))
        )

    @_effective_in_named_stages("TILT", "PUSH")
    def _reward_bin_tilt_steady(self):
        """2026-10-03 push2: dense tilt-braking / steady-hold reward in TILT and
        PUSH. Pays inside the tilt band [10, 22] deg with grip maintained when
        angular rate |d theta/dt| is small (Gaussian width bin_tilt_steady_std_dps,
        default 15 deg/s), directly rewarding the robot for arresting the
        backward swing before/after the TILT->PUSH handover."""
        return self._tilt_steady_weight() * self._push_motion_anti_stall_weight()

    @_effective_in_named_stages("GRASP")
    def _reward_grasp_advance(self):
        """2026-10-04 full2: one-time milestone bonus on the step that GRASP
        satisfies _stage_2_to_3_advance_condition() and advances into TILT
        (or PUSH when enable_tilt_stage is False). When grasp_is_final is False,
        complete only fires at the end of PUSH; this preserves the gate bonus at
        GRASP->TILT while continuing live into TILT."""
        next_stage = self.STAGE_TILT if self._tilt_enabled else self.STAGE_PUSH
        cond = (
            (self.stage_buf == self.STAGE_GRASP)
            & (self.current_max_stage_buf < next_stage)
            & ~self.just_resetted_buf
            & ~self.reset_buf.bool()
            & self._stage_2_to_3_advance_condition()
        )
        return cond.float()

    @_effective_in_named_stages("TILT")
    def _reward_tilt_advance(self):
        """2026-10-03 push3: one-time milestone bonus on the step that TILT
        satisfies _stage_3_to_4_advance_condition() and advances into PUSH
        (when tilt_is_final is False, complete only fires at the end of PUSH;
        this preserves tilt5/tilt7's +20 gate bonus while continuing live into
        PUSH)."""
        if not self._tilt_enabled:
            return torch.zeros(self.num_envs, device=self.device)
        cond = (
            (self.stage_buf == self.STAGE_TILT)
            & (self.current_max_stage_buf < self.STAGE_PUSH)
            & ~self.just_resetted_buf
            & ~self.reset_buf.bool()
            & self._stage_3_to_4_advance_condition()
        )
        return cond.float()

    @_effective_in_named_stages("TILT", "PUSH")  # 2026-09-29 overhand: + TILT
    def _reward_bin_tilt_back(self):
        """2026-09-25: POSITIVE reward for tipping the bin back onto its wheels.

        Gaussian in signed back-tilt, peaking at bin_tilt_target_deg (15) with
        a wide std (bin_tilt_back_std_deg, 10) so an upright bin (0 deg) still
        sees a gradient (exp(-2.25) ~ 0.11). The existing penalty_bin_tilt only
        penalized deviation (max 0.068/step) -- too weak to shape anything, and
        the bin can't roll without tipping (front base drags on the floor).
        """
        t = float(self.config.get("bin_tilt_target_deg", 15.0))
        std = float(self.config.get("bin_tilt_back_std_deg", 10.0))
        raw = torch.exp(-torch.square((self._bin_tilt_back_deg() - t) / std))
        if bool(self.config.get("bin_tilt_back_zero_at_upright", False)):
            # 2026-10-03 push4/push5: subtract the Gaussian value at
            # bin_tilt_back_min_deg_tilt (default 0.0 deg; 4.0 deg in push5) in
            # TILT and gate by grip maintained, so parking flat or slightly
            # rocked (<= min_deg_tilt) in TILT earns 0.0 instead of farming the
            # Gaussian tail.
            min_deg_t = float(self.config.get("bin_tilt_back_min_deg_tilt", 0.0))
            b0 = math.exp(-((min_deg_t - t) / std) ** 2)
            norm_tilt = ((raw - b0) / max(1.0 - b0, 1e-4)).clamp(min=0.0, max=1.0)
            in_tilt = self.stage_buf == self.STAGE_TILT
            raw = torch.where(in_tilt, norm_tilt, raw) * self._grip_maintained_mask().float()
        return raw * self._push_motion_anti_stall_weight()

    @_effective_in_named_stages("PUSH")
    def _reward_penalty_bin_tilt(self):
        """Penalize the bin deviating from a TARGET backward tilt during PUSH.

        Two changes (this run):
        1. Gated to PUSH only. Previously active in ALL stages, it punished the
           robot for the incidental bin contact needed to REACH/grasp the rim —
           the policy learned to keep its hands away (swipe-and-retreat) to avoid
           it. Tipping only matters once we're actually driving the bin.
        2. Target is NOT upright — it's the natural 2-wheel push posture: the bin
           tilted BACK toward the robot by `bin_tilt_target_deg` (~15deg, handles
           ~10-15cm down). This is how a person wheels a curbside bin — tilt onto
           the wheels to distribute weight — so we reward that pose instead of
           bolt-upright. Penalize squared deviation of the bin up-axis from the
           tilted target up-axis.
        """
        # get_task_root_state returns wxyz (same as L375); do NOT reorder.
        bin_quat_wxyz = self.simulator.get_task_root_state("bin")[:, 3:7]
        bin_up = quat_apply(bin_quat_wxyz,
                            torch.tensor([[0.0, 0.0, 1.0]], device=self.device).repeat(self.num_envs, 1))

        # Target up-axis = vertical tilted back toward the robot by target angle.
        # "Back toward robot" = opposite the push direction (robot pushes +push_axis).
        # tilted_up = cos(t)*Zhat - sin(t)*push_dir_xy   (top leans against push dir)
        t = torch.deg2rad(torch.tensor(
            float(self.config.get("bin_tilt_target_deg", 15.0)), device=self.device))
        push_xy = F.normalize(self.push_axis_world[:, :2], dim=-1)  # (num_envs, 2)
        target_up = torch.zeros_like(bin_up)
        target_up[:, 0] = -torch.sin(t) * push_xy[:, 0]
        target_up[:, 1] = -torch.sin(t) * push_xy[:, 1]
        target_up[:, 2] = torch.cos(t)
        return torch.sum(torch.square(bin_up - target_up), dim=-1)

    @_effective_in_named_stages("GRASP", "TILT", "PUSH")  # 2026-09-29 overhand: + TILT
    def _reward_penalty_asymmetric_grasp(self):
        """Penalize imbalance between left and right hand bar distances.

        2026-08-29: bar-distance-based (see _hand_bar_distances) -- same
        imbalance concept, now in meters instead of a link-count delta. May
        need its associated reward-scale retuned given the different
        magnitude (0-0.04ish typical vs. 0-8 for counts); watch
        rew_penalty_asymmetric_grasp once tested.
        """
        left_dist, right_dist = self._hand_bar_distances()
        return torch.abs(left_dist - right_dist)

    def _reward_penalty_upright(self):
        """Penalize torso tilt from vertical (squared deviation of torso up-axis from +Z)."""
        upright_vec = torch.tensor([[0.0, 0.0, 1.0]], device=self.device).repeat(self.num_envs, 1)
        torso_quat_wxyz = xyzw_to_wxyz(self.simulator._rigid_body_rot[:, self.torso_index])
        rotated_vec = quat_apply(torso_quat_wxyz, upright_vec)
        return torch.sum(torch.square(rotated_vec - upright_vec), dim=-1)

    def _reward_penalty_humanly_dof_limit(self):
        """Hinge penalty for DoFs outside the 'humanly reasonable' limits."""
        lower_violations = -1.0 * (
            self.simulator.dof_pos - self.dof_pos_humanly_lower_limit
        ).clip(max=0.0).sum(dim=-1)
        upper_violations = (
            self.simulator.dof_pos - self.dof_pos_humanly_upper_limit
        ).clip(min=0.0).sum(dim=-1)
        return lower_violations + upper_violations

    @override
    def _reward_penalty_dof_acc(self):
        return torch.sum(
            torch.square(self.simulator.dof_acc[:, self._upper_non_finger_dof_idx]), dim=-1
        )

    @override
    def _reward_penalty_dof_vel(self):
        return torch.sum(
            torch.square(self.simulator.dof_vel[:, self._upper_non_finger_dof_idx]), dim=-1
        )

    def _reward_penalty_dof_overspeed(self):
        return (
            torch.maximum(
                torch.abs(self.simulator.dof_vel[:, self._upper_non_finger_dof_idx]) - 2.0,
                torch.zeros_like(self.simulator.dof_vel[:, self._upper_non_finger_dof_idx]),
            ) ** 2
        ).sum(dim=-1)

    @override
    def _reward_penalty_undesired_contact(self):
        return torch.sum(
            torch.norm(
                self.simulator.contact_forces[:, self.penalised_contact_indices, :], dim=-1
            ) > 1,
            dim=1, dtype=torch.float,
        )

    def _reward_penalty_homie_action_limit(self):
        """Penalize extreme locomotion commands (beyond ±0.8 soft cap)."""
        homie_commands = self.get_physical_homie_commands()
        return (
            torch.maximum(
                torch.abs(homie_commands) - 0.8,
                torch.zeros_like(homie_commands),
            ) ** 2
        ).sum(dim=-1)

    # ``_reward_penalty_delta_action_rate`` is provided by ``DeltaActionBase`` and
    # is intentionally NOT overridden here — the base implementation computes the
    # squared delta-action rate directly and is correct as-is.

    # ================================================================
    # RESETS
    # ================================================================

    @override
    def _reset_root_states(self, env_ids, target_root_states=None):
        # Start from base_init_state (sets proper Z height = 0.75m).
        self.target_robot_root_states[env_ids] = self.base_init_state
        self.target_robot_root_states[env_ids, :3] += self.env_origins[env_ids]

        # Zero Z lin vel to avoid drop/float; small XY jitter.
        self.target_robot_root_states[env_ids, 7:9] = torch_rand_float(
            -0.3, 0.3, (len(env_ids), 2), device=str(self.device)
        )
        self.target_robot_root_states[env_ids, 9] = 0.0
        self.target_robot_root_states[env_ids, 10:13] = torch_rand_float(
            -0.3, 0.3, (len(env_ids), 3), device=str(self.device)
        )

        r, p, _ = euler_xyz_from_quat(self.target_robot_root_states[env_ids, 3:7])

        # 2026-10-06 full3: Polar STAGE_APPROACH Spawn DR with FOV-locked yaw.
        # Samples radial distance d ~ U[d_min, d_max] and azimuth phi ~ U[-phi_max, +phi_max]
        # behind the grasp stop target (stop_ox, stop_oy) = (-0.8733, 0.0), then sets
        # robot yaw = atan2(bar_y - y, bar_x - x) + delta_psi_fov so the bin handle bar
        # always starts inside the d435_link camera horizontal FOV.
        if getattr(self, "_spawn_polar_dr", False):
            d_min, d_max, phi_max_deg, fov_yaw_max_deg = self._spawn_polar_active_bounds()
            n_ids = len(env_ids)
            d = torch_rand_float(
                float(d_min), float(d_max), (n_ids, 1), device=str(self.device)
            )[:, 0]
            phi_max_rad = math.radians(float(phi_max_deg))
            phi = torch_rand_float(
                -phi_max_rad, phi_max_rad, (n_ids, 1), device=str(self.device)
            )[:, 0]
            stop_ox = float(self.config.get("spawn_polar_stop_offset_x", -0.8733))
            stop_oy = float(self.config.get("spawn_polar_stop_offset_y", 0.0))
            bar_ox = float(self.config.get("spawn_polar_bar_offset_x", -0.3413))
            bar_oy = float(self.config.get("spawn_polar_bar_offset_y", 0.0))
            rel_x0 = stop_ox - d * torch.cos(phi)
            rel_y0 = stop_oy + d * torch.sin(phi)
            dyaw = (
                self._spawn_bin_yaw_rad[env_ids]
                if hasattr(self, "_spawn_bin_yaw_rad")
                else torch.zeros_like(d)
            )
            c_bin = torch.cos(dyaw)
            s_bin = torch.sin(dyaw)
            rel_x = c_bin * rel_x0 - s_bin * rel_y0
            rel_y = s_bin * rel_x0 + c_bin * rel_y0
            self.target_robot_root_states[env_ids, 0] = self.env_origins[env_ids, 0] + rel_x
            self.target_robot_root_states[env_ids, 1] = self.env_origins[env_ids, 1] + rel_y

            psi_look = torch.atan2(bar_oy - rel_y0, bar_ox - rel_x0) + dyaw
            fov_yaw_rad = math.radians(float(fov_yaw_max_deg))
            delta_psi = torch_rand_float(
                -fov_yaw_rad, fov_yaw_rad, (n_ids, 1), device=str(self.device)
            )[:, 0]
            random_yaw = psi_look + delta_psi
            zero_rp = torch.zeros_like(random_yaw)
            self.target_robot_root_states[env_ids, 3:7] = quat_from_euler_xyz(
                zero_rp, zero_rp, random_yaw
            )
            self._spawn_dr_init_dist[env_ids] = d
            self._spawn_dr_init_phi_deg[env_ids] = phi * (180.0 / math.pi)
            self._spawn_dr_init_yaw_deg[env_ids] = random_yaw * (180.0 / math.pi)
            self._spawn_dr_ep_level[env_ids] = float(self.spawn_dr_dist_level_m)
            return

        # Push axis is +X world (v1 locked). Spawn G1 ~2 m behind bin at env origin,
        # so along -X. Small XY jitter, tight yaw range so we face the bin.
        # spawn_x_range config-driven so an ISOLATION experiment (Run 12b) can spawn
        # the robot right at the stop pose (e.g. [-0.5, -0.4]) to train reach/grasp
        # in isolation with walking disabled. Default = normal ~2m-behind spawn.
        spawn_x = self.config.get("spawn_x_range", [-2.3, -1.7])
        self.target_robot_root_states[env_ids, 0:1] = (
            self.env_origins[env_ids, 0:1]
            + torch_rand_float(float(spawn_x[0]), float(spawn_x[1]), (len(env_ids), 1), device=str(self.device))
        )
        # 2026-09-29 reach_grasp: lateral / yaw jitter config-driven
        # (spawn_y_range, spawn_yaw_range_deg). Defaults = old hardcoded
        # ±0.3 m / ±15°; the reach->grasp exp spawns inside the stop-pose
        # tolerance (stop_transition_lateral_tolerance 0.12).
        spawn_y = self.config.get("spawn_y_range", [-0.3, 0.3])
        self.target_robot_root_states[env_ids, 1:2] = (
            self.env_origins[env_ids, 1:2]
            + torch_rand_float(float(spawn_y[0]), float(spawn_y[1]), (len(env_ids), 1), device=str(self.device))
        )

        # Yaw facing +X (toward the bin). ±15° jitter.
        yaw_jitter = math.radians(float(self.config.get("spawn_yaw_range_deg", 15.0)))
        random_yaw = torch_rand_float(
            -yaw_jitter,
            yaw_jitter,
            (len(env_ids), 1), device=str(self.device)
        )[:, 0]
        zero_rp = torch.zeros_like(random_yaw)
        self.target_robot_root_states[env_ids, 3:7] = quat_from_euler_xyz(
            zero_rp, zero_rp, random_yaw
        )

    @override
    def _reset_dofs(self, env_ids, target_state=None):
        # Fix (2026-08-24): this override only ever wrote wrist_dof_idx and
        # finger_dof_idx below, leaving every other DOF (legs, torso, other
        # arm joints) at target_robot_dof_state's raw torch.zeros() buffer
        # value -- i.e. every episode reset was placing the robot at a flat
        # 0-rad pose instead of the configured standing pose. Seed the full
        # body to the real resting_dof_pos first; the wrist/finger-specific
        # randomization below still overwrites those columns as before.
        self.target_robot_dof_state[env_ids, :, 0] = self.resting_dof_pos

        # Optional pre-positioned arm pose (2026-09-12). When set, the 14 arm
        # DOFs are seeded to an explicit configuration instead of resting_dof_pos
        # (arms down), so an episode can begin with the palms already at the
        # handles. Used by the reach/grasp isolation config to study the GRASP
        # and PUSH rewards -- whether the policy lowers the handle and drives the
        # bin the way a human does -- without first having to solve approach and
        # reach. Left unset (None) for the normal task.
        spawn_arm_dof_pos = self.config.get("spawn_arm_dof_pos", None)
        if spawn_arm_dof_pos is not None:
            arm_idx = torch.cat([self._left_arm_dof_idx, self._right_arm_dof_idx])
            arm_pose = torch.tensor(
                list(spawn_arm_dof_pos), device=self.device, dtype=torch.float32
            )
            assert arm_pose.numel() == arm_idx.numel(), (
                f"spawn_arm_dof_pos has {arm_pose.numel()} entries, expected "
                f"{arm_idx.numel()} (left arm 7 then right arm 7)"
            )
            xx, yy = torch.meshgrid(env_ids, arm_idx)
            self.target_robot_dof_state[xx, yy, 0] = arm_pose[None, :]

        # Reset randomization is blended toward resting_dof_pos by these factors
        # (1.0 = the original full-range randomization, 0.0 = exactly resting).
        # 2026-09-12: made configurable rather than fixed. Every episode was
        # starting with wrists randomized across +-80deg and fingers across
        # their ENTIRE joint range, which is a lot of initial-state noise for a
        # task whose gate wants both palms held inside 4cm/3cm for
        # grasp_transition_hold_steps consecutive steps. Default is unchanged;
        # the reach/grasp isolation config narrows it.
        wrist_scale = float(self.config.get("reset_wrist_random_scale", 1.0))
        finger_scale = float(self.config.get("reset_finger_random_scale", 1.0))

        xx, yy = torch.meshgrid(env_ids, self.wrist_dof_idx)
        wrist_rest = self.resting_dof_pos[:, self.wrist_dof_idx]
        self.target_robot_dof_state[xx, yy, 0] = wrist_rest + wrist_scale * (
            torch_rand_float(
                -1.39626, 1.39626,
                (len(env_ids), len(self.wrist_dof_idx)),
                device=str(self.device),
            )
            - wrist_rest
        )

        xx, yy = torch.meshgrid(env_ids, self.finger_dof_idx)
        upper_limit = torch.tensor(
            self.simulator.robot_config.dof_pos_upper_limit_list, device=str(self.device)
        )[None, self.finger_dof_idx]
        lower_limit = torch.tensor(
            self.simulator.robot_config.dof_pos_lower_limit_list, device=str(self.device)
        )[None, self.finger_dof_idx]
        finger_random = lower_limit + (
            upper_limit - lower_limit
        ) * torch_rand_float(
            0.0, 1.0, (len(env_ids), len(self.finger_dof_idx)), device=str(self.device)
        )
        finger_rest = self.resting_dof_pos[:, self.finger_dof_idx]
        self.target_robot_dof_state[xx, yy, 0] = finger_rest + finger_scale * (
            finger_random - finger_rest
        )

        self.target_robot_dof_state[env_ids, :, 1] = 0.0

    @override
    def set_to_stage(self, env_ids, stage):
        super().set_to_stage(env_ids, stage)
        # 2026-10-01: remember the stage the new episode starts in (stage-0
        # fresh start, staged-reset snapshot, or grasp-init TILT).
        if hasattr(self, "_ep_start_stage"):
            self._ep_start_stage[env_ids] = stage

    def _record_episode_outcomes(self, env_ids):
        """2026-10-01 success metrics (logging only). For every env in env_ids
        whose episode actually ran (total_time_buf > 0, i.e. not the startup
        reset), push one row into a ring buffer of the last
        `success_metrics_window` (512) ended episodes and recompute:

        - episode_success_frac: completed (the task's complete condition was
          met during the episode -- works for grasp_is_final and tilt_is_final).
        - grasp_reliable_frac: of episodes that were in GRASP (started at or
          before GRASP and reached it), the fraction that later reached TILT.
          Episodes that STARTED in TILT (grasp-init / TILT snapshots) are
          excluded.
        - tilt_success_frac: of episodes that were in TILT (incl. TILT
          starts), the fraction that completed.
        - time_to_success_s: mean episode time at completion.
        - grasp_success_frac: of episodes that STARTED at or before GRASP, the
          fraction that reached TILT (pass the grasp gate). Same quantity as
          Env/average_goal_reached in the grasp_is_final runs (reach_grasp2),
          for the pre/post-switch comparison.

        Written to log_dict only here; log_dict persists between steps, so the
        trainer's per-step average carries the last value through steps with no
        resets instead of being diluted by them. A metric whose denominator is
        empty in the window keeps its previous value."""
        if not hasattr(self, "_succ_buf") or len(env_ids) == 0:
            return
        ran = self.total_time_buf[env_ids] > 0
        ids = env_ids[ran]
        if len(ids) == 0:
            return
        g, t = self.STAGE_GRASP, self.STAGE_TILT
        start = self._ep_start_stage[ids]
        mx = self.current_max_stage_buf[ids]
        done = self.current_completed_task_buf[ids]
        entered_grasp = (start <= g) & (mx >= g)
        if t >= 0:
            reached_tilt = entered_grasp & (mx >= t)
            entered_tilt = mx >= t
        else:
            reached_tilt = torch.zeros_like(entered_grasp)
            entered_tilt = torch.zeros_like(entered_grasp)
        tsc = self.time_since_completion_buf[ids]
        t_done = (self.total_time_buf[ids] - (tsc - 1).clamp(min=0)).float() * float(self.dt)
        # 2026-10-03 push1: reached PUSH, and the episode's bin body mass (still
        # the ended episode's: _apply_bin_mass_curriculum re-samples after this
        # call in _reset_tasks_callback).
        p = self.STAGE_PUSH
        reached_push = (mx >= p) if p >= 0 else torch.zeros_like(entered_grasp)
        started_approach = start == self.STAGE_APPROACH
        reached_reach = started_approach & (mx >= self.STAGE_REACH)
        rows = torch.stack(
            [done.float(), entered_grasp.float(), reached_tilt.float(),
             entered_tilt.float(), torch.where(done, t_done, torch.zeros_like(t_done)),
             (start <= g).float(), reached_push.float(), self._bin_mass_env[ids].float(),
             started_approach.float(), reached_reach.float()],
            dim=-1,
        )[-self._succ_win:]
        k = rows.shape[0]
        idx = (torch.arange(k, device=self.device) + self._succ_ptr) % self._succ_win
        self._succ_buf[idx] = rows
        self._succ_ptr = (self._succ_ptr + k) % self._succ_win
        self._succ_count = min(self._succ_count + k, self._succ_win)
        b = self._succ_buf[: self._succ_count] if self._succ_count < self._succ_win else self._succ_buf
        c, eg, rt, et, tt, sg, rp, mk, sa, rr = b.unbind(-1)
        self.log_dict["episode_success_frac"] = c.mean()
        self.log_dict["success_window_episodes"] = torch.tensor(float(b.shape[0]), device=self.device)
        if sa.sum() > 0:
            self.log_dict["approach_reached_frac"] = (rr * sa).sum() / sa.sum()
            self.log_dict["approach_ep_success_frac"] = (c * sa).sum() / sa.sum()
        if eg.sum() > 0:
            self.log_dict["grasp_reliable_frac"] = (rt * eg).sum() / eg.sum()
        if sg.sum() > 0:
            self.log_dict["grasp_success_frac"] = (rt * sg).sum() / sg.sum()
        if et.sum() > 0:
            self.log_dict["tilt_success_frac"] = (c * et).sum() / et.sum()
        if c.sum() > 0:
            self.log_dict["time_to_success_s"] = (tt * c).sum() / c.sum()
        # 2026-10-03 push1 (push_stage_logs, default False): with tilt_is_final
        # off, "completed" = PUSH complete, so tilt_success_frac above becomes
        # the PUSH success of TILT episodes. push_reached_frac = of episodes in
        # TILT, the fraction that passed the TILT->PUSH gate (the old tilt
        # success); push_success_frac = of those, the fraction that completed.
        push_logs = bool(self.config.get("push_stage_logs", False))
        if push_logs:
            if et.sum() > 0:
                self.log_dict["push_reached_frac"] = (rp * et).sum() / et.sum()
            if rp.sum() > 0:
                self.log_dict["push_success_frac"] = (c * rp).sum() / rp.sum()
        # 2026-10-03 push1 (success_by_mass_edges_kg, default None): episode
        # success (and PUSH reached, with push_stage_logs) per bin body-mass
        # bucket [lo, hi) over the same window, e.g. [6, 9, 11, 13] ->
        # episode_success_frac_m6_9 / _m9_11 / _m11_13.
        edges = self.config.get("success_by_mass_edges_kg", None)
        if edges:
            edges = [float(e) for e in edges]
            for lo, hi in zip(edges[:-1], edges[1:]):
                sel = ((mk >= lo) & (mk < hi)).float()
                n = sel.sum()
                if n > 0:
                    tag = f"m{lo:g}_{hi:g}"
                    self.log_dict[f"episode_success_frac_{tag}"] = (c * sel).sum() / n
                    if push_logs:
                        self.log_dict[f"push_reached_frac_{tag}"] = (rp * sel).sum() / n

    @override
    def load_env_state_dict(self, state_dict):
        """2026-10-02 grasp_tilt4 (bin_mass_level_reset_kg, default None = old
        behaviour). The base load restores bin_mass_level_kg from the
        checkpoint's env state (= log_dict), so bin_mass_start_kg alone cannot
        LOWER the level on a resume (m10 sat at 6 kg: 4-6 kg bins are
        edge-held, 2-3 kg bins tip and succeed). If bin_mass_level_reset_kg is
        set, put the level back to it and clear the level counts ONCE: the
        checkpoints of the run then carry bin_mass_level_reset_applied_kg
        (written in _update_bin_mass_curriculum), and a later resume with the
        same reset value keeps the restored (possibly advanced) level."""
        state_dict = dict(state_dict)
        # popped so the base load does not warn "Key ... not found"
        applied = state_dict.pop("bin_mass_level_reset_applied_kg", None)
        push_reset_applied = state_dict.pop("push_distance_level_reset_applied_m", None)
        spawn_reset_applied = state_dict.pop("spawn_polar_dist_level_reset_applied_m", None)
        super().load_env_state_dict(state_dict)
        self._post_load_tilt_rate_gate()  # 2026-10-03 tilt7
        self._post_load_push_distance_curriculum(push_reset_applied)  # 2026-10-03 push2 / full3
        self._post_load_spawn_dr_curriculum(spawn_reset_applied)  # 2026-10-06 full3
        reset_kg = self.config.get("bin_mass_level_reset_kg", None)
        if reset_kg is None or not getattr(self, "_bin_mass_cur", False):
            return
        reset_kg = float(reset_kg)
        if applied is not None and abs(float(applied) - reset_kg) < 1e-6:
            logger.info(
                f"bin mass curriculum: level reset to {reset_kg:.2f} kg already applied "
                f"(checkpoint marker); keeping restored level "
                f"{float(self.bin_mass_level_kg):.2f} kg"
            )
            return
        old = float(self.bin_mass_level_kg)
        self.bin_mass_level_kg = torch.tensor(reset_kg, device=self.device)
        self._bin_mass_lvl_eps = 0
        self._bin_mass_lvl_succ = 0
        self.log_dict["bin_mass_level_kg"] = self.bin_mass_level_kg.clone()
        self.log_dict["bin_mass_level_reset_applied_kg"] = torch.tensor(
            reset_kg, device=self.device
        )
        logger.info(
            f"bin mass curriculum: level reset {old:.2f} -> {reset_kg:.2f} kg "
            f"(bin_mass_level_reset_kg)"
        )

    def _post_load_push_distance_curriculum(self, reset_applied=None):
        """2026-10-03 push2: clamp restored target_bin_push_distance_m to
        [push_distance_start_m, push_distance_end_m] after checkpoint load,
        and optionally apply push_distance_level_reset_m once."""
        if not getattr(self, "_push_dist_cur", False):
            return
        start = float(self.config.get("push_distance_start_m", 0.10))
        end = max(float(self.config.get("push_distance_end_m", 1.00)), start)
        lvl = min(max(float(self.target_bin_push_distance_m), start), end)
        reset_m = self.config.get("push_distance_level_reset_m", None)
        if reset_m is not None:
            reset_m = float(reset_m)
            if reset_applied is None or abs(float(reset_applied) - reset_m) > 1e-6:
                logger.info(
                    f"push distance curriculum: level reset {lvl:.2f} -> {reset_m:.2f} m "
                    f"(push_distance_level_reset_m)"
                )
                lvl = reset_m
                self.log_dict["push_distance_level_reset_applied_m"] = torch.tensor(
                    reset_m, device=self.device
                )
        self.target_bin_push_distance_m = torch.tensor(lvl, device=self.device)
        self._push_dist_ep_level[:] = lvl
        self._push_dist_lvl_eps = 0
        self._push_dist_lvl_succ = 0
        self.log_dict["target_bin_push_distance_m"] = self.target_bin_push_distance_m.clone()
        logger.info(
            f"push distance curriculum: target {lvl:.2f} m after checkpoint load "
            f"(range [{start:.2f}, {end:.2f}] m)"
        )

    def _post_load_spawn_dr_curriculum(self, reset_applied=None):
        """2026-10-06 full3: clamp restored spawn_dr_dist_level_m to
        [spawn_polar_dist_start_m, spawn_polar_dist_max_m] after checkpoint load,
        and optionally apply spawn_polar_dist_level_reset_m once."""
        if not getattr(self, "_spawn_polar_dr", False):
            return
        start = float(self.config.get("spawn_polar_dist_start_m", 0.25))
        end = max(float(self.config.get("spawn_polar_dist_max_m", 1.10)), start)
        if getattr(self, "_spawn_polar_cur", False):
            lvl = min(max(float(self.spawn_dr_dist_level_m), start), end)
            reset_m = self.config.get("spawn_polar_dist_level_reset_m", None)
            if reset_m is not None:
                reset_m = float(reset_m)
                if reset_applied is None or abs(float(reset_applied) - reset_m) > 1e-6:
                    logger.info(
                        f"spawn polar DR curriculum: level reset {lvl:.2f} -> {reset_m:.2f} m "
                        f"(spawn_polar_dist_level_reset_m)"
                    )
                    lvl = reset_m
                    self.log_dict["spawn_polar_dist_level_reset_applied_m"] = torch.tensor(
                        reset_m, device=self.device
                    )
        else:
            lvl = end
        self.spawn_dr_dist_level_m = torch.tensor(lvl, device=self.device)
        self._spawn_dr_ep_level[:] = lvl
        self._spawn_dr_lvl_eps = 0
        self._spawn_dr_lvl_reach = 0
        self._spawn_dr_lvl_succ = 0
        _, _, phi_cur, yaw_cur = self._spawn_polar_active_bounds()
        self.log_dict["spawn_dr_dist_level_m"] = self.spawn_dr_dist_level_m.clone()
        self.log_dict["spawn_dr_phi_level_deg"] = torch.tensor(phi_cur, device=self.device)
        self.log_dict["spawn_dr_fov_yaw_level_deg"] = torch.tensor(yaw_cur, device=self.device)
        logger.info(
            f"spawn polar DR: dist_level={lvl:.2f} m, phi_max={phi_cur:.1f} deg, "
            f"fov_yaw_max={yaw_cur:.1f} deg after checkpoint load"
        )

    def _update_spawn_dr_curriculum(self, env_ids):
        """2026-10-06 full3 (spawn_polar_curriculum, default False): count ended
        episodes that started in STAGE_APPROACH at the current spawn_dr_dist_level_m,
        how many reached STAGE_REACH, and how many completed the full task. Once >=
        spawn_polar_curriculum_min_episodes (800) are counted: if reach_rate >=
        spawn_polar_curriculum_advance_reach (0.75) AND ep_succ_rate >=
        spawn_polar_curriculum_advance_success (0.50), advance spawn_dr_dist_level_m
        by spawn_polar_dist_step_m (0.15 m, capped at spawn_polar_dist_max_m 1.10 m)
        and reset counts; otherwise halve counts. Must run before
        super()._reset_tasks_callback."""
        if not getattr(self, "_spawn_polar_cur", False) or len(env_ids) == 0:
            return
        lvl = float(self.spawn_dr_dist_level_m)
        ran = self.total_time_buf[env_ids] > 0
        cur = (
            ran
            & (self._ep_start_stage[env_ids] == self.STAGE_APPROACH)
            & ((self._spawn_dr_ep_level[env_ids] - lvl).abs() < 1e-4)
        )
        n = int(cur.sum())
        if n > 0:
            self._spawn_dr_lvl_eps += n
            self._spawn_dr_lvl_reach += int(
                (cur & (self.current_max_stage_buf[env_ids] >= self.STAGE_REACH)).sum()
            )
            self._spawn_dr_lvl_succ += int(
                (cur & self.current_completed_task_buf[env_ids]).sum()
            )
        end = float(self.config.get("spawn_polar_dist_max_m", 1.10))
        need = int(self.config.get("spawn_polar_curriculum_min_episodes", 800))
        thr_reach = float(self.config.get("spawn_polar_curriculum_advance_reach", 0.75))
        thr_succ = float(self.config.get("spawn_polar_curriculum_advance_success", 0.50))
        if self._spawn_dr_lvl_eps >= need:
            reach_rate = self._spawn_dr_lvl_reach / max(self._spawn_dr_lvl_eps, 1)
            succ_rate = self._spawn_dr_lvl_succ / max(self._spawn_dr_lvl_eps, 1)
            if reach_rate >= thr_reach and succ_rate >= thr_succ and lvl < end - 1e-6:
                step = float(self.config.get("spawn_polar_dist_step_m", 0.15))
                new = min(end, round(lvl + step, 4))
                self.spawn_dr_dist_level_m = torch.tensor(new, device=self.device)
                _, _, phi_new, yaw_new = self._spawn_polar_active_bounds()
                logger.info(
                    f"spawn polar DR curriculum: {lvl:.2f} -> {new:.2f} m "
                    f"(phi_max={phi_new:.1f} deg, fov_yaw_max={yaw_new:.1f} deg; "
                    f"reach={reach_rate:.3f}, succ={succ_rate:.3f} over "
                    f"{self._spawn_dr_lvl_eps} APPROACH episodes)"
                )
                self._spawn_dr_lvl_eps = 0
                self._spawn_dr_lvl_reach = 0
                self._spawn_dr_lvl_succ = 0
            elif lvl < end - 1e-6:
                self._spawn_dr_lvl_eps //= 2
                self._spawn_dr_lvl_reach //= 2
                self._spawn_dr_lvl_succ //= 2
        _, _, phi_cur, yaw_cur = self._spawn_polar_active_bounds()
        self.log_dict["spawn_dr_dist_level_m"] = self.spawn_dr_dist_level_m.clone()
        self.log_dict["spawn_dr_phi_level_deg"] = torch.tensor(phi_cur, device=self.device)
        self.log_dict["spawn_dr_fov_yaw_level_deg"] = torch.tensor(yaw_cur, device=self.device)
        self.log_dict["spawn_dr_level_episodes"] = torch.tensor(
            float(self._spawn_dr_lvl_eps), device=self.device
        )
        if self._spawn_dr_lvl_eps > 0:
            self.log_dict["spawn_dr_level_reach_frac"] = torch.tensor(
                self._spawn_dr_lvl_reach / self._spawn_dr_lvl_eps, device=self.device
            )
            self.log_dict["spawn_dr_level_succ_frac"] = torch.tensor(
                self._spawn_dr_lvl_succ / self._spawn_dr_lvl_eps, device=self.device
            )
        reset_m = self.config.get("spawn_polar_dist_level_reset_m", None)
        if reset_m is not None:
            self.log_dict["spawn_polar_dist_level_reset_applied_m"] = torch.tensor(
                float(reset_m), device=self.device
            )

    def _update_push_distance_curriculum(self, env_ids):
        """2026-10-03 push2 (push_distance_curriculum, default False): count
        ended episodes that ran at the current target_bin_push_distance_m and
        entered PUSH, and how many completed. Once >=
        push_distance_advance_min_episodes (1500) are counted: success >=
        push_distance_advance_success (0.40) -> target += push_distance_step_m
        (0.05, capped at push_distance_end_m 1.0) and reset counts; otherwise
        halve counts. Must run before super()._reset_tasks_callback."""
        if not getattr(self, "_push_dist_cur", False) or len(env_ids) == 0:
            return
        lvl = float(self.target_bin_push_distance_m)
        ran = self.total_time_buf[env_ids] > 0
        cur = ran & ((self._push_dist_ep_level[env_ids] - lvl).abs() < 1e-4)
        if self.STAGE_PUSH >= 0:
            cur = cur & (self.current_max_stage_buf[env_ids] >= self.STAGE_PUSH)
        n = int(cur.sum())
        if n > 0:
            self._push_dist_lvl_eps += n
            self._push_dist_lvl_succ += int(
                (cur & self.current_completed_task_buf[env_ids]).sum()
            )
        end = float(self.config.get("push_distance_end_m", 1.00))
        need = int(self.config.get("push_distance_advance_min_episodes", 1500))
        thr = float(self.config.get("push_distance_advance_success", 0.40))
        if self._push_dist_lvl_eps >= need:
            rate = self._push_dist_lvl_succ / max(self._push_dist_lvl_eps, 1)
            if rate >= thr and lvl < end - 1e-6:
                step = float(self.config.get("push_distance_step_m", 0.05))
                sched = self.config.get("push_distance_step_schedule", None)
                if sched:
                    for max_lvl, s_val in sched:
                        if lvl < float(max_lvl) - 1e-6:
                            step = float(s_val)
                            break
                new = min(end, round(lvl + step, 4))
                logger.info(
                    f"push distance curriculum: {lvl:.2f} -> {new:.2f} m (push success "
                    f"{rate:.3f} over {self._push_dist_lvl_eps} PUSH episodes)"
                )
                self.target_bin_push_distance_m = torch.tensor(new, device=self.device)
                self._push_dist_lvl_eps = 0
                self._push_dist_lvl_succ = 0
            elif lvl < end - 1e-6:
                self._push_dist_lvl_eps //= 2
                self._push_dist_lvl_succ //= 2
        self.log_dict["target_bin_push_distance_m"] = self.target_bin_push_distance_m.clone()
        reset_m = self.config.get("push_distance_level_reset_m", None)
        if reset_m is not None:
            self.log_dict["push_distance_level_reset_applied_m"] = torch.tensor(
                float(reset_m), device=self.device
            )
        self.log_dict["push_dist_level_episodes"] = torch.tensor(
            float(self._push_dist_lvl_eps), device=self.device
        )
        if self._push_dist_lvl_eps > 0:
            self.log_dict["push_dist_level_success"] = torch.tensor(
                self._push_dist_lvl_succ / self._push_dist_lvl_eps, device=self.device
            )

    def _post_load_tilt_rate_gate(self):
        """2026-10-03 tilt7: the base load restored tilt_gate_max_rate_dps if
        the checkpoint has it (a pre-tilt7 checkpoint does not -> the start
        value stays). Fixed threshold (no tilt_rate_curriculum_start_dps):
        always the configured tilt_advance_max_rate_dps. Curriculum: clamp to
        [tilt_advance_max_rate_dps, tilt_rate_curriculum_start_dps]. Then give
        it to the episodes already running and clear the level counts."""
        if not getattr(self, "_tilt_rate_gate", False):
            return
        floor = float(self.config["tilt_advance_max_rate_dps"])
        if self._tilt_rate_cur:
            start = max(float(self.config["tilt_rate_curriculum_start_dps"]), floor)
            thr = min(max(float(self.tilt_gate_max_rate_dps), floor), start)
        else:
            thr = floor
        self.tilt_gate_max_rate_dps = torch.tensor(thr, device=self.device)
        self._tilt_rate_thr_env[:] = thr
        self._tilt_rate_lvl_eps = 0
        self._tilt_rate_lvl_succ = 0
        self.log_dict["tilt_gate_max_rate_dps"] = self.tilt_gate_max_rate_dps.clone()
        logger.info(
            f"tilt rate gate: threshold {thr:.2f} deg/s after checkpoint load "
            f"(floor {floor:.2f}, curriculum {'on' if self._tilt_rate_cur else 'off'})"
        )

    def _update_tilt_rate_curriculum(self, env_ids):
        """2026-10-03 tilt7 (tilt_rate_curriculum_start_dps, default None =
        fixed threshold): count the ended episodes that ran with the CURRENT
        threshold (_tilt_rate_thr_env, set at their reset) and entered TILT,
        and how many of them completed (= tilt_success_frac's definition with
        tilt_is_final). Once >= tilt_rate_curriculum_min_episodes (3000) are
        counted: success >= tilt_rate_curriculum_success (0.5) -> threshold *=
        tilt_rate_curriculum_factor (0.9; floor tilt_advance_max_rate_dps) and
        reset the counts; success < tilt_rate_curriculum_loosen_success
        (default None = never) -> threshold /= factor (cap: the start value)
        and reset the counts; otherwise halve both counts (as in
        _update_bin_mass_curriculum). Must run before
        super()._reset_tasks_callback (current-episode buffers)."""
        if not getattr(self, "_tilt_rate_cur", False) or len(env_ids) == 0:
            return
        thr = float(self.tilt_gate_max_rate_dps)
        ran = self.total_time_buf[env_ids] > 0
        cur = ran & ((self._tilt_rate_thr_env[env_ids] - thr).abs() < 1e-3)
        if self.STAGE_TILT >= 0:
            cur = cur & (self.current_max_stage_buf[env_ids] >= self.STAGE_TILT)
        n = int(cur.sum())
        if n > 0:
            self._tilt_rate_lvl_eps += n
            if bool(self.config.get("tilt_is_final", False)):
                gate_ok = self.current_completed_task_buf[env_ids]
            else:
                gate_ok = self.current_max_stage_buf[env_ids] >= self.STAGE_PUSH
            self._tilt_rate_lvl_succ += int((cur & gate_ok).sum())
        need = int(self.config.get("tilt_rate_curriculum_min_episodes", 3000))
        if self._tilt_rate_lvl_eps >= need:
            floor = float(self.config["tilt_advance_max_rate_dps"])
            start = max(float(self.config["tilt_rate_curriculum_start_dps"]), floor)
            factor = float(self.config.get("tilt_rate_curriculum_factor", 0.9))
            adv = float(self.config.get("tilt_rate_curriculum_success", 0.5))
            loosen = self.config.get("tilt_rate_curriculum_loosen_success", None)
            rate = self._tilt_rate_lvl_succ / max(self._tilt_rate_lvl_eps, 1)
            new = None
            if rate >= adv and thr > floor + 1e-6:
                new = max(floor, thr * factor)
            elif loosen is not None and rate < float(loosen) and thr < start - 1e-6:
                new = min(start, thr / factor)
            if new is not None:
                logger.info(
                    f"tilt rate curriculum: {thr:.2f} -> {new:.2f} deg/s (tilt success "
                    f"{rate:.3f} over {self._tilt_rate_lvl_eps} TILT episodes)"
                )
                self.tilt_gate_max_rate_dps = torch.tensor(new, device=self.device)
                self._tilt_rate_lvl_eps = 0
                self._tilt_rate_lvl_succ = 0
            else:
                self._tilt_rate_lvl_eps //= 2
                self._tilt_rate_lvl_succ //= 2
        self.log_dict["tilt_gate_max_rate_dps"] = self.tilt_gate_max_rate_dps.clone()
        self.log_dict["tilt_rate_level_episodes"] = torch.tensor(
            float(self._tilt_rate_lvl_eps), device=self.device
        )
        if self._tilt_rate_lvl_eps > 0:
            self.log_dict["tilt_rate_level_success"] = torch.tensor(
                self._tilt_rate_lvl_succ / self._tilt_rate_lvl_eps, device=self.device
            )

    def _update_bin_mass_curriculum(self, env_ids):
        """2026-10-01 tilt3 (bin_mass_curriculum, default off): count the ended
        episodes whose bin mass was sampled at the CURRENT level and that
        entered TILT, and how many of them completed (= tilt_success_frac's
        definition). Once >= bin_mass_advance_min_episodes (3000) are counted:
        tilt success >= bin_mass_advance_success (0.3) -> level +=
        bin_mass_step_kg (1, capped at bin_mass_end_kg 15) and reset the
        counts; otherwise halve both counts (forget the oldest half, so a
        policy that improves is not held back by its early failures). Must run
        before super()._reset_tasks_callback (current-episode buffers)."""
        if not getattr(self, "_bin_mass_cur", False) or len(env_ids) == 0:
            return
        lvl = float(self.bin_mass_level_kg)
        ran = self.total_time_buf[env_ids] > 0
        cur = ran & ((self._bin_mass_ep_level[env_ids] - lvl).abs() < 1e-3)
        if self.STAGE_TILT >= 0:
            cur = cur & (self.current_max_stage_buf[env_ids] >= self.STAGE_TILT)
        n = int(cur.sum())
        if n > 0:
            self._bin_mass_lvl_eps += n
            self._bin_mass_lvl_succ += int((cur & self.current_completed_task_buf[env_ids]).sum())
        end = float(self.config.get("bin_mass_end_kg", 15.0))
        need = int(self.config.get("bin_mass_advance_min_episodes", 3000))
        thr = float(self.config.get("bin_mass_advance_success", 0.3))
        if self._bin_mass_lvl_eps >= need:
            rate = self._bin_mass_lvl_succ / max(self._bin_mass_lvl_eps, 1)
            if rate >= thr and lvl < end - 1e-6:
                new = min(end, lvl + float(self.config.get("bin_mass_step_kg", 1.0)))
                logger.info(
                    f"bin mass curriculum: {lvl:.2f} -> {new:.2f} kg (tilt success "
                    f"{rate:.3f} over {self._bin_mass_lvl_eps} TILT episodes)"
                )
                self.bin_mass_level_kg = torch.tensor(new, device=self.device)
                self._bin_mass_lvl_eps = 0
                self._bin_mass_lvl_succ = 0
            elif lvl < end - 1e-6:
                self._bin_mass_lvl_eps //= 2
                self._bin_mass_lvl_succ //= 2
        self.log_dict["bin_mass_level_kg"] = self.bin_mass_level_kg.clone()
        # 2026-10-02 grasp_tilt4: checkpoint marker for bin_mass_level_reset_kg
        # (log_dict is saved as the env state dict; see load_env_state_dict).
        reset_kg = self.config.get("bin_mass_level_reset_kg", None)
        if reset_kg is not None:
            self.log_dict["bin_mass_level_reset_applied_kg"] = torch.tensor(
                float(reset_kg), device=self.device
            )
        # bin_mass_mean_kg is logged in _apply_bin_mass_curriculum (after the
        # resetting envs got their new mass; logging it here read the old one).
        self.log_dict["bin_mass_level_episodes"] = torch.tensor(
            float(self._bin_mass_lvl_eps), device=self.device
        )
        if self._bin_mass_lvl_eps > 0:
            self.log_dict["bin_mass_level_tilt_success"] = torch.tensor(
                self._bin_mass_lvl_succ / self._bin_mass_lvl_eps, device=self.device
            )

    def _apply_bin_mass_curriculum(self, env_ids):
        """2026-10-01 tilt3 (bin_mass_curriculum, default off): re-sample the
        bin BODY mass of the resetting envs from U[max(bin_mass_start_kg,
        level - bin_mass_spread_kg), level] (wheels unchanged) and scale the
        body inertia by the same ratio, through the bin articulation's PhysX
        view (CPU tensors; same calls as tilt_feasibility A5). Updates
        _privileged_bin_mass with the live sampled per-env mass."""
        if not getattr(self, "_bin_mass_cur", False) or len(env_ids) == 0:
            return
        art = self.simulator.scene.articulations["bin"]
        view = art.root_physx_view
        if not hasattr(self, "_bin_mass_base"):
            self._bin_body_bi = list(art.body_names).index("body")
            self._bin_mass_cpu = view.get_masses().clone()
            self._bin_inertia_cpu = view.get_inertias().clone()
            self._bin_mass_base = self._bin_mass_cpu[:, self._bin_body_bi].clone()
            self._bin_inertia_base = self._bin_inertia_cpu[:, self._bin_body_bi].clone()
            logger.info(
                f"bin mass curriculum: base body mass {float(self._bin_mass_base[0]):.2f} kg, "
                f"bodies {list(art.body_names)}, level {float(self.bin_mass_level_kg):.2f} kg"
            )
        lvl = float(self.bin_mass_level_kg)
        lo = max(
            float(self.config.get("bin_mass_start_kg", 15.0)),
            lvl - float(self.config.get("bin_mass_spread_kg", 0.0)),
        )
        lo = min(lo, lvl)
        ids_cpu = env_ids.detach().to("cpu", dtype=torch.long)
        m = lo + (lvl - lo) * torch.rand(len(ids_cpu))
        ratio = m / self._bin_mass_base[ids_cpu]
        self._bin_mass_cpu[ids_cpu, self._bin_body_bi] = m
        self._bin_inertia_cpu[ids_cpu, self._bin_body_bi] = (
            self._bin_inertia_base[ids_cpu] * ratio[:, None]
        )
        idx = ids_cpu.to(torch.int32)
        view.set_masses(self._bin_mass_cpu, idx)
        view.set_inertias(self._bin_inertia_cpu, idx)
        self._bin_mass_env[env_ids] = m.to(self.device)
        self._privileged_bin_mass[env_ids] = self._bin_mass_env[env_ids]
        self._bin_mass_ep_level[env_ids] = lvl
        self.log_dict["bin_mass_mean_kg"] = self._bin_mass_env.mean()

    @override
    def _reset_tasks_callback(self, env_ids):
        # 2026-10-01: must run BEFORE super(), which overwrites the
        # current-episode buffers (max stage / completed) for env_ids.
        self._record_episode_outcomes(env_ids)
        self._update_bin_mass_curriculum(env_ids)  # 2026-10-01 tilt3
        self._update_tilt_rate_curriculum(env_ids)  # 2026-10-03 tilt7
        self._update_push_distance_curriculum(env_ids)  # 2026-10-03 push2
        self._update_spawn_dr_curriculum(env_ids)  # 2026-10-06 full3
        bin_yaw_range_deg = float(self.config.get("spawn_bin_yaw_range_deg", 0.0))
        if bin_yaw_range_deg > 0.0 and len(env_ids) > 0:
            max_bin_yaw = math.radians(bin_yaw_range_deg)
            dyaw = torch_rand_float(
                -max_bin_yaw, max_bin_yaw, (len(env_ids), 1), device=str(self.device)
            )[:, 0]
            self._spawn_bin_yaw_rad[env_ids] = dyaw
            self.push_axis_world[env_ids, 0] = torch.cos(dyaw)
            self.push_axis_world[env_ids, 1] = torch.sin(dyaw)
            self.push_axis_world[env_ids, 2] = 0.0
        elif len(env_ids) > 0:
            self._spawn_bin_yaw_rad[env_ids] = 0.0
        super()._reset_tasks_callback(env_ids)
        self._apply_bin_mass_curriculum(env_ids)  # 2026-10-01 tilt3
        self._approach_transition_ready_steps[env_ids] = 0
        self._reach_transition_ready_steps[env_ids] = 0
        self._grasp_valid_steps[env_ids] = 0
        self._push_invalid_grasp_steps[env_ids] = 0
        self._left_finger_close_latched[env_ids] = False
        self._right_finger_close_latched[env_ids] = False
        self._tilt_hold_steps[env_ids] = 0  # 2026-09-29 overhand
        # 2026-10-03 tilt7: the new episode runs its steady-hold gate with the
        # current threshold (0-dim tensor broadcast, no host sync).
        self._tilt_rate_thr_env[env_ids] = self.tilt_gate_max_rate_dps
        self._push_dist_ep_level[env_ids] = self.target_bin_push_distance_m
        self._spawn_dr_ep_level[env_ids] = self.spawn_dr_dist_level_m
        # 2026-09-30 overhand grasp_init: every reset starts un-flagged; the
        # stage-0 subset may be re-flagged in _reset_robot_states_callback
        # (staged-reset snapshot envs never pass through that callback).
        self._grasp_init_mask[env_ids] = False

    @override
    def _reset_object_states_callback(self, env_ids):
        # 2026-09-30 overhand: stage-0 resets never touched the bin before
        # (only staged snapshot resets write its state), so a pushed / toppled
        # bin carried over into the next episode. `reset_bin_on_reset`
        # (default False = old behaviour) puts it back at its default pose;
        # grasp-init envs always get the default bin pose because the canned
        # robot pose is composed onto it.
        if bool(self.config.get("reset_bin_on_reset", False)):
            bin_ids = env_ids
        else:
            bin_ids = env_ids[self._grasp_init_mask[env_ids]]
        if len(bin_ids) > 0:
            self._write_bin_default_state(bin_ids)
        self._reset_bin_states(env_ids)
        if len(bin_ids) > 0:
            # the sim read-back may lag the write; use the written pose.
            self._initial_bin_pos_xy[bin_ids] = self._bin_default_root_state(bin_ids)[:, :2]
            self._initial_bar_pos_xy[bin_ids] = self._bar_mid_xy()[bin_ids]
        return super()._reset_object_states_callback(env_ids)

    # ---- 2026-09-30 overhand: grasp-state init (spec section 6 / B5) ----
    # A fraction `grasp_init_ratio` (default 0.0 = off) of the envs that reset
    # into stage 0 instead start directly in TILT, already holding the bar
    # with the validated overhand claw grasp (th 10 deg, a 0.080, gap 2 mm):
    #   * bin at its default pose; robot root = the canned bin-relative pose
    #     composed onto it (same rigid transform -> grasp geometry preserved,
    #     no independent root noise);
    #   * all joints from the canned state (legs/waist = the hf4 GRASP frame,
    #     arms = IK + settled, fingers = settled claw), zero velocity;
    #     optional `grasp_init_joint_noise` (rad, default 0) on arms+fingers;
    #   * delta-action accumulator = arm targets / fingers closed / HOMIE
    #     height+pitch commands of the canned posture, so the first policy
    #     step does not snap the arms, open the fingers or stand the torso up.
    # Canned state: grasp_init_path (npz from scratch/robot_claw3.py DUMP=...).

    def _grasp_init_state(self):
        """Lazy-load the canned grasped state and precompute everything in the
        env's dof / action spaces."""
        if getattr(self, "_gi_state", None) is not None:
            return self._gi_state
        import numpy as np

        from gr00t.rl.utils.paths import resolve_repo_path

        path = str(resolve_repo_path(str(self.config.get("grasp_init_path"))))
        assert os.path.isfile(path), f"grasp_init_path not found: {path}"
        data = np.load(path)
        names = [str(n) for n in data["joint_names"]]
        dof_names = list(self.dof_names)
        missing = [n for n in dof_names if n not in names]
        assert not missing, f"grasp_init state lacks joints {missing}"
        perm = [names.index(n) for n in dof_names]
        dev = self.device
        # "pulled" (default) = settled after the 2 cm pull test (fingers
        # hooked behind the bar); "closed" = settled after close + hold,
        # before the pull (palm ~on its IK target).
        snap = str(self.config.get("grasp_init_snapshot", "pulled"))
        q_key, t_key = {
            "pulled": ("joint_pos", "joint_target"),
            "closed": ("joint_pos_closed", "joint_target_closed"),
        }[snap]
        q = torch.tensor(data[q_key][perm], device=dev, dtype=torch.float32)
        # Arm TARGETS: "settled" = the settled arm pose (default, no pull
        # tension), "target" = the commanded IK targets of that snapshot.
        arm_mode = str(self.config.get("grasp_init_arm_target", "settled"))
        q_tgt = q.clone()
        if arm_mode == "target":
            q_tgt = torch.tensor(data[t_key][perm], device=dev, dtype=torch.float32)
        scale = float(self.config.robot.control.action_scale)
        default = self.default_dof_pos[0]
        # HOMIE command guess for the canned posture (pelvis height, torso
        # pitch), clipped like HomieBase.step does; overridable.
        h_cmd = self.config.get("grasp_init_homie_height", None)
        p_cmd = self.config.get("grasp_init_homie_pitch", None)
        h_cmd = float(data["homie_height_cmd"]) if h_cmd is None else float(h_cmd)
        p_cmd = float(data["homie_pitch_cmd"]) if p_cmd is None else float(p_cmd)
        if self.config.get("clip_homie_command", False):
            h_cmd = min(max(h_cmd, float(self.config.clip_homie_lower_height_threshold)),
                        float(self.config.clip_homie_upper_height_threshold))
            p_cmd = min(max(p_cmd, float(self.config.clip_homie_torso_pitch_lower_threshold)),
                        float(self.config.clip_homie_torso_pitch_upper_threshold))
        cmd_scale = float(self.config.homie_command_scale)
        n_cmd = self._num_homie_commands  # 7: vx vy wz height roll pitch yaw
        processed = self._default_policy_actions[0].clone()  # full commands + upper actions
        processed[3] = (h_cmd - float(self.config.obs.homie_default_height)) / cmd_scale
        processed[5] = p_cmd / cmd_scale
        n_lower = self._num_lower_dof
        n_upper_non_prim = self._num_non_primitive_dof - n_lower  # 14 arm dofs
        for k in range(n_upper_non_prim):
            j = n_lower + k
            processed[n_cmd + k] = (q_tgt[j] - default[j]) / scale
        # 2026-09-30 negative control grasp_init_debug_variant=open: fingers
        # commanded (and placed, see _reset_robot_states_callback) at p0.
        finger_act = (
            self._finger_open_action
            if self._grasp_init_debug_variant() == "open"
            else self._finger_closed_action
        )
        for k in range(len(self._primitive_action_map)):
            processed[n_cmd + n_upper_non_prim + k] = finger_act
        # policy action (= delta accumulator) vector: active command slots.
        act = processed[self._homie_active_command_index]
        delta = torch.zeros(len(self._delta_action_indices), device=dev)
        for di, ai in enumerate(self._delta_action_indices.tolist()):
            delta[di] = act[ai]
        clip = float(self.config.get("delta_action_clip", 100.0))
        assert delta.abs().max() <= clip, "grasp_init delta exceeds delta_action_clip"
        # whole-body action vector (lower HOMIE dofs + upper actions).
        lower = (q[:n_lower] - default[:n_lower]) / scale
        whole_body = torch.cat([lower, processed[n_cmd:]])
        homie_cmd = processed[:n_cmd] * cmd_scale
        homie_cmd[3] += float(self.config.obs.homie_default_height)
        t = float(self.config.get("bin_tilt_target_deg", 15.0))
        std = float(self.config.get("bin_tilt_back_std_deg", 10.0))
        self._gi_state = dict(
            q=q,
            root_pos_in_bin=torch.tensor(data["root_pos_in_bin"], device=dev, dtype=torch.float32),
            root_quat_in_bin=torch.tensor(
                data["root_quat_in_bin_wxyz"], device=dev, dtype=torch.float32
            ),
            delta=delta,
            whole_body=whole_body,
            homie_cmd=homie_cmd,
            upright_tilt_potential=math.exp(-((0.0 - t) / std) ** 2),
        )
        logger.info(
            f"grasp_init: loaded {path} (arm targets: {arm_mode}), root_in_bin "
            f"{data['root_pos_in_bin'].round(4).tolist()}, homie height {h_cmd:.3f} "
            f"pitch {p_cmd:.3f}, finger hull dist cm "
            f"{data['finger_hull_dist_cm_pulled'].round(2).tolist()}"
        )
        return self._gi_state

    def _grasp_init_debug_variant(self):
        return str(self.config.get("grasp_init_debug_variant", "") or "")

    def _bin_default_root_state(self, env_ids):
        """(n, 13) default bin root state in world (pos, quat wxyz, vel = 0)."""
        bin_art = self.simulator.scene.articulations["bin"]
        state = bin_art.data.default_root_state[env_ids].clone()
        state[:, :3] += self.simulator.scene.env_origins[env_ids]
        if float(self.config.get("spawn_bin_yaw_range_deg", 0.0)) > 0.0 and len(env_ids) > 0:
            dyaw = self._spawn_bin_yaw_rad[env_ids]
            z = torch.zeros_like(dyaw)
            q_yaw = quat_from_euler_xyz(z, z, dyaw)
            state[:, 3:7] = quat_mul(q_yaw, state[:, 3:7])
        state[:, 7:] = 0.0
        return state

    def _write_bin_default_state(self, env_ids):
        bin_art = self.simulator.scene.articulations["bin"]
        root = torch.zeros(self.num_envs, 13, device=self.device)
        root[env_ids] = self._bin_default_root_state(env_ids)
        self.simulator.set_task_root_state_tensor(env_ids, {"bin": root})
        n_j = bin_art.num_joints
        if n_j > 0:
            jp = torch.zeros(self.num_envs, n_j, device=self.device)
            jp[env_ids] = bin_art.data.default_joint_pos[env_ids]
            jv = torch.zeros_like(jp)
            self.simulator.set_task_dof_state_tensor(
                env_ids, {"bin": (jp, jv, torch.arange(n_j, device=self.device))}
            )
        if float(self.config.get("spawn_bin_yaw_range_deg", 0.0)) > 0.0:
            for sname in (
                "left_hand_frame_transformer",
                "right_hand_frame_transformer",
                "left_hand_bar_frame_transformer",
                "right_hand_bar_frame_transformer",
                "head_target_frame_transformer",
            ):
                if sname in self.simulator.scene.sensors:
                    self.simulator.scene.sensors[sname].update(dt=0.0, force_recompute=True)

    @override
    def _reset_robot_states_callback(self, env_ids, target_states=None):
        super()._reset_robot_states_callback(env_ids, target_states)
        ratio = float(self.config.get("grasp_init_ratio", 0.0))
        if ratio <= 0.0 or target_states is not None or len(env_ids) == 0:
            return
        assert self._tilt_enabled, "grasp_init needs enable_tilt_stage"
        pick = torch.rand(len(env_ids), device=self.device) < ratio
        gi = env_ids[pick]
        if len(gi) == 0:
            return
        self._grasp_init_mask[gi] = True
        st = self._grasp_init_state()
        n = len(gi)
        bin_root = self._bin_default_root_state(gi)
        pos = bin_root[:, :3] + quat_apply(
            bin_root[:, 3:7], st["root_pos_in_bin"][None].expand(n, 3)
        )
        quat = quat_mul(bin_root[:, 3:7], st["root_quat_in_bin"][None].expand(n, 4))
        self.target_robot_root_states[gi, 0:3] = pos
        self.target_robot_root_states[gi, 3:7] = quat  # wxyz (isaaclab write)
        self.target_robot_root_states[gi, 7:13] = 0.0
        q = st["q"][None].expand(n, -1).clone()
        noise = float(self.config.get("grasp_init_joint_noise", 0.0))
        if noise > 0.0:
            idx = torch.cat(
                [self._left_arm_dof_idx, self._right_arm_dof_idx, self.finger_dof_idx]
            )
            q[:, idx] += noise * torch.randn(n, len(idx), device=self.device)
        # 2026-09-30 offline negative controls for the wrap gate (default ""
        # = off): "open" = same arm pose, fingers at p0 (open claw);
        # "raised" = whole robot raised grasp_init_debug_raise (m, 0.03), so
        # the palm is that much above its grasp pose, fingers at p1.
        variant = self._grasp_init_debug_variant()
        if variant in ("open", "raised"):
            lp = self._left_p0 if variant == "open" else self._left_p1
            rp = self._right_p0 if variant == "open" else self._right_p1
            q[:, self._left_hand_dof_idx] = lp
            q[:, self._right_hand_dof_idx] = rp
        if variant == "raised":
            self.target_robot_root_states[gi, 2] += float(
                self.config.get("grasp_init_debug_raise", 0.03)
            )
        elif variant not in ("", "open"):
            raise ValueError(f"unknown grasp_init_debug_variant {variant!r}")
        self.target_robot_dof_state[gi, :, 0] = q
        self.target_robot_dof_state[gi, :, 1] = 0.0

    def _resting_action_state(self):
        """2026-10-07 approach_dr2 fix: precompute the delta-action accumulator
        and whole-body action vector corresponding to `resting_dof_pos` (arms at
        waist with wrist_roll = +-1.5708 rad, fingers open at p0, HOMIE at
        default height/zero vel). Without this, DeltaActionBase resets
        `_delta_actions` to 0.0 (= default_dof_pos = 0.0 on all 14 arm DOFs),
        which yanks both wrists by 90 deg on Step 1 of every STAGE_APPROACH
        episode."""
        if getattr(self, "_rest_act_state", None) is not None:
            return self._rest_act_state
        dev = self.device
        q_rest = torch.tensor(
            list(self.config.resting_dof_pos), device=dev, dtype=torch.float32
        )
        scale = float(self.config.robot.control.action_scale)
        default = self.default_dof_pos[0]
        n_cmd = self._num_homie_commands
        n_lower = self._num_lower_dof
        n_upper_non_prim = self._num_non_primitive_dof - n_lower  # 14 arm dofs
        processed = self._default_policy_actions[0].clone()
        for k in range(n_upper_non_prim):
            j = n_lower + k
            processed[n_cmd + k] = (q_rest[j] - default[j]) / scale
        finger_open_act = -1.0 / scale
        for k in range(len(self._primitive_action_map)):
            processed[n_cmd + n_upper_non_prim + k] = finger_open_act
        act = processed[self._homie_active_command_index]
        delta = torch.zeros(len(self._delta_action_indices), device=dev)
        arm_delta_cols = []
        upper_delta_cols = []
        for di, ai in enumerate(self._delta_action_indices.tolist()):
            delta[di] = act[ai]
            orig_idx = int(self._homie_active_command_index[ai])
            if n_cmd <= orig_idx < n_cmd + n_upper_non_prim:
                arm_delta_cols.append(di)
                upper_delta_cols.append(di)
            elif orig_idx >= n_cmd + n_upper_non_prim:
                upper_delta_cols.append(di)
        lower = (q_rest[:n_lower] - default[:n_lower]) / scale
        whole_body = torch.cat([lower, processed[n_cmd:]])
        cmd_scale = float(self.config.homie_command_scale)
        homie_cmd = processed[:n_cmd] * cmd_scale
        homie_cmd[3] += float(self.config.obs.homie_default_height)
        self._rest_act_state = dict(
            delta=delta,
            whole_body=whole_body,
            homie_cmd=homie_cmd,
            arm_delta_cols=torch.tensor(arm_delta_cols, device=dev, dtype=torch.long),
            upper_delta_cols=torch.tensor(upper_delta_cols, device=dev, dtype=torch.long),
        )
        return self._rest_act_state

    def _apply_resting_action_state(self, s0_ids):
        st = self._resting_action_state()
        self._delta_actions[s0_ids] = st["delta"]
        self._last_delta_actions[s0_ids] = 0.0
        wb = st["whole_body"]
        if self.actions.shape[1] == wb.numel():
            self.actions[s0_ids] = wb
            self.last_actions[s0_ids] = wb
            self.actions_after_delay[s0_ids] = wb
            if hasattr(self, "action_queue"):
                self.action_queue[s0_ids] = wb
        self._homie_actions[s0_ids] = wb[: self._num_lower_dof]
        self._homie_commands[s0_ids] = st["homie_cmd"]
        self._homie_commands_unclipped[s0_ids] = st["homie_cmd"]
        self._last_homie_commands[s0_ids] = st["homie_cmd"]

    @override
    def reset_envs_idx(self, env_ids, target_states=None, target_buf=None):
        super().reset_envs_idx(env_ids, target_states, target_buf)
        if len(env_ids) > 0 and bool(self.config.get("reset_delta_actions_to_resting", False)):
            s0 = env_ids[
                (self.stage_buf[env_ids] == self.STAGE_APPROACH)
                & ~self._grasp_init_mask[env_ids]
            ]
            if len(s0) > 0:
                self._apply_resting_action_state(s0)
        # 2026-09-30 overhand grasp_init: after the base classes put the
        # stage-0 envs in APPROACH, move the grasp-init subset to TILT and make
        # every holding-related buffer consistent with the canned grasp.
        if len(env_ids) > 0:
            gi = env_ids[self._grasp_init_mask[env_ids]]
            if len(gi) > 0:
                self._apply_grasp_init_task_state(gi)

    def _apply_grasp_init_task_state(self, gi):
        st = self._grasp_init_state()
        self.set_to_stage(gi, torch.full_like(gi, self.STAGE_TILT))
        # Delta accumulator = canned arm targets / fingers closed / HOMIE
        # height+pitch; zero last increment (delta-rate penalty + obs).
        self._delta_actions[gi] = st["delta"]
        self._last_delta_actions[gi] = 0.0
        wb = st["whole_body"]
        if self.actions.shape[1] == wb.numel():
            self.actions[gi] = wb
            self.last_actions[gi] = wb
            self.actions_after_delay[gi] = wb
            if hasattr(self, "action_queue"):
                self.action_queue[gi] = wb
        self._homie_actions[gi] = wb[: self._num_lower_dof]
        self._homie_commands[gi] = st["homie_cmd"]
        self._homie_commands_unclipped[gi] = st["homie_cmd"]
        self._last_homie_commands[gi] = st["homie_cmd"]
        # Grasp bookkeeping: gate counted as passed, grip fresh, TILT fresh.
        self._grasp_valid_steps[gi] = int(self.config.get("grasp_transition_hold_steps", 8))
        self._push_invalid_grasp_steps[gi] = 0
        self._tilt_hold_steps[gi] = 0
        self._left_finger_close_latched[gi] = False
        self._right_finger_close_latched[gi] = False
        self._prev_tilt_potential[gi] = st["upright_tilt_potential"]
        self._initial_bin_pos_xy[gi] = self._bin_default_root_state(gi)[:, :2]
        self._initial_bar_pos_xy[gi] = self._bar_mid_xy()[gi]

    def _grasp_init_debug_report(self, step, ids):
        """Offline acceptance printout (grasp_init_debug_steps); logging only."""
        ovh = self._overhand_finger_terms() if self._overhand_mode() else None
        ld, rd = self._hand_bar_distances()
        valid = self._grasp_pose_valid_mask()
        grip = self._grip_maintained_mask()
        theta = self._bin_tilt_back_deg()
        lc, rc = self._finger_closure_fractions()
        root_z = self.simulator.robot_root_states[:, 2]

        def s(x):
            x = x[ids].float()
            return f"mean {x.mean():.4f} min {x.min():.4f} max {x.max():.4f}"

        lines = [
            f"[grasp_init_debug] episode_step={step} n={len(ids)}",
            f"  stage==TILT frac {(self.stage_buf[ids] == self.STAGE_TILT).float().mean():.3f}"
            f"  stages {self.stage_buf[ids].tolist()}",
            f"  palm->bar dist L {s(ld)} | R {s(rd)}"
            f"  (grasp_bar_valid_dist {float(self.config.get('grasp_bar_valid_dist', 0.03))})",
            f"  grasp_pose_valid frac {valid[ids].float().mean():.3f}"
            f"  grip_maintained frac {grip[ids].float().mean():.3f}",
            f"  closure L {s(lc)} | R {s(rc)}",
            f"  bin tilt_back deg {s(theta)}  pelvis z {s(root_z)}",
        ]
        # 2026-09-30: palm height above the bar axis (bin frame) + readiness.
        pdl, pdr = self._palm_bar_deltas_bin()
        la, ra = self._palm_above_bar_masks()
        lr, rr = self._finger_close_ready()
        lines.append(
            f"  palm dz cm L {s(100 * pdl[:, 2])} | R {s(100 * pdr[:, 2])}"
            f"  (grasp_palm_min_dz {self._palm_min_dz()}) above L {la[ids].float().mean():.3f}"
            f" R {ra[ids].float().mean():.3f}  finger_close_ready L {lr[ids].float().mean():.3f}"
            f" R {rr[ids].float().mean():.3f}"
        )
        # 2026-09-30: grasp-point target (grasp_bar_reward_target="grasp_target").
        lgp, rgp = self._palm_grasp_points()
        lgt, rgt = self._grasp_point_targets()
        q_inv = quat_inv(self.simulator.get_task_root_state("bin")[:, 3:7])
        tl = quat_apply(q_inv, lgt - self._left_bar_raw_pos())
        tr = quat_apply(q_inv, rgt - self._right_bar_raw_pos())
        lines.append(
            f"  grasp_target dist cm L {s(100 * (lgt - lgp).norm(dim=-1))} | R {s(100 * (rgt - rgp).norm(dim=-1))}"
            f"  target-bar (bin) cm L y {s(100 * tl[:, 1])} z {s(100 * tl[:, 2])}"
            f" | R y {s(100 * tr[:, 1])} z {s(100 * tr[:, 2])}"
        )
        if ovh is not None:
            names = ("index", "middle", "thumb")
            for side, (score, dsurf_tip, dsurf_seg, hand_score, wrapped) in ovh.items():
                slot, below, thumb_front, prox, dy, dz = self._ovh_last_sub[side]
                lines.append(
                    f"  {side}: hand_score {s(hand_score)} wrapped frac "
                    f"{wrapped[ids].float().mean():.3f} | variants "
                    + " ".join(
                        f"{m}={v[ids].mean():.3f}(min {v[ids].min():.3f})"
                        for m, v in self._ovh_last_variants[side].items()
                    )
                )
                for k, fname in enumerate(names):
                    lines.append(
                        f"    {fname:6s} dsurf_seg_cm mean {100 * dsurf_seg[ids, k].mean():.3f} "
                        f"max {100 * dsurf_seg[ids, k].max():.3f} | dsurf_tip_cm "
                        f"{100 * dsurf_tip[ids, k].mean():.3f} | score {score[ids, k].mean():.3f}"
                        f" slot {slot[ids, k].mean():.3f} below {below[ids, k].mean():.3f}"
                        f" thumb_front {thumb_front[ids, k].mean():.3f} prox {prox[ids, k].mean():.3f}"
                        f" | tip dy {100 * dy[ids, k].mean():.2f} dz {100 * dz[ids, k].mean():.2f} cm"
                    )
        print("\n".join(lines), flush=True)

    def _grasp_init_debug_csv_row(self):
        """2026-09-30 video aid (grasp_init_debug_csv = path, default off):
        append one row per step for envs 0..grasp_init_debug_csv_envs-1 with
        the grasp-gate quantities (burned into / shipped next to the mp4)."""
        path = str(self.config.get("grasp_init_debug_csv"))
        n = min(int(self.config.get("grasp_init_debug_csv_envs", 2)), self.num_envs)
        cols = (
            "sim_step,env,episode_step,stage,grasp_init,hand_score_L,hand_score_R,"
            "wrapped_L,wrapped_R,grasp_pose_valid,grip_maintained,palm_bar_dist_L,"
            "palm_bar_dist_R,closure_L,closure_R,bin_tilt_back_deg,pelvis_z,"
            "invalid_grasp_steps,reset"
        )
        # 2026-09-30 (reach_grasp1 video): extended per-hand diagnostics,
        # appended after the original columns. Positions in the BIN frame
        # relative to that hand's bar point (x along the bar, y horizontal
        # across it: - = behind/lid side, z up); tips = the overhand score's
        # tip points. palm_rerr = rotation vector (deg) of target^-1 * palm,
        # in the target frame; palm_fwd / palm_nrm = palm-link +X (finger
        # direction) and palm-face contact axis, expressed in the bin frame.
        fingers = ("index", "middle", "thumb")
        ext = []
        for s in ("L", "R"):
            ext += [f"lid_{s}", f"palm_twist_deg_{s}", f"palm_down_align_{s}"]
            ext += [f"palm_rerr_{a}_{s}" for a in "xyz"]
            ext += [f"palm_gp_{a}_{s}" for a in "xyz"]
            ext += [f"palm_fwd_{a}_{s}" for a in "xyz"]
            ext += [f"palm_nrm_{a}_{s}" for a in "xyz"]
            for fn in fingers:
                ext += [
                    f"{fn}_{k}_{s}"
                    for k in ("dseg", "tipx", "tipy", "tipz", "slot", "below", "front", "prox", "score")
                ]
        # 2026-10-01 (default False): per hand link |F| (N) against the bin,
        # bin-frame dy/dz from that hand's bar point and the lid-region flag
        # of _lid_contact_per_hand, to see WHICH links make lid contact.
        lid_links = bool(self.config.get("grasp_init_debug_csv_lid_links", False))
        if lid_links:
            self._init_hand_link_indices()
            sim_names = list(self.simulator.body_names)
            for s, side in (("L", "left"), ("R", "right")):
                for i in self._hand_link_sim_idx[side].tolist():
                    nm = sim_names[i].replace(f"{side}_hand_", "")
                    ext += [f"lk_{nm}_{s}_{k}" for k in ("F", "dy", "dz", "inlid")]
        # 2026-10-01 tilt2 video aid (default False): hand moment about the
        # wheel contact line (Nm, + tips back), total hand-on-bin |F| (N),
        # body (non-hand/wrist/foot) bin contact (N) and the physical HOMIE
        # height / pitch / vx commands (squat + lean observability).
        tilt_cols = bool(self.config.get("grasp_init_debug_csv_tilt", False))
        if tilt_cols:
            ext += ["tilt_moment_Nm", "hand_force_N", "body_bin_contact_N",
                    "homie_height_cmd", "homie_pitch_cmd", "homie_vx_cmd", "homie_height_cmd_raw",
                    "bin_mass_kg"]
            # 2026-10-01 tilt3: normal-only / friction-only hand moments (the
            # friction ones NaN unless task.target_obj_contact_track_friction)
            # and the wrist links' contact |F| / moment (not in either sum).
            ext += ["tilt_moment_normal_Nm", "tilt_moment_fric_Nm", "fric_force_N",
                    "wrist_contact_N", "wrist_moment_Nm"]
        # 2026-10-02 tilt5 video aid (default False): bin-mass curriculum level
        # (global, and the level this env's episode was sampled at), WHOLE-bin
        # mass (all bin bodies incl. wheels, kg, read back from PhysX) and the
        # per-link |F| (N) of every link summed by _body_bin_contact_force, to
        # see WHICH body part (knee/shin vs torso vs elbow) touches the bin.
        body_links = bool(self.config.get("grasp_init_debug_csv_body_links", False))
        if body_links:
            self._body_bin_contact_force()  # builds self._body_contact_idx
            bl_names = list(self.simulator.body_names)
            ext += ["bin_level_kg", "bin_ep_level_kg", "bin_total_kg"]
            ext += [f"bc_{bl_names[i]}" for i in self._body_contact_idx.tolist()]
        cols = cols + "," + ",".join(ext)
        if not getattr(self, "_gi_csv_started", False):
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, "w") as f:
                f.write(cols + "\n")
            self._gi_csv_started = True
            self._gi_csv_step = 0
        self._gi_csv_step += 1
        ovh = None
        if self._overhand_mode():
            ovh = self._overhand_finger_terms()
            hs_l, hs_r = ovh["left"][3], ovh["right"][3]
            w_l, w_r = ovh["left"][4], ovh["right"][4]
        else:
            hs_l, hs_r = self._wrap_hand_scores()
            w_l = w_r = torch.zeros_like(hs_l, dtype=torch.bool)
        ld, rd = self._hand_bar_distances()
        valid = self._grasp_pose_valid_mask()
        grip = self._grip_maintained_mask()
        lc, rc = self._finger_closure_fractions()
        theta = self._bin_tilt_back_deg()
        root_z = self.simulator.robot_root_states[:, 2]
        # --- extended diagnostics ---
        lid = self._lid_contact_per_hand()
        twist = self._palm_twist_errors()
        align = self._palm_down_alignments()
        gp = self._palm_grasp_points()
        contact_axes = self._palm_contact_axes()
        q_bin_inv = quat_inv(self.simulator.get_task_root_state("bin")[:, 3:7])
        targets = (
            xyzw_to_wxyz(self._left_handle_target_rot()),
            xyzw_to_wxyz(self._right_handle_target_rot()),
        )
        ext_vals = []  # list of (N, k) tensors, in column order
        for h, (side, bar, palm_idx) in enumerate(
            (
                ("left", self._left_bar_raw_pos(), self.left_palm_idx),
                ("right", self._right_bar_raw_pos(), self.right_palm_idx),
            )
        ):
            palm_q = xyzw_to_wxyz(self.simulator._rigid_body_rot[:, palm_idx, :])
            q_err = quat_mul(quat_inv(targets[h]), palm_q)
            q_err = torch.where(q_err[:, :1] < 0, -q_err, q_err)
            v = q_err[:, 1:]
            vn = v.norm(dim=-1, keepdim=True).clamp(min=1e-9)
            ang = 2.0 * torch.atan2(vn, q_err[:, :1])
            rerr = torch.rad2deg(v / vn * ang)
            gp_b = quat_apply(q_bin_inv, gp[h] - bar)
            fwd_b = quat_apply(
                q_bin_inv,
                quat_apply(palm_q, torch.tensor([[1.0, 0.0, 0.0]], device=self.device).expand(self.num_envs, 3)),
            )
            nrm_b = quat_apply(q_bin_inv, quat_apply(palm_q, contact_axes[h]))
            ext_vals += [
                lid[h][:, None], torch.rad2deg(twist[h])[:, None], align[h][:, None],
                rerr, gp_b, fwd_b, nrm_b,
            ]
            if ovh is not None:
                score, _, dseg, _, _ = ovh[side]
                slot, below, front, prox, _, _ = self._ovh_last_sub[side]
                tip = self._ovh_last_tip[side]
                for k in range(3):
                    ext_vals.append(
                        torch.stack(
                            [dseg[:, k], tip[:, k, 0], tip[:, k, 1], tip[:, k, 2],
                             slot[:, k], below[:, k], front[:, k], prox[:, k], score[:, k]],
                            dim=-1,
                        )
                    )
            else:
                ext_vals.append(torch.full((self.num_envs, 27), float("nan"), device=self.device))
        if lid_links:
            forces = self.simulator.object_to_hand_contact_forces[:, 0, :, :].norm(dim=-1)
            dy_max = float(self.config.get("grasp_lid_dy_max", -0.024))
            dz_min = float(self.config.get("grasp_lid_dz_min", 0.0))
            for side, bar in (
                ("left", self._left_bar_raw_pos()),
                ("right", self._right_bar_raw_pos()),
            ):
                d = self._bar_frame_deltas(self._hand_link_sim_idx[side], bar)
                fl = forces[:, self._hand_link_contact_idx[side]]
                inl = ((d[..., 1] < dy_max) & (d[..., 2] > dz_min)).float()
                ext_vals.append(
                    torch.stack([fl, d[..., 1], d[..., 2], inl], dim=-1).reshape(self.num_envs, -1)
                )
        if tilt_cols:
            m_csv, f_csv = self._bin_tilt_moment_terms()
            # _homie_commands columns: vx, vy, yaw, height, roll, pitch, yaw_t
            # (clip_homie_* threshold order in homie_base). height_raw = before
            # the clip_homie_*_height_threshold clamp (saturation check).
            hc = self.get_physical_homie_commands()
            h_raw = self._homie_commands_unclipped[:, 3]
            # 2026-10-01 tilt3: bin body mass READ BACK from PhysX (checks
            # that _apply_bin_mass_curriculum's set_masses took effect).
            art = self.simulator.scene.articulations["bin"]
            bi = list(art.body_names).index("body")
            m_rb = art.root_physx_view.get_masses()[:, bi].to(self.device).float()
            ext_vals.append(
                torch.stack(
                    [m_csv, f_csv, self._body_bin_contact_force(), hc[:, 3], hc[:, 5], hc[:, 0], h_raw, m_rb],
                    dim=-1,
                )
            )
            # 2026-10-01 tilt3: normal-only and friction-only (RAW sign, i.e.
            # friction_forces_w as reported) hand moments, so both sign
            # conventions can be checked offline against the static limit.
            self._init_hand_link_indices()
            f_n = self.BIN_TILT_MOMENT_FORCE_SIGN * self.simulator.object_to_hand_contact_forces[:, 0, :, :]
            m_n = self._bin_hand_moment(f_n)
            fr_raw = getattr(self.simulator, "object_to_hand_friction_forces", None)
            if fr_raw is not None:
                fr_raw = fr_raw[:, 0, :, :]
                m_f = self._bin_hand_moment(fr_raw)
                f_f = fr_raw.norm(dim=-1).sum(dim=-1)
            else:
                m_f = torch.full_like(m_n, float("nan"))
                f_f = torch.full_like(m_n, float("nan"))
            w_f, w_m = self._wrist_bin_contact_terms()
            ext_vals.append(torch.stack([m_n, m_f, f_f, w_f, w_m], dim=-1))
        if body_links:  # 2026-10-02 tilt5 video aid (same order as the names)
            lvl_g = float(getattr(self, "bin_mass_level_kg", float("nan")))
            art_b = self.simulator.scene.articulations["bin"]
            m_tot = art_b.root_physx_view.get_masses().sum(dim=-1).to(self.device).float()
            bc = self.simulator.contact_forces[:, self._body_contact_idx, :].norm(dim=-1).float()
            ext_vals.append(
                torch.cat(
                    [
                        torch.full((self.num_envs, 1), lvl_g, device=self.device),
                        self._bin_mass_ep_level.float()[:, None],
                        m_tot[:, None],
                        bc,
                    ],
                    dim=-1,
                )
            )
        ext_t = torch.cat(ext_vals, dim=-1)
        rows = []
        for e in range(n):
            vals = [
                self._gi_csv_step, e, int(self.episode_length_buf[e]), int(self.stage_buf[e]),
                int(self._grasp_init_mask[e]), f"{float(hs_l[e]):.4f}", f"{float(hs_r[e]):.4f}",
                int(w_l[e]), int(w_r[e]), int(valid[e]), int(grip[e]),
                f"{float(ld[e]):.4f}", f"{float(rd[e]):.4f}", f"{float(lc[e]):.4f}",
                f"{float(rc[e]):.4f}", f"{float(theta[e]):.3f}", f"{float(root_z[e]):.4f}",
                int(self._push_invalid_grasp_steps[e]), int(self.reset_buf[e]),
            ]
            vals += [f"{x:.4f}" for x in ext_t[e].tolist()]
            rows.append(",".join(str(v) for v in vals))
        with open(path, "a") as f:
            f.write("\n".join(rows) + "\n")

    @override
    def _compute_reward(self):
        """2026-10-02 rewcsv: play-analysis aid (grasp_init_debug_csv_rewards =
        path, default off -> plain super()). Logs the EXACT per-step reward of
        every term (incl. termination) as the delta of episode_sums across
        super()._compute_reward(), so no reward function is re-evaluated
        (potential-based terms such as bin_tilt_progress keep their state).
        Written here -- after _check_termination, before reset_envs_idx --
        so reset / timeout / completed and the episode step belong to the same
        step as the reward (the grasp_init_debug_csv row is written earlier in
        the step, from _update_custom_logs, and repeated on reset steps)."""
        path = self.config.get("grasp_init_debug_csv_rewards", None)
        if not path:
            return super()._compute_reward()
        before = {k: v.clone() for k, v in self.episode_sums.items()}
        out = super()._compute_reward()
        self._rew_csv_row(str(path), before)
        return out

    def _rew_csv_row(self, path, before):
        """2026-10-02 rewcsv: append one row per step for envs
        0..grasp_init_debug_csv_rewards_envs-1 (fallback grasp_init_debug_csv_envs,
        then 2): step/stage/termination flags, bin tilt (deg), hand moment
        (Nm, + tips back, same as the bin_tilt_moment reward), hand |F| (N),
        grip, tilt-hold counter, sampled bin body mass (kg), r_total (= rew_buf)
        and r_<term> for every reward term (already x dt and x scale, i.e. the
        per-step values PPO sees)."""
        n = min(
            int(self.config.get(
                "grasp_init_debug_csv_rewards_envs", self.config.get("grasp_init_debug_csv_envs", 2)
            )),
            self.num_envs,
        )
        names = list(self.episode_sums.keys())
        if not getattr(self, "_rew_csv_started", False):
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            head = (
                "sim_step,env,episode_step,stage,time_in_stage,reset,timeout,completed,"
                "bin_tilt_back_deg,tilt_moment_Nm,hand_force_N,grip,tilt_hold_steps,bin_mass_kg,r_total"
            )
            with open(path, "w") as f:
                f.write(head + "," + ",".join(f"r_{k}" for k in names) + "\n")
            self._rew_csv_started = True
            self._rew_csv_names = names
            self._rew_csv_step = 0
        self._rew_csv_step += 1
        m_t, f_t = self._bin_tilt_moment_terms()
        cols = [
            self.episode_length_buf.float(), self.stage_buf.float(), self.time_in_stage_buf.float(),
            self.reset_buf.float(), self.time_out_buf.float(), self.current_completed_task_buf.float(),
            self._bin_tilt_back_deg().float(), m_t.float(), f_t.float(),
            self._grip_maintained_mask().float(), self._tilt_hold_steps.float(),
            self._bin_mass_env.float(), self.rew_buf.float(),
        ]
        cols += [(self.episode_sums[k] - before[k]).float() for k in self._rew_csv_names]
        vals = torch.stack([c[:n] for c in cols], dim=-1).cpu().tolist()
        rows = []
        for e in range(n):
            v = vals[e]
            ints = ",".join(str(int(x)) for x in v[:6])
            rest = ",".join(f"{x:.5g}" for x in v[6:])
            rows.append(f"{self._rew_csv_step},{e},{ints},{rest}")
        with open(path, "a") as f:
            f.write("\n".join(rows) + "\n")

    @override
    def step(self, actor_state):
        # 2026-09-30: offline acceptance aid (default False): zero the RL
        # policy's delta increments so the delta accumulator (arm targets,
        # fingers, HOMIE height/pitch) holds its reset value; the HOMIE
        # lower-body actions (last columns) are left untouched.
        if bool(self.config.get("debug_zero_policy_actions", False)):
            actor_state["actions"][:, self._delta_action_indices] = 0.0
        return super().step(actor_state)

    def _reset_bin_states(self, env_ids):
        """Populate privileged-info buffers and cache the initial bin XY pose.

        The bin USD's dynamic parameters (mass, wheel drive, floor friction) are set
        by the spawner. Here we just snapshot the initial XY position of the bin so
        push-progress observations are relative to a stable reference. Randomization
        of these physics params — if we want to expose them to privileged obs — is a
        follow-up task; for now we just copy the constant that the spawner set.
        """
        # Cache the initial XY position of the bin in world frame — used by
        # bin_pose_progress obs + _stage_3_to_complete_condition.
        bin_xy = self.simulator.get_task_root_state("bin")[env_ids, :2].clone()
        self._initial_bin_pos_xy[env_ids] = bin_xy
        self._initial_bar_pos_xy[env_ids] = self._bar_mid_xy()[env_ids]

        # Privileged info values (live per-env bin mass from _bin_mass_env, which
        # defaults to 15.0 when bin_mass_curriculum is off; ground friction; wheel stiffness).
        self._privileged_bin_mass[env_ids] = self._bin_mass_env[env_ids]
        self._privileged_ground_friction[env_ids] = 0.9
        self._privileged_wheel_stiffness[env_ids] = 0.5

    def _store_initial_bin_pos_xy(self, env_ids):
        return self._initial_bin_pos_xy[env_ids]

    def _load_initial_bin_pos_xy(self, env_ids, data):
        self._initial_bin_pos_xy[env_ids] = data
        self._initial_bar_pos_xy[env_ids] = self._bar_mid_xy()[env_ids]
        if float(self.config.get("spawn_bin_yaw_range_deg", 0.0)) > 0.0 and len(env_ids) > 0:
            q_bin = self.simulator.get_task_root_state("bin")[env_ids, 3:7]
            ex = torch.tensor([[1.0, 0.0, 0.0]], device=self.device).expand(len(env_ids), 3)
            axle_w = quat_apply(q_bin, ex)
            push_xy = F.normalize(torch.stack([axle_w[:, 1], -axle_w[:, 0]], dim=-1), dim=-1)
            self.push_axis_world[env_ids, :2] = push_xy
            self.push_axis_world[env_ids, 2] = 0.0

    @override
    def _check_termination(self):
        super()._check_termination()

        not_just_resetted = self.episode_length_buf > 20

        # Distance to bin > 5m — robot must be near-ish to the bin.
        too_far = (self.relative_bin_pos_buf.norm(dim=-1) > 5.0) & not_just_resetted
        self.reset_buf |= too_far

        # DOF overspeed (only after settling).
        dof_overspeed = torch.any(
            torch.abs(self.simulator.dof_vel[:, self._upper_non_finger_dof_idx])
            > self.termination_level * 20.0,
            dim=-1,
        )
        self.reset_buf |= dof_overspeed & not_just_resetted

        grasp_loss_grace = int(self.config.get("push_grasp_loss_grace_steps", 10))
        # 2026-09-29 overhand: grip loss terminates in every grasp-holding
        # stage (TILT + PUSH; == PUSH only when the TILT stage is disabled).
        grip_lost = (
            self._holding_stage_mask()
            & (self._push_invalid_grasp_steps > grasp_loss_grace)
            & not_just_resetted
        )
        self.reset_buf |= grip_lost
        task_failed = too_far | (dof_overspeed & not_just_resetted) | grip_lost

        # 2026-09-29 overhand (B6): bin toppled forward / fallen back past its
        # self-balance angle -> failure (any stage, after settling).
        if bool(self.config.get("bin_topple_terminate", False)):
            theta = self._bin_tilt_back_deg()
            lo = float(self.config.get("bin_topple_min_deg", -8.0))
            hi = float(self.config.get("bin_topple_max_deg", 30.0))
            toppled = ((theta < lo) | (theta > hi)) & not_just_resetted
            self.reset_buf |= toppled
            task_failed = task_failed | toppled
        # With overtime_is_timeout, a task failure is never labelled a timeout
        # even if the stage also ran over this step (old configs: unchanged).
        if self.config.get("overtime_is_timeout", False):
            self.time_out_buf &= ~task_failed
            if not bool(self.config.get("tilt_overtime_is_timeout", True)):
                # 2026-10-03 push3: running out the 4 s TILT clock without
                # passing the TILT->PUSH gate is a failure to tilt (not a
                # harmless timeout), so stalling below 10 deg in TILT cannot
                # farm an infinite-horizon value bootstrap.
                tilt_overtime = (
                    (self.stage_buf == self.STAGE_TILT)
                    & (self.time_in_stage_buf >= self.max_stage_time[self.stage_buf])
                    & ~self.just_completed_task_buf
                )
                self.time_out_buf &= ~tilt_overtime
            if not bool(self.config.get("pre_push_overtime_is_timeout", True)):
                # 2026-10-04 full2: running out the stage clock in any pre-PUSH
                # stage (APPROACH, REACH, GRASP, TILT) is a failure to advance
                # (not a bootstrapped timeout), so stalling in REACH or GRASP
                # cannot farm a free +gamma*V(s) value bootstrap to avoid TILT.
                final_stage = self.STAGE_PUSH if self._tilt_enabled else 3
                pre_push_overtime = (
                    (self.stage_buf < final_stage)
                    & (self.time_in_stage_buf >= self.max_stage_time[self.stage_buf])
                    & ~self.just_completed_task_buf
                )
                self.time_out_buf &= ~pre_push_overtime
        # 2026-09-30 offline acceptance aid (only with grasp_init_debug_steps):
        # report every grasp-init env that terminates and why.
        if self.config.get("grasp_init_debug_steps", None):
            term = self._grasp_init_mask & (self.reset_buf | self.time_out_buf)
            if bool(term.any()):
                ids = term.nonzero(as_tuple=False).flatten()
                print(
                    f"[grasp_init_debug] TERMINATION n={len(ids)} env {ids.tolist()} "
                    f"ep_len {self.episode_length_buf[ids].tolist()} "
                    f"grip_lost {grip_lost[ids].int().tolist()} "
                    f"invalid_steps {self._push_invalid_grasp_steps[ids].tolist()} "
                    f"timeout {self.time_out_buf[ids].int().tolist()} "
                    f"stage {self.stage_buf[ids].tolist()}",
                    flush=True,
                )

    @property
    def ground_height(self):
        return 0.0

    def scene_creation_callback(self, simulator):
        """Register the bin-body contact sensor, retarget the right-hand frame
        transformer to `grasp_target_right` (the simulator initially points both
        hands at `grasp_target_left`, which is `task.target_obj_transform_sub_prim_path`),
        and add a head-target frame transformer for the student obj-pred head.
        """
        # Bin-body contact sensor — reads foot / hand contacts against the bin.
        # Must point at the actual RigidBody prim (`.../bin/body`), not the
        # top-level Xform (`.../bin`) which has no RigidBodyAPI. Cf. drawer
        # env which watches `.../cabinet/Counter093`.
        bin_body_contact_sensor_config = ContactSensorCfg(
            prim_path=f"/World/envs/env_.*/{simulator.task_config.target_obj}/body",
        )
        simulator.scene.sensors["bin_body_contact_sensor"] = ContactSensor(
            bin_body_contact_sensor_config
        )

        # Dedicated WHEEL contact sensor for penalty_bin_wheel_kick (Run 6 bugfix):
        # the penalty must read only the wheels, not the whole bin body. The bin
        # USD exposes wheel_left / wheel_right rigid bodies (see prepare_bin_asset).
        # 2026-09-25: filtered to robot knee/ankle links. Unfiltered
        # net_forces_w included the FLOOR holding the bin up, so the penalty
        # fired every PUSH step and grew when the bin tipped back onto its wheels.
        # 2026-09-27 handlefix4: ONE sensor per wheel. A single sensor over
        # wheel_.* (2 bodies/env) made PhysX reject the per-env leg filters
        # ("expected 4096, found 2048") -> empty force matrix -> penalty was 0.
        for wheel in ("wheel_left", "wheel_right"):
            simulator.scene.sensors[f"bin_{wheel}_contact_sensor"] = ContactSensor(
                ContactSensorCfg(
                    prim_path=f"/World/envs/env_.*/{simulator.task_config.target_obj}/{wheel}",
                    filter_prim_paths_expr=[
                        f"/World/envs/env_.*/Robot/{side}_{link}"
                        for side in ("left", "right")
                        for link in ("knee_link", "ankle_pitch_link", "ankle_roll_link")
                    ],
                )
            )

        # Dedicated BAR contact sensors -- ADDED 2026-08-27, REVERTED
        # 2026-08-29. The original motivation was real: `_hand_contact_counts()`
        # / `grasp_contact_links_required` used to read the framework's
        # generic `object_to_hand_contact_forces` (isaacsim.py, driven by
        # `task_config.target_obj_contact_sub_prim_path="body"`), which
        # reports contact against the ENTIRE bin body -- so a flat palm
        # resting anywhere on the lid satisfied the contact-count gate just
        # as easily as fingers actually wrapped around the bar (found via
        # direct video review, see project memory). Tried fixing it with two
        # task-specific ContactSensors here pointed at grasp_target_left/
        # right, backed by a new collision sphere authored directly onto the
        # USD asset -- this reliably hung `simulation_app.close()` at
        # training shutdown (confirmed 3 times: once burned ~22hrs of idle
        # VM time before being caught, then reproduced twice more in a
        # tightly-bounded retest even after fixing a purpose/visibility
        # mismatch -- guide+invisible vs. the proven default/inherited
        # convention the wheel colliders below use -- that looked like a
        # promising root cause but wasn't the whole story). Not worth
        # further VM time chasing a Kit-internal teardown bug. Reverted the
        # USD collision sphere and both sensors; `_grasp_pose_valid_mask`
        # and every other consumer of the old `_hand_contact_counts()` now
        # use `_hand_bar_distances()` instead -- a pure kinematic
        # palm-to-bar-target distance check, zero new geometry/sensors, zero
        # shutdown-hang risk. See that method's docstring for the full
        # history and every consumer it replaced.
        #
        # left_hand_paths = [
        #     f"/World/envs/env_.*/Robot/{n}"
        #     for n in simulator.robot_config.body_names
        #     if ("left_hand" in n or "right_hand" in n)
        # ]
        # bin_bar_left_contact_sensor_config = ContactSensorCfg(
        #     prim_path=f"/World/envs/env_.*/{simulator.task_config.target_obj}/grasp_target_left",
        #     filter_prim_paths_expr=left_hand_paths,
        # )
        # simulator.scene.sensors["bin_bar_left_contact_sensor"] = ContactSensor(
        #     bin_bar_left_contact_sensor_config
        # )
        # bin_bar_right_contact_sensor_config = ContactSensorCfg(
        #     prim_path=f"/World/envs/env_.*/{simulator.task_config.target_obj}/grasp_target_right",
        #     filter_prim_paths_expr=left_hand_paths,
        # )
        # simulator.scene.sensors["bin_bar_right_contact_sensor"] = ContactSensor(
        #     bin_bar_right_contact_sensor_config
        # )

        # Bimanual handle assignment (Run 16 CROSSING FIX): the simulator
        # (isaacsim.py:1251) creates both hand transformers pointing at
        # `grasp_target_left`. We overwrite BOTH so each palm targets the handle on
        # ITS OWN side — NO cross-body reaching.
        #
        # Geometry (verified by runtime probe): robot faces +X, so its LEFT hand is
        # on the +Y side and RIGHT hand on the -Y side. After the 90deg bin yaw,
        # `grasp_target_right` sits on +Y (near the robot's LEFT hand) and
        # `grasp_target_left` on -Y (near the robot's RIGHT hand). So the handle
        # NAMES are backwards relative to the robot's anatomy. Cross-assign:
        #   left_hand_frame_transformer  -> grasp_target_right  (+Y, robot's left)
        #   right_hand_frame_transformer -> grasp_target_left   (-Y, robot's right)
        # Both authored targets point the hands' local +X finger axis along the
        # bin's lateral direction. Reorient them so +X points inward through the
        # handle face while each mirrored anatomical palm side points bin-down.
        # At the initial 90-degree bin yaw these become world Rx(+90) for the
        # left hand and Rx(-90) for the right hand. Both authored frames have
        # local +Z along the bin push direction, so negative local Z provides
        # a shared stand-off away from the bin without changing orientation.
        obj = simulator.task_config.target_obj
        target_standoff = float(self.config.get("grasp_target_standoff", 0.04))
        target_height_offset = float(
            self.config.get("grasp_target_height_offset", 0.01)
        )
        target_position_offset = (
            0.0,
            target_height_offset,
            -target_standoff,
        )
        left_hand_frame_transformer_config = FrameTransformerCfg(
            prim_path="/World/envs/env_.*/Robot/left_hand_palm_link",
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path=f"/World/envs/env_.*/{obj}/grasp_target_right",
                    offset=OffsetCfg(
                        pos=target_position_offset,
                        # 2026-09-28 handlefix5: configurable (front hook).
                        rot=tuple(self.config.get(
                            "grasp_target_rot_left",
                            [0.70710678, 0.0, -0.70710678, 0.0])),
                    ),
                ),
            ],
        )
        simulator.scene.sensors["left_hand_frame_transformer"] = FrameTransformer(
            left_hand_frame_transformer_config
        )
        right_hand_frame_transformer_config = FrameTransformerCfg(
            prim_path="/World/envs/env_.*/Robot/right_hand_palm_link",
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path=f"/World/envs/env_.*/{obj}/grasp_target_left",
                    offset=OffsetCfg(
                        pos=target_position_offset,
                        rot=tuple(self.config.get(
                            "grasp_target_rot_right",
                            [0.0, -0.70710678, 0.0, -0.70710678])),
                    ),
                ),
            ],
        )
        simulator.scene.sensors["right_hand_frame_transformer"] = FrameTransformer(
            right_hand_frame_transformer_config
        )

        # RAW (no standoff/height offset) bar-target frame transformers,
        # 2026-08-29: the two transformers above return a pre-contact HOVER
        # point (offset by grasp_target_standoff=4cm + grasp_target_height_
        # offset=1cm from the true bar), designed for REACH-stage approach
        # guidance -- reusing them for grasp VALIDITY conflated "close to
        # the hover point" with "close to the bar itself" (found reviewing
        # _hand_bar_distances with the user). These two track
        # grasp_target_left/right's raw world pose directly (no offset), so
        # _hand_bar_distances can measure genuine palm-to-bar distance for
        # the validity gate, while REACH-stage guidance keeps using the
        # hover-point transformers above unchanged. Same CROSSING FIX
        # assignment (left hand -> grasp_target_right, right hand ->
        # grasp_target_left) -- see the comment above.
        left_hand_bar_frame_transformer_config = FrameTransformerCfg(
            prim_path="/World/envs/env_.*/Robot/left_hand_palm_link",
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path=f"/World/envs/env_.*/{obj}/grasp_target_right",
                ),
            ],
        )
        simulator.scene.sensors["left_hand_bar_frame_transformer"] = FrameTransformer(
            left_hand_bar_frame_transformer_config
        )
        right_hand_bar_frame_transformer_config = FrameTransformerCfg(
            prim_path="/World/envs/env_.*/Robot/right_hand_palm_link",
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path=f"/World/envs/env_.*/{obj}/grasp_target_left",
                ),
            ],
        )
        simulator.scene.sensors["right_hand_bar_frame_transformer"] = FrameTransformer(
            right_hand_bar_frame_transformer_config
        )

        # Head-target frame transformer — used by the student's object-pos aux
        # head. Points at grasp_target_left (an arbitrary choice; the student
        # only needs one anchor point on the bin).
        head_target_config = FrameTransformerCfg(
            prim_path="/World/envs/env_.*/Robot/head_link",
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path=simulator.scene.sensors["left_hand_frame_transformer"]
                    .cfg.target_frames[0].prim_path
                ),
            ],
        )
        simulator.scene.sensors["head_target_frame_transformer"] = FrameTransformer(
            head_target_config
        )
