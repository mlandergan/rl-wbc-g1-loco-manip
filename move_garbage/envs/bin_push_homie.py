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


class BinPushHomie(
    StagedTaskBase,
    DeltaActionBase,
    WarpedActionBase,
    HomieBase,
    FingerPrimitiveBase,
):
    """Bimanual bin-push task with a 4-stage curriculum."""

    STAGE_APPROACH = 0
    STAGE_REACH = 1
    STAGE_GRASP = 2
    STAGE_PUSH = 3

    # ================================================================
    # SETUP
    # ================================================================

    def __init__(self, config, device):
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
        for stage in range(4):
            self.log_dict[f"stage_{stage}_frac"] = (self.stage_buf == stage).float().mean()

        left_palm, right_palm = self._palm_grasp_points()
        left_delta = self._left_handle_target() - left_palm
        right_delta = self._right_handle_target() - right_palm
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
        approach_speed_ready = root_speed < float(
            self.config.get("stage_advance_vel", 0.2)
        )
        approach_arms_ready = upper_deviation < 0.3
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
        self.log_dict["approach_transition_ready_frac_s0"] = (
            approach_stance_ready & approach_speed_ready & approach_arms_ready
        )[approach_mask].float().mean()

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
        bin_quat_xyzw = self.simulator.get_task_root_state("bin")[:, 3:7]
        bin_up = quat_apply(
            xyzw_to_wxyz(bin_quat_xyzw),
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

    def _bin_push_progress(self):
        """Bin displacement along pushAxisWorld (scalar per env, meters)."""
        current_xy = self.simulator.get_task_root_state("bin")[:, :2]
        delta_xy = current_xy - self._initial_bin_pos_xy
        axis_xy = self.push_axis_world[:, :2]
        axis_norm = axis_xy.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        axis_unit = axis_xy / axis_norm
        return (delta_xy * axis_unit).sum(dim=-1)

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
        left_close = (
            left_palm - self._left_handle_target()
        ).norm(dim=-1) < transition_dist
        right_close = (
            right_palm - self._right_handle_target()
        ).norm(dim=-1) < transition_dist
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
        left_delta = self._left_handle_target() - left_palm
        right_delta = self._right_handle_target() - right_palm

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
            return left_r > threshold, right_r > threshold

        # --- Original 5-condition AND-gate (LIVE DEFAULT as of 2026-08-27) ---
        # 2026-08-29: "contact" condition is now bar-distance-based, see
        # _hand_bar_distances.
        bar_valid_dist = float(self.config.get("grasp_bar_valid_dist", 0.03))
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

        left_delta = self._left_handle_target() - left_palm
        right_delta = self._right_handle_target() - right_palm
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
        return left_ready, right_ready

    def _reach_transition_ready_mask(self):
        """Instantaneous REACH readiness before applying the hold duration."""
        left_palm, right_palm = self._palm_grasp_points()
        left_close = (
            left_palm - self._left_handle_target()
        ).norm(dim=-1) < 0.15
        right_close = (
            right_palm - self._right_handle_target()
        ).norm(dim=-1) < 0.15
        left_open = (
            self.simulator.dof_pos[:, self._left_hand_dof_idx] - self._left_p0
        ).abs().mean(dim=-1) < 0.174533
        right_open = (
            self.simulator.dof_pos[:, self._right_hand_dof_idx] - self._right_p0
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
        in_push = self.stage_buf == self.STAGE_PUSH
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
        pass

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
        pre_push = self.stage_buf[env_ids] != self.STAGE_PUSH
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
        return (left_dist < bar_valid_dist) & (right_dist < bar_valid_dist)

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
        force_sustained = self._grasp_force_sustained_mask()[env_ids]
        invalid_push_steps = self._push_invalid_grasp_steps[env_ids]
        self._push_invalid_grasp_steps[env_ids] = torch.where(
            active
            & (self.stage_buf[env_ids] == self.STAGE_PUSH)
            & ~force_sustained,
            invalid_push_steps + 1,
            torch.zeros_like(invalid_push_steps),
        )

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

    def _stage_0_to_1_advance_condition(self):
        # Pelvis near the grasp-relative stop pose (XY), robot mostly stopped,
        # upper body near resting pose. Must match _reward_stop_at_bin's target so
        # the reward and the stage transition agree on WHERE to stop.
        cond = self._standoff_ready_mask()

        # Require a genuine stop, not a cruise-through. Run 6 showed the robot
        # hovering the stop zone at ~0.27 m/s — just under the old 0.3 gate — so
        # it flickered into REACH without settling. Tightened to 0.2.
        root_vel = self.simulator.robot_root_states[:, 7:10].norm(dim=-1)
        cond &= root_vel < float(self.config.get("stage_advance_vel", 0.2))

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
        self._initial_bin_pos_xy[env_ids] = self.simulator.get_task_root_state("bin")[
            env_ids, :2
        ].clone()

    def _stage_3_reward_condition(self):
        return self._grasp_pose_valid_mask()

    def _stage_3_to_complete_condition(self):
        target = float(self.config.get("target_bin_push_distance", 1.0))
        return self._grasp_pose_valid_mask() & (self._bin_push_progress() > target)

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
        target_vel = self.config.get("target_root_vel", 0.3) * target_dir
        vel_reward = self._tracking_reward_util(
            torch.linalg.norm(root_vel - target_vel, dim=-1),
            std=0.15, target=0.0, scale=1.0, offset=0.0,
        )
        # Tight taper measured to the stop target (not bin center): full beyond
        # walk_taper_far, linearly to 0 at walk_taper_near.
        stop_xy = self._stop_target_xy()
        dist = (root_pos[:, :2] - stop_xy).norm(dim=-1)
        near = float(self.config.get("walk_taper_near", 0.4))
        far = float(self.config.get("walk_taper_far", 0.6))
        taper = ((dist - near) / (far - near)).clamp(0.0, 1.0)
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
            self._left_p0,
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
            self._right_p0,
            right_alignment,
        )

    @StagedTaskBase.effective_in_stage([STAGE_APPROACH, STAGE_REACH])
    def _reward_reach_finger_dof_pos_l1(self):
        """Both hands OPEN (fingers near *_p0). Averaged L+R Gaussian on dof deviation."""
        left_diff = self.simulator.dof_pos[:, self._left_hand_dof_idx] - self._left_p0
        right_diff = self.simulator.dof_pos[:, self._right_hand_dof_idx] - self._right_p0
        left = self._tracking_reward_util(left_diff, std=0.3, target=0.0, scale=1.0, offset=0.0).mean(dim=-1)
        right = self._tracking_reward_util(right_diff, std=0.3, target=0.0, scale=1.0, offset=0.0).mean(dim=-1)
        return 0.5 * (left + right)

    @StagedTaskBase.effective_in_stage([STAGE_GRASP, STAGE_PUSH])
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
            return self._tracking_reward_util(
                diff, std=0.3, target=0.0, scale=1.0, offset=0.0
            ).mean(dim=-1)

        def _one_hand_unconditional(hand_dof_idx, p1):
            diff = self.simulator.dof_pos[:, hand_dof_idx] - p1[None, :]
            return self._tracking_reward_util(
                diff, std=0.3, target=0.0, scale=1.0, offset=0.0
            ).mean(dim=-1)

        if self.config.get("enable_pregrasp_curl", False):
            left_frac, right_frac = self._pregrasp_curl_fraction()
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
        return 0.5 * (left + right)

    @StagedTaskBase.effective_in_stage([STAGE_GRASP, STAGE_PUSH])
    def _reward_grasp_target_distance(self):
        """Both palms close to their handle targets. Averaged L+R Gaussian."""
        lp, rp = self._palm_grasp_points()
        ld = torch.norm(self._left_handle_target() - lp, dim=-1)
        rd = torch.norm(self._right_handle_target() - rp, dim=-1)
        left = self._tracking_reward_util(ld, std=0.1, target=0.0, scale=1.0, offset=0.0)
        right = self._tracking_reward_util(rd, std=0.1, target=0.0, scale=1.0, offset=0.0)
        return 0.5 * (left + right)

    @StagedTaskBase.effective_in_stage([STAGE_GRASP, STAGE_PUSH])
    def _reward_grasp_target_anisotropic(self):
        """Axis-aware grasp-point shaping with a wider approach region.

        The hard gate remains 4 cm horizontal and 3 cm vertical. Wider widths
        preserve useful signal at the current 8-11 cm operating point while
        making vertical error more costly than the isotropic reward does.
        """
        lp, rp = self._palm_grasp_points()
        ld = self._left_handle_target() - lp
        rd = self._right_handle_target() - rp
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

        return 0.5 * (_one_hand(ld) + _one_hand(rd))

    @StagedTaskBase.effective_in_stage([STAGE_GRASP, STAGE_PUSH])
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
        left_vert_error = (self._left_handle_target() - lp)[:, 2].abs()
        right_vert_error = (self._right_handle_target() - rp)[:, 2].abs()
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

    @StagedTaskBase.effective_in_stage([STAGE_GRASP, STAGE_PUSH])
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
        left_dist, right_dist = self._hand_bar_distances()
        left = self._tracking_reward_util(left_dist, std=0.1, target=0.0, scale=1.0, offset=0.0)
        right = self._tracking_reward_util(right_dist, std=0.1, target=0.0, scale=1.0, offset=0.0)
        return 0.5 * (left + right)

    @StagedTaskBase.effective_in_stage([STAGE_GRASP, STAGE_PUSH])
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
        left_dist, right_dist = self._hand_bar_distances()
        max_step_error = float(self.config.get("grasp_bar_progress_norm", 0.3))
        valid = (self.episode_length_buf > 1).float()
        left_progress = (
            (self._prev_left_bar_dist - left_dist) / max_step_error * valid
        )
        right_progress = (
            (self._prev_right_bar_dist - right_dist) / max_step_error * valid
        )
        self._prev_left_bar_dist = left_dist.detach().clone()
        self._prev_right_bar_dist = right_dist.detach().clone()
        return 0.5 * (left_progress + right_progress)

    @StagedTaskBase.effective_in_stage([STAGE_GRASP, STAGE_PUSH])
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

        return 0.5 * (left_near * left_force_term + right_near * right_force_term)

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

    @StagedTaskBase.effective_in_stage([STAGE_GRASP, STAGE_PUSH])
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
        if not hasattr(self, "_left_flat_press_mask"):
            g1_hand_links = [
                n for n in self.simulator.robot_config.body_names
                if ("left_hand" in n or "right_hand" in n)
            ]
            fingertip_substrings = ("index_1", "middle_1", "thumb_2")
            self._left_flat_press_mask = [
                i for i, n in enumerate(g1_hand_links)
                if "left_hand" in n and not any(s in n for s in fingertip_substrings)
            ]
            self._right_flat_press_mask = [
                i for i, n in enumerate(g1_hand_links)
                if "right_hand" in n and not any(s in n for s in fingertip_substrings)
            ]
        left_flat_force = forces[:, self._left_flat_press_mask, :].norm(dim=-1).sum(dim=-1)
        right_flat_force = forces[:, self._right_flat_press_mask, :].norm(dim=-1).sum(dim=-1)
        max_force = float(self.config.get("grasp_flat_press_force_max", 5.0))
        left_penalty_term = (left_flat_force / max_force).clamp(max=1.0)
        right_penalty_term = (right_flat_force / max_force).clamp(max=1.0)

        return 0.5 * (left_near * left_penalty_term + right_near * right_penalty_term)

    @StagedTaskBase.effective_in_stage([STAGE_REACH, STAGE_GRASP, STAGE_PUSH])
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

        def _hand_force_reward(contact_mask, rot, axis):
            link_forces = forces[:, contact_mask, :]  # (num_envs, num_links, 3)
            num_links = link_forces.shape[1]
            rot_expand = rot[:, None, :].expand(-1, num_links, -1).reshape(-1, 4)
            force_flat = link_forces.reshape(-1, 3)
            force_palm_frame = quat_apply(quat_inv(rot_expand), force_flat)
            axis_expand = axis[:, None, :].expand(-1, num_links, -1).reshape(-1, 3)
            outward = (force_palm_frame * axis_expand).sum(dim=-1)
            lateral = (force_palm_frame - outward[:, None] * axis_expand).norm(dim=-1)
            reward = (outward - lateral).clamp(min=-10.0, max=10.0)
            return reward.reshape(self.num_envs, num_links).mean(dim=-1)

        left_reward = _hand_force_reward(self._left_contact_mask, left_rot, left_axis)
        right_reward = _hand_force_reward(self._right_contact_mask, right_rot, right_axis)
        reward = left_reward + right_reward

        in_reach = self.stage_buf == self.STAGE_REACH
        return torch.where(in_reach, -1.0 * torch.abs(reward), reward)

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

    @StagedTaskBase.effective_in_stage([STAGE_GRASP, STAGE_PUSH])
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

    @StagedTaskBase.effective_in_stage([STAGE_REACH, STAGE_GRASP, STAGE_PUSH])
    def _reward_palm_down_alignment(self):
        """Align each mirrored palm's contact face with the bin's downward normal."""
        left_alignment, right_alignment = self._palm_down_alignments()
        # Keep the objective signed so dorsal-hand approaches are penalized
        # instead of receiving a smaller positive reward. The minimum term makes
        # the worse hand the bilateral bottleneck without removing each hand's
        # independent rotation signal.
        left_score = left_alignment.clamp(-1.0, 1.0)
        right_score = right_alignment.clamp(-1.0, 1.0)
        return 0.25 * (left_score + right_score) + 0.5 * torch.minimum(
            left_score, right_score
        )

    @StagedTaskBase.effective_in_stage([STAGE_REACH, STAGE_GRASP, STAGE_PUSH])
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
        return 0.5 * (left_score + right_score)

    @StagedTaskBase.effective_in_stage([STAGE_REACH, STAGE_GRASP, STAGE_PUSH])
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

    @StagedTaskBase.effective_in_stage(STAGE_PUSH)
    def _reward_maintain_grasp_while_pushing(self):
        """Reward sustained bimanual bar proximity during the push.

        2026-08-29: bar-distance-based (see _hand_bar_distances) -- reward
        ramps 1.0 at dist=0 down to 0.0 at dist>=grasp_bar_valid_dist,
        inverted from the old count/required linear ramp (higher count was
        better; lower distance is better).
        """
        left_dist, right_dist = self._hand_bar_distances()
        bar_valid_dist = float(self.config.get("grasp_bar_valid_dist", 0.03))
        left_reward = (1.0 - left_dist / bar_valid_dist).clamp(min=0.0, max=1.0)
        right_reward = (1.0 - right_dist / bar_valid_dist).clamp(min=0.0, max=1.0)
        return 0.25 * (left_reward + right_reward) + 0.5 * (
            left_reward * right_reward
        )

    @StagedTaskBase.effective_in_stage(STAGE_PUSH)
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
        target_vel = float(self.config.get("push_target_vel", 0.30))
        vel_reward = (vel_along / target_vel).clamp(min=0.0, max=1.0)

        progress = self._bin_push_progress()                    # meters along axis
        target = float(self.config.get("target_bin_push_distance", 1.0))
        pos_reward = progress.clamp(min=0.0, max=target) / target
        velocity_weight = float(self.config.get("push_velocity_weight", 0.50))
        reward = velocity_weight * vel_reward + (1.0 - velocity_weight) * pos_reward
        return reward
        # --- Original grasp-gated version (superseded 2026-08-29) ---
        # return reward * self._grasp_pose_valid_mask().float()

    @StagedTaskBase.effective_in_stage(STAGE_PUSH)
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
        reward = self._tracking_reward_util(
            fwd_vel, std=0.15, target=0.2, scale=1.0, offset=0.0
        )
        return reward
        # --- Original grasp-gated version (superseded 2026-08-29) ---
        # return reward * self._grasp_pose_valid_mask().float()

    @StagedTaskBase.effective_in_stage(STAGE_PUSH)
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

    @StagedTaskBase.effective_in_stage([STAGE_APPROACH])
    def _reward_penalty_upper_body_non_finger_deviation_l1(self):
        """L1 deviation of non-finger upper-body DoFs from resting pose (arms down while walking)."""
        return torch.abs(
            self.simulator.dof_pos[:, self._upper_non_finger_dof_idx]
            - self.resting_dof_pos[:, self._upper_non_finger_dof_idx]
        ).sum(dim=-1)

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

    @StagedTaskBase.effective_in_stage([STAGE_REACH, STAGE_GRASP])
    def _reward_penalty_not_standing_still(self):
        """Penalize nonzero HOMIE locomotion command while reaching / grasping."""
        return torch.norm(self.get_physical_homie_commands()[:, :3], dim=1)

    @StagedTaskBase.effective_in_stage([STAGE_PUSH])
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
        wheel_contact = self.simulator.scene.sensors[
            "bin_wheel_contact_sensor"
        ].data.net_forces_w
        return wheel_contact.norm(dim=-1).sum(dim=-1)

    @StagedTaskBase.effective_in_stage([STAGE_PUSH])
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
        bin_quat_xyzw = self.simulator.get_task_root_state("bin")[:, 3:7]
        bin_up = quat_apply(xyzw_to_wxyz(bin_quat_xyzw),
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

    @StagedTaskBase.effective_in_stage([STAGE_GRASP, STAGE_PUSH])
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
        self.target_robot_root_states[env_ids, 1:2] = (
            self.env_origins[env_ids, 1:2]
            + torch_rand_float(-0.3, 0.3, (len(env_ids), 1), device=str(self.device))
        )

        # Yaw facing +X (toward the bin). ±15° jitter.
        random_yaw = torch_rand_float(
            -torch.pi / 12,
            torch.pi / 12,
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

        xx, yy = torch.meshgrid(env_ids, self.wrist_dof_idx)
        self.target_robot_dof_state[xx, yy, 0] = torch_rand_float(
            -1.39626, 1.39626,
            (len(env_ids), len(self.wrist_dof_idx)),
            device=str(self.device),
        )

        xx, yy = torch.meshgrid(env_ids, self.finger_dof_idx)
        upper_limit = torch.tensor(
            self.simulator.robot_config.dof_pos_upper_limit_list, device=str(self.device)
        )[None, self.finger_dof_idx]
        lower_limit = torch.tensor(
            self.simulator.robot_config.dof_pos_lower_limit_list, device=str(self.device)
        )[None, self.finger_dof_idx]
        self.target_robot_dof_state[xx, yy, 0] = lower_limit + (
            upper_limit - lower_limit
        ) * torch_rand_float(
            0.0, 1.0, (len(env_ids), len(self.finger_dof_idx)), device=str(self.device)
        )

        self.target_robot_dof_state[env_ids, :, 1] = 0.0

    @override
    def _reset_tasks_callback(self, env_ids):
        super()._reset_tasks_callback(env_ids)
        self._reach_transition_ready_steps[env_ids] = 0
        self._grasp_valid_steps[env_ids] = 0
        self._push_invalid_grasp_steps[env_ids] = 0
        self._left_finger_close_latched[env_ids] = False
        self._right_finger_close_latched[env_ids] = False

    @override
    def _reset_object_states_callback(self, env_ids):
        self._reset_bin_states(env_ids)
        return super()._reset_object_states_callback(env_ids)

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

        # Privileged info fallback values (bin mass, ground friction, wheel stiffness).
        # A follow-up commit will read these from the spawner's per-env DR tape.
        self._privileged_bin_mass[env_ids] = 15.0
        self._privileged_ground_friction[env_ids] = 0.9
        self._privileged_wheel_stiffness[env_ids] = 0.5

    def _store_initial_bin_pos_xy(self, env_ids):
        return self._initial_bin_pos_xy[env_ids]

    def _load_initial_bin_pos_xy(self, env_ids, data):
        self._initial_bin_pos_xy[env_ids] = data

    @override
    def _check_termination(self):
        super()._check_termination()

        not_just_resetted = self.episode_length_buf > 20

        # Distance to bin > 5m — robot must be near-ish to the bin.
        self.reset_buf |= (self.relative_bin_pos_buf.norm(dim=-1) > 5.0) & not_just_resetted

        # DOF overspeed (only after settling).
        dof_overspeed = torch.any(
            torch.abs(self.simulator.dof_vel[:, self._upper_non_finger_dof_idx])
            > self.termination_level * 20.0,
            dim=-1,
        )
        self.reset_buf |= dof_overspeed & not_just_resetted

        grasp_loss_grace = int(self.config.get("push_grasp_loss_grace_steps", 10))
        self.reset_buf |= (
            (self.stage_buf == self.STAGE_PUSH)
            & (self._push_invalid_grasp_steps > grasp_loss_grace)
            & not_just_resetted
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
        bin_wheel_contact_sensor_config = ContactSensorCfg(
            prim_path=f"/World/envs/env_.*/{simulator.task_config.target_obj}/wheel_.*",
        )
        simulator.scene.sensors["bin_wheel_contact_sensor"] = ContactSensor(
            bin_wheel_contact_sensor_config
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
                        rot=(0.70710678, 0.0, -0.70710678, 0.0)
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
                        rot=(0.0, -0.70710678, 0.0, -0.70710678)
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
