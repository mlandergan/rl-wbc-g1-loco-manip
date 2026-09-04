# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import torch
from typing_extensions import override

from gr00t.rl.envs.base_task.finger_primitive_base import FingerPrimitiveBase
from gr00t.rl.envs.legged_base_task.legged_robot_base import LeggedRobotBase
from gr00t.rl.utils.torch_utils import quat_rotate


class HomieBase(LeggedRobotBase):
    def __init__(self, config, device):
        super().__init__(config, device)
        self._homie_history_length = self.config.obs.homie_history_length
        self._num_body_dof = self.config.robot.homie_dof_obs_size
        self._num_lower_dof = self.config.robot.lower_body_actions_dim
        self._num_upper_dof = self.config.robot.upper_body_actions_dim
        self._num_homie_commands = self.config.obs.obs_dims.b_homie_commands
        if self.config.robot.get("use_primitive", False):
            self._num_non_homie_command_actions = 14 + 2
        else:
            self._num_non_homie_command_actions = (
                self.config.robot.actions_dim - self._num_lower_dof
            )  # 14 + 14
        self._homie_commands = torch.zeros(
            self.num_envs, self._num_homie_commands, device=self.device, requires_grad=False
        )  # active homie command
        self._homie_commands_unclipped = torch.zeros(
            self.num_envs, self._num_homie_commands, device=self.device, requires_grad=False
        )
        self._last_homie_commands = torch.zeros(
            self.num_envs, self._num_homie_commands, device=self.device, requires_grad=False
        )
        self._homie_actions = torch.zeros(
            self.num_envs, self._num_lower_dof, device=self.device, requires_grad=False
        )  # lower body dof actions outputed by homie
        self._default_policy_actions = torch.zeros(
            self.num_envs,
            self._num_homie_commands + self._num_non_homie_command_actions,
            device=self.device,
            requires_grad=False,
        )  # full 7 homie commad + non-homie dof actions
        index = []
        for key, value in self.config.obs.homie_command_default.items():
            if key not in self.config.obs.homie_command_keys:
                self._default_policy_actions[:, self.config.obs.homie_command_index[key]] = (
                    torch.tensor(value, device=self.device)
                )
            else:
                index += self.config.obs.homie_command_index[key]
        self._homie_active_command_index = torch.tensor(index, device=self.device, dtype=torch.long)
        self._num_homie_active_commands = len(self._homie_active_command_index)

        # self.last_processed_commands = torch.zeros(self.num_envs, self._num_homie_commands + self._num_non_homie_command_actions, device=self.device, requires_grad=False)

        self._clip_homie_command = self.config.get("clip_homie_command", False)
        self._clip_upper_actions = self.config.get("clip_upper_actions", False)
        if self._clip_homie_command:
            self._homie_command_low_thres = torch.tensor(
                [
                    [
                        -self.config.clip_homie_linvel_x_threshold,
                        -self.config.clip_homie_linvel_y_threshold,
                        -self.config.clip_homie_angvel_threshold,
                        self.config.clip_homie_lower_height_threshold,
                        self.config.clip_homie_torso_roll_lower_threshold,
                        self.config.clip_homie_torso_pitch_lower_threshold,
                        self.config.clip_homie_torso_yaw_lower_threshold,
                    ]
                ],
                device=self.device,
            )
            self._homie_command_high_thres = torch.tensor(
                [
                    [
                        self.config.clip_homie_linvel_x_threshold,
                        self.config.clip_homie_linvel_y_threshold,
                        self.config.clip_homie_angvel_threshold,
                        self.config.clip_homie_upper_height_threshold,
                        self.config.clip_homie_torso_roll_upper_threshold,
                        self.config.clip_homie_torso_pitch_upper_threshold,
                        self.config.clip_homie_torso_yaw_upper_threshold,
                    ]
                ],
                device=self.device,
            )
        if self._clip_upper_actions:
            self.arm_vel_threshold = self.config.get("arm_vel_threshold", 0.2)
            self._clip_upper_actions_threshold = self.config.get(
                "clip_upper_actions_threshold", 0.5
            )

    @override
    def _reset_buffers_callback(self, env_ids, target_buf):
        super()._reset_buffers_callback(env_ids, target_buf)
        self._homie_actions[env_ids, :] = 0.0

    @override
    def step(self, actor_state):
        actions = actor_state["actions"]
        self.processed_commands = (
            self._default_policy_actions.clone()
        )  # [num_envs, num_homie_commands + upper_body_actions (7 + 28)]
        self.processed_commands[:, self._homie_active_command_index] = actions[
            :, : self._num_homie_active_commands
        ]
        # if actions.shape[1] == self._num_homie_active_commands + self._num_lower_dof:
        #     # in case the policy only outputs homie commands
        #     self.processed_commands[:, self._num_homie_commands:] = 0.0
        # else:
        #     # non-homie actions
        #     self.processed_commands[:, self._num_homie_commands:] = actions[:, self._num_homie_active_commands:-self._num_lower_dof]

        self._homie_commands = (
            self.processed_commands[:, : self._num_homie_commands] * self.config.homie_command_scale
        )
        self._homie_commands[:, 3] += self.config.obs.homie_default_height  # height
        self._homie_commands_unclipped[:] = self._homie_commands

        if self._clip_homie_command:
            self._homie_commands = torch.clamp(
                self._homie_commands, self._homie_command_low_thres, self._homie_command_high_thres
            )

        self._homie_actions = actions[:, -self._num_lower_dof :]

        if self._clip_upper_actions:
            raw_arm_commands = self.processed_commands[:, self._num_homie_commands : -2]
            num_lower_dof = self._homie_actions.shape[-1]
            arm_default_pos = self.default_dof_pos[:, num_lower_dof : num_lower_dof + 14]
            dof_pos = self.simulator.dof_pos[:, num_lower_dof : num_lower_dof + 14]
            # dof_vel = self.simulator.dof_vel[:, num_lower_dof:num_lower_dof + 14]
            target_raw = raw_arm_commands * self.config.robot.control.action_scale + arm_default_pos
            target_clipped = torch.clamp(
                target_raw,
                -self._clip_upper_actions_threshold + dof_pos,
                self._clip_upper_actions_threshold + dof_pos,
            )
            self.processed_commands[:, self._num_homie_commands : -2] = (
                target_clipped - arm_default_pos
            ) / self.config.robot.control.action_scale
            # clip_condition = ((dof_vel > self.arm_vel_threshold) | (dof_vel < -self.arm_vel_threshold))
            # clip_value = torch.where(clip_condition, dof_pos, target_raw)
            # self.processed_commands[:, self._num_homie_commands:-2] = (clip_value - arm_default_pos) / self.config.robot.control.action_scale
            # alpha = 0.8
            # self.processed_commands[:, self._num_homie_commands:-2] = alpha * self.processed_commands[:, self._num_homie_commands:-2] + (1 - alpha) * self.last_processed_commands[:, self._num_homie_commands:-2]
            # self.last_processed_commands[:, self._num_homie_commands:-2] = self.processed_commands[:, self._num_homie_commands:-2].clone()
        whole_body_actions = torch.cat(
            [
                self._homie_actions,  # lower-body actions
                self.processed_commands[
                    :, self._num_homie_commands :
                ],  # non-homie upper body actions
            ],
            dim=-1,
        )

        self._pre_physics_step(whole_body_actions)
        self._physics_step()
        self._post_physics_step()

        self._last_homie_commands[:] = self._homie_commands

        return self.obs_buf_dict, self.rew_buf, self.reset_buf, self.extras

    @override
    def _action_backmap(self):
        whole_body_action_backmap = super()._action_backmap()

        return torch.cat(
            [
                torch.zeros(self.num_envs, self._num_homie_active_commands, device=self.device),
                whole_body_action_backmap[:, self._homie_actions.shape[1] :],
            ],
            dim=-1,
        )

    def _reward_penalty_homie_action_rate(self):
        return torch.sum(torch.square(self._last_homie_commands - self._homie_commands), dim=1)

    def _reward_penalty_homie_action_limit(self):
        return torch.sum(torch.square(self._homie_commands_unclipped - self._homie_commands), dim=1)

    def get_physical_homie_commands(self):
        commands = self._homie_commands.clone()
        commands[:, :2] *= 2.0
        commands[:, 2] *= 0.5
        return commands

    @property
    def ground_height(self):
        return 0.0

    def _get_obs_b_homie_commands(self):
        commands = self._homie_commands.clone()
        commands[:, :2] *= 2.0
        commands[:, 2] *= 0.5
        commands[commands[:, :3].norm(dim=-1) < 0.1, :3] = 0.0
        return commands

    def _get_obs_e_homie_dof_pos(self):
        return (
            self.simulator.dof_pos[:, : self._num_body_dof]
            - self.default_dof_pos[:, : self._num_body_dof]
        )

    def _get_obs_f_homie_dof_vel(self):
        return self.simulator.dof_vel[:, : self._num_body_dof]

    def _get_obs_c_homie_base_ang_vel(self):
        return self.base_ang_vel

    def _get_obs_d_homie_projected_gravity(self):
        return self.projected_gravity

    def _get_obs_g_homie_body_actions(self):
        return self._homie_actions

    def _get_obs_a_history_homie(self):
        assert "a_history_homie" in self.config.obs.obs_auxiliary.keys()
        history_config = self.config.obs.obs_auxiliary["a_history_homie"]
        history_tensors = []
        for item in range(self._homie_history_length):
            for key in history_config.keys():
                history_tensor = self.history_handler.query(key)[:, item]
                history_tensors.append(history_tensor)
        return torch.cat(history_tensors, dim=-1)


class TestHomieBase(HomieBase):
    def _reward_test_homie(self):
        root_vel = self.simulator.robot_root_states[:, 7:10]
        ref_root_vel = torch.tensor([[0.2, 0.3, 0.0]], device=self.device).repeat(self.num_envs, 1)
        base_quat = self.simulator.base_quat
        ref_root_vel = quat_rotate(base_quat, ref_root_vel)
        root_height = self.simulator.robot_root_states[:, 2]
        ref_root_height = self.config.ref_height
        return self._tracking_reward_util(
            root_vel - ref_root_vel, std=0.1, target=0.0, scale=1.0, offset=0.0
        ).mean(dim=-1) + self._tracking_reward_util(
            root_height - ref_root_height, std=0.1, target=0.0, scale=1.0, offset=0.0
        )

    def _reward_penalty_upper_body_deviation_l1(self):
        return torch.abs(
            self.simulator.dof_pos[:, self.upper_dof_indices]
            - self.default_dof_pos[:, self.upper_dof_indices]
        ).sum(dim=-1)


class TestHomieWithFingerPrimitive(TestHomieBase, FingerPrimitiveBase):
    def __init__(self, config, device):
        super().__init__(config, device)
        self._left_p0 = torch.tensor(
            self.config.robot.finger_primitive.primitive_action_map.left.pos_0,
            device=self.device,
            requires_grad=False,
        )
        self._left_p1 = torch.tensor(
            self.config.robot.finger_primitive.primitive_action_map.left.pos_1,
            device=self.device,
            requires_grad=False,
        )
        self._right_p0 = torch.tensor(
            self.config.robot.finger_primitive.primitive_action_map.right.pos_0,
            device=self.device,
            requires_grad=False,
        )
        self._right_p1 = torch.tensor(
            self.config.robot.finger_primitive.primitive_action_map.right.pos_1,
            device=self.device,
            requires_grad=False,
        )

        self._left_dof_idx = [
            self.dof_names.index(name)
            for name in self.config.robot.finger_primitive.primitive_action_map.left.dof_names
        ]
        self._right_dof_idx = [
            self.dof_names.index(name)
            for name in self.config.robot.finger_primitive.primitive_action_map.right.dof_names
        ]
        self._upper_non_finger_dof_idx = [
            i
            for i in self.upper_dof_indices
            if i not in self._left_dof_idx and i not in self._right_dof_idx
        ]

        self._target_period = self.config.target_period

    def _reward_penalty_upper_body_non_finger_deviation_l1(self):
        return torch.abs(
            self.simulator.dof_pos[:, self._upper_non_finger_dof_idx]
            - self.default_dof_pos[:, self._upper_non_finger_dof_idx]
        ).sum(dim=-1)

    def _reward_finger_sine_wave(self):
        phase = (
            torch.sin(self.episode_length_buf * 2 * torch.pi * self.dt / self._target_period) + 1.0
        ) / 2
        left_target = torch.lerp(self._left_p0[None, :], self._left_p1[None, :], phase[:, None])
        right_target = torch.lerp(self._right_p0[None, :], self._right_p1[None, :], phase[:, None])

        left_current = self.simulator.dof_pos[:, self._left_dof_idx]
        right_current = self.simulator.dof_pos[:, self._right_dof_idx]

        left_reward = self._tracking_reward_util(
            left_current - left_target, std=0.1, target=0.0, scale=1.0, offset=0.0
        )
        right_reward = self._tracking_reward_util(
            right_current - right_target, std=0.1, target=0.0, scale=1.0, offset=0.0
        )

        return left_reward.mean(dim=-1) + right_reward.mean(dim=-1)

    def _get_obs_phase(self):
        return torch.sin(self.episode_length_buf * 2 * torch.pi * self.dt / self._target_period)[
            :, None
        ]
