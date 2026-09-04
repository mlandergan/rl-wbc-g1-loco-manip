# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import torch
from typing_extensions import override

from gr00t.rl.envs.legged_base_task.legged_robot_base import LeggedRobotBase


class FingerPrimitiveBase(LeggedRobotBase):
    """
    Base class for finger primitive tasks.
    Overrides _apply_force_in_physics_step
    """

    def __init__(self, config, device):
        super().__init__(config, device)

        # parse config to map action to finger dof
        self._num_non_primitive_dof = self.config.robot.finger_primitive.num_non_primitive_dof
        self._primitive_action_map = []

        num_primitive_dof = 0
        primitive_dof_idx = set()
        for (
            action_name,
            primitive_cfg,
        ) in self.config.robot.finger_primitive.primitive_action_map.items():
            self._primitive_action_map.append(
                {
                    "action_name": action_name,
                    "dof_names": primitive_cfg.dof_names,
                    "pos_0": torch.tensor(
                        primitive_cfg.pos_0, device=self.device, requires_grad=False
                    ),
                    "pos_1": torch.tensor(
                        primitive_cfg.pos_1, device=self.device, requires_grad=False
                    ),
                    "inheritrange": primitive_cfg.inheritrange,
                    "dof_idx": [
                        self.simulator.dof_names.index(dof_name)
                        for dof_name in primitive_cfg.dof_names
                    ],
                    "mode": primitive_cfg.mode,
                }
            )
            assert (
                len(primitive_cfg.dof_names) == len(primitive_cfg.pos_0) == len(primitive_cfg.pos_1)
            )
            num_primitive_dof += len(primitive_cfg.dof_names)
            len_old = len(primitive_dof_idx)
            primitive_dof_idx.update(self._primitive_action_map[-1]["dof_idx"])
            assert len(primitive_dof_idx) - len_old == len(
                primitive_cfg.dof_names
            ), "Duplicate dof names in primitive action map"

        assert num_primitive_dof + self._num_non_primitive_dof == self.num_dofs
        self.num_primitive_dof = num_primitive_dof
        primitive_dof_idx = sorted(list(primitive_dof_idx))
        gt_primitive_dof_idx = list(
            range(self._num_non_primitive_dof, self._num_non_primitive_dof + num_primitive_dof)
        )
        assert (
            primitive_dof_idx == gt_primitive_dof_idx
        ), "Primitive dof must be after all non-primitive dof in robot config"

        self._primitive_action_map = sorted(
            self._primitive_action_map, key=lambda x: x["action_name"]
        )

        for i, primitive_cfg in enumerate(self._primitive_action_map):
            print(f"Primitive '{primitive_cfg['action_name']}' mapped to the {i}-th action")

        self.primitive_action_over_limit_buf = torch.zeros(
            self.num_envs, len(self._primitive_action_map), device=self.device
        )

    @override
    def _apply_force_in_physics_step(self):
        if self.config.simulator.config.name == "isaacsim":
            actions_scaled = self.actions_after_delay * self.config.robot.control.action_scale
            combined_target = torch.zeros(
                self.num_envs, self.num_dofs, device=self.device
            )  # num_dofs = 43
            combined_target[:, : self._num_non_primitive_dof] = (
                actions_scaled[:, : self._num_non_primitive_dof]
                + self.default_dof_pos[:, : self._num_non_primitive_dof]
            )
            for i, primitive_cfg in enumerate(self._primitive_action_map):
                inheritrange = primitive_cfg["inheritrange"]
                p0 = primitive_cfg["pos_0"]
                p1 = primitive_cfg["pos_1"]
                dof_idx = primitive_cfg["dof_idx"]

                # actions_scaled_i: [-1, 1]
                actions_scaled_i = actions_scaled[:, self._num_non_primitive_dof + i]
                self.primitive_action_over_limit_buf[:, i] = torch.maximum(
                    actions_scaled_i.abs() - 1.1 * inheritrange, torch.zeros_like(actions_scaled_i)
                )
                actions_scaled_i = actions_scaled_i.clamp(min=-inheritrange, max=inheritrange)

                if primitive_cfg["mode"] == "linear":
                    # actions_scaled_i_mapped: [0, 1]
                    actions_scaled_i_mapped = (actions_scaled_i + 1.0) / 2
                    combined_target[:, dof_idx] = torch.lerp(
                        p0[None, :], p1[None, :], actions_scaled_i_mapped[:, None]
                    )
                elif primitive_cfg["mode"] == "discrete":
                    combined_target[:, dof_idx] = torch.where(
                        actions_scaled_i[:, None] >= 0, p1[None, :], p0[None, :]
                    )
                else:
                    raise ValueError(f"Invalid mode: {primitive_cfg['mode']}")

            self.simulator._robot.set_joint_position_target(
                combined_target, joint_ids=self.simulator.dof_ids
            )
        else:
            raise NotImplementedError(
                f"Simulator {self.config.simulator.config.name} not implemented"
            )

    @override
    def _action_backmap(self):
        action = torch.zeros(
            self.num_envs,
            self._num_non_primitive_dof + len(self._primitive_action_map),
            device=self.device,
        )
        action[:, : self._num_non_primitive_dof] = (
            self.simulator.dof_pos[:, : self._num_non_primitive_dof]
            - self.default_dof_pos[:, : self._num_non_primitive_dof]
        ) / self.config.robot.control.action_scale

        for i, primitive_cfg in enumerate(self._primitive_action_map):
            action[:, self._num_non_primitive_dof + i] = torch.mean(
                (
                    (self.simulator.dof_pos[:, primitive_cfg["dof_idx"]] - primitive_cfg["pos_0"])
                    / (primitive_cfg["pos_1"] + 1e-6 - primitive_cfg["pos_0"])
                ).clamp(min=0.0, max=1.0)
                * 2
                - 1,
                dim=-1,
            )
            action[:, self._num_non_primitive_dof + i] /= self.config.robot.control.action_scale

        return action

    @override
    def _reset_buffers_callback(self, env_ids, target_buf=None):
        self.primitive_action_over_limit_buf[env_ids] = 0.0
        return super()._reset_buffers_callback(env_ids, target_buf)

    @override
    def _reward_limits_primitive_action(self):
        return torch.sum(self.primitive_action_over_limit_buf, dim=1)
