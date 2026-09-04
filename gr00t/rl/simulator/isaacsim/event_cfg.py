# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from pathlib import Path

try:
    with Path(__file__).with_name(".isaacsim_version").open("r", encoding="utf-8") as f:
        DEFAULT_ISAACSIM_VERSION = f.read().strip()
except FileNotFoundError:
    DEFAULT_ISAACSIM_VERSION = "4.5"

if DEFAULT_ISAACSIM_VERSION == "4.5":
    ### IsaacSim 4.5
    from isaaclab.utils import configclass

elif DEFAULT_ISAACSIM_VERSION == "4.2":
    from omni.isaac.lab.utils import configclass
else:
    raise ValueError(f"Unsupported IsaacSim version: {DEFAULT_ISAACSIM_VERSION}")

# @configclass
# class EventCfg:
#     """Configuration for events."""

#     scale_body_mass = EventTerm(
#         func=mdp.randomize_rigid_body_mass,
#         mode="startup",
#         params={
#             "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
#             "mass_distribution_params": (0.8, 1.2),
#             "operation": "scale",
#         },
#     )

#     random_joint_friction = EventTerm(
#         func=mdp.randomize_joint_parameters,
#         mode="startup",
#         params={
#             "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
#             "friction_distribution_params": (0.5, 1.25),
#             "operation": "scale",
#         },
#     )


@configclass
class EventCfg:
    """Configuration for events."""

    scale_body_mass = None
    random_joint_friction = None
