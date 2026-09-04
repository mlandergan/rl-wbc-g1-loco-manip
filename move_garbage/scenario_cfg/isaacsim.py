# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Authoritative Isaac Sim object configuration for the bin-push task.

The simulator loads the bundled compatibility wrapper under
``gr00t/rl/data/tasks/bin/scenario_cfg``, which imports this module.
"""

from __future__ import annotations

import logging
import os

logging.getLogger("asyncio").setLevel(logging.WARNING)

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from gr00t.rl.utils.paths import resolve_repo_path
from move_garbage.isaac_utils.bin import RecyclingBinSpawnerCfg, spawn_recycling_bin


# The dynamic bin asset is bundled. BIN_ASSET_PATH can select a local variant.
_default_asset_path = os.environ.get(
    "BIN_ASSET_PATH",
    str(
        resolve_repo_path(
            "move_garbage/data/objects/bin/"
            "RecyclingContainerWithWheels_A01_Blue_01_dynamic.usd"
        )
    ),
)
print(f"[BinScenario] Using asset: {_default_asset_path}")


bin_spawner_cfg = RecyclingBinSpawnerCfg(
    func=spawn_recycling_bin,
    articulation_props=sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=False,
        solver_position_iteration_count=8,
        solver_velocity_iteration_count=4,
        fix_root_link=False,  # the bin body is dynamic — it must be free to translate
    ),
    # Turn off Isaac Lab's automatic contact-sensor activation. The env's
    # scene_creation_callback registers a dedicated bin_body_contact_sensor;
    # letting the @clone wrapper also do it here trips
    # `activate_contact_sensors` at prim `/World/Template/Asset_0000` which
    # can't see the referenced USD's rigid bodies at that composition stage.
    activate_contact_sensors=False,
    asset_usd_path=_default_asset_path,
    add_floor=True,
)


multi_spawner_cfg = sim_utils.MultiAssetSpawnerCfg(
    assets_cfg=[bin_spawner_cfg] * 4096,
    random_choice=False,
    activate_contact_sensors=False,
    rigid_props=sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        retain_accelerations=False,
        linear_damping=0.0,
        angular_damping=0.0,
        max_linear_velocity=1000.0,
        max_angular_velocity=1000.0,
        max_depenetration_velocity=1.0,
    ),
)


# DEBUG (2026-07-11): Try plain UsdFileCfg first to isolate whether the
# articulation parse failure is in our custom spawner or in the USD itself.
# If this works, the bug is in spawn_recycling_bin or MultiAssetSpawnerCfg.
_plain_spawn_cfg = sim_utils.UsdFileCfg(
    usd_path=_default_asset_path,
    # The dynamic USD is now authored NATIVELY IN METERS by prepare_bin_asset.py
    # (metersPerUnit=1.0, geometry + joints + colliders all in m). No spawn-time
    # scale needed. (Earlier we used scale=(0.01,...) to shrink the cm asset,
    # but PhysX handles scaled articulations poorly — see TRAINING_LOG 2026-07-12.)
    activate_contact_sensors=True,  # apply PhysxContactReportAPI to all rigid bodies
    articulation_props=sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=False,
        solver_position_iteration_count=8,
        solver_velocity_iteration_count=4,
        fix_root_link=False,
    ),
    rigid_props=sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        retain_accelerations=False,
        linear_damping=0.0,
        angular_damping=0.0,
        max_linear_velocity=1000.0,
        max_angular_velocity=1000.0,
        max_depenetration_velocity=1.0,
    ),
)

TaskObjCfgDict = {
    "bin": ArticulationCfg(
        spawn=_plain_spawn_cfg,  # was: multi_spawner_cfg  — see debug note above
        init_state=ArticulationCfg.InitialStateCfg(
            # Bin sits ~2m in front of the robot at env origin.
            pos=(0.0, 0.0, 0.0),
            # 90deg CCW yaw about +Z. The grasp handles sit on the bin's lateral
            # (+Y/-Y) rim, so rotating the bin 90deg turns that face dead-on to
            # the robot's +X approach (wheels/handles facing the G1), instead of
            # letting it walk to the bin's side.  wxyz = (cos45, 0, 0, sin45)
            rot=(0.70711, 0.0, 0.0, 0.70711),
            joint_pos={".*wheel.*": 0.0},
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=0.9,
        actuators={
            "wheels": ImplicitActuatorCfg(
                joint_names_expr=[".*wheel.*"],
                velocity_limit_sim=100.0,
                stiffness=None,
                damping=None,
            ),
        },
    )
}
