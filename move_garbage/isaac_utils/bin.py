# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recycling-bin spawner for the bin-push task.

Loads a pre-built articulated recycling-bin USD (produced by
``move_garbage/scripts/prepare_bin_asset.py``) and augments it with:

- Randomized wheel drive (stiffness + damping)
- Randomized bin body mass
- Optional ground plane with randomized friction + slight slope

Mirrors ``gr00t/rl/isaac_utils/playground/env_rand/cabinet.py`` — simpler
because the bin has fewer joints and no material randomization in v1.
"""

from __future__ import annotations

import math
from typing import Optional

import isaaclab.sim as sim_utils
import isaacsim.core.utils.prims as prim_utils
import numpy as np
import omni.usd
from isaaclab.sim.utils import clone
from isaaclab.utils import configclass
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

from gr00t.rl.isaac_utils.playground.utils.math_utils import set_prim_transform
from gr00t.rl.isaac_utils.playground.utils.usd_utils import (
    create_plane,
    write_custom_data_to_prim,
)


@configclass
class RecyclingBinSpawnerCfg(sim_utils.RigidObjectSpawnerCfg):
    """Configuration for :func:`spawn_recycling_bin`."""

    articulation_props: sim_utils.ArticulationRootPropertiesCfg = None

    # Path to the base bin USD asset (dynamic, articulated).
    asset_usd_path: str = ""

    # Wheel joint drive randomization ranges (per-env, uniform).
    wheel_drive_stiffness: tuple[float, float] = (0.0, 1.0)
    wheel_drive_damping: tuple[float, float] = (0.05, 0.30)

    # Bin body mass range in kg (per-env, uniform).
    bin_mass_range: tuple[float, float] = (10.0, 25.0)

    # Ground plane options.
    add_floor: bool = True
    floor_size: tuple[float, float] = (6.0, 6.0)
    ground_friction: tuple[float, float] = (0.6, 1.2)
    # Slope in radians (±3° default); interpreted as the max tilt magnitude around
    # a random horizontal axis. The spawner picks a random axis each env and
    # applies a rotation by ``uniform(*ground_slope_rad)`` about it.
    ground_slope_rad: tuple[float, float] = (-0.0523598776, 0.0523598776)  # ±3°

    # Z offset to place bin base on ground; computed automatically if None.
    z_offset: Optional[float] = None


@clone
def spawn_recycling_bin(
    prim_path: str,
    cfg: RecyclingBinSpawnerCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> "Usd.Prim":
    """Spawn a dynamic recycling-bin USD with randomized physics.

    Mirrors ``spawn_cabinet``. This function is called once per env by Isaac Lab
    when the ArticulationCfg is instantiated (via @clone).
    """
    stage: Usd.Stage = omni.usd.get_context().get_stage()

    if not cfg.asset_usd_path:
        raise ValueError(
            "RecyclingBinSpawnerCfg.asset_usd_path must point to a dynamic bin USD"
        )

    # === Load base asset as a USD reference ===
    bin_prim = prim_utils.create_prim(
        prim_path, "Xform", translation=translation, orientation=orientation
    )
    bin_prim.GetReferences().AddReference(cfg.asset_usd_path)

    # The dynamic USD applies ArticulationRootAPI to `/Bin` (its defaultPrim).
    # When referenced under a new prim path, the API doesn't propagate to the
    # referencing prim — Isaac Lab looks for ArticulationRootAPI ON the top
    # asset prim itself, not on its children, so it fails to locate the
    # articulation. Apply it here to the referencing prim.
    if not bin_prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        UsdPhysics.ArticulationRootAPI.Apply(bin_prim)

    # === Compute Z offset to place bin base on ground ===
    z_offset = cfg.z_offset
    if z_offset is None:
        try:
            temp_stage = Usd.Stage.Open(cfg.asset_usd_path)
            temp_root = temp_stage.GetPrimAtPath("/root")
            if not temp_root or not temp_root.IsValid():
                # Fall back to the pseudo-root if the asset doesn't have /root.
                temp_root = temp_stage.GetPseudoRoot()
            bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
            bbox = bbox_cache.ComputeWorldBound(temp_root)
            z_min = bbox.ComputeAlignedRange().GetMin()[2]
            z_offset = -float(z_min)
            del temp_stage
        except Exception:
            # If we can't open the USD (e.g., asset is a placeholder), assume the
            # asset is already at Z=0 — the env code will still work.
            z_offset = 0.0

    xformable = UsdGeom.Xformable(bin_prim)
    translate_op = None
    for op in xformable.GetOrderedXformOps():
        if op.GetOpName() == "xformOp:translate":
            translate_op = op
            break
    if translate_op is not None:
        translate_op.Set(Gf.Vec3d(0, 0, z_offset))
    else:
        xformable.AddTranslateOp().Set(Gf.Vec3d(0, 0, z_offset))

    # === Randomize wheel joint drives ===
    # Wheel joints are named according to whatever prepare_bin_asset.py chose.
    # We identify them by inspecting all revolute joints under the bin prim and
    # matching on names containing "wheel".
    root_prim = stage.GetPrimAtPath(prim_path)
    if root_prim.IsValid():
        for prim in Usd.PrimRange(root_prim):
            if not prim.IsA(UsdPhysics.RevoluteJoint):
                continue
            name = prim.GetName().lower()
            if "wheel" not in name:
                continue
            stiffness = float(
                np.random.uniform(*cfg.wheel_drive_stiffness)
            )
            damping = float(np.random.uniform(*cfg.wheel_drive_damping))
            UsdPhysics.DriveAPI.Apply(prim, "angular")
            prim.CreateAttribute(
                "drive:angular:physics:type", Sdf.ValueTypeNames.Token
            ).Set("force")
            prim.CreateAttribute(
                "drive:angular:physics:stiffness", Sdf.ValueTypeNames.Float
            ).Set(stiffness)
            prim.CreateAttribute(
                "drive:angular:physics:damping", Sdf.ValueTypeNames.Float
            ).Set(damping)
            prim.CreateAttribute(
                "drive:angular:physics:targetPosition", Sdf.ValueTypeNames.Float
            ).Set(0.0)

    # === Randomize bin body mass ===
    # The bin body prim is the ArticulationRoot itself (a RigidBody with mass).
    bin_mass = float(np.random.uniform(*cfg.bin_mass_range))
    body_prim = stage.GetPrimAtPath(prim_path)
    if body_prim.IsValid():
        mass_api = UsdPhysics.MassAPI.Apply(body_prim)
        mass_api.CreateMassAttr().Set(bin_mass)

    # === Ground plane (optional) with random friction + slope ===
    floor_friction = 0.9
    slope_rad = 0.0
    slope_axis = (1.0, 0.0)
    if cfg.add_floor:
        import os as _os

        env_root = _os.path.dirname(prim_path)  # e.g. /World/envs/env_0
        floor_path = f"{env_root}/floor"
        if not stage.GetPrimAtPath(floor_path).IsValid():
            create_plane(stage, floor_path, cfg.floor_size)

            floor_friction = float(np.random.uniform(*cfg.ground_friction))
            slope_rad = float(np.random.uniform(*cfg.ground_slope_rad))
            # Random horizontal axis for the slope direction.
            axis_theta = float(np.random.uniform(0.0, 2.0 * math.pi))
            slope_axis = (math.cos(axis_theta), math.sin(axis_theta))

            # Apply the tilt to the floor xform. slope_axis lies in the XY plane;
            # we tilt about it by slope_rad. Roll = slope_rad * sin(theta), Pitch
            # = slope_rad * cos(theta) is a small-angle approximation that's fine
            # for ±3° tilts.
            roll = slope_rad * slope_axis[1]
            pitch = slope_rad * slope_axis[0]
            set_prim_transform(
                stage, floor_path, (0.0, 0.0, 0.001), (roll, pitch, 0.0), (1.0, 1.0, 1.0)
            )

            # Attach a physics material with the randomized friction.
            mat_path = f"{env_root}/floor_material"
            if not stage.GetPrimAtPath(mat_path).IsValid():
                mat_prim = UsdPhysics.MaterialAPI.Apply(
                    stage.DefinePrim(mat_path, "Material")
                )
                mat_prim.CreateStaticFrictionAttr().Set(floor_friction)
                mat_prim.CreateDynamicFrictionAttr().Set(
                    max(0.5 * floor_friction, 0.1)
                )
                mat_prim.CreateRestitutionAttr().Set(0.0)

    # === Write metadata (auxiliary — env reads its own customData from the USD) ===
    metadata = {
        "binAssetPath": cfg.asset_usd_path,
        "zOffset": z_offset,
        "binMass": bin_mass,
        "floorFriction": floor_friction,
        "groundSlopeRad": slope_rad,
        "groundSlopeAxisXY": [slope_axis[0], slope_axis[1]],
    }
    write_custom_data_to_prim(stage, prim_path, metadata)

    return bin_prim
