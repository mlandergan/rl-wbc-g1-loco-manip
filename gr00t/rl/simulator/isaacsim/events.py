# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import gc
from typing import Literal

## IsaacSim 4.5
import isaaclab.utils.math as math_utils
import omni.client
import omni.kit.commands
import omni.usd
import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim.utils import bind_visual_material
from loguru import logger
from pxr import Sdf, Usd, UsdLux, UsdShade

from .utils import Rotation, Transform


def resolve_dist_fn(
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    dist_fn = math_utils.sample_uniform

    if distribution == "uniform":
        dist_fn = math_utils.sample_uniform
    elif distribution == "log_uniform":
        dist_fn = math_utils.sample_log_uniform
    elif distribution == "gaussian":
        dist_fn = math_utils.sample_gaussian
    else:
        raise ValueError(f"Unrecognized distribution {distribution}")

    return dist_fn


def randomize_body_com(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    distribution_params: tuple[float, float] | tuple[torch.Tensor, torch.Tensor],
    operation: Literal["add", "abs", "scale"],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    num_envs: int = 1,  # number of environments
):
    """Randomize the com of the bodies by adding, scaling or setting random values.

    This function allows randomizing the center of mass of the bodies of the asset. The function samples random values from the
    given distribution parameters and adds, scales or sets the values into the physics simulation based on the operation.

    .. tip::
        This function uses CPU tensors to assign the body masses. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # get the current masses of the bodies (num_assets, num_bodies)
    coms = asset.root_physx_view.get_coms()
    # apply randomization on default values
    # import ipdb; ipdb.set_trace()
    coms[env_ids[:, None], body_ids] = env.default_coms[env_ids[:, None], body_ids].clone()

    dist_fn = resolve_dist_fn(distribution)

    if isinstance(distribution_params[0], torch.Tensor):
        distribution_params = (
            distribution_params[0].to(coms.device),
            distribution_params[1].to(coms.device),
        )

    env.base_com_bias[env_ids, :] = dist_fn(
        *distribution_params, (env_ids.shape[0], env.base_com_bias.shape[1]), device=coms.device
    )

    # sample from the given range
    if operation == "add":
        coms[env_ids[:, None], body_ids, :3] += env.base_com_bias[env_ids[:, None], :]
    elif operation == "abs":
        coms[env_ids[:, None], body_ids, :3] = env.base_com_bias[env_ids[:, None], :]
    elif operation == "scale":
        coms[env_ids[:, None], body_ids, :3] *= env.base_com_bias[env_ids[:, None], :]
    else:
        raise ValueError(
            f"Unknown operation: '{operation}' for property randomization. Please use 'add', 'abs' or 'scale'."
        )
    # set the mass into the physics simulation
    asset.root_physx_view.set_coms(coms, env_ids)


def randomize_dome_light(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    texture_files: list[str],
    intensity_min_max: tuple[float, float] | tuple[torch.Tensor, torch.Tensor],
    yaw_min_max: tuple[float, float] | tuple[torch.Tensor, torch.Tensor],
    intensity_distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    yaw_distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize the dome light with improved memory management."""
    # Cache the dome light prim to avoid repeated USD queries
    if not hasattr(env, "_cached_dome_light_prim"):
        env._cached_dome_light_prim = UsdLux.DomeLight(env.scene[asset_cfg.name])

    dome_light_prim = env._cached_dome_light_prim

    # Early exit if dome light is invalid
    if not dome_light_prim.GetPrim().IsValid():
        return

    # Pre-allocate and cache transform matrices to avoid repeated allocations
    if not hasattr(env, "_cached_dome_light_transforms"):
        env._cached_dome_light_transforms = {}

    # Set a random texture file
    if texture_files:
        random_texture_file_choice = int(torch.randint(0, len(texture_files), (1,)).item())
        selected_texture = texture_files[random_texture_file_choice]

        # Only set texture if it's different from current one to avoid unnecessary operations
        current_texture = dome_light_prim.GetTextureFileAttr().Get()
        if current_texture != selected_texture:
            dome_light_prim.GetTextureFileAttr().Set(selected_texture)

    # Set a random intensity with optimized tensor operations
    intensity_min, intensity_max = intensity_min_max
    dist_fn = resolve_dist_fn(intensity_distribution)

    # Use cached tensor if available to reduce allocations
    if not hasattr(env, "_dome_light_intensity_tensor"):
        env._dome_light_intensity_tensor = torch.zeros(1, device=env.device)

    # Reuse the tensor instead of creating new ones
    intensity = dist_fn(intensity_min, intensity_max, 1, device=env.device)
    dome_light_prim.GetIntensityAttr().Set(float(intensity.item()))

    # Set a random yaw with cached transform matrices
    yaw_min, yaw_max = yaw_min_max
    yaw_dist_fn = resolve_dist_fn(yaw_distribution)
    yaw = yaw_dist_fn(yaw_min, yaw_max, 1, device=env.device)
    yaw_value = float(yaw.item())

    # Use cached transform matrix if available
    if yaw_value not in env._cached_dome_light_transforms:
        # Create transform matrix and cache it
        rotation = Rotation.from_euler_xyz([0.0, 0.0, yaw_value])
        transform = Transform(rotation=rotation)
        env._cached_dome_light_transforms[yaw_value] = transform.to_matrix()

    rotation_transform = env._cached_dome_light_transforms[yaw_value]

    # Use the cached transform matrix
    omni.kit.commands.execute(
        "TransformPrimCommand",
        path=dome_light_prim.GetPath(),
        new_transform_matrix=rotation_transform,
    )

    # Periodic cleanup of cached transforms to prevent unbounded growth
    if len(env._cached_dome_light_transforms) > 100:
        # Keep only the most recent 50 transforms
        items = list(env._cached_dome_light_transforms.items())
        env._cached_dome_light_transforms = dict(items[-50:])

    # Force cleanup of any temporary variables
    del intensity, yaw, yaw_value, rotation_transform

    # Periodic garbage collection every 100 calls
    if not hasattr(env, "_dome_light_call_count"):
        env._dome_light_call_count = 0
    env._dome_light_call_count += 1

    if env._dome_light_call_count % 100 == 0:
        cleanup_usd_memory()


def randomize_robot_material(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    shader_prim_paths: list[str],
    roughness_range: tuple[float, float] | tuple[torch.Tensor, torch.Tensor],
    metallic_range: tuple[float, float] | tuple[torch.Tensor, torch.Tensor],
    specular_range: tuple[float, float] | tuple[torch.Tensor, torch.Tensor],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize the material of the robot."""
    stage = omni.usd.get_context().get_stage()

    # get the randomization function
    dist_fn = resolve_dist_fn(distribution)

    # Generate random values once for all shaders
    metallic_min, metallic_max = metallic_range
    metallic = dist_fn(metallic_min, metallic_max, 1, device=env.device)

    # set a random roughness

    # set a random roughness
    roughness_min, roughness_max = roughness_range
    roughness = dist_fn(roughness_min, roughness_max, 1, device=env.device)

    # set a random specular

    # set a random specular
    specular_min, specular_max = specular_range
    specular = dist_fn(specular_min, specular_max, 1, device=env.device)

    for shader_prim_path in shader_prim_paths:
        # get the material prim
        shader: UsdShade.Shader = UsdShade.Shader.Get(stage, shader_prim_path)

        if not shader:
            continue

        # Get or create inputs more efficiently
        enable_orm_input = shader.GetInput("enable_ORM_texture")
        if not enable_orm_input:
            enable_orm_input = shader.CreateInput("enable_ORM_texture", Sdf.ValueTypeNames.Bool)
        enable_orm_input.Set(False)

        metallic_input = shader.GetInput("metallic_constant")
        if not metallic_input:
            metallic_input = shader.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float)
        metallic_input.Set(metallic.item())

        roughness_input = shader.GetInput("reflection_roughness_constant")
        if not roughness_input:
            roughness_input = shader.CreateInput(
                "reflection_roughness_constant", Sdf.ValueTypeNames.Float
            )
        roughness_input.Set(roughness.item())

        specular_input = shader.GetInput("specular_level")
        if not specular_input:
            specular_input = shader.CreateInput("specular_level", Sdf.ValueTypeNames.Float)
        specular_input.Set(specular.item())


def randomize_floor_material(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    material_count: int,
):
    """Randomize the material of the floor."""
    floor_geom_prim_path = "/World/ground/terrain/Environment/Geometry"
    floor_geom_prim: Usd.Prim = (
        omni.usd.get_context().get_stage().GetPrimAtPath(floor_geom_prim_path)
    )
    random_material_index = int(torch.randint(0, material_count, (1,)).item())
    random_material_prim_path = f"/World/Looks/FloorMaterials/FloorMaterial_{random_material_index}"

    # Get existing material instead of creating new one
    stage = omni.usd.get_context().get_stage()
    mtl_path = Sdf.Path(random_material_prim_path)
    existing_material = UsdShade.Material.Get(stage, mtl_path)

    if existing_material:
        # Use existing material
        bindingAPI = UsdShade.MaterialBindingAPI.Apply(floor_geom_prim)
        bindingAPI.Bind(existing_material)
    else:
        # Fallback: create new material if it doesn't exist (shouldn't happen in normal operation)
        mtl = UsdShade.Material.Define(stage, mtl_path)
        bindingAPI = UsdShade.MaterialBindingAPI.Apply(floor_geom_prim)
        bindingAPI.Bind(mtl)


def randomize_table_material(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    table_prim_paths: list[str],
    material_count: int,
):
    """Randomize the material of table objects by selecting from pre-loaded materials.

    This function randomly selects and applies materials from a pre-loaded set of materials
    to table objects, similar to how floor materials are randomized.

    Args:
        env: The environment instance
        env_ids: Environment IDs to apply randomization to (None for all)
        table_prim_paths: List of USD prim paths to table objects
        material_count: Number of available materials to choose from
    """
    stage = omni.usd.get_context().get_stage()

    for table_prim_path in table_prim_paths:
        # get the table prim
        table_prim: Usd.Prim = stage.GetPrimAtPath(table_prim_path)
        if not table_prim.IsValid():
            print(f"Warning: Table prim not found at path: {table_prim_path}")
            continue

        # Select a random material from the pre-loaded materials
        random_material_index = int(torch.randint(0, material_count, (1,)).item())
        random_material_prim_path = (
            f"/World/Looks/TableMaterials/TableMaterial_{random_material_index}"
        )

        # Get existing material instead of creating new one
        mtl_path = Sdf.Path(random_material_prim_path)
        existing_material = UsdShade.Material.Get(stage, mtl_path)

        if existing_material:
            # Use existing material
            bindingAPI = UsdShade.MaterialBindingAPI.Apply(table_prim)
            bindingAPI.Bind(existing_material)
        else:
            # Fallback: create new material if it doesn't exist (shouldn't happen in normal operation)
            mtl = UsdShade.Material.Define(stage, mtl_path)
            bindingAPI = UsdShade.MaterialBindingAPI.Apply(table_prim)
            bindingAPI.Bind(mtl)


def randomize_object_material(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    object_prim_paths: list[str],
    material_count: int,
):
    """Randomize the material of task objects by selecting from pre-loaded materials.

    This function randomly selects and applies materials from a pre-loaded set of materials
    to task objects, similar to how table and floor materials are randomized.

    Args:
        env: The environment instance
        env_ids: Environment IDs to apply randomization to (None for all)
        object_prim_paths: List of USD prim paths to task objects
        material_count: Number of available materials to choose from
    """
    stage = omni.usd.get_context().get_stage()
    # import ipdb; ipdb.set_trace()
    object_prim_paths_env_ids = [object_prim_paths[i] for i in env_ids.tolist()]

    for object_prim_path in object_prim_paths_env_ids:
        # get the object prim
        object_prim: Usd.Prim = stage.GetPrimAtPath(object_prim_path)

        if not object_prim.IsValid():
            print(f"Warning: Object prim not found at path: {object_prim_path}")
            continue

        # Select a random material from the pre-loaded materials
        random_material_index = int(torch.randint(0, material_count, (1,)).item())
        random_material_prim_path = (
            f"/World/Looks/ObjectMaterials/ObjectMaterial_{random_material_index}"
        )

        # Get existing material instead of creating new one
        mtl_path = Sdf.Path(random_material_prim_path)
        existing_material = UsdShade.Material.Get(stage, mtl_path)

        if existing_material:
            # Use existing material
            # bindingAPI = UsdShade.MaterialBindingAPI.Apply(object_prim)
            # bindingAPI.Bind(existing_material)

            bind_visual_material(
                prim_path=object_prim_path,
                material_path=mtl_path,
                stage=stage,
                stronger_than_descendants=True,
            )
        else:
            # Fallback: create new material if it doesn't exist (shouldn't happen in normal operation)
            # mtl = UsdShade.Material.Define(stage, mtl_path)
            # bindingAPI = UsdShade.MaterialBindingAPI.Apply(object_prim)
            # bindingAPI.Bind(mtl)

            bind_visual_material(
                prim_path=object_prim_path,
                material_path=mtl_path,
                stage=stage,
                stronger_than_descendants=True,
            )

        logger.warning(f"Randomized material {random_material_prim_path} to {object_prim_path}")


def cleanup_usd_memory():
    """
    Utility function to help cleanup USD memory.
    This can be called periodically to help with memory management.
    """
    try:
        # Force garbage collection
        gc.collect()

        # Clear torch cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Additional cleanup for Omniverse
        try:
            import omni.kit.app

            app = omni.kit.app.get_app()
            if app:
                app.update()  # Update the app to process any pending operations
        except:
            pass

    except Exception:
        # Silently handle any cleanup errors
        pass


def optimize_material_randomization_memory():
    """
    One-time optimization function to help with material randomization memory usage.
    Call this once during initialization to set up optimal memory management.
    """
    try:
        # Basic memory optimization - just force garbage collection
        gc.collect()

        # Get the USD stage for basic checks
        stage = omni.usd.get_context().get_stage()
        if stage:
            # Just ensure the stage is accessible - avoid making changes that could cause issues
            pass

    except Exception as e:
        print(f"Warning: Material randomization memory optimization failed: {e}")
