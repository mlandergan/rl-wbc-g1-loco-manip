#!/usr/bin/env python3
"""Show only the two DEX hands at the bin's final target poses."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--frames",
        type=int,
        default=0,
        help="Exit after this many render frames; zero keeps the viewer open.",
    )
    parser.add_argument(
        "--save-stage",
        type=Path,
        help="Optionally export the composed validation stage as USDA.",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


ARGS = parse_args()
simulation_app = AppLauncher(ARGS).app

import omni.usd
from isaacsim.core.utils.viewports import set_camera_view
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade


REPO_ROOT = Path(__file__).resolve().parents[2]
G1_ASSET = REPO_ROOT / "gr00t/rl/data/robots/g1/g1_43dof_s3.usda"
BIN_ASSET = (
    REPO_ROOT
    / "move_garbage/data/objects/bin/RecyclingContainerWithWheels_A01_Blue_01_dynamic.usd"
)
G1_ROOT = "/g1_29dof_with_hand_rev_1_0"
GRASP_OFFSET = 0.0464
TARGET_STANDOFF = 0.04
TARGET_HEIGHT_OFFSET = 0.01
GRASP_AXES = {
    "left": np.array((0.9717, -0.2356, -0.0181), dtype=np.float64),
    "right": np.array((0.9717, 0.2359, 0.0155), dtype=np.float64),
}
FINGER_POSES = {
    "left": {
        "index_0": -1.5708,
        "index_1": -1.5708,
        "middle_0": -1.5708,
        "middle_1": -1.5708,
        "thumb_0": 0.0,
        "thumb_1": 0.785398,
        "thumb_2": 1.5708,
    },
    "right": {
        "index_0": 1.5708,
        "index_1": 1.5708,
        "middle_0": 1.5708,
        "middle_1": 1.5708,
        "thumb_0": 0.0,
        "thumb_1": -0.785398,
        "thumb_2": -1.5708,
    },
}


def quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.array(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dtype=np.float64,
    )


def axis_angle_quat(axis: str, angle: float) -> np.ndarray:
    result = np.zeros(4, dtype=np.float64)
    result[0] = math.cos(angle / 2.0)
    result[{"X": 1, "Y": 2, "Z": 3}[axis]] = math.sin(angle / 2.0)
    return result


def rotate(quat_wxyz: np.ndarray, vector: np.ndarray) -> np.ndarray:
    _, x, y, z = quat_wxyz
    skew = np.array(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))
    return vector + 2.0 * (
        quat_wxyz[0] * (skew @ vector) + skew @ (skew @ vector)
    )


def set_pose(prim: Usd.Prim, position: np.ndarray, quaternion: np.ndarray) -> None:
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*position))
    xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Quatd(float(quaternion[0]), Gf.Vec3d(*quaternion[1:]))
    )


def make_material(
    stage: Usd.Stage, path: str, color: tuple[float, float, float]
) -> UsdShade.Material:
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.75)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.3)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def closed_link_poses(side: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    values = FINGER_POSES[side]
    lateral = 1.0 if side == "left" else -1.0
    poses: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "palm": (np.zeros(3), np.array((1.0, 0.0, 0.0, 0.0)))
    }

    def add(
        name: str,
        parent: str,
        offset: tuple[float, float, float],
        axis: str,
    ) -> None:
        parent_pos, parent_quat = poses[parent]
        position = parent_pos + rotate(parent_quat, np.array(offset))
        quaternion = quat_multiply(
            parent_quat, axis_angle_quat(axis, values[name])
        )
        poses[name] = (position, quaternion)

    add("index_0", "palm", (0.0777, 0.0016 * lateral, 0.0285), "Z")
    add("index_1", "index_0", (0.0458, 0.0, 0.0), "Z")
    add("middle_0", "palm", (0.0777, 0.0016 * lateral, -0.0285), "Z")
    add("middle_1", "middle_0", (0.0458, 0.0, 0.0), "Z")
    add("thumb_0", "palm", (0.0255, 0.0, 0.0), "Y")
    add("thumb_1", "thumb_0", (-0.0025, -0.0193 * lateral, 0.0), "Z")
    add("thumb_2", "thumb_1", (0.0, -0.0458 * lateral, 0.0), "Z")
    return poses


def add_hand(
    stage: Usd.Stage,
    side: str,
    palm_position: np.ndarray,
    palm_quaternion: np.ndarray,
    material: UsdShade.Material,
) -> None:
    hand = UsdGeom.Xform.Define(stage, f"/World/Hands/{side.capitalize()}").GetPrim()
    set_pose(hand, palm_position, palm_quaternion)
    for short_name, (position, quaternion) in closed_link_poses(side).items():
        link_name = f"{side}_hand_{short_name}_link"
        link_path = f"/World/Hands/{side.capitalize()}/{link_name}"
        link = UsdGeom.Xform.Define(stage, link_path).GetPrim()
        set_pose(link, position, quaternion)
        mesh = UsdGeom.Mesh.Define(stage, f"{link_path}/visuals")
        source_path = f"{G1_ROOT}/{link_name}/visuals"
        mesh.GetPrim().GetReferences().AddReference(str(G1_ASSET), source_path)
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)


def add_sphere(
    stage: Usd.Stage,
    path: str,
    position: np.ndarray,
    radius: float,
    color: tuple[float, float, float],
) -> None:
    sphere = UsdGeom.Sphere.Define(stage, path)
    sphere.CreateRadiusAttr(radius)
    sphere.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    set_pose(sphere.GetPrim(), position, np.array((1.0, 0.0, 0.0, 0.0)))


def add_axes(
    stage: Usd.Stage,
    path: str,
    position: np.ndarray,
    quaternion: np.ndarray,
    length: float = 0.12,
) -> None:
    axes = UsdGeom.BasisCurves.Define(stage, path)
    points = []
    for axis in np.eye(3):
        points.extend((Gf.Vec3f(*position), Gf.Vec3f(*(position + rotate(quaternion, axis) * length))))
    axes.CreateTypeAttr(UsdGeom.Tokens.linear)
    axes.CreateCurveVertexCountsAttr([2, 2, 2])
    axes.CreatePointsAttr(points)
    axes.CreateWidthsAttr([0.006] * 6)
    axes.CreateDisplayColorAttr(
        [
            Gf.Vec3f(0.9, 0.1, 0.1),
            Gf.Vec3f(0.1, 0.8, 0.2),
            Gf.Vec3f(0.1, 0.3, 0.95),
        ]
    )
    axes.SetWidthsInterpolation(UsdGeom.Tokens.vertex)
    axes.GetDisplayColorPrimvar().SetInterpolation(UsdGeom.Tokens.uniform)


def find_descendant(stage: Usd.Stage, name: str) -> Usd.Prim:
    for prim in stage.Traverse():
        if prim.GetName() == name and prim.GetPath().HasPrefix("/World/Bin"):
            return prim
    raise RuntimeError(f"Could not find {name} under the composed bin")


def world_pose(cache: UsdGeom.XformCache, prim: Usd.Prim) -> tuple[np.ndarray, np.ndarray]:
    transform = Gf.Transform(cache.GetLocalToWorldTransform(prim))
    translation = np.array(transform.GetTranslation(), dtype=np.float64)
    quat = transform.GetRotation().GetQuat()
    quaternion = np.array(
        (quat.GetReal(), *quat.GetImaginary()), dtype=np.float64
    )
    return translation, quaternion / np.linalg.norm(quaternion)


def build_stage() -> Usd.Stage:
    context = omni.usd.get_context()
    context.new_stage()
    stage = context.get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, "/World/Hands")

    bin_prim = UsdGeom.Xform.Define(stage, "/World/Bin").GetPrim()
    bin_prim.GetReferences().AddReference(str(BIN_ASSET))
    set_pose(
        bin_prim,
        np.zeros(3),
        np.array((math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5))),
    )

    hand_material = make_material(stage, "/World/Looks/DEXHand", (0.23, 0.29, 0.34))
    cache = UsdGeom.XformCache()
    # The authored target frames send palm-local +X (the finger direction)
    # sideways along the bin. Apply the same offsets as BinPushHomie so +X
    # points inward through the handle face while the mirrored palm sides face
    # down. Move each pose 4 cm back along authored local -Z and 1 cm up along
    # local +Y, which are world -X and +Z at the initial bin yaw.
    right_offset = quat_multiply(
        quat_multiply(
            axis_angle_quat("X", -math.pi / 2.0),
            axis_angle_quat("Z", -math.pi / 2.0),
        ),
        axis_angle_quat("X", -math.pi / 2.0),
    )
    assignments = {
        "left": ("grasp_target_right", axis_angle_quat("Y", -math.pi / 2.0)),
        "right": ("grasp_target_left", right_offset),
    }
    for side, (target_name, orientation_offset) in assignments.items():
        target_position, target_quaternion = world_pose(
            cache, find_descendant(stage, target_name)
        )
        target_position += rotate(
            target_quaternion,
            np.array((0.0, TARGET_HEIGHT_OFFSET, -TARGET_STANDOFF)),
        )
        target_quaternion = quat_multiply(target_quaternion, orientation_offset)
        axis = GRASP_AXES[side]
        axis /= np.linalg.norm(axis)
        palm_position = target_position - rotate(
            target_quaternion, axis * GRASP_OFFSET
        )
        add_hand(
            stage,
            side,
            palm_position,
            target_quaternion,
            hand_material,
        )
        marker_color = (0.1, 0.9, 0.35) if side == "left" else (1.0, 0.65, 0.05)
        add_sphere(
            stage,
            f"/World/Markers/{side.capitalize()}Target",
            target_position,
            0.014,
            marker_color,
        )
        add_sphere(
            stage,
            f"/World/Markers/{side.capitalize()}PalmOrigin",
            palm_position,
            0.009,
            (0.2, 0.65, 1.0),
        )
        add_axes(
            stage,
            f"/World/Markers/{side.capitalize()}TargetAxes",
            target_position,
            target_quaternion,
        )
        reconstructed = palm_position + rotate(
            target_quaternion, axis * GRASP_OFFSET
        )
        error = np.linalg.norm(reconstructed - target_position)
        print(
            f"{side}: target_pos={target_position.round(6).tolist()} "
            f"target_quat_wxyz={target_quaternion.round(6).tolist()} "
            f"palm_pos={palm_position.round(6).tolist()} "
            f"grasp_point_error={error:.3e} m"
        )
    return stage


stage = build_stage()
set_camera_view(
    eye=np.array((1.5, -1.8, 1.35)),
    target=np.array((-0.25, 0.0, 0.82)),
    camera_prim_path="/OmniverseKit_Persp",
)

if ARGS.save_stage:
    output = ARGS.save_stage.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    stage.GetRootLayer().Export(str(output))
    print(f"Saved validation stage: {output}")

frame = 0
while simulation_app.is_running() and (ARGS.frames <= 0 or frame < ARGS.frames):
    simulation_app.update()
    frame += 1

simulation_app.close()
