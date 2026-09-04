# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from dataclasses import dataclass, field
from typing import List
import math

import torch
from pxr import Gf, Usd, UsdGeom

from isaaclab.utils.math import quat_from_euler_xyz, euler_xyz_from_quat, wrap_to_pi, quat_from_matrix

@dataclass
class Translation:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    @classmethod
    def from_dict(cls, data: dict) -> "Translation":
        return Translation(
            x=data["x"],
            y=data["y"],
            z=data["z"]
        )

    def __str__(self) -> str:
        return f"({self.x}, {self.y}, {self.z})"

    def to_list(self) -> List[float]:
        return [self.x, self.y, self.z]

@dataclass
class Rotation:
    w: float = 1.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    @classmethod
    def from_dict(cls, data: dict) -> "Rotation":
        return Rotation(
            w=data["w"],
            x=data["x"],
            y=data["y"],
            z=data["z"]
        )

    @classmethod
    def from_euler_xyz(cls, euler: List[float]) -> "Rotation":
        euler = torch.tensor(euler)
        w, x, y, z = quat_from_euler_xyz(euler[0], euler[1], euler[2]).tolist()
        return cls(w=w, x=x, y=y, z=z)

    @classmethod
    def from_matrix(cls, matrix: torch.Tensor) -> "Rotation":
        w, x, y, z = quat_from_matrix(matrix).tolist()
        return cls(w=w, x=x, y=y, z=z)

    def __str__(self) -> str:
        return f"({self.w}, {self.x}, {self.y}, {self.z})"

    def to_list(self) -> List[float]:
        return [self.w, self.x, self.y, self.z]

    def to_euler_xyz(self, degrees: bool = False) -> List[float]:
        r, p, y = euler_xyz_from_quat(torch.tensor([[self.w, self.x, self.y, self.z]]))
        r = wrap_to_pi(r)
        p = wrap_to_pi(p)
        y = wrap_to_pi(y)
        if degrees:
            r = r * 180.0 / math.pi
            p = p * 180.0 / math.pi
            y = y * 180.0 / math.pi
        return [r[0], p[0], y[0]]

@dataclass
class Transform:
    translation: Translation = field(default_factory=Translation)
    rotation: Rotation = field(default_factory=Rotation)

    @classmethod
    def from_dict(cls, data: dict) -> "Transform":
        return Transform(
            translation=Translation.from_dict(data["translation"]),
            rotation=Rotation.from_dict(data["rotation"])
        )

    @classmethod
    def from_gf(cls, translation: Gf.Vec3d, rotation: Gf.Quaternion) -> "Transform":
        return Transform(
            translation=Translation(x=translation[0], y=translation[1], z=translation[2]),
            rotation=Rotation(w=rotation.GetReal(), x=rotation.GetImaginary()[0], y=rotation.GetImaginary()[1], z=rotation.GetImaginary()[2])
        )

    @classmethod
    def from_prim(cls, prim, use_scale_as_translation: bool = False, zero_z_translation: bool = False) -> "Transform":
        xform = UsdGeom.Xformable(prim)
        local_transformation: Gf.Matrix4d = xform.GetLocalTransformation()
        translation: Gf.Vec3d = local_transformation.ExtractTranslation()
        if use_scale_as_translation:
            tf = Gf.Transform(local_transformation)
            translation = tf.GetScale()
        if zero_z_translation:
            translation[2] = 0.0
        ops: List[UsdGeom.XformOp] = xform.GetOrderedXformOps()
        rotation: Gf.Quaternion = Gf.Rotation()
        for op in ops:
            if op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                rotation = UsdGeom.XformOp.GetOpTransform(op, Usd.TimeCode.Default()).ExtractRotation().GetQuaternion()

        return cls.from_gf(translation, rotation)

    @classmethod
    def from_matrix(cls, matrix: torch.Tensor) -> "Transform":
        return Transform(
            translation=Translation(x=matrix[0, 3], y=matrix[1, 3], z=matrix[2, 3]),
            rotation=Rotation.from_matrix(matrix[:3, :3])
        )

    def to_matrix(self) -> Gf.Matrix4d:
        translation = Gf.Vec3d(self.translation.x, self.translation.y, self.translation.z)
        rotation = Gf.Rotation()
        rotation.SetQuaternion(Gf.Quaternion(self.rotation.w, Gf.Vec3d(self.rotation.x, self.rotation.y, self.rotation.z).GetNormalized()))
        transform = Gf.Matrix4d()
        return Gf.Matrix4d.SetTransform(transform, rotation, translation)

    def to_list(self) -> List[float]:
        return self.translation.to_list() + self.rotation.to_list()

    def to_tensor(self, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        return torch.tensor([self.translation.x, self.translation.y, self.translation.z, self.rotation.w, self.rotation.x, self.rotation.y, self.rotation.z], device=device)

    def __str__(self) -> str:
        return f"Translation: {self.translation}\nRotation: {self.rotation}"


def set_prim_transform(stage, prim_path, translation, rotation, scale):
    """
    Set the translation, rotation, and scale of a USD prim at the same time.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path to the prim.
        translation (tuple): The translation as a tuple of three floats (x, y, z).
        rotation (tuple): The rotation as a tuple of three floats (x, y, z) in degrees.
        scale (tuple): The scale as a tuple of three floats (x, y, z).
    """
    prim = stage.GetPrimAtPath(prim_path)

    if not prim or not prim.IsA(UsdGeom.Xformable):
        raise ValueError("Prim is not valid or not of type Xformable")

    xformable = UsdGeom.Xformable(prim)

    xformable.ClearXformOpOrder()

    translate_op = xformable.AddTranslateOp()
    translate_op.Set(Gf.Vec3f(*translation))

    roll, pitch, yaw = rotation
    cr = math.cos(math.radians(roll)/2)
    sr = math.sin(math.radians(roll)/2)
    cp = math.cos(math.radians(pitch)/2)
    sp = math.sin(math.radians(pitch)/2)
    cy = math.cos(math.radians(yaw)/2)
    sy = math.sin(math.radians(yaw)/2)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    rotation = Gf.Quatd(w, Gf.Vec3d(x, y, z))
    rotate_op = xformable.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble)
    rotate_op.Set(rotation)

    scale_op = xformable.AddScaleOp(precision=UsdGeom.XformOp.PrecisionDouble)
    scale_op.Set(Gf.Vec3f(*scale))
