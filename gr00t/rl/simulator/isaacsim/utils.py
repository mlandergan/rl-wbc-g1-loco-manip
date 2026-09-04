# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from dataclasses import dataclass, field
from typing import List

import torch
from isaaclab.utils.math import quat_from_euler_xyz
from pxr import Gf, Usd, UsdGeom


@dataclass
class Translation:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    @classmethod
    def from_dict(cls, data: dict) -> "Translation":
        return Translation(x=data["x"], y=data["y"], z=data["z"])

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
        return Rotation(w=data["w"], x=data["x"], y=data["y"], z=data["z"])

    @classmethod
    def from_euler_xyz(cls, euler: List[float]) -> "Rotation":
        euler = torch.tensor(euler)
        w, x, y, z = quat_from_euler_xyz(euler[0], euler[1], euler[2]).tolist()
        return cls(w=w, x=x, y=y, z=z)

    def __str__(self) -> str:
        return f"({self.w}, {self.x}, {self.y}, {self.z})"

    def to_list(self) -> List[float]:
        return [self.w, self.x, self.y, self.z]


@dataclass
class Transform:
    translation: Translation = field(default_factory=Translation)
    rotation: Rotation = field(default_factory=Rotation)

    @classmethod
    def from_dict(cls, data: dict) -> "Transform":
        return Transform(
            translation=Translation.from_dict(data["translation"]),
            rotation=Rotation.from_dict(data["rotation"]),
        )

    @classmethod
    def from_gf(cls, translation: Gf.Vec3d, rotation: Gf.Quaternion) -> "Transform":
        return Transform(
            translation=Translation(x=translation[0], y=translation[1], z=translation[2]),
            rotation=Rotation(
                w=rotation.GetReal(),
                x=rotation.GetImaginary()[0],
                y=rotation.GetImaginary()[1],
                z=rotation.GetImaginary()[2],
            ),
        )

    @classmethod
    def from_prim(
        cls, prim, use_scale_as_translation: bool = False, zero_z_translation: bool = False
    ) -> "Transform":
        xform = UsdGeom.Xformable(prim)
        local_transformation: Gf.Matrix4d = xform.GetLocalTransformation()
        translation: Gf.Vec3d = local_transformation.ExtractTranslation()
        if use_scale_as_translation:
            tf = Gf.Transform(local_transformation)
            translation = tf.GetScale()
        if zero_z_translation:
            translation[2] = 0.0
        # rotation: Gf.Quaternion = local_transformation.ExtractRotation().GetQuaternion()
        ops: List[UsdGeom.XformOp] = xform.GetOrderedXformOps()
        rotation: Gf.Quaternion = Gf.Rotation()
        for op in ops:
            if op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                rotation = (
                    UsdGeom.XformOp.GetOpTransform(op, Usd.TimeCode.Default())
                    .ExtractRotation()
                    .GetQuaternion()
                )

        return cls.from_gf(translation, rotation)

    def to_matrix(self) -> Gf.Matrix4d:
        translation = Gf.Vec3d(self.translation.x, self.translation.y, self.translation.z)
        rotation = Gf.Rotation()
        rotation.SetQuaternion(
            Gf.Quaternion(
                self.rotation.w,
                Gf.Vec3d(self.rotation.x, self.rotation.y, self.rotation.z).GetNormalized(),
            )
        )
        transform = Gf.Matrix4d()
        return Gf.Matrix4d.SetTransform(transform, rotation, translation)

    def to_list(self) -> List[float]:
        return self.translation.to_list() + self.rotation.to_list()

    def __str__(self) -> str:
        return f"Translation: {self.translation}\nRotation: {self.rotation}"
