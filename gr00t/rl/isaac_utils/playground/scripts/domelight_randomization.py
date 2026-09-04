# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from typing_extensions import override
import os

import numpy as np
from pxr import Usd, Sdf, UsdLux
import omni.kit.commands
from omni.kit.scripting import BehaviorScript

from gr00t.rl.isaac_utils.playground.utils.usd_utils import get_custom_data_from_prim, write_custom_data_to_prim
from gr00t.rl.isaac_utils.playground.utils.math_utils import set_prim_transform


DOME_LIGHT_RANDOMIZATION_SCRIPT_PATH = os.path.join(os.path.dirname(__file__), "domelight_randomization.py")


class DomeLightRandomization(BehaviorScript):
    @override
    def on_init(self):
        custom_data = get_custom_data_from_prim(self.stage, self.prim.GetPrim().GetPath())
        self.randomization_interval = custom_data["randomizationInterval"]
        self.randomization_texture_file_list = custom_data["randomizationTextureFileList"]
        self.intensity_range = custom_data["intensityRange"]
        self.last_randomization_time = 0.0

    @override
    def on_update(self, current_time: float, delta_time: float):
        if current_time - self.last_randomization_time < self.randomization_interval:
            return

        self.last_randomization_time = current_time

        new_texture_file_path = np.random.choice(self.randomization_texture_file_list)
        dome_light_prim = UsdLux.DomeLight(self.prim.GetPrim())
        dome_light_prim.GetTextureFileAttr().Set(new_texture_file_path)
        dome_light_prim.GetIntensityAttr().Set(np.random.uniform(self.intensity_range[0], self.intensity_range[1]))
        set_prim_transform(self.stage, self.prim.GetPrim().GetPath(), (0.0, 0.0, 0.0), (0.0, 0.0, np.random.uniform(0, 360)), (1.0, 1.0, 1.0))

    @override
    def on_play(self):
        self.last_randomization_time = 0.0

    @staticmethod
    def add_to_prim(stage: Usd.Stage, prim_path: str, randomization_interval: float, randomization_texture_file_list: list[str], intensity_range: tuple[float, float]):
        write_custom_data_to_prim(stage, prim_path, {
            "randomizationInterval": randomization_interval,
            "randomizationTextureFileList": randomization_texture_file_list,
            "intensityRange": list(intensity_range)
        })
        omni.kit.commands.execute("ApplyScriptingAPICommand", paths=[Sdf.Path(prim_path)])
        stage.GetPrimAtPath(prim_path).CreateAttribute("omni:scripting:scripts", Sdf.ValueTypeNames.AssetArray).Set([DOME_LIGHT_RANDOMIZATION_SCRIPT_PATH])
