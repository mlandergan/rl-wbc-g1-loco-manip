# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import weakref
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch

import omni.kit.app
import omni.timeline

if TYPE_CHECKING:
    from isaaclab.envs import DirectRLEnv, ManagerBasedEnv, ViewerCfg


class ViewportCameraController:
    """Controls the camera associated with a viewport in the simulator.

    Supports tracking different origin types:
    - **world**: the center of the world (static)
    - **env**: the center of an environment (static)
    - **asset_root**: the root of an asset in the scene (dynamic tracking)
    """

    def __init__(self, env, cfg: ViewerCfg):
        self._env = env
        self._cfg = copy.deepcopy(cfg)
        self.default_cam_eye = np.array(self._cfg.eye)
        self.default_cam_lookat = np.array(self._cfg.lookat)

        if self.cfg.origin_type == "env":
            self.set_view_env_index(self.cfg.env_index)
            self.update_view_to_env()
        elif self.cfg.origin_type == "asset_root":
            if self.cfg.asset_name is None:
                raise ValueError(
                    f"No asset name provided for viewer with origin type: '{self.cfg.origin_type}'."
                )
        else:
            self.update_view_to_world()

        app_interface = omni.kit.app.get_app_interface()
        app_event_stream = app_interface.get_post_update_event_stream()
        self._viewport_camera_update_handle = app_event_stream.create_subscription_to_pop(
            lambda event, obj=weakref.proxy(self): obj._update_tracking_callback(event)
        )

    def __del__(self):
        if (
            hasattr(self, "_viewport_camera_update_handle")
            and self._viewport_camera_update_handle is not None
        ):
            self._viewport_camera_update_handle.unsubscribe()
            self._viewport_camera_update_handle = None

    @property
    def cfg(self) -> ViewerCfg:
        return self._cfg

    def set_view_env_index(self, env_index: int):
        if env_index < 0 or env_index >= self._env.config.scene.num_envs:
            raise ValueError(
                f"Out of range value for attribute 'env_index': {env_index}."
                f" Expected a value between 0 and {self._env.config.scene.num_envs - 1}."
            )
        self.cfg.env_index = env_index
        if self.cfg.origin_type == "env":
            self.update_view_to_env()

    def update_view_to_world(self):
        self.cfg.origin_type = "world"
        self.viewer_origin = torch.zeros(3)
        self.update_view_location()

    def update_view_to_env(self):
        self.cfg.origin_type = "env"
        self.viewer_origin = self._env.scene.env_origins[self.cfg.env_index]
        self.update_view_location()

    def update_view_to_asset_root(self, asset_name: str):
        if self.cfg.asset_name != asset_name:
            asset_entities = [
                *self._env.scene.rigid_objects.keys(),
                *self._env.scene.articulations.keys(),
            ]
            if asset_name not in asset_entities:
                raise ValueError(
                    f"Asset '{asset_name}' is not in the scene. Available: {asset_entities}."
                )
        self.cfg.asset_name = asset_name
        self.cfg.origin_type = "asset_root"
        self.viewer_origin = self._env.scene[self.cfg.asset_name].data.root_pos_w[
            self.cfg.env_index
        ]
        self.update_view_location()

    def update_view_location(
        self, eye: Sequence[float] | None = None, lookat: Sequence[float] | None = None
    ):
        if eye is not None:
            self.default_cam_eye = np.asarray(eye)
        if lookat is not None:
            self.default_cam_lookat = np.asarray(lookat)
        viewer_origin = self.viewer_origin.detach().cpu().numpy()
        cam_eye = viewer_origin + self.default_cam_eye
        cam_target = viewer_origin + self.default_cam_lookat
        self._env.sim.set_camera_view(eye=cam_eye, target=cam_target)

    def _update_tracking_callback(self, event):
        if self.cfg.origin_type == "asset_root" and self.cfg.asset_name is not None:
            self.update_view_to_asset_root(self.cfg.asset_name)
