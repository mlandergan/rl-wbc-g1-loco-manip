# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import torch
from loguru import logger
from termcolor import colored
from torch import Tensor


class HistoryHandler:

    def __init__(self, num_envs, history_config, obs_dims, device, env_config=None):
        self.obs_dims = obs_dims
        self.device = device
        self.num_envs = num_envs
        self.env_config = env_config
        self.history = {}

        self.buffer_config = {}
        for aux_key, aux_config in history_config.items():
            for obs_key, obs_num in aux_config.items():
                if obs_key in self.buffer_config:
                    self.buffer_config[obs_key] = max(self.buffer_config[obs_key], obs_num)
                else:
                    self.buffer_config[obs_key] = obs_num

        for key in self.buffer_config.keys():
            # Handle different observation shapes
            if key == "rgb_image":
                # For vision observations, read camera resolution directly from env config
                if (
                    env_config is not None
                    and hasattr(env_config, "simulator")
                    and hasattr(env_config.simulator, "config")
                ):
                    camera_config = env_config.simulator.config.cameras
                    if hasattr(camera_config, "camera_resolutions"):
                        camera_resolutions = camera_config.camera_resolutions
                        # Determine number of channels based on camera type
                        if hasattr(camera_config, "camera_types"):
                            camera_types = camera_config.camera_types
                            if isinstance(camera_types, list) and len(camera_types) > 0:
                                camera_type = camera_types[0]
                                if "depth" in camera_type and camera_type.get("depth", False):
                                    if (
                                        hasattr(camera_config, "stack_depth_image")
                                        and camera_config.stack_depth_image
                                    ):
                                        num_channels = 3
                                    else:
                                        num_channels = 1
                                else:
                                    num_channels = 3  # RGB
                            else:
                                num_channels = 3  # Default to RGB
                        else:
                            num_channels = 3  # Default to RGB

                        # Create shape tuple: (H, W, C)
                        shape = tuple(camera_resolutions + [num_channels])
                        logger.info(f"Using camera resolution from config: {shape}")
                    else:
                        logger.warning(
                            "Camera resolutions not found in config, falling back to inference"
                        )
                        shape = self._infer_image_shape_from_obs_dim(obs_dims[key])
                else:
                    raise ValueError("Environment config not provided, falling back to inference")

                # Create history buffer: [num_envs, history_length, H, W, C]
                self.history[key] = torch.zeros(
                    num_envs, self.buffer_config[key], *shape, device=self.device
                )
                logger.info(
                    f"Vision history buffer for {key}: shape {shape}, history length {self.buffer_config[key]}"
                )
                logger.info(f"Total history buffer shape: {self.history[key].shape}")
            else:
                # Regular scalar/vector observations: [num_envs, history_length, obs_dim]
                self.history[key] = torch.zeros(
                    num_envs, self.buffer_config[key], obs_dims[key], device=self.device
                )

        logger.info(colored("History Handler Initialized", "green"))
        for key, value in self.buffer_config.items():
            logger.info(f"Key: {key}, Value: {value}")

    def reset(self, reset_ids):
        if len(reset_ids) == 0:
            return
        assert set(self.buffer_config.keys()) == set(
            self.history.keys()
        ), f"History keys mismatch\n{self.buffer_config.keys()}\n{self.history.keys()}"
        for key in self.history.keys():
            self.history[key][reset_ids] *= 0.0

    def add(self, key: str, value: Tensor):
        assert key in self.history.keys(), f"Key {key} not found in history"
        # Handle vision observations with shape reconstruction
        if key == "rgb_image":
            # Ensure value has the right shape [num_envs, H, W, C]
            expected_shape = self.history[key].shape[2:]  # (H, W, C)
            if value.shape[1:] != expected_shape:
                raise ValueError(
                    f"Vision observation shape mismatch: got {value.shape[1:]}, expected {expected_shape}"
                )

        # Shift history: move all entries one step back
        val = self.history[key].clone()
        # self.history[key][:, 1:] = val[:, :-1]
        # self.history[key][:, 0] = value.clone()
        self.history[key][:, :-1] = val[:, 1:]
        self.history[key][:, -1] = value.clone()

    def query(self, key: str):
        """Return the full history buffer for a key."""
        assert key in self.history.keys(), f"Key {key} not found in history"
        return self.history[key].clone()

    def query_for_actor(self, key: str):
        """
        Query history formatted for vision actors.
        For vision observations, returns [num_envs, history_length, H, W, C]
        For other observations, returns [num_envs, history_length, obs_dim]
        """
        if key not in self.history:
            raise ValueError(f"Key {key} not found in history")

        history = self.query(key)

        if key == "rgb_image":
            # For vision, return as [num_envs, history_length, H, W, C]
            # This is already the format we store, so just return it
            return history
        else:
            # For other observations, return as [num_envs, history_length, obs_dim]
            return history
