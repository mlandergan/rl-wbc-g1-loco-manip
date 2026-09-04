# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import copy
import os
from typing import Any, Dict, List

import torch
from loguru import logger
from omegaconf import OmegaConf


def class_to_dict(obj) -> dict:
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result


def pre_process_config(config) -> None:

    # compute observation_dim
    # config.robot.policy_obs_dim = -1
    # config.robot.critic_obs_dim = -1

    obs_dim_dict = dict()
    _obs_key_list = config.env.config.obs.obs_dict
    _aux_obs_key_list = config.env.config.obs.obs_auxiliary

    assert set(config.env.config.obs.noise_scales.keys()) == set(
        config.env.config.obs.obs_scales.keys()
    )

    # convert obs_dims to list of dicts
    each_dict_obs_dims = {k: v for d in config.env.config.obs.obs_dims for k, v in d.items()}
    config.env.config.obs.obs_dims = each_dict_obs_dims
    logger.info(f"obs_dims: {each_dict_obs_dims}")

    # parse config.env.config.obs.obs_dims, turn string to integer BEFORE processing auxiliary observations
    for key, value in config.env.config.obs.obs_dims.items():
        if isinstance(value, str):
            config.env.config.obs.obs_dims[key] = eval(value)

    auxiliary_obs_dims = {}
    for aux_obs_key, aux_config in _aux_obs_key_list.items():
        auxiliary_obs_dims[aux_obs_key] = 0
        for _key, _num in aux_config.items():
            if _key not in config.env.config.obs.obs_dims:
                raise KeyError(f"auxiliary observation {aux_obs_key!r} references unknown key {_key!r}")
            auxiliary_obs_dims[aux_obs_key] += config.env.config.obs.obs_dims[_key] * _num
    logger.info(f"auxiliary_obs_dims: {auxiliary_obs_dims}")

    for obs_key, obs_config in _obs_key_list.items():
        obs_dim_dict[obs_key] = 0
        for key in obs_config:
            if key.endswith("_raw"):
                key = key[:-4]
            past_length = config.env.config.obs.get("past_length", 1)
            if key in config.env.config.obs.obs_dims.keys():
                obs_dim_dict[obs_key] += config.env.config.obs.obs_dims[key] * past_length
                logger.info(f"{obs_key}: {key} has dim: {config.env.config.obs.obs_dims[key]}")
            else:
                obs_dim_dict[obs_key] += auxiliary_obs_dims[key] * past_length
                logger.info(f"{obs_key}: {key} has dim: {auxiliary_obs_dims[key]}")
    config.robot.algo_obs_dim_dict = obs_dim_dict

    OmegaConf.set_struct(config.env.config.obs.obs_dims, False)
    config.env.config.obs.obs_dims.update(
        auxiliary_obs_dims
    )  # ZL: adding auxiliary obs dims to obs_dims

    logger.info(f"algo_obs_dim_dict: {config.robot.algo_obs_dim_dict}")

    # compute action_dim for ppo
    # for agent in config.algo.config.network_dict.keys():
    #     for network in config.algo.config.network_dict[agent].keys():
    #         output_dim = config.algo.config.network_dict[agent][network].output_dim
    #         if output_dim == "action_dim":
    #             config.algo.config.network_dict[agent][network].output_dim = config.env.config.robot.actions_dim

    # print the config

    logger.debug("PPO CONFIG")
    if hasattr(config.algo.config, "module_dict"):
        logger.debug(f"{config.algo.config.module_dict}")
    # logger.debug(f"{config.algo.config.network_dict}")


def parse_observation(
    cls: Any,
    key_list: List,
    buf_dict: Dict,
    obs_scales: Dict,
    noise_scales: Dict,
    current_noise_curriculum_value: Any,
    use_noise: bool = True,
) -> None:
    """Parse observations for the legged_robot_base class"""
    # TOOD: Parse observations for manipulation tasks

    for obs_key in key_list:
        if use_noise:
            obs_noise = noise_scales[obs_key] * current_noise_curriculum_value
        else:
            obs_noise = 0.0

        # print(f"obs_key: {obs_key}, obs_noise: {obs_noise}")

        actor_obs = getattr(cls, f"_get_obs_{obs_key}")().clone()
        obs_scale = obs_scales[obs_key]

        buf_dict[obs_key] = (
            actor_obs + (torch.rand_like(actor_obs) * 2.0 - 1.0) * obs_noise
        ) * obs_scale


def export_policy_as_jit(actor_critic, path):
    if hasattr(actor_critic, "memory_a"):
        # assumes LSTM: TODO add GRU
        exporter = PolicyExporterLSTM(actor_critic)
        exporter.export(path)
    else:
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, "policy_1.pt")
        model = copy.deepcopy(actor_critic.actor).to("cpu")
        traced_script_module = torch.jit.script(model)
        traced_script_module.save(path)


class PolicyExporterLSTM(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.is_recurrent = actor_critic.is_recurrent
        self.memory = copy.deepcopy(actor_critic.memory_a.rnn)
        self.memory.cpu()
        self.register_buffer(
            "hidden_state", torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size)
        )
        self.register_buffer(
            "cell_state", torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size)
        )

    def forward(self, x):
        out, (h, c) = self.memory(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h
        self.cell_state[:] = c
        return self.actor(out.squeeze(0))

    @torch.jit.export
    def reset_memory(self):
        self.hidden_state[:] = 0.0
        self.cell_state[:] = 0.0

    def export(self, path):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, "policy_lstm_1.pt")
        self.to("cpu")
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)


def dict_to_true_list(input_dict):
    """Convert a dictionary or list of dictionaries to a list containing only keys where values are True.

    Args:
        input_dict (dict or list[dict]): Input dictionary or list of dictionaries to process

    Returns:
        list: List of keys where values are True
    """
    # Handle list of dictionaries
    result = []
    for d in input_dict:
        result.extend([key for key, value in d.items() if value is True])
    return result
