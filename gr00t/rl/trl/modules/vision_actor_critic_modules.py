# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn
from hydra.utils import instantiate
from torch.distributions import Normal

from gr00t.rl.utils.running_mean_std import RunningMeanStd


class VisionActor(nn.Module):

    is_recurrent = False

    def __init__(
        self,
        env_config,
        algo_config,
        backbone,
        module_dim_dict={},
        running_mean_std=False,
        max_rollout_history=1,
        input_key="actor_obs",
    ):
        super(VisionActor, self).__init__()
        self.input_key = input_key

        obs_dim_dict = env_config.robot.algo_obs_dim_dict

        init_noise_std = algo_config.init_noise_std

        self.max_rollout_history = max_rollout_history
        self.vision_module = instantiate(
            backbone.vision_module,
            env_config=env_config,
            algo_config=algo_config,
            obs_dim_dict=obs_dim_dict,
            module_dim_dict=module_dim_dict,
            _recursive_=False,
        )
        self.mlp_module = instantiate(
            backbone.mlp_module,
            env_config=env_config,
            algo_config=algo_config,
            obs_dim_dict=obs_dim_dict,
            module_dim_dict=module_dim_dict,
            _recursive_=False,
        )
        self.vision_module_config_dict = backbone.vision_module.module_config_dict

        self.running_mean_std = None
        if running_mean_std:
            self.running_mean_std = RunningMeanStd(
                (obs_dim_dict[self.input_key],), per_channel=True
            )

        # Action noise
        self.num_actions = self.mlp_module.output_dim
        self.std = nn.Parameter(init_noise_std * torch.ones(self.num_actions))

        if algo_config.get("freeze_noise_std", False):
            self.std.requires_grad = False

        self.clamp_noise_std = algo_config.get("clamp_noise_std", False)
        if self.clamp_noise_std:
            self.max_noise_std = algo_config.get("max_noise_std", 1.0)

        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

        # Initialize observation buffer for rollout
        self.obs_dict_buffer = {}
        self.is_eval_mode = False

    def reset(self, dones=None):
        pass

    def forward(self, obs_dict, **kwargs):
        if self.running_mean_std is not None:
            obs_dict[self.input_key] = self.running_mean_std(obs_dict[self.input_key])

        # Handle both vision_obs and rgb_image_history
        if "rgb_image_history" in obs_dict:
            image_input = obs_dict["rgb_image_history"].clone()
        elif "vision_obs" in obs_dict:
            image_input = obs_dict["vision_obs"].clone()
        else:
            raise ValueError("Neither 'rgb_image_history' nor 'vision_obs' found in obs_dict")

        original_shape = image_input.shape

        if len(original_shape) == 5:
            batch_size, seq_len = original_shape[0], original_shape[1]
            image_reshaped = image_input.reshape(-1, *original_shape[2:])
            effective_batch_size = batch_size * seq_len
        else:
            batch_size = original_shape[0]
            seq_len = None
            image_reshaped = image_input
            effective_batch_size = batch_size

        encoder_type = self.vision_module_config_dict.layer_config.type

        if encoder_type in ["CNN", "ResNet"]:
            image_permuted = image_reshaped.permute(0, 3, 1, 2)
            latent = self.vision_module(image_permuted).reshape(effective_batch_size, -1)
        else:
            flat_image = image_reshaped.reshape(effective_batch_size, -1).contiguous()
            latent = self.vision_module(flat_image).reshape(effective_batch_size, -1)

        if seq_len is not None:
            latent = latent.reshape(batch_size, seq_len, -1).contiguous()

        concated_obs = torch.cat([obs_dict[self.input_key], latent], dim=-1)
        result = self.mlp_module(concated_obs)
        return result

    @property
    def has_normalized_actions(self):
        return False

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, obs_dict, episode_attnmask=None, last_step_only=False, **kwargs):
        mean = self.forward(obs_dict, episode_attnmask=episode_attnmask, **kwargs)
        if last_step_only:
            mean = mean[:, -1]

        if self.clamp_noise_std:
            with torch.no_grad():
                self.std.clamp_(max=self.max_noise_std)

        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, obs_dict, episode_attnmask=None, **kwargs):
        self.update_distribution(obs_dict, episode_attnmask=episode_attnmask, **kwargs)

        actions = self.distribution.sample()
        return {
            "actions": actions,
            "action_mean": self.action_mean,
            "action_sigma": self.action_std,
        }

    def _update_obs_buffer(self, obs_dict, episode_attnmask=None):
        update_episode_attnmask = False
        for key in obs_dict.keys():
            if key not in self.obs_dict_buffer:
                self.obs_dict_buffer[key] = obs_dict[key].unsqueeze(1)
            else:
                self.obs_dict_buffer[key] = torch.cat(
                    [self.obs_dict_buffer[key], obs_dict[key].unsqueeze(1)], dim=1
                )
            if self.obs_dict_buffer[key].shape[1] > self.max_rollout_history:
                update_episode_attnmask = True
                self.obs_dict_buffer[key] = self.obs_dict_buffer[key][
                    :, -self.max_rollout_history :
                ]

        if episode_attnmask is not None and update_episode_attnmask:
            episode_attnmask = episode_attnmask[
                :, -self.max_rollout_history :, -self.max_rollout_history :
            ]

        return episode_attnmask

    def rollout(self, obs_dict, episode_attnmask=None, **kwargs):
        episode_attnmask = self._update_obs_buffer(obs_dict, episode_attnmask)

        self.update_distribution(
            obs_dict=self.obs_dict_buffer,
            episode_attnmask=episode_attnmask,
            last_step_only=True,
            **kwargs,
        )
        return {
            "actions": self.distribution.sample(),
            "action_mean": self.action_mean,
            "action_sigma": self.action_std,
        }

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, obs_dict, episode_attnmask=None, **kwargs):
        episode_attnmask = self._update_obs_buffer(obs_dict, episode_attnmask)
        actions_mean = self.forward(
            obs_dict=self.obs_dict_buffer, episode_attnmask=episode_attnmask, **kwargs
        )
        # last step only
        actions_mean = actions_mean[:, -1]
        return actions_mean

    def to_cpu(self):
        if self.running_mean_std is not None:
            self.running_mean_std.to("cpu")
        self.vision_module = deepcopy(self.vision_module).to("cpu")
        self.mlp_module = deepcopy(self.mlp_module).to("cpu")
        self.std.to("cpu")

    def init_rollout(self):
        """Initialize the observation buffer for rollout phase."""
        self.obs_dict_buffer = {}

    def clear_rollout(self):
        """Clear the observation buffer after rollout phase."""
        self.obs_dict_buffer = {}

    def eval_mode(self):
        self.is_eval_mode = True

    def train_mode(self):
        self.is_eval_mode = False


class VisionCritic(nn.Module):
    def __init__(
        self,
        env_config,
        algo_config,
        backbone,
        module_dim_dict={},
        running_mean_std=False,
    ):
        super(VisionCritic, self).__init__()

        obs_dim_dict = env_config.robot.algo_obs_dim_dict
        self.vision_module = instantiate(
            backbone.vision_module,
            env_config=env_config,
            algo_config=algo_config,
            obs_dim_dict=obs_dim_dict,
            module_dim_dict=module_dim_dict,
            _recursive_=False,
        )
        self.mlp_module = instantiate(
            backbone.mlp_module,
            env_config=env_config,
            algo_config=algo_config,
            obs_dim_dict=obs_dim_dict,
            module_dim_dict=module_dim_dict,
            _recursive_=False,
        )
        self.vision_module_config_dict = backbone.vision_module.module_config_dict
        self.running_mean_std = None
        if running_mean_std:
            self.running_mean_std = RunningMeanStd((obs_dim_dict["critic_obs"],), per_channel=True)

    # @property
    # def critic(self):
    #     return self.critic_module

    def reset(self, dones=None):
        pass

    def evaluate(self, obs_dict, **kwargs):
        # Handle both vision_obs and rgb_image_history
        if "rgb_image_history" in obs_dict:
            image_input = obs_dict["rgb_image_history"].clone()
        elif "vision_obs" in obs_dict:
            image_input = obs_dict["vision_obs"].clone()
        else:
            raise ValueError("Neither 'rgb_image_history' nor 'vision_obs' found in obs_dict")

        original_shape = image_input.shape

        if len(original_shape) == 5:
            batch_size, seq_len = original_shape[0], original_shape[1]
            image_reshaped = image_input.reshape(-1, *original_shape[2:])
            effective_batch_size = batch_size * seq_len
        else:
            batch_size = original_shape[0]
            seq_len = None
            image_reshaped = image_input
            effective_batch_size = batch_size

        encoder_type = self.vision_module_config_dict.layer_config.type

        if encoder_type in ["CNN", "ResNet"]:
            image_permuted = image_reshaped.permute(0, 3, 1, 2)
            latent = self.vision_module(image_permuted).reshape(effective_batch_size, -1)
        else:
            flat_image = image_reshaped.reshape(effective_batch_size, -1).contiguous()
            latent = self.vision_module(flat_image).reshape(effective_batch_size, -1)

        if seq_len is not None:
            latent = latent.reshape(batch_size, seq_len, -1).contiguous()

        concated_obs = torch.cat([obs_dict["critic_obs"], latent], dim=-1)
        value = self.mlp_module(concated_obs)
        return value
