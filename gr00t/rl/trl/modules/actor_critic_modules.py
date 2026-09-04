# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal

from gr00t.rl.trl.utils.common import custom_instantiate
from gr00t.rl.trl.utils.rl import compute_episode_attnmask
from gr00t.rl.utils.running_mean_std import RunningMeanStd


class Actor(nn.Module):

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
        super(Actor, self).__init__()

        self.algo_config = algo_config
        obs_dim_dict = env_config.robot.algo_obs_dim_dict
        self.input_key = input_key

        init_noise_std = algo_config.init_noise_std

        self.max_rollout_history = max_rollout_history
        self.actor_module = custom_instantiate(
            backbone,
            env_config=env_config,
            algo_config=algo_config,
            obs_dim_dict=obs_dim_dict,
            module_dim_dict=module_dim_dict,
            _resolve=False,
        )

        self.running_mean_std = None
        if running_mean_std:
            self.running_mean_std = RunningMeanStd(
                (obs_dim_dict[self.input_key],), per_channel=True
            )

        # Action noise
        self.num_actions = self.actor_module.output_dim
        self.std = nn.Parameter(init_noise_std * torch.ones(self.num_actions))

        if algo_config.get("freeze_noise_std", False):
            self.std.requires_grad = False

        self.clamp_noise_std = algo_config.get("clamp_noise_std", False)
        if self.clamp_noise_std:
            self.max_noise_std = algo_config.get("max_noise_std", 1.0)
            self.min_noise_std = algo_config.get("min_noise_std", 0.0)

        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

        # Initialize observation buffer for rollout
        self.obs_dict_buffer = TensorDict()
        self.dones_buffer = None
        self.steps = 0
        self.is_eval_mode = False

    def reset(self, dones=None):
        pass

    def forward(self, obs_dict, **kwargs):
        obs_dict = obs_dict.copy()
        if self.running_mean_std is not None:
            with torch.no_grad():
                obs_dict[self.input_key] = self.running_mean_std(obs_dict[self.input_key])

        if self.algo_config.get("use_clampped_std", False):
            self.std.clamp_(min=self.algo_config.std_clamp_min, max=self.algo_config.std_clamp_max)

        return self.actor_module(obs_dict[self.input_key], **kwargs)

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
        # obs_dict[self.input_key] = torch.zeros_like(obs_dict[self.input_key]) # DEBUG
        # logger.warning(f"obs_dict[self.input_key]: {obs_dict[self.input_key]}")
        mean = self.forward(obs_dict, episode_attnmask=episode_attnmask, **kwargs)
        if last_step_only:
            mean = mean[:, -1]

        if self.clamp_noise_std:
            with torch.no_grad():
                self.std.clamp_(max=self.max_noise_std)
        # import ipdb; ipdb.set_trace()
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, obs_dict, episode_attnmask=None, **kwargs):
        self.update_distribution(obs_dict, episode_attnmask=episode_attnmask, **kwargs)
        actions = self.distribution.sample()
        return TensorDict(
            {
                "actions": actions,
                "action_mean": self.action_mean,
                "action_sigma": self.action_std,
            }
        )

    def update_dones_buffer_and_compute_episode_attnmask(self, cur_dones):
        if self.steps > 0 and self.max_rollout_history > 1:
            if self.dones_buffer is None:
                self.dones_buffer = cur_dones.clone().unsqueeze(1)
            else:
                self.dones_buffer = torch.cat(
                    [self.dones_buffer, cur_dones.clone().unsqueeze(1)], dim=1
                )
            if self.dones_buffer.shape[1] > self.max_rollout_history - 1:
                self.dones_buffer = self.dones_buffer[:, -self.max_rollout_history + 1 :]
            dones = torch.cat(
                [self.dones_buffer, torch.zeros_like(self.dones_buffer[:, :1])], dim=1
            )
        else:
            dones = torch.zeros_like(cur_dones.unsqueeze(1))
        episode_attnmask_from_dones = compute_episode_attnmask(dones)
        return episode_attnmask_from_dones

    def _update_obs_buffer(self, obs_dict, episode_attnmask=None, cur_dones=None):
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

        if cur_dones is not None:
            episode_attnmask_from_dones = self.update_dones_buffer_and_compute_episode_attnmask(
                cur_dones
            )
            if episode_attnmask is not None:
                assert (episode_attnmask == episode_attnmask_from_dones).all()
            if episode_attnmask is None:
                episode_attnmask = episode_attnmask_from_dones

        return episode_attnmask

    def rollout(self, obs_dict, episode_attnmask=None, cur_dones=None, **kwargs):
        episode_attnmask = self._update_obs_buffer(obs_dict, episode_attnmask, cur_dones)
        self.update_distribution(
            obs_dict=self.obs_dict_buffer,
            episode_attnmask=episode_attnmask,
            last_step_only=True,
            **kwargs,
        )
        self.steps += 1
        return TensorDict(
            {
                "actions": self.distribution.sample(),
                "action_mean": self.action_mean,
                "action_sigma": self.action_std,
            }
        )

    def rollout_with_obj_pred(self, obs_dict, episode_attnmask=None, cur_dones=None, **kwargs):
        """
        Rollout method compatible with object prediction trainer.
        This base Actor class doesn't predict object positions, so it doesn't return predicted_obj_pos.
        """
        episode_attnmask = self._update_obs_buffer(obs_dict, episode_attnmask, cur_dones)
        self.update_distribution(
            obs_dict=self.obs_dict_buffer,
            episode_attnmask=episode_attnmask,
            last_step_only=True,
            **kwargs,
        )
        self.steps += 1
        return TensorDict(
            {
                "actions": self.distribution.sample(),
                "action_mean": self.action_mean,
                "action_sigma": self.action_std,
                # Note: predicted_obj_pos is NOT included since this base actor doesn't predict object positions
            }
        )

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, obs_dict, episode_attnmask=None, cur_dones=None, **kwargs):
        episode_attnmask = self._update_obs_buffer(obs_dict, episode_attnmask, cur_dones)
        actions_mean = self.forward(
            obs_dict=self.obs_dict_buffer, episode_attnmask=episode_attnmask, **kwargs
        )
        # last step only
        actions_mean = actions_mean[:, -1]
        self.steps += 1
        return actions_mean

    def act_pure_inference(self, obs_dict, episode_attnmask=None, **kwargs):
        """
        Pure inference mode for temporal models like transformer.
        Need to construct obs_dict and episode_attnmask outside the model.
        """
        actions_mean = self.forward(obs_dict=obs_dict, episode_attnmask=episode_attnmask, **kwargs)
        # last step only
        actions_mean = actions_mean[:, -1]
        return actions_mean

    def to_cpu(self):
        if self.running_mean_std is not None:
            self.running_mean_std.to("cpu")
        self.actor = deepcopy(self.actor).to("cpu")
        self.std.to("cpu")

    def init_rollout(self):
        """Initialize the observation buffer for rollout phase."""
        self.obs_dict_buffer = TensorDict()
        self.dones_buffer = None
        self.steps = 0

    def clear_rollout(self):
        """Clear the observation buffer after rollout phase."""
        self.obs_dict_buffer = TensorDict()
        self.dones_buffer = None
        self.steps = 0
        del self.distribution

    def eval_mode(self):
        self.is_eval_mode = True

    def train_mode(self):
        self.is_eval_mode = False


class Critic(nn.Module):
    def __init__(
        self, env_config, algo_config, backbone, module_dim_dict={}, running_mean_std=False
    ):
        super(Critic, self).__init__()

        obs_dim_dict = env_config.robot.algo_obs_dim_dict
        self.critic_module = custom_instantiate(
            backbone,
            env_config=env_config,
            algo_config=algo_config,
            obs_dim_dict=obs_dim_dict,
            module_dim_dict=module_dim_dict,
            _resolve=False,
        )

        self.running_mean_std = None
        if running_mean_std:
            self.running_mean_std = RunningMeanStd((obs_dim_dict["critic_obs"],), per_channel=True)

    @property
    def critic(self):
        return self.critic_module

    def reset(self, dones=None):
        pass

    def evaluate(self, obs_dict, **kwargs):
        obs_dict = obs_dict
        if self.running_mean_std is not None:
            with torch.no_grad():
                obs_dict["critic_obs"] = self.running_mean_std(obs_dict["critic_obs"])
        value = self.critic(obs_dict, **kwargs)
        return value
