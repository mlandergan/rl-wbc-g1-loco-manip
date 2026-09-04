# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

from copy import deepcopy

import torch
from hydra.utils import instantiate
from torch.distributions import Normal

from gr00t.rl.trl.modules.vision_actor_critic_modules import VisionActor


class VisionActorObjPred(VisionActor):
    """
    VisionActor with object position prediction capability.
    Inherits from VisionActor and adds an additional MLP head for object position prediction.
    Also supports state_history handling with different fusion strategies.
    """

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
        state_history_fusion_strategy="transformer",  # "concat" or "transformer"
        use_temporal_attention=False,  # Enable temporal attention for vision features
    ):
        super(VisionActorObjPred, self).__init__(
            env_config,
            algo_config,
            backbone,
            module_dim_dict,
            running_mean_std,
            max_rollout_history,
            input_key,
        )

        # Initialize object position prediction MLP
        obs_dim_dict = env_config.robot.algo_obs_dim_dict

        # Create object position prediction MLP
        self.obj_pred_mlp = instantiate(
            backbone.obj_pred_mlp,
            env_config=env_config,
            algo_config=algo_config,
            obs_dim_dict=obs_dim_dict,
            module_dim_dict=module_dim_dict,
            _recursive_=False,
        )

        self.state_history_fusion_strategy = state_history_fusion_strategy
        self.state_history_module = None
        self.use_temporal_attention = use_temporal_attention

        state_history_module_config = getattr(backbone, "state_history_module", None)
        if state_history_module_config is not None:
            self.state_history_module = instantiate(
                state_history_module_config,
                env_config=env_config,
                algo_config=algo_config,
                obs_dim_dict=obs_dim_dict,
                module_dim_dict=module_dim_dict,
                _recursive_=False,
            )

        # Initialize temporal attention module for vision features
        self.temporal_attention_module = None
        temporal_attention_config = getattr(backbone, "temporal_attention_module", None)
        if self.use_temporal_attention and temporal_attention_config is not None:
            # Get vision feature dimension from the vision module
            vision_feature_dim = module_dim_dict.get("vision_feature_dim", 512)

            self.temporal_attention_module = instantiate(
                temporal_attention_config,
                env_config=env_config,
                algo_config=algo_config,
                mode="temporal_vision",
                query_dim=vision_feature_dim,
                key_value_dim=vision_feature_dim,
                _recursive_=False,
            )

    def forward(self, obs_dict, **kwargs):
        """
        Forward pass that returns both action predictions and object position predictions.
        Handles state_history_obs if present.
        """
        if self.running_mean_std is not None:
            obs_dict[self.input_key] = self.running_mean_std(obs_dict[self.input_key])

        if (
            self.state_history_fusion_strategy is not None
            and self.state_history_module is not None
            and any(key for key in obs_dict.keys() if "state_history_obs" in key)
        ):

            obs_dict_copy = {
                "actor_obs": obs_dict["actor_obs"],
                "state_history_obs": obs_dict["state_history_obs"],
            }

            # Input:
            # - actor_obs: [batch_size, obs_dim] in rollout, [batch_size, num_seq, obs_dim] in training loop
            # - state_history_obs: [batch_size, num_seq, history_length, obs_dim]
            # Output:
            # - features: [batch_size, num_seq, feature_dim]
            if self.state_history_fusion_strategy == "transformer":
                state_history_features = self.state_history_module(obs_dict_copy, **kwargs)
            elif self.state_history_fusion_strategy == "concat":
                obs_dict_copy["state_history_obs"] = torch.flatten(
                    obs_dict_copy["state_history_obs"], start_dim=2
                )
                obs_dict_copy["actor_obs"] = torch.cat(
                    [obs_dict_copy["actor_obs"], obs_dict_copy["state_history_obs"]], dim=-1
                )
                state_history_features = self.state_history_module(obs_dict_copy, **kwargs)

        else:
            # only single frame actor_obs
            state_history_features = obs_dict["actor_obs"]

        # Handle vision processing for both action and object prediction
        if "rgb_image_history" in obs_dict:
            image_input = obs_dict["rgb_image_history"].clone()
        elif "vision_obs" in obs_dict:
            image_input = obs_dict["vision_obs"].clone()
        else:
            raise ValueError("Neither 'rgb_image_history' nor 'vision_obs' found in obs_dict")

        original_shape = image_input.shape

        if len(original_shape) == 6:
            batch_size, num_seq, history_length, H, W, C = original_shape
            image_reshaped = image_input.reshape(-1, H, W, C)
        elif len(original_shape) == 5:
            batch_size, num_seq, H, W, C = original_shape
            history_length = 1
            image_reshaped = image_input.reshape(-1, H, W, C)
        elif len(original_shape) == 4:
            batch_size, H, W, C = original_shape
            num_seq, history_length = 1, 1
            image_reshaped = image_input
        else:
            raise ValueError(f"Unsupported image input shape: {original_shape}")

        # [N, H, W, C] to [N, C, H, W]
        image_reshaped = image_reshaped.permute(0, 3, 1, 2)

        # [batch_size * num_seq * history_length, C, H, W] to [batch_size * num_seq * history_length, feature_dim]
        vision_features = self.vision_module(image_reshaped)

        # [batch_size * num_seq * history_length, feature_dim] to [batch_size, num_seq, history_length, feature_dim]
        feature_dim = vision_features.shape[-1]
        vision_features = vision_features.reshape(batch_size, num_seq, history_length, feature_dim)

        # Temporal aggregation: [batch_size, num_seq, history_length, feature_dim] to [batch_size, num_seq, aggregated_feature_dim]
        if (
            self.use_temporal_attention
            and self.temporal_attention_module is not None
            and history_length > 1
        ):
            # Current frame as query, all history frames as key/value
            # Output: [batch_size, num_seq, feature_dim]
            aggregated_vision_features = self.temporal_attention_module(vision_features)
        elif history_length > 1:
            # Concatenate features along history dimension: [batch_size, num_seq, history_length * feature_dim]
            aggregated_vision_features = vision_features.reshape(
                batch_size, num_seq, history_length * feature_dim
            )
        else:
            aggregated_vision_features = vision_features.squeeze(2)

        combined_features = [state_history_features, aggregated_vision_features]
        combined_features = torch.cat(combined_features, dim=-1)

        actions = self.mlp_module(combined_features)

        obj_pred = self.obj_pred_mlp(aggregated_vision_features)

        return {"actions": actions, "obj_pred": obj_pred}

    def update_distribution(self, obs_dict, episode_attnmask=None, last_step_only=False, **kwargs):
        """
        Update action distribution (only uses action predictions).
        """
        result = self.forward(obs_dict, episode_attnmask=episode_attnmask, **kwargs)
        mean = result["actions"]

        if last_step_only:
            mean = mean[:, -1]

        if self.clamp_noise_std:
            with torch.no_grad():
                self.std.clamp_(max=self.max_noise_std)

        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, obs_dict, episode_attnmask=None, **kwargs):
        """
        Action sampling (only uses action predictions).
        """
        self.update_distribution(obs_dict, episode_attnmask=episode_attnmask, **kwargs)

        actions = self.distribution.sample()
        return {
            "actions": actions,
            "action_mean": self.action_mean,
            "action_sigma": self.action_std,
        }

    def act_with_obj_pred(self, obs_dict, episode_attnmask=None, **kwargs):
        """
        Action sampling with object position prediction for distillation training.
        """
        # Single forward pass to get both action and object position predictions
        result = self.forward(obs_dict, episode_attnmask=episode_attnmask, **kwargs)
        action_mean = result["actions"]
        obj_pos_pred = result["obj_pred"]

        # Create distribution using the cached action mean
        if self.clamp_noise_std:
            with torch.no_grad():
                self.std.clamp_(max=self.max_noise_std)

        self.distribution = Normal(action_mean, action_mean * 0.0 + self.std)

        actions = self.distribution.sample()
        return {
            "actions": actions,
            "action_mean": self.action_mean,
            "action_sigma": self.action_std,
            "predicted_obj_pos": obj_pos_pred,
        }

    def action_mean_with_obj_pred(self, obs_dict, episode_attnmask=None, **kwargs):
        """
        Action sampling with object position prediction for distillation training.
        """
        # Single forward pass to get both action and object position predictions
        result = self.forward(obs_dict, episode_attnmask=episode_attnmask, **kwargs)
        action_mean = result["actions"]
        obj_pos_pred = result["obj_pred"]

        # Create distribution using the cached action mean
        if self.clamp_noise_std:
            with torch.no_grad():
                self.std.clamp_(max=self.max_noise_std)

        self.distribution = Normal(action_mean, action_mean * 0.0 + self.std)

        actions = self.distribution.sample()
        return {
            "actions": self.action_mean,
            "action_mean": self.action_mean,
            "action_sigma": self.action_std,
            "predicted_obj_pos": obj_pos_pred,
        }

    def rollout_with_obj_pred(self, obs_dict, episode_attnmask=None, **kwargs):
        """
        Rollout with object position prediction (handles observation buffering properly).
        """
        episode_attnmask = self._update_obs_buffer(obs_dict, episode_attnmask)

        # Forward pass using buffered observations
        result = self.forward(
            obs_dict=self.obs_dict_buffer, episode_attnmask=episode_attnmask, **kwargs
        )
        action_mean = result["actions"]
        obj_pos_pred = result["obj_pred"]

        # Take last step only for rollout
        if len(action_mean.shape) == 3:
            action_mean = action_mean[:, -1]
            obj_pos_pred = obj_pos_pred[:, -1]

        # Create distribution
        if self.clamp_noise_std:
            with torch.no_grad():
                self.std.clamp_(max=self.max_noise_std)

        self.distribution = Normal(action_mean, action_mean * 0.0 + self.std)

        actions = self.distribution.sample()
        return {
            "actions": self.action_mean,
            "action_mean": self.action_mean,
            "action_sigma": self.action_std,
            "predicted_obj_pos": obj_pos_pred,
        }

    def rollout(self, obs_dict, episode_attnmask=None, **kwargs):
        """
        Rollout (only uses action predictions).
        """
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

    def act_inference(self, obs_dict, episode_attnmask=None, **kwargs):
        """
        Inference mode (only uses action predictions).
        """
        episode_attnmask = self._update_obs_buffer(obs_dict, episode_attnmask)
        result = self.forward(
            obs_dict=self.obs_dict_buffer, episode_attnmask=episode_attnmask, **kwargs
        )
        actions_mean = result["actions"]
        # last step only
        actions_mean = actions_mean[:, -1]
        return actions_mean

    def to_cpu(self):
        """
        Move model to CPU.
        """
        if self.running_mean_std is not None:
            self.running_mean_std.to("cpu")
        self.vision_module = deepcopy(self.vision_module).to("cpu")
        self.mlp_module = deepcopy(self.mlp_module).to("cpu")
        self.obj_pred_mlp = deepcopy(self.obj_pred_mlp).to("cpu")
        if self.state_history_module is not None:
            self.state_history_module = deepcopy(self.state_history_module).to("cpu")
        if self.temporal_attention_module is not None:
            self.temporal_attention_module = deepcopy(self.temporal_attention_module).to("cpu")
        self.std.to("cpu")
