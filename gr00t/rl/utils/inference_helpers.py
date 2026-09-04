# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import copy
import os

import torch
from torch import nn


def export_policy_as_jit(actor_critic, path, exported_policy_name):
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, exported_policy_name)
    model = copy.deepcopy(actor_critic.actor).to("cpu")
    traced_script_module = torch.jit.script(model)
    traced_script_module.save(path)


def export_policy_as_onnx(inference_model, path, exported_policy_name, example_obs_dict):
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, exported_policy_name)

    actor = copy.deepcopy(inference_model["actor"]).to("cpu")
    actor.eval()

    class PPOWrapper(nn.Module):
        def __init__(self, actor):
            """
            model: The original PyTorch model.
            input_keys: List of input names as keys for the input dictionary.
            """
            super(PPOWrapper, self).__init__()
            self.actor = actor

        def forward(self, obs_dict):
            """
            Dynamically creates a dictionary from the input keys and args.
            """
            return self.actor.act_inference(obs_dict)

    wrapper = PPOWrapper(actor)
    example_input_list = {"obs_dict": example_obs_dict}
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            example_input_list,  # Pass x1 and x2 as separate inputs
            path,
            verbose=True,
            input_names=["obs_dict"],  # Specify the input names
            output_names=["action"],  # Name the output
            opset_version=13,  # Specify the opset version, if needed
        )


def export_policy_z_as_onnx(inference_model, env, path, exported_policy_name, example_obs_dict):
    ###### Harded coded for now.

    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, exported_policy_name)

    actor = copy.deepcopy(inference_model["actor"]).to("cpu")
    z_actor = copy.deepcopy(env.z_actor).to("cpu")
    prop_obs_filter = torch.zeros(env.dim_obs, dtype=torch.bool)
    actor_obs_keys = sorted(env.config.obs["obs_dict"]["actor_obs"])
    counter = 0

    for key in actor_obs_keys:
        obs_dim = env.config.obs.obs_dims[key]
        if key in env.z_config.obs.obs_dict.proprioception_obs:
            prop_obs_filter[counter : counter + obs_dim] = True
        counter += obs_dim
    prop_obs_filter = prop_obs_filter.nonzero().squeeze()

    class PPOWrapper(nn.Module):
        def __init__(self, actor, z_actor, prop_obs_filter):
            """
            model: The original PyTorch model.
            input_keys: List of input names as keys for the input dictionary.
            """
            super(PPOWrapper, self).__init__()
            self.actor = actor
            self.z_actor = z_actor
            self.prop_obs_filter = prop_obs_filter

        def forward(self, actor_obs):
            """
            Dynamically creates a dictionary from the input keys and args.
            """
            prop_obs = actor_obs[:, self.prop_obs_filter].clone()
            actor_z = self.actor({"actor_obs": actor_obs})
            prop_obs_mean = self.z_actor.running_mean_std_proprioception(prop_obs)
            x = self.z_actor.prior(prop_obs_mean)
            prior_mu = self.z_actor.prior_mu_layer(x)
            new_z = prior_mu + actor_z
            actions_mean = self.z_actor.motion_decoder(torch.cat([prop_obs_mean, new_z], dim=-1))
            return actions_mean

    wrapper = PPOWrapper(actor, z_actor, prop_obs_filter)
    example_input_list = example_obs_dict["actor_obs"]
    torch.onnx.export(
        wrapper,
        example_input_list,  # Pass x1 and x2 as separate inputs
        path,
        verbose=True,
        input_names=["actor_obs"],  # Specify the input names
        output_names=["action"],  # Name the output
        opset_version=13,  # Specify the opset version, if needed
    )


def export_policy_CNN_as_onnx(inference_model, env, path, exported_policy_name, example_obs_dict):
    ###### Harded coded for now.

    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, exported_policy_name)

    # import ipdb; ipdb.set_trace()
    try:
        actor_mlp = copy.deepcopy(inference_model["actor"].actor_module.mlp_module).to("cpu")
        actor_encoder = copy.deepcopy(inference_model["actor"].actor_module.encoder).to("cpu")
        if inference_model["actor"].running_mean_std is not None:
            running_mean_std = copy.deepcopy(inference_model["actor"].running_mean_std).to("cpu")
        else:
            running_mean_std = None
    except:
        actor_mlp = copy.deepcopy(inference_model["actor"].mlp_module).to("cpu")
        actor_encoder = copy.deepcopy(inference_model["actor"].vision_module).to("cpu")
        if inference_model["actor"].running_mean_std is not None:
            running_mean_std = copy.deepcopy(inference_model["actor"].running_mean_std).to("cpu")
        else:
            running_mean_std = None

    class PPOWrapper(nn.Module):
        def __init__(self, actor_mlp, actor_encoder, running_mean_std):
            """
            model: The original PyTorch model.
            input_keys: List of input names as keys for the input dictionary.
            """
            super(PPOWrapper, self).__init__()
            self.actor_mlp = actor_mlp
            self.actor_encoder = actor_encoder
            self.running_mean_std = running_mean_std

        def forward(self, inputs):
            """
            Dynamically creates a dictionary from the input keys and args.
            """
            actor_obs, vision_obs = inputs
            if self.running_mean_std is not None:
                actor_obs = self.running_mean_std(actor_obs)
            latent = self.actor_encoder(vision_obs.permute(0, 3, 1, 2))
            concated_obs = torch.cat([actor_obs, latent], dim=-1)
            actions_mean = self.actor_mlp(concated_obs)
            return actions_mean

    wrapper = PPOWrapper(actor_mlp, actor_encoder, running_mean_std)
    example_input_list = [example_obs_dict["actor_obs"], example_obs_dict["vision_obs"]]
    # import ipdb; ipdb.set_trace()
    torch.onnx.export(
        wrapper,
        example_input_list,  # Pass x1 and x2 as separate inputs
        path,
        verbose=True,
        input_names=["actor_obs", "vision_obs"],  # Specify the input names
        output_names=["action"],  # Name the output
        opset_version=13,  # Specify the opset version, if needed
    )


def export_policy_CNN_as_onnx_with_obj_pred(
    inference_model, env, path, exported_policy_name, example_obs_dict
):

    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, exported_policy_name)

    # import ipdb; ipdb.set_trace()
    try:
        actor_mlp = copy.deepcopy(inference_model["actor"].actor_module.mlp_module).to("cpu")
        actor_encoder = copy.deepcopy(inference_model["actor"].actor_module.encoder).to("cpu")
        obj_pred_mlp = copy.deepcopy(inference_model["actor"].actor_module.obj_pred_mlp).to("cpu")
        if inference_model["actor"].running_mean_std is not None:
            running_mean_std = copy.deepcopy(inference_model["actor"].running_mean_std).to("cpu")
        else:
            running_mean_std = None
        if (
            hasattr(inference_model["actor"], "is_recurrent")
            and inference_model["actor"].is_recurrent
        ):
            memory = copy.deepcopy(inference_model["actor"].memory).to("cpu")
        else:
            memory = None

    except:
        actor_mlp = copy.deepcopy(inference_model["actor"].mlp_module).to("cpu")
        actor_encoder = copy.deepcopy(inference_model["actor"].vision_module).to("cpu")
        obj_pred_mlp = copy.deepcopy(inference_model["actor"].obj_pred_mlp).to("cpu")
        if inference_model["actor"].running_mean_std is not None:
            running_mean_std = copy.deepcopy(inference_model["actor"].running_mean_std).to("cpu")
        else:
            running_mean_std = None
        if (
            hasattr(inference_model["actor"], "is_recurrent")
            and inference_model["actor"].is_recurrent
        ):
            memory = copy.deepcopy(inference_model["actor"].memory).to("cpu")
        else:
            memory = None

    class PPOWrapper(nn.Module):
        def __init__(self, actor_mlp, actor_encoder, obj_pred_mlp, running_mean_std, memory):
            """
            model: The original PyTorch model.
            input_keys: List of input names as keys for the input dictionary.
            """
            super(PPOWrapper, self).__init__()
            self.actor_mlp = actor_mlp
            self.actor_encoder = actor_encoder
            self.obj_pred_mlp = obj_pred_mlp
            self.running_mean_std = running_mean_std
            self.memory = memory

        def forward(self, inputs):
            """
            Dynamically creates a dictionary from the input keys and args.
            """
            if self.memory is not None:
                if isinstance(self.memory.rnn, nn.LSTM):
                    actor_obs, vision_obs, hidden_state, cell_state = inputs
                    hidden_states = (
                        hidden_state.transpose(0, 1),
                        cell_state.transpose(0, 1),
                    )  # [num_layers, batch_size, hidden_dim]
                elif isinstance(self.memory.rnn, nn.GRU):
                    actor_obs, vision_obs, hidden_states = inputs
                    hidden_states = hidden_states.transpose(
                        0, 1
                    )  # [num_layers, batch_size, hidden_dim]
                else:
                    raise ValueError(f"Unsupported memory type: {type(self.memory.rnn)}")
            else:
                actor_obs, vision_obs = inputs

            if self.running_mean_std is not None:
                actor_obs = self.running_mean_std(actor_obs)

            latent = self.actor_encoder(vision_obs.permute(0, 3, 1, 2))
            concated_obs = torch.cat([actor_obs, latent], dim=-1)

            if self.memory is not None:
                concated_obs = concated_obs.unsqueeze(0)
                concated_obs, new_hidden_states = self.memory(
                    concated_obs,
                    hidden_states=hidden_states,
                    return_new_hidden_states=True,
                    masks=1,
                )
                concated_obs = concated_obs.squeeze(1)
                if isinstance(new_hidden_states, tuple):
                    new_hidden_states = tuple(
                        new_hidden_state.transpose(0, 1) for new_hidden_state in new_hidden_states
                    )  # [batch_size, num_layers, hidden_dim]
                else:
                    new_hidden_states = (
                        new_hidden_states.transpose(0, 1),
                    )  # [batch_size, num_layers, hidden_dim]
            actions_mean = self.actor_mlp(concated_obs)
            obj_3d_pos = self.obj_pred_mlp(latent)

            if self.memory is not None:
                return actions_mean, obj_3d_pos, *new_hidden_states
            else:
                return actions_mean, obj_3d_pos

    wrapper = PPOWrapper(actor_mlp, actor_encoder, obj_pred_mlp, running_mean_std, memory)
    example_input_list = [example_obs_dict["actor_obs"], example_obs_dict["vision_obs"]]
    # import ipdb; ipdb.set_trace()
    input_names = ["actor_obs", "vision_obs"]
    output_names = ["action", "obj_3d_pos"]
    if wrapper.memory is not None:
        hidden_size = wrapper.memory.rnn.hidden_size
        num_layers = wrapper.memory.rnn.num_layers
        if isinstance(wrapper.memory.rnn, nn.LSTM):
            input_names.extend(["hidden_state", "cell_state"])
            output_names.extend(["new_hidden_state", "new_cell_state"])
            example_input_list.extend(
                [torch.zeros(1, num_layers, hidden_size), torch.zeros(1, num_layers, hidden_size)]
            )
        elif isinstance(wrapper.memory.rnn, nn.GRU):
            input_names.extend(["hidden_state"])
            output_names.extend(["new_hidden_state"])
            example_input_list.extend([torch.zeros(1, num_layers, hidden_size)])
    torch.onnx.export(
        wrapper,
        example_input_list,  # Pass x1 and x2 as separate inputs
        path,
        verbose=True,
        input_names=input_names,  # Specify the input names
        output_names=output_names,  # Name the output
        opset_version=13,  # Specify the opset version, if needed
    )


def export_vae_as_onnx(inference_model, path, exported_policy_name, example_obs_dict):
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, exported_policy_name)
    prior = copy.deepcopy(inference_model.prior).to("cpu")
    motion_decoder = copy.deepcopy(inference_model.motion_decoder).to("cpu")
    prior_mu_layer = copy.deepcopy(inference_model.prior_mu_layer).to("cpu")
    prior_sigma_layer = copy.deepcopy(inference_model.prior_sigma_layer).to("cpu")
    running_mean_std_proprioception = copy.deepcopy(
        inference_model.running_mean_std_proprioception
    ).to("cpu")
    running_mean_std = copy.deepcopy(inference_model.running_mean_std).to("cpu")

    class PPOWrapper(nn.Module):
        def __init__(
            self,
            prior,
            prior_mu_layer,
            prior_sigma_layer,
            motion_decoder,
            running_mean_std_proprioception,
            running_mean_std,
        ):
            """
            model: The original PyTorch model.
            input_keys: List of input names as keys for the input dictionary.
            """
            super(PPOWrapper, self).__init__()
            self.prior = prior
            self.prior_mu_layer = prior_mu_layer
            self.prior_sigma_layer = prior_sigma_layer
            self.motion_decoder = motion_decoder
            self.running_mean_std_proprioception = running_mean_std_proprioception
            self.running_mean_std = running_mean_std

        def forward(self, proprioception_obs):
            """
            Dynamically creates a dictionary from the input keys and args.
            """
            proprioception_obs = self.running_mean_std_proprioception(proprioception_obs)
            z = self.prior(proprioception_obs)
            z_mu = self.prior_mu_layer(z)
            z_sigma = self.prior_sigma_layer(z)
            std = torch.exp(0.5 * z_sigma)
            eps = torch.randn_like(std)
            z = z_mu + eps * z_sigma
            decoder_input = torch.cat([proprioception_obs, z], dim=-1)
            return self.motion_decoder(decoder_input)

    wrapper = PPOWrapper(
        prior,
        prior_mu_layer,
        prior_sigma_layer,
        motion_decoder,
        running_mean_std_proprioception,
        running_mean_std,
    )
    example_input_list = example_obs_dict["proprioception_obs"]
    torch.onnx.export(
        wrapper,
        example_input_list,  # Pass x1 and x2 as separate inputs
        path,
        verbose=True,
        input_names=["proprioception_obs"],  # Specify the input names
        output_names=["action"],  # Name the output
        opset_version=13,  # Specify the opset version, if needed
    )


def export_policy_and_estimator_as_onnx(
    inference_model, path, exported_policy_name, example_obs_dict
):
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, exported_policy_name)

    actor = copy.deepcopy(inference_model["actor"]).to("cpu")
    left_hand_force_estimator = copy.deepcopy(inference_model["left_hand_force_estimator"]).to(
        "cpu"
    )
    right_hand_force_estimator = copy.deepcopy(inference_model["right_hand_force_estimator"]).to(
        "cpu"
    )

    class PPOForceEstimatorWrapper(nn.Module):
        def __init__(self, actor, left_hand_force_estimator, right_hand_force_estimator):
            """
            model: The original PyTorch model.
            input_keys: List of input names as keys for the input dictionary.
            """
            super(PPOForceEstimatorWrapper, self).__init__()
            self.actor = actor
            self.left_hand_force_estimator = left_hand_force_estimator
            self.right_hand_force_estimator = right_hand_force_estimator

        def forward(self, inputs):
            """
            Dynamically creates a dictionary from the input keys and args.
            """
            actor_obs, history_for_estimator = inputs
            left_hand_force_estimator_output = self.left_hand_force_estimator(history_for_estimator)
            right_hand_force_estimator_output = self.right_hand_force_estimator(
                history_for_estimator
            )
            input_for_actor = torch.cat(
                [actor_obs, left_hand_force_estimator_output, right_hand_force_estimator_output],
                dim=-1,
            )
            return (
                self.actor.act_inference(input_for_actor),
                left_hand_force_estimator_output,
                right_hand_force_estimator_output,
            )

    wrapper = PPOForceEstimatorWrapper(actor, left_hand_force_estimator, right_hand_force_estimator)
    example_input_list = [
        example_obs_dict["actor_obs"],
        example_obs_dict["long_history_for_estimator"],
    ]
    torch.onnx.export(
        wrapper,
        example_input_list,  # Pass x1 and x2 as separate inputs
        path,
        verbose=True,
        input_names=["actor_obs", "long_history_for_estimator"],  # Specify the input names
        output_names=[
            "action",
            "left_hand_force_estimator_output",
            "right_hand_force_estimator_output",
        ],  # Name the output
        opset_version=13,  # Specify the opset version, if needed
    )
