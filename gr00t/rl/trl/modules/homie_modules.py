# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal


class HIMEstimator(nn.Module):
    def __init__(
        self,
        temporal_steps,
        num_one_step_obs,
        num_height_points,
        enc_hidden_dims=[256, 256],
        tar_hidden_dims=[256, 256],
        latent_dim=32,
        activation="elu",
        learning_rate=1e-3,
        max_grad_norm=10.0,
        num_prototype=64,
        temperature=3.0,
        **kwargs,
    ):
        if kwargs:
            print(
                "Estimator_CL.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super(HIMEstimator, self).__init__()
        activation = get_activation(activation)

        self.temporal_steps = temporal_steps
        self.num_one_step_obs = num_one_step_obs
        self.num_height_points = num_height_points
        self.num_latent = latent_dim
        self.max_grad_norm = max_grad_norm
        self.temperature = temperature

        # Encoder
        enc_input_dim = self.temporal_steps * self.num_one_step_obs + self.num_height_points
        enc_layers = []
        for l in range(len(enc_hidden_dims)):
            enc_layers += [nn.Linear(enc_input_dim, enc_hidden_dims[l]), activation]
            enc_input_dim = enc_hidden_dims[l]
        enc_layers += [nn.Linear(enc_input_dim, 3 + latent_dim)]
        self.encoder = nn.Sequential(*enc_layers)

        # Target
        tar_input_dim = self.num_one_step_obs
        tar_layers = []
        for l in range(len(tar_hidden_dims)):
            tar_layers += [nn.Linear(tar_input_dim, tar_hidden_dims[l]), activation]
            tar_input_dim = tar_hidden_dims[l]
        tar_layers += [nn.Linear(tar_input_dim, latent_dim)]
        self.target = nn.Sequential(*tar_layers)

        # Prototype
        self.proto = nn.Embedding(num_prototype, latent_dim)

        # Optimizer
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    def get_latent(self, obs_history):
        vel, z = self.encode(obs_history)
        return vel.detach(), z.detach()

    def forward(self, obs_history):
        parts = self.encoder(obs_history.detach())
        vel, z = parts[..., :3], parts[..., 3:]
        z = F.normalize(z, dim=-1, p=2)
        return vel.detach(), z.detach()

    def compute_response(self, obs_history):
        parts = self.encoder(obs_history.detach())
        vel, z = parts[..., :3], parts[..., 3:]
        z = F.normalize(z, dim=-1, p=2)
        return vel.detach(), z.detach()

    def compute_feedback(self, obs):
        z = self.target(obs)
        z = F.normalize(z, dim=-1, p=2)
        return z.detach()

    def encode(self, obs_history):
        parts = self.encoder(obs_history.detach())
        vel, z = parts[..., :3], parts[..., 3:]
        z = F.normalize(z, dim=-1, p=2)
        return vel, z

    def update(self, obs_history, next_critic_obs, lr=None):
        if lr is not None:
            self.learning_rate = lr
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.learning_rate

        vel = next_critic_obs[:, self.num_one_step_obs : self.num_one_step_obs + 3].detach()
        next_obs = next_critic_obs.detach()[:, 3 : self.num_one_step_obs + 3]

        z_s = self.encoder(obs_history)
        z_t = self.target(next_obs)
        pred_vel, z_s = z_s[..., :3], z_s[..., 3:]

        z_s = F.normalize(z_s, dim=-1, p=2)
        z_t = F.normalize(z_t, dim=-1, p=2)

        with torch.no_grad():
            w = self.proto.weight.data.clone()
            w = F.normalize(w, dim=-1, p=2)
            self.proto.weight.copy_(w)

        score_s = z_s @ self.proto.weight.T
        score_t = z_t @ self.proto.weight.T

        with torch.no_grad():
            q_s = sinkhorn(score_s)
            q_t = sinkhorn(score_t)

        log_p_s = F.log_softmax(score_s / self.temperature, dim=-1)
        log_p_t = F.log_softmax(score_t / self.temperature, dim=-1)

        swap_loss = -0.5 * (q_s * log_p_t + q_t * log_p_s).mean()
        estimation_loss = F.mse_loss(pred_vel, vel)
        losses = estimation_loss + swap_loss

        self.optimizer.zero_grad()
        losses.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return estimation_loss.item(), swap_loss.item()


@torch.no_grad()
def sinkhorn(out, eps=0.05, iters=3):
    Q = torch.exp(out / eps).T
    K, B = Q.shape[0], Q.shape[1]
    Q /= Q.sum()

    for it in range(iters):
        # normalize each row: total weight per prototype must be 1/K
        Q /= torch.sum(Q, dim=1, keepdim=True)
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B
    return (Q * B).T


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None


class HIMActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_one_step_obs,
        num_one_step_critic_obs,
        actor_history_length,
        critic_history_length,
        num_actions=19,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        init_noise_std=1.0,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super(HIMActorCritic, self).__init__()

        activation = get_activation(activation)
        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs
        self.num_one_step_obs = num_one_step_obs
        self.num_one_step_critic_obs = num_one_step_critic_obs
        self.actor_history_length = actor_history_length
        self.critic_history_length = critic_history_length
        self.actor_proprioceptive_obs_length = self.actor_history_length * self.num_one_step_obs
        self.critic_proprioceptive_obs_length = (
            self.critic_history_length * self.num_one_step_critic_obs
        )
        self.num_height_points = self.num_actor_obs - self.actor_proprioceptive_obs_length
        self.num_critic_height_points = self.num_critic_obs - self.critic_proprioceptive_obs_length
        self.actor_use_height = True if self.num_height_points > 0 else False
        self.num_actions = num_actions

        self.dynamic_latent_dim = 32
        self.terrain_latent_dim = 32
        if self.actor_use_height:
            mlp_input_dim_a = (
                num_one_step_obs + 3 + self.dynamic_latent_dim + self.terrain_latent_dim
            )
        else:
            mlp_input_dim_a = num_one_step_obs + 3 + self.dynamic_latent_dim
        mlp_input_dim_c = num_critic_obs

        # Estimator
        self.estimator = HIMEstimator(
            temporal_steps=self.actor_history_length,
            num_one_step_obs=self.num_one_step_obs,
            num_height_points=0,
            latent_dim=self.dynamic_latent_dim,
        )

        # Terrain Encoder
        if self.actor_use_height:
            self.terrain_encoder = nn.Sequential(
                nn.Linear(self.num_one_step_obs + self.num_height_points, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, self.terrain_latent_dim),
            )

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
                # actor_layers.append(nn.Tanh())
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")
        print(f"Estimator: {self.estimator.encoder}")
        if self.actor_use_height:
            print(f"Terrain Encoder: {self.terrain_encoder}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, obs_history):
        with torch.no_grad():
            vel, dynamic_latent = self.estimator(
                obs_history[:, 0 : self.actor_proprioceptive_obs_length]
            )
        if self.actor_use_height:
            terrain_latent = self.terrain_encoder(
                obs_history[:, -(self.num_height_points + self.num_one_step_obs) :]
            )
            actor_input = torch.cat(
                (
                    obs_history[
                        :,
                        -(self.num_height_points + self.num_one_step_obs) : -self.num_height_points,
                    ],
                    vel,
                    dynamic_latent,
                    terrain_latent,
                ),
                dim=-1,
            )
        else:
            actor_input = torch.cat(
                (obs_history[:, -self.num_one_step_obs :], vel, dynamic_latent), dim=-1
            )
        action_mean = self.actor(actor_input)
        self.distribution = Normal(action_mean, action_mean * 0.0 + self.std)

    def act(self, obs_history=None, **kwargs):
        self.update_distribution(obs_history)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, obs_history, observations=None):
        with torch.no_grad():
            vel, dynamic_latent = self.estimator(
                obs_history[:, 0 : self.actor_proprioceptive_obs_length]
            )
        if self.actor_use_height:
            terrain_latent = self.terrain_encoder(
                obs_history[:, -(self.num_height_points + self.num_one_step_obs) :]
            )
            actor_input = torch.cat(
                (
                    obs_history[
                        :,
                        -(self.num_height_points + self.num_one_step_obs) : -self.num_height_points,
                    ],
                    vel,
                    dynamic_latent,
                    terrain_latent,
                ),
                dim=-1,
            )
        else:
            actor_input = torch.cat(
                (obs_history[:, -self.num_one_step_obs :], vel, dynamic_latent), dim=-1
            )
        action_mean = self.actor(actor_input)
        return action_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value

    def update_estimator(self, obs_history, next_critic_obs, lr=None):
        return self.estimator.update(
            obs_history[:, 0 : self.actor_proprioceptive_obs_length],
            next_critic_obs[:, 0 : self.critic_proprioceptive_obs_length],
            lr,
        )


class HomieActorModule(nn.Module):
    def __init__(self, full_model):
        super().__init__()
        self.actor = full_model.actor
        self.estimator = full_model.estimator
        self.terrain_encoder = full_model.terrain_encoder if full_model.actor_use_height else None

        self.actor_use_height = full_model.actor_use_height
        self.num_one_step_obs = full_model.num_one_step_obs
        self.num_height_points = full_model.num_height_points
        self.std = full_model.std
        self.num_actions = 15
        self.distribution = None

    def forward(self, obs_history, export=False):
        vel, dynamic_latent = self.estimator(obs_history[..., 0 : self.num_one_step_obs * 6])
        if self.actor_use_height and self.terrain_encoder is not None:
            terrain_latent = self.terrain_encoder(
                obs_history[..., -(self.num_one_step_obs + self.num_height_points) :]
            )
            actor_input = torch.cat(
                [
                    obs_history[
                        ...,
                        -(self.num_one_step_obs + self.num_height_points) : -self.num_height_points,
                    ],
                    vel,
                    dynamic_latent,
                    terrain_latent,
                ],
                dim=-1,
            )
        else:
            actor_input = torch.cat(
                [obs_history[..., -self.num_one_step_obs :], vel, dynamic_latent], dim=-1
            )
        action_mean = self.actor(actor_input)

        self.distribution = Normal(action_mean, self.std.expand_as(action_mean))
        if export:
            return {"action": action_mean}
        else:
            return {
                "actions": self.distribution.mean,  # NOTE(qben): use.sample(),
                "action_mean": self.distribution.mean,
                "action_sigma": self.distribution.stddev,
                "entropy": self.distribution.entropy().sum(dim=-1),
            }

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)


init_actor_critic_dict = {
    "num_actor_obs": 516,
    "num_critic_obs": 89,
    "num_one_step_obs": 86,
    "num_one_step_critic_obs": 89,
    "actor_history_length": 6,
    "critic_history_length": 1,
    "num_actions": 15,
    "actor_hidden_dims": [512, 256, 256],
    "critic_hidden_dims": [512, 256, 256],
    "activation": "elu",
    "init_noise_std": 0.0,
}
