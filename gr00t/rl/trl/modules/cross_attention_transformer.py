# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import math

import torch
import torch.nn as nn


class CrossAttentionTransformer(nn.Module):
    """
    Flexible cross-attention transformer using PyTorch's transformer encoder.
    Can be configured for different Q/K/V setups including:
    - State history attention (actor_obs as Q, state_history as K/V)
    - Temporal vision attention (current frame as Q, history frames as K/V)
    """

    def __init__(
        self,
        env_config,
        algo_config,
        obs_dim_dict=None,
        input_obs_names=["actor_obs"],
        history_obs_names=["state_history_obs"],
        output_dim=None,
        hidden_dim=512,
        num_layers=6,
        num_heads=8,
        dropout=0.1,
        state_history_length=None,
        mode="state_history",  # "state_history" or "temporal_vision"
        query_dim=None,
        key_value_dim=None,
        **kwargs,
    ):
        super().__init__()

        if obs_dim_dict is None:
            obs_dim_dict = env_config.robot.algo_obs_dim_dict

        if output_dim is None:
            output_dim = hidden_dim
        elif isinstance(output_dim, str):
            output_dim = eval(output_dim)

        self.input_obs_names = input_obs_names
        self.history_obs_names = history_obs_names
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.state_history_length = state_history_length
        self.mode = mode

        # Setup projections based on mode
        if mode == "state_history":
            self._setup_state_history_projections(obs_dim_dict)
        elif mode == "temporal_vision":
            self._setup_temporal_vision_projections(query_dim, key_value_dim)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Cross-attention layers using PyTorch's MultiheadAttention
        self.cross_attn_layers = nn.ModuleList(
            [
                CrossAttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
                for _ in range(num_layers)
            ]
        )

        # Feature projection layer
        if output_dim != hidden_dim:
            self.feature_proj = nn.Linear(hidden_dim, output_dim)
        else:
            self.feature_proj = None

    def _setup_state_history_projections(self, obs_dim_dict):
        """Setup projections for state history attention mode."""
        # Projection for actor_obs (Query)
        self.obs_proj_modules = nn.ModuleDict()
        all_obs_dim = 0
        for obs_name in self.input_obs_names:
            if obs_name in obs_dim_dict:
                self.obs_proj_modules[obs_name] = nn.Linear(obs_dim_dict[obs_name], self.hidden_dim)
                all_obs_dim += self.hidden_dim

        if all_obs_dim > 0:
            self.obs_proj_linear = nn.Linear(all_obs_dim, self.hidden_dim)
        else:
            self.obs_proj_linear = None

        # Projection for state_history (Key, Value)
        self.prop_hist_proj_modules = nn.ModuleDict()
        for prop_hist_name in self.history_obs_names:
            if prop_hist_name in obs_dim_dict:
                prop_hist_dim = obs_dim_dict[prop_hist_name]
                self.prop_hist_proj_modules[prop_hist_name] = nn.Linear(
                    prop_hist_dim // self.state_history_length, self.hidden_dim
                )

    def _setup_temporal_vision_projections(self, query_dim, key_value_dim):
        """Setup projections for temporal vision attention mode."""
        if query_dim is None or key_value_dim is None:
            raise ValueError(
                "query_dim and key_value_dim must be specified for temporal_vision mode"
            )

        # Projection for query (current frame)
        self.query_proj = nn.Linear(query_dim, self.hidden_dim)

        # Projection for key/value (history frames)
        self.key_value_proj = nn.Linear(key_value_dim, self.hidden_dim)

    def forward(self, input_data, episode_attnmask=None, **kwargs):
        """
        Forward pass using cross-attention.

        For state_history mode:
            input_data: Dictionary containing:
                - actor_obs: [batch_size, seq_len, obs_dim]
                - state_history_obs: [batch_size, seq_len, num_hist, obs_dim]

        For temporal_vision mode:
            input_data: vision_features tensor [batch_size, num_seq, history_length, feature_dim]

        Returns:
            Feature tensor of shape [batch_size, seq_len, output_dim]
        """
        if self.mode == "state_history":
            return self._forward_state_history(input_data, episode_attnmask, **kwargs)
        elif self.mode == "temporal_vision":
            return self._forward_temporal_vision(input_data, episode_attnmask, **kwargs)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _forward_state_history(self, actor_obs_dict, episode_attnmask=None, **kwargs):
        """Forward pass for state history attention mode."""
        assert len(actor_obs_dict["actor_obs"].shape) == 3
        assert len(actor_obs_dict["state_history_obs"].shape) == 4

        if isinstance(actor_obs_dict, torch.Tensor):
            actor_obs_dict = {self.input_obs_names[0]: actor_obs_dict}

        # Process actor_obs as Query
        query = None
        if self.obs_proj_linear is not None:
            x_obs = []
            for obs_name in self.input_obs_names:
                if obs_name in actor_obs_dict:
                    obs = actor_obs_dict[obs_name]  # [batch_size, seq_len, obs_dim]
                    obs = self.obs_proj_modules[obs_name](obs)
                    x_obs.append(obs)

            if x_obs:
                x_obs = torch.cat(x_obs, dim=-1)
                query = self.obs_proj_linear(x_obs)  # [batch_size, seq_len, hidden_dim]

        # Process state_history as Key and Value
        key_value = None
        for prop_hist_name in self.history_obs_names:
            if prop_hist_name in actor_obs_dict:
                prop_hist = actor_obs_dict[
                    prop_hist_name
                ]  # [batch_size, seq_len, num_hist, obs_dim]
                if len(prop_hist.shape) == 4:
                    batch_size, seq_len, num_hist, obs_dim = prop_hist.shape

                    # Reshape to [batch_size * seq_len, num_hist, obs_dim]
                    prop_hist_reshaped = prop_hist.reshape(batch_size * seq_len, num_hist, obs_dim)

                    # Project each history step
                    key_value = self.prop_hist_proj_modules[prop_hist_name](prop_hist_reshaped)
                    # key_value: [batch_size * seq_len, num_hist, hidden_dim]
                    break

        if query is None or key_value is None:
            raise ValueError("Both actor_obs and state_history must be present")

        batch_size, seq_len, hidden_dim = query.shape

        # Reshape query to match key_value batch dimension
        # query: [batch_size, seq_len, hidden_dim] -> [batch_size * seq_len, 1, hidden_dim]
        query_reshaped = query.reshape(batch_size * seq_len, 1, hidden_dim)

        # Apply cross-attention layers
        x = query_reshaped
        for layer in self.cross_attn_layers:
            x = layer(x, key_value, key_value)  # Use same tensor for K and V

        # Reshape back to [batch_size, seq_len, hidden_dim]
        x = x.reshape(batch_size, seq_len, hidden_dim)

        # Feature projection
        if self.feature_proj is not None:
            x = self.feature_proj(x)

        return x

    def _forward_temporal_vision(self, vision_features, episode_attnmask=None, **kwargs):
        """
        Forward pass for temporal vision attention mode.

        Args:
            vision_features: [batch_size, num_seq, history_length, feature_dim]

        Returns:
            Aggregated features: [batch_size, num_seq, output_dim]
        """
        batch_size, num_seq, history_length, feature_dim = vision_features.shape

        if history_length <= 1:
            # No history to attend to, just return the single frame
            features = vision_features.squeeze(2)
            if self.feature_proj is not None:
                features = self.feature_proj(features)
            return features

        # Use the first frame (current) as query
        # query: [batch_size, num_seq, 1, feature_dim]
        query = vision_features[:, :, 0, :]

        # Use all frames as key and value
        # key_value: [batch_size, num_seq, history_length, feature_dim]
        key_value = vision_features

        # Reshape for processing
        # query: [batch_size * num_seq, 1, feature_dim]
        query_reshaped = query.reshape(batch_size * num_seq, 1, feature_dim)
        # key_value: [batch_size * num_seq, history_length, feature_dim]
        key_value_reshaped = key_value.reshape(batch_size * num_seq, history_length, feature_dim)

        # Project to hidden dimension
        query_proj = self.query_proj(query_reshaped)  # [batch_size * num_seq, 1, hidden_dim]
        key_value_proj = self.key_value_proj(
            key_value_reshaped
        )  # [batch_size * num_seq, history_length, hidden_dim]

        # Apply cross-attention layers
        x = query_proj
        for layer in self.cross_attn_layers:
            x = layer(x, key_value_proj, key_value_proj)  # Use same tensor for K and V

        # Reshape back to [batch_size, num_seq, hidden_dim]
        x = x.reshape(batch_size, num_seq, self.hidden_dim)

        # Feature projection
        if self.feature_proj is not None:
            x = self.feature_proj(x)

        return x


class CrossAttentionLayer(nn.Module):
    """
    Single cross-attention layer using PyTorch's MultiheadAttention.
    """

    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Cross-attention layer
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, query, key, value):
        """
        Forward pass of cross-attention layer.

        Args:
            query: [batch_size * seq_len, 1, hidden_dim]
            key: [batch_size * seq_len, num_hist, hidden_dim]
            value: [batch_size * seq_len, num_hist, hidden_dim]
        """
        # Cross-attention with residual connection and layer norm
        attn_output, _ = self.cross_attention(query, key, value)
        query = self.norm1(query + attn_output)

        # Feed-forward with residual connection and layer norm
        ffn_output = self.ffn(query)
        query = self.norm2(query + ffn_output)

        return query


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for transformer inputs.
    """

    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(0), :]
