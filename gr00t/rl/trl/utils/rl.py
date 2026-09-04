# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import torch
from tensordict import TensorDict


def split_and_pad_trajectories(
    tensor: torch.Tensor | TensorDict, dones: torch.Tensor
) -> tuple[torch.Tensor | TensorDict, torch.Tensor]:
    """Splits trajectories at done indices. Then concatenates them and pads with zeros up to the length of the longest
    trajectory. Returns masks corresponding to valid parts of the trajectories.

    Args:
        tensor: [num_steps, num_envs, ...]
        dones: [num_steps, num_envs]

    Returns:
        padded_trajectories: [max_traj_len, num_trajectories, ...]
        trajectory_masks: [max_traj_len, num_trajectories]

    Example:
        Input: [[a1, a2, a3, a4 | a5, a6],
                 [b1, b2 | b3, b4, b5 | b6]]

        Output:[[a1, a2, a3, a4], | [[True, True, True, True],
                [a5, a6, 0, 0],   |  [True, True, False, False],
                [b1, b2, 0, 0],   |  [True, True, False, False],
                [b3, b4, b5, 0],  |  [True, True, True, False],
                [b6, 0, 0, 0]]    |  [True, False, False, False]]

    Assumes that the input has the following order of dimensions: [time, number of envs, additional dimensions]
    """

    dones = dones.clone()
    dones[-1] = 1

    # Shape validation
    assert len(dones.shape) == 2, f"dones should be [num_steps, num_envs], got shape {dones.shape}"
    num_steps, num_envs = dones.shape
    if not isinstance(tensor, TensorDict):
        assert (
            tensor.shape[0] == num_steps
        ), f"tensor first dim {tensor.shape[0]} should match dones first dim {num_steps}"
        assert (
            tensor.shape[1] == num_envs
        ), f"tensor second dim {tensor.shape[1]} should match dones second dim {num_envs}"

    # Permute the buffers to have order (num_envs, num_transitions_per_env, ...), for correct reshaping
    flat_dones = dones.transpose(1, 0).reshape(-1, 1)
    # Get length of trajectory by counting the number of successive not done elements
    done_indices = torch.cat(
        (flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero()[:, 0])
    )
    trajectory_lengths = done_indices[1:] - done_indices[:-1]
    trajectory_lengths_list = trajectory_lengths.tolist()
    # Extract the individual trajectories
    if isinstance(tensor, TensorDict):
        padded_trajectories = {}
        for k, v in tensor.items():
            # split the tensor into trajectories
            trajectories = torch.split(v.transpose(1, 0).flatten(0, 1), trajectory_lengths_list)
            # add at least one full length trajectory
            trajectories = list(trajectories) + [
                torch.zeros(v.shape[0], *v.shape[2:], device=v.device)
            ]
            # pad the trajectories to the length of the longest trajectory
            padded_trajectories[k] = torch.nn.utils.rnn.pad_sequence(trajectories)
            # remove the added tensor
            padded_trajectories[k] = padded_trajectories[k][:, :-1]
        padded_trajectories = TensorDict(
            padded_trajectories, batch_size=[tensor.batch_size[0], len(trajectory_lengths_list)]
        )
    else:
        # split the tensor into trajectories
        trajectories = torch.split(tensor.transpose(1, 0).flatten(0, 1), trajectory_lengths_list)
        # add at least one full length trajectory
        trajectories = list(trajectories) + [
            torch.zeros(tensor.shape[0], *tensor.shape[2:], device=tensor.device)
        ]
        # pad the trajectories to the length of the longest trajectory
        padded_trajectories = torch.nn.utils.rnn.pad_sequence(trajectories)
        # remove the added tensor
        padded_trajectories = padded_trajectories[:, :-1]
    # create masks for the valid parts of the trajectories
    # Use padded_trajectories.shape[0] (max_traj_len) instead of tensor.shape[0] (num_steps)
    trajectory_masks = trajectory_lengths > torch.arange(
        0, padded_trajectories.shape[0], device=tensor.device
    ).unsqueeze(1)
    return padded_trajectories, trajectory_masks


def unsplit_trajectories(padded_trajectories, masks, original_dones):
    """
    Reverse operation of split_and_pad_trajectories.
    Reconstructs the original [num_envs, num_steps, ...] format from trajectory format.

    Key insight: After extracting valid elements with boolean indexing, they are already
    in the correct order (env-major, step-minor) from how split_and_pad_trajectories works.
    We just need to reshape back to [num_envs, num_steps, ...].

    Args:
        padded_trajectories: [num_trajectories, max_traj_len, ...]
        masks: [num_trajectories, max_traj_len] - True for valid timesteps
        original_dones: [num_envs, num_steps] - Original done flags

    Returns:
        reconstructed: [num_envs, num_steps, ...] - Back in original format
    """
    # Shape validation
    assert (
        padded_trajectories.shape[0] == masks.shape[0]
    ), f"padded_trajectories first dim {padded_trajectories.shape[0]} should match masks first dim {masks.shape[0]}"
    assert (
        padded_trajectories.shape[1] == masks.shape[1]
    ), f"padded_trajectories second dim {padded_trajectories.shape[1]} should match masks second dim {masks.shape[1]}"
    assert (
        len(original_dones.shape) == 2
    ), f"original_dones should be [num_envs, num_steps], got shape {original_dones.shape}"

    num_envs, num_steps = original_dones.shape
    feature_shape = padded_trajectories.shape[2:]

    # Extract valid elements only (removes padding)
    # Boolean indexing extracts elements in order, which matches the env-major order from split
    valid_elements = padded_trajectories[masks]  # Shape: [total_valid_elements, ...]

    # The valid elements are in env-major, step-minor order, so we can directly reshape
    # to [num_envs, num_steps, ...]
    reconstructed = valid_elements.reshape(num_envs, num_steps, *feature_shape)

    # Validate output shape
    assert (
        reconstructed.shape[0] == num_envs
    ), f"Output first dim {reconstructed.shape[0]} should match num_envs {num_envs}"
    assert (
        reconstructed.shape[1] == num_steps
    ), f"Output second dim {reconstructed.shape[1]} should match num_steps {num_steps}"
    assert (
        reconstructed.shape[2:] == feature_shape
    ), f"Output feature dims {reconstructed.shape[2:]} should match input feature dims {feature_shape}"

    return reconstructed


def compute_episode_attnmask(dones):
    """
    Compute an attention mask that prevents the model from attending to observations from different episodes.

    Args:
        dones (torch.Tensor): A tensor of shape (num_envs, num_steps) indicating when each environment episode ends.
                                A value of 1.0 indicates the end of an episode.

    Returns:
        torch.Tensor: An attention mask of shape (num_envs, num_steps, num_steps) where True values indicate
                        positions that should be masked (i.e., the model should not attend to these positions).
    """
    # Create cumulative sum of dones to identify different episodes
    episode_starts = torch.roll(dones, 1, dims=1)
    episode_starts[:, 0] = True  # First step is always start of an episode
    episode_ids = torch.cumsum(episode_starts, dim=1)  # (num_envs, num_steps)

    # Expand episode_ids for broadcasting
    episode_ids_i = episode_ids.unsqueeze(2)  # (num_envs, num_steps, 1)
    episode_ids_j = episode_ids.unsqueeze(1)  # (num_envs, 1, num_steps)

    # Create mask where True indicates positions from different episodes
    attnmask = episode_ids_i != episode_ids_j
    return attnmask
