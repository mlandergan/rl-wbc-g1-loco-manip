# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import torch
import torch.nn as nn

"""
updates statistic from a full data

Memory optimization note: If you encounter CUDA out of memory errors, consider setting:
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
This can help reduce memory fragmentation.
"""


class RunningMeanStd(nn.Module):

    def __init__(self, insize, epsilon=1e-05, per_channel=False, norm_only=False):
        super(RunningMeanStd, self).__init__()
        print("RunningMeanStd: ", insize)
        self.insize = insize
        self.mean_size = insize[0]
        self.epsilon = epsilon

        self.norm_only = norm_only
        self.per_channel = per_channel
        if per_channel:
            if len(self.insize) == 3:
                self.axis = [0, 2, 3]
            if len(self.insize) == 2:
                self.axis = [0, 2]
            if len(self.insize) == 1:
                self.axis = [0]
            in_size = self.insize[0]
        else:
            self.axis = [0]
            in_size = insize

        self.register_buffer("running_mean", torch.zeros(in_size, dtype=torch.float32))
        self.register_buffer("running_var", torch.ones(in_size, dtype=torch.float32))
        self.register_buffer("count", torch.ones((), dtype=torch.float32))

        self.frozen = False
        self.frozen_partial = False

    def freeze(self):
        self.frozen = True

    def unfreeze(self):
        self.frozen = False

    def freeze_partial(self, diff):
        self.frozen_partial = True
        self.diff = diff

    def _update_mean_var_count_from_moments(
        self, mean, var, count, batch_mean, batch_var, batch_count
    ):
        delta = batch_mean - mean
        tot_count = count + batch_count

        new_mean = mean + delta * batch_count / tot_count
        m_a = var * count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count
        return new_mean, new_var, new_count

    def forward(self, input, unnorm=False):
        # change shape
        input_shape = input.shape
        if len(input.shape) == 3:
            input = input.reshape(-1, input_shape[-1])

        if self.per_channel:
            if len(self.insize) == 3:
                # Use broadcasting instead of expand_as to avoid memory issues
                current_mean = self.running_mean.view([1, self.insize[0], 1, 1])
                current_var = self.running_var.view([1, self.insize[0], 1, 1])
            elif len(self.insize) == 2:
                current_mean = self.running_mean.view([1, self.insize[0], 1])
                current_var = self.running_var.view([1, self.insize[0], 1])
            elif len(self.insize) == 1:
                current_mean = self.running_mean.view([1, self.insize[0]])
                current_var = self.running_var.view([1, self.insize[0]])

        else:
            current_mean = self.running_mean
            current_var = self.running_var
        # get output

        if unnorm:
            y = torch.clamp(input, min=-5.0, max=5.0)
            y = torch.sqrt(current_var + self.epsilon) * y + current_mean
        else:
            if self.norm_only:
                y = input / torch.sqrt(current_var + self.epsilon)
            else:
                # Use in-place operations where possible to reduce memory usage
                y = input - current_mean
                y = y / torch.sqrt(current_var + self.epsilon)
                y = torch.clamp(y, min=-5.0, max=5.0)

        # update After normalization, so that the values used for training and testing are the same.
        if self.training and not self.frozen:
            mean = input.mean(self.axis)  # along channel axis
            var = input.var(self.axis)
            new_mean, new_var, new_count = self._update_mean_var_count_from_moments(
                self.running_mean, self.running_var, self.count, mean, var, input.size()[0]
            )
            if self.frozen_partial:
                # Only update the last bit (futures)
                self.running_mean[-self.diff :], self.running_var[-self.diff :], self.count = (
                    new_mean[-self.diff :],
                    new_var[-self.diff :],
                    new_count,
                )
            else:
                self.running_mean, self.running_var, self.count = new_mean, new_var, new_count

        if len(input_shape) == 3:
            y = y.view(input_shape)

        return y


class RunningMeanStdObs(nn.Module):

    def __init__(self, insize, epsilon=1e-05, per_channel=False, norm_only=False):
        assert insize is dict
        super(RunningMeanStdObs, self).__init__()
        self.running_mean_std = nn.ModuleDict(
            {k: RunningMeanStd(v, epsilon, per_channel, norm_only) for k, v in insize.items()}
        )

    def forward(self, input, unnorm=False):
        res = {k: self.running_mean_std[k](v, unnorm) for k, v in input.items()}
        return res
