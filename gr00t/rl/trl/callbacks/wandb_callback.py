# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0



import wandb
from transformers import TrainerCallback

from gr00t.rl.trl.utils.common import wandb_run_exists


class WandbCallback(TrainerCallback):
    """Callback to save model state_dict during training."""

    def __init__(
        self,
    ):
        super().__init__()

    def on_log(self, args, state, control, logs=None, **kwargs):

        if state.is_world_process_zero and wandb_run_exists():
            wandb.log(logs, step=state.global_step)
