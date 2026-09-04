# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import copy
import os
import subprocess
import sys
from pathlib import Path

import torch
from transformers import TrainerCallback


class ModelSaveCallback(TrainerCallback):
    """Callback to save model state_dict during training."""

    def __init__(self, save_dir, save_frequency=1000):
        """
        Args:
            save_dir (str): Directory to save model checkpoints
            save_frequency (int): Save model every N steps
        """
        self.save_dir = Path(save_dir)
        self.save_frequency = save_frequency
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def on_step_end(self, args, state, control, **kwargs):
        """Save model state_dict at the end of each step if frequency matches."""
        model = kwargs.get("model")
        optimizer = kwargs.get("optimizer")
        lr_scheduler = kwargs.get("lr_scheduler")
        env = kwargs.get("env")

        if (
            state.is_world_process_zero
            and state.global_step % self.save_frequency == 0
            and not env.is_evaluating
        ):
            env_state_dict = env.get_env_state_dict()
            ModelSaveCallback.save_checkpoint(
                model,
                optimizer,
                lr_scheduler,
                state,
                env_state_dict,
                args,
                f"{self.save_dir}/model_step_{state.global_step:06d}.pt",
            )

        if state.is_world_process_zero and state.global_step % 50 == 0 and not env.is_evaluating:
            env_state_dict = env.get_env_state_dict()
            ModelSaveCallback.save_checkpoint(
                model,
                optimizer,
                lr_scheduler,
                state,
                env_state_dict,
                args,
                f"{self.save_dir}/last.pt",
            )
            # self.export_policy_to_onnx(env, state, model)

    def get_example_obs(self, env):
        obs_dict = copy.deepcopy(env.obs_buf_dict)
        for obs_key in obs_dict.keys():
            print(obs_key, sorted(env.config.obs.obs_dict[obs_key]))
        # move to cpu
        for k in obs_dict:
            obs_dict[k] = obs_dict[k].cpu()[0:1]
        return obs_dict

    def export_policy_to_onnx(self, env, state, model):
        checkpoint_path = os.path.join(env.config.experiment_dir, "last.pt")
        cmd = [
            sys.executable,
            "gr00t/rl/eval_agent_trl.py",
            f"+checkpoint={checkpoint_path}",
            "+num_envs=1",
            "+headless=true",
            "+export_onnx_only=true",
        ]
        result = subprocess.run(cmd, capture_output=False, text=True, cwd=os.getcwd())
        onnx_last_path = os.path.join(env.config.experiment_dir, "exported", "last.onnx")
        onnx_step_path = os.path.join(
            env.config.experiment_dir, "exported", f"model_step_{state.global_step:06d}.onnx"
        )
        shutil.copy(onnx_last_path, onnx_step_path)

    @classmethod
    def save_checkpoint(
        cls, model, optimizer, lr_scheduler, state, env_state_dict, args, save_path
    ):
        if model is not None:
            # Save model, optimizer, scheduler and training state
            _state = copy.deepcopy(state)
            _state.__dict__.pop("log_history")
            checkpoint = {
                "policy_state_dict": model.policy.state_dict(),
                "value_state_dict": (
                    model.value_model.state_dict() if model.value_model is not None else None
                ),  # Value model is not always preset (e.g. for distillation)
                "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
                "lr_scheduler_state_dict": (
                    lr_scheduler.state_dict() if lr_scheduler is not None else None
                ),
                "state": _state,
                "args": args,
                "env_state_dict": env_state_dict,
            }
            if hasattr(model, "homie_model") and model.homie_model is not None:
                checkpoint["homie_state_dict"] = model.homie_model.state_dict()
            for attempt in range(5):
                try:
                    torch.save(checkpoint, save_path)
                    print(f"Saved model checkpoint to {save_path}")
                    break
                except Exception as e:
                    if attempt == 4:  # Last attempt
                        print(f"Failed to save checkpoint after 5 attempts. Error: {e}")
                        raise
                    print(f"Attempt {attempt + 1} failed to save checkpoint. Retrying...")
