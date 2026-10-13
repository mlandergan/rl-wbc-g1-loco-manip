# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
2026-10-01 TILT FEASIBILITY probe (setup copied from play_bin_push.py; the
policy is never queried -- only the frozen HOMIE stand/walk models drive the
legs, every other action is scripted).

  feas_mode=A1  external force at the bar midpoint (ramps 0->150 N in several
                directions + tilt-hold PI), robot standing far away.
  feas_mode=A2  canned overhand grasp in TILT, arm targets offset by J^+ dx
                to pull both hands back / down; fingers at the canned command.
  feas_mode=A3  as A2 plus HOMIE height / torso-pitch command offsets
                (whole-body pull).
  feas_mode=A4  as A3 plus a backward HOMIE vx command (walk back while
                holding).
  feas_mode=A5  bin body-mass sweep (15/10/7/5 kg) x 4 pull strategies.
  feas_mode=A6  force-controlled closed-loop IK (bar-attached palm target +
                admittance force loop; 6D wrist-follows-bin vs position-only),
                external-force-while-holding control, non-hand link contacts.

Usage (beast, repo root; copy a checkpoint + its config.yaml to /tmp/tiltfeas
so nothing is written into a live run dir):
  venv/bin/python -u move_garbage/scripts/probe_tilt_feasibility.py \
    ++checkpoint=/tmp/tiltfeas/model_step_001800.pt ++headless=True \
    ++feas_mode=A1 ++num_envs=12 ++env.config.grasp_init_ratio=0.0 ... \
    "hydra.searchpath=[file://$PWD/move_garbage/config]" hydra.run.dir=/tmp/tiltfeas/hydra_A1
"""

import logging
import os
import shutil
import sys
from pathlib import Path

import hydra
import yaml
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from loguru import logger
from omegaconf import OmegaConf

from gr00t.rl.utils.config_utils import register_rl_resolvers

register_rl_resolvers()

_CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "gr00t/rl/config")


def process_output_dim_in_config(config):
    """Process and adapt output dimensions for actor and teacher_actor backbones.

    When output_dim is set to -1 in the config, this function auto-calculates
    the correct dimension based on the homie command keys.
    """

    def calculate_homie_output_dim():
        output_dim = 0
        for key in config.obs["homie_command_keys"].keys():
            output_dim += len(config.obs["homie_command_default"][key])
        return output_dim

    def adapt_backbone_output_dim(backbone_config, config_name=""):
        try:
            if hasattr(backbone_config, "module_config_dict"):
                if backbone_config.module_config_dict.output_dim[0] == -1:
                    output_dim = calculate_homie_output_dim()
                    backbone_config.module_config_dict.output_dim = [output_dim]
                    return True
            elif hasattr(backbone_config, "mlp_module") and hasattr(
                backbone_config.mlp_module, "module_config_dict"
            ):
                if backbone_config.mlp_module.module_config_dict.output_dim[0] == -1:
                    output_dim = calculate_homie_output_dim()
                    backbone_config.mlp_module.module_config_dict.output_dim = [output_dim]
                    return True
        except (AttributeError, IndexError) as e:
            logger.warning(f"Could not adapt {config_name} backbone output_dim: {e}")
        return False

    if (
        config.algo.config.get("use_new_actor_critic", False)
        and hasattr(config.algo.config, "actor")
        and hasattr(config.algo.config.actor, "backbone")
    ):
        adapt_backbone_output_dim(config.algo.config.actor.backbone, "actor")

        if (
            getattr(config.algo.config, "use_dagger", False)
            and hasattr(config.algo.config, "teacher_actor")
            and hasattr(config.algo.config.teacher_actor, "backbone")
        ):
            adapt_backbone_output_dim(config.algo.config.teacher_actor.backbone, "teacher_actor")


@hydra.main(
    config_path=_CONFIG_DIR,
    config_name="base_eval",
)
def main(override_config: OmegaConf):
    # --- Logging setup ---
    hydra_log_path = os.path.join(HydraConfig.get().runtime.output_dir, "eval.log")
    logger.remove()
    logger.add(hydra_log_path, level="DEBUG")
    console_log_level = os.environ.get("LOGURU_LEVEL", "INFO").upper()
    logger.add(sys.stdout, level=console_log_level, colorize=True)

    from gr00t.rl.utils.logging import HydraLoggerBridge

    logging.basicConfig(level=logging.INFO)
    # TorchAudio probes optional FFmpeg ABIs at DEBUG level. Forwarding those
    # exception records through Loguru recursively can produce unbounded logs.
    logging.getLogger("torio").setLevel(logging.CRITICAL)
    logging.getLogger().addHandler(HydraLoggerBridge())
    os.chdir(hydra.utils.get_original_cwd())

    # --- Load and merge training config from checkpoint directory ---
    if override_config.checkpoint is not None:
        has_config = True
        checkpoint = Path(override_config.checkpoint)
        config_path = checkpoint.parent / "config.yaml"
        if not config_path.exists():
            config_path = checkpoint.parent.parent / "config.yaml"
            if not config_path.exists():
                has_config = False
                logger.error(f"Could not find config path: {config_path}")

        if has_config:
            logger.info(f"Loading training config file from {config_path}")
            with open(config_path) as file:
                train_config = OmegaConf.load(file)

            if train_config.eval_overrides is not None:
                train_config = OmegaConf.merge(train_config, train_config.eval_overrides)

            config = OmegaConf.merge(train_config, override_config)
        else:
            config = override_config

        config.experiment_dir = checkpoint.parent
    else:
        if override_config.eval_overrides is not None:
            config = override_config.copy()
            eval_overrides = OmegaConf.to_container(config.eval_overrides, resolve=True)
            for arg in sys.argv[1:]:
                if not arg.startswith("+"):
                    key = arg.split("=")[0]
                    if key in eval_overrides:
                        del eval_overrides[key]
            config.eval_overrides = OmegaConf.create(eval_overrides)
            config = OmegaConf.merge(config, eval_overrides)
        else:
            config = override_config

    # Resume wandb run if meta.yaml exists
    meta_path = Path(config.experiment_dir) / "meta.yaml"
    if meta_path.exists():
        meta = yaml.safe_load(open(meta_path, "r"))
        config.wandb.wandb_id = meta["wandb_run"]
        print(f"resume wandb from run: {config.wandb.wandb_id}")

    # --- Setup Isaac Sim ---
    simulator_type = config.simulator["_target_"].split(".")[-1]
    if simulator_type == "IsaacSim":
        try:
            with open("./rl/simulator/isaacsim/.isaacsim_version", "r", encoding="utf-8") as f:
                DEFAULT_ISAACSIM_VERSION = f.read().strip()
        except FileNotFoundError:
            DEFAULT_ISAACSIM_VERSION = "4.5"

        if DEFAULT_ISAACSIM_VERSION == "4.5":
            from isaaclab.app import AppLauncher
        elif DEFAULT_ISAACSIM_VERSION == "4.2":
            logger.warning("Using IsaacSim 4.2, replacing isaaclab with omni.isaac.lab")
            from omni.isaac.lab.app import AppLauncher

        import argparse

        import isaaclab

        parser = argparse.ArgumentParser(description="Evaluate an RL agent.")
        AppLauncher.add_app_launcher_args(parser)

        args_cli, hydra_args = parser.parse_known_args()
        sys.argv = [sys.argv[0]] + hydra_args
        args_cli.num_envs = config.num_envs
        args_cli.seed = config.seed
        args_cli.env_spacing = config.env.config.env_spacing
        args_cli.output_dir = config.output_dir
        args_cli.enable_cameras = (
            config.simulator.config.cameras.enable_cameras
            or config.simulator.config.get("render_results", False)
        )
        args_cli.headless = config.headless

        app_launcher = AppLauncher(args_cli)
        simulation_app = app_launcher.app

    # --- Imports that must come after Isaac Sim initialization ---
    from accelerate import Accelerator
    from transformers import HfArgumentParser
    from trl import ModelConfig, PPOConfig, ScriptArguments

    from gr00t.rl.agents.modules.ppo_modules import (
        PPOCritic,
        PPOStateActor,
        PPOStateActorFixSigma,
    )
    from gr00t.rl.trl.utils.common import custom_instantiate
    from gr00t.rl.utils.common import seeding
    from gr00t.rl.utils.helpers import pre_process_config

    # --- Config processing ---
    unresolved_conf = OmegaConf.to_container(config, resolve=False)
    os.chdir(hydra.utils.get_original_cwd())
    pre_process_config(config)

    # Set rendering output directory
    ckpt_num = config.checkpoint.split("/")[-1].split("_")[-1].split(".")[0]
    if config.env.config.get("save_rendering_dir", None) is None:
        if config.simulator.config.get("render_results", False):
            from datetime import datetime

            parent_dir = checkpoint.parent.name
            config.env.config.save_rendering_dir = str(
                Path("logs_rl") / "renderings" / datetime.now().strftime("%Y-%m-%d") / parent_dir
            )
        else:
            config.env.config.save_rendering_dir = str(
                checkpoint.parent / "renderings" / f"ckpt_{ckpt_num}"
            )
    config.env.config.ckpt_dir = str(checkpoint.parent)

    # --- Setup TRL training args (reused for eval) ---
    config.algo.trl.output_dir = str(Path(config.experiment_dir))
    parser = HfArgumentParser((ScriptArguments, PPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_dict(config.algo.trl)
    if "eval_output_dir" in config:
        training_args.eval_output_dir = config.eval_output_dir

    ref_model = None
    value_model = None

    # --- Accelerator and seeding ---
    accelerator = Accelerator()
    device = str(accelerator.device)
    if device == "cuda":
        device = "cuda:0"
    config.multi_gpu = accelerator.num_processes > 1
    if config.multi_gpu:
        config.global_rank = accelerator.process_index
        config.seed += accelerator.process_index
        config.algo.config.global_rank = accelerator.process_index
        config.algo.config.world_size = accelerator.num_processes
    seeding(config.seed)

    # --- Create environment ---
    env = instantiate(config=config.env, device=device)

    # --- Build policy and value models ---
    process_output_dim_in_config(config)

    if config.algo.config.get("use_new_actor_critic", False):
        module_dim_dict = getattr(config.algo.config, "module_dim", {})
        policy = instantiate(
            config.algo.config.actor,
            env_config=env.config,
            algo_config=config.algo.config,
            module_dim_dict=module_dim_dict,
            _recursive_=False,
        ).to(device)
        if getattr(config.algo.config, "use_dagger", False):
            ref_model = instantiate(
                config.algo.config.teacher_actor,
                env_config=env.config,
                algo_config=config.algo.config,
                module_dim_dict=module_dim_dict,
                _recursive_=False,
                input_key="teacher_obs",
            ).to(device)
        if not getattr(config.algo.config, "distill_only", False) and hasattr(
            config.algo.config, "critic"
        ):
            value_model = instantiate(
                config.algo.config.critic,
                env_config=env.config,
                algo_config=config.algo.config,
                module_dim_dict=module_dim_dict,
                _recursive_=False,
            ).to(device)
    else:
        if getattr(config.algo.config, "use_dagger", False):
            module_dim_dict = getattr(config.algo.config, "module_dim", {})
            policy = PPOStateActorFixSigma(
                obs_dim_dict=env.config.robot.algo_obs_dim_dict,
                module_config_dict=config.algo.config.module_dict.actor,
                num_actions=env.config.robot.actions_dim,
                module_dim_dict=module_dim_dict,
            ).to(device)
            ref_model = PPOStateActorFixSigma(
                obs_dim_dict=env.config.robot.algo_obs_dim_dict,
                module_config_dict=config.algo.config.module_dict.teacher_actor,
                num_actions=env.config.robot.actions_dim,
                module_dim_dict=module_dim_dict,
            ).to(device)
        else:
            policy = PPOStateActor(
                obs_dim_dict=env.config.robot.algo_obs_dim_dict,
                module_config_dict=config.algo.config.module_dict.actor,
                num_actions=env.config.robot.actions_dim,
                input_key="actor_obs",
                init_noise_std=config.algo.config.init_noise_std,
            ).to(device)
            value_model = PPOCritic(
                env.config.robot.algo_obs_dim_dict,
                config.algo.config.module_dict.critic,
            ).to(device)

    accelerator.wait_for_everyone()

    # --- Callbacks ---
    callbacks = []
    for callback in config.callbacks.values():
        callbacks.append(instantiate(callback))

    # --- Build trainer (loads checkpoint weights) ---
    trainer = custom_instantiate(
        config.trainer,
        args=training_args,
        config=config.algo.config,
        env=env,
        model=policy,
        ref_model=ref_model,
        use_ref_model=getattr(config.algo.config, "use_dagger", False),
        value_model=value_model,
        train_dataset=None,
        eval_dataset=None,
        callbacks=callbacks,
        checkpoint=config.checkpoint,
        local_seed=config.seed,
        accelerator=accelerator,
    )

    # =====================================================================
    # 2026-10-01 TILT FEASIBILITY (appended to play_bin_push.py's setup).
    #   feas_mode=A1: external force at the bar midpoint on the bin, robot
    #                 standing far away (no contact). Ramps / tilt-hold PI.
    #   feas_mode=A2: canned overhand grasp in TILT (grasp_init_ratio 1.0),
    #                 arm joint targets offset by J^+ dx to pull both hands
    #                 back / down; fingers stay at the canned (p1) command.
    # Moment about the wheel-ground contact line, + = tips the top back
    # toward the robot (same sign as _bin_tilt_back_deg):
    #   M = ((p - c) x F) . k,  c = mean wheel centre - r*z,  k = a x z,
    #   a = horizontal push axis (away from the robot).
    # =====================================================================
    import math

    import numpy as np
    import torch
    import torch.nn.functional as F

    mode = str(config.get("feas_mode", "A1"))
    out_path = str(config.get("feas_out", f"/tmp/tiltfeas/{mode}.npz"))
    env_u = env
    dev = env_u.device
    N = env_u.num_envs
    dt = float(env_u.dt)
    model = trainer.model
    homie_walk_model = model.homie_walk_model
    homie_stand_model = model.homie_stand_model
    thr = getattr(trainer, "homie_switch_threshold", 0.5)
    from gr00t.rl.trl.modules.homie_modules import init_actor_critic_dict as _iacd

    n1 = _iacd["num_one_step_obs"]
    trainer._eval_mode()
    n_pol = int(env_u.config.robot.actions_dim)
    print(f"[feas] mode={mode} N={N} dt={dt} n_pol={n_pol} "
          f"delta_shape={tuple(env_u._delta_actions.shape)}", flush=True)

    def homie(obs):
        h = obs["homie_obs"]
        w = homie_walk_model(h)
        s = homie_stand_model(h)
        cmd = h[..., -n1:-(n1 - 3)]
        m = torch.norm(cmd, dim=-1, keepdim=True) > thr
        while m.dim() < w["actions"].dim():
            m = m.unsqueeze(-1)
        return torch.where(m, w["action_mean"], s["action_mean"])

    state = {"obs": {k: v.to(dev) for k, v in env_u.reset_all().items()}}

    def step():
        with torch.no_grad():
            a = torch.cat([torch.zeros(N, n_pol, device=dev), homie(state["obs"])], dim=-1)
            obs, _rew, dones, _inf = env_u.step({"actions": a})
        state["obs"] = {k: v.to(dev) for k, v in obs.items()}
        return dones.bool().reshape(N)

    bin_art = env_u.simulator.scene.articulations["bin"]
    bnames = list(bin_art.body_names)
    bi = bnames.index("body")
    wi = [bnames.index("wheel_left"), bnames.index("wheel_right")]
    zhat = torch.tensor([0.0, 0.0, 1.0], device=dev).expand(N, 3).contiguous()

    def axes():
        a2 = F.normalize(env_u.push_axis_world[:, :2].float(), dim=-1)
        a = torch.cat([a2, torch.zeros(N, 1, device=dev)], -1)
        return a, torch.cross(a, zhat, dim=-1)

    def contact_pt():
        c = bin_art.data.body_pos_w[:, wi, :].mean(1).clone()
        c[:, 2] -= env_u.wheel_radius
        return c

    def bar_mid():
        return 0.5 * (env_u._left_bar_raw_pos() + env_u._right_bar_raw_pos())

    masses = bin_art.root_physx_view.get_masses().to(dev).float()
    # 2026-10-01: optional uniform bin body-mass override (any mode), e.g.
    # ++feas_body_mass=5.0 with feas_mode=A1 to check the mass lever physically.
    if config.get("feas_body_mass", None) is not None:
        _m = bin_art.root_physx_view.get_masses().clone()
        _i = bin_art.root_physx_view.get_inertias().clone()
        _r = float(config.feas_body_mass) / _m[:, bi]
        _m[:, bi] = float(config.feas_body_mass)
        _i[:, bi] = _i[:, bi] * _r[:, None]
        _ids = torch.arange(N, dtype=torch.int32)
        bin_art.root_physx_view.set_masses(_m, _ids)
        bin_art.root_physx_view.set_inertias(_i, _ids)
        masses = bin_art.root_physx_view.get_masses().to(dev).float()
        print(f"[feas] feas_body_mass override -> body {float(masses[0, bi]):.2f} kg", flush=True)

    def com_pos():
        cp = getattr(bin_art.data, "body_com_pos_w", None)
        return bin_art.data.body_pos_w if cp is None else cp

    def com_all():
        cp = com_pos()
        return (cp * masses[..., None]).sum(1) / masses.sum(1, keepdim=True)

    def mom(p, f, c, k):
        return (torch.cross(p - c, f, dim=-1) * k).sum(-1)

    def tilt():
        return env_u._bin_tilt_back_deg()

    def geometry_report(tag):
        a, k = axes()
        c, bm, cm = contact_pt(), bar_mid(), com_all()
        mtot = masses.sum(1)
        Mg = mom(cm, torch.stack([0 * mtot, 0 * mtot, -9.81 * mtot], -1), c, k)
        off = (com_pos()[:, bi] - bin_art.data.body_pos_w[:, bi]).norm(dim=-1)
        e = 0
        print(f"[feas][geom {tag}] bin bodies {bnames} masses(env0) "
              f"{[round(float(x), 3) for x in masses[e]]} total {float(mtot[e]):.2f} kg "
              f"(min/max over envs {float(mtot.min()):.2f}/{float(mtot.max()):.2f})", flush=True)
        print(f"[feas][geom {tag}] wheel_radius {float(env_u.wheel_radius[e]):.3f} | "
              f"COM - contact: along a {float(((cm - c) * a).sum(-1)[e]) * 100:+.1f} cm, "
              f"z {float((cm - c)[e, 2]) * 100:+.1f} cm | bar - contact: along a "
              f"{float(((bm - c) * a).sum(-1)[e]) * 100:+.1f} cm, z {float((bm - c)[e, 2]) * 100:+.1f} cm | "
              f"gravity moment about contact {float(Mg[e]):+.2f} Nm | body com-vs-origin "
              f"{float(off[e]) * 100:.1f} cm | tilt {float(tilt()[e]):+.2f} deg", flush=True)
        return Mg

    def summarize_print(rows, header):
        print("[feas] " + header, flush=True)
        for r in rows:
            print("[feas] " + r, flush=True)

    log = {}

    def rec(key, val):
        log.setdefault(key, []).append(val.detach().float().cpu().numpy() if torch.is_tensor(val) else val)

    # settle (both modes): zero policy actions, HOMIE stand.
    for _ in range(3):
        step()
    if mode == "A1":
        # ---------------- A1: external force on the bin ----------------
        # env layout: 0-4 ramps at phi=-30/0/30/60/90 (deg below horizontal,
        # toward the robot; 90 = straight down), 5-8 tilt-hold PI
        # (theta_t, phi) = (10,0) (15,0) (10,30) (15,30), 9 control (F=0),
        # 10-11 slow ramps phi=0/30.
        settle = int(round(0.6 / dt))
        for _ in range(settle):
            step()
        Mg0 = geometry_report("A1 t0")
        T_total = float(config.get("feas_T", 12.0))
        n_steps = int(round(T_total / dt))
        phis = torch.zeros(N, device=dev)
        kind = ["ctrl"] * N
        rate = torch.zeros(N, device=dev)
        theta_t = torch.zeros(N, device=dev)
        layout = [("ramp", -30, 25.0), ("ramp", 0, 25.0), ("ramp", 30, 25.0), ("ramp", 60, 25.0),
                  ("ramp", 90, 25.0), ("hold", 0, 10.0), ("hold", 0, 15.0), ("hold", 30, 10.0),
                  ("hold", 30, 15.0), ("ctrl", 0, 0.0), ("ramp", 0, 12.5), ("ramp", 30, 12.5)]
        for e in range(min(N, len(layout))):
            kd_, ph, v = layout[e]
            kind[e] = kd_
            phis[e] = math.radians(ph)
            if kd_ == "ramp":
                rate[e] = v
            elif kd_ == "hold":
                theta_t[e] = v
        a0, k0 = axes()
        u = torch.cos(phis)[:, None] * (-a0) + torch.sin(phis)[:, None] * (-zhat)
        c0, bm0 = contact_pt(), bar_mid()
        lever = (torch.cross(bm0 - c0, u, dim=-1) * k0).sum(-1)  # M per N of force
        F_star = -Mg0 / lever.clamp(min=1e-6)
        F_star = torch.where(lever > 1e-6, F_star, torch.full_like(F_star, float("nan")))
        th0 = tilt().clone()
        x0 = (bin_art.data.root_pos_w[:, :2] * a0[:, :2]).sum(-1).clone()
        Iacc = torch.zeros(N, device=dev)
        th_prev = th0.clone()
        mag = torch.zeros(N, device=dev)
        alive = torch.ones(N, dtype=torch.bool, device=dev)
        hold_mask = torch.tensor([k_ == "hold" for k_ in kind], device=dev)
        kp, ki, kd = 4.0, 8.0, 1.0
        is_ramp = torch.tensor([k_ == "ramp" for k_ in kind], device=dev)
        for i in range(n_steps):
            t = i * dt
            th = tilt()
            thd = (th - th_prev) / dt
            th_prev = th.clone()
            err = theta_t - th
            Iacc = (Iacc + err * dt).clamp(-40.0, 40.0)
            m_hold = (kp * err + ki * Iacc - kd * thd).clamp(0.0, 250.0)
            m_ramp = (rate * t).clamp(max=150.0)
            mag = torch.where(hold_mask, m_hold, torch.where(is_ramp, m_ramp, torch.zeros_like(mag)))
            a, k = axes()
            u = torch.cos(phis)[:, None] * (-a) + torch.sin(phis)[:, None] * (-zhat)
            Fw = mag[:, None] * u
            bm = bar_mid()
            comb = com_pos()[:, bi]
            tau = torch.cross(bm - comb, Fw, dim=-1)
            pwc = getattr(bin_art, "permanent_wrench_composer", None)
            if pwc is not None and hasattr(pwc, "_link_poses_updated"):
                pwc._link_poses_updated = False  # refresh link pose for global->local
            bin_art.set_external_force_and_torque(
                Fw[:, None, :].contiguous(), tau[:, None, :].contiguous(), body_ids=[bi], is_global=True
            )
            c = contact_pt()
            Mext = mom(bm, Fw, c, k)
            cm = com_all()
            mtot = masses.sum(1)
            Mg = mom(cm, torch.stack([0 * mtot, 0 * mtot, -9.81 * mtot], -1), c, k)
            dones = step()
            alive &= ~dones
            x = (bin_art.data.root_pos_w[:, :2] * a[:, :2]).sum(-1)
            hand_f = env_u.simulator.object_to_hand_contact_forces[:, 0].norm(dim=-1).sum(-1)
            rec("t", t)
            rec("mag", mag)
            rec("tilt", tilt())
            rec("Mext", Mext)
            rec("Mg", Mg)
            rec("dx_a", x - x0)
            rec("alive", alive)
            rec("hand_f", hand_f)
        np.savez(out_path, **{k_: np.stack(v) if isinstance(v[0], np.ndarray) else np.array(v)
                              for k_, v in log.items()},
                 kind=np.array(kind), phi_deg=np.rad2deg(phis.cpu().numpy()),
                 theta_t=theta_t.cpu().numpy(), F_star=F_star.cpu().numpy(), Mg0=Mg0.cpu().numpy())
        L = {k_: np.stack(v) if isinstance(v[0], np.ndarray) else np.array(v) for k_, v in log.items()}
        tl, mg, me, dx, al, tt = L["tilt"], L["mag"], L["Mext"], L["dx_a"], L["alive"], L["t"]
        hf = L["hand_f"]
        rows = []
        for e in range(N):
            ok = al[:, e].astype(bool)
            d0 = tl[:, e] - float(th0[e])
            if kind[e] == "ramp":
                on = np.where((d0 > 0.5) & ok)[0]
                on1 = np.where((d0 > 2.0) & ok)[0]
                s05 = (f"F {mg[on[0], e]:6.1f} N  M {me[on[0], e]:6.1f} Nm  t {tt[on[0]]:4.1f}s "
                       f"dx {dx[on[0], e] * 100:+5.1f}cm" if len(on) else "no onset")
                s2 = f"F {mg[on1[0], e]:6.1f} N  M {me[on1[0], e]:6.1f} Nm" if len(on1) else "-"
                r10 = np.where((tl[:, e] >= 10.0) & ok)[0]
                r15 = np.where((tl[:, e] >= 15.0) & ok)[0]
                rows.append(
                    f"env{e:2d} ramp phi {np.rad2deg(float(phis[e])):+4.0f} rate {float(rate[e]):4.1f}N/s | "
                    f"analytic F* {float(F_star[e]):6.1f} N (M* {-float(Mg0[e]):5.1f} Nm) | onset(+0.5deg): {s05} | "
                    f"+2deg: {s2} | F@10deg {mg[r10[0], e] if len(r10) else float('nan'):6.1f} "
                    f"F@15deg {mg[r15[0], e] if len(r15) else float('nan'):6.1f} | max tilt "
                    f"{tl[ok, e].max() if ok.any() else float('nan'):+5.1f} | dx_end {dx[ok, e][-1] * 100 if ok.any() else float('nan'):+5.1f}cm "
                    f"| alive {int(ok.sum())}/{len(ok)} | max hand contact {hf[:, e].max():.1f}N"
                )
            elif kind[e] == "hold":
                w = slice(int(len(tt) * 0.75), None)
                okw = ok[w]
                rows.append(
                    f"env{e:2d} hold theta_t {float(theta_t[e]):4.1f} phi {np.rad2deg(float(phis[e])):+4.0f} | "
                    f"last 25%: tilt {tl[w, e][okw].mean() if okw.any() else float('nan'):+5.1f}+-"
                    f"{tl[w, e][okw].std() if okw.any() else float('nan'):.1f} deg  F {mg[w, e][okw].mean() if okw.any() else float('nan'):6.1f} N  "
                    f"M {me[w, e][okw].mean() if okw.any() else float('nan'):6.1f} Nm  Mg {L['Mg'][w, e][okw].mean() if okw.any() else float('nan'):+6.1f} Nm | "
                    f"dx_end {dx[ok, e][-1] * 100 if ok.any() else float('nan'):+5.1f}cm | alive {int(ok.sum())}/{len(ok)}"
                )
            else:
                rows.append(f"env{e:2d} control | tilt range {tl[:, e].min():+.2f}..{tl[:, e].max():+.2f} | "
                            f"dx_end {dx[-1, e] * 100:+.1f}cm | max hand contact {hf[:, e].max():.1f}N")
        summarize_print(rows, "A1 SUMMARY (M = external moment about wheel contact line, + tips back)")
    else:
        # ---------------- A2: robot pulls from the canned grasp ----------------
        robot = env_u.simulator._robot
        if mode == "A5":
            # bin body mass sweep (wheels 1 kg each unchanged), inertia scaled.
            # 2026-10-01: env e -> mass index e % 4 (strategy is e // 4)
            body_mass = torch.tensor([15.0, 10.0, 7.0, 5.0] * 4)[:N].reshape(-1)
            mcpu = bin_art.root_physx_view.get_masses().clone()
            icpu = bin_art.root_physx_view.get_inertias().clone()
            ratio = body_mass / mcpu[:, bi]
            mcpu[:, bi] = body_mass
            icpu[:, bi] = icpu[:, bi] * ratio[:, None]
            ids_cpu = torch.arange(N, dtype=torch.int32)
            bin_art.root_physx_view.set_masses(mcpu, ids_cpu)
            bin_art.root_physx_view.set_inertias(icpu, ids_cpu)
            masses.copy_(bin_art.root_physx_view.get_masses().to(dev).float())
            print(f"[feas] A5 body masses {masses[:, bi].tolist()}", flush=True)
        settle = int(round(0.4 / dt))
        for _ in range(settle):
            step()
        geometry_report("A2 t0")
        print(f"[feas] stage after settle {env_u.stage_buf.tolist()} tilt "
              f"{[round(float(x), 2) for x in tilt()]}", flush=True)
        env_u._init_hand_link_indices()
        n_cmd = int(env_u._num_homie_commands)
        n_lower = int(env_u._num_lower_dof)
        act_idx = env_u._homie_active_command_index
        dai = env_u._delta_action_indices
        scale = float(env_u.config.robot.control.action_scale)
        clip = float(env_u.config.get("delta_action_clip", 100.0))
        default = env_u.default_dof_pos[0]

        def delta_col(j):
            kk = j - n_lower
            p = int((act_idx == n_cmd + kk).nonzero()[0].item())
            return int((dai == p).nonzero()[0].item())

        sides = ("left", "right")
        arm_env = {"left": env_u._left_arm_dof_idx.tolist(), "right": env_u._right_arm_dof_idx.tolist()}
        cols = {s: [delta_col(j) for j in arm_env[s]] for s in sides}
        dof_ids = list(env_u.simulator.dof_ids)
        J_full = robot.root_physx_view.get_jacobians()
        floating = J_full.shape[1] == robot.num_bodies
        off6 = 6 if J_full.shape[3] == robot.num_joints + 6 else 0
        print(f"[feas] jacobian {tuple(J_full.shape)} num_bodies {robot.num_bodies} num_joints "
              f"{robot.num_joints} floating-convention {floating} col offset {off6} | delta cols {cols}",
              flush=True)
        if mode == "A5":
            strat = [(0, 0.0, 0.0, 0.0, 0.0), (0, 0.0, -0.10, 0.0, -0.3), (30, 0.08, -0.10, 0.0, 0.0),
                     (0, 0.0, -0.15, 0.0, 0.0)]
            layout = [st_ for st_ in strat for _m in range(4)]  # env e: strategy e//4, mass e%4
        elif mode == "A4":
            # walk backward (HOMIE vx command < 0) while holding, +- arm pull /
            # lower height / pitch: (phi_deg, dx_m, dheight_m, dpitch_rad, vx)
            layout = [(0, 0.0, 0.0, 0.0, 0.0), (0, 0.0, 0.0, 0.0, -0.1), (0, 0.0, 0.0, 0.0, -0.2),
                      (0, 0.0, 0.0, 0.0, -0.3), (0, 0.0, 0.0, 0.0, -0.5),
                      (30, 0.08, 0.0, 0.0, -0.1), (30, 0.08, 0.0, 0.0, -0.2), (30, 0.08, 0.0, 0.0, -0.3),
                      (30, 0.08, 0.0, 0.0, -0.5), (0, 0.0, -0.10, 0.0, -0.1), (0, 0.0, -0.10, 0.0, -0.2),
                      (0, 0.0, -0.10, 0.0, -0.3), (0, 0.0, -0.10, 0.0, -0.5), (0, 0.0, 0.0, 0.3, -0.3),
                      (0, 0.0, 0.0, -0.3, -0.3), (0, 0.15, 0.0, 0.0, -0.3)]
        elif mode == "A3":
            # whole-body: HOMIE height / torso-pitch command offsets (+ arm pull)
            # (phi_deg, dx_m, dheight_m, dpitch_rad)
            layout = [(0, 0.0, 0.0, 0.0), (0, 0.0, -0.05, 0.0), (0, 0.0, -0.10, 0.0), (0, 0.0, -0.15, 0.0),
                      (0, 0.0, 0.0, 0.2), (0, 0.0, 0.0, 0.4), (0, 0.0, 0.0, -0.2), (0, 0.0, 0.0, -0.4),
                      (30, 0.08, -0.10, 0.0), (30, 0.08, 0.0, 0.3), (30, 0.08, 0.0, -0.3),
                      (30, 0.08, -0.10, 0.3), (30, 0.08, -0.10, -0.3), (0, 0.15, -0.15, 0.3),
                      (0, 0.15, -0.15, -0.3), (30, 0.08, 0.0, 0.0)]
        else:
            layout = [(0, 0.04), (0, 0.08), (0, 0.15), (30, 0.04), (30, 0.08), (30, 0.15),
                      (60, 0.04), (60, 0.08), (60, 0.15), (90, 0.04), (90, 0.08), (90, 0.15),
                      (0, 0.0), (-20, 0.08), (-20, 0.15), (30, 0.25)]
            layout = [(p_, m_, 0.0, 0.0) for p_, m_ in layout]
        layout = [tuple(e_) + (0.0,) * (5 - len(e_)) for e_ in layout]
        layout = (layout + [(0, 0.0, 0.0, 0.0, 0.0)] * N)[:N]
        phis = torch.tensor([math.radians(e_[0]) for e_ in layout], device=dev)
        mags = torch.tensor([e_[1] for e_ in layout], device=dev)
        dh = torch.tensor([e_[2] for e_ in layout], device=dev)
        dp = torch.tensor([e_[3] for e_ in layout], device=dev)
        vx = torch.tensor([e_[4] for e_ in layout], device=dev)
        cmd_scale = float(env_u.config.homie_command_scale)

        def act_col(proc_idx):
            p_ = int((act_idx == proc_idx).nonzero()[0].item())
            return int((dai == p_).nonzero()[0].item())

        h_col, p_col = act_col(3), act_col(5)
        try:
            vx_col = act_col(0)
        except Exception:
            vx_col = None
        print(f"[feas] vx col {vx_col}", flush=True)
        h0 = env_u._delta_actions[:, h_col].clone()
        p0 = env_u._delta_actions[:, p_col].clone()
        print(f"[feas] homie cols h {h_col} p {p_col} cmd_scale {cmd_scale} | delta h0 {float(h0[0]):.3f} "
              f"p0 {float(p0[0]):.3f} -> height cmd {float(h0[0]) * cmd_scale + float(env_u.config.obs.homie_default_height):.3f} "
              f"pitch cmd {float(p0[0]) * cmd_scale:.3f} | clip_homie_command {env_u.config.get('clip_homie_command', False)}",
              flush=True)
        root0 = robot.data.root_pos_w[:, :2].clone()
        bin0 = bin_art.data.root_pos_w[:, :2].clone()
        if mode == "A6":
            # ---------------- A6: force-controlled closed-loop IK ----------------
            # 2026-10-01 (parent agent). A2-A5 used ONE fixed J^+ dx offset computed
            # at t0 (position only): the arms reached their joint targets with only
            # ~25 N per hand and the bin slid along. A6 instead, every step:
            #   * palm target = the t0 grasp pose carried with the bin body pose
            #     (bar-attached frame -> the target follows the bar arc / slide)
            #     + an admittance offset d along u (phi below horizontal, toward
            #     the robot) that grows until the measured hand force on the bin
            #     along u reaches F_des/2 per hand (force control on top of PD);
            #   * ori=1: palm orientation target rotates with the bin (6D DLS IK,
            #     the wrist never fights the bar rotation); ori=0: position-only.
            #   * ext>0: robot only holds (F_des 0) and an external force ramps at
            #     the bar (as A1) -> onset vs A1's no-robot onset = hidden
            #     resistance from the grasp.
            #   * logs net contact on every non-hand/non-foot robot link (forearm /
            #     wrist / torso touching the bin).
            # layout rows: (F_des_N, phi_deg, ori, dh_m, vx_mps, body_mass_kg, ext_rate_N_per_s)
            import json as _json

            lay6 = [
                (0, 30, 1, 0.0, 0.0, 15.0, 25.0),
                (0, 30, 0, 0.0, 0.0, 15.0, 25.0),
                (40, 30, 1, 0.0, 0.0, 15.0, 0.0),
                (60, 30, 1, 0.0, 0.0, 15.0, 0.0),
                (80, 30, 1, 0.0, 0.0, 15.0, 0.0),
                (60, 30, 0, 0.0, 0.0, 15.0, 0.0),
                (60, 60, 1, 0.0, 0.0, 15.0, 0.0),
                (100, 75, 1, 0.0, 0.0, 15.0, 0.0),
                (60, 0, 1, 0.0, 0.0, 15.0, 0.0),
                (60, 30, 1, -0.10, 0.0, 15.0, 0.0),
                (80, 45, 1, -0.10, 0.0, 15.0, 0.0),
                (60, 30, 1, 0.0, -0.2, 15.0, 0.0),
                (40, 30, 1, 0.0, 0.0, 5.0, 0.0),
                (60, 30, 1, 0.0, 0.0, 10.0, 0.0),
                (60, 30, 0, 0.0, 0.0, 5.0, 0.0),
                (0, 30, 1, 0.0, 0.0, 15.0, 0.0),
            ]
            lf = config.get("feas_a6_layout_file", None)
            if lf:
                lay6 = [tuple(r_) for r_ in _json.load(open(str(lf)))]
            if mode == "A6" and config.get("feas_a7", False):
                # A7 (2026-10-01): tilt-feedback force control. Column 8 = tilt target
                # (deg); F_des becomes the force CAP, the commanded force is a PID on
                # tilt (may go negative = hold back past the target; the bin is
                # unstable beyond ~23 deg back).
                lay6 = [
                    (120, 90, 1, 0.0, 0.0, 15.0, 0.0, 15.0),
                    (160, 90, 1, 0.0, 0.0, 15.0, 0.0, 15.0),
                    (120, 90, 1, -0.10, 0.0, 15.0, 0.0, 15.0),
                    (160, 90, 1, -0.10, 0.0, 15.0, 0.0, 15.0),
                    (120, 75, 1, 0.0, 0.0, 15.0, 0.0, 15.0),
                    (160, 75, 1, 0.0, 0.0, 15.0, 0.0, 15.0),
                    (120, 75, 1, -0.10, 0.0, 15.0, 0.0, 15.0),
                    (160, 75, 1, -0.10, 0.0, 15.0, 0.0, 15.0),
                    (160, 60, 1, -0.10, 0.0, 15.0, 0.0, 15.0),
                    (160, 90, 0, -0.10, 0.0, 15.0, 0.0, 15.0),
                    (160, 75, 0, -0.10, 0.0, 15.0, 0.0, 15.0),
                    (160, 90, 1, -0.10, -0.1, 15.0, 0.0, 15.0),
                    (160, 75, 1, -0.15, 0.0, 15.0, 0.0, 15.0),
                    (100, 90, 1, 0.0, 0.0, 10.0, 0.0, 15.0),
                    (80, 90, 1, 0.0, 0.0, 5.0, 0.0, 15.0),
                    (160, 45, 1, -0.10, 0.0, 15.0, 0.0, 15.0),
                ]
                if lf:
                    lay6 = [tuple(r_) for r_ in _json.load(open(str(lf)))]
            lay6 = [tuple(r_) + (0.0,) * (8 - len(r_)) for r_ in lay6]
            lay6 = (lay6 + [(0, 30, 1, 0.0, 0.0, 15.0, 0.0, 0.0)] * N)[:N]
            Fdes, phi6, ori6, dh6, vx6, m6, ext6, tgt6 = (
                torch.tensor([float(r_[j]) for r_ in lay6], device=dev) for j in range(8)
            )
            kp6 = float(config.get("feas_tilt_kp", 10.0))  # N per deg
            ki6 = float(config.get("feas_tilt_ki", 5.0))
            kd6 = float(config.get("feas_tilt_kd", 2.0))
            I6 = torch.zeros(N, device=dev)
            a8 = bool(config.get("feas_a8", False))
            if a8:
                # A8 (2026-10-01): tilt-feedback ARC position control. Each hand's
                # palm target = its bar-attached pose rotated about the current
                # wheel-ground contact line by D = g*(th_des - th) - kd*th_dot
                # (deg, clamped [-10, 15]); orientation rotated by D too. The arm PD
                # stiffness turns the arc error into force and it reverses sign on
                # overshoot (A7's force PID + slow admittance overshot past the
                # ~23 deg balance point). Extra push bias b0 (m) along phi, fading
                # out over the last 5 deg before th_des, to break the tip without
                # rolling the bin into the torso. th_des ramps 0 -> tgt over
                # feas_t_ramp8 s. Row: (tgt_deg, g, kd, b0_m, phi_deg, ori, dh_m, body_mass_kg)
                lay8 = [
                    (15, 1.0, 0.0, 0.00, 90, 1, 0.0, 15.0),
                    (15, 2.0, 0.0, 0.00, 90, 1, 0.0, 15.0),
                    (15, 3.0, 0.0, 0.00, 90, 1, 0.0, 15.0),
                    (15, 2.0, 0.3, 0.00, 90, 1, 0.0, 15.0),
                    (15, 2.0, 0.3, 0.05, 90, 1, 0.0, 15.0),
                    (15, 2.0, 0.3, 0.10, 90, 1, 0.0, 15.0),
                    (15, 2.0, 0.3, 0.05, 60, 1, 0.0, 15.0),
                    (15, 2.0, 0.3, 0.05, 90, 1, -0.10, 15.0),
                    (15, 3.0, 0.3, 0.10, 90, 1, -0.10, 15.0),
                    (15, 2.0, 0.3, 0.05, 90, 0, 0.0, 15.0),
                    (12, 2.0, 0.3, 0.05, 90, 1, 0.0, 15.0),
                    (18, 2.0, 0.3, 0.05, 90, 1, 0.0, 15.0),
                    (15, 2.0, 0.3, 0.05, 90, 1, 0.0, 10.0),
                    (15, 2.0, 0.3, 0.05, 90, 1, 0.0, 5.0),
                    (15, 1.0, 0.3, 0.10, 60, 1, -0.10, 15.0),
                    (15, 3.0, 0.5, 0.05, 45, 1, -0.10, 15.0),
                ]
                if lf:
                    lay8 = [tuple(r_) for r_ in _json.load(open(str(lf)))]
                lay8 = (lay8 + [lay8[-1]] * N)[:N]
                tgt8, g8, kd8, b08 = (torch.tensor([float(r_[j]) for r_ in lay8], device=dev) for j in range(4))
                T_r8 = float(config.get("feas_t_ramp8", 2.5))
                lay6 = [(r_[3] * 100.0, r_[4], r_[5], r_[6], 0.0, r_[7], 0.0, r_[0]) for r_ in lay8]
                Fdes, phi6, ori6, dh6, vx6, m6, ext6, tgt6 = (
                    torch.tensor([float(r_[j]) for r_ in lay6], device=dev) for j in range(8)
                )
                print("[feas] A8 rows (tgt, g, kd, b0, phi, ori, dh, mass): "
                      + " | ".join(f"env{e} {lay8[e]}" for e in range(N)), flush=True)
            phi6 = torch.deg2rad(phi6)
            k_adm = float(config.get("feas_k_adm", 0.004))  # m per (N s)
            w_rot = float(config.get("feas_w_rot", 0.5))
            lam6 = float(config.get("feas_lambda", 0.05))
            d_max = float(config.get("feas_d_max", 0.30))
            e_cap = float(config.get("feas_e_cap", 0.25))
            t_ramp6 = float(config.get("feas_t_ramp", 1.5))
            T6 = float(config.get("feas_T", 6.0))

            # per-env bin body mass (wheels unchanged), inertia scaled
            mcpu = bin_art.root_physx_view.get_masses().clone()
            icpu = bin_art.root_physx_view.get_inertias().clone()
            ratio6 = m6.cpu() / mcpu[:, bi]
            mcpu[:, bi] = m6.cpu()
            icpu[:, bi] = icpu[:, bi] * ratio6[:, None]
            ids_cpu = torch.arange(N, dtype=torch.int32)
            bin_art.root_physx_view.set_masses(mcpu, ids_cpu)
            bin_art.root_physx_view.set_inertias(icpu, ids_cpu)
            masses.copy_(bin_art.root_physx_view.get_masses().to(dev).float())
            for _ in range(int(round(0.3 / dt))):
                step()
            print(f"[feas] A6 body masses {masses[:, bi].tolist()}", flush=True)

            def qmul(a_, b_):
                aw, ax, ay, az = a_.unbind(-1)
                bw, bx, by, bz = b_.unbind(-1)
                return torch.stack([aw * bw - ax * bx - ay * by - az * bz,
                                    aw * bx + ax * bw + ay * bz - az * by,
                                    aw * by - ax * bz + ay * bw + az * bx,
                                    aw * bz + ax * by - ay * bx + az * bw], -1)

            def qconj(q_):
                return q_ * torch.tensor([1.0, -1.0, -1.0, -1.0], device=q_.device)

            def qrot(q_, v_):
                qv = torch.cat([torch.zeros_like(v_[..., :1]), v_], -1)
                return qmul(qmul(q_, qv), qconj(q_))[..., 1:]

            def qlog(q_):  # world-frame axis-angle of a unit quaternion (wxyz)
                q_ = q_ * torch.where(q_[..., :1] < 0, -torch.ones_like(q_[..., :1]), torch.ones_like(q_[..., :1]))
                v_ = q_[..., 1:]
                s_ = v_.norm(dim=-1, keepdim=True).clamp(min=1e-9)
                return v_ / s_ * (2.0 * torch.atan2(s_, q_[..., :1]))

            pb6 = {s: robot.body_names.index(f"{s}_hand_palm_link") for s in sides}
            jb6 = {s: (pb6[s] if floating else pb6[s] - 1) for s in sides}
            jc6 = {s: [dof_ids[j] + off6 for j in arm_env[s]] for s in sides}
            sim6 = {s: [dof_ids[j] for j in arm_env[s]] for s in sides}
            lim_t = None
            for nm in ("joint_effort_limits", "joint_effort_limit"):
                if getattr(robot.data, nm, None) is not None:
                    lim_t = getattr(robot.data, nm)
                    if lim_t.dim() == 1:
                        lim_t = lim_t[None].expand(N, -1)
                    break
            if lim_t is not None:
                print(f"[feas] A6 arm effort limits L {[round(float(lim_t[0, j]), 1) for j in sim6['left']]}",
                      flush=True)
            bnames_r = list(env_u.simulator.body_names)
            other = [i for i, n in enumerate(bnames_r)
                     if not any(x in n for x in ("hand", "ankle", "foot", "toe"))]
            other_names = [bnames_r[i] for i in other]

            xb0 = bin_art.data.body_pos_w[:, bi].clone()
            qb0 = bin_art.data.body_quat_w[:, bi].clone()
            prel, qrel = {}, {}
            for s in sides:
                prel[s] = qrot(qconj(qb0), robot.data.body_pos_w[:, pb6[s]] - xb0)
                qrel[s] = qmul(qconj(qb0), robot.data.body_quat_w[:, pb6[s]])
            dmag = {s: torch.zeros(N, device=dev) for s in sides}
            Ff = {s: torch.zeros(N, device=dev) for s in sides}
            th0 = tilt().clone()
            mt = masses.sum(1)
            Mg6 = mom(com_all(), torch.stack([0 * mt, 0 * mt, -9.81 * mt], -1), contact_pt(), axes()[1])
            root0 = robot.data.root_pos_w[:, :2].clone()
            bin0 = bin_art.data.root_pos_w[:, :2].clone()
            alive = torch.ones(N, dtype=torch.bool, device=dev)
            use6 = ori6 > 0.5
            has_ext = bool((ext6 > 0).any())
            n6 = int(round(T6 / dt))
            th_prev6 = torch.zeros(N, device=dev)
            # optional replay trajectories (play_bin_push.py traj format, env origin
            # subtracted) for move_garbage/scripts/replay_traj.py videos
            tenvs = [int(x) for x in (config.get("feas_traj_envs", None) or [])]
            org = env_u.simulator.scene.env_origins
            trj = {e: {k_: [] for k_ in ("robot_pos", "robot_quat", "bin_pos", "bin_quat", "robot_root",
                                         "robot_jpos", "bin_root", "bin_jpos")} for e in tenvs}
            for i in range(n6):
                t = i * dt
                r = min(1.0, t / t_ramp6)
                a, k = axes()
                u = torch.cos(phi6)[:, None] * (-a) + torch.sin(phi6)[:, None] * (-zhat)
                xb = bin_art.data.body_pos_w[:, bi]
                qb = bin_art.data.body_quat_w[:, bi]
                Fall = env_u.simulator.object_to_hand_contact_forces[:, 0]
                Jf = robot.root_physx_view.get_jacobians()
                th_now = tilt() - th0
                thd6 = (th_now - th_prev6) / dt if i > 0 else torch.zeros_like(th_now)
                th_prev6 = th_now.clone()
                err6 = tgt6 - th_now
                I6 = (I6 + err6 * dt).clamp(-20.0, 20.0)
                Fpid = torch.maximum(torch.minimum(kp6 * err6 + ki6 * I6 - kd6 * thd6, Fdes), -0.5 * Fdes)
                Fcmd = torch.where(tgt6 > 0, r * Fpid, r * Fdes)
                rec("F_cmd", Fcmd)
                if a8:
                    th_des = tgt8 * min(1.0, t / T_r8)
                    D8 = torch.deg2rad((g8 * (th_des - th_now) - kd8 * thd6).clamp(-10.0, 15.0))
                    kax = axes()[1]
                    qD = torch.cat([torch.cos(D8 / 2)[:, None], kax * torch.sin(D8 / 2)[:, None]], -1)
                    c8 = contact_pt()
                    bias8 = b08 * ((th_des - th_now) / 5.0).clamp(0.0, 1.0)
                    rec("th_des", th_des)
                    rec("D8", torch.rad2deg(D8))
                Fh = {}
                clipped = torch.zeros(N, dtype=torch.bool, device=dev)
                for s in sides:
                    Fh[s] = Fall[:, env_u._hand_link_contact_idx[s], :].sum(1)
                    Ff[s] = 0.8 * Ff[s] + 0.2 * (Fh[s] * u).sum(-1)
                    d_lo = torch.where(tgt6 > 0, torch.full_like(Fdes, -0.10), torch.zeros_like(Fdes))
                    dmag[s] = torch.maximum(torch.minimum(dmag[s] + k_adm * (Fcmd / 2.0 - Ff[s]) * dt,
                                                          torch.full_like(Fdes, d_max)), d_lo)
                    pp = robot.data.body_pos_w[:, pb6[s]]
                    pq = robot.data.body_quat_w[:, pb6[s]]
                    xt = xb + qrot(qb, prel[s]) + dmag[s][:, None] * u
                    qtgt = qmul(qb, qrel[s])
                    if a8:
                        p_att = xb + qrot(qb, prel[s])
                        xt = c8 + qrot(qD, p_att - c8) + bias8[:, None] * u
                        qtgt = qmul(qD, qtgt)
                    ep = xt - pp
                    ep = ep * (e_cap / ep.norm(dim=-1, keepdim=True).clamp(min=e_cap))
                    er = qlog(qmul(qtgt, qconj(pq)))
                    J6 = Jf[:, jb6[s], :6, :][:, :, jc6[s]].clone()  # (N,6,7)
                    e6 = torch.cat([ep, w_rot * er], -1)
                    J6[:, 3:, :] *= w_rot
                    J6[~use6, 3:, :] = 0.0
                    e6[~use6, 3:] = 0.0
                    JJt = J6 @ J6.transpose(1, 2) + (lam6 ** 2) * torch.eye(6, device=dev)[None]
                    dq6 = (J6.transpose(1, 2) @ torch.linalg.solve(JJt, e6[..., None]))[..., 0]
                    qt = env_u.simulator.dof_pos[:, arm_env[s]] + dq6.clamp(-0.6, 0.6)
                    for jj, c in enumerate(cols[s]):
                        j = arm_env[s][jj]
                        raw = (qt[:, jj] - default[j]) / scale
                        clipped |= raw.abs() > clip
                        env_u._delta_actions[:, c] = raw.clamp(-clip, clip)
                env_u._delta_actions[:, h_col] = h0 + r * dh6 / cmd_scale
                if vx_col is not None:
                    env_u._delta_actions[:, vx_col] = r * vx6 / cmd_scale
                bm = bar_mid()
                Fext_mag = (ext6 * t).clamp(max=150.0)
                if has_ext:
                    Fw = Fext_mag[:, None] * u
                    tau = torch.cross(bm - com_pos()[:, bi], Fw, dim=-1)
                    pwc = getattr(bin_art, "permanent_wrench_composer", None)
                    if pwc is not None and hasattr(pwc, "_link_poses_updated"):
                        pwc._link_poses_updated = False
                    bin_art.set_external_force_and_torque(
                        Fw[:, None, :].contiguous(), tau[:, None, :].contiguous(), body_ids=[bi], is_global=True
                    )
                c = contact_pt()
                Mb = mom(bm, Fh["left"] + Fh["right"], c, k)
                Mext = mom(bm, Fext_mag[:, None] * u, c, k)
                dones = step()
                alive &= ~dones
                for e in tenvs:
                    o_ = org[e]
                    rr = robot.data.root_state_w[e, :7].clone()
                    rr[:3] -= o_
                    br = bin_art.data.root_state_w[e, :7].clone()
                    br[:3] -= o_
                    trj[e]["robot_pos"].append((robot.data.body_pos_w[e] - o_).cpu().numpy())
                    trj[e]["robot_quat"].append(robot.data.body_quat_w[e].cpu().numpy())
                    trj[e]["bin_pos"].append((bin_art.data.body_pos_w[e] - o_).cpu().numpy())
                    trj[e]["bin_quat"].append(bin_art.data.body_quat_w[e].cpu().numpy())
                    trj[e]["robot_root"].append(rr.cpu().numpy())
                    trj[e]["robot_jpos"].append(robot.data.joint_pos[e].cpu().numpy())
                    trj[e]["bin_root"].append(br.cpu().numpy())
                    trj[e]["bin_jpos"].append(bin_art.data.joint_pos[e].cpu().numpy())
                lc, rc = env_u._finger_closure_fractions()
                cf = env_u.simulator.contact_forces.norm(dim=-1)[:, other]
                rec("t", t)
                rec("tilt", tilt())
                rec("M_bar", Mb)
                rec("M_ext", Mext)
                rec("F_ext", Fext_mag)
                rec("Fu_L", (Fh["left"] * u).sum(-1))
                rec("Fu_R", (Fh["right"] * u).sum(-1))
                rec("Fz", (Fh["left"] + Fh["right"])[:, 2])
                rec("d_L", dmag["left"])
                rec("d_R", dmag["right"])
                rec("grip", env_u._grip_maintained_mask())
                rec("cl_L", lc)
                rec("cl_R", rc)
                rec("bin_da", ((bin_art.data.root_pos_w[:, :2] - bin0) * a[:, :2]).sum(-1))
                rec("root_da", ((robot.data.root_pos_w[:, :2] - root0) * a[:, :2]).sum(-1))
                rec("root_z", robot.data.root_pos_w[:, 2].clone())
                rec("clipped", clipped)
                rec("other_cf", cf)
                rec("alive", alive)
                if lim_t is not None:
                    ids_ = sim6["left"] + sim6["right"]
                    rec("tq_ratio", (robot.data.applied_torque[:, ids_].abs()
                                     / lim_t[:, ids_].clamp(min=1e-6)).max(dim=-1).values)
            for e in tenvs:
                tp = out_path.replace(".npz", f"_env{e}_traj.npz")
                np.savez_compressed(tp, **{k_: np.stack(v) for k_, v in trj[e].items()},
                                    robot_body_names=np.array(list(robot.data.body_names)),
                                    bin_body_names=np.array(list(bin_art.data.body_names)),
                                    robot_joint_names=np.array(list(robot.data.joint_names)),
                                    bin_joint_names=np.array(list(bin_art.data.joint_names)))
                print(f"[feas] traj env {e} -> {tp}", flush=True)
            L = {k_: np.stack(v) if isinstance(v[0], np.ndarray) else np.array(v) for k_, v in log.items()}
            np.savez(out_path, **L, layout=np.array(lay6, dtype=np.float32), Mg6=Mg6.cpu().numpy(),
                     other_names=np.array(other_names))
            tt = L["t"]
            hold = tt >= t_ramp6
            rows = []
            for e in range(N):
                ok = L["alive"][:, e].astype(bool)
                okh = ok & hold
                tl = L["tilt"][:, e] - float(th0[e])
                i_max = int(np.argmax(np.where(ok, tl, -1e9)))
                t2 = np.where(ok & (tl >= 2.0))[0]
                g = L["grip"][:, e].astype(bool)
                lost = np.where(~g & ok)[0]
                ocf = L["other_cf"][okh, e, :].mean(0) if okh.any() else np.zeros(len(other_names))
                top = np.argsort(-ocf)[:2]
                top_s = ", ".join(f"{other_names[j]} {ocf[j]:.0f}N" for j in top if ocf[j] > 2.0) or "none"
                ext_s = ""
                if lay6[e][6] > 0:
                    on = np.where(ok & (tl >= 0.5))[0]
                    ext_s = (f" | EXT onset F {L['F_ext'][on[0], e]:.1f} N M {L['M_ext'][on[0], e]:.1f} Nm"
                             if len(on) else " | EXT no onset")
                def mh(key):
                    return L[key][okh, e].mean() if okh.any() else float("nan")
                tq_s = f"{L['tq_ratio'][okh, e].max():.2f}" if ("tq_ratio" in L and okh.any()) else "-"
                last = ok & (tt >= tt[-1] - 3.0)
                tli = tl[last] if last.any() else np.array([np.nan])
                ti_ = other_names.index("torso_link") if "torso_link" in other_names else None
                torso_s = f"{L['other_cf'][last, e, ti_].mean():.0f}N" if (ti_ is not None and last.any()) else "-"
                hold_s = (f"LAST3s tilt {np.nanmean(tli):+5.1f}+-{np.nanstd(tli):4.1f} band10-22 "
                          f"{np.mean((tli >= 10) & (tli <= 22)):.2f} grip {L['grip'][last, e].mean():.2f} "
                          f"torso {torso_s} Fcmd {L['F_cmd'][last, e].mean():+5.0f} || ")
                rows.append(
                    hold_s + f"env{e:2d} tgt {lay6[e][7]:4.1f} F {lay6[e][0]:4.0f} phi {lay6[e][1]:+3.0f} ori {int(lay6[e][2])} dh {lay6[e][3]:+.2f} "
                    f"vx {lay6[e][4]:+.1f} m {lay6[e][5]:4.1f} (Mg {-float(Mg6[e]):4.1f}) | TILT max {tl[i_max]:+5.1f} "
                    f"end {tl[ok][-1] if ok.any() else float('nan'):+5.1f} t2deg "
                    f"{tt[t2[0]] if len(t2) else float('nan'):.2f}s | M_bar mean {mh('M_bar'):+5.1f} "
                    f"p90 {np.percentile(L['M_bar'][okh, e], 90) if okh.any() else float('nan'):+5.1f} | "
                    f"Fu L/R {mh('Fu_L'):5.1f}/{mh('Fu_R'):5.1f} Fz {mh('Fz'):+5.1f} | d L/R "
                    f"{mh('d_L') * 100:4.1f}/{mh('d_R') * 100:4.1f}cm | grip "
                    f"{(tt[lost[0]] if len(lost) else -1):.2f}s frac {g[ok].mean():.2f} cl min "
                    f"{L['cl_L'][ok, e].min():.2f}/{L['cl_R'][ok, e].min():.2f} | bin da {L['bin_da'][-1, e] * 100:+.1f} "
                    f"root da {L['root_da'][-1, e] * 100:+.1f} z {L['root_z'][-1, e]:.3f} | tq {tq_s} clip "
                    f"{L['clipped'][ok, e].mean():.2f} | other contact: {top_s}{ext_s} | alive {int(ok.sum())}/{len(ok)}"
                )
            summarize_print(rows, "A6 SUMMARY (force-controlled closed-loop IK; tilt relative to t0)")
            print(f"[feas] saved {out_path}", flush=True)
            sys.stdout.flush()
            os._exit(0)
        a0, k0 = axes()
        dxw = (torch.cos(phis)[:, None] * (-a0) + torch.sin(phis)[:, None] * (-zhat)) * mags[:, None]
        dq = {}
        q0 = {}
        lam = 0.02
        for s in sides:
            pb = robot.body_names.index(f"{s}_hand_palm_link")
            jb = pb if floating else pb - 1
            jc = [dof_ids[j] + off6 for j in arm_env[s]]
            J = J_full[:, jb, :3, :][:, :, jc]  # (N,3,7)
            JJt = J @ J.transpose(1, 2) + (lam ** 2) * torch.eye(3, device=dev)[None]
            dq[s] = (J.transpose(1, 2) @ torch.linalg.solve(JJt, dxw[..., None]))[..., 0]
            q0[s] = torch.stack([default[j] + scale * env_u._delta_actions[:, c] for j, c in
                                 zip(arm_env[s], cols[s])], -1).clone()
            print(f"[feas] {s} |dq| per env {[round(float(x), 3) for x in dq[s].norm(dim=-1)]}", flush=True)
        t_ramp = 1.0
        T_total = float(config.get("feas_T", 4.0))
        n_steps = int(round(T_total / dt))
        alive = torch.ones(N, dtype=torch.bool, device=dev)
        tq = getattr(env_u, "torques", None)
        tlim = getattr(env_u, "torque_limits", None)
        arm_all = arm_env["left"] + arm_env["right"]
        for i in range(n_steps):
            t = i * dt
            r = min(1.0, t / t_ramp)
            qerr = torch.zeros(N, device=dev)
            for s in sides:
                qt = q0[s] + r * dq[s]
                for jj, c in enumerate(cols[s]):
                    j = arm_env[s][jj]
                    env_u._delta_actions[:, c] = ((qt[:, jj] - default[j]) / scale).clamp(-clip, clip)
                qerr = qerr + (qt - env_u.simulator.dof_pos[:, arm_env[s]]).abs().mean(-1) / 2
            env_u._delta_actions[:, h_col] = h0 + r * dh / cmd_scale
            env_u._delta_actions[:, p_col] = p0 + r * dp / cmd_scale
            if vx_col is not None:
                env_u._delta_actions[:, vx_col] = r * vx / cmd_scale
            dones = step()
            alive &= ~dones
            a, k = axes()
            c = contact_pt()
            Fall = env_u.simulator.object_to_hand_contact_forces[:, 0]  # (N,L,3)
            Ml = torch.zeros(N, device=dev)
            Fsum = torch.zeros(N, 3, device=dev)
            per_hand = []
            for s in sides:
                ci = env_u._hand_link_contact_idx[s]
                si = env_u._hand_link_sim_idx[s]
                Fh = Fall[:, ci, :]
                P = env_u.simulator._rigid_body_pos[:, si, :]
                Ml = Ml + (torch.cross(P - c[:, None, :], Fh, dim=-1) * k[:, None, :]).sum(-1).sum(-1)
                Fsum = Fsum + Fh.sum(1)
                per_hand.append(Fh.norm(dim=-1).sum(-1))
            Mb = mom(bar_mid(), Fsum, c, k)
            lc, rc = env_u._finger_closure_fractions()
            grip = env_u._grip_maintained_mask()
            ld, rd = env_u._hand_bar_distances()
            ovh = env_u._overhand_finger_terms()
            rec("t", t)
            rec("tilt", tilt())
            rec("M_link", Ml)
            rec("M_bar", Mb)
            rec("F_a", (Fsum * a).sum(-1))
            rec("F_z", Fsum[:, 2])
            rec("F_L", per_hand[0])
            rec("F_R", per_hand[1])
            rec("cl_L", lc)
            rec("cl_R", rc)
            rec("grip", grip)
            rec("d_L", ld)
            rec("d_R", rd)
            rec("dsurf_L", ovh["left"][2].max(dim=-1).values)
            rec("dsurf_R", ovh["right"][2].max(dim=-1).values)
            rec("wrap_L", ovh["left"][4])
            rec("wrap_R", ovh["right"][4])
            rec("stage", env_u.stage_buf.clone())
            rec("qerr", qerr)
            rec("root_da", ((robot.data.root_pos_w[:, :2] - root0) * a[:, :2]).sum(-1))
            rec("root_z", robot.data.root_pos_w[:, 2].clone())
            rec("bin_da", ((bin_art.data.root_pos_w[:, :2] - bin0) * a[:, :2]).sum(-1))
            rec("alive", alive)
            if tq is not None and tlim is not None:
                tl_ = tlim if tlim.dim() == 1 else tlim[0]
                rec("tq_sat", (tq[:, arm_all].abs() / tl_[arm_all].clamp(min=1e-6)).max(dim=-1).values)
        L = {k_: np.stack(v) if isinstance(v[0], np.ndarray) else np.array(v) for k_, v in log.items()}
        np.savez(out_path, **L, phi_deg=np.rad2deg(phis.cpu().numpy()), mag_m=mags.cpu().numpy(),
                 dh=dh.cpu().numpy(), dp=dp.cpu().numpy(), vx=vx.cpu().numpy())
        tt = L["t"]
        rows = []
        for e in range(N):
            ok = L["alive"][:, e].astype(bool)
            g = L["grip"][:, e].astype(bool)
            i_max = int(np.argmax(np.where(ok, L["tilt"][:, e], -1e9)))
            lost = np.where(~g & ok)[0]
            lost_s = f"{tt[lost[0]]:.2f}s" if len(lost) else "never"
            hold = slice(int(round(t_ramp / dt)), None)
            sat = f"{L['tq_sat'][hold, e].mean():.2f}" if "tq_sat" in L else "-"
            rows.append(
                f"env{e:2d} m_body {float(masses[e, bi]):4.1f} phi {np.rad2deg(float(phis[e])):+4.0f} dx {float(mags[e]) * 100:4.0f}cm "
                f"dh {float(dh[e]):+.2f} dp {float(dp[e]):+.1f} vx {float(vx[e]):+.1f} | "
                f"tilt max {L['tilt'][i_max, e]:+5.1f} end {L['tilt'][ok, e][-1] if ok.any() else float('nan'):+5.1f} | "
                f"M_link mean(hold) {L['M_link'][hold, e].mean():+6.1f} max {L['M_link'][ok, e].max():+6.1f} | "
                f"M_bar mean {L['M_bar'][hold, e].mean():+6.1f} | F_a {L['F_a'][hold, e].mean():+6.1f} F_z "
                f"{L['F_z'][hold, e].mean():+6.1f} | |F| L/R {L['F_L'][hold, e].mean():5.1f}/{L['F_R'][hold, e].mean():5.1f} "
                f"max {(L['F_L'][:, e] + L['F_R'][:, e]).max():5.1f} | closure min L/R {L['cl_L'][ok, e].min():.2f}/"
                f"{L['cl_R'][ok, e].min():.2f} | dsurf max L/R {L['dsurf_L'][ok, e].max() * 100:4.1f}/"
                f"{L['dsurf_R'][ok, e].max() * 100:4.1f}cm | grip lost {lost_s} (grip frac {g[ok].mean():.2f}) | "
                f"wrapped frac L/R {L['wrap_L'][ok, e].mean():.2f}/{L['wrap_R'][ok, e].mean():.2f} | "
                f"qerr(hold) {L['qerr'][hold, e].mean():.3f}rad | bin da {L['bin_da'][-1, e] * 100:+.1f}cm | root da {L['root_da'][-1, e] * 100:+.1f}cm z "
                f"{L['root_z'][0, e]:.3f}->{L['root_z'][-1, e]:.3f} | M_bar p90 {np.percentile(L['M_bar'][hold, e], 90):+.1f} | "
                f"tq_sat {sat} | stage max {int(L['stage'][:, e].max())} | alive {int(ok.sum())}/{len(ok)}"
            )
        summarize_print(rows, "A2 SUMMARY (contact tensor used as-is, s=+1; F_a = sum force . push axis)")
    print(f"[feas] saved {out_path}", flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
