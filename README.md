# Move Garbage

Teacher-policy reinforcement learning for a Unitree G1 humanoid that approaches,
grasps, and pushes a two-wheeled recycling bin. The policy controls locomotion
commands, both arms, and hand primitives while frozen HOMIE policies provide
lower-body walking and standing control.

This repository is a focused, portable extraction of the runtime needed for the
bin-push task. Unrelated tasks, datasets, deployment bundles, experiment logs,
and intermediate checkpoints are intentionally excluded.

## Origin and license

This repository is a fork of NVIDIA's
[GR00T-VisualSim2Real](https://github.com/NVlabs/GR00T-VisualSim2Real)
(`doorman` branch), extracted and modified for the bin-push task. Several
design decisions in `move_garbage/envs/bin_push_homie.py` are modeled directly
on that branch's `envs/door/door_open_homie.py` — see the in-code comments
citing it.

It is licensed under the Apache License 2.0, the same license as the upstream
project — see `LICENSE`. See `THIRD_PARTY_NOTICES.md` for runtime dependencies
and binary-asset redistribution notes.

## Included

- Bin-push environment, rewards, observations, and experiment configuration.
- Minimal GR00T PPO/TRL and Isaac Lab runtime used by the task.
- Inlined G1 simulation asset.
- Local bin USD and textures.
- Frozen HOMIE walking and standing policies.
- Stable baseline checkpoint at iteration 12,900 and its effective run config.
- Quantitative evaluation and trajectory replay tools.

## External prerequisites

- Ubuntu 22.04
- Python 3.11
- NVIDIA GPU and a CUDA 12.8-compatible driver
- Isaac Sim 5.1.0
- Isaac Lab 2.3.2.post1

Isaac Sim and Isaac Lab are platform dependencies and are not vendored here.

## Environment

Create and activate a Python 3.11 environment, then install the simulation
platform according to its official instructions. Install the remaining declared
dependencies and this project:

```bash
pip install -r requirements-training.txt
pip install -e .
python scripts/preflight.py
```

The PyTorch command may need the CUDA-specific package index for your system.
The validated environment uses Python 3.11.13, PyTorch 2.7.0 with CUDA 12.8,
and the exact direct dependency versions in `requirements-training.txt`.
Optional ONNX export and interactive visualization dependencies are available
as the `export` and `visualization` extras.

## Smoke test

Run from the repository root:

```bash
HYDRA_FULL_ERROR=1 WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0 \
python -u gr00t/rl/train_agent_trl.py \
  +exp=wbmanip/bin_push_homie_lstm \
  ++num_envs=4 \
  ++algo.trl.num_total_batches=1 \
  ++use_wandb=False \
  'hydra.searchpath=[file://'"$PWD"'/move_garbage/config]'
```

## Reference checkpoint

`move_garbage/checkpoints/model_step_012900.pt` is the stable baseline snapshot
selected for this release:

```text
SHA-256: f148b48925d57dea4a1ff277e5a272ce7d98b21caac24613f171f8f46666e1ed
```

The checkpoint contains policy, value, optimizer, scheduler, and staged
environment state. Over its final 20 iterations through 12,900, the baseline
averaged 82.4% bilateral contact, 88.7% broad bilateral proximity, 62.0%
bilateral alignment, and 1.56 cm lateral error. It had not yet crossed the
strict bilateral finger-close gate.

`move_garbage/checkpoints/config.yaml` records the effective baseline
configuration, with its checkpoint path normalized for this package.

## Resume with the latest rewards

The current reward configuration adds anisotropic grasp-point shaping with an
8 cm horizontal width and a tighter 6 cm vertical width. The hard finger-close
gate remains 4 cm horizontal and 3 cm vertical. This command reproduces the
current fine-tuning recipe, including TensorBoard logging:

```bash
CKPT="$PWD/move_garbage/checkpoints/model_step_012900.pt"

HYDRA_FULL_ERROR=1 WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0 \
python -u gr00t/rl/train_agent_trl.py \
  +exp=wbmanip/bin_push_homie_lstm \
  ++num_envs=4096 \
  checkpoint="$CKPT" \
  experiment_name=bin_push_anisotropic_resume \
  ++algo.trl.num_total_batches=50000 \
  ++callbacks.model_save.save_frequency=100 \
  ++algo.config.load_optimizer=True \
  ++algo.trl.report_to=tensorboard \
  '++algo.trl.logging_dir=${experiment_dir}/tensorboard' \
  ++use_wandb=False \
  env.config.stop_reward_mode=additive \
  env.config.stage_advance_require_resting_arms=False \
  rewards.reward_scales.grasp_target_anisotropic=6.0 \
  env.config.grasp_target_anisotropic_horiz_std=0.08 \
  env.config.grasp_target_anisotropic_vert_std=0.06 \
  'hydra.searchpath=[file://'"$PWD"'/move_garbage/config]'
```

To continue the exact baseline reward instead, use the same command with
`rewards.reward_scales.grasp_target_anisotropic=0.0`. The baseline and current
fine-tune both preserve the checkpoint optimizer state.

## Preserve ground-contact fidelity

The checkpoint and frozen HOMIE locomotion policies expect Isaac Lab's native
infinite ground plane. Keep `TerrainImporterCfg(terrain_type="plane")` in
`gr00t/rl/simulator/isaacsim/isaacsim.py`. Do not replace it with a generated
`MeshPlaneTerrainCfg`: although the rendered surfaces look equivalent, their
contact behavior is not equivalent for this policy.

A generated mesh produced repeated low-height terminations, pelvis heights near
the 0.50 m cutoff, and ankle-link origins below the ground. With the same
checkpoint and seed on the native plane, pelvis height remained 0.728-0.785 m
and the only reset in 300 steps was the configured episode timeout. When
changing assets, terrain, or simulator versions, repeat this one-checkpoint A/B
rollout before continuing training.

## Visualize final gripper targets

Launch Isaac Sim with only the recycling bin and the two DEX hands:

```bash
python -u move_garbage/scripts/show_gripper_targets.py
```

The script derives the closed finger pose from the checkpoint-era primitive,
cross-assigns the named bin targets to the anatomical hands, orients each
hand's fingers inward through the handle face with its palm side down, and
places the target 4 cm behind and 1 cm above the physical handle before
compensating for the 4.64 cm virtual grasp-point offset. Green/orange spheres mark the exact
left/right stand-off targets, blue spheres mark the palm origins, and RGB axes
show each target orientation. No torso, arms, or other robot geometry is
spawned.

For a headless composition check:

```bash
python -u move_garbage/scripts/show_gripper_targets.py \
  --headless --frames 2 --save-stage /tmp/gripper_targets_validation.usda
```

## Evaluate

Quantitative rollout:

```bash
python -u move_garbage/scripts/measure_standoff.py \
  +checkpoint="$PWD/move_garbage/checkpoints/model_step_012900.pt" \
  ++num_envs=128 ++headless=True
```

Single-policy rollout and trajectory capture:

```bash
python -u move_garbage/scripts/play_bin_push.py \
  +checkpoint="$PWD/move_garbage/checkpoints/model_step_012900.pt" \
  ++num_envs=1
```

Generate a rollout video for every 500-iteration checkpoint written by a
running experiment:

```bash
python -u move_garbage/scripts/watch_training_videos.py \
  --run-dir "$PWD/logs_rl/g1_bin_push_homie/<experiment-directory>" \
  --interval 500 --fps 25 --stride 2
```

The watcher waits for each checkpoint file to become stable, records a policy
trajectory, renders it through `replay_traj.py`, and stores the MP4 plus logs in
the run's `videos_every_500/` directory. Its state file prevents duplicate
renders after restart. Run the watcher in a separate simulation process and,
when available, assign it a GPU other than the training GPU.

### Replay at simulation speed

The policy timestep is the physics timestep multiplied by control decimation:
`0.005 s * 4 = 0.020 s`, or 50 policy steps per second. When replay skips
trajectory frames, preserve wall-clock speed with:

```text
output_fps = 1 / (physics_dt * control_decimation * stride)
```

The default `stride=2` therefore requires `fps=25` for 1x playback. Encoding
those frames at 30 FPS produces 1.2x playback. The watcher command above pins
the correct pairing explicitly.

For a captured `/tmp/bin_traj.npz`, render manually at 1x with:

```bash
python -u move_garbage/scripts/replay_traj.py \
  --headless --enable_cameras \
  --traj /tmp/bin_traj.npz --out /tmp/bin_rollout_1x.mp4 \
  --fps 25 --stride 2
```

## Layout

```text
gr00t/                         Minimal training framework
move_garbage/                  Task code, configuration, and bin assets
models/                        Frozen HOMIE policies
move_garbage/checkpoints/      Stable baseline checkpoint and saved config
scripts/preflight.py           Runtime and artifact validation
move_garbage/scripts/show_gripper_targets.py
                               Hand-only target-pose visualizer
```

See `move_garbage/STAGE_REWARD_FLOW.md` for the staged reward and transition
design. See `THIRD_PARTY_NOTICES.md` before redistributing binary assets.
