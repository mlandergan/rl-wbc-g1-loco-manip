# Base Task Usages

## Staged Task Base

### Purpose

`StagedTaskBase` is suited for RL tasks requiring different behaviors at different stages, which requires formulating different rewards for different stages. It manages a linear state machine with timing constraints enforced at every stage. It checks for user-defined state transition condition and automatically advance stage of the task. It also provides a helper decorator to constrain the activation of a reward function only to particular stages.

### Basic Usage

#### Define Staging Logics

The user should start by having a mental picture of how many stage is required in their task. This part of the semantics is not managed by `StagedTaskBase`, and it is totally up to the user to define the meaning of every stage. For example, a grasping task can be decomposed into 3 stages: reaching pre-grasp pose, grasping object, and lifting object. More meticulous seperation of stage can give more fine-grained behavior, but it could hurt RL learning, as well.

Once the number of stages, and how long each stage should be is determined, write up the env config:

1. `max_stage_time`: the time limit of each stage (in number of steps)
2. `stage_reward_scale`: the base reward value for being in that stage (to be used below with `stage` reward)
3. `award_remaining_time_on_advance`: Default to `true`. When a stage transition happens before the time limit hits, transfer the remaining time to the next stage.
4. `no_reward_for_awarded_time`: Default to `false`. Deactivate stage reward during the awarded time from last stage. This can be useful to discourage the RL agent from rushing into the next stage where there might be more total reward than the last.
5. `accumulate_stage_reward`: Default to `true`. Add up the base stage reward from previous stages. For example, if the `stage_reward_scale` is `[1, 2, 1]`, then the actual stage reward scale is `[1, 1 + 2 & cond(), 1 + 2 + 1 & cond()]`, where `cond()` is the stage reward condition of the current stage. It encourages the agent to quickly moving into the next stage, because it is always guaranteed there to be more base reward. However, it makes it harder to balance total reward amount for tasks with many stages.
6. `reset_on_overtime`: Default to `true`. Terminate when the stage time limit hits, and apply termination penalty.
7. `reset_on_complete`: Default to `false`. Reset the env when the task completion condition is met, without termination penalty.
8. `no_reward_for_overtime`: Default to `false`. Only effective when `reset_on_overtime` is `false`. Disable stage reward when over the current stage's time limit.

Here is an example config:

```yaml
env:
  config:
    task:
      max_stage_time: [100, 50, 100]
      stage_reward_scale: [1.0, 1.0, 1.0]
      award_remaining_time_on_advance: true
      no_reward_for_awarded_time: false
      accumulate_stage_reward: true
      reset_on_overtime: true
      reset_on_complete: false
      # no_reward_for_overtime: false  ## only useful if reset_on_overtime is false.
```

Afterwards, make sure that `env.config.max_episode_length_s` is longer than the total `max_stage_time` (mind that they have different units). Otherwise, the RL agent may not be penalized when it has not completed the task but has reached the env time limit.

Then, you need to define the transition logics for every stage in your env subclass. They are in the format of:

1. `_stage_{i}_to_{i+1}_advance_condition`: what condition should the agent satisfy to advance onto the next stage?
2. `stage_{i}_reward_condition`: what condition should be met to give the current stage's base reward?
3. `_stage_{i_max}_to_complete_condition`: what condition signals task completion?
4. `_stage_{i}_to_{i+1}_advance_callback` (optional): what to do when the stage actually advances?

Note that you do not have to check if agent is actually in stage `i` in these condition functions. It is handled automatically.

For example:

```python
class PickUpCube(StagedTaskBase):
  def _stage_0_reward_condition(self):
    # always assign stage 0 reward (this is just an example)
    return torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

  def _stage_0_to_1_advance_condition(self):
    # check if the hand reaches the pre-grasp pose
    ...

  def _stage_0_to_1_advance_callback(self, env_ids):
    # do something when the stage 0 to 1 advance actually happens
    ...

  def _stage_1_reward_condition(self):
    # stay at the pre-grasp pose to get stage 1 reward
    ...

  def _stage_1_to_2_advance_condition(self):
    # check if the cube is grasped
    ...

  def _stage_2_reward_condition(self):
    # keep grasping the cube
    ...

  def _stage_2_to_complete_condition(self):
    # check if target lifting pose is reached
    ...
```

These function names will be automatically picked up by `StageTaskBase`.

#### Limit Reward Scope to Particular Stages

If you want a reward to be only effective in select stage, use `@StagedTaskBase.effective_in_stage` decorator.

```python
class PickUpCube(StagedTaskBase):
    @StagedTaskBase.effective_in_stage(0)
    def _reward_approach_cube(self):
        ...

    @StagedTaskBase.effective_in_stage([1, 2])
    def _reward_grasp_cube(self):
        ...

    @StagedTaskBase.effective_in_stage(2)
    def _reward_lift_cube(self):
        ...
```

#### Hook Up Stage Observations and Rewards

Here are optional reward terms from `StagedTaskBase`:

1. `stage` (highly recommended): gives the base stage reward value mentioned above.
2. `penalty_overtime`: only effective when `reset_on_overtime` is `false`. Gives a constant penalty when the current stage time exceeds limit.
3. `time_saving`: only effective when `award_remaining_time_on_advance` is `true`. Gives a constant reward during the awarded time from the last stage.
4. `transition`: gives an one-time reward at stage transition. Typically it needs a high scale to take effect. Use with caution as it introduces spike to the value function.
5. `complete`: gives an one-time reward at task completion.

Here are optional observation terms from `StageTaskBase`:

1. `stage`: one-hot-encoded current stage number. Length is `${eval:'len(${env.config.max_stage_time})'}`.
2. `time_in_stage`: time spent in the current stage, minus awarded time from the last stage. It is negative when the env is using the awarded time from the last stage. **It should be scaled properly as it might be in the hundreds.**
3. `actual_time_in_stage`: time spent in the current stage, including awarded time from the last stage. It is non-negative.  **It should be scaled properly as it might be in the hundreds.**
4. `transition`: 1 if a transition just happened. 0 otherwise.
5. `complete`: 1 if the task completion condition is met. 0 otherwise.

### Staged Reset (Advance Usage)

To assist with exploration of later stages, `StagedTaskBase` supports resetting an environment directly to later stages that the agent has already been in. It maintains a rolling buffer that takes a snapshot of the task, including the states of the robot and task objects, whenever the robot enters a stage. It then randomly sample from this rolling buffer when resetting.

#### Step 1: Set Config

The parameters for this feature are:

1. `enable_staged_reset`: Default to `False`. Enable staged reset.
2. `staged_reset_ratios`: A list of floats specifying the probability of resetting to a particular stage. Note you should always have non-zero ratio for stage 0.
3. `staged_reset_max_samples_per_stage`: Length of the rolling buffer that stores the snapshots.

```yaml
env:
  config:
    task:
      enable_staged_reset: true
      staged_reset_ratios: [0.3, 0.4, 0.3]  # 30% stage 0, 40% stage 1, etc.
      staged_reset_max_samples_per_stage: 100  # store 100 snapshots per env per stage
```

Note that the rolling buffer is per-env to support envs with different task objects.

#### Step 2: Hook Up Custom Task Object and Buffer

By default, `StagedTaskBase` only tracks the robot state in the buffer. To also letting it track custom task objects or buffers, use these methods at initialization:

- `self._register_task_state_to_track(obj: Articulation | RigidObject, name: str)` for tracking custom objects in IsaacSim. Make sure the name matches the simulator's record.
- `self._register_buffer_to_track(name: str, shape: tuple[int, ...], store_callback: Callable, load_callback: Callable, dtype: torch.dtype = torch.float)` for tracking custom buffer. `shape` should be `(num_envs, ...)`. `store_callback` will be called at snapshot time with `env_ids` in its argument, and expect a returned tensor of shape `(len(env_ids), ...)`. `load_callback` will be called when resetting with `env_ids` and `data` in its argument, where `data` is of shape `(len(env_ids), ...)`.

## Homie Base

### Purpose

`HomieBase` manages a robot doing loco-manipulation tasks, whose lower body is controlled by a HOMIE policy. The user's RL agent outputs the commands, such as linear velocity, torso rotation, to control the HOMIE policy, and at the same time outputs upper body target joint angles to perform manipulation.

**It makes the following important assumptions in the robot config**:

1. In `dof_names`, All lower-body DoFs are declared first. They are controlled by HOMIE.
2. In `dof_names`, All upper-body DoFs follow lower-body DoFs.

### How to Use

Refer to `groot/rl/config/exp/loco_manip/homie_base.yaml` and `TestHomieBase` about the setup of HOMIE tasks.

#### What Is the Dimension of My RL Actor Output?

Your RL model controls two things:

1. The high-level command sent to HOMIE. The full HOMIE controller takes 7 commands: linear vel x, linear vel y, angular vel z, base height, torso roll, torso pitch, and torso yaw. You can selectively activate some of them to control in `obs.homie_command_keys`, while leaving the rest to default values set in `obs.homie_command_default`. The length of everything in `obs.homie_command_keys` is `HomieBase._num_homie_active_commands`. It is automatically computed.

2. The upper-body DoFs of your robot. For example, on Unitree G1 29 DoF version with two Dex 3-1 hands, that is 29 + 2 * 7 - 15 = 28. If using other base classes like `FingerPrimitive`, Deduce accordingly. The length of this part is `HomieBase._num_non_homie_command_actions`. It is automatically computed.

Now add up`_num_homie_active_commands` `_num_non_homie_command_actions`. Put the value in `algo.config.actor.backbone.module_config_dict.output_dim`. Your model will output HOMIE commands in the front, and the rest of the non-homie actions follow.

#### How is My Subclass Env Affected?

Your subclass will remain mostly agnostic about the fact that it is actually HOMIE that is managing locomotion. It will receive the same action as if HOMIE is not used. This is because under the hood, `HomieBase` concatenates HOMIE policy output (lower body control) and part of your own actor output (upper body control) as action given to the subclass env. This is also why it is critical to order the DoFs in robot config in the way mentioned above.

You should take advantage of this fact which will allow easy ablation study on HOMIE vs no-HOMIE.

## Finger Primitive Base

### Purpose

Instead of learning the full finger movements as individual actions, This class bundles them into simple open-close primitives, such that an entire hand can be controlled with a single action in a gripper style.

**It makes the following important assumptions in the robot config**:

1. In `dof_names`, All finger DoFs are at the end of the list.

If using with `HomieBase`, both classes' requirements need to be satisfied at the same time.

### How to Use

Include in robot configuration:

```yaml
robot:
  actions_dim: 31  # Unitree G1: body dof 29 + finger primitive 2
  check_action_dim: false  # very likely your robot action dim will not match dof dim after using primitive
  finger_primitive:
    num_non_primitive_dof: 29  # Unitree G1: total dof 43 - finger dof 14
    primitive_action_map:
      left:
        dof_names: [
          'left_hand_index_0_joint', 'left_hand_middle_0_joint', 'left_hand_thumb_0_joint',
          'left_hand_index_1_joint', 'left_hand_middle_1_joint', 'left_hand_thumb_1_joint',
          'left_hand_thumb_2_joint'
        ]
        pos_0: null
        pos_1: null  
        inheritrange: 1.0  # same function as Mujoco inheritrange
        mode: "linear"  # "linear" or "discrete"
      right:
        dof_names: [
          'right_hand_index_0_joint', 'right_hand_middle_0_joint', 'right_hand_thumb_0_joint',
          'right_hand_index_1_joint', 'right_hand_middle_1_joint', 'right_hand_thumb_1_joint',
          'right_hand_thumb_2_joint'
        ]
        pos_0: null
        pos_1: null
        inheritrange: 1.0  # same function as Mujoco inheritrange
        mode: "linear"  # "linear" or "discrete"
```

Explanations:

1. `pos_0` and `pos_1`: two poses at the two ends of the primitive action map. They should be filled as lists of floats in the order of `dof_names` defined in the primitive action map.
2. `mode`: `linear` means the action output interpolates between `pos_0` and `pos_1`, with -1 being `pos_0` and 1 being `pos_1`. `discrete` means the action switches between `pos_0` and `pos_1` based on whether the action is less than or greater or equal to 0.
2. `inheritrange`: effective only when `mode` is `linear`. Similar to Mujoco setting with the same name, it allows control effort beyond the joint limits.

See example in `groot/rl/config/exp/loco_manip/homie_primitive_test.yaml` and `TestHomieWithFingerPrimitive`.

## Delta Action Base

### Purpose

`DeltaActionBase` converts some action indices to delta action and maintains an internal buffer that accumulates the delta actions.

### How to Use

**Important: If using `DeltaActionBase` with the other base controllers, it is critical to use the inheritance order as below:**

```python
class MyTask(StagedTaskBase, DeltaActionBase, HomieBase, FingerPrimitiveBase, ResetFromDataset):
    ...
```

Parameters:

1. `delta_action_indices`: list of action indices to convert to delta action. These indices correspond to `actor_state["actions"]` passed to the environment's `step()` function by the trainer.
2. `delta_action_scale`: scale the delta action, typically smaller than 1 (such as 0.1, 0.3, 0.5).
3. `reset_delta_actions_with_backmap`: default to `false`. If true, at environment reset, heuristically compute the values in delta action buffer from joint angle, such that it does not jerk. If false, the delta action buffer is filled with zeros.

Observations:

1. `delta_actions` (Required): returns the maintained delta action buffer.

If using staged reset function from `StagedTaskBase`, it is recommended to store the delta action buffer at initialization of your custom task:

```python
self._register_buffer_to_track("delta_actions", self._get_delta_actions_buffer_shape(), self._store_delta_actions_buffer, self._load_delta_actions_buffer, dtype=torch.float32)
```