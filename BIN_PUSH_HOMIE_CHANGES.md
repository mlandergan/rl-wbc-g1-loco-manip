# `bin_push_homie` changes since `da8a7f9`

> This file covers the **uncommitted** working-tree changes made 2026-09-24 → 2026-10-07.
> - **Base commit:** `da8a7f9` (2026-09-12), "Fix grasp reference-frame mismatch; add bar-relative obs and torso DOFs".
> - **Raw diff:** `git diff da8a7f9 -- move_garbage/envs/bin_push_homie.py`.
> - **Finding a change in the code:** every edited block carries a dated tag such as `# 2026-09-29 overhand:` or `# 2026-10-01 tilt3:`. Grep for the tag to find all the edits for one theme, e.g. `grep -n "2026-10-01 tilt2" move_garbage/envs/bin_push_homie.py`.

## TL;DR

- **New task.** It went from a palm-down "grasp → push" to a 5-stage task: **APPROACH → REACH → GRASP (overhand claw) → TILT (~15° back) → PUSH**. That is the human motion of tipping the bin onto its wheels, then rolling it.
- **Size.** Almost all the code is in [`move_garbage/envs/bin_push_homie.py`](move_garbage/envs/bin_push_homie.py) (+3400+ / −68):
  - 95+ new methods and 48 modified, none removed;
  - 18 new reward terms;
  - 140+ new config keys.
- **Backward compatible by construction.** Every new behaviour sits behind a config key whose default reproduces the old behaviour. `bin_push_homie_lstm` and the older checkpoints still compose and run.
- **One standalone YAML config + one python command.** All 16 intermediate overhand stage configs have been flattened into a single standalone config [`move_garbage/config/exp/wbmanip/bin_push_homie_overhand_approach_dr1.yaml`](move_garbage/config/exp/wbmanip/bin_push_homie_overhand_approach_dr1.yaml) (inheriting directly from `/exp/wbmanip/bin_push_homie_lstm`). Everything the old `run_*.sh` / `train_loco_manip.sh` wrappers used to pass now lives in that YAML.
- **Progress summary (through 2026-10-07):**
  1. **REACH → GRASP:** solved (`reach_grasp2` iter 1700).
  2. **GRASP → TILT:** solved across `6–13 kg` (`grasp_tilt5` iter 5100: 96% success at 20 rad/s).
  3. **TILT → PUSH (5.0 m straight walk):** solved (`tilt_push6` iter 8400+: 99.9% success across `U[6, 13] kg` at `5.00 m`).
  4. **5-stage manipulation (`REACH → GRASP → TILT → PUSH` at 3.50 m):** solved (`overhand_full2` iter 2900: **97.2% end-to-end success** across `8..15 kg` total bin mass).
  5. **Polar `STAGE_APPROACH` Spawn DR (`d ∈ [0.02, 1.10] m`, `φ ∈ [-45°, +45°]`, `Δψ_fov ∈ [-20°, +20°]`) + Resting-Arm Approach:** `approach_dr1` iter 3300 solved the full `1.10 m / ±45° / ±20°` polar cone (`86.5%` success), and `approach_dr2..dr4` added resting-arm delta-action reset + decay (`reset_delta_actions_to_resting: true`, `approach_arm_delta_decay: 0.50`) so arms stay naturally tucked (`100%` within `0.35 rad`, mean max dev `4.8°`) while walking up to the bin.
  6. **MuJoCo Sim2Sim verification (`sim2sim_mujoco/`):** standalone MuJoCo 3.x verifier for the 43-DoF G1 + split-handle wheeled bin (`full2_2900` and `approach_dr1`).

## Files

| File | Status | Lines | What |
|---|---|---|---|
| [`move_garbage/envs/bin_push_homie.py`](move_garbage/envs/bin_push_homie.py) | modified | +3400+ / −68 | everything in the sections below |
| [`gr00t/rl/simulator/isaacsim/isaacsim.py`](gr00t/rl/simulator/isaacsim/isaacsim.py) | modified | +21 | opt-in friction tracking on the object↔hand contact sensor (§9) |
| [`gr00t/rl/envs/base_task/staged_task_base.py`](gr00t/rl/envs/base_task/staged_task_base.py) | modified | +8 | opt-in `overtime_is_timeout` (§5) |
| [`move_garbage/scripts/probe_reach_grasp_feasibility.py`](move_garbage/scripts/probe_reach_grasp_feasibility.py) | modified | +59 | `probe_finger_frac` (fixed finger closure) and `probe_descend_to_bar` (IK descent from hover onto the bar) |
| [`move_garbage/config/exp/wbmanip/bin_push_homie_overhand_approach_dr1.yaml`](move_garbage/config/exp/wbmanip/bin_push_homie_overhand_approach_dr1.yaml) | new | 362 | standalone flattened experiment config (inherits `/exp/wbmanip/bin_push_homie_lstm`; toggle `approach_is_final: false` for full 5-stage training) |
| `move_garbage/data/objects/bin/…_dynamic_splithandle_wide.usd` | new | | bin with a real handle collider; used by every overhand exp (§2) |
| `move_garbage/data/objects/bin/…_dynamic_splithandle.usd` | new | | first, narrow-gap version; used only by handlefix 1–3 |
| [`move_garbage/scripts/make_split_handle_bin.py`](move_garbage/scripts/make_split_handle_bin.py) | new | | generates the split-handle USDs |
| [`move_garbage/scripts/probe_tilt_feasibility.py`](move_garbage/scripts/probe_tilt_feasibility.py) | new | | offline tilt probes A1–A8 (tilt-onset moment, topple angle, canned-grasp pulls) |
| [`grasp_metrics.py`](grasp_metrics.py) | new | | TensorBoard funnel, trend table and PASS/FAIL checks for the staged runs: `venv/bin/python grasp_metrics.py [run_dir]` |
| `sim2sim_mujoco/` | new | | standalone MuJoCo 3.x sim-to-sim harness (`g1_43dof_bin_scene.xml`, `sim2sim_obs_actions.py`, `rollout_sim2sim.py`) |
| `move_garbage/data/grasp_init/overhand_th10_a080.npz` | new, **gitignored** (`*.npz`) | 13 KB | canned grasped state for grasp-init (§6). **Needs `git add -f`.** |

## Changes by theme (chronological)

### 1. Finger pre-shape (09-24; runs `preshape35`, `preshape45d`)
- **Why:** a deadlock.
  - With open fingers, the hands rested flat on the lid and could not get within the 5 cm grasp gate.
  - But the fingers only closed once inside it.
  - A diagnostic rollout showed a best worst-hand distance of 6.4 cm, with the palms ~5 cm above the bar.
- **What:**
  - Distance-graded "hook" pre-shape (`_finger_preshape_poses`). Fingers stay open in APPROACH. In REACH the curl ramps from 0 to `finger_preshape_frac` as the grasp point closes from 25 cm to 5 cm (`finger_preshape_start_dist` / `_full_dist`).
  - Finger gates and rewards are measured against the pre-shape.
  - GRASP never asks for less curl than the pre-shape.
  - Looser close radius: `grasp_finger_close_bar_dist`.

### 2. Bin handle collider + quaternion fix (09-25; `handlefix`)
- **Why, collider:** the lid's convex-hull collider included the hinge rod. Lid and handle were therefore one solid slab that swallowed the bar (the bar reference sat ~1.5 cm inside it), so the hands could never get between the lid and the handle.
- **Why, quaternion:** `get_task_root_state` returns **wxyz**, but two places reordered it as xyzw. That gave the wrong bin up-vector.
- **What:**
  - `make_split_handle_bin.py` builds a USD with separate lid-panel and bracket hulls plus a capsule `handle_collider`.
  - The USD's grasp-target prims are moved onto the handle axis (`--move-target-to-handle`).
  - The quaternion is fixed in `_palm_down_alignments` and `_reward_penalty_bin_tilt`.
  - handlefix4 added the **wide-gap** variant: lid collider cut at y 0.300 and capsule r 0.0165 (`_BAR_RADIUS`), which leaves a ~2.4 cm slot for the 1.77 cm-thick distal finger. Regenerate it with roughly:
    `make_split_handle_bin.py --split-body --move-target-to-handle --lid-cut-y 0.300 --handle-radius 0.0165 --out <usd>`
    This needs Isaac Sim's pxr python, and the flags are reconstructed from the old launcher headers, so verify them.

### 3. Wrap score, lid contact, wheel-kick fix (09-25 → 09-27; `handlefix2`–`4`)
- **Why, grasp** (seen in video and in a wrap probe):
  - The index and middle tips went *over* the bar and landed on the lid.
  - The 5 cm grasp-point gate was satisfied by a fist above the bar.
  - The closure reward paid for closing in the air.
- **Why, wheel kick:**
  - The penalty's unfiltered force included the floor holding the bin up, so it fired on every PUSH step.
  - Moving to a single sensor over both wheels made PhysX reject the per-env filters ("expected 4096, found 2048"). The force matrix came back empty, so the penalty was silently 0.
- **What:**
  - `grasp_finger_wrap`: each fingertip is scored by its angle around the bar (0 on top → 1 underneath, linear in φ) times its proximity. A worst-finger blend means every finger has to wrap.
  - `penalty_lid_contact`: hand-link contact force on the lid side of the bar, above it.
  - The closure reward only pays near the bar (`grasp_finger_near_gate_floor`).
  - The finger rewards blend in the worst joint, so one finger jammed on the lid costs reward.
  - Wheel-kick penalty: **one contact sensor per wheel**, filtered to the robot's knee/ankle links.
  - First positive `bin_tilt_back` reward, plus PUSH observability logs.

### 4. Front hook + arm IK branch (09-27 → 09-28; `handlefix5`–`5e`)
- **Why:**
  - The 75° hook target needed ~30° of waist lean: upright, the left hand fell 9 cm short.
  - The left forearm twisted onto the wrong IK branch while walking (wrist roll 80–100° by 0.8 s).
  - HOMIE switches to its STAND model when |cmd| < 0.1. GRASP had trained the command to 0, so in PUSH the robot never walked.
  - Hand link origins are the PIP/IP joints, ~5.2 cm short of the real fingertip.
- **What:**
  - Palm alignment is measured against the configured target axis (`grasp_palm_align_from_target`).
  - Wrap scoring gained a thumb-over option and uses real fingertip points.
  - The grasp gate requires a wrap score (`grasp_gate_min_wrap`).
  - New arm terms: `torso_lean`, `left_arm_posture`, `penalty_left_wrist_roll`, `penalty_arm_branch_guard`, `penalty_left_arm_posture_l1`.
  - `push_forward_command`: rewards a forward HOMIE command above the deadband.
  - Posture and closure logs.
  - All the hook-specific terms are **off** in the overhand exps.

### 5. Overhand claw grasp + 5-stage TILT (09-29; `bin_push_homie_overhand.yaml`)
- **Why:** the goal became overhand grasp → ~15° back-tilt → push. A review of the hf5e config found six blockers:
  - targets and arm terms were tuned for the hook;
  - the wrap score rejected the overhand grip;
  - there was no tilt phase, yet forward force and command were paid on a flat bin, which can only topple it;
  - staged resets only replay stages the policy has already reached, so PUSH was never visited;
  - nothing terminated the episode when the bin toppled.
- **What, grasp:**
  - Overhand claw target, validated in sim 15/15 over standoff 7.6–8.2 cm × wrist tilt 5–15°.
  - The finger primitive now runs from the claw pre-shape to the closed claw (set in the exp yaml).
  - `grasp_wrap_mode: overhand` → `_overhand_finger_terms`: slot / below / front / proximity sub-scores per finger, using real tip points.
  - Gate: hand score > `grasp_gate_min_wrap` **and** every finger within `grasp_finger_max_bar_dist` (1.8 cm) of the bar surface.
- **What, stages:**
  - `enable_tilt_stage` gives 0 APPROACH, 1 REACH, 2 GRASP, 3 TILT, 4 PUSH. The class constants keep the 4-stage layout when it is off.
  - Rewards are declared **by stage name** (`@_effective_in_named_stages`), so they resolve correctly in both layouts.
  - TILT holds the grasp the same way PUSH does (`_holding_stage_mask`).
  - TILT→PUSH: back-tilt 10–22° held for 10 steps with the grip maintained.
  - PUSH complete: ≥ 1.0 m **and** tilt 5–25° **and** grip maintained.
  - The push-progress baseline is taken at TILT→PUSH.
- **What, rewards:**
  - `bin_tilt_progress` (potential-based).
  - `push_force_nomoment`: hand force along the handle → wheel-contact line, which rolls a tilted bin without adding tipping moment. It replaces `push_bin_force`.
  - Push progress and the forward command only pay inside the tilt band (`_tilt_band_weight`).
- **What, robustness:**
  - Topple termination: `bin_topple_*`, at −8° / +30°.
  - Grip check in TILT/PUSH: before this, the strict grasp gate ended the episode after 0.2 s of grip loss. `push_grip_loose_check` replaces it with a loose check (palm < 8 cm, closure ≥ threshold) and a 25-step grace.
  - `overtime_is_timeout`: a stage overtime bootstraps instead of counting as a −20 failure.
  - `reset_bin_on_reset`: previously a pushed or toppled bin carried over into the next episode.
  - Configurable spawn jitter: `spawn_y_range`, `spawn_yaw_range_deg`.
  - `bin_asset_path` config key; the `BIN_ASSET_PATH` env var still wins.

### 6. Grasp-state init, reach→grasp stage, palm above bar (09-30)
- **Why:**
  - TILT and PUSH were never visited, so their rewards never trained.
  - In `reach_grasp1` the distance gate turned out to be direction-blind: the left palm settled *under* the bar (dz −4 to −5 cm) at the same 5 cm.
- **What, grasp-init:**
  - `grasp_init_ratio` (default 0) makes that fraction of stage-0 resets start directly in TILT, already holding the bar with the canned claw state from the `grasp_init_path` npz.
  - The bin goes to its default pose and the robot pose is composed onto it. Joints and the delta accumulator are set consistently.
  - Negative controls (`grasp_init_debug_variant` = open / raised), an acceptance report, and a per-step debug CSV for videos.
- **What, staging:** `grasp_is_final` / `tilt_is_final` end the episode as a success at GRASP→TILT or at TILT→PUSH. That is what lets each stage train in isolation (the `reach_grasp` and `reach_grasp_tilt*` exps).
- **What, palm above the bar:**
  - `grasp_palm_min_dz` is added to the gate, plus `penalty_palm_below_bar`.
  - `grasp_bar_reward_target: grasp_target` measures distance to the actual grasp-point target instead of the bar centre.
  - `grasp_ovh_prox_mode: segment_prox`: at the validated claw pose the tips sit ~2 cm off the bar, because contact is mid-phalanx, so tip-based proximity scored a correct grasp at only 0.15.
- **What, testing:** `debug_zero_policy_actions` for offline acceptance tests.

### 7. Success metrics + closure gate (10-01)
- **Why:**
  - The staged runs needed per-episode success rates.
  - `reach_grasp2` entered TILT with the right hand only 35–39 % closed, because the gate was geometry-only, and then loosened its grip.
- **What:**
  - `set_to_stage` / `_record_episode_outcomes`: episode success, grasp reliability and tilt success over a ring buffer of `success_metrics_window` episodes.
  - `grasp_gate_min_closure`: both hands' closure must be ≥ the value.
  - Closure logs for the TILT stage only.

### 8. tilt2: tipping-moment shaping (10-01; `…_tilt2.yaml`)
- **Why:**
  - `tilt1` grasped well (86 % reached TILT, grip kept 82–92 %), but the bin never rotated (max tilt ≤ 0.09°).
  - The tilt rewards were outcome-only. A wheeled bin needs a finite moment about its wheel–ground contact line before it moves at all (measured onset ~30 Nm for the full-mass bin, probe A1), so below that moment there was no gradient.
- **What:**
  - `bin_tilt_moment` = clamp(M / 30 Nm, 0, 1) × grip, where M is the hand-force moment about the wheel contact line.
  - It fades to 0 between 12° and 16°. This is a topple guard: a steep push carries the bin past its ~23° balance point and onto the robot.
  - `bin_tilt_back_std_deg` raised to 10, so an upright bin still sees a gradient.
  - `penalty_body_bin_contact`: no chest bracing.
  - TILT logs for moment, hand force, body contact and closure.

### 9. tilt3: friction-corrected moment + bin-mass curriculum (10-01; `…_tilt3.yaml`)
- **Why:** tilt2 plateaued.
  - The hold was firm (grip kept 100 %), but the moment rose to ~6.5 Nm by iter 2100 and stalled. The bin never left its rest pose.
  - **Root cause:** Isaac Lab's contact `force_matrix_w` reports the **normal force only**. About 27 of the ~63 N claw grip goes through friction, so the moment the reward paid for was wrong.
  - Evidence: the normal-only moment showed sustained values above the bin's static tip limit while the bin stayed put, which is physically impossible. With friction included, there were 0 such violations.
  - Also, 15 kg needs ~23–30 Nm, far beyond anything the policy explored.
- **What:**
  - `bin_tilt_moment_friction`, together with `task.target_obj_contact_track_friction` and `target_obj_contact_max_data_per_prim: 64` in `isaacsim.py`: the moment is now normal + friction.
  - `bin_mass_curriculum`:
    - the bin **body** mass is sampled from U[max(6, level − 2), level] kg, with the inertia scaled to match (wheels unchanged);
    - the level rises from 6 to 15 kg in 1 kg steps once tilt success ≥ 30 % over ≥ 3000 episodes at the current level;
    - the level is saved in the checkpoint env state, so a resume keeps it.
  - Diagnostics:
    - normal-only moment log (`tilt_moment_normal_Nm_tilt`);
    - wrist contact;
    - friction columns in the debug CSV;
    - bin mass read back from PhysX.

### 10. TILT-only: grasp_tilt3 → m10 → tilt4 → tilt5 (10-01 → 10-02; `…_grasp_tilt3*.yaml`, `…_grasp_tilt4.yaml`, `…_grasp_tilt5.yaml`)
- **Why TILT-only:** `tilt3` (and `tilt1` before it) lost its grasp (97 % → 23–37 %) once it learned to hold in TILT: early closing in GRASP broke the right claw.
- **grasp_tilt3** (from rg2 1700):
  - every episode starts holding the bar (`grasp_init_ratio` 1.0, `staged_reset_ratios` [1, 0, 0, 0, 0]);
  - `bin_tilt_moment_min_ratio` −1: the moment reward is signed, clamp(M / 30, −1, 1) × grip × topple-fade. tilt3 had held 100 % grip at M = −1.8 Nm (pushing the bin toward tipping forward) at no cost.
  - Result: hold learned (grip 100 %), moment recovered to +1.8…+2.7 Nm, too slowly.
- **m10** (from grasp_tilt3 2300): `bin_tilt_moment` 3 → 10.
  - Result: moment 4.5 → 8.6 Nm, grip 100 %, but tilt max fell to 0.09° and success stayed 0; mass level stuck at 6 kg.
  - **Edge-hold diagnosis** (play of 4600): for 4–6 kg the hand moment equals the bin's static tipping moment (M ≈ 1.49 × mass − 0.15 Nm; COM ~15 cm ahead of the wheel line). The front feet unload and the bin never rotates. The moment reward is saturated at that physical cap, and a tipped bin needs *less* moment (balance point 23°), so tipping paid nothing extra.
  - Same checkpoint, mass sweep: 2–3 kg 83 % success (tips to 12–14°), 3–4 kg 27 %, 4–6 kg 0 %; topple 0.
- **tilt4** (from m10 4600), one agreed package:
  - `bin_mass_level_reset_kg` 3 (new key, default None) + `bin_mass_start_kg` 2: `load_env_state_dict` puts the restored level back to 3 kg **once**. The checkpoint marker `bin_mass_level_reset_applied_kg` (written with the level in `log_dict`) makes later resumes of the same run keep their level.
  - `bin_tilt_back` 6 → 20. The moment pays at most 0.2/step; tipping into the band gains 0.01776 × scale per step, so tipping beats holding at the edge for every mass iff the scale is above ~11.3. At 6 the crossover was ~13–14 kg.
  - `bin_mass_end_kg` 15 → 25.
  - Result: 3 → 8 kg in ~100 it (4601–4705), then collapsed at level 8 (6–8 kg): 0 % success from ~4817 on. At 4900 even 3–4 kg bins no longer tip within 3 s (4700: 100 %).
- **Diagnosis** (per-step reward CSV, plays of 4700 / 4900 with a 10 s window; `tilt_window_reward_analysis.md`):
  - Heavier bins wait longer at the tipping edge: 0.4 s at 3–4 kg, 2.5 s at 6–7 kg, 5.5 s at 8–9 kg. The rotation itself takes ~0.5 s at every mass.
  - So level-8 tips took 3.2–3.7 s, longer than the 3 s TILT window (a timeout: no success, no promotion credit).
  - Inside 3 s a tip and an edge-hold earned the same return (~147 vs ~149), and the success bonus was +1.
- **tilt5** (from tilt4 4700, before the collapse), agreed package:
  - `max_stage_time[3]` 150 → 200: TILT window 3 → 4 s. Not 5 s: at the level-8 masses 4700 succeeds 16 / 55 / 56 % within 3 / 4 / 5 s but terminates on grip loss 0 / 4 / 38 %, because the 7–8 kg grip losses happen at 4–5 s (see the bootstrapping gotcha).
  - `complete` 50 → 1000: +1 → +20 once, on the success step.
  - `bin_mass_level_reset_kg` 6: one-time level reset 8 → 6 kg (bins U[4,6]). It triggers because 6 ≠ the 4700 marker 3.0.
- **Per-step reward CSV** (new, default off; used for the diagnosis): `grasp_init_debug_csv_rewards` (path) and `grasp_init_debug_csv_rewards_envs` (default: `grasp_init_debug_csv_envs`, else 2). `_compute_reward` snapshots `episode_sums` around the base call, and `_rew_csv_row` writes each term's exact per-step value plus stage, time, tilt, moment, hand force, grip and bin mass.
- **Body-link CSV columns** (new, default off; tilt5 video overlay): `grasp_init_debug_csv_body_links` appends `bin_level_kg` (current level), `bin_ep_level_kg` (level the episode was sampled at), `bin_total_kg` (all bin bodies = body + 2.0 kg) and `bc_<link>` (per-link |F| of every link summed by `_body_bin_contact_force`) to the `grasp_init_debug_csv` rows. It is how the self-contact gotcha below was found.

### 11. tilt5 outcome → tilt_push1: PUSH on (10-02 → 10-03; `…_tilt_push1.yaml`)
- **tilt5 results** (deterministic sweeps, 64 envs, seed 11, body U[6,13]):
  - 4950: 75 % at 20 rad/s vs 2 rad/s as trained. The overspeed curriculum had decayed to its floor (gotcha).
  - **5100 at 20 rad/s: 96 %**; 6–9 kg 98–99 %, 9–11 kg 95–98 %, 11–12 kg 89 %, 12–13 kg 91 %. 18 / 19 failures at 11–13 kg are topple > 30°; t_succ ~2.4 s.
  - 5300 as trained: 20 %. It is 99.5 % at 20–25 kg, a heavy-bin specialist from the U[L − 2, L] window (gotcha).
  - Training reached the 25 kg cap at ~5650 (~98 % success there).
- **tilt6** (4950 + `min_level` 1.0 + level reset 11 kg, cap 13) was written but never launched; superseded by the 5100 sweep.
- **tilt_push1** (from tilt5 5100), one agreed package:
  - `tilt_is_final` false: the TILT→PUSH gate (10–22° held 10 steps + grip) no longer ends the episode. Success = PUSH complete: 1.0 m along +x from the bin pose at the handover, tilt 5–25°, grip held.
  - `max_stage_time[4]` 50 → 400: PUSH window 1 → 8 s.
  - Body mass fixed at U[6, 13]: `bin_mass_level_reset_kg` 13 (19 → 13 once), `bin_mass_end_kg` 13, `bin_mass_spread_kg` 7.
  - `termination_curriculum.min_level` 1.0: overspeed limit pinned at 20 rad/s.
  - PUSH rewards are the existing overhand ones: `push_bin_progress` 16, `step_forward_while_pushing` 4, `push_forward_command` 2, `maintain_grasp_while_pushing` 3, `push_force_nomoment` 0.1, `bin_tilt_back` 20 (TILT + PUSH).
- **Zero-shot baseline** (5100 with this config, deterministic): 96 % reach PUSH, then **100 % of PUSH entries topple within ~0.4 s** (tilt ~23° = past the balance point; TILT-only training never had to brake). Expect a ~100 % fail spike at the start; the PASS/FAIL list is in the yaml header and `grasp_metrics.py` (`NOTE_TILT_PUSH1`).
- **New logging keys** (default off; the yaml switches them on):
  - `push_stage_logs`: `push_reached_frac` (of episodes that reached TILT) and `push_success_frac` (of PUSH-reached), from `_record_episode_outcomes`. Also the per-step PUSH logs from `_update_overhand_tilt_logs`, computed only when some env is in PUSH: `push_frac_envs`, `push_progress_m_push`, `push_progress_max_m_push`, `tilt_back_deg_push`, `tilt_in_band_frac_push` (5–25°), `homie_vx_cmd_push`, `walk_cmd_frac_push` (vx > 0.1), `pelvis_fwd_vel_push`, `bin_fwd_vel_push`, `body_bin_contact_frac_push`.
  - `success_by_mass_edges_kg` (list): `episode_success_frac_m{lo}_{hi}` per body-mass bin, plus `push_reached_frac_m{lo}_{hi}` with `push_stage_logs`.
  - The episode-outcome ring buffer `_succ_buf` grew from 6 to 8 columns (+reached_push, +episode body mass). It is not in the env state dict, so old checkpoints still load.

### 12. push1 diagnosis → grasp_tilt7 & tilt_push2 (10-03; `…_grasp_tilt7.yaml`, `…_tilt_push2.yaml`)
- **Why `tilt_push1` failed (gate avoidance):**
  1. **Mid-swing handover:** `tilt5` `model_step_005100` enters the `[10°, 22°]` band with mean $|\dot{\theta}| \approx 56^\circ/\text{s}$ (`+31°/s` at handover) and topples past `30°` within `~0.35 s` in 100% of zero-shot PUSH entries.
  2. **4× self-contact penalty jump on PUSH entry:** `body_bin_contact_push_scale` was `4.0` vs `1.0` in TILT, penalizing ~110 N of unfiltered left-arm ↔ torso self-contact at `-1.76/s` in PUSH vs `-0.44/s` in TILT. Combined with 100% topple termination (`-20`, no bootstrap) vs free TILT timeout bootstrap, PPO learned within ~150 iterations to stall at `8–9°` in TILT (`push_reached_frac` `90% -> 23%` by iter 5292, `0%` by 5330).
  3. **Fake push progress from backward rotation:** `bin_root` is at the bottom of the bin (`z = 0`) while the handle bar is at `z ≈ 0.83 m`. Tipping backward from `11°` to `28°` around the handle swings `bin_root` `+0.158 m` (`+0.5 m/s`) forward along `+x` while the handle moves backward (`-0.052 m`).
- **`grasp_tilt7` (TILT-only steady-hold gate alternative):** adds `tilt_advance_max_rate_dps` + curriculum (`40 -> 10 °/s`) on `|dtheta/dt|` at `_update_grasp_validity_counters` and `_stage_3_to_4_advance_condition`, with handover rate logs (`tilt_gate_rate_dps`, `tilt_gate_rate_p90_dps`, `tilt_rate_abs_dps_band_tilt`, `tilt_steady_frac_band_tilt`).
- **`tilt_push2` (full TILT → brake/hold → PUSH + walk sequence, Option B):**
  1. **Rotation-invariant push progress (`push_progress_use_bar_min: true`):** `_bin_push_progress` and `_reward_push_bin_progress` take `min(bin_root, bar_midpoint)` along `+x` for both displacement and velocity (`_bar_mid_xy`, `_bar_fwd_vel`, `_bin_bar_push_progress`, `_initial_bar_pos_xy`), preventing backward tipping from earning fake progress or short-distance completions.
  2. **In-band tilt braking (`bin_tilt_steady: 12.0`, `bin_tilt_moment_off_deg: 14.0`, `bin_tilt_progress_clip_deg: 14.0`):** fades out `bin_tilt_moment` over `10 -> 14°`, clips positive `bin_tilt_progress` above `14°` (preserving the negative penalty above `15°`), and adds `_reward_bin_tilt_steady` ($w_{\text{band}}(\theta) \cdot \text{grip} \cdot \exp(-(\dot{\theta}/15)^2)$ in TILT + PUSH). `tilt_advance_max_rate_dps` is left `None` so all 96% of `5100`'s grasps enter PUSH from iter 1.
  3. **Smooth TILT → PUSH walk-command ramp (`tilt_walk_ramp_in_band: true`, `tilt_walk_ramp_scale: 0.5`, `tilt_walk_ramp_rate_std_dps: 25.0`):** once inside `[10°, 22°]` with settling tilt rate, `penalty_not_standing_still` relaxes on `vx > 0` and `push_forward_command` + `step_forward_while_pushing` ramp in at `0.5x` inside TILT.
  4. **Eliminate gate-avoidance incentive (`body_bin_contact_push_scale: 1.0`, `stage_reward_scale: [0.1, 0.25, 0.5, 1.0, 2.5]`):** removes the 4× self-contact penalty jump on PUSH entry and raises PUSH stage reward from `1.5` to `2.5`.
  5. **Push distance curriculum (`push_distance_curriculum: true`, `0.10 -> 1.00 m` in `0.05 m` steps at `>= 40%` PUSH success over `>= 1500` PUSH episodes):** makes the `+20` completion bonus achievable early while requiring genuine forward translation of both the bin base and the handle bar.

### 13. push2 outcome → tilt_push3: steady-rate gate + `+20` stage-advance bonus + no TILT-stall bootstrap (10-03; `…_tilt_push3.yaml`)
- **Why `tilt_push2` still collapsed into gate avoidance (`PUSH reached` `96.4% -> 50.9% -> 1.4% -> 0.4%` by iter 5250):**
  1. **`push_progress_use_bar_min: true` worked:** `rew_push_bin_progress` was `0.000` during backward tipping (`bin_root` `+0.11 m/s`, `bar` `-0.40 m/s`).
  2. **Missing `+20` payout at the `TILT -> PUSH` gate:** In `grasp_tilt5`, passing the `TILT` gate paid `+20` (`complete: 1000.0`). In `tilt_push1/2`, `complete` moved to the end of `PUSH`, leaving `+0` at the `TILT -> PUSH` transition while entering `PUSH` mid-fling (`+42..+49 °/s` with `tilt_advance_max_rate_dps: None`) yanked the bar out of the hands (`grip maintained PUSH = 42%`, `-20` termination).
  3. **`TILT` overtime bootstrap (`overtime_is_timeout: true`):** Parking the bin flat at `0.08°` for all 200 `TILT` steps had `0%` termination and received the full timeout value bootstrap $+\gamma V(s) \approx +100..+140$.
- **`tilt_push3` (`tilt_is_final: false`, live `TILT -> PUSH` transition):**
  1. **`+20` stage-advance milestone bonus (`_reward_tilt_advance`, `rewards.reward_scales.tilt_advance: 1000.0`):** pays `+20` once per episode on the exact step `STAGE_TILT` satisfies `_stage_3_to_4_advance_condition`, before advancing live into `STAGE_PUSH` (with `+20` `complete` still paid at the end of `PUSH`).
  2. **Steady-hold `TILT -> PUSH` rate gate + curriculum (`40 -> 12 °/s`):** requires `|dtheta/dt| <= threshold` for all 10 hold steps + gate step in `[10°, 22°]`. `_update_tilt_rate_curriculum` counts `current_max_stage_buf >= STAGE_PUSH` when `tilt_is_final: false` so the rate curriculum tightens `40 -> 12 °/s` during `TILT + PUSH` training.
  3. **Close `TILT` overtime bootstrap (`tilt_overtime_is_timeout: false`):** running out the 4.0 s `TILT` clock without advancing to `PUSH` clears `time_out_buf` for those envs (treated as failure `-20`, no value bootstrap), while `PUSH` overtime stays a value-bootstrapped timeout.
  4. **Wider braking Gaussian (`bin_tilt_steady_std_dps: 35.0`, was `15.0`):** gives strong dense damping gradient across `20–45 °/s`.

### 14. push3 outcome → tilt_push4: eliminating static edge-hold farming + survivable PUSH handover + pre-deadband walk ramp (10-03; `…_tilt_push4.yaml`)
- **Why `tilt_push3` still collapsed into static edge-holding (`PUSH reached` `44.5% -> 25.0% -> 2.3%` by iter 5310):**
  1. **200-step static edge-hold return exceeded early-`PUSH` failure return (`+58.2` vs `+41.5`):** Even with `tilt_overtime_is_timeout: false` (`-20.0` at step 200), holding the bin flat at `-0.1°` just below static tipping (`+13.6 Nm`) for 200 steps paid `+0.391/step` (`+78.2` dense − `20.0` overtime = **`+58.2` net return**), driven by static grasp-shaping posture rewards (`+0.251/step`), `bin_tilt_moment` (`+0.193/step` = `+38.6`), and `bin_tilt_back` (`+0.042/step` = `+8.4` at `0°`). Meanwhile, episodes advancing into `STAGE_PUSH` at step ~35 (`+9.1` TILT dense + `+20.0` `tilt_advance`) died ~24 steps (`0.48 s`) later in `STAGE_PUSH` (`+32.4` PUSH dense − `20.0` `grip_lost`/`bin_topple` = **`+41.5` net return**).
  2. **Why `STAGE_PUSH` failed in `~0.48 s`:**
     - **High-angle / high-rate handover (`[10°, 22°]` at `<= 40 °/s`):** Entering `PUSH` at `16–21°` while rotating at `+25..+38 °/s` crossed the `~23°` unstable balance point within `0.15 s` (`tilt_back_deg_push = 20.4°`).
     - **Tight `0.08 m` / `0.60` grip termination during dynamic braking:** Arresting angular momentum stretched palm-to-bar distance to `0.082–0.095 m`, triggering instant `-20` `grip_lost` (`grip_maintained_frac_push = 51–55%`).
     - **Dead-zone in `_reward_push_forward_command` on `vx in [0.0, 0.10]`:** Because `vx` is a delta-action accumulator starting at `0.0` on `PUSH` entry and both `_reward_push_forward_command` (`((vx - 0.10) / 0.20).clamp(0, 1)`) and HOMIE's locomotion controller have a `0.10 m/s` deadband, small positive delta-action steps saw **zero gradient from every reward term**, trapping `homie_vx_cmd_push` at `+0.03..+0.05 m/s` (`0%` walking).
- **`tilt_push4` (5 structural fixes + coupled 3-axis curricula):**
  1. **Make static edge-holding strictly negative (`-5.5` net return vs `+108.5` for `TILT -> PUSH`):**
     - `grasp_shape_in_tilt_push_scale: 0.25` (`_grasp_shape_stage_scale`): scales the 9 grasp-acquisition posture terms to `0.25x` in `TILT` and `PUSH` while keeping `maintain_grasp_while_pushing` at `1.0x` (`10.0` in `TILT`, `17.5` in `PUSH`).
     - `bin_tilt_back_zero_at_upright: true`: subtracts the `0°` Gaussian baseline in `STAGE_TILT` so parking flat (`<= 0°`) pays `0.000` instead of `+0.042/step`.
     - `bin_tilt_moment_stall_decay_steps: 40`, `bin_tilt_moment_stall_span_steps: 35`, `bin_tilt_moment_stall_min_deg: 2.0`: linearly decays positive `bin_tilt_moment` to `0` over `0.8 -> 1.5 s` in `STAGE_TILT` if `tilt_back_max_deg < 2.0°` (keeping full initiation reward and negative push-forward penalty).
     - `stage_max_Seconds: 2.5` for `STAGE_TILT` (`125` steps, down from `4.0 s` / `200` steps; successful tips take `0.7–1.2 s`).
     - `rewards.reward_scales.tilt_advance: 4000.0` (`+80.0` milestone bonus) and `stage_reward_scale: [0.1, 0.25, 0.5, 0.3, 3.5]`.
  2. **Safer `TILT -> PUSH` handover (`[10°, 17°]`, `25 -> 12 °/s`, `15` steps = `0.30 s`):**
     - Tightens `tilt_advance_max_deg: 17.0` (giving `>= 6°` margin below the `23°` balance point and `13°` below `30°` topple), starts `tilt_rate_curriculum_start_dps: 25.0`, and requires `tilt_hold_steps_for_advance: 15`.
  3. **Dynamic-braking grip tolerance in `STAGE_PUSH` (`0.10 m` / `0.50` closure) + bar-settle re-anchor:**
     - `push_stage_grip_max_bar_dist: 0.10` and `push_stage_grip_min_closure: 0.50` in `_grip_maintained_mask` and `_reward_maintain_grasp_while_pushing` (while `STAGE_TILT` keeps `0.08 m` / `0.60`).
     - `reanchor_bar_on_push_settle: true` (`push_settle_reanchor_steps: 15`): re-anchors `_initial_bar_pos_xy` if the bin settles deeper inside `[10°, 17°]` during the first `0.3 s` of `STAGE_PUSH`, eliminating the `-4..6 cm` handle-bar setback debt before forward walking starts.
  4. **Continuous pre-deadband walk-command ramp (`push_cmd_pre_deadband_weight: 0.25`):**
     - Gives `_reward_push_forward_command` a positive linear slope across `vx in [0.0, 0.10]` (`0 -> 0.25`) followed by `0.25 -> 1.0` over `[0.10, 0.30]`, with `push_forward_command: 4.0`, `step_forward_while_pushing: 6.0`, and `push_bin_progress: 35.0`.
  5. **Coupled 3-axis curricula (`kg` mass `11 -> 13 kg`, tilt rate `25 -> 12 °/s`, push distance `0.10 -> 1.00 m`):**
     - Resets `bin_mass_level_reset_kg: 11.0` with `bin_mass_spread_kg: 7.0` (`U[6, 11] kg` initially -> `U[6, 13] kg` once `PUSH` completion reaches `>= 50%`).

### 15. push4 30-min diagnosis → tilt_push5: preserving 5100's grasp equilibrium + 10° moment stall gate + active moment braking in PUSH (10-03; `…_tilt_push5.yaml`)
- **What `tilt_push4` (steps `5101 -> 5253`) showed:**
  1. **Immediate improvements at iter `5101`:** With `[10°, 17°]` / `25 °/s` handover (`12.5°`, `+16 °/s`), `0.10 m / 0.50` PUSH grip tolerance, and `push_cmd_pre_deadband_weight: 0.25`, **`grip maintained PUSH` started at `97%`**, `walk cmd frac` jumped from `0%` (`tp3`) to **`60%`**, and `pelvis_fwd_vel_push` turned positive (`+0.04..+0.05 m/s` vs `-0.03 m/s` in `tp3`).
  2. **Why `PUSH reached` slipped after iter `5175` (`43.1% -> 12.5%` at `5253`) and `grip maintained PUSH` fell (`97% -> 20%`):**
     - **Scaling `grasp_shape_in_tilt_push_scale: 0.15` broke `5100`'s grasp force equilibrium:** Cutting `grasp_bar_distance` and `grasp_finger_wrap` by 85% while keeping `bin_tilt_moment: 10.0` at 100% caused the policy to yank its arms into its torso (`body contact 28 N -> 396 N`), collapsing the right palm downward (`R palm dz +3.8 -> +2.1 cm`, `R wrapped 96% -> 66%`), creating `-9.9 Nm` of downward friction torque that opposed tipping, and causing the unwrapped hands to lose grip in `STAGE_PUSH` (`97% -> 20%`).
     - **The `< 10°` moment-farming loophole:** Because `bin_tilt_moment_stall_min_deg` was `2.0°` and `max_stage_time` wasn't overridden (`200` steps), holding `+1.25..+2.1°` bypassed stall decay.
     - **Wound-up backward arm pull in `STAGE_PUSH` (`bar_fwd_vel = -0.24 m/s`, `tilt = 20.7°`):** In delta-action space, the arm joint targets remained wound up at `+130 N` (`+12 Nm`) on entering `STAGE_PUSH` because `_reward_bin_tilt_moment` went to `0.0` above `12°` and in `STAGE_PUSH`.
- **`tilt_push5` fixes:**
  1. **Restore `grasp_shape_in_tilt_push_scale: 1.0`:** preserves `5100`'s grasp force equilibrium (`97%` initial `STAGE_PUSH` grip, `96%` wrap, low body self-contact).
  2. **Shorten `STAGE_TILT` (`max_stage_time: [60, 100, 150, 125, 400]` = `2.5 s`) + `bin_tilt_moment_stall_min_deg: 10.0` (`decay: 30`, `span: 25`) + `bin_tilt_back_min_deg_tilt: 4.0` + `tilt_advance: 2500.0` (`+50`):** decays positive moment to `0` by `1.1 s` anywhere below the `10°` gate band and zeroes `bin_tilt_back` at `<= 4°` (`125`-step stall nets `+14.0` vs `>= +62.6` for entering `PUSH`).
  3. **Direct 1st-order over-pull moment braking (`bin_tilt_moment_brake_in_push: true`, `brake_start_deg: 11.0`, `steady_hold_Nm: 7.0`):** active in `TILT` (`>= 11°`) and `PUSH`; penalizes excess backward pull $M > M_{\text{allow}}(\theta, \dot{\theta})$ once tilted and rotating backward ($\dot{\theta} > 2^\circ/\text{s}$) or overshooting ($\theta > 14^\circ$), directly teaching the policy to unwind the `130 N` backward yank on entering the band.

### 16. tilt_push5 success → tilt_push6: STAGE_PUSH anti-stall principle, non-positive velocity penalty, and 5.0 m straight-walk curriculum (10-04; `…_tilt_push6.yaml`)
- **What `tilt_push5` (steps `5101 -> 6133+`) achieved:**
  1. **Completely solved `TILT -> PUSH` handover & balance:** `PUSH reached (of TILT)` climbed to `99.8–100.0%`, `BOTH wrapped = 97.8%`, `grip maintained PUSH = 98.0%`, and `rew termination (=-fail) = -0.5%` (`99.5%` survival, `0.0%` topple).
  2. **All 3 curricula ratcheted:**
     - `gate max rate`: `25.0 -> 15.8 -> 12.0 °/s` floor (`handover rate mean = -1.9 °/s`, `|p90| = 7.0 °/s` at `11.8°`).
     - `bin mass level`: `11.0 -> 12.0 -> 13.0 kg` (full `U[6, 13] kg` range).
     - `push target dist`: `0.10 m -> 0.15 m`, reaching `33.1%` overall episode success (`48.7%` on `6–9 kg`, `38.5%` on `9–11 kg`) and `walk cmd frac (vx > 0.1)` rising from `2% (@6000) -> 19% (@6133)`.
  3. **Why `model_step_005900.pt` pushed with its arms at `0.10–0.15 m` instead of walking immediately:**
     - `0.10–0.15 m` is physically reachable by extending the shoulders/elbows while keeping `vx_cmd < 0.10 m/s` (HOMIE stand mode).
     - Standing still in `STAGE_PUSH` (`vx_cmd = 0, pelvis_fwd_vel = 0`) while holding the bin at `~13.5°` still farmed `~+1.55/step` (`+77/s`) from static `bin_tilt_back` (`+0.39/step`), `bin_tilt_steady` (`+0.23/step`), static `pos_reward` (`50%` of `push_bin_progress`), and the `0.169` zero-velocity Gaussian tail of `step_forward_while_pushing`.
- **`tilt_push6` changes (`bin_push_homie_overhand_tilt_push6.yaml`):**
  1. **`STAGE_PUSH` Anti-Stall Velocity Gate (`push_anti_stall_enable: true`):**
     - `_push_motion_anti_stall_weight(self)`: in `STAGE_PUSH`, after `push_stall_grace_steps` (`25` steps = `0.50 s` settle window), ramps over `push_stall_ramp_steps` (`15` steps, fully active by `0.80 s`) a forward-velocity gate $w_{\text{move}} \in [0.15, 1.0]$ based on $v_{\text{eff}} = \min(v_{\text{pelvis},x}, v_{\text{bin},x}, v_{\text{bar},x})$ (`0.15` at $\le 0.02\text{ m/s}$, `1.0` at $\ge 0.12\text{ m/s}$).
     - Multiplies `_reward_bin_tilt_back` and `_reward_bin_tilt_steady` in `STAGE_PUSH` (while leaving `grasp_shape_in_tilt_push_scale: 1.0` untouched so the `98%` overhand claw wrap equilibrium is never weakened).
     - `push_progress_require_positive_vel: true` (`push_progress_vel_gate_mps: 0.05`) gates the static `pos_reward` term of `_reward_push_bin_progress` by positive forward velocity so a parked bin earns `0.000`.
  2. **Explicit Non-Positive Velocity & Sub-Deadband Command Penalties + Straight-Line Walk Guard:**
     - `step_forward_zero_at_rest: true` subtracts the zero-velocity Gaussian baseline (`0.169`) in `_reward_step_forward_while_pushing`.
     - `step_forward_stall_penalty: 1.0` (`step_forward_min_vel: 0.05`) explicitly penalizes $v_{\text{eff}} = \min(v_{\text{pelvis},x}, v_{\text{bar},x}) < 0.05\text{ m/s}$ in `STAGE_PUSH` after the `0.50 s` settle window (`-0.14/step` at rest, `-0.28/step` moving backward, vs `+0.28/step` walking forward at `+0.25 m/s` with scale `14.0`).
     - `push_cmd_sub_deadband_penalty: 0.5` penalizes `vx_cmd < 0.10 m/s` in `STAGE_PUSH` in `_reward_push_forward_command` (scale `6.0`).
     - `penalize_lateral_yaw_cmd_in_push: true` extends `_reward_penalty_not_standing_still` to `STAGE_PUSH` for `vy_cmd`, `vyaw_cmd`, and `vx_cmd < 0` (leaving `vx_cmd > 0` free) so the robot walks straight forward with the bin.
  3. **`5.0 m` Straight-Walk Curriculum & `18.0 s` `STAGE_PUSH` Budget:**
     - `episode_length_s: 22.0`, `max_stage_time: [60, 100, 150, 125, 900]` (`900` steps = `18.0 s` in `STAGE_PUSH`).
     - `push_distance_end_m: 5.00` with `push_distance_step_schedule: [[0.25, 0.05], [1.00, 0.15], [5.00, 0.25]]`.
     - `push_cmd_deadband: 0.50` and `push_cmd_target: 0.70` so `_reward_push_forward_command` drives `vx_cmd` across `ppo_trainer_homie_api.py`'s actual `0.50` `homie_stand_model -> homie_walk_model` switch threshold (`0.25 m/s` physical velocity).
- **`tilt_push6` overnight results (steps `6401 -> 8400+`, resumed from `tilt_push5` step `6400`):**
  - **Curriculum swept all the way from `0.20 m` to the full `5.00 m` cap by step `7000`** (`gate max rate = 12.0 °/s`, `bin mass level = 13.0 kg`).
  - **`99.9%` episode success at `5.00 m` across `U[6, 13] kg`** (`99.9%` at `6–9 kg`, `99.8%` at `9–11 kg`, `99.9%` at `11–13 kg`; `PUSH reached = 100.0%`, `PUSH success = 99.9%`, `0.1%` fail / `0.0%` topple).
  - **Sustained straight-line walking with tilted bin:** `HOMIE vx cmd PUSH = +1.00` (`100%` walk cmd), `pelvis / bin / bar fwd vel = +0.42 m/s`, `push progress max = 5.02 m` (`time to success = 12.7 s`), `tilt mean PUSH = +12.4°` (`100%` in `[5°, 25°]` band, `|rate| = 3.1 °/s`), `grip maintained PUSH = 98.3%`, `BOTH wrapped = 96.5%`.

### 17. Full 5-stage stitching: full1 & full2 (10-04 → 10-06)
- **`overhand_full1` (`grasp_init_ratio: 0.15` from scratch, iter `0 -> 1065`):**
  - Reached `4.25 m` in `STAGE_PUSH` on the `15%` grasp-init episodes, but `85%` `STAGE_REACH` episodes stalled at `0%` `GRASP` entry (`44/50 cm` palm distance, left wrist pinned at `+92.5°` joint limit) because:
    1. `grasp_tilt3.yaml` had inherited `grasp_tilt.yaml`'s `spawn_x_range: [-0.84, -0.74]` (`35 cm` standoff instead of `46 cm`, pinning the elbows/wrists).
    2. `15%` grasp-init episodes paid high `STAGE_TILT/PUSH` rewards immediately (`stage_reward_scale = 5.0`), dominating the policy gradient and starving `STAGE_REACH -> STAGE_GRASP`.
- **`overhand_full2` (resumed from `reach_grasp2` `model_step_001700.pt`, iter `1701 -> 2950`, `grasp_init_ratio: 0.0`):**
  1. Restored `spawn_x_range: [-0.95, -0.87]` (`46 cm` standoff).
  2. Added `_reward_grasp_advance` (`grasp_advance: 1500.0` = `+30` one-shot milestone bonus on `GRASP -> TILT` transition).
  3. Set `pre_push_overtime_is_timeout: false` (`-20` termination on stalling in `APPROACH`, `REACH`, `GRASP`, or `TILT`).
  4. Added `arm_branch_overhand_mode: true` (`penalty_arm_branch_guard: -4.0`) to keep shoulder yaw and wrist roll/pitch/yaw inside the overhand grasp basin during `APPROACH/REACH/GRASP`.
  5. **Result (`overhand_full2` `model_step_002900.pt`):** **Solved the full manipulation pipeline (`REACH → GRASP → TILT → PUSH` at `3.50 m` across `8..15 kg` total bin mass) with `97.2%` end-to-end episode success** (`99.1%` grasp-reliable, `99.8%` PUSH reached, `97.4%` PUSH success, `0.0%` topple).

### 18. Polar `STAGE_APPROACH` Spawn DR & Approach-to-Stop Curriculum: full3 & approach_dr1 (10-06 → 10-07)
- **Polar Spawn DR (`spawn_polar_dr: true`, `_sample_polar_spawn_states`):**
  - Randomizes bin yaw $\psi_{\text{bin}} \sim \mathcal{U}[-\pi, +\pi]$ and samples robot pelvis pose in polar coordinates relative to the nominal stop target $(x^*_{\text{stop}}, y^*_{\text{stop}}) = p_{\text{bar}} - d_{\text{stop}}\hat{u}_{\text{bin}}$ (`0.46 m` behind handle bar):
    - Distance $d \sim \mathcal{U}[0.02\text{ m}, d_{\max}]$, approach angle $\phi \sim \mathcal{U}[-\phi_{\max}, +\phi_{\max}]$, and initial robot heading $\psi_{\text{robot}} = \text{atan2}(p_{\text{bar},y} - p_{\text{robot},y}, p_{\text{bar},x} - p_{\text{robot},x}) + \Delta\psi_{\text{fov}}$ with $\Delta\psi_{\text{fov}} \sim \mathcal{U}[-\psi_{\text{fov},\max}, +\psi_{\text{fov},\max}]$ so the bin always starts inside the head D435 camera's horizontal FOV.
  - Expanding polar curriculum (`spawn_polar_curriculum: true`, `_update_polar_spawn_curriculum`): advances $(d_{\max}, \phi_{\max}, \psi_{\text{fov},\max})$ from `(0.25 m, 15°, 10°)` to `(1.10 m, 45°, 20°)` whenever `STAGE_REACH` entry $\ge 80\%$.
- **Squaring-up `STAGE_APPROACH -> STAGE_REACH` gate & rewards:**
  - `_stage_0_to_1_advance_condition` requires: 2D stop error `< 0.15 m`, heading error $|\Delta\psi_{\text{bin}}| < 12^\circ$ (`stop_transition_yaw_tolerance_deg: 12.0`), $|v_{xy}| < 0.20\text{ m/s}$, $|\omega_z| < 0.40\text{ rad/s}$ (`stage_advance_ang_vel: 0.40`), `|homie_cmd[:3]| <= 0.10`, and `approach_transition_hold_steps: 5` consecutive steps.
  - Added `_reward_approach_walk_command` (`6.0`, bridges HOMIE's `0.50` stand→walk switch when $d_{\text{stop}} > 0.15\text{ m}$ and rewards zero command inside `< 0.12 m`) and `_reward_approach_advance` (`1000.0` = `+20` milestone bonus).
  - Updated `_reward_stop_at_bin` (`stop_at_bin_include_yaw: true`, `stop_pos_coarse_std: 0.60`, `stop_yaw_std_deg: 15.0`) and `_reward_walk_to_bin` (`walk_taper_near: 0.06`, `walk_taper_far: 0.22`, `walk_to_bin_zero_at_rest: true`).
- **`approach_dr1` (`approach_is_final: true`, resumed from `full2` `2900`, iter `2901 -> 3350`):**
  - Swept all 6 polar curriculum levels from `0.25 m` to the full `1.10 m / ±45° / ±20°` cap in 190 iterations (`step 3090`) and reached **`86.5%` success** (`96.2%` reached `< 15 cm`, `93.8%` yaw `< 12°`, mean stop err `8.8 cm`, yaw err `5.0°`).

### 19. Resting-Arm Approach Reset & Decay (`approach_dr2..dr4`) + MuJoCo Sim2Sim Harness (10-07)
- **Diagnosing & fixing the raised left arm during `STAGE_APPROACH`:**
  - In `approach_dr1`, the robot walked with its left elbow raised outward (`max_arm_dev = 1.15 rad = 66°`, `arms_ready = 0%`).
  - Root cause: `default_dof_pos` in `g1_43dof.yaml` has a hardware pre-bent left arm (`elbow = -0.3 rad`, `shoulder_roll = +0.5 rad`) while `_resting_arm_dof_pos()` is `[0.25, 0.25, 0, 0.98, 0, 0, 0]`. Because `_reset_robot_states_callback` set `dof_pos` to `_resting_arm_dof_pos()` while `_reset_buffers_callback` zeroed `_last_delta_actions = 0`, frame 1 PD targets immediately snapped toward `default_dof_pos` (`1.31 rad` jump on the left elbow!). Furthermore, `stage_advance_require_resting_arms` was disabled in `STAGE_APPROACH` by `not self.enable_stop_near_bin`.
  - **Fix (`bin_push_homie.py` + `bin_push_homie_overhand_approach_dr1.yaml`):**
    1. `reset_delta_actions_to_resting: true`: initializes `_last_delta_actions` at episode reset to `(resting_arm_dof_pos - default_dof_pos) / delta_action_scale` so zero policy delta-actions hold the symmetric tucked resting pose from step 0.
    2. `approach_arm_delta_decay: 0.50`: in `STAGE_APPROACH`, blends `_last_delta_actions` toward the resting offset (`50%` per 50 Hz step) so Gaussian exploration noise cannot integrate into a random walk on the arms while walking.
    3. `stage_advance_require_resting_arms: true` (`stage_advance_arm_tolerance: 0.35`): enforces all 14 arm joints within `0.35 rad` (`20°`) of `_resting_arm_dof_pos` in `_stage_0_to_1_advance_condition`.
    4. Result verified in `approach_dr2/dr3` videos: **`approach_arms_ready_frac_s0 = 100.0%`** and **`approach_max_arm_dev_s0 = 0.084 rad (4.8°)`** (both arms stay smoothly tucked at the sides throughout the entire walk and stop).
- **Standalone MuJoCo 3.x Sim2Sim Harness (`sim2sim_mujoco/`):**
  - Built a standalone MuJoCo 3.x physics harness (`g1_43dof_bin_scene.xml`, `sim2sim_obs_actions.py`, `rollout_sim2sim.py`) that converts `g1_43dof_s3.usda` + `RecyclingContainerWithWheels_A01_Blue_01_dynamic_splithandle_wide.usd` into an exact MJCF scene (`500 Hz` RK4, exact PhysX joint armatures/PD gains, 5-stage state machine, 262-D privileged actor obs, and frozen HOMIE walk/stand policies) and verified `full2_2900` (`REACH → GRASP → TILT → PUSH`) and `approach_dr1` in MuJoCo.

## New reward terms (18)

| Term | Stages | Purpose | Scale in current config |
|---|---|---|---|
| `approach_walk_command` | APPROACH | forward HOMIE `vx_cmd >= 0.50` when far (`>15 cm`), zero command when near (`<12 cm`) | 6.0 |
| `approach_advance` | APPROACH | one-shot milestone bonus on `APPROACH -> REACH` transition | 1000.0 (+20) |
| `grasp_advance` | GRASP | one-shot milestone bonus on `GRASP -> TILT` transition | 1500.0 (+30) |
| `grasp_finger_wrap` | GRASP, TILT, PUSH | fingertips wrap around the bar | 8.0 |
| `penalty_lid_contact` | REACH → PUSH | fingers/palm pressing on the lid | −1.5 |
| `penalty_palm_below_bar` | REACH, GRASP | palm grasp point below the bar | −2.0 |
| `bin_tilt_back` | TILT, PUSH | Gaussian on back-tilt, peak `13.5°` (zero at `<= 4°` in TILT; anti-stall gated in PUSH) | 20.0 |
| `bin_tilt_progress` | TILT | potential-based tilt shaping (clipped `>= 13.5°`) | 4.0 |
| `bin_tilt_moment` | TILT, PUSH | hand moment about wheels (`<10°` stall decay + `>=11°` over-pull brake) | 10.0 |
| `bin_tilt_steady` | TILT, PUSH | in-band angular-velocity damping $\exp(-(\dot{\theta}/\sigma)^2)$ (anti-stall gated in PUSH) | 12.0 |
| `tilt_advance` | TILT | one-shot milestone bonus on `TILT -> PUSH` transition | 2500.0 (+50) |
| `penalty_body_bin_contact` | TILT, PUSH | no chest/body bracing | −0.5 |
| `push_force_nomoment` | PUSH | force along handle → wheel contact | 0.1 |
| `push_forward_command` | TILT (ramp), PUSH | forward HOMIE command (`0.50` deadband, sub-deadband penalty in PUSH) | 6.0 |
| `torso_lean` | REACH, GRASP | hook only: waist lean | 0 |
| `left_arm_posture` | REACH, GRASP | hook only: IK good branch | 0 |
| `penalty_left_wrist_roll` | REACH, GRASP | hook only | 0 |
| `penalty_arm_branch_guard` | APPROACH → GRASP | overhand shoulder-yaw / wrist-roll-pitch-yaw box guard (`arm_branch_overhand_mode: true`) | −4.0 |
| `penalty_left_arm_posture_l1` | REACH, GRASP | hook only: L1 pull to the IK pose | 0 |

Scale changes across the project (§10–§19): `bin_tilt_moment` signed (min ratio −1) since grasp_tilt3 and 3 → 10 since m10; `bin_tilt_back` 6 → 20 in tilt4; `complete` 50 → 1000 in tilt5; `bin_tilt_steady` 12.0 in tilt_push2; `tilt_advance` 1000.0 in tilt_push3 → 2500.0 in tilt_push5; `push_forward_command` 2.0 → 5.0 (push4/5) → 6.0 (push6); `step_forward_while_pushing` 3.0 → 8.0 (push5) → 14.0 (push6); `push_bin_progress` 25.0 → 30.0 (push5) → 35.0 (push6); `grasp_advance` 1500.0 and `penalty_arm_branch_guard` −4.0 in full2; `approach_walk_command` 6.0, `approach_advance` 1000.0, `walk_to_bin` 2.0 → 8.0, `stop_at_bin` 4.0 → 10.0, and `penalty_upper_body_non_finger_deviation_l1` −0.1 → −1.0 in full3 / approach_dr1..dr4.

<details>
<summary><b>Function map of <code>bin_push_homie.py</code> (95+ new, 48 modified, 0 removed)</b></summary>

**New methods, by theme**
- **Stage plumbing (APPROACH / GRASP / TILT / PUSH):** `_named_stage_mask`, `_holding_stage_mask`, `_push_motion_anti_stall_weight`, `_grasp_shape_stage_scale`, `_reward_approach_walk_command`, `_reward_approach_advance`, `_reward_grasp_advance`, `_stage_3_to_4_advance_condition`, `_stage_3_to_4_advance_callback`, `_stage_4_reward_condition`, `_stage_4_to_complete_condition`
- **Polar Spawn DR & curriculum:** `_sample_polar_spawn_states`, `_post_load_polar_spawn_curriculum`, `_update_polar_spawn_curriculum`, `_resting_arm_delta_actions`
- **Fingers / grip:** `_finger_preshape_poses`, `_finger_closure_fractions`, `_grip_maintained_mask`
- **Wrap + lid (handlefix3–5):** `_init_hand_link_indices`, `_bar_frame_deltas`, `_finger_wrap_scores`, `_wrap_is_thumb`, `_wrap_tip_offsets`, `_wrap_terms`, `_wrap_hand_scores`, `_reward_grasp_finger_wrap`, `_lid_contact_per_hand`, `_reward_penalty_lid_contact`
- **Overhand claw:** `_overhand_mode`, `_overhand_finger_terms`, `_overhand_link_idx`, `_overhand_tip_offsets`, `_update_overhand_tilt_logs`
- **Palm above bar / grasp distance:** `_palm_bar_deltas_bin`, `_palm_min_dz`, `_palm_above_bar_masks`, `_reward_penalty_palm_below_bar`, `_grasp_bar_reward_distances`, `_grasp_point_targets`
- **Arm posture (hook era + overhand box guard):** `_dof_idx`, `_reward_torso_lean`, `_reward_left_arm_posture`, `_reward_penalty_left_wrist_roll`, `_arm_branch_q`, `_reward_penalty_arm_branch_guard`, `_reward_penalty_left_arm_posture_l1`
- **Tilt:** `_bin_tilt_back_deg`, `_bin_tilt_rate_dps`, `_tilt_rate_gate_ok`, `_tilt_steady_weight`, `_tilt_potential`, `_reward_bin_tilt_back`, `_reward_bin_tilt_progress`, `_reward_bin_tilt_steady`, `_reward_tilt_advance`, `_tilt_band_weight`, `_bin_wheel_contact_point`, `_bin_tilt_moment_terms`, `_hand_friction_forces`, `_bin_hand_moment`, `_wrist_bin_contact_terms`, `_reward_bin_tilt_moment`, `_body_bin_contact_force`, `_reward_penalty_body_bin_contact`
- **Push:** `_bar_mid_xy`, `_bar_fwd_vel`, `_bin_bar_push_progress`, `_target_bin_push_distance_m`, `_push_force_nomoment_raw`, `_reward_push_force_nomoment`, `_reward_push_forward_command`
- **Episode metrics / curricula:** `set_to_stage`, `_record_episode_outcomes`, `_update_bin_mass_curriculum`, `_apply_bin_mass_curriculum`, `_post_load_tilt_rate_curriculum`, `_update_tilt_rate_curriculum`, `_post_load_push_distance_curriculum`, `_update_push_distance_curriculum`, `load_env_state_dict`
- **Grasp-state init / resets / debug:** `_grasp_init_state`, `_grasp_init_debug_variant`, `_bin_default_root_state`, `_write_bin_default_state`, `_reset_robot_states_callback`, `reset_envs_idx`, `_apply_grasp_init_task_state`, `_grasp_init_debug_report`, `_grasp_init_debug_csv_row`, `step` (`debug_zero_policy_actions`), `_compute_reward` + `_rew_csv_row` (per-step reward CSV, §10)

**Modified methods** (lines +added/−removed)
- **Core:**
  - `__init__` +28
  - `_init_buffers` +60
  - `_update_custom_logs` +145/−1
  - `_check_termination` +45/−3 (topple, grip loss in every holding stage, `tilt_overtime_is_timeout`, `pre_push_overtime_is_timeout`)
  - `scene_creation_callback` +24/−8 (per-wheel sensors)
  - `_reset_root_states` +9/−3 (spawn jitter + polar spawn DR)
  - `_reset_tasks_callback` +18
  - `_reset_object_states_callback` +18 (`reset_bin_on_reset`)
- **Grasp & Approach gates:**
  - `_stage_0_to_1_advance_condition` (yaw alignment, angular velocity, resting-arm check, hold counter)
  - `_grasp_pose_valid_mask` +28/−1 (wrap, finger distance, palm dz, min closure)
  - `_update_grasp_validity_counters` +45/−3
  - `_finger_close_ready` +18/−3
  - `_palm_down_alignments` +13/−2 (wxyz fix, target axis)
  - `_reach_transition_ready_mask` +5/−2
  - `_finger_control_masks` +2/−1
  - `_load_delta_actions_buffer` +20/−1 (`reset_delta_actions_to_resting`, `approach_arm_delta_decay`)
- **Stage conditions:** `_stage_2_to_3_advance_callback` +6, `_stage_3_reward_condition` +3, `_stage_3_to_complete_condition` +2
- **Rewards with real logic changes:**
  - `_reward_walk_to_bin` +20/−2
  - `_reward_stop_at_bin` +25/−2
  - `_reward_grasp_finger_dof_pos_l1` +25/−5
  - `_reward_maintain_grasp_while_pushing` +28/−3
  - `_reward_grasp_bar_progress` +11/−2
  - `_reward_penalty_bin_wheel_kick` +10/−5
  - `_reward_push_bin_progress` +15/−2
  - `_reward_penalty_grasp_flat_press` +5/−1
  - `_reward_reach_finger_dof_pos_l1` +5/−2
  - `_reward_grasp_bar_distance` +4/−2
  - `_reward_step_forward_while_pushing` +35/−1
  - `_reward_penalty_not_standing_still` +22/−1
  - `_reward_penalty_bin_tilt` +4/−3 (wxyz fix)
- **Rewards with only the stage decorator changed** (`+ "TILT"`, by name): `reach_left`, `reach_right`, `grasp_target_distance`, `grasp_target_anisotropic`, `grasp_vert_progress`, `grasp_fingertip_curl`, `grasp_contact`, `penalty_wrong_twist_contact`, `palm_down_alignment`, `palm_twist_alignment`, `palm_twist_progress`, `push_bin_force`, `penalty_asymmetric_grasp`

**New constants:** `STAGE_TILT` (−1 when TILT is disabled), `BIN_TILT_MOMENT_FORCE_SIGN`, `BIN_TILT_MOMENT_REF_NM`, `_BAR_RADIUS` (0.0165).
</details>

## Other modified files

- **`staged_task_base.py`:** `overtime_is_timeout` (default False). When it is on, a stage overtime is labelled a timeout: the value bootstraps and no failure penalty is applied. Envs that already failed in the same step keep their failure label.
- **`isaacsim.py`:**
  - `task.target_obj_contact_track_friction` (default False) sets `track_friction_forces` on the existing object↔hand `ContactSensorCfg`.
  - `task.target_obj_contact_max_data_per_prim` (default 4) sizes the contact buffer.
  - `refresh_sim_tensors` exposes `object_to_hand_friction_forces`, which has the same layout as the force matrix.
  - Only the existing sensor's flags changed; no new sensors were added.
- **`probe_reach_grasp_feasibility.py`:** `probe_finger_frac` pins the finger primitive at a fixed closure. `probe_descend_to_bar` / `probe_descend_waypoints` add an IK descent from the hover target onto the raw bar, used to sanity-check the pre-shape.

## Experiment configs and runs

All 16 historical overhand stage configs (`bin_push_homie_overhand.yaml` → `reach_grasp*` → `grasp_tilt*` → `tilt_push1..6` → `full1..3` → `approach_dr1`) have been flattened into a **single standalone config** (`move_garbage/config/exp/wbmanip/bin_push_homie_overhand_approach_dr1.yaml`) that inherits directly from `bin_push_homie_lstm` (with backups in `.patch_backups/wbmanip_yamls_20261007/`).

Launch pattern:
```bash
cd ~/Development/rl-wbc-g1-loco-manip
export WANDB_MODE=offline
venv/bin/python -u gr00t/rl/train_agent_trl.py \
  +exp=wbmanip/bin_push_homie_overhand_approach_dr1 \
  headless=True \
  experiment_name=bin_push_overhand_approach_dr4 \
  checkpoint="logs_rl/g1_bin_push_homie/bin_push_overhand_approach_dr1-20261006_181206/model_step_003300.pt" \
  "hydra.searchpath=[file://$PWD/move_garbage/config]"
```
*(To switch from `APPROACH`-only (`approach_is_final: true`) to the full 5-stage task (`APPROACH → REACH → GRASP → TILT → PUSH`), set `approach_is_final: false` in `bin_push_homie_overhand_approach_dr1.yaml` or pass `env.config.approach_is_final=False`.)*

Runs in `logs_rl/g1_bin_push_homie/`. Each run dir keeps its fully resolved `config.yaml`, so the old `run_*.sh` launchers aren't needed to reproduce anything.

| Run | Started from | Iters | Outcome / what it led to |
|---|---|---|---|
| `bin_push_fresh_2k` (09-23) | scratch | → 4400 | baseline; 800 and 4400 are the later restart points |
| `preshape35`, `preshape45d` (09-24) | fresh_2k 4400 | → 4800 / 8500 | pre-shape; still blocked by the lid → §2 |
| `handlefix` (09-25) | fresh_2k 4400 | → 5200 | split-handle bin + quaternion fix |
| `handlefix2`, `handlefix3` (09-25) | previous run | → 5700 / 6600 | wheel-kick, tilt-back, worst-joint; then wrap + lid → §3 |
| `handlefix4` (09-27) | fresh_2k 4400 | → 5600 | wide-gap bin, per-wheel sensors |
| `handlefix5` → `5e` (09-27/28) | fresh_2k 800, then hf5b 4300 | → 5100 | front hook + arm-branch fixes → §4; replaced by the overhand redesign |
| `overhand_reach_grasp1` (09-29) | scratch | → 2400 | palm settled *under* the bar → palm-above-bar gate (§6) |
| `overhand_reach_grasp2` (09-30) | scratch | → 1700 | **best grasp; iter 1700 is the start of tilt1–3, grasp_tilt3, and full2** |
| `overhand_reach_grasp_tilt1` (09-30) | rg2 1700 | → 2000 | grasps, but the bin never rotates → §8 |
| `overhand_reach_grasp_tilt2` (10-01) | rg2 1700 | → 2300 | moment plateau ~6.5 Nm, 0 % tilt → §9 |
| `overhand_reach_grasp_tilt3` (10-01) | rg2 1700 | → 2000 | friction-corrected moment + 6 → 15 kg mass curriculum; lost its grasp once it held in TILT → §10 |
| `overhand_grasp_tilt3` (10-01) | rg2 1700 | → 2300 | TILT-only + signed moment: hold learned, moment +1.8…+2.7 Nm (too slow) → m10 |
| `overhand_grasp_tilt3_m10` (10-01/02) | grasp_tilt3 2300 | → 4663 | moment 8.6 Nm = the 6 kg bin's static tip limit (edge-hold), 0 % success, level stuck at 6 kg → tilt4 |
| `overhand_grasp_tilt4` (10-02) | m10 4600 | → 5089 | 3 → 8 kg by ~4705, then collapsed at 8 kg: tips took longer than the 3 s window and a tip earned no more than an edge-hold; 0 % success from ~4817. **4700 = last good checkpoint** → tilt5 |
| `overhand_grasp_tilt5` (10-02) | tilt4 4700 | → ~5700 (stopped) | 6 → 13 kg by ~4950. The 4950 play at 11–13 kg gave 0 / 48 successes (4 s timeouts, edge-hold), and the 2 rad/s overspeed floor (gotcha) failed most fast tips at 6–10 kg. At 20 rad/s the deterministic sweep over U[6,13] gives 75 % for 4950 and **96 % for 5100** (89–91 % at 11–13 kg), then 25 kg cap by ~5650 with ~98 % training success: a heavy-bin specialist (5300: 0 % at 11–17 kg). Its rising "body contact" is left-arm ↔ torso self-contact (gotcha). **5100 = start of tilt_push1/2/3/4/5** |
| `overhand_grasp_tilt6` (10-02) | — | never launched | 4950 + 20 rad/s + level reset 11 kg; superseded by the 5100 sweep → tilt_push1 |
| `overhand_tilt_push1` (10-03) | tilt5 5100 | → ~5490 | **FAIL: gate avoidance.** PUSH reached 85–94 % with ~100 % topple until ~5180, then 59 % at 5201, 11 % at 5211 and 0 % from ~5330. From then on every episode was a 4 s TILT timeout with the bin held flat (tilt max 4–7°, fail 0 %). PUSH never improved first (grip held 26–41 %, tilt ~21–22°). Side effects: body contact > 50 N 15 → 78 %, R wrapped 100 → 71 %. Cause: see the two gotchas (handover mid-swing; stalling is free) |
| `overhand_tilt_push2` (10-03) | tilt5 5100 | → 5316 | **FAIL: gate avoidance.** `push_progress_use_bar_min: true` prevented fake backward-tip progress, but without the `+20` milestone bonus at the TILT→PUSH gate, unchecked mid-swing entry (`+42..+49 °/s`, 42 % grip) plus free TILT overtime value bootstrap drove PUSH reached `96.4 % → 50.9 % → 1.4 % → 0.4 %` by iter 5250 (bin parked flat at `0.08°`) → §13 (`tilt_push3`) |
| `overhand_tilt_push3` (10-03) | tilt5 5100 | → 5310 (stopped) | **FAIL: static edge-hold farming.** Even with `tilt_overtime_is_timeout: false` (`-20` at step 200) and `+20` `tilt_advance`, 200 steps of static edge-holding at `-0.1°` (`+13.6 Nm`) earned `+78.2` dense (`+58.2` net) vs `+41.5` net when advancing to `PUSH` and failing `0.48 s` later (`[10°,22°]` / `40 °/s` handover + `0.08 m` grip loss + `vx` deadband trap) → §14 (`tilt_push4`) |
| `overhand_tilt_push4` (10-03) | tilt5 5100 | → ~5260 | **PARTIAL → §15:** `97%` initial PUSH grip and `60%` walk cmd (`+0.05 m/s` pelvis vel), but `grasp_shape_in_tilt_push_scale: 0.15` broke `5100`'s grasp force equilibrium (`body contact 28 -> 396 N`, `R wrapped 96 -> 66%`, `PUSH grip 97 -> 20%`) and wound-up backward arm pull in `PUSH` needed active moment braking → §15 (`tilt_push5`) |
| `overhand_tilt_push5` (10-03) | tilt5 5100 | → 6400 | **SUCCESS on TILT→PUSH & short push:** `99.8–100%` PUSH reached, `98%` both wrapped, `98%` PUSH grip, `99.5%` survival (`0%` topple); all 3 curricula ratcheted (`12.0 °/s` rate floor, `13.0 kg` mass cap, `0.20 m` push dist). At `0.10–0.20 m` `vx_cmd = +0.27` stayed below the `0.50` `homie_walk_model` switch → §16 (`tilt_push6`) |
| `overhand_tilt_push6` (10-03/04) | tilt_push5 6400 | → 8400+ | **SOLVED 5.0 m straight-walk (`99.9%` success across `U[6,13] kg`):** swept `0.20 -> 5.00 m` by iter `7000`; `HOMIE vx_cmd = +1.00`, `+0.42 m/s` pelvis/bin/bar velocity (`5.0 m` in `12.7 s`), `12.4°` tilt (`100%` in band), `98.3%` PUSH grip, `96.5%` both wrapped, `0%` topple (§16) |
| `overhand_full1` (10-04) | scratch (iter=0) | → 1065 (stopped) | **PARTIAL (`TILT->PUSH` reached `4.25 m`, `REACH->GRASP` stuck at `0%`):** 15% grasp-init dominated gradients so `STAGE_REACH` stalled (`44/50 cm`) and left wrist pinned at `+92.5°` joint limit; also inherited `grasp_tilt.yaml`'s `[-0.84,-0.74]` spawn bug → `overhand_full2` (§17) |
| `overhand_full2` (10-04) | rg2 1700 | → 2950 (stopped) | **SOLVED 5-stage pipeline (`97.2%` end-to-end success across `8..15 kg` at `3.50 m`):** `spawn_x_range: [-0.95,-0.87]`, `grasp_advance: 1500` (+30), `pre_push_overtime_is_timeout: false`, `arm_branch_overhand_mode: true` (`penalty_arm_branch_guard: -4.0`), `4.0 -> 13.0 kg` mass + `0.15 -> 3.50 m` push curricula; `99.1%` grasp-reliable, `99.8%` PUSH reached, `97.4%` PUSH success, `0%` topple (§17) |
| `overhand_full3` (10-06) | full2 2900 | → 2930 (stopped) | Polar `STAGE_APPROACH` Spawn DR + full 5-stage task; switched to `approach_dr1` (`approach_is_final: true`) first for faster stage-0 throughput (§18) |
| `overhand_approach_dr1` (10-06) | full2 2900 | → 3350 (stopped) | **SOLVED Polar `STAGE_APPROACH` (`86.5%` success across `d ∈ [0.02, 1.10] m`, `φ ∈ [-45°, +45°]`, `Δψ_fov ∈ [-20°, +20°]`):** swept all 6 polar levels in 190 iters (`3090`), `96.2%` pos `< 15 cm` (`mean = 8.8 cm`), `93.8%` yaw `< 12°` (`mean = 5.0°`), but left arm raised (`1.15 rad`) due to `default_dof_pos != resting_arm_dof_pos` → `approach_dr2..dr4` (§18, §19) |
| `overhand_approach_dr2..dr4` (10-07) | approach_dr1 3300 | → 3505 (stopped) | **SOLVED Resting-Arm Polar Approach (`99.4%` success at full `1.10 m / ±45° / ±20°` cap, `100%` arms tucked, mean max dev `5.1°`):** `reset_delta_actions_to_resting: true` + `approach_arm_delta_decay: 0.50` + `stage_advance_require_resting_arms: true` (`0.35 rad`), `penalty_upper_body_non_finger_deviation_l1: -1.0`, polar level reset to `0.35 m` in `dr4` (§19) |
| `overhand_full4` (10-07) | approach_dr4 3500 | → 3674 (stopped) | **DIAGNOSED (`3%` `REACH → GRASP`):** `approach_dr1.yaml` still had `episode_length_s: 5.0` and `staged_reset_ratios: [1.0, 0, 0, 0, 0]` (`grasp_init_ratio: 0.0`), and 600 iters of `APPROACH`-only training had overwritten `REACH/GRASP/TILT/PUSH` in the shared LSTM → `overhand_full5` |
| `overhand_full5` (10-07) | full2 2900 | → 3650+ | **SOLVED Full 5-Stage Pipeline with Polar Approach & Tucked Arms (`96.7%` `APPROACH → REACH`, `91.8%` `APPROACH → COMPLETE` at full `1.10 m / ±45° / ±20°` polar cap & `3.50 m` push across `6..13 kg` body mass):** restored `episode_length_s: 28.0`, `staged_reset_ratios: [0.50, 0.18, 0.16, 0.16, 0.0]`, `grasp_init_ratio: 0.05`, `approach_is_final: false`; `98.1%` grasp-reliable (`88.7%` both wrapped), `95.0%` PUSH success at `3.50 m` (`99.8%` grip, `+13.4°` tilt, `+0.29 m/s`), `61.0%` overall success across all staged resets (`63.0% / 61.0% / 58.1%` on `6–9 / 9–11 / 11–13 kg`) |

## Gotchas learned along the way

- **Contact forces are normal-only.** Isaac Lab's `ContactSensor` `force_matrix_w` / `net_forces_w` report the **normal** force only. For friction you need `track_friction_forces=True` plus a `max_contact_data_count_per_prim` large enough for the contact count (64 for a two-hand claw).
- **One sensor per body.** A single `ContactSensor` spanning several bodies per env, combined with per-env filters, makes PhysX reject the filters, and the force matrix comes back **silently empty**.
- **New sensors can hang shutdown.** Adding brand-new contact sensors once made `simulation_app.close()` hang. Changing flags on an existing sensor is fine.
- **Quaternion order.** `get_task_root_state` quaternions are **wxyz**; don't reorder them.
- **Fingertip offsets.** G1 finger link origins sit at the PIP/IP joints, ~5.2 cm short of the tip. Use the real tip offsets for proximity and wrap scores.
- **HOMIE deadband (`0.10` obs zeroing vs `0.50` stand→walk switch).** `homie_base.py` multiplies `_homie_commands[:, :2]` by `2.0` in both `get_physical_homie_commands()` and `_get_obs_b_homie_commands()`, and zeros `b_homie_commands[:, :3]` when `|cmd[:3]| < 0.10`. **However, `ppo_trainer_homie_api.py` (lines 47 & 789) only switches from `homie_stand_model` to `homie_walk_model` when `|cmd[:3]| > homie_switch_threshold = 0.50` (`0.25 m/s` physical velocity)!** For any `|cmd[:3]| <= 0.50`, `ppo_trainer_homie_api.py` sets `walk_mask = False`, zeros out the velocity commands (`reshaped_obs[..., :3] = 0.0`), and runs `homie_stand_model`. Setting `push_cmd_target = 0.30` in `tilt_push1..5` parked the policy at `vx_cmd = +0.27` (98% of steps > 0.10 at iter 6371 of `tilt_push5`) — saturating `push_forward_command` while still inside the `[0.0, 0.50]` `homie_stand_model` dead zone. `tilt_push6` sets `push_cmd_deadband: 0.50` and `push_cmd_target: 0.70` so `push_forward_command` drives `vx_cmd` across the actual `0.50` `homie_walk_model` switch threshold.
- **Unvisited stages.** Staged resets only replay stages the policy has already reached. Use grasp-init to put episodes into TILT/PUSH.
- **Curriculum level survives resumes.** `bin_mass_level_kg` is restored from the checkpoint's env state, in both training resume and `play_bin_push.py`, so `bin_mass_start_kg` alone cannot lower it. Use `bin_mass_level_reset_kg` (§10).
- **`bin_mass_start_kg` is also the floor of the sampled range** U[max(start, level − spread), level]. m10 (start 6, level 6) trained only on bins of exactly 6.0 kg.
- **Edge-hold.** A held bin can sit at its static tipping moment, front feet unloaded, and never rotate. A moment-only reward saturates there, so the tilt angle itself has to pay more than holding (§10).
- **"Body-bin contact" is not bin-filtered.**
  - `_body_bin_contact_force` (penalty `penalty_body_bin_contact`, metric `body_bin_contact_frac_tilt`) and `_wrist_bin_contact_terms` read the robot-wide net-contact sensor `simulator.contact_forces`.
  - Self-collisions are on: `robot.asset.self_collisions: 0` → `enabled_self_collisions=True`.
  - In the tilt5 4950 play, 100 % of the logged body contact was the left upper arm (`left_shoulder_yaw_link`) plus the left wrist pressing on `torso_link`: torso |F| = arm |F| + wrist |F| (ratio 1.00 in 99 % of the > 50 N steps). The video shows a gap between the chest and the bin.
  - So the `< 10 %` body-contact checks and the penalty measured self-contact there. A real bin-only measure needs a bin-filtered sensor on the body links.
- **Timeouts bootstrap, failures don't.**
  - Success and stage overtime (`overtime_is_timeout`) are both labelled timeouts, so the trainer adds γ·V(s_t) (`ppo_trainer_homie_api.py`, γ 0.9975). Only failures (grip loss, topple, …) end the return.
  - At ~1 reward/step, V ≈ 400, so a failure costs ≈ −20 − V ≈ −420, about 20× tilt5's success bonus.
  - Lengthening a stage window turns late failures, which used to be harmless timeouts, into terminations. That is why tilt5 uses 4 s, not 5 s (§10).
- **A level reset needs a new value.** `bin_mass_level_reset_kg` only triggers if it differs from the checkpoint's `bin_mass_level_reset_applied_kg` marker. tilt4 checkpoints carry 3.0, so 3.0 would silently keep the restored level.
- **Success row in the reward / debug CSVs.** On success the stage has already advanced to PUSH (`award_remaining_time_on_advance`), so that row shows stage = PUSH and a negative `time_in_stage`. Use `episode_step` for time-to-success.
- **`Episode/rew_*` TB tags** are per-episode sums divided by `max_episode_length_s` (20). A failure is −20, so `-Episode/rew_termination` = fail rate. With `complete` 1000 (+20), `Episode/rew_complete` ≈ success rate; with 50 it was success / 20.
- **Training-loop quirks:**
  - training processes hang on exit (pre-existing);
  - `++algo.trl.num_total_batches` counts from the resume point;
  - checkpoints are saved every `save_frequency` (100) iterations, plus `last.pt` every 50.
- **The overspeed termination has its own curriculum.** The `dof_overspeed` limit is `termination_level × 20 rad/s`. `termination_level` shrinks ×(1 − 0.0001) per update while `average_goal_reached` < 0.5 (grows above 0.7), down to `min_level`. In tilt5 it had decayed to the 0.1 floor = **2 rad/s**, so every fast tip of a light bin was a failure. Pin it with `termination_curriculum.min_level: 1.0` (§11). It logs 0.9999, not 1.0: the logged tensor is multiplied in place before the clip. The limit actually used is 20 rad/s.
- **A narrow mass window makes a specialist.** With body mass U[L − 2, L] and the level at the 25 kg cap, tilt5 5300 succeeds 99.5 % at 20–25 kg and 0 % at 11–17 kg (heavy pull, 300–500 N, at any mass). Previously the privileged mass observation was a constant (15.0), so the policy couldn't condition on mass; as of 2026-10-03, `_privileged_bin_mass` is updated with the live per-env sampled mass `_bin_mass_env` in `_apply_bin_mass_curriculum` and `_reset_bin_states`.
- **TILT-only does not transfer to PUSH zero-shot.** With `tilt_is_final` true the episode ends at the TILT→PUSH gate, so the policy never learns to brake. Zero-shot PUSH of 5100: 96 % reach PUSH, then 100 % topple (> 30°) within ~0.4 s.
- **The 10-step TILT→PUSH "hold" is a pass-through.** The checkpoint's saved PUSH-entry snapshots (stage 4, 4 per env × 2048 envs) show the handover at tilt 15° (p10–p90 13–17°) but still rotating at **+31°/s** (p10 19, p90 41–46), in tilt5 5100 and in push1 alike. 10 steps × 0.02 s × 31°/s ≈ 6°, i.e. a fling through the 10–22° band satisfies the gate. From 15° at 31°/s the bin passes the ~23° balance point in ~0.25 s and the 30° limit in ~0.5 s. So tilt5's 96 % "success" was a fling, not a hold. The snapshots are in `ckpt["env_state_dict"]["_staged_reset_state"]["buffers"]["bin"]["root_state"][4]` (tilt rate ≈ −ω_y for push axis +x).
- **Stage overtime as a timeout makes stalling free.** With `overtime_is_timeout` the TILT overtime bootstraps V (a stall episode paid ≈ 0.26/step, so V ≈ 100), while a PUSH topple is a −20 termination with no bootstrap (≈ −120 relative). Once PUSH stopped being the success (`tilt_is_final` false) and was ~100 % fatal, PPO learned within ~30 iterations (5181–5211) to keep the bin out of the gate band.
- **Where `stage_reward_scale` came from (Base `gr00t` vs. what we added):**
  -It is a hybrid—the base mechanism was already built into `gr00t`'s `StagedTaskBase`, and we customized it (plus added a companion stage-scaling helper) to solve the `TILT -> PUSH` transition:
  1. **What was already in base `gr00t` (`StagedTaskBase._reward_stage()` in `gr00t/rl/envs/base_task/staged_task_base.py`):**
     - `StagedTaskBase` defines a single baseline reward term `stage` configured by `stage_reward_scale: [s_0, s_1, ..., s_{K-1}]` and `accumulate_stage_reward: True`.
     - When an environment is in stage $k$, `_reward_stage()` pays the **cumulative sum of all completed stages' scales** plus the current stage's scale $s_k$ (if `stage_k_reward_condition()` holds):
       $$r_{\text{stage}}(s \text{ in stage } k) = \sum_{i=0}^{k-1} s_i \;+\; s_k \cdot \mathbb{I}\!\left[\text{cond}_k(s)\right]$$
     - The original design intent in `gr00t` is a **monotonically increasing survival staircase** across stages so that advancing to stage $k+1$ always raises the per-step baseline reward above stage $k$, preventing the policy from preferring an earlier stage just to stay alive.
  2. **What we invented/customized on top of it in `bin_push_homie.py`:**
     - **Zero-step `TILT` + massive `PUSH` cliff (`stage_reward_scale: [0.1, 0.25, 0.5, 0.0, 5.0]`):** Originally the YAML had uniform/gentle increments (`[0.1, 0.25, 0.5, 0.75, 1.0]`), which wasn't nearly enough to beat the dense grasp/pull rewards in `STAGE_TILT`. Setting $s_3 = 0.0$ (`STAGE_TILT` gives zero baseline bump over `GRASP`) and $s_4 = 5.0$ (`STAGE_PUSH` immediately jumps the per-step baseline from `0.85` to **`5.85/step`**) created an unmistakable value-function cliff across the `TILT -> PUSH` boundary.
     - **Stage-dependent shaping suppression (`_grasp_shape_stage_scale()` / `grasp_shape_in_tilt_push_scale: 0.25`):** Base `gr00t`'s `stage_reward_scale` only adds an offset—it does *not* rescale individual shaping terms (`grasp_bar_distance`, `grasp_Enclosure`, `grasp_hand_pitch`, etc.) across stages. We added `_grasp_shape_stage_scale()` so static grasp-holding rewards are paid at `100%` during `STAGE_GRASP` (to learn a clean overhand wrap) and can be attenuated inside `STAGE_TILT` and `STAGE_PUSH` so the policy can't farm static grip points by standing still.
