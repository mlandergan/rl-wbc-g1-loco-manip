#!/usr/bin/env python3
"""Grasp-progress report from the run's TensorBoard file (exact logged values).

Usage (on beast, repo root):
  venv/bin/python grasp_metrics.py                  # newest bin_push_* run (now overhand reach_grasp_tilt1)
  venv/bin/python grasp_metrics.py <run_dir> --cols 8 --win 10
  venv/bin/python grasp_metrics.py --base <run_dir> # compare against a specific baseline run
  venv/bin/python grasp_metrics.py --list           # print available scalar tags
Sections:
  0. PULSE: one line -- episode success (vs ~200 it ago), grasp-reliable,
     tilt-success, both wrapped, gate, L/R wrapped, tilt in band ('-' = not logged).
  1. GRASP SCORECARD: the reach->grasp funnel (REACH -> palms near -> palm ABOVE
     bar -> orientation -> close-ready -> overhand wrap L/R -> gate -> complete),
     latest value vs target, PASS/FAIL, and the first failing step = current
     BOTTLENECK. BASE column = end of the previous run (default: newest
     bin_push_overhand_reach_grasp2-* run, else the 2nd-newest bin_push_* run
     with checkpoints). Then the tilt1 / tilt2 PASS/FAIL CHECKS (see NOTE;
     the NOTE + CHECKS block is picked from the run name: *tilt2* -> tilt2,
     otherwise tilt1; 2026-10-02: *grasp_tilt3* -> the TILT-only grasp_tilt3
     block, *tilt3* -> tilt3). BOTTLENECK skips "in GRASP stage" when the run has a
     TILT stage (it falls naturally as envs advance).
  2. LEFT vs RIGHT per-hand table (latest values).
  3. Trend table over iterations (each cell = mean of the last --win points).
"""
import argparse, glob, math, os, sys

NOTE_TILT1 = """\
================================ WHAT WE EXPECT ================================
Run: bin_push_overhand_reach_grasp_tilt1 (2026-10-01) = RESUMED from the newest
  reach_grasp2 checkpoint (weights + optimizer + global_step, so steps continue
  from ~1700). REACH -> GRASP -> TILT; GRASP -> TILT no longer ends the episode,
  success (complete) = TILT -> PUSH gate (bin held 10-22 deg back with the grip
  maintained; tilt_is_final). 20 % of fresh starts begin grasped in TILT
  (canned overhand claw). Changes vs reach_grasp2:
  * grasp_gate_min_closure 0.4: both hands' p0->p1 closure >= 0.4 to pass the
    grasp gate (reach_grasp2 entered TILT with R at 0.35-0.39, then let go).
    Canned claw closure 0.46-0.52 -> passes.
  * penalty_lid_contact 0.0 (was -3.0): the "lid contact" is the index/middle
    DISTAL links behind/above the bar while the tips are in the slot = the hook.
  * BASE = reach_grasp2 (grasp_is_final): its "grasp success" is
    Env/average_goal_reached; here that is grasp_success_frac (episodes started
    at/before GRASP that reached TILT). The gate is now stricter (closure).
  * PASS/FAIL (declared before launch; "it" = iterations since the resume):
    ~300 it : grasp_success_frac >= 50% of reach_grasp2's final
              average_goal_reached, AND grip_maintained_frac_tilt > 0.6.
    ~1000 it: tilt_in_band_frac_tilt > 0.1 AND tilt_success_frac > 0.
    always  : bin_topple_frac ~0.
  * RED FLAGS: gate closure-ok stays low (policy does not squeeze harder ->
    grasp success collapses); closure in TILT drops below 0.4 (grip loosens);
    tilt back angle goes negative or topples; lid contact rises with hands
    leaving the bar.
================================================================================"""

NOTE_TILT2 = """\
================================ WHAT WE EXPECT ================================
Run: bin_push_overhand_reach_grasp_tilt2 (2026-10-01) = RESUMED again from
  reach_grasp2 model_step_001700.pt (same as tilt1). tilt1 grasped well (86 %
  reach TILT, grip kept 82-92 %, closure 0.48/0.51) but the bin NEVER rotated
  (tilt_back_max_deg_tilt <= 0.09 deg): the TILT rewards were outcome-only and
  the wheeled bin needs a finite moment about its wheel contact line first.
  Changes vs tilt1 (gate closure 0.4 + lid penalty 0 kept):
  * NEW bin_tilt_moment 3.0: clamp(M / 30 Nm, 0, 1) * grip-maintained, TILT
    only. M = hand-on-bin moment about the wheel-ground contact line
    (+ = tips back); 30 Nm = measured tilt-onset moment (feasibility A1).
  * bin_tilt_back_std_deg 10 (was 6): upright bin now sees exp(-2.25)=0.11.
  * New logs: tilt_moment_Nm_tilt, tilt_moment_frac_tilt (= clamped M/ref, not
    grip-gated), hand_force_N_tilt.
  * After the scripted tilt probes (A6-A8: 15 kg tipped + held 11-14 deg with
    both claws via squat + down-back push, but always with a chest brace):
    bin_tilt_moment fades 12->16 deg (topple guard) and NEW
    penalty_body_bin_contact -0.5 (x1 TILT, x4 PUSH) on non-hand/wrist/foot
    link contact; logs body_bin_contact_N_tilt / _frac_tilt (>50 N).
  * PASS/FAIL (declared before launch; "it" = iterations since the resume):
    ~300 it : tilt_moment_frac_tilt > 0.3 AND grip_maintained_frac_tilt > 0.6
              AND grasp success >= 50% of reach_grasp2's average_goal_reached.
    ~1000 it: tilt_back_max_deg_tilt > 5 deg AND tilt_in_band_frac_tilt > 0.05.
    ~2000 it: tilt_success_frac > 0.
    always  : bin_topple_frac ~0.
  * RED FLAGS: moment up but grip_maintained down (pulling the hands off the
    bar); hand_force up with moment ~0 (pressing down/sideways); tilt
    overshoots past 22-30 deg (topple); grasp success collapses; body contact
    frac high with tilt still ~0 (leaning on the bin without tipping it).
================================================================================"""

NOTE_TILT3 = """\
================================ WHAT WE EXPECT ================================
Run: bin_push_overhand_reach_grasp_tilt3 (2026-10-01) = RESUMED from
  reach_grasp2 model_step_001700 (the clean grasp policy, like tilt1/tilt2 --
  NOT tilt2 2300, to avoid inheriting its squeeze habit). tilt2 kept a 100 %
  two-claw grip in TILT but its tipping
  moment plateaued at ~6.5 Nm vs the ~23-30 Nm a 15 kg body needs: the bin
  never moved (tilt max -0.04 deg) in 500+ iterations, so the angle rewards
  never fired. HOMIE height pinned at its 0.75 m clip.
  TWO changes vs tilt2 (agreed 2026-10-01):
  1 bin_mass_curriculum. Bin BODY mass (wheels unchanged) re-sampled per reset
    from U[max(start, level-2), level] kg (inertia scaled); level starts at
    bin_mass_start_kg and steps +1 kg (to 15) whenever >= 3000 TILT episodes at
    the current level reach tilt success >= 30 % (else the counts are halved).
    Level saved in the checkpoint. Privileged mass obs stays 15.
  2 bin_tilt_moment_friction. The bin-hand contact sensor reported NORMAL force
    only; with friction (~27 of ~63 N grip) the true moment on a 6 kg bin sat
    right at its 8.1 Nm tip limit while the normal-only number read up to 20 Nm
    with the bin at rest. bin_tilt_moment and tilt_moment_*_tilt /
    hand_force_N_tilt now use normal + friction.
  New logs: bin_mass_level_kg, bin_mass_mean_kg, bin_mass_level_tilt_success,
  bin_mass_level_episodes, tilt_moment_normal_Nm_tilt (old normal-only moment).
  * PASS/FAIL (declared before launch; "it" = iterations since the resume):
    ~100 it : tilt_back_max_deg_tilt > 3 deg (the light bin starts to move).
    ~300 it : tilt_success_frac > 0.05 AND grip_maintained_frac_tilt > 0.6
              AND grasp success >= 50% of reach_grasp2's.
    ~1000 it: bin_mass_level_kg >= 10.
    ~2000 it: bin_mass_level_kg >= 15.
    always  : bin_topple_frac < 2%.
  * RED FLAGS: topple rising (light bin pushed past its ~23 deg balance
    point); level stuck at the start mass after ~500 it; grip / grasp success
    collapsing; body contact rising (chest brace); normal-only moment rising
    while the true (friction) moment stays flat (squeezing, not tipping).
================================================================================"""

NOTE_GRASP_TILT3 = """\
================================ WHAT WE EXPECT ================================
Run: bin_push_overhand_grasp_tilt3 (2026-10-02) = TILT-ONLY, RESUMED from
  reach_grasp2 model_step_001700. Every episode starts already holding the bar
  (grasp_init_ratio 1.0, staged_reset_ratios [1,0,0,0,0]) and ends as a success
  at the TILT -> PUSH gate (10..22 deg held 10 steps with the grip), or on
  topple / grip loss / TILT overtime (3 s). REACH / GRASP rows are n/a.
  = tilt3 (bin-mass curriculum 6 -> 15 kg, friction-corrected moment) plus TWO
  changes (agreed 2026-10-02):
  1 TILT only: tilt3 lost its grasp (97 % -> 23-37 %) once it learned to hold
    in TILT (early closing in GRASP broke the right claw); tilt1 did the same.
  2 bin_tilt_moment_min_ratio -1: moment reward = clamp(M/30, -1, 1) * grip *
    topple-fade (was clamped at 0). tilt3 held 100 % grip at 82 N but at
    M = -1.8 Nm (pushing the bin toward tipping FORWARD) at no cost.
  * START (smoke from 1700, 2048 envs, it 1-3): true moment +3..+5 Nm, grip
    0.61, tilt max 1.2-1.5 deg; episodes end on grip loss after ~1.2 s (the
    grasp policy never held in TILT). The first success point is noise (36 %
    from ONE episode in a near-empty window) -- ignore it.
  * PASS/FAIL (declared before launch; "it" = iterations since the resume):
    ~100 it : grip_maintained_frac_tilt > 0.8 AND tilt_moment_Nm_tilt > +3 Nm
              (learn to hold WITHOUT flipping the push; tilt3 went to -1..-5 Nm
              exactly when its grip improved).
    ~300 it : tilt_back_max_deg_tilt > 3 deg AND tilt_success_frac > 0.05.
    ~1000 it: bin_mass_level_kg >= 10.
    ~2000 it: bin_mass_level_kg >= 15.
    always  : bin_topple_frac < 2 % AND body_bin_contact_frac_tilt < 10 %.
  * RED FLAGS: moment trending toward <= 0 as the grip improves (the tilt3
    pattern); grip dropping (letting go to dodge the negative moment term);
    topple rising; body contact rising (chest brace); mass level stuck at 6 kg
    after ~500 it.
================================================================================"""

NOTE_GRASP_TILT3_M10 = """\
================================ WHAT WE EXPECT ================================
Run: bin_push_overhand_grasp_tilt3_m10 (2026-10-02) = grasp_tilt3 (TILT-ONLY,
  signed moment, 6 -> 15 kg curriculum) RESUMED from grasp_tilt3 model_step_002300
  with ONE change: reward scale bin_tilt_moment 3 -> 10.
  Why: grasp_tilt3 learned the hold (grip 100 %, full 3 s episodes) and its
  moment recovered from -2.6 Nm (1900) to +1.8..+2.7 Nm (2200-2300), but far too
  slowly -- a 6 kg bin needs ~8-12 Nm; tilt max fell to 0.02 deg while hand
  force rose to ~80 N as SQUEEZE (link magnitudes add, net force cancels).
  Grasp-quality terms ~89 % of the positive reward, moment 0.4 %; a tip-level
  push was worth ~1/7 of grasp_finger_wrap at scale 3, ~1/2 at scale 10.
  * START (grasp_tilt3 at 2300): moment +1.8 Nm, grip 1.00, tilt max 0.02 deg,
    hand force ~80 N, episode length 150.
  * PASS/FAIL (declared before launch; "it" = iterations since the resume):
    ~50 it  : tilt_moment_Nm_tilt > +4 Nm AND grip_maintained_frac_tilt > 0.9.
    ~150 it : tilt_back_max_deg_tilt > 3 deg.
    ~300 it : tilt_success_frac > 0.05.
    ~1000 it: bin_mass_level_kg >= 10.
    ~2000 it: bin_mass_level_kg >= 15.
    always  : bin_topple_frac < 2 % AND body_bin_contact_frac_tilt < 10 %.
  * RED FLAGS: grip < 85 % (trading the grip for the moment term); topple
    rising; body contact rising (chest brace); moment not above the start
    (+2 Nm) at ~150 it.
================================================================================"""

# 2026-10-02 grasp_tilt4
NOTE_GRASP_TILT4 = """\
================================ WHAT WE EXPECT ================================
Run: bin_push_overhand_grasp_tilt4 (2026-10-02) = grasp_tilt3_m10 (TILT-ONLY,
  signed moment x10) RESUMED from m10 model_step_004600 with THREE changes (A+B+C,
  agreed as one package):
  A bin_mass_level_reset_kg 3 + bin_mass_start_kg 2: level 6 -> 3 kg ONCE
    (masses U[2,3] at 3, U[2,4] at 4, then U[lvl-2, lvl]).
  B bin_tilt_back 6 -> 20: tipping beats holding at the edge for every mass
    (crossover ~11.3; at 6 it was ~13-14 kg).
  C bin_mass_end_kg 15 -> 25.
  Why: m10 edge-held 4-6 kg bins (hand moment = static tipping moment ~1.49 *
  mass Nm, front feet unload, tilt 0.09 deg); 2-3 kg bins tip and succeed 83 %
  (play of 4600). Level stuck at 6 kg, success 0.
  * START: log "level reset 6.00 -> 3.00 kg"; grip 100 %, success at 2-3 kg
    ~60-85 % (play was deterministic; training samples actions).
  * PASS/FAIL (declared before launch; "it" = iterations since the resume):
    ~50 it  : tilt_success_frac >= 0.30 AND bin_mass_level_kg >= 4.
    ~300 it : bin_mass_level_kg >= 6.
    ~1000 it: bin_mass_level_kg >= 10.
    ~2000 it: bin_mass_level_kg >= 15.
    always  : bin_topple_frac < 2 %, body_bin_contact_frac_tilt < 10 %,
              grip_maintained_frac_tilt >= 85 %.
  * RED FLAGS: closure L/R TILT < 42 % (grip gate 40 %); BOTH wrapped < 90 %;
    level stuck >= 300 it with tilt success < 10 % (esp. 6-7 kg = m10's
    edge-hold); topple or body contact rising.
================================================================================"""

# 2026-10-02 grasp_tilt5
NOTE_GRASP_TILT5 = """\
================================ WHAT WE EXPECT ================================
Run: bin_push_overhand_grasp_tilt5 (2026-10-02) = grasp_tilt4 RESUMED from
  tilt4 model_step_004700 (pre-collapse) with TWO changes (A+B, agreed):
  A TILT window 3 -> 4 s: max_stage_time [60,100,150,200,50].
  B success bonus complete 50 -> 1000 (+1 -> +20 once).
  + one-time level reset 8 -> 6 kg (bin_mass_level_reset_kg 6 -> U[4,6]).
  Why: tilt4 collapsed at level 8 (6-8 kg): those tips take 3.2-3.7 s > the
  3 s window (timeout, no success), and inside 3 s a tip ~ an edge-hold
  return (+1 bonus) -> 0 % success since ~4817, even at 3-4 kg (4900 play).
  4 s not 5 s (4700 play, level-8 masses): 3 s 16 % success / 0 % fail,
  4 s 55 % / 4 %, 5 s 56 % / 38 % (grip lost at 4-5 s). A termination costs
  -20 AND the bootstrapped future (V ~ 400): ~ -420 vs +20 per success.
  * START: log "level reset 8.00 -> 6.00 kg"; Scale complete = 1000.0.
    4700 play within 4 s: L6 100 %, L7 98 %, L8 55 %, L9 7 % success
    (deterministic; training samples actions -> lower, e.g. tilt4 L7: play
    67 % within 3 s vs 40 % in training).
  * PASS/FAIL (declared before launch; "it" = iterations since the resume):
    ~30 it  : bin_mass_level_kg >= 8 AND tilt_success_frac >= 0.30.
    ~300 it : bin_mass_level_kg >= 9 (past tilt4's wall).
    ~1000 it: bin_mass_level_kg >= 11.
    ~2000 it: bin_mass_level_kg >= 15.
    always  : topple < 2 %, body contact < 10 %, grip >= 85 %,
              termination (fail) rate < 5 % (= -rew_termination).
  * RED FLAGS: tilt success < 10 % for >= 100 it after >= 30 % (tilt4's
    collapse); tilt max -> 0; time_to_success -> 4 s; fail rate > 5 % and
    rising (grip wall at >= 7-8 kg); level stuck >= 300 it, success < 10 %.
================================================================================"""

# 2026-10-02 grasp_tilt6
NOTE_GRASP_TILT6 = """\
================================ WHAT WE EXPECT ================================
Run: bin_push_overhand_grasp_tilt6 (2026-10-02) = grasp_tilt5 RESUMED from
  tilt5 step 4950 (_ckpt_tilt5_4950) with ONE change:
  A dof_overspeed termination 2 -> 20 rad/s (termination_curriculum.min_level
    1.0). The limit is termination_level * 20 rad/s and termination_level had
    decayed to its floor 0.1 (shrinks while average_goal_reached < 0.5).
  + one-time level reset 13 -> 11 kg; bin_mass_end_kg 25 -> 13 (target 11-13).
  Why: deterministic sweeps (U[6,13] kg, 4 s window), success by body kg:
    kg      6-7  7-8  8-9  9-10 10-11 11-12 12-13
    4950@2   36    4    7   41   75    24     0   (6-10 kg: 59-96 % FAIL at
    4950@20  92   81   96  100   97    35     0    ~2 s = arms > 2 rad/s)
  At 2 rad/s every fast tip was a failure (~ -420) -> slow tips / edge-hold.
  * START: log "level reset 13.00 -> 11.00 kg"; Env/termination_level = 1.0.
  * PASS/FAIL (declared before launch; "it" = iterations since the resume):
    ~30 it  : bin_mass_level_kg >= 12 AND tilt_success_frac >= 0.50.
    ~150 it : bin_mass_level_kg >= 13 (the cap).
    ~300 it : tilt_success_frac >= 0.50 at 13; deterministic sweep of that
              checkpoint 11-12 kg >= 60 %, 12-13 kg >= 40 % (offline).
    always  : fail rate < 5 % (= -rew_termination), topple < 2 %, grip >= 85 %.
  * RED FLAGS: fail rate > 5 % and rising (faster tips -> topple / grip);
    success at 13 falling >= 100 it; time_to_success -> 4 s. Body contact is
    left-arm <-> torso SELF-contact (unfiltered sensor): watch only.
================================================================================"""

# 2026-10-03 tilt_push1
NOTE_TILT_PUSH1 = """\
================================ WHAT WE EXPECT ================================
Run: bin_push_overhand_tilt_push1 (2026-10-03) = grasp_tilt5 RESUMED from
  tilt5 model_step_005100 with the PUSH stage on (package agreed):
  A tilt_is_final false: the TILT->PUSH gate (10-22 deg held 10 steps + grip)
    no longer ends the episode; success = bin pushed 1.0 m (+x) with tilt
    5-25 deg and grip held (+20, reset).
  B PUSH window 1 -> 8 s: max_stage_time [60,100,150,200,400].
  C body mass FIXED U[6,13] kg (level reset 19 -> 13, end 13, spread 7).
  D overspeed limit pinned at 20 rad/s (termination_curriculum.min_level 1).
  Why 5100: deterministic TILT sweep U[6,13] @20 rad/s = 96 % (11-13 kg ~90 %).
  Zero-shot PUSH of 5100: 96 % reach PUSH, then 100 % topple (> 30 deg)
  within ~0.4 s -- TILT-only never had to brake / hold (episode ended at the
  gate); the bin's balance point is ~23 deg. Smoke start (5101-5105, 16 envs):
  push_reached 0.89-0.92, fail 100 %, grip in PUSH 0.31-0.36, progress 0.04 m,
  pelvis vx -0.07..-0.02 m/s, tilt in 5-25 band (PUSH) 0.70-0.76.
  * START: log "level reset 19.00 -> 13.00 kg"; termination_level >= 0.999
    (logs 0.9999 = clip display artifact); fail ~100 % EXPECTED.
  * PASS/FAIL (declared before launch; "it" = iterations since the resume;
    fail rate = -Episode/rew_termination):
    start   : push_reached_frac >= 0.85.
    ~100 it : fail rate < 60 %; grip_maintained_frac_push >= 0.6.
    ~300 it : fail rate < 25 %; push_progress_m_push >= 0.2 m;
              pelvis_fwd_vel_push > +0.1 m/s (walking, not an arm-only shove).
    ~1000 it: push_success_frac >= 0.30; episode success >= 25 %;
              success 11-13 kg >= 15 %.
  * RED FLAGS: push_reached < 50 % and falling (gate avoidance: hold < 10 deg
    in TILT, farm bin_tilt_back); fail > 50 % at 300 it; L or R wrapped < 80 %
    (tilt5 drifted to L 39 % after 5100); pelvis vx <= 0 at 1000 it.
  Body contact is mostly arm <-> torso SELF-contact (unfiltered): watch only.
================================================================================"""

# 2026-10-03 grasp_tilt7
NOTE_GRASP_TILT7 = """\
================================ WHAT WE EXPECT ================================
Run: bin_push_overhand_grasp_tilt7 (2026-10-03) = grasp_tilt5 RESUMED from
  tilt5 model_step_005100 (TILT-only, tilt_is_final) with ONE mechanism:
  A steady-hold TILT->PUSH gate: each of the 10 hold steps and the gate step
    also need |d tilt/dt| <= threshold; threshold curriculum 40 -> 10 deg/s
    (x0.9 when tilt success >= 0.5 over >= 3000 TILT episodes at the current
    threshold, /0.9 if < 0.2, else counts halved; 14 tightenings to the floor).
  + as push1: body mass FIXED U[6,13] kg (level reset 19 -> 13), 20 rad/s.
  Why: push1 FAIL. tilt5 hands over MID-SWING (stage-4 snapshots: 15 deg at
  +31 deg/s, p90 41-46) -> bin passes its ~23 deg balance point in ~0.25 s;
  zero-shot PUSH toppled 100 %, then push1 learned gate avoidance (bin flat).
  * START: log "level reset 19.00 -> 13.00 kg" and "tilt rate gate: threshold
    40.00 deg/s after checkpoint load"; termination_level >= 0.999.
  * PASS/FAIL (declared before launch; "it" = iterations since the resume;
    fail rate = -Episode/rew_termination):
    start (<= 20 it): threshold 40; tilt_success_frac >= 0.6.
    ~100 it : threshold <= 25 AND tilt success >= 0.5.
    ~300 it : threshold = 10 (floor) AND tilt success >= 0.5.
    ~500 it : tilt success >= 0.7 at 10 deg/s; success 11-13 kg >= 0.5;
              handover |rate| p90 <= 10.
    offline : zero-shot PUSH play (push1 config) of the ~500 it checkpoint:
              >= 50 % of PUSH entries upright (5-25 deg) + gripped 1 s after
              the handover (5100: 0 %).
    always  : fail rate < 20 %; grip_maintained_frac_tilt >= 85 %.
  * RED FLAGS: tilt success < 0.2 for >= 100 it (edge-hold below 10 deg);
    threshold stuck > 25 for >= 300 it; fail > 20 % for >= 100 it; tilt max
    -> 0; L or R wrapped < 80 %.
  Body contact is mostly arm <-> torso SELF-contact (unfiltered): watch only.
================================================================================"""

# 2026-10-03 tilt_push2
NOTE_TILT_PUSH2 = """\
================================ WHAT WE EXPECT ================================
Run: bin_push_overhand_tilt_push2 (2026-10-03) = tilt_push1 RESUMED from
  tilt5 model_step_005100 with five root-cause fixes for push1's failure modes:
  1 push_progress_use_bar_min true: push velocity & distance use min(bin_root,
    bar_midpoint) along +x so backward tipping cannot earn fake push progress.
  2 In-band tilt braking: bin_tilt_moment fades out over 10->14 deg, positive
    bin_tilt_progress is clipped >= 14 deg, and bin_tilt_steady (scale 12.0 in
    TILT+PUSH, exp(-(dtheta/15)^2) * band * grip) rewards arresting the swing.
  3 Smooth TILT->PUSH walk ramp (tilt_walk_ramp_in_band true): once in [10,22]
    deg and settling, penalty_not_standing_still relaxes on vx > 0 and
    push_forward_command + step_forward_while_pushing ramp in at 0.5x in TILT.
  4 Eliminate gate-avoidance penalty jump: body_bin_contact_push_scale 4 -> 1
    and PUSH stage_reward_scale 1.5 -> 2.5 (entering PUSH strictly higher-value
    than stalling in TILT).
  5 Push distance curriculum: 0.10 m -> 1.00 m in 0.05 m steps (advances at
    >= 40 % PUSH success over >= 1500 PUSH episodes).
  * START   : level reset 19 -> 13 kg; target_bin_push_distance_m = 0.10 m;
              push_reached_frac >= 0.85; fail rate ~90-100 % initially.
  * ~150 it : push_reached_frac >= 0.70 (no gate-avoidance collapse);
              tilt_rate_abs_dps_push < 35 deg/s; fail rate < 65 %.
  * ~300 it : fail rate < 35 %; push_bar_progress_m_push >= 0.08 m;
              pelvis_fwd_vel_push > +0.08 m/s; push_success_frac >= 0.25.
  * ~800 it : target_bin_push_distance_m >= 0.25 m; pelvis_fwd_vel_push > +0.12;
              grip_maintained_frac_push >= 0.75.
  * RED FLAGS: push_reached < 50 % and falling (gate avoidance); pelvis vx <= 0
    at 500 it; L or R wrapped < 80 %.
================================================================================"""

# 2026-10-03 tilt_push3
NOTE_TILT_PUSH3 = """\
================================ WHAT WE EXPECT ================================
Run: bin_push_overhand_tilt_push3 (2026-10-03) = tilt_push2 RESUMED from
  tilt5 model_step_005100 (tilt_is_final: false, live TILT -> PUSH transition)
  with four fixes for tilt_push2's gate-avoidance collapse (where PUSH reached
  dropped 96 % -> 0.4 % by iter 5250 as the policy parked flat at 0.08 deg):
  1 +20 stage-advance milestone bonus (rewards.reward_scales.tilt_advance 1000):
    restores the +20 payout at the TILT -> PUSH gate while continuing live into
    PUSH (plus +20 complete at the end of PUSH).
  2 Steady-hold TILT -> PUSH rate gate + curriculum (40 -> 12 deg/s): all 10
    hold steps + gate step require |d(tilt)/dt| <= threshold so uncontrolled
    flings cannot enter PUSH mid-swing.
  3 tilt_overtime_is_timeout false: stalling in TILT for 4.0 s without reaching
    PUSH no longer gets a free timeout value bootstrap (treated as failure -20),
    while PUSH overtime stays a clean timeout.
  4 Wider braking Gaussian (bin_tilt_steady_std_dps 15 -> 35 deg/s) for strong
    damping gradient across 20-45 deg/s.
  * START   : level reset 19 -> 13 kg; tilt_gate_max_rate_dps = 40.0 deg/s;
              target_bin_push_distance_m = 0.10 m; push_reached_frac >= 0.60;
              Episode/rew_tilt_advance > 0.
  * ~150 it : push_reached_frac >= 0.65 (no collapse to 0 %);
              tilt_gate_max_rate_dps <= 32.4 deg/s; grip_maintained_frac_push >= 0.55.
  * ~300 it : tilt_gate_max_rate_dps <= 25 deg/s; push_bar_progress_m_push >= 0.05 m;
              pelvis_fwd_vel_push > +0.05 m/s; push_success_frac >= 0.20.
  * ~800 it : tilt_gate_max_rate_dps <= 15 deg/s; target_bin_push_distance_m >= 0.20 m;
              pelvis_fwd_vel_push > +0.10 m/s; grip_maintained_frac_push >= 0.70.
  * RED FLAGS: push_reached < 30 % for >= 100 it (gate avoidance); tilt max -> 0;
    L or R wrapped < 80 %.
================================================================================"""

# 2026-10-03 tilt_push4
NOTE_TILT_PUSH4 = """\
================================ WHAT WE EXPECT ================================
Run: bin_push_overhand_tilt_push4 (2026-10-03) = grounded architectural redesign
  of TILT -> PUSH transition and 3-axis curriculum, resumed from tilt5 5100:
  A Eliminate STAGE_TILT static survival farming (strict return dominance):
    - grasp_shape_in_tilt_push_scale: 0.25 scales the 9 grasp-acquisition posture
      terms by 0.25x in TILT/PUSH (+0.251 -> +0.063/step, freeing wrist compliance).
    - bin_tilt_back_zero_at_upright: true zeroes the 0-deg Gaussian tail in TILT.
    - bin_tilt_moment_stall_decay_steps: 40 decays positive moment reward to 0
      over 35 steps if theta < 2.0 deg after 0.8 s in TILT (TILT window 2.5 s).
    - stage_reward_scale: [0.1, 0.25, 0.5, 0.3, 3.5] + tilt_advance: 4000 (+80).
    -> 125-step TILT stall yields -5.5 net return vs +108.5 for crossing into PUSH
       even if failing 0.48 s later (+114.0 advantage for advancing to PUSH).
  B Safer [10, 17] deg handover band (held 15 steps = 0.30 s, 14.0 deg target,
    bin_tilt_moment_off_deg 13.0 -> >= 6 deg margin below ~23 deg CoM apex).
  C PUSH grip compliance (push_stage_grip_max_bar_dist 0.10 m, min_closure 0.50)
    + reanchor_bar_on_push_settle (15 steps) + pre-deadband vx ramp
    (push_cmd_pre_deadband_weight 0.25).
  D Coupled 3-axis curricula: Rate (25 -> 12 deg/s), Push Distance (0.10 -> 1.00 m),
    Bin Body Mass (starts at level 11.0 kg = U[6, 11] kg, ratchets 11->12->13 kg
    as full PUSH completion >= 50 %).
  * START   : mass level reset 19 -> 11 kg; tilt_gate_max_rate_dps = 25.0 deg/s;
              target_bin_push_distance_m = 0.10 m; Episode/rew_tilt_advance > 0.
  * ~100 it : push_reached_frac >= 0.60 (no gate avoidance); grip_maintained_push
              >= 0.70; homie_vx_cmd_push >= +0.10 m/s.
  * ~300 it : push_reached_frac >= 0.75; pelvis_fwd_vel_push >= +0.08 m/s;
              push_bar_progress_m_push >= +0.05 m; push_success_frac >= 0.25.
  * ~800 it : target_bin_push_distance_m >= 0.20 m; tilt_gate_max_rate_dps <= 20;
              bin_mass_level_kg >= 12.0 kg.
  * RED FLAGS: push_reached < 25 % for >= 100 it; tilt_back_max_deg_tilt < 3 deg;
    L or R wrapped < 80 %.
================================================================================"""

# 2026-10-03 tilt_push5
NOTE_TILT_PUSH5 = """\
================================ WHAT WE EXPECT ================================
Run: bin_push_overhand_tilt_push5 (2026-10-03) = tilt_push4 with the two loopholes
  exposed by the 30-min tilt_push4 run (steps 5101..5253) closed:
  1 Restore 5100's Grasp Force Equilibrium (grasp_shape_in_tilt_push_scale: 1.0):
    - In tilt_push4, scaling grasp-posture/wrap terms to 0.15x let bin_tilt_moment
      yank the arms into the torso (body contact 28 -> 396 N), collapsing R wrap
      (96 -> 66 %) and R palm dz (+3.8 -> +2.1 cm), which eroded PUSH grip from
      97 % (iter 5101) down to 20 % (iter 5253). Restoring 1.0x preserves 5100's
      97 % initial PUSH grip retention and 96 % overhand wrap.
  2 Close the < 10 deg "twitch-and-hold" loophole + 2.5 s (125-step) TILT window:
    - max_stage_time: [60, 100, 150, 125, 400] (TILT 200 -> 125 steps = 2.5 s).
    - bin_tilt_moment_stall_min_deg: 10.0 (checks current theta < 10 deg, decay
      steps 30, span 25 -> 0 moment by 1.1 s anywhere below the 10 deg gate).
    - bin_tilt_back_min_deg_tilt: 4.0 (0 tilt_back reward at <= 4 deg in TILT).
    - tilt_advance: 2500 (+50 bonus): 125-step TILT stall nets +14.0 vs >= +62.6
      for crossing into PUSH (+48.6 strict return advantage).
  3 Direct 1st-Order Over-Pull Moment Braking in TILT (>= 11 deg) and PUSH:
    - bin_tilt_moment_brake_in_push: true (brake_start_deg 11.0, steady_hold 7 Nm)
      penalizes excess backward pull M > M_allow once tilted and rotating backward
      so the policy unwinds the 130 N backward yank on entering [10, 17] deg.
  * START   : mass level reset 19 -> 11 kg; tilt_gate_max_rate_dps = 25.0 deg/s;
              target_bin_push_distance_m = 0.10 m; Episode/rew_tilt_advance > 0.
  * ~100 it : push_reached_frac >= 0.60 (no gate avoidance); grip_maintained_push
              >= 0.70; homie_vx_cmd_push >= +0.10 m/s; L & R wrapped >= 80 %.
  * ~300 it : push_reached_frac >= 0.75; pelvis_fwd_vel_push >= +0.08 m/s;
              push_bar_progress_m_push >= +0.05 m; push_success_frac >= 0.25.
  * ~800 it : target_bin_push_distance_m >= 0.20 m; tilt_gate_max_rate_dps <= 20;
              bin_mass_level_kg >= 12.0 kg.
  * RED FLAGS: push_reached < 25 % for >= 100 it; tilt_back_max_deg_tilt < 3 deg;
    L or R wrapped < 80 %.
================================================================================"""

# 2026-10-04 tilt_push6
NOTE_TILT_PUSH6 = """\
================================ WHAT WE EXPECT ================================
Run: bin_push_overhand_tilt_push6 (2026-10-04) = tilt_push5 + STAGE_PUSH
  anti-stall principle, explicit non-positive velocity penalty, and 0.15 -> 5.0 m
  straight-walk curriculum (900 steps = 18.0 s in STAGE_PUSH, episode 22.0 s).
  1 STAGE_PUSH Anti-Stall Gate (push_anti_stall_enable: true): after 25 steps
    (0.5 s) in STAGE_PUSH, ramps over 15 steps a forward-velocity gate w_move
    on bin_tilt_back and bin_tilt_steady based on min(pelvis_fwd_vel,
    bin_root_fwd_vel, bar_fwd_vel) (0.15 at <=0.02 m/s -> 1.0 at >=0.12 m/s),
    plus push_progress_require_positive_vel: true on pos_reward.
  2 Non-Positive Velocity & Sub-Deadband Penalties: step_forward_zero_at_rest:
    true + step_forward_stall_penalty: 1.0 (penalizes v_eff < 0.05 m/s in PUSH)
    + push_cmd_sub_deadband_penalty: 0.5 (penalizes vx_cmd < 0.10 m/s in PUSH)
    + penalize_lateral_yaw_cmd_in_push: true (straight-line walking).
  3 5.0 m Push Curriculum: push_distance_end_m: 5.00 (+0.05 m steps to 0.25 m,
    +0.15 m steps to 1.00 m, +0.25 m steps to 5.00 m).
================================================================================"""

# 2026-10-03 grasp_tilt7: steady-hold gate rows, appended to the trend table
# only for *grasp_tilt7* / *tilt_push3* runs (tags exist with tilt_advance_max_rate_dps; the
# handover / in-band rate tags are logged by every run from env 72d8dcc6 on).
METRICS_TILT7 = [
    ("Env/tilt_gate_max_rate_dps", "gate max rate [deg/s]", 1, "{:.1f}", "-", "40->10 @300"),
    ("Env/tilt_rate_level_success", "tilt success @rate lvl", 100, "{:.0f}%", "+", ">=50%"),
    ("Env/tilt_rate_level_episodes", "rate lvl episodes", 1, "{:.0f}", "+", "info"),
    ("Env/tilt_gate_rate_dps", "handover rate mean", 1, "{:+.1f}", "-", "<= thr"),
    ("Env/tilt_gate_rate_p90_dps", "handover |rate| p90", 1, "{:.1f}", "-", "<=10 @500"),
    ("Env/tilt_gate_tilt_deg", "handover tilt [deg]", 1, "{:.1f}", "-", "info 10..22"),
    ("Env/tilt_rate_abs_dps_band_tilt", "|rate| in band TILT", 1, "{:.1f}", "-", "falls"),
    ("Env/tilt_steady_frac_band_tilt", "steady frac in band", 100, "{:.0f}%", "+", "up"),
    ("Env/episode_success_frac_m6_9", "success 6-9 kg", 100, "{:.1f}%", "+", "up"),
    ("Env/episode_success_frac_m9_11", "success 9-11 kg", 100, "{:.1f}%", "+", "up"),
    ("Env/episode_success_frac_m11_13", "success 11-13 kg", 100, "{:.1f}%", "+", ">=50% @500"),
]

# 2026-10-03 tilt_push1 / tilt_push2 / tilt_push3: PUSH-stage rows, appended to the trend table
# for *tilt_push* runs (tags exist with push_stage_logs / success_by_mass_edges_kg).
METRICS_PUSH = [
    ("Env/stage_4_frac", "stage_4 PUSH frac", 100, "{:.1f}%", "+", "up"),
    ("Env/push_reached_frac", "PUSH reached (of TILT)", 100, "{:.1f}%", "+", ">=85%"),
    ("Env/push_success_frac", "PUSH success (of reached)", 100, "{:.1f}%", "+", ">=25% @300"),
    ("Env/target_bin_push_distance_m", "push target dist [m]", 1, "{:.2f}", "+", "0.10->5.00"),
    ("Env/grip_maintained_frac_push", "grip maintained PUSH", 100, "{:.0f}%", "+", ">=60% @100"),
    ("Env/tilt_back_deg_push", "tilt mean [deg] PUSH", 1, "{:+.1f}", "-", "~15 (5..25)"),
    ("Env/tilt_in_band_frac_push", "tilt in 5-25 band PUSH", 100, "{:.0f}%", "+", ">50%"),
    ("Env/tilt_rate_abs_dps_push", "|rate| in PUSH [deg/s]", 1, "{:.1f}", "-", "<35 @150"),
    ("Env/tilt_steady_weight_push", "steady weight PUSH", 1, "{:.2f}", "+", "up"),
    ("Env/push_progress_m_push", "push progress [m] PUSH", 1, "{:.2f}", "+", "up"),
    ("Env/push_bar_progress_m_push", "bar progress [m] PUSH", 1, "{:+.2f}", "+", ">=0.08 @300"),
    ("Env/push_progress_max_m_push", "push progress max [m]", 1, "{:.2f}", "+", "->target"),
    ("Env/homie_vx_cmd_band_tilt", "HOMIE vx in TILT band", 1, "{:+.2f}", "+", "up (ramp)"),
    ("Env/homie_vx_cmd_push", "HOMIE vx cmd PUSH", 1, "{:+.2f}", "+", "->0.3"),
    ("Env/walk_cmd_frac_push", "walk cmd frac (vx>0.1)", 100, "{:.0f}%", "+", "up"),
    ("Env/pelvis_fwd_vel_push", "pelvis fwd vel [m/s]", 1, "{:+.2f}", "+", ">0.08 @300"),
    ("Env/bin_fwd_vel_push", "bin root fwd vel [m/s]", 1, "{:+.2f}", "+", "->0.3"),
    ("Env/bar_fwd_vel_push", "bar fwd vel [m/s] PUSH", 1, "{:+.2f}", "+", "->0.3"),
    ("Env/push_anti_stall_weight_push", "anti-stall weight PUSH", 1, "{:.2f}", "+", "->1.0"),
    ("Env/body_bin_contact_frac_push", "body contact>50N PUSH", 100, "{:.0f}%", "-", "info (self)"),
    ("Episode/rew_grasp_advance", "rew grasp advance (+30)", 1, "{:+.3f}", "+", "up"),
    ("Episode/rew_tilt_advance", "rew tilt advance (+50)", 1, "{:+.3f}", "+", "up"),
    ("Episode/rew_penalty_arm_branch_guard", "rew arm/wrist guard", 1, "{:+.4f}", "+", "->0"),
    ("Episode/rew_bin_tilt_steady", "rew bin tilt steady", 1, "{:+.3f}", "+", "up"),
    ("Episode/rew_push_bin_progress", "rew push progress", 1, "{:+.3f}", "+", "up"),
    ("Episode/rew_step_forward_while_pushing", "rew step forward", 1, "{:+.3f}", "+", "up"),
    ("Env/episode_success_frac_m6_9", "success 6-9 kg", 100, "{:.1f}%", "+", "up"),
    ("Env/episode_success_frac_m9_11", "success 9-11 kg", 100, "{:.1f}%", "+", "up"),
    ("Env/episode_success_frac_m11_13", "success 11-13 kg", 100, "{:.1f}%", "+", ">=15% @1000"),
    ("Env/push_reached_frac_m11_13", "PUSH reached 11-13 kg", 100, "{:.0f}%", "+", "up"),
]

# 2026-10-06 full3: Polar STAGE_APPROACH Spawn DR & approach locomotion rows
METRICS_APPROACH_DR = [
    ("Env/stage_0_frac", "stage_0 APPROACH frac", 100, "{:.1f}%", "-", "info"),
    ("Env/spawn_dr_dist_level_m", "spawn DR dist lvl [m]", 1, "{:.2f}", "+", "0.25->1.10"),
    ("Env/spawn_dr_phi_level_deg", "spawn DR phi lvl [deg]", 1, "{:.1f}", "+", "15->45"),
    ("Env/spawn_dr_fov_yaw_level_deg", "spawn DR fov yaw [deg]", 1, "{:.1f}", "+", "10->20"),
    ("Env/approach_reached_frac", "APPROACH->REACH frac", 100, "{:.1f}%", "+", ">=75%"),
    ("Env/approach_ep_success_frac", "APPROACH->COMPLETE frac", 100, "{:.1f}%", "+", ">=50%"),
    ("Env/spawn_dr_level_reach_frac", "DR lvl reach frac", 100, "{:.1f}%", "+", ">=75%"),
    ("Env/spawn_dr_level_succ_frac", "DR lvl succ frac", 100, "{:.1f}%", "+", ">=50%"),
    ("Env/spawn_dr_level_episodes", "DR lvl episodes", 1, "{:.0f}", "+", "info"),
    ("Env/approach_standoff_error_s0", "standoff along err [m]", 1, "{:+.2f}", "-", "<0.08"),
    ("Env/approach_lateral_error_s0", "standoff lat err [m]", 1, "{:.2f}", "-", "<0.12"),
    ("Env/approach_yaw_error_deg_s0", "approach yaw err [deg]", 1, "{:.1f}", "-", "<12.0"),
    ("Env/approach_yaw_ready_frac_s0", "approach yaw ready s0", 100, "{:.1f}%", "+", "up"),
    ("Env/approach_stance_ready_frac_s0", "approach xy ready s0", 100, "{:.1f}%", "+", "up"),
    ("Env/approach_transition_ready_frac_s0", "approach xy+yaw+v ready", 100, "{:.1f}%", "+", "up"),
    ("Env/approach_homie_cmd_norm_s0", "HOMIE cmd norm s0", 1, "{:.2f}", "+", ">0.50 walk"),
    ("Episode/rew_walk_to_bin", "rew walk_to_bin", 1, "{:+.3f}", "+", "up"),
    ("Episode/rew_stop_at_bin", "rew stop_at_bin", 1, "{:+.3f}", "+", "up"),
    ("Episode/rew_approach_walk_command", "rew approach_walk_cmd", 1, "{:+.3f}", "+", "up"),
    ("Episode/rew_approach_advance", "rew approach_advance", 1, "{:+.3f}", "+", "up"),
]

GATE_CM = 7.0  # grasp_bar_valid_dist 0.07 (overhand)
# ---- grasp funnel: (tag, label, scale, fmt, target test, target text)
FUNNEL = [
    ("Env/stage_1_frac", "in REACH stage", 100, "{:.0f}%", lambda v: v <= 0.5, "<=50%"),
    ("Env/reach_both_close_frac_s1", "REACH: both palms <15cm", 100, "{:.0f}%", lambda v: v >= 0.2, ">=20%"),
    ("Env/stage_2_frac", "in GRASP stage", 100, "{:.0f}%", lambda v: v >= 0.3, ">=30%"),
    ("Env/palm_dz_left_s12", "L palm above bar [cm]", 1, "{:+.1f}", lambda v: v >= 1.0, ">=+1.0"),
    ("Env/palm_dz_right_s12", "R palm above bar [cm]", 1, "{:+.1f}", lambda v: v >= 1.0, ">=+1.0"),
    ("Env/dist_worst_grasppoint_to_bar_p50_s2", "worst-hand dist p50 [cm]", 100, "{:.1f}", lambda v: v * 100 <= GATE_CM, f"<={GATE_CM}"),
    ("Env/grasp_both_twist_ok_frac_s2", "both wrists twist-ok", 100, "{:.0f}%", lambda v: v >= 0.5, ">=50%"),
    ("Env/grasp_both_finger_close_ready_frac_s2", "both close-ready", 100, "{:.0f}%", lambda v: v >= 0.5, ">=50%"),
    ("Env/ovh_wrapped_frac_left_s23", "L overhand wrapped", 100, "{:.0f}%", lambda v: v >= 0.2, ">=20%"),
    ("Env/ovh_wrapped_frac_right_s23", "R overhand wrapped", 100, "{:.0f}%", lambda v: v >= 0.2, ">=20%"),
    ("Env/ovh_both_wrapped_frac_s23", "BOTH wrapped", 100, "{:.1f}%", lambda v: v >= 0.05, ">=5%"),
    ("Env/grasp_gate_closure_ok_frac_s2", "gate closure >=0.4 both", 100, "{:.0f}%", lambda v: v >= 0.5, ">=50%"),
    ("Env/grasp_full_pose_valid_frac_s2", "GRASP GATE valid", 100, "{:.1f}%", lambda v: v > 0.05, ">5%"),
    # 2026-10-01 tilt1: episode-outcome metrics (ring buffer of the last 512
    # ended episodes) + TILT-stage logs.
    ("Env/grasp_success_frac", "grasp success (->TILT)", 100, "{:.1f}%", lambda v: v >= 0.05, ">=5%"),
    ("Env/grasp_reliable_frac", "grasp reliable GRASP->TILT", 100, "{:.1f}%", lambda v: v >= 0.2, ">=20%"),
    ("Env/stage_3_frac", "in TILT stage", 100, "{:.1f}%", lambda v: v >= 0.1, ">=10%"),
    ("Env/grip_maintained_frac_tilt", "TILT: grip maintained", 100, "{:.0f}%", lambda v: v > 0.6, ">60%"),
    # 2026-10-01 tilt2: force/moment in TILT (moment frac = clamped M / ref).
    ("Env/hand_force_N_tilt", "TILT: hand force on bin [N]", 1, "{:.1f}", lambda v: v > 5.0, ">5"),
    ("Env/tilt_moment_Nm_tilt", "TILT: moment, + tips [Nm]", 1, "{:+.1f}", lambda v: v > 0.0, ">0"),
    ("Env/tilt_moment_frac_tilt", "TILT: moment / onset ref", 100, "{:.0f}%", lambda v: v > 0.3, ">30%"),
    # 2026-10-01 tilt2: current bin back-tilt (+ = toward robot) over the envs
    # in TILT: mean, and the max over ALL TILT envs at each log step (window-
    # averaged). Upright bin at rest reads about -0.15 deg.
    ("Env/tilt_back_deg_tilt", "TILT: bin tilt mean [deg]", 1, "{:+.2f}", lambda v: v >= 10.0, "10..22"),
    ("Env/tilt_back_max_deg_tilt", "TILT: bin tilt max (any env)", 1, "{:+.2f}", lambda v: v > 5.0, ">5 @1000"),
    ("Env/body_bin_contact_frac_tilt", "TILT: body-bin contact >50N", 100, "{:.0f}%", lambda v: v < 0.5, "<50%"),
    ("Env/tilt_in_band_frac_tilt", "TILT: bin in 10-22deg band", 100, "{:.1f}%", lambda v: v > 0.1, ">10%"),
    ("Env/tilt_success_frac", "tilt success (TILT->done)", 100, "{:.1f}%", lambda v: v > 0, ">0"),
    ("Env/episode_success_frac", "EPISODE SUCCESS", 100, "{:.1f}%", lambda v: v > 0, ">0"),
    ("Episode/rew_complete", "rew_complete (success)", 1, "{:.3f}", lambda v: v > 0, ">0"),
    ("Env/bin_topple_frac_all", "bin toppled", 100, "{:.1f}%", lambda v: v <= 0.05, "<=5%"),
]

# BASE fallback: grasp_is_final runs (reach_grasp2) have no grasp_success_frac;
# their equivalent is the fraction of envs whose last episode completed.
BASE_ALIAS = {"Env/grasp_success_frac": "Env/average_goal_reached"}

# ---- per-hand rows: (label, left tag, right tag, scale, fmt, better '+'/'-', target)
HANDS = [
    ("REACH palm->target [cm]", "Env/reach_left_dist_s1", "Env/reach_right_dist_s1", 100, "{:.1f}", "-", "<15"),
    ("palm dz above bar [cm]", "Env/palm_dz_left_s12", "Env/palm_dz_right_s12", 1, "{:+.1f}", "+", ">=+1 (tgt +3.35)"),
    ("palm BELOW bar frac", "Env/palm_below_frac_left_s12", "Env/palm_below_frac_right_s12", 100, "{:.0f}%", "-", "->0 (<20%)"),
    ("grasp pt->bar [cm]", "Env/grasp_left_bar_dist_s2", "Env/grasp_right_bar_dist_s2", 100, "{:.1f}", "-", f"<{GATE_CM}"),
    ("  horiz [cm]", "Env/dist_left_grasppoint_to_bar_horiz_s2", "Env/dist_right_grasppoint_to_bar_horiz_s2", 100, "{:.1f}", "-", "~3.0"),
    ("  vert [cm]", "Env/dist_left_grasppoint_to_bar_vert_s2", "Env/dist_right_grasppoint_to_bar_vert_s2", 100, "{:.1f}", "-", "~3.4"),
    ("grasp pt->above-tgt [cm]", "Env/grasp_target_dist_left_s2", "Env/grasp_target_dist_right_s2", 100, "{:.1f}", "-", "->0 (canned 1-2)"),
    ("wrist twist err [rad]", "Env/grasp_left_twist_rad_s2", "Env/grasp_right_twist_rad_s2", 1, "{:.2f}", "-", "<1.05"),
    ("palm-down align", "Env/grasp_left_palm_down_s2", "Env/grasp_right_palm_down_s2", 1, "{:.2f}", "+", ">0.9"),
    ("finger close-ready", "Env/grasp_left_finger_close_ready_frac_s2", "Env/grasp_right_finger_close_ready_frac_s2", 100, "{:.0f}%", "+", ">50%"),
    ("closure (p0->p1)", "Env/closure_left_s23", "Env/closure_right_s23", 100, "{:.0f}%", "+", "~45-55% on bar"),
    ("closure in TILT", "Env/closure_left_tilt", "Env/closure_right_tilt", 100, "{:.0f}%", "+", ">40% (gate), canned 46-52"),
    ("overhand hand score", "Env/ovh_score_left_s23", "Env/ovh_score_right_s23", 1, "{:.2f}", "+", ">0.5 gate"),
    ("overhand wrapped", "Env/ovh_wrapped_frac_left_s23", "Env/ovh_wrapped_frac_right_s23", 100, "{:.0f}%", "+", ">20%"),
    ("index dsurf [cm]", "Env/ovh_left_index_dsurf_cm_s23", "Env/ovh_right_index_dsurf_cm_s23", 1, "{:.1f}", "-", "<1.8"),
    ("middle dsurf [cm]", "Env/ovh_left_middle_dsurf_cm_s23", "Env/ovh_right_middle_dsurf_cm_s23", 1, "{:.1f}", "-", "<1.8"),
    ("thumb dsurf [cm]", "Env/ovh_left_thumb_dsurf_cm_s23", "Env/ovh_right_thumb_dsurf_cm_s23", 1, "{:.1f}", "-", "<1.8"),
    ("lid contact", "Env/lid_contact_left_s23", "Env/lid_contact_right_s23", 1, "{:.2f}", "-", "->0"),
]

# ---- trend table: (tag, label, scale, fmt, expected direction, target)
METRICS = [
    ("Env/stage_1_frac", "stage_1 REACH frac", 100, "{:.0f}%", "-", "falls"),
    ("Env/stage_2_frac", "stage_2 GRASP frac", 100, "{:.0f}%", "+", ">5% @300"),
    ("Env/reach_transition_ready_frac_s1", "REACH->GRASP ready", 100, "{:.1f}%", "+", "up"),
    ("Env/reach_left_dist_s1", "L reach dist [cm]", 100, "{:.1f}", "-", "<15"),
    ("Env/reach_right_dist_s1", "R reach dist [cm]", 100, "{:.1f}", "-", "<15"),
    ("Env/palm_dz_left_s12", "L palm dz [cm]", 1, "{:+.1f}", "+", ">0 @1000"),
    ("Env/palm_dz_right_s12", "R palm dz [cm]", 1, "{:+.1f}", "+", ">0 @1000"),
    ("Env/palm_below_frac_left_s12", "L palm below frac", 100, "{:.0f}%", "-", "<20% @1000"),
    ("Env/palm_below_frac_right_s12", "R palm below frac", 100, "{:.0f}%", "-", "<20%"),
    ("Episode/rew_penalty_palm_below_bar", "rew pen palm below", 1, "{:.3f}", "+", "->0"),
    ("Env/grasp_left_bar_dist_s2", "L bar dist [cm]", 100, "{:.1f}", "-", f"<{GATE_CM}"),
    ("Env/grasp_right_bar_dist_s2", "R bar dist [cm]", 100, "{:.1f}", "-", f"<{GATE_CM}"),
    ("Env/grasp_target_dist_left_s2", "L to above-tgt [cm]", 100, "{:.1f}", "-", "->0"),
    ("Env/grasp_target_dist_right_s2", "R to above-tgt [cm]", 100, "{:.1f}", "-", "->0"),
    ("Env/dist_worst_grasppoint_to_bar_p50_s2", "worst dist p50 [cm]", 100, "{:.1f}", "-", f"<{GATE_CM}"),
    ("Env/grasp_both_finger_close_ready_frac_s2", "both close-ready", 100, "{:.0f}%", "+", ">50%"),
    ("Env/ovh_score_left_s23", "L overhand score", 1, "{:.2f}", "+", ">0.5"),
    ("Env/ovh_score_right_s23", "R overhand score", 1, "{:.2f}", "+", ">0.5"),
    ("Env/ovh_wrapped_frac_left_s23", "L wrapped", 100, "{:.0f}%", "+", ">20% @1500"),
    ("Env/ovh_wrapped_frac_right_s23", "R wrapped", 100, "{:.0f}%", "+", ">20% @1500"),
    ("Env/ovh_both_wrapped_frac_s23", "BOTH wrapped", 100, "{:.1f}%", "+", ">5%"),
    ("Env/grasp_full_pose_valid_frac_s2", "grasp gate valid", 100, "{:.2f}%", "+", ">0"),
    ("Env/grasp_gate_closure_ok_frac_s2", "gate closure ok", 100, "{:.0f}%", "+", "up"),
    ("Env/grasp_success_frac", "grasp success", 100, "{:.1f}%", "+", ">=50% base"),
    ("Env/grasp_reliable_frac", "grasp reliable", 100, "{:.1f}%", "+", "up"),
    ("Env/stage_3_frac", "stage_3 TILT frac", 100, "{:.1f}%", "+", "up"),
    ("Env/closure_left_tilt", "L closure TILT", 100, "{:.0f}%", "+", ">40%"),
    ("Env/closure_right_tilt", "R closure TILT", 100, "{:.0f}%", "+", ">40%"),
    ("Env/grip_maintained_frac_tilt", "grip maintained TILT", 100, "{:.0f}%", "+", ">60% @300"),
    ("Env/tilt_back_deg_tilt", "tilt mean [deg] TILT", 1, "{:+.2f}", "+", "10..22"),
    ("Env/tilt_back_max_deg_tilt", "tilt max any env [deg]", 1, "{:+.2f}", "+", ">5 @1000"),
    ("Env/hand_force_N_tilt", "hand force TILT [N]", 1, "{:.1f}", "+", "up"),
    ("Env/tilt_moment_Nm_tilt", "moment TILT [Nm]", 1, "{:+.1f}", "+", "-> ref"),
    # 2026-10-01 tilt3: the row above includes friction in tilt3; this is the
    # old normal-only number (tilt1/tilt2's moment) for comparison.
    ("Env/tilt_moment_normal_Nm_tilt", "moment normal-only [Nm]", 1, "{:+.1f}", "+", "info"),
    ("Env/tilt_moment_frac_tilt", "moment/ref TILT", 100, "{:.0f}%", "+", ">30% @300"),
    # 2026-10-02 grasp_tilt3: the moment reward itself (signed when
    # bin_tilt_moment_min_ratio < 0; the row above is the clamped 0..1 ratio).
    ("Episode/rew_bin_tilt_moment", "rew tilt moment", 1, "{:+.3f}", "+", "up"),
    # 2026-10-02 grasp_tilt4: the tilt-angle reward (scale 6 -> 20 in tilt4).
    ("Episode/rew_bin_tilt_back", "rew tilt back (angle)", 1, "{:+.3f}", "+", "up"),
    ("Env/body_bin_contact_N_tilt", "body-bin contact TILT [N]", 1, "{:.0f}", "-", "low"),
    ("Env/body_bin_contact_frac_tilt", "body contact>50N TILT", 100, "{:.0f}%", "-", "falls"),
    ("Env/tilt_in_band_frac_tilt", "tilt in band TILT", 100, "{:.1f}%", "+", ">10% @1000"),
    ("Env/tilt_success_frac", "tilt success", 100, "{:.1f}%", "+", ">0 @1000"),
    ("Env/episode_success_frac", "EPISODE SUCCESS", 100, "{:.1f}%", "+", "up"),
    ("Env/time_to_success_s", "time to success [s]", 1, "{:.1f}", "-", "falls"),
    ("Episode/rew_complete", "rew_complete", 1, "{:.3f}", "+", ">0"),
    # 2026-10-02 grasp_tilt5: termination reward per episode = -fail rate
    # (failure = -1000 * dt = -20; Episode/rew_* are divided by the 20 s
    # max episode length). A 4 s window exposes the >= 7-8 kg grip losses.
    ("Episode/rew_termination", "rew termination (=-fail)", 100, "{:+.1f}%", "+", ">-5% (t5)"),
    ("Env/lid_contact_any_frac_s23", "any lid contact", 100, "{:.0f}%", "-", "->0"),
    ("Env/bin_topple_frac_all", "bin toppled", 100, "{:.1f}%", "-", "~0"),
    # 2026-10-01 tilt3: bin body-mass curriculum.
    ("Env/bin_mass_level_kg", "bin mass level [kg]", 1, "{:.1f}", "+", "->15 (t4+ 25)"),
    ("Env/bin_mass_mean_kg", "bin mass mean [kg]", 1, "{:.1f}", "+", "up"),
    ("Env/bin_mass_level_tilt_success", "tilt success @level", 100, "{:.0f}%", "+", ">=30% steps"),
    ("Env/average_episode_length", "episode length", 1, "{:.0f}", "-", "falls w/ success"),
]

# PULSE line tags (2026-10-01)
PULSE_TAGS = ["Env/episode_success_frac", "Env/grasp_reliable_frac", "Env/tilt_success_frac",
              "Env/ovh_both_wrapped_frac_s23", "Env/grasp_full_pose_valid_frac_s2",
              "Env/ovh_wrapped_frac_left_s23", "Env/ovh_wrapped_frac_right_s23",
              "Env/tilt_in_band_frac_tilt",
              "Env/termination_level"]  # 2026-10-02 tilt6: overspeed limit = level * 20 rad/s
DEFAULT_BASE_GLOB = "bin_push_overhand_reach_grasp2-*"

BASE_DIR = os.path.expanduser("~/Development/rl-wbc-g1-loco-manip/logs_rl/g1_bin_push_homie")


def runs_by_recency():
    # Newest = run whose TensorBoard event file was written most recently
    # (dir mtime only changes when files are added/removed). Smoke runs excluded.
    ev = glob.glob(os.path.join(BASE_DIR, "bin_push_*", "**", "events.out.tfevents*"), recursive=True)
    if not ev: sys.exit("no bin_push_* run with tfevents found; pass a run dir")
    seen = []
    for e in sorted(ev, key=os.path.getmtime, reverse=True):
        r = os.path.join(BASE_DIR, os.path.relpath(e, BASE_DIR).split(os.sep)[0])
        if r not in seen: seen.append(r)
    return seen


def load(run):
    ev = sorted(glob.glob(os.path.join(run, "**", "events.out.tfevents*"), recursive=True), key=os.path.getmtime)
    if not ev: sys.exit(f"no tfevents under {run}")
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    acc = EventAccumulator(os.path.dirname(ev[-1]), size_guidance={"scalars": 0}); acc.Reload()
    tags = acc.Tags()["scalars"]

    def resolve(t):
        for c in (t, "train/" + t):
            if c in tags: return c
        return None
    want = ({t for t, *_ in METRICS} | {t for t, *_ in FUNNEL} | {h[1] for h in HANDS} | {h[2] for h in HANDS}
            | set(PULSE_TAGS) | set(BASE_ALIAS.values())
            | {t for t, *_ in METRICS_PUSH}  # 2026-10-03 tilt_push1
            | {t for t, *_ in METRICS_TILT7}  # 2026-10-03 grasp_tilt7
            | {t for t, *_ in METRICS_APPROACH_DR})  # 2026-10-06 full3
    return tags, {t: [(e.step, e.value) for e in acc.Scalars(resolve(t))] for t in want if resolve(t)}


def nice(x):
    for v in (1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000):
        if v >= x: return v
    return int(math.ceil(x / 1000) * 1000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?")
    ap.add_argument("--base", help="baseline run dir (default: newest %s, else 2nd-newest bin_push_* run)" % DEFAULT_BASE_GLOB)
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--win", type=int, default=10, help="points averaged per column")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    order = runs_by_recency() if not (a.run and a.base) else []
    run = a.run or order[0]
    rg2 = [r for r in sorted(glob.glob(os.path.join(BASE_DIR, DEFAULT_BASE_GLOB)), key=os.path.getmtime, reverse=True)
           if os.path.abspath(r) != os.path.abspath(run)]
    base = a.base or (rg2[0] if rg2 else None) or next((r for r in order if os.path.abspath(r) != os.path.abspath(run)
                           and glob.glob(os.path.join(r, "model_step_*.pt"))), None)
    tags, data = load(run)
    if a.list:
        print("\n".join(sorted(tags))); return
    if not data: sys.exit("none of the metric tags found; try --list")
    bdata = load(base)[1] if base else {}

    def last(d, t):
        pts = [v for _, v in d.get(t, [])][-a.win:]
        return sum(pts) / len(pts) if pts else None

    def blast(t):  # BASE value, with the grasp_is_final alias
        b = last(bdata, t)
        return last(bdata, BASE_ALIAS[t]) if b is None and t in BASE_ALIAS else b

    ref = data.get("Env/stage_2_frac") or next(iter(data.values()))
    steps = [s for s, _ in ref]
    first, lastc = steps[0], steps[-1]

    # ---- 0. PULSE (one line)
    def pct(t):
        v = last(data, t)
        return "-" if v is None else f"{100 * v:.1f}%"

    def pct2(t):
        v = last(data, t)
        return "-" if v is None else f"{100 * v:.0f}"

    sv = last(data, "Env/episode_success_frac")
    old = [v for s, v in data.get("Env/episode_success_frac", []) if s <= lastc - 200][-a.win:]
    if sv is None or not old:
        trend = "(- vs ~200 it ago)"
    else:
        o = sum(old) / len(old)
        arrow = "^" if sv > o + 1e-9 else ("v" if sv < o - 1e-9 else "=")
        trend = f"({arrow} vs {100 * o:.1f}% ~200 it ago)"
    lw_, rw_ = pct2("Env/ovh_wrapped_frac_left_s23"), pct2("Env/ovh_wrapped_frac_right_s23")
    print(f"PULSE @{lastc}: success {pct('Env/episode_success_frac')} {trend} | "
          f"grasp-reliable {pct('Env/grasp_reliable_frac')} | tilt-success {pct('Env/tilt_success_frac')} | "
          f"both wrapped {pct('Env/ovh_both_wrapped_frac_s23')} | gate {pct('Env/grasp_full_pose_valid_frac_s2')} | "
          f"L/R wrapped {lw_}/{rw_}% | tilt in band {pct('Env/tilt_in_band_frac_tilt')}")
    rname = os.path.basename(os.path.normpath(run))
    is_full = "full" in rname
    has_spawn_dr = "Env/spawn_dr_dist_level_m" in data or "full3" in rname
    # 2026-10-03 tilt_push1..6: tilt5 5100 resumed with the PUSH stage on.
    is_tp6 = "tilt_push6" in rname or "_tp6" in rname
    is_tp5 = "tilt_push5" in rname or "_tp5" in rname or is_tp6
    is_tp4 = "tilt_push4" in rname or "_tp4" in rname or is_tp5
    is_tp3 = "tilt_push3" in rname or "_tp3" in rname or is_tp4
    is_tp2 = "tilt_push2" in rname or is_tp3
    is_tp1 = "tilt_push1" in rname or is_tp2
    if has_spawn_dr:
        def num_ap(t, f):
            v = last(data, t)
            return "-" if v is None else f.format(v)
        print(f"APPR  @{lastc}: dist_lvl {num_ap('Env/spawn_dr_dist_level_m', '{:.2f}')}m "
              f"(phi +-{num_ap('Env/spawn_dr_phi_level_deg', '{:.0f}')}d, fov +-{num_ap('Env/spawn_dr_fov_yaw_level_deg', '{:.0f}')}d) | "
              f"APPR->REACH {pct('Env/approach_reached_frac')} | APPR->DONE {pct('Env/approach_ep_success_frac')} | "
              f"cmd_norm {num_ap('Env/approach_homie_cmd_norm_s0', '{:.2f}')} | "
              f"along_err {num_ap('Env/approach_standoff_error_s0', '{:+.2f}')}m | "
              f"lat_err {num_ap('Env/approach_lateral_error_s0', '{:.2f}')}m | "
              f"yaw_err {num_ap('Env/approach_yaw_error_deg_s0', '{:.1f}')}d "
              f"(ready {pct('Env/approach_yaw_ready_frac_s0')})")
    if is_tp1 or is_full:
        rtp = last(data, "Episode/rew_termination")  # = -fail rate
        def num(t, f):
            v = last(data, t)
            return "-" if v is None else f.format(v)
        print(f"PUSH  @{lastc}: reached {pct('Env/push_reached_frac')} | push-success {pct('Env/push_success_frac')} | "
              f"target {num('Env/target_bin_push_distance_m', '{:.2f}')}m | "
              f"fail {'-' if rtp is None else f'{-100 * rtp:.0f}%'} | grip {pct('Env/grip_maintained_frac_push')} | "
              f"tilt {num('Env/tilt_back_deg_push', '{:.1f}')} deg (|rate| {num('Env/tilt_rate_abs_dps_push', '{:.0f}')} d/s) | "
              f"prog {num('Env/push_progress_m_push', '{:.2f}')}m (bar {num('Env/push_bar_progress_m_push', '{:+.2f}')}m) | "
              f"pelvis vx {num('Env/pelvis_fwd_vel_push', '{:+.2f}')} m/s | 11-13kg {pct('Env/episode_success_frac_m11_13')}")
    # 2026-10-02 grasp_tilt6: tilt5 resumed from 4950 with the overspeed
    # termination 2 -> 20 rad/s (+ level reset 13 -> 11 kg, cap 13 kg).
    is_gt6 = "grasp_tilt6" in rname
    # 2026-10-03 grasp_tilt7: tilt5 5100 resumed with the steady-hold gate
    # (|tilt rate| threshold curriculum 40 -> 10 deg/s), body U[6,13], 20 rad/s.
    is_gt7 = "grasp_tilt7" in rname
    if is_gt7 or is_tp3 or is_full:
        rt7 = last(data, "Episode/rew_termination")  # = -fail rate
        def num7(t, f):
            v = last(data, t)
            return "-" if v is None else f.format(v)
        print(f"GATE  @{lastc}: thr {num7('Env/tilt_gate_max_rate_dps', '{:.1f}')} deg/s | "
              f"success @thr {pct('Env/tilt_rate_level_success')} | "
              f"handover rate {num7('Env/tilt_gate_rate_dps', '{:+.1f}')} (|p90| "
              f"{num7('Env/tilt_gate_rate_p90_dps', '{:.1f}')}) deg/s at "
              f"{num7('Env/tilt_gate_tilt_deg', '{:.1f}')} deg | |rate| in band "
              f"{num7('Env/tilt_rate_abs_dps_band_tilt', '{:.1f}')} (steady {pct('Env/tilt_steady_frac_band_tilt')}) | "
              f"fail {'-' if rt7 is None else f'{-100 * rt7:.0f}%'} | 6-9/9-11/11-13kg "
              f"{pct('Env/episode_success_frac_m6_9')}/{pct('Env/episode_success_frac_m9_11')}/"
              f"{pct('Env/episode_success_frac_m11_13')}")
    # 2026-10-02 grasp_tilt5: tilt4 resumed from 4700 with the TILT window
    # 3 -> 4 s and complete 50 -> 1000 (+ one-time level reset 8 -> 6 kg).
    is_gt5 = "grasp_tilt5" in rname
    # 2026-10-02 grasp_tilt4: TILT-only too (resumed from m10 4600; level
    # reset 3 kg, bin_tilt_back 20, mass end 25 kg).
    is_gt4 = "grasp_tilt4" in rname
    # 2026-10-02: grasp_tilt3 = TILT-only (every episode starts grasped).
    # 2026-10-03: tilt_push1/2/3/4/5/6 also start grasped in TILT -> same funnel logic.
    # 2026-10-03: grasp_tilt7 too (TILT-only).
    is_gt3 = "grasp_tilt3" in rname or is_gt4 or is_gt5 or is_gt6 or is_tp1 or is_gt7
    # 2026-10-02: grasp_tilt3_m10 = grasp_tilt3 with bin_tilt_moment 3 -> 10
    # (still TILT-only, so is_gt3 stays True for the row / bottleneck logic).
    is_m10 = "grasp_tilt3_m10" in rname
    is_tilt3 = "tilt3" in rname and not is_gt3
    is_tilt2 = "tilt2" in rname
    print(NOTE_TILT_PUSH6 if (is_tp6 or is_full) else (NOTE_TILT_PUSH5 if is_tp5 else (NOTE_TILT_PUSH4 if is_tp4 else (NOTE_TILT_PUSH3 if is_tp3 else (NOTE_TILT_PUSH2 if is_tp2 else (NOTE_GRASP_TILT7 if is_gt7 else (
        NOTE_TILT_PUSH1 if is_tp1 else (NOTE_GRASP_TILT6 if is_gt6 else (NOTE_GRASP_TILT5 if is_gt5 else (NOTE_GRASP_TILT4 if is_gt4 else (
            NOTE_GRASP_TILT3_M10 if is_m10 else (NOTE_GRASP_TILT3 if is_gt3 else (
                NOTE_TILT3 if is_tilt3 else (NOTE_TILT2 if is_tilt2 else NOTE_TILT1))))))))))))))
    print(f"run : {os.path.basename(run)}   steps {first}..{lastc} ({lastc - first} it this run)")
    print(f"base: {os.path.basename(base) if base else '-'}   (BASE = its last {a.win} points)\n")

    # ---- 1. scorecard
    print("=" * 30 + " GRASP SCORECARD " + "=" * 33)
    lw = max(len(f[1]) for f in FUNNEL)
    print(f"{'step':<3} {'grasp funnel':<{lw}} | {'BASE':>7} {'NOW':>7} | {'target':<11} status")
    print("-" * 80)
    bottleneck = None
    # 2026-10-01: with a TILT stage, GRASP occupancy falls as envs advance --
    # not a bottleneck signal.
    has_tilt = any(v > 0 for _, v in data.get("Env/stage_3_frac", []))
    skip_bn = {"Env/stage_2_frac"} if has_tilt else set()
    # 2026-10-02: TILT-only runs never visit REACH / GRASP -> those funnel rows
    # (everything before "in TILT stage") are not bottlenecks there.
    if is_gt3:
        ti = [f[0] for f in FUNNEL].index("Env/stage_3_frac")
        skip_bn |= {f[0] for f in FUNNEL[:ti]}
    # 2026-10-03 tilt_push1/2/3/4/5/6 & full1: TILT no longer ends the episode -> tilt success
    # (TILT-final completions) is 0 by construction; not a bottleneck.
    if is_tp1 or is_full:
        skip_bn |= {"Env/tilt_success_frac"}
    for i, (t, label, sc, fmt, ok, tgt) in enumerate(FUNNEL, 1):
        v, b = last(data, t), blast(t)
        fv = fmt.format(v * sc) if v is not None else "-"
        fb = fmt.format(b * sc) if b is not None else "-"
        if v is None: st = "n/a"
        else:
            st = "PASS" if ok(v) else "FAIL"
            if st == "FAIL" and bottleneck is None and t not in skip_bn: bottleneck = label
        arrow = "" if (v is None or b is None or abs(v - b) < 1e-9) else ("  (^)" if v > b else "  (v)")
        print(f"{i:<3} {label:<{lw}} | {fb:>7} {fv:>7} | {tgt:<11} {st}{arrow}")
    print("-" * 80)
    print(f"BOTTLENECK: {bottleneck or 'none -- all funnel steps pass'}")
    print("(^)/(v) = NOW higher/lower than BASE; for dist rows lower is better.")
    print("grasp success BASE = reach_grasp2 Env/average_goal_reached (grasp_is_final).\n")

    # ---- 1b. tilt1 PASS/FAIL checks (see NOTE)
    def chk(name, ok, detail):
        st = "n/a " if ok is None else ("PASS" if ok else "FAIL")
        print(f"  [{st}] {name:<44} {detail}")
    gs, gb = last(data, "Env/grasp_success_frac"), blast("Env/grasp_success_frac")
    if is_tilt3:
        print(f"TILT3 CHECKS ({lastc - first} it since resume; ~100 / ~300 / ~1000 / ~2000 it milestones):")
        tm3 = last(data, "Env/tilt_back_max_deg_tilt")
        chk("~100: tilt_back_max_deg_tilt > 3 deg", None if tm3 is None else tm3 > 3.0,
            "-" if tm3 is None else f"{tm3:+.1f} deg")
        ts3 = last(data, "Env/tilt_success_frac")
        chk("~300: tilt_success_frac > 0.05", None if ts3 is None else ts3 > 0.05, pct("Env/tilt_success_frac"))
        gm3 = last(data, "Env/grip_maintained_frac_tilt")
        chk("~300: grip_maintained_frac_tilt > 0.6", None if gm3 is None else gm3 > 0.6, pct("Env/grip_maintained_frac_tilt"))
        chk("~300: grasp success >= 50% of BASE", None if gs is None or not gb else gs >= 0.5 * gb,
            f"now {pct('Env/grasp_success_frac')} vs base {'-' if gb is None else f'{100 * gb:.1f}%'}")
        lv = last(data, "Env/bin_mass_level_kg")
        lvs = "-" if lv is None else f"{lv:.1f} kg"
        chk("~1000: bin_mass_level_kg >= 10", None if lv is None else lv >= 10.0 - 1e-3, lvs)
        chk("~2000: bin_mass_level_kg >= 15", None if lv is None else lv >= 15.0 - 1e-3, lvs)
        tp3 = last(data, "Env/bin_topple_frac_all")
        chk("always: bin_topple_frac < 2%", None if tp3 is None else tp3 < 0.02, pct("Env/bin_topple_frac_all"))
        print()
    if is_tilt2:
        print(f"TILT2 CHECKS ({lastc - first} it since resume; ~300 / ~1000 / ~2000 it milestones):")
        mf = last(data, "Env/tilt_moment_frac_tilt")
        chk("~300: tilt_moment_frac_tilt > 0.3", None if mf is None else mf > 0.3, pct("Env/tilt_moment_frac_tilt"))
        gm2 = last(data, "Env/grip_maintained_frac_tilt")
        chk("~300: grip_maintained_frac_tilt > 0.6", None if gm2 is None else gm2 > 0.6, pct("Env/grip_maintained_frac_tilt"))
        chk("~300: grasp success >= 50% of BASE", None if gs is None or not gb else gs >= 0.5 * gb,
            f"now {pct('Env/grasp_success_frac')} vs base {'-' if gb is None else f'{100 * gb:.1f}%'}")
        tm = last(data, "Env/tilt_back_max_deg_tilt")
        chk("~1000: tilt_back_max_deg_tilt > 5 deg", None if tm is None else tm > 5.0,
            "-" if tm is None else f"{tm:+.1f} deg")
        ib2 = last(data, "Env/tilt_in_band_frac_tilt")
        chk("~1000: tilt_in_band_frac_tilt > 0.05", None if ib2 is None else ib2 > 0.05, pct("Env/tilt_in_band_frac_tilt"))
        ts2 = last(data, "Env/tilt_success_frac")
        chk("~2000: tilt_success_frac > 0", None if ts2 is None else ts2 > 0, pct("Env/tilt_success_frac"))
        tp2 = last(data, "Env/bin_topple_frac_all")
        chk("always: bin_topple_frac ~0 (<1%)", None if tp2 is None else tp2 < 0.01, pct("Env/bin_topple_frac_all"))
        print()
    if is_gt3:
        lbl = "TILT_PUSH6" if is_tp6 else ("TILT_PUSH5" if is_tp5 else ("TILT_PUSH4" if is_tp4 else ("TILT_PUSH3" if is_tp3 else ("TILT_PUSH2" if is_tp2 else ("GRASP_TILT7" if is_gt7 else ("TILT_PUSH1" if is_tp1 else ("GRASP_TILT6" if is_gt6 else ("GRASP_TILT5" if is_gt5 else (
            "GRASP_TILT4" if is_gt4 else ("GRASP_TILT3_M10" if is_m10 else "GRASP_TILT3"))))))))))
        mil = "start / ~100 / ~300 / ~800" if is_tp4 else ("start / ~150 / ~300 / ~800" if is_tp2 else ("start / ~100 / ~300 / ~500" if is_gt7 else ("start / ~100 / ~300 / ~1000" if is_tp1 else (
            "~30 / ~150 / ~300" if is_gt6 else ((("~30 / ~300 / ~1000" if is_gt5 else (
                "~50 / ~300" if is_gt4 else ("~50 / ~150 / ~300" if is_m10 else "~100 / ~300")) + " / ~1000")
                + " / ~2000"))))))
        print(f"{lbl} CHECKS ({lastc - first} it since resume; {mil} it milestones):")
        gmg = last(data, "Env/grip_maintained_frac_tilt")
        mo = last(data, "Env/tilt_moment_Nm_tilt")
        tmg = last(data, "Env/tilt_back_max_deg_tilt")
        tsg = last(data, "Env/tilt_success_frac")
        lvg = last(data, "Env/bin_mass_level_kg")
        lvgs = "-" if lvg is None else f"{lvg:.1f} kg"
        if is_gt7:
            # 2026-10-03 grasp_tilt7 (pre-registered): steady-hold gate, threshold
            # curriculum 40 -> 10 deg/s; 5100 hands over at ~+31 deg/s (p90 41-46).
            def fmt7(v, f):
                return "-" if v is None else f.format(v)

            def roll7(t):  # (step, mean of the last --win points) per point
                sp = data.get(t, [])
                return [(sp[i][0], sum(v for _, v in sp[max(0, i - a.win + 1):i + 1]) / (i - max(0, i - a.win + 1) + 1))
                        for i in range(len(sp))]
            tl = last(data, "Env/termination_level")
            chk("start: termination_level >= 0.999 (20 rad/s)", None if tl is None else tl >= 0.999, fmt7(tl, "{:.4f}"))
            th = last(data, "Env/tilt_gate_max_rate_dps")
            ths = fmt7(th, "{:.1f} deg/s")
            th0 = [v for s, v in data.get("Env/tilt_gate_max_rate_dps", []) if s <= first + 20]
            chk("start: threshold 40 deg/s", None if not th0 else abs(th0[0] - 40.0) < 1e-3,
                fmt7(th0[0] if th0 else None, "{:.1f} deg/s (first point)"))
            ts0 = [v for s, v in data.get("Env/tilt_success_frac", []) if s <= first + 20]
            ts0m = sum(ts0) / len(ts0) if ts0 else None
            chk("start (<=20 it): tilt_success_frac >= 0.6", None if ts0m is None else ts0m >= 0.6,
                fmt7(None if ts0m is None else 100 * ts0m, "{:.1f}% (mean of the first 20 it)"))
            chk("~100: threshold <= 25 AND tilt success >= 0.5",
                None if th is None or tsg is None else (th <= 25.0 + 1e-3 and tsg >= 0.5),
                f"{ths}, {pct('Env/tilt_success_frac')}")
            chk("~300: threshold = 10 (floor) AND success >= 0.5",
                None if th is None or tsg is None else (th <= 10.0 + 1e-3 and tsg >= 0.5),
                f"{ths}, {pct('Env/tilt_success_frac')}")
            chk("~500: tilt success >= 0.7 at 10 deg/s",
                None if th is None or tsg is None else (th <= 10.0 + 1e-3 and tsg >= 0.7),
                f"{pct('Env/tilt_success_frac')} at {ths}")
            eh = last(data, "Env/episode_success_frac_m11_13")
            chk("~500: success 11-13 kg >= 0.5", None if eh is None else eh >= 0.5,
                pct("Env/episode_success_frac_m11_13"))
            gp = last(data, "Env/tilt_gate_rate_p90_dps")
            chk("~500: handover |rate| p90 <= 10 deg/s", None if gp is None else gp <= 10.0 + 1e-3,
                fmt7(gp, "{:.1f} deg/s"))
            rtt = last(data, "Episode/rew_termination")
            fr = None if rtt is None else -rtt
            chk("always: fail rate < 20%", None if fr is None else fr < 0.20,
                fmt7(None if fr is None else 100 * fr, "{:.1f}%"))
            rs = [m for s, m in roll7("Env/tilt_success_frac") if s > lastc - 100]
            chk("flag: no edge-hold (success < 0.2 for 100 it)",
                None if lastc - first < 100 or not rs else max(rs) >= 0.2,
                "-" if not rs else f"last-100-it max {100 * max(rs):.0f}%")
            chk("flag: threshold <= 25 by 300 it", None if lastc - first < 300 or th is None else th <= 25.0 + 1e-3, ths)
            rf = [0.0 - m for s, m in roll7("Episode/rew_termination") if s > lastc - 100]
            chk("flag: fail not > 20% for 100 it", None if lastc - first < 100 or not rf else min(rf) <= 0.20,
                "-" if not rf else f"last-100-it min {100 * min(rf):.0f}%")
            chk("flag: tilt max > 5 deg (still tipping)", None if tmg is None else tmg > 5.0, fmt7(tmg, "{:+.1f} deg"))
            lwv, rwv = last(data, "Env/ovh_wrapped_frac_left_s23"), last(data, "Env/ovh_wrapped_frac_right_s23")
            chk("flag: L and R wrapped >= 80%", None if lwv is None or rwv is None else min(lwv, rwv) >= 0.80,
                f"{pct2('Env/ovh_wrapped_frac_left_s23')}/{pct2('Env/ovh_wrapped_frac_right_s23')}%")
        elif is_tp4:
            # 2026-10-03 tilt_push4 (pre-registered): anti-edge-hold TILT rewards
            # + [10, 17] deg / 25 deg/s gate + 0.10m/0.50 grip + pre-deadband vx ramp.
            def fmt4(v, f):
                return "-" if v is None else f.format(v)
            tl = last(data, "Env/termination_level")
            chk("start: termination_level >= 0.999 (20 rad/s)", None if tl is None else tl >= 0.999, fmt4(tl, "{:.4f}"))
            rta = last(data, "Episode/rew_tilt_advance")
            chk("start: Episode/rew_tilt_advance > 0 (+80 gate bonus)", None if rta is None else rta > 0.0, fmt4(rta, "{:+.3f}"))
            th = last(data, "Env/tilt_gate_max_rate_dps")
            ths = fmt4(th, "{:.1f} deg/s")
            pr = last(data, "Env/push_reached_frac")
            chk("~100: push_reached_frac >= 0.60 (no gate avoidance)", None if pr is None else pr >= 0.60, pct("Env/push_reached_frac"))
            gp = last(data, "Env/grip_maintained_frac_push")
            chk("~100: grip_maintained_frac_push >= 0.70", None if gp is None else gp >= 0.70, pct("Env/grip_maintained_frac_push"))
            vc = last(data, "Env/homie_vx_cmd_push")
            chk("~100: homie_vx_cmd_push >= +0.10 m/s", None if vc is None else vc >= 0.10, fmt4(vc, "{:+.2f} m/s"))
            chk("~300: push_reached_frac >= 0.75", None if pr is None else pr >= 0.75, pct("Env/push_reached_frac"))
            pv = last(data, "Env/pelvis_fwd_vel_push")
            chk("~300: pelvis_fwd_vel_push >= +0.08 m/s", None if pv is None else pv >= 0.08, fmt4(pv, "{:+.2f} m/s"))
            pbp = last(data, "Env/push_bar_progress_m_push")
            chk("~300: push_bar_progress_m_push >= +0.05 m", None if pbp is None else pbp >= 0.05, fmt4(pbp, "{:+.2f} m"))
            ps = last(data, "Env/push_success_frac")
            chk("~300: push_success_frac >= 0.25", None if ps is None else ps >= 0.25, pct("Env/push_success_frac"))
            td = last(data, "Env/target_bin_push_distance_m")
            chk("~800: target_bin_push_distance_m >= 0.20 m", None if td is None else td >= 0.20 - 1e-4, fmt4(td, "{:.2f} m"))
            chk("~800: threshold <= 20 deg/s", None if th is None else th <= 20.0 + 1e-3, ths)
            chk("~800: bin_mass_level_kg >= 12.0 kg", None if lvg is None else lvg >= 12.0 - 1e-3, lvgs)
            lwv, rwv = last(data, "Env/ovh_wrapped_frac_left_s23"), last(data, "Env/ovh_wrapped_frac_right_s23")
            chk("flag: L and R wrapped >= 80%", None if lwv is None or rwv is None else min(lwv, rwv) >= 0.80,
                f"{pct2('Env/ovh_wrapped_frac_left_s23')}/{pct2('Env/ovh_wrapped_frac_right_s23')}%")
        elif is_tp3:
            # 2026-10-03 tilt_push3 (pre-registered): steady-hold rate gate +
            # +20 tilt_advance milestone bonus + tilt_overtime_is_timeout: false.
            def fmtv(v, f):
                return "-" if v is None else f.format(v)
            tl = last(data, "Env/termination_level")
            chk("start: termination_level >= 0.999 (20 rad/s)", None if tl is None else tl >= 0.999, fmtv(tl, "{:.4f}"))
            rta = last(data, "Episode/rew_tilt_advance")
            chk("start: Episode/rew_tilt_advance > 0 (+20 gate bonus)", None if rta is None else rta > 0.0, fmtv(rta, "{:+.3f}"))
            th = last(data, "Env/tilt_gate_max_rate_dps")
            ths = fmtv(th, "{:.1f} deg/s")
            pr = last(data, "Env/push_reached_frac")
            chk("~150: push_reached_frac >= 0.65 (no gate avoidance)", None if pr is None else pr >= 0.65, pct("Env/push_reached_frac"))
            chk("~150: threshold <= 32.4 deg/s", None if th is None else th <= 32.4 + 1e-3, ths)
            gp = last(data, "Env/grip_maintained_frac_push")
            chk("~150: grip_maintained_frac_push >= 0.55", None if gp is None else gp >= 0.55, pct("Env/grip_maintained_frac_push"))
            chk("~300: threshold <= 25 deg/s", None if th is None else th <= 25.0 + 1e-3, ths)
            pbp = last(data, "Env/push_bar_progress_m_push")
            chk("~300: push_bar_progress_m_push >= 0.05 m", None if pbp is None else pbp >= 0.05, fmtv(pbp, "{:+.2f} m"))
            pv = last(data, "Env/pelvis_fwd_vel_push")
            chk("~300: pelvis_fwd_vel_push > +0.05 m/s", None if pv is None else pv > 0.05, fmtv(pv, "{:+.2f} m/s"))
            ps = last(data, "Env/push_success_frac")
            chk("~300: push_success_frac >= 0.20", None if ps is None else ps >= 0.20, pct("Env/push_success_frac"))
            chk("~800: threshold <= 15 deg/s", None if th is None else th <= 15.0 + 1e-3, ths)
            td = last(data, "Env/target_bin_push_distance_m")
            chk("~800: target_bin_push_distance_m >= 0.20 m", None if td is None else td >= 0.20 - 1e-4, fmtv(td, "{:.2f} m"))
            chk("~800: grip_maintained_frac_push >= 0.70", None if gp is None else gp >= 0.70, pct("Env/grip_maintained_frac_push"))
            lwv, rwv = last(data, "Env/ovh_wrapped_frac_left_s23"), last(data, "Env/ovh_wrapped_frac_right_s23")
            chk("flag: L and R wrapped >= 80%", None if lwv is None or rwv is None else min(lwv, rwv) >= 0.80,
                f"{pct2('Env/ovh_wrapped_frac_left_s23')}/{pct2('Env/ovh_wrapped_frac_right_s23')}%")
        elif is_tp2:
            # 2026-10-03 tilt_push2 (pre-registered): bar/root min push progress,
            # in-band tilt braking, TILT->PUSH walk ramp, 0.10->1.00 m curriculum.
            def fmtv(v, f):
                return "-" if v is None else f.format(v)
            tl = last(data, "Env/termination_level")
            chk("start: termination_level >= 0.999 (20 rad/s)", None if tl is None else tl >= 0.999, fmtv(tl, "{:.4f}"))
            pr = last(data, "Env/push_reached_frac")
            chk("~150: push_reached_frac >= 0.70 (no gate avoidance)", None if pr is None else pr >= 0.70, pct("Env/push_reached_frac"))
            trp = last(data, "Env/tilt_rate_abs_dps_push")
            chk("~150: |tilt rate| in PUSH < 35 deg/s", None if trp is None else trp < 35.0, fmtv(trp, "{:.1f} deg/s"))
            rtt = last(data, "Episode/rew_termination")
            fr = None if rtt is None else -rtt
            frs = fmtv(None if fr is None else 100 * fr, "{:.1f}%")
            chk("~150: fail rate < 65%", None if fr is None else fr < 0.65, frs)
            chk("~300: fail rate < 35%", None if fr is None else fr < 0.35, frs)
            pbp = last(data, "Env/push_bar_progress_m_push")
            chk("~300: push_bar_progress_m_push >= 0.08 m", None if pbp is None else pbp >= 0.08, fmtv(pbp, "{:+.2f} m"))
            pv = last(data, "Env/pelvis_fwd_vel_push")
            chk("~300: pelvis_fwd_vel_push > +0.08 m/s", None if pv is None else pv > 0.08, fmtv(pv, "{:+.2f} m/s"))
            ps = last(data, "Env/push_success_frac")
            chk("~300: push_success_frac >= 0.25", None if ps is None else ps >= 0.25, pct("Env/push_success_frac"))
            td = last(data, "Env/target_bin_push_distance_m")
            chk("~800: target_bin_push_distance_m >= 0.25 m", None if td is None else td >= 0.25 - 1e-4, fmtv(td, "{:.2f} m"))
            gp = last(data, "Env/grip_maintained_frac_push")
            chk("~800: grip_maintained_frac_push >= 0.75", None if gp is None else gp >= 0.75, pct("Env/grip_maintained_frac_push"))
            lwv, rwv = last(data, "Env/ovh_wrapped_frac_left_s23"), last(data, "Env/ovh_wrapped_frac_right_s23")
            chk("flag: L and R wrapped >= 80%", None if lwv is None or rwv is None else min(lwv, rwv) >= 0.80,
                f"{pct2('Env/ovh_wrapped_frac_left_s23')}/{pct2('Env/ovh_wrapped_frac_right_s23')}%")
        elif is_tp1:
            # 2026-10-03 tilt_push1 (pre-registered): zero-shot 5100 = 96 % reach
            # PUSH, 100 % topple within ~0.4 s; smoke start fail 100 %, grip in
            # PUSH 0.31-0.36, progress 0.04 m, pelvis vx -0.07..-0.02 m/s.
            def fmtv(v, f):
                return "-" if v is None else f.format(v)
            tl = last(data, "Env/termination_level")
            chk("start: termination_level >= 0.999 (20 rad/s)", None if tl is None else tl >= 0.999, fmtv(tl, "{:.4f}"))
            pr = last(data, "Env/push_reached_frac")
            chk("start: push_reached_frac >= 0.85", None if pr is None else pr >= 0.85, pct("Env/push_reached_frac"))
            rtt = last(data, "Episode/rew_termination")
            fr = None if rtt is None else -rtt
            frs = fmtv(None if fr is None else 100 * fr, "{:.1f}%")
            chk("~100: fail rate < 60%", None if fr is None else fr < 0.60, frs)
            gp = last(data, "Env/grip_maintained_frac_push")
            chk("~100: grip_maintained_frac_push >= 0.6", None if gp is None else gp >= 0.60,
                pct("Env/grip_maintained_frac_push"))
            chk("~300: fail rate < 25%", None if fr is None else fr < 0.25, frs)
            pp = last(data, "Env/push_progress_m_push")
            chk("~300: push_progress_m_push >= 0.2 m", None if pp is None else pp >= 0.20, fmtv(pp, "{:.2f} m"))
            pv = last(data, "Env/pelvis_fwd_vel_push")
            chk("~300: pelvis_fwd_vel_push > +0.1 m/s", None if pv is None else pv > 0.10, fmtv(pv, "{:+.2f} m/s"))
            ps = last(data, "Env/push_success_frac")
            chk("~1000: push_success_frac >= 0.30", None if ps is None else ps >= 0.30, pct("Env/push_success_frac"))
            es = last(data, "Env/episode_success_frac")
            chk("~1000: episode_success_frac >= 0.25", None if es is None else es >= 0.25, pct("Env/episode_success_frac"))
            eh = last(data, "Env/episode_success_frac_m11_13")
            chk("~1000: success 11-13 kg >= 0.15", None if eh is None else eh >= 0.15,
                pct("Env/episode_success_frac_m11_13"))
            chk("flag: push_reached >= 0.5 (no gate avoidance)", None if pr is None else pr >= 0.50,
                pct("Env/push_reached_frac"))
            lwv, rwv = last(data, "Env/ovh_wrapped_frac_left_s23"), last(data, "Env/ovh_wrapped_frac_right_s23")
            chk("flag: L and R wrapped >= 80%", None if lwv is None or rwv is None else min(lwv, rwv) >= 0.80,
                f"{pct2('Env/ovh_wrapped_frac_left_s23')}/{pct2('Env/ovh_wrapped_frac_right_s23')}%")
        elif is_gt6:
            # 2026-10-02 grasp_tilt6: start = level reset 13 -> 11 kg (U[9,11]);
            # 4950 @20 rad/s deterministic: 9-11 kg 97-100 %, 11-12 35 %, 12-13 0 %.
            tl = last(data, "Env/termination_level")
            chk("start: termination_level = 1.0 (20 rad/s)", None if tl is None else tl >= 0.999,
                "-" if tl is None else f"{tl:.4f}")
            chk("~30: bin_mass_level_kg >= 12", None if lvg is None else lvg >= 12.0 - 1e-3, lvgs)
            chk("~30: tilt_success_frac >= 0.50", None if tsg is None else tsg >= 0.50, pct("Env/tilt_success_frac"))
            chk("~150: bin_mass_level_kg >= 13 (cap)", None if lvg is None else lvg >= 13.0 - 1e-3, lvgs)
            chk("~300: tilt success >= 0.50 at level 13",
                None if tsg is None or lvg is None else (lvg >= 13.0 - 1e-3 and tsg >= 0.50),
                f"{pct('Env/tilt_success_frac')} at {lvgs} (+ offline deterministic sweep)")
        elif is_gt5:
            # 2026-10-02 grasp_tilt5: start = level reset 8 -> 6 kg (U[4,6]); play
            # of 4700 within 4 s: L6 100 %, L7 98 %, L8 55 %, L9 7 % success.
            chk("~30: bin_mass_level_kg >= 8", None if lvg is None else lvg >= 8.0 - 1e-3, lvgs)
            chk("~30: tilt_success_frac >= 0.30", None if tsg is None else tsg >= 0.30, pct("Env/tilt_success_frac"))
            chk("~300: bin_mass_level_kg >= 9 (tilt4 wall)", None if lvg is None else lvg >= 9.0 - 1e-3, lvgs)
            chk("~1000: bin_mass_level_kg >= 11", None if lvg is None else lvg >= 11.0 - 1e-3, lvgs)
        elif is_gt4:
            # 2026-10-02 grasp_tilt4: start = level reset to 3 kg (U[2,3]); play of
            # m10 4600 succeeded 83 % at 2-3 kg, 0 % at 4-6 kg.
            chk("~50: tilt_success_frac >= 0.30", None if tsg is None else tsg >= 0.30, pct("Env/tilt_success_frac"))
            chk("~50: bin_mass_level_kg >= 4", None if lvg is None else lvg >= 4.0 - 1e-3, lvgs)
            chk("~300: bin_mass_level_kg >= 6", None if lvg is None else lvg >= 6.0 - 1e-3, lvgs)
        elif is_m10:
            # 2026-10-02 m10: start (grasp_tilt3 2300) moment +1.8 Nm, grip 1.00.
            chk("~50: tilt_moment_Nm_tilt > +4 Nm", None if mo is None else mo > 4.0,
                "-" if mo is None else f"{mo:+.1f} Nm")
            chk("~50: grip_maintained_frac_tilt > 0.9", None if gmg is None else gmg > 0.9,
                pct("Env/grip_maintained_frac_tilt"))
            chk("~150: tilt_back_max_deg_tilt > 3 deg", None if tmg is None else tmg > 3.0,
                "-" if tmg is None else f"{tmg:+.1f} deg")
            chk("~300: tilt_success_frac > 0.05", None if tsg is None else tsg > 0.05, pct("Env/tilt_success_frac"))
        else:
            # 2026-10-02: milestones relative to the measured start (smoke from 1700:
            # moment +3..+5 Nm, grip 0.61, tilt max ~1.5 deg), so "moment > 0" does
            # not pass trivially; ~100 it = learn to hold without flipping the push.
            chk("~100: grip_maintained_frac_tilt > 0.8", None if gmg is None else gmg > 0.8,
                pct("Env/grip_maintained_frac_tilt"))
            chk("~100: tilt_moment_Nm_tilt > +3 Nm", None if mo is None else mo > 3.0,
                "-" if mo is None else f"{mo:+.1f} Nm")
            chk("~300: tilt_back_max_deg_tilt > 3 deg", None if tmg is None else tmg > 3.0,
                "-" if tmg is None else f"{tmg:+.1f} deg")
            chk("~300: tilt_success_frac > 0.05", None if tsg is None else tsg > 0.05, pct("Env/tilt_success_frac"))
        # 2026-10-03 tilt_push1/2/3: fixed U[6,13] kg (no level-ups) -> no level checks.
        # 2026-10-03 grasp_tilt7: same fixed range.
        if not (is_gt5 or is_gt6 or is_tp1 or is_gt7):
            chk("~1000: bin_mass_level_kg >= 10", None if lvg is None else lvg >= 10.0 - 1e-3, lvgs)
        if not (is_gt6 or is_tp1 or is_gt7):  # tilt6 caps the level at 13 kg
            chk("~2000: bin_mass_level_kg >= 15", None if lvg is None else lvg >= 15.0 - 1e-3, lvgs)
        tpg = last(data, "Env/bin_topple_frac_all")
        chk("always: bin_topple_frac < 2%", None if tpg is None else tpg < 0.02, pct("Env/bin_topple_frac_all"))
        bcg = last(data, "Env/body_bin_contact_frac_tilt")
        if is_gt6 or is_tp1 or is_gt7:
            # 2026-10-02: the sensor is unfiltered; in tilt5 4950 this was 100 %
            # left-arm <-> torso self-contact, not the bin -> watch only.
            print(f"  [info] watch: body contact >50N (= self-contact)    {pct('Env/body_bin_contact_frac_tilt')}")
        else:
            chk("always: body contact >50N < 10%", None if bcg is None else bcg < 0.10,
                pct("Env/body_bin_contact_frac_tilt"))
        if is_gt4 or is_gt5 or is_gt6 or is_tp1 or is_gt7:
            chk("always: grip_maintained_frac_tilt >= 85%", None if gmg is None else gmg >= 0.85,
                pct("Env/grip_maintained_frac_tilt"))
        if is_gt5 or is_gt6 or is_gt7:
            if not is_gt7:  # 2026-10-03: grasp_tilt7 checks fail < 20 % above
                # 2026-10-02 grasp_tilt5: fail rate = -Episode/rew_termination (a 4 s
                # window starts to expose the >= 7-8 kg grip losses; 4700 play at
                # level 8: 4 % within 4 s, 38 % within 5 s).
                rt = last(data, "Episode/rew_termination")
                chk("always: termination (fail) rate < 5%", None if rt is None else -rt < 0.05,
                    "-" if rt is None else f"{-100 * rt:.1f}%")
            # tilt4's collapse signature: success >= 30 % (10-pt mean) at some
            # point, then < 10 % over the whole last 100 iterations.
            sp = data.get("Env/tilt_success_frac", [])
            roll = [(sp[i][0], sum(v for _, v in sp[max(0, i - a.win + 1):i + 1]) / (i - max(0, i - a.win + 1) + 1))
                    for i in range(len(sp))]
            peak = max((m for _, m in roll), default=None)
            rec = [m for s, m in roll if s > lastc - 100]
            coll = None if peak is None or peak < 0.30 or lastc - first < 100 else max(rec) < 0.10
            chk("flag: no collapse (>=30% then <10% for 100 it)", None if coll is None else not coll,
                "-" if peak is None else f"peak {100 * peak:.0f}%, last-100-it max "
                f"{'-' if not rec else f'{100 * max(rec):.0f}%'}")
        if is_gt4 or is_gt5 or is_gt6 or is_tp1 or is_gt7:
            # red flags (bin_tilt_back 20 must not undercut the hold)
            cl, cr = last(data, "Env/closure_left_tilt"), last(data, "Env/closure_right_tilt")
            chk("flag: closure L/R TILT >= 42% (gate 40%)",
                None if cl is None or cr is None else min(cl, cr) >= 0.42,
                f"{pct2('Env/closure_left_tilt')}/{pct2('Env/closure_right_tilt')}%")
        if is_gt4 or is_gt5 or is_gt6:  # tilt_push1 checks L / R wrapped >= 80 % above
            bw = last(data, "Env/ovh_both_wrapped_frac_s23")
            chk("flag: BOTH wrapped >= 90%", None if bw is None else bw >= 0.90, pct("Env/ovh_both_wrapped_frac_s23"))
        print()
    if not is_gt3:  # TILT-only runs have no GRASP stage (grasp-success checks n/a)
        print(f"TILT1 CHECKS ({lastc - first} it since resume; ~300 / ~1000 it milestones):")
        chk("~300: grasp success >= 50% of BASE", None if gs is None or not gb else gs >= 0.5 * gb,
            f"now {pct('Env/grasp_success_frac')} vs base {'-' if gb is None else f'{100 * gb:.1f}%'}")
        gm = last(data, "Env/grip_maintained_frac_tilt")
        chk("~300: grip_maintained_frac_tilt > 0.6", None if gm is None else gm > 0.6, pct("Env/grip_maintained_frac_tilt"))
        ib = last(data, "Env/tilt_in_band_frac_tilt")
        chk("~1000: tilt_in_band_frac_tilt > 0.1", None if ib is None else ib > 0.1, pct("Env/tilt_in_band_frac_tilt"))
        ts = last(data, "Env/tilt_success_frac")
        chk("~1000: tilt_success_frac > 0", None if ts is None else ts > 0, pct("Env/tilt_success_frac"))
        tp = last(data, "Env/bin_topple_frac_all")
        chk("always: bin_topple_frac ~0 (<1%)", None if tp is None else tp < 0.01, pct("Env/bin_topple_frac_all"))
        print()

    # ---- 2. per-hand
    print("=" * 30 + " LEFT vs RIGHT (NOW, BASE in brackets) " + "=" * 11)
    hw = max(len(h[0]) for h in HANDS)
    print(f"{'metric':<{hw}} | {'LEFT':>15} | {'RIGHT':>15} | target")
    print("-" * 80)
    for label, lt, rt, sc, fmt, d, tgt in HANDS:
        def cell(t):
            v, b = last(data, t), last(bdata, t)
            s = fmt.format(v * sc) if v is not None else "-"
            return s + (f" [{fmt.format(b * sc)}]" if b is not None else "")
        print(f"{label:<{hw}} | {cell(lt):>15} | {cell(rt):>15} | {tgt}")
    print()

    # ---- 3. trend table
    delta = nice(max(1, (lastc - first) / max(1, a.cols - 1)))
    cols, c = [], lastc
    while c > first and len(cols) < a.cols - 1:
        cols.append(c); c = (c // delta) * delta if c % delta else c - delta
    cols.append(first); cols = sorted(set(cols))

    def at(t, col):
        pts = [v for s, v in data[t] if s <= col][-a.win:]
        return sum(pts) / len(pts) if pts else None

    print("=" * 30 + f" TREND (col spacing ~{delta}, mean of last {a.win} pts) " + "=" * 5)
    # 2026-10-03 tilt_push1/2/3 & full1: PUSH rows appended under a separator.
    # 2026-10-03 grasp_tilt7 / tilt_push3 / full1: GATE rows likewise; a str row = section label.
    rows = (
        (["APPROACH DR"] + METRICS_APPROACH_DR if has_spawn_dr else [])
        + METRICS
        + (["PUSH"] + METRICS_PUSH if (is_tp1 or is_full) else [])
        + (["GATE"] + METRICS_TILT7 if (is_gt7 or is_tp3 or is_full) else [])
    )
    lw = max(len(m[1]) for m in rows if not isinstance(m, str))
    hdr = f"{'metric':<{lw}} | " + " ".join(f"{c:>8}" for c in cols) + " | exp  target      trend"
    print(hdr); print("-" * len(hdr))
    for row in rows:
        if isinstance(row, str):
            print("-" * 30 + f" {row} " + "-" * (len(hdr) - 32 - len(row))); continue
        t, label, sc, fmt, d, tgt = row
        if t not in data:
            print(f"{label:<{lw}} | (tag {t} not logged)"); continue
        vals = [at(t, c) for c in cols]
        cells = " ".join(f"{(fmt.format(v * sc) if v is not None else '-'):>8}" for v in vals)
        v0, v1 = vals[0], vals[-1]
        tol = 2 * 10 ** -int(fmt.split(".")[1][0])  # 2 units of the last shown digit
        if v0 is None or v1 is None or abs(v1 - v0) * sc <= tol + 0.03 * max(abs(v0), abs(v1)) * sc: tr = "flat"
        else: tr = "GOOD" if (v1 > v0) == (d == "+") else "bad"
        print(f"{label:<{lw}} | {cells} | {'up' if d == '+' else 'down':<4} {tgt:<11} {tr}")
    print("\nNote: success / grasp_* _frac / closure TILT / gate closure rows exist from reach_grasp_tilt1 on; the BASE run shows - for them.")


if __name__ == "__main__":
    main()
