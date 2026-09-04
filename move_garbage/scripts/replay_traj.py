# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PASS 2: replay a logged rollout trajectory and record a clean video.

Reads the .npz produced by ``play_bin_push.py`` (pass 1) — per-step root pose +
joint positions for the G1 and the bin — and replays it KINEMATICALLY in a
standalone SimulationContext scene, capturing with the proven vidtest_d camera
(raw UsdGeom.Camera + omni.replicator, driven by sim.step(render=True) — the ONE
capture technique that works in this Isaac Sim build).

Kinematic = we write root pose + joint state each frame and DON'T run physics
control; the recorded trajectory already encodes the trained policy's motion.

Run:
    CUDA_VISIBLE_DEVICES=3 conda run --no-capture-output -n isaacsim5.1 python -u \\
        move_garbage/scripts/replay_traj.py --headless --enable_cameras \\
        --traj /tmp/bin_traj.npz --out /tmp/bin_rollout.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--traj", type=str, default="/tmp/bin_traj.npz")
parser.add_argument("--out", type=str, default="/tmp/bin_rollout.mp4")
parser.add_argument("--fps", type=int, default=25)
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--stride", type=int, default=2, help="Render every Nth trajectory frame.")
parser.add_argument(
    "--force_exit_after_write",
    action="store_true",
    help="Exit immediately after closing the video writer, bypassing Isaac Sim shutdown.",
)
parser.add_argument("--warmup", type=int, default=10,
                    help="RTX denoiser warmup steps before recording. Higher = cleaner "
                         "start; 0-2 = strong 'analog-TV warmup' grain that resolves on camera.")
parser.add_argument("--bin_yaw_deg", type=float, default=0.0,
                    help="Extra yaw (deg, CCW about world +Z) applied to the logged bin "
                         "pose at replay time — preview a new spawn orientation without a "
                         "fresh rollout.")
parser.add_argument("--debug_markers", action="store_true",
                    help="Draw reward-debug markers: RED at the grasp-relative stop target, "
                         "GREEN at the two grasp-handle targets. Uses the logged bin_pos body "
                         "positions so the markers track the bin every frame.")
parser.add_argument("--proposed_stop", action="store_true",
                    help="Also draw the PROPOSED grasp-relative stop target (BLUE ball + "
                         "acceptance ring): handle_mid + standoff*face_normal at pelvis height.")
parser.add_argument("--stop_standoff", type=float, default=0.42,
                    help="Standoff (m) back from handle midpoint along the grasp-face normal "
                         "for the proposed stop target.")
parser.add_argument("--stop_pos_std", type=float, default=0.20,
                    help="Along-axis Gaussian std (m); the debug ring is a 1.5x reference contour.")
parser.add_argument("--pregrasp_arc_height", type=float, default=0.10,
                    help="Height (m) above each handle at the PEAK of the triangular pre-grasp "
                         "hump (Run 20). Matches _one_hand_reach's pregrasp_arc_height.")
parser.add_argument("--pregrasp_arc_peak_frac", type=float, default=0.75,
                    help="Fraction of pregrasp_blend_dist where the hump peaks. Matches "
                         "_one_hand_reach's pregrasp_arc_peak_frac.")
parser.add_argument("--pregrasp_blend_dist", type=float, default=0.30,
                    help="Horizontal distance (m) at/beyond which the pregrasp waypoint sits at "
                         "the true handle height (hump=0); the hump rises and falls inside this "
                         "radius, peaking at pregrasp_arc_peak_frac. Matches _one_hand_reach.")
parser.add_argument("--grasp_finger_close_horiz_dist", type=float, default=0.04,
                    help="Horizontal tolerance (m) for the finger-close gate (Run 19 decomposition).")
parser.add_argument("--grasp_finger_close_vert_dist", type=float, default=0.03,
                    help="Vertical tolerance (m) for the finger-close gate (Run 19 decomposition).")
parser.add_argument("--palm_grasp_point_offset", type=float, default=0.0464,
                    help="Offset (m) from the palm-link origin into the finger-closing cavity, "
                         "along the per-hand palm_grasp_axis (60/40 blend toward the thumb's "
                         "curled position). Matches env's _palm_grasp_points config default.")
parser.add_argument("--show_palm_markers", action="store_true",
                    help="Draw the ACTUAL virtual grasp point for each palm (offset 5cm into the "
                         "finger cavity along its contact axis, matching _palm_grasp_points), colored "
                         "by whether it currently passes the decomposed horiz+vert+twist finger-close "
                         "gate: GREEN=ready, YELLOW=close but not ready, MAGENTA=far. Also draws a thin "
                         "white line from each palm to its live top-down arc waypoint, so you can see "
                         "whether the palm is tracking the arc down onto the rim or hovering above it.")
parser.add_argument("--show_full_arc", action="store_true",
                    help="Draw the ENTIRE episode's palm path and intended top-down arc waypoint path "
                         "as static point trails, visible all at once in every frame (instead of one "
                         "point per hand that only shows the CURRENT frame). Small dots = actual palm "
                         "grasp-point path over time (dimmer green/red for left/right); small white "
                         "dots = the intended waypoint arc over the same window. Implies "
                         "--show_palm_markers's geometry (uses the same targets/offsets) but is drawn "
                         "independently and works even without --show_palm_markers.")
parser.add_argument("--cam_dir", type=float, nargs=3, default=[-1.0, 1.0, 0.6],
                    help="Camera view direction (offset from target). Default diagonal 3/4. "
                         "Side view (down the handle axis, both arms visible): 0 1 0.25")
parser.add_argument("--cam_zoom", type=float, default=1.0,
                    help="Multiplier on camera standoff distance; <1 zooms in (e.g. 0.3 for a "
                         "close-up on the hands to validate small markers).")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

app_launcher = AppLauncher(args)
sim_app = app_launcher.app

import os

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg

import omni.replicator.core as rep
from pxr import Gf, Usd, UsdGeom

_REPO = Path(__file__).resolve().parents[2]
BIN_USD = str(_REPO / "move_garbage" / "data" / "objects" / "bin"
              / "RecyclingContainerWithWheels_A01_Blue_01_dynamic.usd")
G1_USD = str(_REPO / "gr00t" / "rl" / "data" / "robots" / "g1" / "g1_43dof_s3.usda")


def _quat_apply_wxyz(q, v):
    """Rotate 3-vector v by quaternion q (wxyz). Pure numpy, no batching needed here."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    qvec = np.array([x, y, z])
    cross1 = np.cross(qvec, v)
    cross2 = np.cross(qvec, cross1)
    return v + 2.0 * w * cross1 + 2.0 * cross2


def _quat_mul_wxyz(q1, q2):
    """Hamilton product q1*q2, both wxyz."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def _quat_apply_wxyz_batch(q, v):
    """Batched version of _quat_apply_wxyz: q is (T,4), v is (T,3) or (3,) broadcast."""
    w = q[:, 0:1]
    qvec = q[:, 1:4]
    if v.ndim == 1:
        v = np.broadcast_to(v, (q.shape[0], 3))
    cross1 = np.cross(qvec, v)
    cross2 = np.cross(qvec, cross1)
    return v + 2.0 * w * cross1 + 2.0 * cross2


def _quat_angle_error_wxyz(q1, q2):
    """Angular distance (radians) between two wxyz quaternions."""
    dot = np.clip(np.abs(np.dot(q1, q2)), -1.0, 1.0)
    return 2.0 * np.arccos(dot)


def _arc_hump(horiz, blend_dist, peak_frac):
    """Smooth (half-cosine) hump in horiz-space: 0 at horiz=0 and
    horiz>=blend_dist, 1 at horiz=peak_frac*blend_dist, zero slope at both ends
    and at the peak. Mirrors _one_hand_reach's Run 20 pregrasp arc exactly
    (numpy, scalar-or-array safe)."""
    s = np.clip(horiz / blend_dist, 0.0, 1.0)
    rising_phase = np.clip(s / peak_frac, 0.0, 1.0)
    falling_phase = np.clip((1.0 - s) / (1.0 - peak_frac), 0.0, 1.0)
    phase = np.where(s < peak_frac, rising_phase, falling_phase)
    return 0.5 * (1.0 - np.cos(np.pi * phase))


def _look_at(eye, target, up=Gf.Vec3d(0, 0, 1)):
    fn = (target - eye).GetNormalized()
    r = Gf.Cross(fn, up).GetNormalized()
    u = Gf.Cross(r, fn).GetNormalized()
    m = Gf.Matrix4d(1.0)
    m.SetRow(0, Gf.Vec4d(r[0], r[1], r[2], 0.0))
    m.SetRow(1, Gf.Vec4d(u[0], u[1], u[2], 0.0))
    m.SetRow(2, Gf.Vec4d(-fn[0], -fn[1], -fn[2], 0.0))
    m.SetRow(3, Gf.Vec4d(eye[0], eye[1], eye[2], 1.0))
    return m


def _reorder(src_names, dst_names, arr_last_axis):
    """Return index list mapping dst order -> src columns (for joint reorder)."""
    src = {n: i for i, n in enumerate(src_names)}
    return [src[n] for n in dst_names]


def main() -> None:
    d = np.load(args.traj, allow_pickle=True)
    T = d["robot_root"].shape[0]
    print(f"[replay] loaded {args.traj}: {T} frames")

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 200.0))
    sim_utils.GroundPlaneCfg().func("/World/GroundPlane", sim_utils.GroundPlaneCfg())
    dome = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.98, 0.95, 0.88))
    dome.func("/World/DomeLight", dome)

    # Articulations (kinematic replay — fix_root so we set pose directly, and
    # zero-stiffness actuators so joint targets we write aren't fought).
    # Free root (no fix_root_link — our bin root is an Xform w/o RigidBodyAPI so
    # Isaac Lab can't auto-fix it). Kinematic replay works anyway: we write root
    # pose + joint state + zero velocity every frame, and disable gravity so the
    # bodies don't drift between our writes.
    g1_cfg = ArticulationCfg(
        prim_path="/World/G1",
        spawn=sim_utils.UsdFileCfg(usd_path=G1_USD, activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True)),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0, 0, 0.75)),
        actuators={"all": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=0.0, damping=0.0)},
    )
    bin_cfg = ArticulationCfg(
        prim_path="/World/Bin",
        spawn=sim_utils.UsdFileCfg(usd_path=BIN_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True)),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0, 0, 0.1)),
        actuators={"wheels": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=0.0, damping=0.0)},
    )
    g1 = Articulation(g1_cfg)
    bin_art = Articulation(bin_cfg)

    import omni.usd
    stage = omni.usd.get_context().get_stage()
    cam = UsdGeom.Camera.Define(stage, "/World/ReplayCam")
    cam.CreateFocalLengthAttr().Set(24.0)
    cam.CreateHorizontalApertureAttr().Set(20.955)
    cam.CreateClippingRangeAttr().Set(Gf.Vec2f(0.05, 1000.0))
    cam_xf = UsdGeom.Xformable(cam.GetPrim())
    cam_xf.ClearXformOpOrder()
    cam_op = cam_xf.AddTransformOp()

    render_product = rep.create.render_product("/World/ReplayCam", resolution=(args.width, args.height))
    rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb_annot.attach([render_product])
    import imageio.v2 as imageio

    # ---- Reward-debug marker spheres (RED=stop anchor, GREEN=grasp handles) ----
    # Colored, unlit spheres whose translate we rewrite each frame from the logged
    # world positions, so they track the bin as it (barely) moves.
    from pxr import UsdShade, Sdf

    def _make_marker(path, rgb, radius):
        sph = UsdGeom.Sphere.Define(stage, path)
        sph.CreateRadiusAttr().Set(radius)
        sph.CreateDisplayColorAttr().Set([Gf.Vec3f(*rgb)])
        xf = UsdGeom.Xformable(sph.GetPrim())
        xf.ClearXformOpOrder()
        op = xf.AddTranslateOp()
        # Emissive-ish unlit look via a simple constant-color preview surface.
        mat = UsdShade.Material.Define(stage, path + "/Mat")
        sh = UsdShade.Shader.Define(stage, path + "/Mat/S")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
        sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.9)
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI(sph.GetPrim()).Bind(mat)
        return op

    def _make_ring(path, rgb, radius, tube=0.02):
        """A flat ring (thin torus) on the ground to show an acceptance radius."""
        tor = UsdGeom.Cylinder.Define(stage, path)  # thin disc outline via scaled cylinder
        tor.CreateRadiusAttr().Set(radius)
        tor.CreateHeightAttr().Set(0.01)
        tor.CreateAxisAttr().Set("Z")
        tor.CreateDisplayColorAttr().Set([Gf.Vec3f(*rgb)])
        tor.CreateDisplayOpacityAttr().Set([0.35])
        xf = UsdGeom.Xformable(tor.GetPrim())
        xf.ClearXformOpOrder()
        return xf.AddTranslateOp()

    markers = None
    if args.debug_markers or args.show_palm_markers or args.show_full_arc:
        bin_body_names = [str(x) for x in d["bin_body_names"]]
        bin_pos_all = d["bin_pos"]  # (T, nB, 3) world positions of each bin body
        gi_l = bin_body_names.index("grasp_target_left")
        gi_r = bin_body_names.index("grasp_target_right")
        gi_body = bin_body_names.index("body")
    if args.debug_markers:
        # RED stop zone = the ACTUAL grasp-relative stop target the reward uses
        # (handle_mid + stop_standoff*face_normal), NOT the old bin-centered 0.9m
        # circle. The ring is a 1.5x along-axis reward-std reference, not the exact
        # anisotropic 8 cm by 12 cm transition gate. A raised red
        # ball marks the target at pelvis height (visible, not occluded).
        # GREEN balls = the two grasp handle targets.
        STOP_RADIUS = args.stop_pos_std * 1.5
        markers = {
            "grasp_l": (_make_marker("/World/DbgGraspL", (0.05, 0.9, 0.05), 0.05), gi_l, 0.0),
            "grasp_r": (_make_marker("/World/DbgGraspR", (0.05, 0.9, 0.05), 0.05), gi_r, 0.0),
        }
        stop_ball_op = _make_marker("/World/DbgStopBall", (0.9, 0.05, 0.05), 0.07)
        stop_ring_op = _make_ring("/World/DbgStopRing", (0.9, 0.1, 0.1), STOP_RADIUS)
        print(f"[replay] debug markers ON: RED ball + {STOP_RADIUS:.2f}m ring = stop target; "
              f"GREEN = grasp handles (idx L={gi_l} R={gi_r})")

    # Shared geometry constants (index lookups, axes, cross-assignment) used by
    # BOTH --show_palm_markers (live per-frame markers) and --show_full_arc
    # (static whole-episode trail). Built once here so the two views can never
    # disagree with each other about which body/axis/offset they're using.
    palm_geom = None
    if args.show_palm_markers or args.show_full_arc:
        robot_body_names = [str(x) for x in d["robot_body_names"]]
        left_palm_bi = robot_body_names.index("left_hand_palm_link")
        right_palm_bi = robot_body_names.index("right_hand_palm_link")
        # Cross-assignment (see scene_creation_callback / TRAINING_LOG July 16):
        # LEFT hand's target is the grasp_target_RIGHT body (no extra offset);
        # RIGHT hand's target is the grasp_target_LEFT body, with a local
        # Rx(180deg) rotation offset ((0,1,0,0) xyzw = (0,0,1,0) wxyz) applied on
        # top so the observed target orientation is palm-down for BOTH hands.
        # OffsetCfg.rot is documented as (w,x,y,z); the env applies
        # OffsetCfg(rot=(0.0, 1.0, 0.0, 0.0)) = Rx(180deg), NOT Ry.
        RX180_WXYZ = np.array([0.0, 1.0, 0.0, 0.0])
        # Grasp-cavity axis (2026-08-01 fix, biased toward thumb per video
        # review): per-hand axis (mostly +X, biased along Y toward the thumb's
        # curl direction — opposite sign per hand) — NOT the face-normal
        # contact axis (+-Y) used elsewhere for palm-down alignment. Matches
        # env's left/right_palm_grasp_axis. See _palm_grasp_axes docstring.
        palm_geom = dict(
            left_palm_bi=left_palm_bi, right_palm_bi=right_palm_bi,
            left_target_gi=gi_r,     # left palm's target body = grasp_target_right
            right_target_gi=gi_l,    # right palm's target body = grasp_target_left
            left_axis=np.array([0.9717, -0.2356, -0.0181]),
            right_axis=np.array([0.9717, 0.2359, 0.0155]),
            offset=args.palm_grasp_point_offset,
            rx180=RX180_WXYZ,
        )

    palm_dbg = None
    if args.show_palm_markers:
        pg0 = palm_geom
        # Live top-down arc waypoint marker (WHITE, small) + a thin connecting
        # line (approximated by a thin cylinder) from each palm's grasp point to
        # its waypoint, so hovering-above-target is visually obvious frame-to-frame.
        waypoint_l_op = _make_marker("/World/DbgWaypointL", (1.0, 1.0, 1.0), 0.025)
        waypoint_r_op = _make_marker("/World/DbgWaypointR", (1.0, 1.0, 1.0), 0.025)
        # Palm grasp-point balls: color rewritten every frame (ready/close/far),
        # so create as plain spheres with a per-frame color override instead of
        # a fixed material.
        palm_l_prim = UsdGeom.Sphere.Define(stage, "/World/DbgPalmL")
        palm_l_prim.CreateRadiusAttr().Set(0.03)
        palm_l_xf = UsdGeom.Xformable(palm_l_prim.GetPrim())
        palm_l_xf.ClearXformOpOrder()
        palm_l_op = palm_l_xf.AddTranslateOp()
        palm_r_prim = UsdGeom.Sphere.Define(stage, "/World/DbgPalmR")
        palm_r_prim.CreateRadiusAttr().Set(0.03)
        palm_r_xf = UsdGeom.Xformable(palm_r_prim.GetPrim())
        palm_r_xf.ClearXformOpOrder()
        palm_r_op = palm_r_xf.AddTranslateOp()

        palm_dbg = dict(
            pg0,
            waypoint_l_op=waypoint_l_op, waypoint_r_op=waypoint_r_op,
            palm_l_prim=palm_l_prim, palm_l_op=palm_l_op,
            palm_r_prim=palm_r_prim, palm_r_op=palm_r_op,
        )
        print(f"[replay] palm markers ON: colored ball = actual virtual grasp point "
              f"(palm + {pg0['offset']}m into finger cavity) per hand — GREEN=finger-close-ready, "
              f"YELLOW=close but gate not satisfied, MAGENTA=far. WHITE ball = live top-down "
              f"arc waypoint (should coincide with the GREEN handle target once close; if the "
              f"colored ball sits well above the green target while horizontally centered, "
              f"that's the July 2026 hovering-above-target bug this run is meant to fix).")

    prop = None
    if args.proposed_stop:
        # BLUE = proposed grasp-relative stop target + its acceptance ring.
        prop_ball_op = _make_marker("/World/DbgPropBall", (0.1, 0.3, 0.95), 0.08)
        prop_ring_op = _make_ring("/World/DbgPropRing", (0.1, 0.3, 0.95), args.stop_pos_std * 1.5)
        prop = (prop_ball_op, prop_ring_op)
        print(f"[replay] PROPOSED stop target ON (BLUE): standoff={args.stop_standoff}m "
              f"along grasp-face normal, accept ring={args.stop_pos_std*1.5:.2f}m")

    sim.reset()

    # Joint-name reorder maps (traj order -> this scene's articulation order).
    g1_traj_jn = [str(x) for x in d["robot_joint_names"]]
    bin_traj_jn = [str(x) for x in d["bin_joint_names"]]
    g1_scene_jn = list(g1.data.joint_names)
    bin_scene_jn = list(bin_art.data.joint_names)
    g1_map = _reorder(g1_traj_jn, g1_scene_jn, None)     # scene[i] = traj[g1_map[i]]
    bin_map = _reorder(bin_traj_jn, bin_scene_jn, None)
    device = sim.device

    robot_root = d["robot_root"]  # (T,7) pos+wxyz
    robot_jpos = d["robot_jpos"]  # (T,43)
    bin_root = d["bin_root"].copy()
    bin_jpos = d["bin_jpos"]

    if palm_dbg is not None or args.show_full_arc:
        # Per-body world pose arrays (only loaded when needed — large for long
        # trajectories). robot_quat/bin_quat are wxyz as Isaac Lab stores them.
        robot_pos_all = d["robot_pos"]    # (T, nB_robot, 3)
        robot_quat_all = d["robot_quat"]  # (T, nB_robot, 4) wxyz
        bin_quat_all = d["bin_quat"]      # (T, nB_bin, 4) wxyz

    # Optional: preview a new bin spawn orientation without re-running the policy.
    # Rotate the logged bin pose by an extra yaw about +Z, pivoting about the
    # bin's OWN center (its t0 world position — the poses carry an env-origin
    # offset of ~(10,-10), so we must NOT pivot about the world origin). This is
    # a rotation-in-place: orientation quat is pre-multiplied by the yaw quat and
    # position is rotated about the pivot. The G1 trajectory is untouched.
    if abs(args.bin_yaw_deg) > 1e-6:
        a = np.deg2rad(args.bin_yaw_deg)
        ca, sa = np.cos(a), np.sin(a)
        cx, cy = float(bin_root[0, 0]), float(bin_root[0, 1])  # pivot = bin t0 center
        px, py = bin_root[:, 0].copy() - cx, bin_root[:, 1].copy() - cy
        bin_root[:, 0] = cx + ca * px - sa * py
        bin_root[:, 1] = cy + sa * px + ca * py
        # quat multiply  q_yaw * q_bin  (wxyz);  q_yaw = (cos a/2, 0, 0, sin a/2)
        qw, qz = np.cos(a / 2), np.sin(a / 2)
        bw, bx, by, bz = (bin_root[:, 3].copy(), bin_root[:, 4].copy(),
                          bin_root[:, 5].copy(), bin_root[:, 6].copy())
        bin_root[:, 3] = qw * bw - qz * bz
        bin_root[:, 4] = qw * bx - qz * by
        bin_root[:, 5] = qw * by + qz * bx
        bin_root[:, 6] = qw * bz + qz * bw
        print(f"[replay] applied +{args.bin_yaw_deg}deg yaw to bin pose (G1 unchanged)")

    if args.show_full_arc:
        # A SIMPLE synthetic sweep per hand, from the palm's OWN starting pose
        # (frame 0 — wherever it actually rests at episode start) straight to
        # its target grasp point. Not derived from the noisy in-between policy
        # trajectory (hard to read, and only shows what THIS rollout happened to
        # do) — just the two fixed endpoints. XY interpolates linearly start ->
        # target; height follows the exact same hump _one_hand_reach rewards
        # (0 at the start, 0 at the target, peaked in between), so this draws
        # the CLEAN intended shape of the arc: rises off the start pose, comes
        # back down to touch the rim exactly at the target.
        pg = palm_geom
        left_target_pos0 = bin_pos_all[0, pg["left_target_gi"], :].astype(float)
        right_target_pos0 = bin_pos_all[0, pg["right_target_gi"], :].astype(float)
        left_palm_pos0 = robot_pos_all[0, pg["left_palm_bi"], :].astype(float)
        left_palm_quat0 = robot_quat_all[0, pg["left_palm_bi"], :].astype(float)
        right_palm_pos0 = robot_pos_all[0, pg["right_palm_bi"], :].astype(float)
        right_palm_quat0 = robot_quat_all[0, pg["right_palm_bi"], :].astype(float)
        left_start0 = left_palm_pos0 + pg["offset"] * _quat_apply_wxyz(left_palm_quat0, pg["left_axis"])
        right_start0 = right_palm_pos0 + pg["offset"] * _quat_apply_wxyz(right_palm_quat0, pg["right_axis"])

        n_pts = 60
        frac = np.linspace(0.0, 1.0, n_pts)  # 0 = start pose, 1 = target
        start_horiz_l = float(np.linalg.norm(left_start0[:2] - left_target_pos0[:2]))
        start_horiz_r = float(np.linalg.norm(right_start0[:2] - right_target_pos0[:2]))
        horiz_sweep_l = start_horiz_l * (1.0 - frac)
        horiz_sweep_r = start_horiz_r * (1.0 - frac)
        hump_sweep_l = _arc_hump(horiz_sweep_l, args.pregrasp_blend_dist, args.pregrasp_arc_peak_frac)
        hump_sweep_r = _arc_hump(horiz_sweep_r, args.pregrasp_blend_dist, args.pregrasp_arc_peak_frac)

        def _sweep_points(start_pos, target_pos, hump_sweep):
            pts = np.zeros((n_pts, 3))
            pts[:, 0] = start_pos[0] + (target_pos[0] - start_pos[0]) * frac
            pts[:, 1] = start_pos[1] + (target_pos[1] - start_pos[1]) * frac
            pts[:, 2] = target_pos[2] + args.pregrasp_arc_height * hump_sweep
            return pts

        left_arc_pts = _sweep_points(left_start0, left_target_pos0, hump_sweep_l)
        right_arc_pts = _sweep_points(right_start0, right_target_pos0, hump_sweep_r)

        def _make_point_trail(path, points_xyz, rgb, radius):
            """Static PointInstancer: one prototype sphere, N positions, no
            per-frame updates needed — draws the whole trail at once."""
            proto_path = path + "/Proto"
            proto = UsdGeom.Sphere.Define(stage, proto_path)
            proto.CreateRadiusAttr().Set(radius)
            proto.CreateDisplayColorAttr().Set([Gf.Vec3f(*rgb)])
            inst = UsdGeom.PointInstancer.Define(stage, path)
            inst.CreatePrototypesRel().SetTargets([proto_path])
            inst.CreatePositionsAttr().Set([Gf.Vec3f(*p) for p in points_xyz])
            inst.CreateProtoIndicesAttr().Set([0] * len(points_xyz))
            return inst

        _make_point_trail("/World/ArcWaypointL", left_arc_pts,
                           (1.0, 1.0, 0.0), 0.018)    # bright yellow = left intended arc
        _make_point_trail("/World/ArcWaypointR", right_arc_pts,
                           (0.0, 1.0, 1.0), 0.018)    # bright cyan = right intended arc
        print(f"[replay] full-arc trail ON: {n_pts} points/hand, synthetic sweep (not episode "
              f"data). BRIGHT YELLOW = left hand's intended arc; BRIGHT CYAN = right hand's "
              f"intended arc. Each should rise smoothly from the rim, peak mid-approach, and "
              f"come back down to touch the rim exactly at the target.")

    def _set_state(art, root7, jpos_traj, jmap):
        root = torch.tensor(root7, dtype=torch.float32, device=device).unsqueeze(0)
        pose = root[:, :7]  # pos(3)+quat wxyz(4)
        art.write_root_pose_to_sim(pose)
        art.write_root_velocity_to_sim(torch.zeros((1, 6), device=device))
        jp = torch.tensor(jpos_traj[jmap], dtype=torch.float32, device=device).unsqueeze(0)
        art.write_joint_state_to_sim(jp, torch.zeros_like(jp))

    # Renderer warmup: each sim.step(render=True) lets the RTX denoiser accumulate
    # more temporal samples, so the first captured frames start grainy and "resolve"
    # (that analog-TV-warmup look). More warmup = cleaner start; fewer = stronger
    # convergence-on-camera effect. Set --warmup 0 to crank it up.
    print(f"[replay] warming renderer ({args.warmup} steps)...")
    for _ in range(args.warmup):
        sim.step(render=True)

    writer = imageio.get_writer(args.out, fps=args.fps, codec="libx264", quality=8)
    frames = 0
    for t in range(0, T, args.stride):
        _set_state(g1, robot_root[t], robot_jpos[t], g1_map)
        _set_state(bin_art, bin_root[t], bin_jpos[t], bin_map)
        g1.write_data_to_sim()
        bin_art.write_data_to_sim()

        # Move reward-debug markers to the logged world body positions. If a bin
        # yaw preview was applied, rotate the marker positions about the same
        # pivot so they stay pinned to the (rotated) bin.
        if markers is not None:
            def _world(bi, zoff=0.0):
                mp = bin_pos_all[t, bi].astype(float).copy()
                if abs(args.bin_yaw_deg) > 1e-6:
                    a = np.deg2rad(args.bin_yaw_deg)
                    ca, sa = np.cos(a), np.sin(a)
                    dx, dy = mp[0] - cx, mp[1] - cy
                    mp[0], mp[1] = cx + ca * dx - sa * dy, cy + sa * dx + ca * dy
                mp[2] += zoff
                return Gf.Vec3d(float(mp[0]), float(mp[1]), float(mp[2]))
            for _op, _bi, _zoff in markers.values():
                _op.Set(_world(_bi, _zoff))
            # RED stop target = handle_mid + stop_standoff*face_normal (same as the
            # reward). Place the ball at pelvis height and the ring on the ground.
            hl = _world(gi_l); hr = _world(gi_r); bctr = _world(gi_body)
            hm_xy = np.array([(hl[0] + hr[0]) / 2, (hl[1] + hr[1]) / 2])
            n_xy = hm_xy - np.array([bctr[0], bctr[1]])
            n_xy = n_xy / (np.linalg.norm(n_xy) + 1e-9)
            st = hm_xy + args.stop_standoff * n_xy
            stop_ball_op.Set(Gf.Vec3d(float(st[0]), float(st[1]), float(robot_root[t, 2])))
            stop_ring_op.Set(Gf.Vec3d(float(st[0]), float(st[1]), 0.02))

            # PROPOSED grasp-relative stop target (BLUE), computed exactly as the
            # new reward will: handle_mid + standoff * face_normal, at pelvis z.
            #   face_normal = normalize(handle_mid_xy - bin_center_xy)
            if prop is not None:
                hl = _world(gi_l); hr = _world(gi_r); bctr = _world(gi_body)
                hm_xy = np.array([(hl[0] + hr[0]) / 2, (hl[1] + hr[1]) / 2])
                n_xy = hm_xy - np.array([bctr[0], bctr[1]])
                n_xy = n_xy / (np.linalg.norm(n_xy) + 1e-9)
                st = hm_xy + args.stop_standoff * n_xy
                pel_z = float(robot_root[t, 2])
                prop[0].Set(Gf.Vec3d(float(st[0]), float(st[1]), pel_z))   # blue ball
                prop[1].Set(Gf.Vec3d(float(st[0]), float(st[1]), 0.03))    # accept ring

        # Palm virtual-grasp-point markers: actual position (palm + anatomical
        # offset), the live top-down arc waypoint, and a ready/close/far color
        # from the SAME decomposed horiz/vert/twist gate the reward now uses
        # (Run 19). Not compatible with --bin_yaw_deg (orientations aren't
        # re-rotated by that preview path) — intended for validating the fix
        # against a real rollout, not a synthetic yaw preview.
        if palm_dbg is not None:
            pg = palm_dbg
            left_palm_pos = robot_pos_all[t, pg["left_palm_bi"]].astype(float)
            left_palm_quat = robot_quat_all[t, pg["left_palm_bi"]].astype(float)
            right_palm_pos = robot_pos_all[t, pg["right_palm_bi"]].astype(float)
            right_palm_quat = robot_quat_all[t, pg["right_palm_bi"]].astype(float)

            left_target_pos = bin_pos_all[t, pg["left_target_gi"]].astype(float)
            left_target_quat = bin_quat_all[t, pg["left_target_gi"]].astype(float)
            right_target_pos = bin_pos_all[t, pg["right_target_gi"]].astype(float)
            right_target_quat = _quat_mul_wxyz(
                bin_quat_all[t, pg["right_target_gi"]].astype(float), pg["rx180"]
            )

            left_grasp_pt = left_palm_pos + pg["offset"] * _quat_apply_wxyz(
                left_palm_quat, pg["left_axis"]
            )
            right_grasp_pt = right_palm_pos + pg["offset"] * _quat_apply_wxyz(
                right_palm_quat, pg["right_axis"]
            )

            left_horiz = float(np.linalg.norm(left_grasp_pt[:2] - left_target_pos[:2]))
            right_horiz = float(np.linalg.norm(right_grasp_pt[:2] - right_target_pos[:2]))
            left_vert = float(left_grasp_pt[2] - left_target_pos[2])
            right_vert = float(right_grasp_pt[2] - right_target_pos[2])
            left_twist = _quat_angle_error_wxyz(left_palm_quat, left_target_quat)
            right_twist = _quat_angle_error_wxyz(right_palm_quat, right_target_quat)

            # Live top-down arc waypoint (matches _one_hand_reach's Run 20 hump
            # exactly): 0 at the target AND at/beyond blend_dist, peaking
            # pregrasp_arc_height above the rim at peak_frac*blend_dist.
            hump_l = _arc_hump(left_horiz, args.pregrasp_blend_dist, args.pregrasp_arc_peak_frac)
            hump_r = _arc_hump(right_horiz, args.pregrasp_blend_dist, args.pregrasp_arc_peak_frac)
            waypoint_l = left_target_pos.copy()
            waypoint_l[2] += args.pregrasp_arc_height * hump_l
            waypoint_r = right_target_pos.copy()
            waypoint_r[2] += args.pregrasp_arc_height * hump_r
            pg["waypoint_l_op"].Set(Gf.Vec3d(*[float(v) for v in waypoint_l]))
            pg["waypoint_r_op"].Set(Gf.Vec3d(*[float(v) for v in waypoint_r]))

            pg["palm_l_op"].Set(Gf.Vec3d(*[float(v) for v in left_grasp_pt]))
            pg["palm_r_op"].Set(Gf.Vec3d(*[float(v) for v in right_grasp_pt]))

            def _gate_color(horiz, vert, twist):
                ready = (
                    horiz < args.grasp_finger_close_horiz_dist
                    and abs(vert) < args.grasp_finger_close_vert_dist
                    and twist < 1.0472
                )
                close = horiz < 0.10 and abs(vert) < 0.10
                if ready:
                    return (0.05, 0.9, 0.05)     # GREEN
                elif close:
                    return (0.95, 0.85, 0.0)     # YELLOW
                else:
                    return (0.9, 0.05, 0.85)     # MAGENTA

            left_color = _gate_color(left_horiz, left_vert, left_twist)
            right_color = _gate_color(right_horiz, right_vert, right_twist)
            pg["palm_l_prim"].GetDisplayColorAttr().Set([Gf.Vec3f(*left_color)])
            pg["palm_r_prim"].GetDisplayColorAttr().Set([Gf.Vec3f(*right_color)])

            if t % 50 == 0:
                print(f"[replay] frame {t}: L horiz={left_horiz:.3f} vert={left_vert:+.3f} "
                      f"twist={np.degrees(left_twist):.0f}deg  |  "
                      f"R horiz={right_horiz:.3f} vert={right_vert:+.3f} "
                      f"twist={np.degrees(right_twist):.0f}deg")

        # Aim camera at the G1<->bin midpoint (from logged world positions).
        rp = robot_root[t, :3]
        bp = bin_root[t, :3]
        tgt = Gf.Vec3d(float((rp[0] + bp[0]) / 2), float((rp[1] + bp[1]) / 2), float((rp[2] + bp[2]) / 2 + 0.4))
        sep = float(np.linalg.norm(rp - bp))
        standoff = min(max(2.5 * sep, 4.0), 9.0) * args.cam_zoom
        vd = np.array(args.cam_dir, dtype=float); vd = vd / np.linalg.norm(vd)
        eye = Gf.Vec3d(float(tgt[0] + standoff * vd[0]),
                       float(tgt[1] + standoff * vd[1]),
                       float(tgt[2] + standoff * vd[2]))
        cam_op.Set(_look_at(eye, tgt))

        sim.step(render=True)
        rgb = rgb_annot.get_data()
        if rgb is not None and hasattr(rgb, "size") and rgb.size > 0:
            arr = np.asarray(rgb)
            if arr.ndim == 3 and arr.shape[-1] == 4:
                arr = arr[..., :3]
            writer.append_data(arr)
            frames += 1
        if t % 50 == 0:
            print(f"[replay] frame {t}/{T}  robot_z={rp[2]:.2f} bin=({bp[0]:.2f},{bp[1]:.2f},{bp[2]:.2f}) saved={frames}")

    writer.close()
    print(f"[replay] wrote {args.out} ({frames} frames, {os.path.getsize(args.out)/1e6:.2f} MB)")
    if args.force_exit_after_write:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    sim_app.close()


if __name__ == "__main__":
    main()
