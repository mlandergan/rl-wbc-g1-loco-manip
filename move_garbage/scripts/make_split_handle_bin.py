#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Make a bin USD variant where the handle has its own collider.

Why: the lid's collider is ``physics:approximation = convexHull``. The lid mesh
also contains the hinge rod the handle wraps around, so the hull turns the
lid + handle into one solid slab that swallows the grasp bar (the bar
reference sits ~1.5 cm inside it). Hands can never get between the lid and
the handle.

What this does (the visual meshes are untouched):
  * Turns off collision on the lid mesh. Optionally (``--split-body``) it also
    turns off collision on the body mesh.
  * Adds collision-only meshes (purpose=guide, so they don't render and aren't
    counted in the spawn bbox):
      - lid_panel_collider: convexHull of the lid faces in front of the handle
        (centroid y < --lid-cut-y)
      - lid_bracket_{neg,pos}_collider: convexHull of the lid end brackets
        (|x| > --rod-half-len, y >= --lid-cut-y)
      - body_shell_collider (only with --split-body): the body mesh minus the
        hinge sleeve, with the same convexDecomposition settings as the original
  * Adds handle_collider: a capsule along X that covers the hinge sleeve + rod.

Usage (Isaac Sim / pxr python):
  python make_split_handle_bin.py --out <path.usd> [--split-body]
Select the variant at train time with the BIN_ASSET_PATH env var (see
move_garbage/scenario_cfg/isaacsim.py).
"""

from __future__ import annotations

import argparse
import os

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, Vt

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SRC = os.path.join(
    HERE, "..", "data", "objects", "bin", "RecyclingContainerWithWheels_A01_Blue_01_dynamic.usd"
)
BODY_MESH = "/Bin/body/visual/SM_RecyclingContainerWithWheels_A01_01"
LID_MESH = BODY_MESH + "/SM_RecyclingContainerWithWheels_A01_Lid_01"
STATIC_NAME = "RecyclingContainerWithWheels_A01_Blue_01_static.usd"


def _mesh_in_body(stage, path):
    """Return (points_body[N,3], faces list[np.ndarray]) for a mesh prim."""
    cache = UsdGeom.XformCache()
    prim = stage.GetPrimAtPath(path)
    mesh = UsdGeom.Mesh(prim)
    pts = np.array(mesh.GetPointsAttr().Get(), dtype=np.float64)
    m_body = cache.GetLocalToWorldTransform(stage.GetPrimAtPath("/Bin/body"))
    m = np.array(cache.GetLocalToWorldTransform(prim) * m_body.GetInverse())  # row-vector
    pb = (np.c_[pts, np.ones(len(pts))] @ m)[:, :3]
    fvc = np.array(mesh.GetFaceVertexCountsAttr().Get())
    fvi = np.array(mesh.GetFaceVertexIndicesAttr().Get())
    faces, k = [], 0
    for c in fvc:
        faces.append(fvi[k:k + c]); k += c
    return pb, faces


def _subset(pb, faces, keep):
    """Compact the kept faces into (points, counts, indices)."""
    sel = [f for f, k in zip(faces, keep) if k]
    used = np.unique(np.concatenate(sel))
    remap = -np.ones(len(pb), dtype=np.int64); remap[used] = np.arange(len(used))
    return pb[used], np.array([len(f) for f in sel]), np.concatenate([remap[f] for f in sel])


def _add_collider_mesh(stage, path, pts, counts, idx, approximation, extra_attrs=None):
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, p)) for p in pts]))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([int(c) for c in counts]))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([int(i) for i in idx]))
    mesh.CreatePurposeAttr(UsdGeom.Tokens.guide)
    prim = mesh.GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim)
    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(approximation)
    for name, (typ, val) in (extra_attrs or {}).items():
        prim.CreateAttribute(name, typ, custom=True).Set(val)
    return prim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split-body", action="store_true",
                    help="also replace the body mesh collider with body-minus-sleeve")
    ap.add_argument("--lid-cut-y", type=float, default=0.312)
    ap.add_argument("--rod-half-len", type=float, default=0.212)
    ap.add_argument("--sleeve-cut-y", type=float, default=0.316)
    ap.add_argument("--sleeve-cut-z", type=float, default=0.835)
    ap.add_argument("--handle-y", type=float, default=0.3425)
    ap.add_argument("--handle-z", type=float, default=0.8583)
    ap.add_argument("--handle-radius", type=float, default=0.019)
    ap.add_argument("--move-target-to-handle", action="store_true",
                    help="re-centre grasp_target_left/right on the handle capsule axis")
    args = ap.parse_args()

    src = os.path.abspath(args.src); out = os.path.abspath(args.out)
    Sdf.Layer.FindOrOpen(src).Export(out)
    stage = Usd.Stage.Open(out)
    if os.path.dirname(out) != os.path.dirname(src):
        # Keep the static visual reference valid when writing elsewhere.
        vis = stage.GetRootLayer().GetPrimAtPath("/Bin/body/visual")
        vis.referenceList.ClearEdits()
        vis.referenceList.Prepend(Sdf.Reference(
            os.path.join(os.path.dirname(src), STATIC_NAME), "/SM_RecyclingContainerWithWheels_A01_01"))
    stage = Usd.Stage.Open(out)  # recompose

    # --- lid: panel + brackets (rod excluded) ---
    lp, lf = _mesh_in_body(stage, LID_MESH)
    cen = np.array([lp[f].mean(0) for f in lf])
    panel = cen[:, 1] < args.lid_cut_y
    br_neg = (~panel) & (cen[:, 0] < -args.rod_half_len)
    br_pos = (~panel) & (cen[:, 0] > args.rod_half_len)
    print(f"lid faces: total {len(lf)} panel {panel.sum()} bracket- {br_neg.sum()} "
          f"bracket+ {br_pos.sum()} dropped(rod/fin) {len(lf) - panel.sum() - br_neg.sum() - br_pos.sum()}")
    root = "/Bin/body/split_colliders"
    UsdGeom.Xform.Define(stage, root)
    for name, keep in (("lid_panel_collider", panel), ("lid_bracket_neg_collider", br_neg),
                       ("lid_bracket_pos_collider", br_pos)):
        p, c, i = _subset(lp, lf, keep)
        _add_collider_mesh(stage, f"{root}/{name}", p, c, i, "convexHull")
        print(f"  {name}: {len(p)} pts bbox {p.min(0).round(4)} {p.max(0).round(4)}")
    stage.GetPrimAtPath(LID_MESH).CreateAttribute(
        "physics:collisionEnabled", Sdf.ValueTypeNames.Bool).Set(False)

    # --- body: optionally drop the hinge sleeve from the decomposition ---
    if args.split_body:
        bp, bf = _mesh_in_body(stage, BODY_MESH)
        cen = np.array([bp[f].mean(0) for f in bf])
        sleeve = (cen[:, 1] > args.sleeve_cut_y) & (cen[:, 2] > args.sleeve_cut_z)
        print(f"body faces: total {len(bf)} sleeve(dropped) {sleeve.sum()}")
        p, c, i = _subset(bp, bf, ~sleeve)
        _add_collider_mesh(stage, f"{root}/body_shell_collider", p, c, i, "convexDecomposition", {
            "physxConvexDecompositionCollision:maxConvexHulls": (Sdf.ValueTypeNames.Int, 32),
            "physxConvexDecompositionCollision:shrinkWrap": (Sdf.ValueTypeNames.Bool, True),
            "physxConvexDecompositionCollision:voxelResolution": (Sdf.ValueTypeNames.Int, 500000),
        })
        stage.GetPrimAtPath(BODY_MESH).CreateAttribute(
            "physics:collisionEnabled", Sdf.ValueTypeNames.Bool).Set(False)

    # --- handle: capsule along X over the sleeve + rod ---
    r = args.handle_radius
    cap = UsdGeom.Capsule.Define(stage, f"{root}/handle_collider")
    cap.CreateAxisAttr(UsdGeom.Tokens.x)
    cap.CreateRadiusAttr(r)
    cap.CreateHeightAttr(2 * args.rod_half_len - 2 * r)
    cap.CreatePurposeAttr(UsdGeom.Tokens.guide)
    cap.AddTranslateOp().Set(Gf.Vec3d(0.0, args.handle_y, args.handle_z))
    UsdPhysics.CollisionAPI.Apply(cap.GetPrim())
    print(f"  handle_collider: capsule r={r} x=+-{args.rod_half_len} at y={args.handle_y} z={args.handle_z}")

    # --- optionally re-centre the grasp targets on the handle axis ---
    if args.move_target_to_handle:
        cache = UsdGeom.XformCache()
        for side in ("left", "right"):
            gt = stage.GetPrimAtPath(f"/Bin/grasp_target_{side}")
            jt = stage.GetPrimAtPath(f"/Bin/joint_grasp_target_{side}_fixed")
            old = cache.GetLocalToWorldTransform(gt).ExtractTranslation()
            xf = UsdGeom.Xformable(gt)
            tops = [op for op in xf.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate]
            new = Gf.Vec3d(old[0], args.handle_y, args.handle_z)
            if tops:
                cur = tops[0].Get()
                tops[0].Set(type(cur)(cur[0], cur[1] + (new[1] - old[1]), cur[2] + (new[2] - old[2])))
            else:
                raise RuntimeError(f"grasp_target_{side} has no translate op")
            j = UsdPhysics.Joint(jt)
            b0 = j.GetBody0Rel().GetTargets(); b1 = j.GetBody1Rel().GetTargets()
            lp0 = j.GetLocalPos0Attr().Get(); lp1 = j.GetLocalPos1Attr().Get()
            print(f"  grasp_target_{side}: {tuple(round(v, 4) for v in old)} -> {tuple(round(v, 4) for v in new)}"
                  f" | joint body0={b0} body1={b1} localPos0={lp0} localPos1={lp1}")
            if b0 and str(b0[0]).endswith("/body"):
                j.GetLocalPos0Attr().Set(Gf.Vec3f(float(lp0[0]), float(lp0[1] + new[1] - old[1]), float(lp0[2] + new[2] - old[2])))
            elif b1 and str(b1[0]).endswith("/body"):
                j.GetLocalPos1Attr().Set(Gf.Vec3f(float(lp1[0]), float(lp1[1] + new[1] - old[1]), float(lp1[2] + new[2] - old[2])))
            else:
                raise RuntimeError(f"unexpected joint bodies {b0} {b1}")
            print(f"     -> localPos0={j.GetLocalPos0Attr().Get()} localPos1={j.GetLocalPos1Attr().Get()}"
                  f" world now {tuple(round(v, 4) for v in UsdGeom.XformCache().GetLocalToWorldTransform(gt).ExtractTranslation())}")

    stage.GetRootLayer().Save()
    print("wrote", out)


if __name__ == "__main__":
    main()
