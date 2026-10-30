#!/usr/bin/env python3
"""Physics-labelled grasp dataset from MuJoCo pick-and-lift trials.

For every object in one or more eval manifests, every planner grasp (up to 12)
is checked for feasibility and, if feasible, executed in MuJoCo
(``src/mujoco_eval.py``); its lift outcome is the label. Features are stored
in the object frame so a model can re-express them in each grasp frame:

    obj_points   [n_obj, P, 3]  surface samples of the true (visual) mesh
    obj_mass, obj_com[3], obj_extents[3], obj_name, obj_group, obj_source
    g_obj        [n_g]          index into the obj_* arrays
    g_rank, g_center[3], g_approach[3], g_axis[3], g_width, g_score
    g_feasible, g_success, g_reason

Usage (from 3d_generator/):

    python scripts/build_grasp_dataset.py --out output/grasp_dataset/journal.npz \\
        --manifests output/journal_10seed/seed_*/eval_manifest.json
    python scripts/build_grasp_dataset.py --out output/grasp_dataset/archetypes.npz \\
        --manifests output/arch_variants/manifest.json

``obj_group`` is the manifest entry's ``method`` (generator or archetype name);
``obj_source`` is the manifest path, so splits can hold out whole seeds.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import mujoco_eval as me  # noqa: E402

_CFG = None


def _init(collision: str) -> None:
    global _CFG
    _CFG = me.MujocoEvalConfig(collision=collision, grasps_per_object=64)


def _one(args) -> dict | None:
    entry, source, n_points = args
    import trimesh
    try:
        obj = me.parse_sdf(entry["sdf"])
        collision = _CFG.collision
        if collision == "coacd":
            try:
                obj["geoms"] = me.coacd_geoms(entry["visual_mesh"], _CFG.coacd_threshold)
            except Exception:      # noqa: BLE001
                collision = "sdf_fallback"
        grasps = entry.get("grasps", [])
        res = me.pick_and_lift(obj, grasps, _CFG)
        mesh = trimesh.load(entry["visual_mesh"], force="mesh")
        seed = zlib.crc32(entry["name"].encode())
        pts, _ = trimesh.sample.sample_surface(mesh, n_points, seed=seed)
    except Exception as e:         # noqa: BLE001
        return {"name": entry["name"], "error": f"{type(e).__name__}: {e}"}
    outcome = {t["rank"]: t for t in res.get("trials", [])}
    reasons = {t["rank"]: t["reason"] for t in res.get("infeasible", [])}
    rows = []
    for k, g in enumerate(grasps):
        t = outcome.get(k)
        rows.append({
            "rank": k, "center": g["center"], "approach": g["approach"],
            "axis": g["axis"], "width": float(g.get("width", 0.0)),
            "score": float(g.get("score", 0.0)),
            "feasible": t is not None,
            "success": bool(t and t.get("success")),
            "reason": t["reason"] if t else reasons.get(k, "not_reached"),
        })
    return {"name": entry["name"], "group": entry.get("method", "?"), "source": source,
            "collision": collision, "mass": obj["mass"], "com": obj["com"],
            "extents": list(np.ptp(mesh.vertices, axis=0)),
            "points": pts.astype(np.float32), "grasps": rows}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifests", nargs="+", required=True, help="paths or globs")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--collision", default="coacd", choices=["coacd", "sdf"])
    ap.add_argument("--points", type=int, default=512)
    ap.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 4))
    ap.add_argument("--limit", type=int, default=0, help="first N objects per manifest")
    args = ap.parse_args()

    jobs = []
    for pat in args.manifests:
        for path in sorted(glob.glob(pat)) or [pat]:
            entries = json.loads(Path(path).read_text())
            if args.limit:
                entries = entries[:args.limit]
            jobs += [(e, str(path), args.points) for e in entries]
    print(f"{len(jobs)} objects, collision={args.collision}, workers={args.workers}")

    t0 = time.time()
    recs, errors = [], 0
    with ProcessPoolExecutor(args.workers, initializer=_init,
                             initargs=(args.collision,)) as ex:
        for i, r in enumerate(ex.map(_one, jobs, chunksize=4)):
            if r is None or "error" in r:
                errors += 1
            else:
                recs.append(r)
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)

    g = [(oi, row) for oi, r in enumerate(recs) for row in r["grasps"]]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        obj_points=np.stack([r["points"] for r in recs]),
        obj_mass=np.array([r["mass"] for r in recs], np.float32),
        obj_com=np.array([r["com"] for r in recs], np.float32),
        obj_extents=np.array([r["extents"] for r in recs], np.float32),
        obj_name=np.array([r["name"] for r in recs]),
        obj_group=np.array([r["group"] for r in recs]),
        obj_source=np.array([r["source"] for r in recs]),
        obj_collision=np.array([r["collision"] for r in recs]),
        g_obj=np.array([oi for oi, _ in g], np.int32),
        g_rank=np.array([row["rank"] for _, row in g], np.int16),
        g_center=np.array([row["center"] for _, row in g], np.float32),
        g_approach=np.array([row["approach"] for _, row in g], np.float32),
        g_axis=np.array([row["axis"] for _, row in g], np.float32),
        g_width=np.array([row["width"] for _, row in g], np.float32),
        g_score=np.array([row["score"] for _, row in g], np.float32),
        g_feasible=np.array([row["feasible"] for _, row in g]),
        g_success=np.array([row["success"] for _, row in g]),
        g_reason=np.array([row["reason"] for _, row in g]),
        config=json.dumps(me.config_dict(_cfg_for_meta(args.collision))),
    )
    feas = np.array([row["feasible"] for _, row in g])
    succ = np.array([row["success"] for _, row in g])
    print(f"wrote {args.out}: {len(recs)} objects ({errors} errors), {len(g)} grasps, "
          f"{feas.sum()} feasible, {succ.sum()} lifted "
          f"({succ.sum() / max(1, feas.sum()):.1%} of feasible) in {time.time() - t0:.0f}s")


def _cfg_for_meta(collision: str) -> me.MujocoEvalConfig:
    return me.MujocoEvalConfig(collision=collision, grasps_per_object=64)


if __name__ == "__main__":
    main()
