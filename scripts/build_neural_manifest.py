#!/usr/bin/env python3
"""E12: cleaned neural meshes (`e12_clean_meshes.py`) to the evaluators' format. Plans grasps with the same planner
and settings as the plain arms and exports an SDF with CoACD collision. An object that fails counts as a failure in
the summary, so none are dropped.

    python scripts/build_neural_manifest.py --model trellis --seed 65 --root output/e12
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from convex_decomp import decompose_file
from grasp_planner import GripperSpec, grasps_as_dicts, plan_grasps
from primitives import CompositeObject

DENSITY = 1000.0
MU = 0.8


class MeshObject(CompositeObject):
    """A fixed mesh behind the CompositeObject calls the grasp planner makes."""

    def __init__(self, mesh, name):
        super().__init__(primitives=[], name=name)
        self.mesh = mesh

    def to_mesh(self, boolean_union=True):
        return self.mesh

    def mesh_mass_properties(self, density=1000.0):
        return self.mesh.volume * density, self.mesh.moment_inertia * density, self.mesh.center_mass

    def center_of_mass(self, density=1000.0):
        return self.mesh.center_mass


SDF = """<?xml version="1.0" ?>
<sdf version="1.7"><model name="{name}"><static>false</static><link name="{name}_link">
  <inertial><mass>{mass:.6f}</mass><pose>{com[0]:.6f} {com[1]:.6f} {com[2]:.6f} 0 0 0</pose><inertia>
    <ixx>{i[0][0]:.9f}</ixx><ixy>{i[0][1]:.9f}</ixy><ixz>{i[0][2]:.9f}</ixz>
    <iyy>{i[1][1]:.9f}</iyy><iyz>{i[1][2]:.9f}</iyz><izz>{i[2][2]:.9f}</izz></inertia></inertial>
  <visual name="{name}_visual"><geometry><mesh><uri>meshes/{name}_visual.obj</uri></mesh></geometry></visual>
  {collisions}
</link></model></sdf>
"""
COL = """<collision name="{name}_collision_{i}"><geometry><mesh><uri>meshes/{file}</uri></mesh></geometry>
    <surface><friction><ode><mu>{mu}</mu><mu2>{mu}</mu2></ode></friction><bounce><restitution_coefficient>0.1</restitution_coefficient></bounce></surface></collision>"""


def export(m, d, name, coacd_dir=None):
    meshes = d / "meshes"
    meshes.mkdir(parents=True, exist_ok=True)
    visual = meshes / f"{name}_visual.obj"
    m.export(visual, include_normals=True)     # Gazebo's mesh loader needs normals
    parts = decompose_file(visual, coacd_dir or d / "coacd")
    cols = []
    for i, p in enumerate(parts):
        rel = f"{name}_col{i}.obj"
        trimesh.load(p, force="mesh").export(meshes / rel, include_normals=True)
        cols.append(COL.format(name=name, i=i, file=rel, mu=MU))
    # moment_inertia is about the centre of mass, in the mesh's axes, for density 1
    (d / f"{name}.sdf").write_text(SDF.format(name=name, mass=m.volume * DENSITY, com=m.center_mass,
                                              i=m.moment_inertia * DENSITY, collisions="\n  ".join(cols)))
    return str((d / f"{name}.sdf").resolve()), str(visual.resolve()), float(m.volume * DENSITY), [float(x) for x in m.center_mass]


@lru_cache(maxsize=1)
def _predictor(path):
    from grasp_predictor import GraspPredictor
    return GraspPredictor(path)


def _rank_by_model(model, obj, grasps, seed):
    """Same ordering as build_eval_manifest: by predicted lift probability, stored as p_success."""
    from grasp_predictor import predict_object
    p = predict_object(model, obj, grasps, seed)
    for g, pi in zip(grasps, p):
        g["p_success"] = float(pi)
    order = np.argsort(-p, kind="stable")
    return [grasps[i] for i in order], float(p.max())


def one(job):
    k, prompt, rec, model, d, seed, rank = job
    name = f"{model}_{k:04d}"
    row = {"name": name, "method": model, "prompt": prompt["prompt"], "archetype": prompt["name"]}
    row.update({a: b for a, b in rec.items() if a not in ("key", "path")})
    if "failed" in row:
        return row
    try:
        m = trimesh.load(rec["path"], force="mesh", process=True)
        if not m.is_watertight:
            raise ValueError("cleaned mesh not watertight here")
        obj = MeshObject(m, name)
        report = plan_grasps(obj, GripperSpec(), n_surface=256, max_pairs=1500, max_returned=12, seed=seed + k,
                             table_check=rank["table_check"])
        grasps = grasps_as_dicts(report.grasps, 12)
        if rank["model"] and grasps:
            grasps, _ = _rank_by_model(_predictor(rank["model"]), obj, grasps, seed + k)
        coacd_dir = rank["coacd_root"] / f"{k:04d}" if rank["coacd_root"] else None
        sdf, visual, mass, com = export(m, d / "manifest_objects" / model / name, name, coacd_dir)
        row.update(sdf=sdf, visual_mesh=visual, mass=mass, com=com, grasps=grasps,
                   n_grasps_synth=len(report.grasps))
    except Exception as e:     # noqa: BLE001
        row["failed"] = f"{type(e).__name__}: {e}"
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--root", type=Path, default=ROOT / "output/e12")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--table-check", action="store_true", help="hand-body-vs-table check in the planner")
    ap.add_argument("--grasp-model", type=Path, help="order grasps by this predictor")
    ap.add_argument("--coacd-root", type=Path, help="share CoACD parts between arms that use the same meshes")
    args = ap.parse_args()
    rank = {"table_check": args.table_check, "model": str(args.grasp_model) if args.grasp_model else None,
            "coacd_root": args.coacd_root}
    d = args.root / args.model / f"seed_{args.seed}"
    cleaned = json.load(open(d / "clean.json"))
    prompts = json.load(open(args.root / "prompts" / f"seed_{args.seed}.json"))
    jobs = []
    for k, p in enumerate(prompts):
        key = f"{k:03d}_{p['name']}"
        if key in cleaned:
            jobs.append((k, p, cleaned[key], args.model, d, args.seed, rank))
    with ProcessPoolExecutor(args.workers) as ex:
        rows = list(ex.map(one, jobs))
    (d / "objects.json").write_text(json.dumps(rows, indent=1))
    ok = [r for r in rows if "failed" not in r]
    (d / "eval_manifest.json").write_text(json.dumps(ok, indent=1))
    print(f"{args.model} seed {args.seed}: {len(ok)} of {len(rows)} usable")
    for r in rows:
        if "failed" in r:
            print(" ", r["name"], r["archetype"], r["failed"][:120])


if __name__ == "__main__":
    main()
