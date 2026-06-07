#!/usr/bin/env python3
"""
Build the evaluation manifest consumed by the ros2_ws downstream nodes.

For each (method, object_idx) entry the manifest stores
    * a unique name and the method that produced it,
    * paths to the exported URDF, SDF, and visual mesh,
    * pre-computed grasp candidates (center + approach + width) so the ROS node
      doesn't need to depend on trimesh.

Usage::

    python3 scripts/build_eval_manifest.py \
        --budget 600 --top-k 25 --seed 42 \
        --out output/eval_manifest.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from generator import RoboticObjectGenerator, GeneratorConfig
from baselines import run_baseline
from grasp_planner import plan_grasps, GripperSpec
from export import URDFExporter, ExportConfig


def _cem_objects(top_k: int, budget: int, seed: int):
    iters = 30
    n_samples = max(20, budget // iters)
    cfg = GeneratorConfig(cem_iterations=iters, cem_samples=n_samples, seed=seed)
    gen = RoboticObjectGenerator(cfg)
    gen.train(verbose=False)
    return gen.generate(top_k, ensure_quality=False)


def _baseline_objects(name: str, top_k: int, budget: int, seed: int):
    r = run_baseline(name, budget=budget, seed=seed, top_k=top_k)
    return r.objects


def _serialise_grasps(grasps, max_keep=5):
    out = []
    for g in grasps[:max_keep]:
        # axis = the antipodal contact line (contact1 -> contact2). The gripper
        # must open its fingers along this so they actually straddle the object.
        axis = np.asarray(g.contact2) - np.asarray(g.contact1)
        n = float(np.linalg.norm(axis))
        axis = (axis / n).tolist() if n > 1e-9 else [0.0, 1.0, 0.0]
        out.append({
            "center": g.center.tolist(),
            "approach": g.approach.tolist(),
            "axis": axis,
            "width": float(g.width),
            "margin": float(g.margin),
        })
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--budget", type=int, default=600)
    p.add_argument("--top-k", type=int, default=25,
                   help="Objects per method to include in the manifest")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--methods", nargs="+",
                   default=["cem", "cmaes", "ga", "random_search", "fixed_cad"])
    p.add_argument("--out", type=Path,
                   default=ROOT / "output" / "eval_manifest.json")
    p.add_argument("--export-root", type=Path,
                   default=ROOT / "output" / "manifest_objects")
    p.add_argument("--n-grasps", type=int, default=5,
                   help="Grasp candidates per object")
    args = p.parse_args()

    # Mesh collision (not convex hull, not simplified) so the object's collision
    # matches its visual mesh exactly and rests flush on the table in gz.
    exporter = URDFExporter(ExportConfig(use_convex_hull=False, simplify_collision=False))
    manifest = []
    for method in args.methods:
        print(f"[{method}] generating top-{args.top_k} objects ...")
        if method == "cem":
            objs = _cem_objects(args.top_k, args.budget, args.seed)
        else:
            objs = _baseline_objects(method, args.top_k, args.budget, args.seed)
        for k, obj in enumerate(objs):
            obj.name = f"{method}_{k:04d}"
            obj_dir = args.export_root / method / obj.name
            paths = exporter.export(obj, obj_dir, obj.name)
            report = plan_grasps(obj, GripperSpec(), n_surface=256,
                                 max_pairs=1500, max_returned=args.n_grasps,
                                 seed=args.seed + k)
            # Mass + center of mass (object frame) so the demo can (a) auto-size
            # the closing force and (b) prefer grasps near the CoM.
            try:
                mass, _, com = obj.mesh_mass_properties(1000.0)
            except Exception:
                mass = obj.total_volume() * 1000.0
                com = obj.center_of_mass(1000.0)
            # AABB + extents (object frame == spawned-mesh frame) so the demo can
            # start a vertical grasp above the object's tallest point.
            try:
                import trimesh
                _vm = trimesh.load(paths["visual_mesh"], force="mesh")
                aabb = [_vm.bounds[0].tolist(), _vm.bounds[1].tolist()]
                extents = (_vm.bounds[1] - _vm.bounds[0]).tolist()
            except Exception:
                aabb, extents = None, None
            manifest.append({
                "name": obj.name,
                "method": method,
                # Absolute paths so gz_sim (launched from a different cwd) can
                # find the SDF/meshes when the demo spawns the object.
                "urdf": str(Path(paths["urdf"]).resolve()),
                "sdf": str(Path(paths["sdf"]).resolve()),
                "visual_mesh": str(Path(paths["visual_mesh"]).resolve()),
                "collision_mesh": str(Path(paths["collision_mesh"]).resolve()),
                "mass": float(mass),
                "com": list(map(float, com)),
                "aabb": aabb,
                "extents": extents,
                "grasps": _serialise_grasps(report.grasps, args.n_grasps),
                "n_grasps_synth": len(report.grasps),
            })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {len(manifest)} entries to {args.out}")


if __name__ == "__main__":
    main()
