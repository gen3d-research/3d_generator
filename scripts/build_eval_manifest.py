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
from grasp_planner import plan_grasps, GripperSpec, grasps_as_dicts

_serialise_grasps = grasps_as_dicts   # name the other manifest builders import
from export import URDFExporter, ExportConfig


def _cem_objects(top_k: int, budget: int, seed: int, grasp_model=None):
    iters = 30
    n_samples = max(20, budget // iters)
    cfg = GeneratorConfig(cem_iterations=iters, cem_samples=n_samples, seed=seed,
                          grasp_model=grasp_model)
    gen = RoboticObjectGenerator(cfg)
    gen.train(verbose=False)
    return gen.generate(top_k, ensure_quality=False)


def _baseline_objects(name: str, top_k: int, budget: int, seed: int,
                      encoding: str = "twobox"):
    kw = {} if name == "fixed_cad" or encoding == "twobox" else {"encoding": encoding}
    r = run_baseline(name, budget=budget, seed=seed, top_k=top_k, **kw)
    return r.objects


def _rank_by_model(model, obj, grasps, seed):
    """Order grasps by the learned predictor's lift probability (stored as
    p_success); returns (grasps, best probability)."""
    from grasp_predictor import predict_object
    p = predict_object(model, obj, grasps, seed)
    for g, pi in zip(grasps, p):
        g["p_success"] = float(pi)
    order = np.argsort(-p, kind="stable")
    return [grasps[i] for i in order], float(p.max())


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
    p.add_argument("--n-grasps", type=int, default=12,
                   help="Grasp candidates per object. Keep a DIVERSE set (not just "
                        "the top few by geometric score) so the demo's runtime "
                        "reachability filter has reachable approaches to choose "
                        "from — e.g. a low side grasp may collide while a "
                        "lower-scored top-down grasp is the only reachable one.")
    p.add_argument("--table-check", action="store_true",
                   help="Drop grasps whose hand body would reach below the table "
                        "(grasp_planner.hand_clears_table). Off by default so "
                        "published manifests reproduce.")
    p.add_argument("--grasp-model", type=Path, default=None,
                   help="model.npz from train_grasp_predictor.py: order each object's "
                        "grasps by predicted physical lift success (stored as p_success)")
    p.add_argument("--train-with-model", action="store_true",
                   help="Also use --grasp-model inside CEM training: each sample's score "
                        "is multiplied by its best grasp's predicted lift success")
    p.add_argument("--concave-decomposition", default="hull", choices=["hull", "coacd"],
                   help="Collision of concave primitives in the exported SDF.")
    p.add_argument("--cem-pool", type=int, default=1,
                   help="CEM generates POOL x top-k objects; with --grasp-model the "
                        "top-k whose best grasp has the highest predicted success "
                        "are kept (physics-aware selection), else the first top-k")
    p.add_argument("--baseline-encoding", default="twobox", choices=["twobox", "multitype"],
                   help="search space of CMA-ES / GA / random search: the published "
                        "two-box 13-D encoding, or the 22-D multi-type parity encoding "
                        "(DISCREPANCIES.md #4)")
    args = p.parse_args()
    model = None
    if args.grasp_model:
        from grasp_predictor import GraspPredictor
        model = GraspPredictor(args.grasp_model)

    # Mesh collision (not convex hull, not simplified) so the object's collision
    # matches its visual mesh exactly and rests flush on the table in gz.
    exporter = URDFExporter(ExportConfig(use_convex_hull=False, simplify_collision=False,
                                         concave_decomposition=args.concave_decomposition))
    manifest = []
    for method in args.methods:
        print(f"[{method}] generating top-{args.top_k} objects ...")
        if method == "cem":
            objs = _cem_objects(args.top_k * max(1, args.cem_pool), args.budget, args.seed,
                                str(args.grasp_model) if args.train_with_model else None)
        else:
            objs = _baseline_objects(method, args.top_k, args.budget, args.seed,
                                     args.baseline_encoding)
        # Plan (and optionally score) grasps for every candidate, then select.
        cands = []
        for k, obj in enumerate(objs):
            report = plan_grasps(obj, GripperSpec(), n_surface=256,
                                 max_pairs=1500, max_returned=args.n_grasps,
                                 seed=args.seed + k, table_check=args.table_check)
            grasps = grasps_as_dicts(report.grasps, args.n_grasps)
            best = -1.0
            if model is not None and grasps:
                grasps, best = _rank_by_model(model, obj, grasps, args.seed + k)
            cands.append((obj, report, grasps, best))
        if method == "cem" and args.cem_pool > 1 and model is not None:
            cands = sorted(cands, key=lambda c: c[3], reverse=True)
        cands = cands[:args.top_k]
        for k, (obj, report, grasps, best) in enumerate(cands):
            obj.name = f"{method}_{k:04d}"
            obj_dir = args.export_root / method / obj.name
            paths = exporter.export(obj, obj_dir, obj.name)
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
                "grasps": grasps,
                "n_grasps_synth": len(report.grasps),
            })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {len(manifest)} entries to {args.out}")


if __name__ == "__main__":
    main()
