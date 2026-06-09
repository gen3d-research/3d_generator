#!/usr/bin/env python3
"""
Build an eval manifest of ARCHETYPE VARIANTS (N parameter variations per archetype)
for the ROS 2 downstream tests — same JSON format as build_eval_manifest.py, so the
Gazebo drop test (gazebo_stability_eval) and MoveIt pick/plan (visual_demo.launch.py /
moveit_planning_eval.launch.py) consume it unchanged.

Each variant's ``method`` field is set to the archetype name, so the evaluators'
per-method summaries become per-ARCHETYPE stability / plan rates, and
``visual_demo.launch.py method:=mug_like`` cycles that archetype's variants.

Variants come from the per-archetype CEM (``ArchetypeTrainer``): it introspects a
factory's signature and samples its scalar/ndarray parameters. ``--train`` first
CEM-tunes each archetype's distribution (optimized-but-narrower variants); the default
samples raw variations around the hand-set defaults (more diverse).

Usage::

    # 100 variants of every archetype (CPU; large — see the runtime note below)
    python3 scripts/build_archetype_manifest.py --variants 100 \
        --out output/arch_variants/manifest.json \
        --export-root output/arch_variants/objects

    # A quick, sim-testable slice: 8 variants of three archetypes
    python3 scripts/build_archetype_manifest.py --variants 8 \
        --archetypes mug_like,wine_bottle,nut \
        --out output/arch_demo/manifest.json --export-root output/arch_demo/objects

Runtime: generation is cheap, but per-object URDF/SDF export + force-closure grasp
synthesis is ~0.5-2 s each, and the ROS 2 sim is sequential (~3-5 s/object). 105
archetypes x 100 variants = 10,500 objects is hours of CPU and 10+ hours of sim — for
the sim, scope it down with --variants / --archetypes, or run the CPU grasp eval on all
and the drop/pick test on a sample.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from archetypes import ARCHETYPE_REGISTRY                    # noqa: E402
from archetype_cem import ArchetypeTrainer                   # noqa: E402
from grasp_planner import plan_grasps, GripperSpec           # noqa: E402
from export import URDFExporter, ExportConfig                # noqa: E402
# Reuse the exact grasp serializer the main manifest builder uses.
sys.path.insert(0, str(ROOT / "scripts"))
from build_eval_manifest import _serialise_grasps           # noqa: E402


def _entry(obj, name, method, exporter, export_root, n_grasps, seed):
    """Export one object + synthesize grasps -> one manifest entry (or None)."""
    obj.name = name
    paths = exporter.export(obj, export_root / method / name, name)
    report = plan_grasps(obj, GripperSpec(), n_surface=256,
                         max_pairs=1500, max_returned=n_grasps, seed=seed)
    try:
        mass, _, com = obj.mesh_mass_properties(1000.0)
    except Exception:
        mass = obj.total_volume() * 1000.0
        com = obj.center_of_mass(1000.0)
    try:
        import trimesh
        vm = trimesh.load(paths["visual_mesh"], force="mesh")
        aabb = [vm.bounds[0].tolist(), vm.bounds[1].tolist()]
        extents = (vm.bounds[1] - vm.bounds[0]).tolist()
    except Exception:
        aabb, extents = None, None
    return {
        "name": name,
        "method": method,                       # = archetype name -> per-archetype grouping
        "urdf": str(Path(paths["urdf"]).resolve()),
        "sdf": str(Path(paths["sdf"]).resolve()),
        "visual_mesh": str(Path(paths["visual_mesh"]).resolve()),
        "collision_mesh": str(Path(paths["collision_mesh"]).resolve()),
        "mass": float(mass),
        "com": list(map(float, com)),
        "aabb": aabb,
        "extents": extents,
        "grasps": _serialise_grasps(report.grasps, n_grasps),
        "n_grasps_synth": len(report.grasps),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", type=int, default=100,
                    help="parameter variants to generate per archetype")
    ap.add_argument("--archetypes", default="",
                    help="comma-separated archetype names (default: all 105)")
    ap.add_argument("--train", action="store_true",
                    help="CEM-tune each archetype first (default: raw sampled variants)")
    ap.add_argument("--iters", type=int, default=15, help="CEM iterations when --train")
    ap.add_argument("--samples", type=int, default=40, help="CEM samples/iter when --train")
    ap.add_argument("--n-grasps", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=ROOT / "output" / "arch_variants" / "manifest.json")
    ap.add_argument("--export-root", type=Path,
                    default=ROOT / "output" / "arch_variants" / "objects")
    args = ap.parse_args()

    names = ([n.strip() for n in args.archetypes.split(",") if n.strip()]
             or list(ARCHETYPE_REGISTRY))
    names = [n for n in names if n in ARCHETYPE_REGISTRY]

    exporter = URDFExporter(ExportConfig(use_convex_hull=False, simplify_collision=False))
    manifest, skipped = [], 0

    for ai, name in enumerate(names):
        trainer = ArchetypeTrainer(ARCHETYPE_REGISTRY[name])
        if args.train:
            trainer.train(iterations=args.iters, samples_per_iter=args.samples)
        made = 0
        for k in range(args.variants):
            try:
                params = trainer.dist.sample(trainer.rng)
                obj = ARCHETYPE_REGISTRY[name](**params)
                if not obj.is_connected():
                    skipped += 1
                    continue
                manifest.append(_entry(obj, f"{name}_{k:04d}", name, exporter,
                                       args.export_root, args.n_grasps,
                                       args.seed + ai * 1000 + k))
                made += 1
            except Exception:
                skipped += 1
        print(f"[{ai + 1}/{len(names)}] {name}: {made}/{args.variants} variants", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {len(manifest)} entries ({len(names)} archetypes, {skipped} skipped) "
          f"to {args.out}")


if __name__ == "__main__":
    main()
