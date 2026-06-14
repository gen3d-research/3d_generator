#!/usr/bin/env python3
"""
Build a drop-/pick-testable manifest from ANY single generation path.

``build_eval_manifest.py`` only knows the five paper methods (cem / cmaes / ga /
random_search / fixed_cad). This runner is the CLI counterpart for the
Python-API paths described in ``docs/GENERATION_PATHS.md`` — the free CEM (④)
plus the one-knob variants ⑥–⑫ — so each becomes a one-liner that emits a
manifest identical in shape to ``build_eval_manifest.py`` (same fields, same
grasp serialisation). Feed the result straight into the GUIDE §5 drop test and
§6 pick-and-place test.

``--path`` selects a preset; the orthogonal flags below it can stack so the
paths compose exactly like their Python recipes (e.g. ⑧ is seed + gate):

    Path  Preset it turns on
    ----  ---------------------------------------------------------------
     4    free CEM, no extra knob (the baseline every other path builds on)
     6    --repair            (re-orient onto a stable resting pose)
     7    --gate              (dynamic-stability gate: kill the tippy tail)
     8    --seed-archetype screwdriver  + --gate   (warm-start, evolve stable)
     9    --palette curved    (train using only curved primitives)
    10    --pareto            (keep the non-dominated stability/grasp subset)
    11    --target-size 0.05  (bias toward a 5 cm longest extent)
    12    --prompt "..."      (text2geometry; REQUIRED for --path 12)

Examples::

    # ⑥ stability-repair, 15 objects -> manifest
    python scripts/build_path_manifest.py --path 6 \
        --out output/path6/manifest.json --export-root output/path6/objects

    # ⑫ text2geometry from a sentence
    python scripts/build_path_manifest.py --path 12 \
        --prompt "a small stable graspable curved bottle" \
        --out output/path12/manifest.json --export-root output/path12/objects

    # compose by hand (seed + gate + curved palette), custom method tag
    python scripts/build_path_manifest.py --seed-archetype mug_like --gate \
        --palette curved --method-tag mug_seeded \
        --out output/mug_seeded/manifest.json --export-root output/mug_seeded/objects

Then drop-test it (GUIDE §5)::

    ros2 launch generated_objects_eval stability_world_gui.launch.py            # window
    ros2_ws/install/generated_objects_eval/bin/gazebo_stability_eval \
        --manifest $(pwd)/output/path6/manifest.json \
        --out $(pwd)/output/path6/gazebo_stability.json --max-objects 6

...or pick-and-place it (GUIDE §6 — pass method:=<tag> printed at the end)::

    ros2 launch generated_objects_eval visual_demo.launch.py \
        manifest:=$(pwd)/output/path6/manifest.json method:=path6 use_gz_control:=true
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))   # for _serialise_grasps reuse
sys.path.insert(0, str(ROOT / "src"))

from generator import RoboticObjectGenerator, GeneratorConfig
from export import URDFExporter, ExportConfig
from grasp_planner import plan_grasps, GripperSpec
from cem import CURVED_TYPES, FACETED_TYPES
from pareto import pareto_objects
from text2gen import text_to_generator
from build_eval_manifest import _serialise_grasps   # identical grasp format


# Presets: --path N flips these on (explicit flags still override/augment).
PRESETS = {
    4:  {},
    6:  {"repair": True},
    7:  {"gate": True},
    8:  {"seed_archetype": "screwdriver", "gate": True},
    9:  {"palette": "curved"},
    10: {"pareto": True},
    11: {"target_size": 0.05},
    12: {"prompt": True},   # sentinel: --prompt is required for this path
}


def _resolve_palette(spec: str):
    """'curved' / 'faceted' / a comma-separated list of PRIMITIVE_SPECS keys."""
    spec = spec.strip().lower()
    if spec == "curved":
        return CURVED_TYPES
    if spec == "faceted":
        return FACETED_TYPES
    return [k.strip() for k in spec.split(",") if k.strip()]


def _entry_for_object(obj, exporter, root, method, n_grasps, seed):
    """Export one object and build its manifest entry (mass/com/aabb/extents/grasps).
    Mirrors build_eval_manifest.py so the manifest shape is identical."""
    paths = exporter.export(obj, root / obj.name, obj.name)
    report = plan_grasps(obj, GripperSpec(), n_surface=256, max_pairs=1500,
                         max_returned=n_grasps, seed=seed)
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
        "name": obj.name,
        "method": method,
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
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--path", type=int, choices=sorted(PRESETS),
                   help="Generation-path preset (see the table above). Omit to "
                        "compose the knobs by hand.")
    # Orthogonal knobs (stack on top of --path, or use standalone):
    p.add_argument("--repair", action="store_true",
                   help="⑥ re-orient each object onto its most-stable resting pose")
    p.add_argument("--gate", action="store_true",
                   help="⑦ dynamic-stability gate (suppress the tippy tail)")
    p.add_argument("--seed-archetype", default=None,
                   help="⑧ warm-start the free CEM from this archetype name")
    p.add_argument("--palette", default=None,
                   help="⑨ restrict + train on a palette: curved | faceted | "
                        "comma-separated PRIMITIVE_SPECS keys")
    p.add_argument("--target-size", type=float, default=None,
                   help="⑪ bias toward this longest-extent in metres (e.g. 0.05)")
    p.add_argument("--pareto", action="store_true",
                   help="⑩ keep only the non-dominated subset (see --pareto-keys)")
    p.add_argument("--pareto-keys", default="stability_score,graspability_score",
                   help="Objectives to maximise for --pareto (comma-separated)")
    p.add_argument("--prompt", default=None,
                   help="⑫ text2geometry prompt; drives seed/palette/size/gates")
    # Budget / output:
    p.add_argument("--n", type=int, default=15, help="Objects to generate")
    p.add_argument("--iterations", type=int, default=30, help="CEM iterations")
    p.add_argument("--samples", type=int, default=100, help="CEM samples/iter")
    p.add_argument("--seed", type=int, default=42, help="RNG seed")
    p.add_argument("--n-grasps", type=int, default=12, help="Grasp candidates/object")
    p.add_argument("--no-train", action="store_true",
                   help="Skip CEM training (sample from the untrained prior)")
    p.add_argument("--method-tag", default=None,
                   help="Manifest 'method' label (default: path<N> or 'custom'). "
                        "Pass this as method:= to the §6 pick test.")
    p.add_argument("--out", type=Path, default=ROOT / "output" / "path_manifest" / "manifest.json")
    p.add_argument("--export-root", type=Path,
                   default=ROOT / "output" / "path_manifest" / "objects")
    args = p.parse_args()

    # Fold the preset in (explicit flags win over preset defaults).
    preset = PRESETS.get(args.path, {})
    repair = args.repair or preset.get("repair", False)
    gate = args.gate or preset.get("gate", False)
    seed_archetype = args.seed_archetype or preset.get("seed_archetype")
    palette = args.palette or preset.get("palette")
    target_size = args.target_size if args.target_size is not None else preset.get("target_size")
    pareto = args.pareto or preset.get("pareto", False)
    if preset.get("prompt") and not args.prompt:
        p.error("--path 12 (text2geometry) requires --prompt \"...\"")
    method = args.method_tag or (f"path{args.path}" if args.path else "custom")

    # Build the configured generator.
    if args.prompt:
        # ⑫: the prompt sets seed / palette / target_extent / gates internally.
        gen, intent = text_to_generator(
            args.prompt, cem_iterations=args.iterations,
            cem_samples=args.samples, seed=args.seed)
        print(f"[prompt] parsed intent: {intent}")
        # let explicit flags still augment the prompt-built generator
        if repair:
            gen.config.repair_stability = True
        if gate:
            gen.config.dynamic_stability_gate = True
            gen.scoring_config.dynamic_stability_gate = True
        if seed_archetype:
            gen.seed_from(seed_archetype)
        if palette:
            gen.constrain_types(_resolve_palette(palette))
        if target_size is not None:
            gen.target_size(target_size)
    else:
        cfg = GeneratorConfig(
            cem_iterations=args.iterations, cem_samples=args.samples, seed=args.seed,
            repair_stability=repair, dynamic_stability_gate=gate,
            target_extent=target_size)
        gen = RoboticObjectGenerator(cfg)
        if seed_archetype:
            gen.seed_from(seed_archetype)
        if palette:
            gen.constrain_types(_resolve_palette(palette))

    knobs = [k for k, v in [("repair", repair), ("gate", gate),
                            ("seed", seed_archetype), ("palette", palette),
                            ("target_size", target_size), ("pareto", pareto),
                            ("prompt", bool(args.prompt))] if v]
    print(f"[{method}] knobs: {knobs or ['none (④ free CEM)']}")

    if not args.no_train:
        print(f"[{method}] training CEM ({args.iterations} iters x {args.samples} samples)...")
        gen.train(verbose=False)

    print(f"[{method}] generating {args.n} objects...")
    objs = gen.generate(args.n)

    if pareto:
        keys = tuple(k.strip() for k in args.pareto_keys.split(","))
        objs, _ = pareto_objects(objs, gen.scorer, keys=keys)
        print(f"[{method}] Pareto front over {keys}: kept {len(objs)} non-dominated")

    # Export + build the manifest (mesh collision, matching build_eval_manifest).
    exporter = URDFExporter(ExportConfig(use_convex_hull=False, simplify_collision=False))
    root = args.export_root / method
    manifest = []
    for k, obj in enumerate(objs):
        obj.name = f"{method}_{k:04d}"
        manifest.append(_entry_for_object(obj, exporter, root, method, args.n_grasps, args.seed + k))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(manifest, f, indent=2)
    abs_out = args.out.resolve()
    print(f"Wrote {len(manifest)} entries (method={method}) to {args.out}")
    print(f"Next: python scripts/patch_sdf_collision.py --manifest {args.out}")
    print(f"      drop test  -> --manifest {abs_out}")
    print(f"      pick test  -> manifest:={abs_out} method:={method}")


if __name__ == "__main__":
    main()
