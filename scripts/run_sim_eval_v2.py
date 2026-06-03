#!/usr/bin/env python3
"""
Sim-in-the-loop driver: export v1 (paper_repro) and v2 generated objects, write
a manifest the existing Gazebo stability evaluator consumes, and (optionally)
run the headless Gazebo settle test on it.

Two stages so the manifest can be built without Gazebo, then evaluated on a
machine where ``gz sim`` runs:

    # 1) build objects + manifest (no Gazebo needed)
    python scripts/run_sim_eval_v2.py build --n 8 --out output/sim_eval

    # 2) run the headless settle test (needs the world running):
    gz sim -s -r ros2_ws/src/generated_objects_eval/worlds/panda_eval_world.sdf &
    python ros2_ws/src/generated_objects_eval/generated_objects_eval/gazebo_stability_eval.py \
        --manifest output/sim_eval/manifest.json \
        --out output/sim_eval/gazebo_stability.json \
        --config ros2_ws/src/generated_objects_eval/config/eval_config.yaml
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from generator import (RoboticObjectGenerator, GeneratorConfig,            # noqa: E402
                       paper_repro_generator)
from export import URDFExporter, ExportConfig                             # noqa: E402


def _export(objs, method, out_dir, exporter):
    entries = []
    for i, o in enumerate(objs):
        name = f"{method}_{i:03d}"
        o.name = name
        paths = exporter.export(o, out_dir / name, name)
        entries.append({"name": name, "method": method,
                        "sdf": str(Path(paths["sdf"]).resolve()),
                        "urdf": str(Path(paths["urdf"]).resolve()),
                        "visual_mesh": str(Path(paths["visual_mesh"]).resolve()),
                        "collision_mesh": str(Path(paths["collision_mesh"]).resolve()),
                        "n_primitives": len(o.primitives)})
    return entries


def cmd_build(args):
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    exporter = URDFExporter(ExportConfig(use_mesh_inertia=True))

    print("[v1] training paper_repro ...")
    v1 = paper_repro_generator(seed=args.seed)
    v1.config.cem_iterations = args.iterations
    v1.train(verbose=False)
    e1 = _export(v1.generate(args.n), "v1", out_dir, exporter)

    print("[v2] training v2 generator ...")
    v2 = RoboticObjectGenerator(GeneratorConfig(
        seed=args.seed, max_primitives=args.max_primitives,
        cem_iterations=args.iterations))
    v2.train(verbose=False)
    e2 = _export(v2.generate(args.n), "v2", out_dir, exporter)

    manifest = e1 + e2
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {out_dir/'manifest.json'} ({len(manifest)} objects: "
          f"{len(e1)} v1 + {len(e2)} v2)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="export objects + write manifest")
    b.add_argument("--n", type=int, default=8)
    b.add_argument("--iterations", type=int, default=18)
    b.add_argument("--max-primitives", type=int, default=8)
    b.add_argument("--seed", type=int, default=42)
    b.add_argument("--out", type=str, default=str(ROOT / "output" / "sim_eval"))
    b.set_defaults(func=cmd_build)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
