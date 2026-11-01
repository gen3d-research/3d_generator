#!/usr/bin/env python3
"""Export the same objects twice, with concave primitives hulled and CoACD-decomposed
in collision, so a Gazebo drop test compares the two on identical objects. Keeps only
objects with at least one concave primitive: the others export identically.

    python scripts/build_collision_pair_manifests.py --seed 42 --out output/gazebo_coacd/seed_42
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_eval_manifest import _baseline_objects, _cem_objects
from export import _CONCAVE_TYPES, ExportConfig, URDFExporter


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--top-k", type=int, default=25)
    ap.add_argument("--budget", type=int, default=1500)
    ap.add_argument("--methods", nargs="+", default=["cem", "fixed_cad"])
    args = ap.parse_args()
    manifests = {"hull": [], "coacd": []}
    n_total = 0
    for method in args.methods:
        objs = (_cem_objects(args.top_k, args.budget, args.seed) if method == "cem"
                else _baseline_objects(method, args.top_k, args.budget, args.seed))
        n_total += len(objs)
        for k, obj in enumerate(objs):
            if not any(p.ptype in _CONCAVE_TYPES for p in obj.primitives):
                continue
            obj.name = f"{method}_{k:04d}"
            for arm in manifests:
                exporter = URDFExporter(ExportConfig(use_convex_hull=False, simplify_collision=False,
                                                     concave_decomposition=arm))
                paths = exporter.export(obj, args.out / arm / method / obj.name, obj.name)
                manifests[arm].append({"name": obj.name, "method": method,
                                       "sdf": str(Path(paths["sdf"]).resolve()),
                                       "visual_mesh": str(Path(paths["visual_mesh"]).resolve())})
    for arm, entries in manifests.items():
        (args.out / f"manifest_{arm}.json").write_text(json.dumps(entries, indent=1))
    print(f"seed {args.seed}: {len(manifests['hull'])} of {n_total} objects have concave primitives")


if __name__ == "__main__":
    main()
