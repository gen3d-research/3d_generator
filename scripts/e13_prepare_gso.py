#!/usr/bin/env python3
"""E13: Google Scanned Objects to the solids the grading pipeline reads. Runs in ~/venv/gen3d_neural (voxel remeshing
needs scikit-image). Native scale and up axis, no rescaling. Writes one prompt file and the same clean.json for each
arm, so `build_neural_manifest.py` grades the same meshes in both.

    python scripts/e13_prepare_gso.py --gso output/e13/gso --root output/e13/graded --arms gso_plain gso_ranked
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e12_clean_meshes import clean_mesh


def one(job):
    key, src, out = job
    row = {"key": key}
    try:
        # Texture seams split vertices and open the surface; merged by position alone most scans are closed.
        raw = trimesh.load(src, force="mesh", process=True, skip_materials=True)
        m = trimesh.Trimesh(raw.vertices, raw.faces, process=False)
        m.merge_vertices(merge_tex=True, merge_norm=True)
        extent = float(m.extents.max())
        m, info = clean_mesh(m, "gso", extent)
        path = out / f"{key}.ply"
        m.export(path)
        if not trimesh.load(path, force="mesh", process=True).is_watertight:    # float32 rounding merged vertices
            raise ValueError("no watertight solid after reload")
        row.update(info, path=str(path.resolve()), extent_m=extent)
    except Exception as e:     # noqa: BLE001 - an unusable scan is a result
        row["failed"] = f"{type(e).__name__}: {e}"
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gso", type=Path, default=Path("output/e13/gso"))
    ap.add_argument("--index", type=Path, default=Path("output/e13/gso_index.json"))
    ap.add_argument("--root", type=Path, default=Path("output/e13/graded"))
    ap.add_argument("--arms", nargs="+", default=["gso_plain", "gso_ranked"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    index = sorted(json.load(open(args.index)), key=lambda m: m["name"])[:args.limit]
    prompts = [{"name": m["name"], "prompt": m.get("description") or m["name"]} for m in index]
    (args.root / "prompts").mkdir(parents=True, exist_ok=True)
    (args.root / "prompts" / "seed_0.json").write_text(json.dumps(prompts, indent=1))
    out = args.root / "clean"
    out.mkdir(exist_ok=True)
    jobs = [(f"{k:03d}_{m['name']}", args.gso / m["name"] / "meshes" / "model.obj", out) for k, m in enumerate(index)]
    with ProcessPoolExecutor(args.workers) as ex:
        rows = list(ex.map(one, jobs))
    table = json.dumps({r["key"]: r for r in rows}, indent=1)
    for arm in args.arms:
        d = args.root / arm / "seed_0"
        d.mkdir(parents=True, exist_ok=True)
        (d / "clean.json").write_text(table)
    print("gso", sum("failed" not in r for r in rows), "of", len(rows), "solid,",
          sum(bool(r.get("remeshed")) for r in rows), "remeshed")


if __name__ == "__main__":
    main()
