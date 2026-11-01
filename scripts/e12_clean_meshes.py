#!/usr/bin/env python3
"""E12: raw neural mesh to one watertight solid at a real size. Runs in ~/venv/gen3d_neural (voxel remeshing needs
scikit-image, which the generator's venv does not have). Sets the up axis, scales the longest side to the prompted
archetype's, keeps the largest connected body, and voxel-remeshes what is not watertight, so the model is graded on
its shape and not on how clean its surface is. A mesh that still gives no solid is recorded as failed.

    python scripts/e12_clean_meshes.py --model trellis --seed 65 --root output/e12
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import trimesh

# Native up axis of each model's output. Hunyuan3D-2 is y-up, the others z-up.
UP_TO_Z = {"shap_e": np.eye(3), "trellis": np.eye(3),
           "hunyuan3d": np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]]), "gso": np.eye(3)}
MAX_FACES = 60000


def _remesh(m, extent):
    pitch = extent / 128.0
    vox = m.voxelized(pitch).fill()
    r = vox.marching_cubes
    r.apply_transform(vox.transform)
    return r


def _largest(m, info):
    parts = m.split(only_watertight=False)
    if len(parts) <= 1:
        return m
    vols = [abs(p.volume) for p in parts]
    info["n_bodies"] = len(parts)
    info["kept_volume_frac"] = float(max(vols) / max(sum(vols), 1e-12))
    return parts[int(np.argmax(vols))]


def clean(path, model, extent):
    return clean_mesh(trimesh.load(path, force="mesh", process=True), model, extent)


def clean_mesh(m, model, extent):
    info = {}
    if len(m.faces) == 0:
        raise ValueError("empty mesh")
    m.apply_transform(np.vstack([np.column_stack([UP_TO_Z[model], np.zeros(3)]), [0, 0, 0, 1]]))
    m.apply_scale(extent / float(m.extents.max()))
    info["input_watertight"] = bool(m.is_watertight)
    if not m.is_watertight:
        m = _remesh(m, extent)
        info["remeshed"] = True
    m = _largest(m, info)
    m.apply_scale(extent / float(m.extents.max()))
    if len(m.faces) > MAX_FACES:
        d = m.simplify_quadric_decimation(face_count=MAX_FACES)
        if d.is_watertight and d.volume > 0:
            m = d
        else:
            info["decimation_skipped"] = True
    if not m.is_watertight:
        m = _largest(_remesh(m, extent), info)
        info["remeshed"] = True
    trimesh.repair.fix_normals(m)
    if not m.is_watertight or m.volume <= 0:
        raise ValueError("no watertight solid")
    c = 0.5 * (m.bounds[0] + m.bounds[1])
    m.apply_translation([-c[0], -c[1], -m.bounds[0][2]])
    return m, info


def one(job):
    key, rec, prompt, model, out = job
    row = {"key": key}
    try:
        if "path" not in rec:
            raise ValueError(f"generation failed: {rec.get('error')}")
        extent = max(prompt["extents_m"])
        m, info = clean(rec["path"], model, extent)
        path = out / f"{key}.ply"
        m.export(path)
        if not trimesh.load(path, force="mesh", process=True).is_watertight:    # float32 rounding merged vertices
            m = _largest(_remesh(m, extent), info)
            trimesh.repair.fix_normals(m)
            c = 0.5 * (m.bounds[0] + m.bounds[1])
            m.apply_translation([-c[0], -c[1], -m.bounds[0][2]])
            m.export(path)
            info["remeshed"] = True
            if not trimesh.load(path, force="mesh", process=True).is_watertight:
                raise ValueError("no watertight solid")
        row.update(info, path=str(path.resolve()))
    except Exception as e:     # noqa: BLE001 - an unusable mesh is a result
        row["failed"] = f"{type(e).__name__}: {e}"
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(UP_TO_Z))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--root", type=Path, default=Path("output/e12"))
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    d = args.root / args.model / f"seed_{args.seed}"
    log = json.load(open(d / "generate.json"))
    prompts = json.load(open(args.root / "prompts" / f"seed_{args.seed}.json"))
    (d / "clean").mkdir(exist_ok=True)
    jobs = []
    for k, p in enumerate(prompts):
        key = f"{k:03d}_{p['name']}"
        if key in log:
            jobs.append((key, log[key], p, args.model, d / "clean"))
    with ProcessPoolExecutor(args.workers) as ex:
        rows = list(ex.map(one, jobs))
    (d / "clean.json").write_text(json.dumps({r["key"]: r for r in rows}, indent=1))
    print(args.model, args.seed, sum("failed" not in r for r in rows), "of", len(rows), "solid")


if __name__ == "__main__":
    main()
