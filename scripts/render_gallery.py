#!/usr/bin/env python3
"""
Render a visual gallery (headless, matplotlib 3D) of:
  * every archetype in ARCHETYPE_REGISTRY  -> docs/gallery/archetypes.png
  * a sample of trained-v2 generated objects -> docs/gallery/v2_samples.png

Run from 3d_generator/:  python scripts/render_gallery.py
"""
import argparse
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from archetypes import ARCHETYPE_REGISTRY                                   # noqa: E402
from generator import RoboticObjectGenerator, GeneratorConfig              # noqa: E402

OUT = ROOT / "docs" / "gallery"


def _render_mesh(ax, mesh, title):
    tris = mesh.triangles  # (F, 3, 3)
    # Simple lambertian-ish shading from face-normal z component.
    nz = mesh.face_normals[:, 2]
    shade = 0.45 + 0.55 * (0.5 * (nz + 1.0))
    base = np.array([0.42, 0.62, 0.85])
    facecolors = np.clip(shade[:, None] * base[None, :], 0, 1)
    pc = Poly3DCollection(tris, facecolors=facecolors, edgecolor="k",
                          linewidths=0.03)
    ax.add_collection3d(pc)
    c = mesh.bounds.mean(axis=0)
    r = float((mesh.bounds[1] - mesh.bounds[0]).max()) * 0.6 + 1e-6
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=22, azim=-55)
    ax.set_axis_off()
    ax.set_title(title, fontsize=6, pad=0)


def _grid(named_meshes, path, cols, title):
    n = len(named_meshes)
    rows = math.ceil(n / cols)
    fig = plt.figure(figsize=(cols * 1.5, rows * 1.55))
    fig.suptitle(title, fontsize=12)
    for i, (name, mesh) in enumerate(named_meshes):
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        _render_mesh(ax, mesh, name)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"wrote {path}  ({n} objects)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cols", type=int, default=10)
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--iterations", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # 1) archetypes
    named = []
    for name, fn in ARCHETYPE_REGISTRY.items():
        try:
            named.append((name, fn().to_mesh(boolean_union=True)))
        except Exception as e:                       # pragma: no cover
            print(f"skip {name}: {e}")
    _grid(named, OUT / "archetypes.png", args.cols,
          f"Archetype library ({len(named)})")

    # 2) trained-v2 generated samples
    print("training v2 generator for samples ...")
    gen = RoboticObjectGenerator(GeneratorConfig(
        seed=args.seed, max_primitives=8,
        cem_iterations=args.iterations, cem_samples=60))
    gen.train(verbose=False)
    objs = gen.generate(args.samples)
    smp = [(f"{len(o.primitives)}p", o.to_mesh(boolean_union=True)) for o in objs]
    _grid(smp, OUT / "v2_samples.png", min(args.cols, 8),
          f"v2 generated samples (max_primitives=8)")


if __name__ == "__main__":
    main()
