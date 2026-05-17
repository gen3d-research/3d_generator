#!/usr/bin/env python3
"""Render a grid of CEM-generated objects for the manuscript.

Loads visual OBJ meshes from ``output/manifest_objects/cem/`` and lays
them out as a 3x4 grid (12 examples) with shaded faces.  Output goes to
the paper's ``images/`` directory by default.

Usage::

    python scripts/render_examples_grid.py \\
        --src output/manifest_objects/cem \\
        --out ../papers/conferences/ICARM/.../images/generated_examples.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def _shade_mesh(ax, mesh: trimesh.Trimesh, base_color=(0.30, 0.55, 0.85)) -> None:
    verts = mesh.vertices
    faces = mesh.faces
    triangles = verts[faces]

    # Per-face shading from the dot product with a light direction.
    normals = mesh.face_normals
    light = np.array([0.4, -0.3, 0.8])
    light /= np.linalg.norm(light)
    intensity = np.clip(normals @ light, 0.0, 1.0)
    base = np.array(base_color)
    face_colors = base * (0.35 + 0.65 * intensity)[:, None]
    face_colors = np.clip(face_colors, 0.0, 1.0)
    face_colors = np.hstack([face_colors, np.ones((len(face_colors), 1))])

    coll = Poly3DCollection(
        triangles,
        facecolors=face_colors,
        edgecolors=(0.0, 0.0, 0.0, 0.15),
        linewidths=0.2,
    )
    ax.add_collection3d(coll)

    # Frame the mesh.
    bounds = mesh.bounds
    center = bounds.mean(axis=0)
    half = (bounds[1] - bounds[0]).max() / 2 * 1.05
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=22, azim=35)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    for spine_axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        spine_axis.pane.set_visible(False)
    ax.grid(False)


# Twelve visually distinct archetypes covering scoring-relevant geometry
# families (handles, fingers-friendly cavities, asymmetric COM, long
# moment arms).  Order is curated for the figure layout.
_DEFAULT_ARCHETYPES = [
    "hammer", "mug", "l_shape", "t_shape",
    "dumbbell", "bottle", "u_shape", "v_shape",
    "monitor", "frying_pan", "snowman", "joystick",
]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path,
                   default=Path("output/archetypes"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--rows", type=int, default=3)
    p.add_argument("--cols", type=int, default=4)
    p.add_argument("--archetypes", nargs="*", default=_DEFAULT_ARCHETYPES,
                   help="Archetype directory names to render, in order.")
    p.add_argument("--label", action="store_true",
                   help="Print the archetype name under each subplot.")
    args = p.parse_args()

    n = args.rows * args.cols
    chosen = args.archetypes[:n]

    fig = plt.figure(figsize=(args.cols * 2.4, args.rows * 2.4), dpi=200)
    for i, name in enumerate(chosen):
        mesh_path = args.src / name / "meshes" / f"{name}_visual.obj"
        if not mesh_path.exists():
            print(f"WARN: missing {mesh_path}", file=sys.stderr)
            continue
        mesh = trimesh.load(mesh_path, force="mesh")
        ax = fig.add_subplot(args.rows, args.cols, i + 1, projection="3d")
        _shade_mesh(ax, mesh)
        if args.label:
            ax.set_title(name.replace("_", " "), fontsize=9, y=-0.05)

    plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02,
                        wspace=0.05, hspace=0.05)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", dpi=200)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
