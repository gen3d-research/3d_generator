#!/usr/bin/env python3
"""
Shared headless (matplotlib 3D) mesh-rendering helpers for the gallery scripts
(``render_gallery.py``, ``render_primitives.py``, ``render_samples.py``).

Renders trimesh meshes with a cheap lambertian face shade and lays them out in a
labelled grid. Uses the Agg backend so it works on a headless CPU box (no GPU /
EGL / OSMesa). Factored out of the original ``render_gallery.py`` so the new
gallery scripts reuse one implementation, plus a ``title_color`` knob to mark
accepted (green) / rejected (red) / optimized (blue) samples.
"""
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # headless: no display needed
import matplotlib.pyplot as plt            # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402
import numpy as np                         # noqa: E402


def render_mesh(ax, mesh, title, title_color="black"):
    """Draw one trimesh into a 3D axis with face-normal-z lambertian shading."""
    tris = mesh.triangles  # (F, 3, 3)
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
    ax.set_title(title, fontsize=6, pad=0, color=title_color)


def grid(items, path, cols, title, cell=1.5):
    """Lay meshes out in a grid and save to ``path``.

    ``items`` entries are either ``(label, mesh)`` or ``(label, mesh, color)``
    where ``color`` tints the cell title (e.g. green/red/blue for a verdict).
    """
    items = list(items)
    n = len(items)
    if n == 0:
        print(f"(nothing to render for {path})")
        return
    rows = math.ceil(n / cols)
    fig = plt.figure(figsize=(cols * cell, rows * (cell + 0.05)))
    fig.suptitle(title, fontsize=12)
    for i, item in enumerate(items):
        label, mesh = item[0], item[1]
        color = item[2] if len(item) > 2 else "black"
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        render_mesh(ax, mesh, label, title_color=color)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"wrote {path}  ({n} cells)")


# Verdict -> title colour, shared by render_samples.py
VERDICT_COLOR = {"accepted": "#1a8a1a", "optimized": "#1456c8"}


def verdict_color(verdict: str) -> str:
    if verdict in VERDICT_COLOR:
        return VERDICT_COLOR[verdict]
    return "#c81414"  # any rejection reason -> red
