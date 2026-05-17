#!/usr/bin/env python3
"""
Rewrite every SDF in the manifest so its <collision> uses a <box> primitive
matched to the visual mesh's AABB.

Gazebo / DART chokes on the trimesh-exported collision OBJs because they
lack vertex normals (see gz_v4.log: "submesh ... does not have a normal
count that matches its vertex count").  Replacing the collision geometry
with a box of the right extents is the surgical fix used by every static
manipulation benchmark we have looked at.
"""

import argparse
import json
import re
from pathlib import Path

import trimesh


def aabb_of(mesh_path: Path) -> tuple[float, float, float]:
    m = trimesh.load(str(mesh_path), force="mesh")
    e = m.bounding_box.extents
    return float(e[0]), float(e[1]), float(e[2])


def patch_sdf(sdf_path: Path, visual_mesh: Path) -> bool:
    sdf_text = sdf_path.read_text()
    ex, ey, ez = aabb_of(visual_mesh)
    new_collision = (
        f"<collision name=\"collision\">"
        f"<geometry><box><size>{ex:.6f} {ey:.6f} {ez:.6f}</size></box></geometry>"
        f"<surface><friction><ode><mu>1.0</mu><mu2>1.0</mu2></ode></friction></surface>"
        f"</collision>"
    )
    sdf_text2 = re.sub(
        r"<collision\b[^>]*>.*?</collision>",
        new_collision,
        sdf_text,
        flags=re.DOTALL,
    )
    if sdf_text2 == sdf_text:
        return False
    sdf_path.write_text(sdf_text2)
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    args = p.parse_args()

    manifest = json.loads(args.manifest.read_text())
    n_patched = 0
    for e in manifest:
        sdf = Path(e["sdf"])
        vis = Path(e["visual_mesh"])
        if patch_sdf(sdf, vis):
            n_patched += 1
    print(f"patched {n_patched}/{len(manifest)} SDFs")


if __name__ == "__main__":
    main()
