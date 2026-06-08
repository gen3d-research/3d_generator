#!/usr/bin/env python3
"""
Render the PRIMITIVE gallery (headless matplotlib 3D): each of the 9 base
primitive types shown with 2-3 variants to illustrate its degrees of freedom
(DOF) -> docs/gallery/primitives.png (+ a copy into docs/static/images/ for the
website).

A type's DOF = how many independent shape parameters it has, i.e. how its
variants can differ:
    Sphere    1 DOF  (radius)
    Cylinder  2 DOF  (radius, height)         Cone/Pyramid/Capsule  2 DOF
    Torus     2 DOF  (major_radius, minor_radius)
    Box       3 DOF  (dx, dy, dz)             Ellipsoid 3 DOF (rx,ry,rz)
    Wedge     3 DOF  (width, depth, height)

Run from 3d_generator/:
    python scripts/render_primitives.py
    python scripts/render_primitives.py --emit-doc      # markdown spec tables
"""
import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from primitives import (Box, Cylinder, Sphere, Capsule, Cone, Pyramid,  # noqa: E402
                        Torus, Ellipsoid, Wedge, HollowShell, Handle,
                        Frustum, Hemisphere, HexPrism, OpenTube, NGonPrism)
from cem import PRIMITIVE_SPECS                                          # noqa: E402
from _render_common import grid                                         # noqa: E402

OUT = ROOT / "docs" / "gallery"
WEB = ROOT / "docs" / "static" / "images"

# (type label, DOF, [(variant label, primitive), ...]) — variants chosen to
# exercise each independent parameter.
GALLERY = [
    ("Sphere  (1 DOF: radius)", [
        ("r=15", Sphere(radius=0.015)),
        ("r=30", Sphere(radius=0.030)),
        ("r=60", Sphere(radius=0.060)),
    ]),
    ("Cylinder  (2 DOF: radius, height)", [
        ("r20 h40 squat", Cylinder(radius=0.020, height=0.040)),
        ("r30 h80", Cylinder(radius=0.030, height=0.080)),
        ("r12 h120 rod", Cylinder(radius=0.012, height=0.120)),
    ]),
    ("Capsule  (2 DOF: radius, height)", [
        ("r15 h30", Capsule(radius=0.015, height=0.030)),
        ("r25 h60", Capsule(radius=0.025, height=0.060)),
        ("r12 h90 pill", Capsule(radius=0.012, height=0.090)),
    ]),
    ("Cone  (2 DOF: radius, height)", [
        ("r20 h40", Cone(radius=0.020, height=0.040)),
        ("r30 h70", Cone(radius=0.030, height=0.070)),
        ("r15 h100 spike", Cone(radius=0.015, height=0.100)),
    ]),
    ("Pyramid  (2 DOF: radius, height)", [
        ("r25 h40", Pyramid(radius=0.025, height=0.040)),
        ("r35 h70", Pyramid(radius=0.035, height=0.070)),
        ("r20 h100", Pyramid(radius=0.020, height=0.100)),
    ]),
    ("Torus  (2 DOF: major, minor)", [
        ("R40 t12", Torus(major_radius=0.040, minor_radius=0.012)),
        ("R60 t8 thin", Torus(major_radius=0.060, minor_radius=0.008)),
        ("R30 t18 fat", Torus(major_radius=0.030, minor_radius=0.018)),
    ]),
    ("Box  (3 DOF: dx, dy, dz)", [
        ("cube 50", Box(dimensions=np.array([0.05, 0.05, 0.05]))),
        ("bar 100x40x40", Box(dimensions=np.array([0.10, 0.04, 0.04]))),
        ("post 30x30x120", Box(dimensions=np.array([0.03, 0.03, 0.12]))),
    ]),
    ("Ellipsoid  (3 DOF: rx, ry, rz)", [
        ("40-30-20", Ellipsoid(radii=np.array([0.04, 0.03, 0.02]))),
        ("60-20-20 cigar", Ellipsoid(radii=np.array([0.06, 0.02, 0.02]))),
        ("30-30-60 egg", Ellipsoid(radii=np.array([0.03, 0.03, 0.06]))),
    ]),
    ("Wedge  (3 DOF: width, depth, height)", [
        ("50x40x40", Wedge(width=0.050, depth=0.040, height=0.040)),
        ("100x40x30 ramp", Wedge(width=0.100, depth=0.040, height=0.030)),
        ("40x40x90 tall", Wedge(width=0.040, depth=0.040, height=0.090)),
    ]),
    ("HollowShell  (4 DOF: outer, wall, height, floor)", [
        ("mug R35 t4 H70", HollowShell(outer_radius=0.035, wall_thickness=0.004, height=0.070, floor_thickness=0.006)),
        ("bowl R55 t5 H35", HollowShell(outer_radius=0.055, wall_thickness=0.005, height=0.035, floor_thickness=0.006)),
        ("jar R30 t3 H100", HollowShell(outer_radius=0.030, wall_thickness=0.003, height=0.100, floor_thickness=0.005)),
    ]),
    ("Handle  (4 DOF: major, tube_a, tube_b, arc)", [
        ("arc=180", Handle(major_radius=0.022, tube_a=0.006, tube_b=0.005, arc_angle=np.pi)),
        ("arc=270 C", Handle(major_radius=0.020, tube_a=0.006, tube_b=0.004, arc_angle=1.5 * np.pi)),
        ("flat tube arc=300", Handle(major_radius=0.020, tube_a=0.008, tube_b=0.004, arc_angle=1.65 * np.pi)),
    ]),
    ("Frustum  (3 DOF: radius_bottom, radius_top, height)", [
        ("taper 40->22", Frustum(radius_bottom=0.040, radius_top=0.022, height=0.060)),
        ("flared 22->40 cup", Frustum(radius_bottom=0.022, radius_top=0.040, height=0.060)),
        ("shallow 50->35", Frustum(radius_bottom=0.050, radius_top=0.035, height=0.030)),
    ]),
    ("Hemisphere  (1 DOF: radius)", [
        ("r=20", Hemisphere(radius=0.020)),
        ("r=35", Hemisphere(radius=0.035)),
        ("r=55", Hemisphere(radius=0.055)),
    ]),
    ("HexPrism  (2 DOF: radius, height)", [
        ("nut r18 h12", HexPrism(radius=0.018, height=0.012)),
        ("bolt-head r16 h10", HexPrism(radius=0.016, height=0.010)),
        ("post r12 h40", HexPrism(radius=0.012, height=0.040)),
    ]),
    ("OpenTube  (3 DOF: outer, wall, height)", [
        ("pipe r15 w4 h100", OpenTube(outer_radius=0.015, wall_thickness=0.004, height=0.100)),
        ("ring r25 w4 h20", OpenTube(outer_radius=0.025, wall_thickness=0.004, height=0.020)),
        ("thick r20 w8 h50", OpenTube(outer_radius=0.020, wall_thickness=0.008, height=0.050)),
    ]),
    ("NGonPrism  (3 DOF: n_sides, radius, height)", [
        ("triangle n3", NGonPrism(n_sides=3, radius=0.022, height=0.03)),
        ("pentagon n5", NGonPrism(n_sides=5, radius=0.022, height=0.03)),
        ("octagon n8", NGonPrism(n_sides=8, radius=0.022, height=0.03)),
    ]),
]


def emit_doc():
    """Print a markdown parameter table per type, pulled from PRIMITIVE_SPECS so
    the docs stay in sync with the code."""
    print("| Primitive | DOF | Parameters | default (m) | clamp min | clamp max |")
    print("|---|---|---|---|---|---|")
    for spec in PRIMITIVE_SPECS:
        name = spec.ptype.value
        params = ", ".join(spec.param_names)
        default = ", ".join(f"{v:.3g}" for v in np.exp(spec.init_log_mean))
        lo = ", ".join(f"{v:.3g}" for v in spec.clamp_lo)
        hi = ", ".join(f"{v:.3g}" for v in spec.clamp_hi)
        print(f"| {name} | {len(spec.param_names)} | {params} | {default} "
              f"| {lo} | {hi} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit-doc", action="store_true",
                    help="print markdown spec tables instead of rendering")
    args = ap.parse_args()

    if args.emit_doc:
        emit_doc()
        return

    cells = []
    for type_label, variants in GALLERY:
        first = True
        for vlabel, prim in variants:
            label = (type_label.split("  ")[0] + "\n" + vlabel) if first else vlabel
            cells.append((label, prim.to_mesh()))
            first = False
    grid(cells, OUT / "primitives.png", cols=3,
         title="Primitive library — variants illustrate each type's DOF", cell=1.7)

    WEB.mkdir(parents=True, exist_ok=True)
    shutil.copy(OUT / "primitives.png", WEB / "primitives.png")
    print(f"copied -> {WEB / 'primitives.png'}")


if __name__ == "__main__":
    main()
