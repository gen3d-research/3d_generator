"""
Central archetype registry + the v2 archetype library.

Every hand-written object archetype is registered here in ``ARCHETYPE_REGISTRY``
(name -> zero-arg factory returning a CompositeObject). This is the single
source of truth: ``generator.create_archetype_set``, the Fixed-CAD baseline, and
the analysis/scale-experiment scripts all enumerate this dict instead of keeping
their own hardcoded lists.

Factories keep zero-required-arg signatures (scalar / ndarray defaults) so
``archetype_cem.ArchetypeDistribution`` can introspect them. Multi-part objects
deliberately OVERLAP their parts so the union is a single connected body
(``CompositeObject.is_connected``).
"""

from __future__ import annotations

import numpy as np
from typing import Callable, Dict

from primitives import (
    CompositeObject, Box, Cylinder, Sphere, Capsule,
    Cone, Pyramid, Torus, Ellipsoid, Wedge, HollowShell, Handle, Frustum, Hemisphere,
    seat_height, Transform,
    # existing v1 factories (registered below)
    create_small_box, create_tall_box, create_flat_box, create_mug_like,
    create_l_shape, create_dumbbell, create_hammer, create_bottle,
    create_t_shape, create_u_shape, create_v_shape, create_monitor,
    create_barbell, create_snowman, create_camera, create_frying_pan,
    create_flashlight, create_spatula, create_remote, create_joystick,
)

ARCHETYPE_REGISTRY: Dict[str, Callable[[], CompositeObject]] = {}


def archetype(name: str):
    """Decorator: register a zero-arg factory under ``name``."""
    def deco(fn: Callable[[], CompositeObject]):
        ARCHETYPE_REGISTRY[name] = fn
        fn._archetype_name = name
        return fn
    return deco


# Register the existing v1 archetypes (defined in primitives.py).
for _nm, _fn in [
    ("small_box", create_small_box), ("tall_box", create_tall_box),
    ("flat_box", create_flat_box), ("mug_like", create_mug_like),
    ("l_shape", create_l_shape), ("dumbbell", create_dumbbell),
    ("hammer", create_hammer), ("bottle", create_bottle),
    ("t_shape", create_t_shape), ("u_shape", create_u_shape),
    ("v_shape", create_v_shape), ("monitor", create_monitor),
    ("barbell", create_barbell), ("snowman", create_snowman),
    ("camera", create_camera), ("frying_pan", create_frying_pan),
    ("flashlight", create_flashlight), ("spatula", create_spatula),
    ("remote", create_remote), ("joystick", create_joystick),
]:
    ARCHETYPE_REGISTRY[_nm] = _fn


# ---------------------------------------------------------------------------
# Composition helpers. Each returns a positioned Primitive. Default z seats the
# part on the ground; pass z explicitly to stack. Positions are part CENTERS.
# ---------------------------------------------------------------------------

def _T(x, y, z, euler=None):
    if euler is None:
        return Transform(translation=np.array([float(x), float(y), float(z)]))
    return Transform.from_euler(np.array([float(x), float(y), float(z)]),
                                np.array(euler, dtype=float))


def _box(dx, dy, dz, x=0.0, y=0.0, z=None, euler=None):
    z = dz / 2 if z is None else z
    return Box(dimensions=np.array([dx, dy, dz]), transform=_T(x, y, z, euler))


def _cyl(r, h, x=0.0, y=0.0, z=None, euler=None):
    z = h / 2 if z is None else z
    return Cylinder(radius=r, height=h, transform=_T(x, y, z, euler))


def _sph(r, x=0.0, y=0.0, z=None):
    z = r if z is None else z
    return Sphere(radius=r, transform=_T(x, y, z))


def _cap(r, h, x=0.0, y=0.0, z=None, euler=None):
    z = (h / 2 + r) if z is None else z
    return Capsule(radius=r, height=h, transform=_T(x, y, z, euler))


def _cone(r, h, x=0.0, y=0.0, z=None, euler=None):
    z = h / 4 if z is None else z
    return Cone(radius=r, height=h, transform=_T(x, y, z, euler))


def _pyr(r, h, x=0.0, y=0.0, z=None, euler=None):
    z = h / 4 if z is None else z
    return Pyramid(radius=r, height=h, transform=_T(x, y, z, euler))


def _tor(R, r, x=0.0, y=0.0, z=None, euler=None):
    z = r if z is None else z
    return Torus(major_radius=R, minor_radius=r, transform=_T(x, y, z, euler))


def _shell(R, wall, h, floor=0.005, x=0.0, y=0.0, z=None, euler=None):
    """Open-top hollow container body. Default z seats it on the ground
    (its centroid sits above the base because of the floor)."""
    p = HollowShell(outer_radius=R, wall_thickness=wall, height=h, floor_thickness=floor)
    z = seat_height(p) if z is None else z
    p.transform = _T(x, y, z, euler)
    return p


def _handle(R, a, b, arc=1.5 * np.pi, x=0.0, y=0.0, z=None, euler=None):
    z = a if z is None else z
    return Handle(major_radius=R, tube_a=a, tube_b=b, arc_angle=arc,
                  transform=_T(x, y, z, euler))


def _frustum(rb, rt, h, x=0.0, y=0.0, z=None, euler=None):
    """Truncated cone. Default z seats its (rb) base on the ground."""
    p = Frustum(radius_bottom=rb, radius_top=rt, height=h)
    z = seat_height(p) if z is None else z
    p.transform = _T(x, y, z, euler)
    return p


def _hemi(r, x=0.0, y=0.0, z=None, euler=None):
    p = Hemisphere(radius=r)
    z = seat_height(p) if z is None else z
    p.transform = _T(x, y, z, euler)
    return p


def _ell(rx, ry, rz, x=0.0, y=0.0, z=None, euler=None):
    z = rz if z is None else z
    return Ellipsoid(radii=np.array([rx, ry, rz]), transform=_T(x, y, z, euler))


def _wedge(w, d, h, x=0.0, y=0.0, z=None, euler=None):
    z = h / 3 if z is None else z
    return Wedge(width=w, depth=d, height=h, transform=_T(x, y, z, euler))


def _co(name, *prims, friction=1.0):
    return CompositeObject(primitives=list(prims), name=name, friction=float(friction))


# ===========================================================================
# Group A — tools
# ===========================================================================

@archetype("screwdriver")
def create_screwdriver(handle_r: float = 0.014, handle_h: float = 0.06,
                       shaft_r: float = 0.004, shaft_h: float = 0.08) -> CompositeObject:
    handle = _cap(handle_r, handle_h)
    top = handle_h / 2 + handle_r
    shaft = _cyl(shaft_r, shaft_h, z=top + shaft_h / 2 - 0.01)
    tip = _cone(shaft_r * 1.2, 0.015, z=top + shaft_h - 0.01)
    return _co("screwdriver", handle, shaft, tip)


@archetype("mallet")
def create_mallet(handle_r: float = 0.012, handle_h: float = 0.12,
                  head_r: float = 0.025, head_h: float = 0.05) -> CompositeObject:
    handle = _cyl(handle_r, handle_h)
    head = _cyl(head_r, head_h, z=handle_h - 0.005, euler=[np.pi / 2, 0, 0])
    return _co("mallet", handle, head)


@archetype("wrench")
def create_wrench(handle_l: float = 0.10, handle_w: float = 0.018,
                  thick: float = 0.008, head_r: float = 0.02) -> CompositeObject:
    handle = _box(handle_w, handle_l, thick, z=thick / 2)
    head = _tor(head_r, thick * 0.6, y=handle_l / 2, z=thick / 2)
    return _co("wrench", handle, head)


@archetype("chisel")
def create_chisel(handle_r: float = 0.013, handle_h: float = 0.06,
                  blade_w: float = 0.02, blade_l: float = 0.07) -> CompositeObject:
    handle = _cap(handle_r, handle_h)
    top = handle_h / 2 + handle_r
    blade = _box(blade_w, 0.004, blade_l, z=top + blade_l / 2 - 0.01)
    return _co("chisel", handle, blade)


@archetype("paintbrush")
def create_paintbrush(handle_r: float = 0.008, handle_h: float = 0.12,
                      ferrule_r: float = 0.011, bristle_h: float = 0.03) -> CompositeObject:
    handle = _cyl(handle_r, handle_h)
    ferrule = _cyl(ferrule_r, 0.02, z=handle_h - 0.005)
    bristles = _box(ferrule_r * 1.6, 0.006, bristle_h, z=handle_h + bristle_h / 2 - 0.005)
    return _co("paintbrush", handle, ferrule, bristles)


@archetype("plunger")
def create_plunger(handle_r: float = 0.011, handle_h: float = 0.12,
                   cup_r: float = 0.04, cup_h: float = 0.05) -> CompositeObject:
    cup = _frustum(cup_r, cup_r * 0.45, cup_h)        # flared rubber bell (not a point)
    handle = _cyl(handle_r, handle_h, z=cup_h - 0.006 + handle_h / 2)
    return _co("plunger", cup, handle)


@archetype("axe")
def create_axe(handle_r: float = 0.012, handle_h: float = 0.14,
               blade_w: float = 0.05, blade_h: float = 0.05) -> CompositeObject:
    handle = _cyl(handle_r, handle_h)
    blade = _wedge(blade_w, 0.012, blade_h, x=blade_w / 2 - 0.005,
                   z=handle_h - blade_h / 3)
    return _co("axe", handle, blade)


@archetype("file_tool")
def create_file_tool(handle_r: float = 0.012, handle_h: float = 0.05,
                     blade_w: float = 0.012, blade_l: float = 0.10) -> CompositeObject:
    handle = _cap(handle_r, handle_h)
    top = handle_h / 2 + handle_r
    blade = _box(blade_w, blade_w, blade_l, z=top + blade_l / 2 - 0.01)
    return _co("file_tool", handle, blade)


@archetype("allen_key")
def create_allen_key(long_l: float = 0.08, short_l: float = 0.03,
                     bar: float = 0.006) -> CompositeObject:
    long_arm = _box(bar, bar, long_l, z=long_l / 2)
    short_arm = _box(short_l, bar, bar, x=short_l / 2 - bar / 2, z=long_l - bar / 2)
    return _co("allen_key", long_arm, short_arm)


@archetype("spirit_level")
def create_spirit_level(length: float = 0.14, w: float = 0.025,
                        h: float = 0.03, vial_r: float = 0.006) -> CompositeObject:
    body = _box(length, w, h, z=h / 2)
    vial = _cyl(vial_r, 0.03, z=h - 0.003, euler=[np.pi / 2, 0, 0])
    return _co("spirit_level", body, vial)


@archetype("scraper")
def create_scraper(handle_r: float = 0.013, handle_h: float = 0.05,
                   blade_w: float = 0.05, blade_l: float = 0.04) -> CompositeObject:
    handle = _cap(handle_r, handle_h)
    top = handle_h / 2 + handle_r
    blade = _box(blade_w, 0.003, blade_l, z=top + blade_l / 2 - 0.01)
    return _co("scraper", handle, blade)


@archetype("drill")
def create_drill(body_w: float = 0.06, body_h: float = 0.06, body_d: float = 0.03,
                 grip_h: float = 0.08, bit_l: float = 0.06) -> CompositeObject:
    body = _box(body_w, body_d, body_h, z=grip_h + body_h / 2)
    grip = _box(body_w * 0.5, body_d, grip_h, z=grip_h / 2)
    bit = _cyl(0.005, bit_l, x=body_w / 2 + bit_l / 2 - 0.005, z=grip_h + body_h / 2,
               euler=[0, np.pi / 2, 0])
    return _co("drill", body, grip, bit)


# ===========================================================================
# Group B — kitchen / household
# ===========================================================================

@archetype("ladle")
def create_ladle(handle_r: float = 0.007, handle_h: float = 0.12,
                 bowl_r: float = 0.03) -> CompositeObject:
    handle = _cyl(handle_r, handle_h)
    bowl = _hemi(bowl_r, z=handle_h - 0.004, euler=[np.pi, 0, 0])   # dome scoop, not a ball
    return _co("ladle", handle, bowl)


@archetype("knife")
def create_knife(handle_l: float = 0.05, handle_w: float = 0.018,
                 blade_l: float = 0.10, blade_w: float = 0.022) -> CompositeObject:
    handle = _box(handle_w, handle_l, 0.014, z=0.007)
    blade = _box(blade_w, blade_l, 0.004, y=handle_l / 2 + blade_l / 2 - 0.005, z=0.007)
    return _co("knife", handle, blade)


@archetype("fork_simple")
def create_fork_simple(handle_l: float = 0.09, handle_w: float = 0.012,
                       head_l: float = 0.04, head_w: float = 0.025) -> CompositeObject:
    handle = _box(handle_w, handle_l, 0.006, z=0.003)
    head = _box(head_w, head_l, 0.006, y=handle_l / 2 + head_l / 2 - 0.005, z=0.003)
    return _co("fork_simple", handle, head)


@archetype("bowl")
def create_bowl(outer_r: float = 0.05, height: float = 0.04) -> CompositeObject:
    base = _cyl(outer_r * 0.5, 0.008)
    # real hollow bowl (open top, walls + floor) instead of a solid disc stack
    body = _shell(outer_r, 0.005, height, floor=0.006, z=0.005 + seat_height(
        HollowShell(outer_radius=outer_r, wall_thickness=0.005, height=height, floor_thickness=0.006)))
    return _co("bowl", base, body)


@archetype("cup")
def create_cup(r: float = 0.035, h: float = 0.07) -> CompositeObject:
    body = _shell(r, 0.004, h, floor=0.006)          # hollow cup body
    handle = _handle(0.017, 0.005, 0.004, arc=1.4 * np.pi,
                     x=r + 0.002, z=h * 0.5, euler=[np.pi / 2, 0, 0])   # C-handle
    return _co("cup", body, handle)


@archetype("mug_like")     # overrides the v1 solid-cylinder + straight-bar version
def create_mug(r: float = 0.04, h: float = 0.09) -> CompositeObject:
    body = _shell(r, 0.004, h, floor=0.006)          # hollow mug body
    handle = _handle(0.02, 0.006, 0.005, arc=1.4 * np.pi,
                     x=r + 0.002, z=h * 0.5, euler=[np.pi / 2, 0, 0])   # C-handle
    return _co("mug_like", body, handle)


@archetype("teapot")
def create_teapot(body_r: float = 0.045, body_h: float = 0.06,
                  spout_l: float = 0.05) -> CompositeObject:
    body = _ell(body_r, body_r, body_h / 2, z=body_h / 2)
    lid = _cap(0.012, 0.012, z=body_h - 0.004)
    spout = _cyl(0.008, spout_l, x=body_r * 0.55, z=body_h * 0.6, euler=[0, np.pi / 3, 0])
    handle = _handle(0.02, 0.006, 0.005, arc=1.5 * np.pi,
                     x=-body_r - 0.002, z=body_h / 2, euler=[np.pi / 2, 0, np.pi])
    return _co("teapot", body, lid, spout, handle)


@archetype("rolling_pin")
def create_rolling_pin(body_r: float = 0.025, body_l: float = 0.12,
                       handle_r: float = 0.01, handle_l: float = 0.03) -> CompositeObject:
    body = _cyl(body_r, body_l, z=body_r, euler=[np.pi / 2, 0, 0])
    h1 = _cyl(handle_r, handle_l, y=body_l / 2 + handle_l / 2 - 0.005, z=body_r, euler=[np.pi / 2, 0, 0])
    h2 = _cyl(handle_r, handle_l, y=-body_l / 2 - handle_l / 2 + 0.005, z=body_r, euler=[np.pi / 2, 0, 0])
    return _co("rolling_pin", body, h1, h2)


@archetype("funnel")
def create_funnel(top_r: float = 0.04, cone_h: float = 0.05,
                  tube_r: float = 0.008, tube_h: float = 0.04) -> CompositeObject:
    tube = _cyl(tube_r, tube_h)
    cone = _cone(top_r, cone_h, z=tube_h - 0.005 + cone_h / 4)
    return _co("funnel", tube, cone)


@archetype("wine_glass")
def create_wine_glass(base_r: float = 0.03, stem_h: float = 0.06,
                      bowl_r: float = 0.03) -> CompositeObject:
    base = _cyl(base_r, 0.006)
    stem = _cyl(0.004, stem_h, z=0.006 + stem_h / 2 - 0.002)
    bowl = _ell(bowl_r, bowl_r, bowl_r, z=0.006 + stem_h + bowl_r - 0.004)
    return _co("wine_glass", base, stem, bowl)


@archetype("pot")
def create_pot(r: float = 0.05, h: float = 0.06) -> CompositeObject:
    body = _shell(r, 0.005, h, floor=0.006)          # hollow pot body
    h1 = _handle(0.014, 0.005, 0.004, arc=1.2 * np.pi,
                 x=r + 0.001, z=h * 0.7, euler=[np.pi / 2, 0, 0])
    h2 = _handle(0.014, 0.005, 0.004, arc=1.2 * np.pi,
                 x=-r - 0.001, z=h * 0.7, euler=[np.pi / 2, 0, np.pi])
    return _co("pot", body, h1, h2)


@archetype("cutting_board")
def create_cutting_board(w: float = 0.12, d: float = 0.09, t: float = 0.012,
                         handle_r: float = 0.012) -> CompositeObject:
    board = _box(w, d, t, z=t / 2)
    hole = _tor(handle_r, 0.004, y=d / 2 - 0.005, z=t / 2)
    return _co("cutting_board", board, hole)


@archetype("pestle")
def create_pestle(r: float = 0.013, h: float = 0.10, head_r: float = 0.02) -> CompositeObject:
    shaft = _cyl(r, h)
    head = _sph(head_r, z=0.005)
    return _co("pestle", head, shaft)


# ===========================================================================
# Group C — electronics
# ===========================================================================

@archetype("mouse")
def create_mouse(w: float = 0.06, d: float = 0.10, h: float = 0.03) -> CompositeObject:
    body = _ell(w / 2, d / 2, h, z=h)
    return _co("mouse", body)


@archetype("speaker")
def create_speaker(w: float = 0.08, d: float = 0.08, h: float = 0.12,
                   cone_r: float = 0.03) -> CompositeObject:
    body = _box(w, d, h, z=h / 2)
    driver = _cone(cone_r, 0.02, y=d / 2 - 0.005, z=h * 0.6, euler=[-np.pi / 2, 0, 0])
    return _co("speaker", body, driver)


@archetype("headphones")
def create_headphones(band_r: float = 0.06, cup_r: float = 0.025) -> CompositeObject:
    band = _tor(band_r, 0.006, z=band_r, euler=[np.pi / 2, 0, 0])
    c1 = _cyl(cup_r, 0.02, x=band_r, z=band_r * 0.4, euler=[0, np.pi / 2, 0])
    c2 = _cyl(cup_r, 0.02, x=-band_r, z=band_r * 0.4, euler=[0, np.pi / 2, 0])
    return _co("headphones", band, c1, c2)


@archetype("game_controller")
def create_game_controller(w: float = 0.11, d: float = 0.06, h: float = 0.03) -> CompositeObject:
    body = _box(w, d, h, z=h / 2)
    g1 = _cap(0.014, 0.03, x=-w / 2 + 0.01, z=h / 2, euler=[0, 0, 0])
    g2 = _cap(0.014, 0.03, x=w / 2 - 0.01, z=h / 2, euler=[0, 0, 0])
    stick = _cyl(0.008, 0.012, z=h)
    return _co("game_controller", body, g1, g2, stick)


@archetype("usb_stick")
def create_usb_stick(l: float = 0.05, w: float = 0.018, h: float = 0.008) -> CompositeObject:
    body = _box(w, l, h, z=h / 2)
    conn = _box(w * 0.7, 0.018, h * 0.6, y=l / 2 + 0.007, z=h / 2)
    return _co("usb_stick", body, conn)


@archetype("webcam")
def create_webcam(r: float = 0.022, base_w: float = 0.05) -> CompositeObject:
    base = _box(base_w, 0.02, 0.012, z=0.006)
    body = _cyl(r, 0.03, z=0.012 + r - 0.003, euler=[np.pi / 2, 0, 0])
    lens = _cyl(0.008, 0.01, y=0.012, z=0.012 + r - 0.003, euler=[np.pi / 2, 0, 0])
    return _co("webcam", base, body, lens)


@archetype("microphone")
def create_microphone(r: float = 0.02, h: float = 0.10, head_r: float = 0.025) -> CompositeObject:
    body = _cyl(r, h)
    head = _sph(head_r, z=h - 0.005)
    return _co("microphone", body, head)


@archetype("tablet")
def create_tablet(w: float = 0.12, d: float = 0.09, t: float = 0.008) -> CompositeObject:
    return _co("tablet", _box(w, d, t, z=t / 2))


@archetype("smartwatch")
def create_smartwatch(face_r: float = 0.022, band_w: float = 0.022) -> CompositeObject:
    face = _box(face_r * 2, face_r * 2, 0.012, z=0.006)
    b1 = _box(band_w, 0.04, 0.004, y=face_r + 0.018, z=0.004)
    b2 = _box(band_w, 0.04, 0.004, y=-face_r - 0.018, z=0.004)
    return _co("smartwatch", face, b1, b2)


@archetype("router")
def create_router(w: float = 0.10, d: float = 0.07, h: float = 0.025,
                  ant_h: float = 0.06) -> CompositeObject:
    body = _box(w, d, h, z=h / 2)
    a1 = _cyl(0.004, ant_h, x=-w / 2 + 0.01, y=-d / 2 + 0.01, z=h + ant_h / 2 - 0.004)
    a2 = _cyl(0.004, ant_h, x=w / 2 - 0.01, y=-d / 2 + 0.01, z=h + ant_h / 2 - 0.004)
    return _co("router", body, a1, a2)


# ===========================================================================
# Group D — toys / sports
# ===========================================================================

@archetype("spinning_top")
def create_spinning_top(r: float = 0.03, cone_h: float = 0.04,
                        stem_h: float = 0.025) -> CompositeObject:
    cone = _cone(r, cone_h, euler=[np.pi, 0, 0], z=cone_h * 0.75)
    stem = _cyl(0.004, stem_h, z=cone_h + stem_h / 2 - 0.005)
    return _co("spinning_top", cone, stem)


@archetype("toy_rocket")
def create_toy_rocket(body_r: float = 0.022, body_h: float = 0.09,
                      nose_h: float = 0.035) -> CompositeObject:
    body = _cyl(body_r, body_h)
    nose = _cone(body_r, nose_h, z=body_h - 0.005 + nose_h / 4)
    f1 = _wedge(0.02, 0.004, 0.03, x=body_r, z=0.015, euler=[0, 0, 0])
    f2 = _wedge(0.02, 0.004, 0.03, x=-body_r, z=0.015, euler=[0, 0, np.pi])
    return _co("toy_rocket", body, nose, f1, f2)


@archetype("die")
def create_die(side: float = 0.04) -> CompositeObject:
    return _co("die", _box(side, side, side))


@archetype("kettlebell")
def create_kettlebell(ball_r: float = 0.035, handle_r: float = 0.022) -> CompositeObject:
    ball = _sph(ball_r)
    handle = _tor(handle_r, 0.006, z=ball_r * 2 - 0.004, euler=[0, np.pi / 2, 0])
    return _co("kettlebell", ball, handle)


@archetype("bowling_pin")
def create_bowling_pin(r: float = 0.025, h: float = 0.12) -> CompositeObject:
    base = _cyl(r, h * 0.4)
    neck = _cap(r * 0.55, h * 0.4, z=h * 0.4 + h * 0.2)
    return _co("bowling_pin", base, neck)


@archetype("trophy")
def create_trophy(base_w: float = 0.05, cup_r: float = 0.03,
                  stem_h: float = 0.04) -> CompositeObject:
    base = _box(base_w, base_w, 0.012, z=0.006)
    stem = _cyl(0.006, stem_h, z=0.012 + stem_h / 2 - 0.002)
    # flared open cup (narrow at the stem, wide rim) instead of a solid inverted cone
    cup = _frustum(0.012, cup_r, 0.045, z=0.01 + stem_h + 0.018)
    return _co("trophy", base, stem, cup)


@archetype("baseball_bat")
def create_baseball_bat(r: float = 0.02, h: float = 0.16, knob_r: float = 0.012) -> CompositeObject:
    bat = _cap(r, h)
    knob = _sph(knob_r, z=0.005)
    return _co("baseball_bat", knob, bat)


@archetype("hockey_puck")
def create_hockey_puck(r: float = 0.038, h: float = 0.025) -> CompositeObject:
    return _co("hockey_puck", _cyl(r, h))


@archetype("ring_stack")
def create_ring_stack(base_r: float = 0.04) -> CompositeObject:
    # Fat central peg so each ring's inner hole clearly overlaps it (one body).
    peg = _cyl(0.032, 0.08)
    r1 = _tor(base_r, 0.018, z=0.018)
    r2 = _tor(base_r * 0.88, 0.016, z=0.044)
    r3 = _tor(base_r * 0.78, 0.015, z=0.066)
    return _co("ring_stack", peg, r1, r2, r3)


@archetype("toy_car")
def create_toy_car(body_l: float = 0.09, body_w: float = 0.04,
                   body_h: float = 0.022, wheel_r: float = 0.014) -> CompositeObject:
    chassis = _box(body_w, body_l, body_h, z=wheel_r)
    cabin = _box(body_w * 0.8, body_l * 0.45, body_h, z=wheel_r + body_h - 0.004)
    w = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            w.append(_cyl(wheel_r, 0.008, x=sx * body_w / 2, y=sy * body_l / 3,
                          z=wheel_r, euler=[0, np.pi / 2, 0]))
    return _co("toy_car", chassis, cabin, *w)


# ===========================================================================
# Group E — containers / office
# ===========================================================================

@archetype("jar")
def create_jar(r: float = 0.035, h: float = 0.08, lid_h: float = 0.015) -> CompositeObject:
    # Open wide-mouth jar with a threaded neck band. A SEALING lid would trap the
    # cavity as an enclosed void -> the union mesh has two disconnected surfaces
    # (outer + inner) and fails is_connected(); the open ring band avoids that.
    body = _shell(r, 0.004, h, floor=0.005)
    rim = _tor(r, 0.006, z=h - 0.004)
    return _co("jar", body, rim)


@archetype("crate")
def create_crate(w: float = 0.10, d: float = 0.08, h: float = 0.07, t: float = 0.008) -> CompositeObject:
    base = _box(w, d, t, z=t / 2)
    s1 = _box(t, d, h, x=w / 2 - t / 2, z=h / 2)
    s2 = _box(t, d, h, x=-w / 2 + t / 2, z=h / 2)
    s3 = _box(w, t, h, y=d / 2 - t / 2, z=h / 2)
    s4 = _box(w, t, h, y=-d / 2 + t / 2, z=h / 2)
    return _co("crate", base, s1, s2, s3, s4)


@archetype("tape_roll")
def create_tape_roll(R: float = 0.04, r: float = 0.012) -> CompositeObject:
    return _co("tape_roll", _tor(R, r))


@archetype("pencil")
def create_pencil(r: float = 0.005, h: float = 0.13, tip_h: float = 0.015) -> CompositeObject:
    body = _cyl(r, h)
    tip = _cone(r, tip_h, z=h - 0.003 + tip_h / 4)
    return _co("pencil", body, tip)


@archetype("marker")
def create_marker(r: float = 0.009, h: float = 0.11, cap_h: float = 0.03) -> CompositeObject:
    body = _cyl(r, h)
    cap = _cyl(r * 1.1, cap_h, z=h - 0.004 + cap_h / 2)
    return _co("marker", body, cap)


@archetype("glue_stick")
def create_glue_stick(r: float = 0.012, h: float = 0.07) -> CompositeObject:
    body = _cyl(r, h)
    cap = _cap(r * 1.05, h * 0.3, z=h - 0.005)
    return _co("glue_stick", body, cap)


@archetype("stapler")
def create_stapler(l: float = 0.12, w: float = 0.03, h: float = 0.025) -> CompositeObject:
    base = _box(w, l, h * 0.5, z=h * 0.25)
    top = _box(w, l * 0.85, h * 0.5, y=-l * 0.05, z=h * 0.7)
    return _co("stapler", base, top)


@archetype("binder_ring")
def create_binder_ring(R: float = 0.03, spine_h: float = 0.10) -> CompositeObject:
    spine = _box(0.015, 0.012, spine_h, z=spine_h / 2)
    r1 = _tor(R, 0.004, y=0.01, z=spine_h * 0.3, euler=[np.pi / 2, 0, 0])
    r2 = _tor(R, 0.004, y=0.01, z=spine_h * 0.7, euler=[np.pi / 2, 0, 0])
    return _co("binder_ring", spine, r1, r2)


# ===========================================================================
# Group F — connectors / brackets / fasteners
# ===========================================================================

@archetype("bolt")
def create_bolt(shaft_r: float = 0.008, shaft_h: float = 0.06,
                head_r: float = 0.016) -> CompositeObject:
    head = _pyr(head_r, 0.012)
    shaft = _cyl(shaft_r, shaft_h, z=0.012 / 2 + shaft_h / 2 - 0.003)
    return _co("bolt", head, shaft)


@archetype("nut")
def create_nut(r: float = 0.018, h: float = 0.012, hole_r: float = 0.007) -> CompositeObject:
    body = _cyl(r, h)
    ring = _tor(r * 0.7, 0.004, z=h / 2)
    return _co("nut", body, ring)


@archetype("washer")
def create_washer(R: float = 0.02, r: float = 0.005) -> CompositeObject:
    return _co("washer", _tor(R, r))


@archetype("hook")
def create_hook(base_h: float = 0.05, hook_r: float = 0.02) -> CompositeObject:
    post = _cyl(0.006, base_h)
    curve = _tor(hook_r, 0.005, z=base_h - 0.004, euler=[np.pi / 2, 0, 0])
    return _co("hook", post, curve)


@archetype("pipe_elbow")
def create_pipe_elbow(r: float = 0.018, seg: float = 0.05) -> CompositeObject:
    horiz = _cyl(r, seg, x=seg / 2 - 0.004, z=r, euler=[0, np.pi / 2, 0])
    vert = _cyl(r, seg, z=r + seg / 2 - 0.004)
    joint = _sph(r, z=r)
    return _co("pipe_elbow", joint, horiz, vert)


@archetype("cross_joint")
def create_cross_joint(r: float = 0.012, arm: float = 0.05) -> CompositeObject:
    c = _sph(r * 1.4, z=r * 1.4)
    z = r * 1.4
    ax = _cyl(r, arm, x=arm / 2 - 0.004, z=z, euler=[0, np.pi / 2, 0])
    ax2 = _cyl(r, arm, x=-arm / 2 + 0.004, z=z, euler=[0, np.pi / 2, 0])
    ay = _cyl(r, arm, y=arm / 2 - 0.004, z=z, euler=[np.pi / 2, 0, 0])
    ay2 = _cyl(r, arm, y=-arm / 2 + 0.004, z=z, euler=[np.pi / 2, 0, 0])
    return _co("cross_joint", c, ax, ax2, ay, ay2)


@archetype("gear_like")
def create_gear_like(r: float = 0.035, h: float = 0.015, n_teeth: int = 8) -> CompositeObject:
    body = _cyl(r, h)
    teeth = []
    for k in range(int(n_teeth)):
        a = 2 * np.pi * k / int(n_teeth)
        teeth.append(_box(0.01, 0.01, h, x=r * np.cos(a), y=r * np.sin(a), z=h / 2))
    return _co("gear_like", body, *teeth)


@archetype("bracket_angle")
def create_bracket_angle(l: float = 0.07, w: float = 0.04, t: float = 0.01) -> CompositeObject:
    horiz = _box(w, l, t, z=t / 2)
    vert = _box(w, t, l, y=-l / 2 + t / 2, z=l / 2)
    return _co("bracket_angle", horiz, vert)


# ---------------------------------------------------------------------------

def all_archetypes() -> Dict[str, Callable[[], CompositeObject]]:
    """Return the full registry (name -> factory)."""
    return dict(ARCHETYPE_REGISTRY)
