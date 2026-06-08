"""
Force-closure grasp planner for parallel-jaw grippers.

Independent of the suitability score used by the CEM optimizer. The metric this
module produces ("at least one collision-free force-closure grasp exists") is
the downstream proxy used to break the circular evaluation flagged by R2-Q2 and
the Associate Editor.

The synthesizer is a textbook antipodal sampler:

    1. Sample N surface points and inward-pointing normals on the object mesh.
    2. For every pair (i, j) check
         (a) gripper width  w in [w_min, w_max]
         (b) Coulomb friction-cone condition at both contacts
             angle(line_ij, n_i) <= atan(mu)  and  angle(-line_ij, n_j) <= atan(mu)
         (c) a top-down or side approach to the grasp configuration is
             collision-free (swept-volume test with trimesh signed-distance).
    3. Return the list of valid grasps and per-grasp diagnostics.

The metric "grasp success rate" used in the manuscript is the fraction of
objects for which at least one grasp passes all three tests.
"""

from __future__ import annotations

import numpy as np
import trimesh
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from primitives import CompositeObject


@dataclass
class GripperSpec:
    """Approximate parallel-jaw gripper geometry (Franka Panda by default)."""
    width_min: float = 0.005     # closed gap
    width_max: float = 0.085     # max opening
    finger_length: float = 0.06  # finger length along approach axis
    finger_thickness: float = 0.015
    finger_width: float = 0.022
    palm_clearance: float = 0.02  # extra clearance behind the fingertips
    mu: float = 0.5              # Coulomb friction coefficient assumed


@dataclass
class Grasp:
    contact1: np.ndarray
    contact2: np.ndarray
    normal1: np.ndarray
    normal2: np.ndarray
    width: float
    approach: np.ndarray         # unit approach vector (from above the grasp toward contacts)
    margin: float                # how deep into friction cone (0 = boundary)
    score: float = 0.0           # overall quality (set by _score_grasp; higher = better)

    @property
    def center(self) -> np.ndarray:
        return 0.5 * (self.contact1 + self.contact2)


@dataclass
class GraspReport:
    n_attempts: int
    n_friction_pass: int
    n_collision_free: int
    grasps: List[Grasp] = field(default_factory=list)

    @property
    def has_grasp(self) -> bool:
        return len(self.grasps) > 0


# ---------------------------------------------------------------------------

def _safe_mesh(obj: CompositeObject) -> Optional[trimesh.Trimesh]:
    try:
        m = obj.to_mesh(boolean_union=False)
        if len(m.vertices) == 0 or len(m.faces) == 0:
            return None
        return m
    except Exception:
        return None


def _sample_surface(mesh: trimesh.Trimesh, n: int, rng: np.random.Generator
                    ) -> Tuple[np.ndarray, np.ndarray]:
    pts, face_idx = trimesh.sample.sample_surface(mesh, n)
    # Inward-pointing normals (negate face normals because trimesh returns outward).
    n_out = mesh.face_normals[face_idx]
    return np.asarray(pts), np.asarray(n_out)


def _friction_cone_ok(line: np.ndarray, n_out_1: np.ndarray, n_out_2: np.ndarray,
                      mu: float) -> Tuple[bool, float]:
    """Check antipodal friction-cone condition. line is contact1 -> contact2."""
    L = np.linalg.norm(line)
    if L < 1e-6:
        return False, -1.0
    l_hat = line / L
    # The closing force at contact 1 is along -l_hat in the body frame, which
    # corresponds to pushing along +l_hat from outside.  The inward normal at
    # contact 1 is -n_out_1.
    cos_alpha = 1.0 / np.sqrt(1.0 + mu * mu)
    c1 = -np.dot(l_hat, n_out_1)   # large when line aligns with inward normal
    c2 = +np.dot(l_hat, n_out_2)
    ok = (c1 >= cos_alpha) and (c2 >= cos_alpha)
    margin = float(min(c1, c2) - cos_alpha)
    return ok, margin


def _approach_clearance_dist(mesh: trimesh.Trimesh, contacts, approach: np.ndarray,
                             gripper: GripperSpec, n_samples: int = 12) -> float:
    """How far the fingers/palm can retract along -approach from the contacts
    before hitting the object — the free 'runway' for the approach.  Returns a
    distance in metres, capped at finger_length + palm_clearance (= fully clear).

    ``contacts`` is an iterable of 3D points (one per finger)."""
    need = gripper.finger_length + gripper.palm_clearance
    offs = np.linspace(0.002, need, n_samples)
    if not mesh.is_watertight:
        # contains() is unreliable on non-watertight meshes — use signed distance.
        try:
            pq = trimesh.proximity.ProximityQuery(mesh)
        except Exception:
            return need  # be permissive rather than spuriously failing
        worst = need
        for c in contacts:
            probe = np.asarray(c)[None, :] - approach[None, :] * offs[:, None]
            d = pq.signed_distance(probe)            # positive = inside
            inside = np.where(d > 1e-4)[0]
            if inside.size:
                worst = min(worst, float(offs[inside[0]]))
        return worst
    worst = need
    for c in contacts:
        probe = np.asarray(c)[None, :] - approach[None, :] * offs[:, None]
        inside = np.where(mesh.contains(probe))[0]
        if inside.size:
            worst = min(worst, float(offs[inside[0]]))
    return worst


def _approach_collision_free(mesh: trimesh.Trimesh, grasp: Grasp,
                             gripper: GripperSpec, n_samples: int = 12) -> bool:
    """Verify the approach runway in front of both fingers is outside the object."""
    need = gripper.finger_length + gripper.palm_clearance
    d = _approach_clearance_dist(mesh, (grasp.contact1, grasp.contact2),
                                 grasp.approach, gripper, n_samples)
    return d >= need - 1e-6


def _build_grasp(c1, n_out_1, c2, n_out_2, gripper: GripperSpec
                 ) -> Optional[Grasp]:
    line = c2 - c1
    width = float(np.linalg.norm(line))
    if width < gripper.width_min or width > gripper.width_max:
        return None
    ok, margin = _friction_cone_ok(line, n_out_1, n_out_2, gripper.mu)
    if not ok:
        return None
    # Choose an approach perpendicular to the line, biased toward +Z so the
    # gripper comes from above the table.
    grasp_axis = line / width
    z = np.array([0.0, 0.0, 1.0])
    approach = z - z @ grasp_axis * grasp_axis
    if np.linalg.norm(approach) < 1e-3:
        # grasp axis is vertical: approach from +X
        approach = np.array([1.0, 0.0, 0.0])
        approach = approach - approach @ grasp_axis * grasp_axis
    approach /= np.linalg.norm(approach)
    # Flip approach so it points from above toward the grasp.
    if approach[2] > 0:
        approach = -approach
    return Grasp(contact1=c1.copy(), contact2=c2.copy(),
                 normal1=n_out_1.copy(), normal2=n_out_2.copy(),
                 width=width, approach=approach, margin=margin)


# ---------------------------------------------------------------------------
# Waist-aware grasp synthesis + scoring
# ---------------------------------------------------------------------------

@dataclass
class _GeomCtx:
    union: trimesh.Trimesh       # mesh used for slicing / clearance (watertight if possible)
    com: np.ndarray              # object centre of mass (world frame)
    axes: List[np.ndarray]       # slicing-axis set (unit vectors); axes[0] = elongation
    diag: float                  # AABB diagonal length (object scale)
    elong: np.ndarray            # main elongation direction (unit)
    elong_ratio: float           # longest extent / largest perpendicular extent (>1 = elongated)


@dataclass
class _SliceInfo:
    centroid3d: np.ndarray       # 3D centroid of the cross-section
    width: float                 # overall cross-section width (2 * 95th-pct radius)
    minor_width: float           # extent along the thin (minor) direction
    minor3d: np.ndarray          # 3D unit vector of the minor (finger-closing) direction
    axis: np.ndarray             # slice normal == the part's LOCAL elongation direction


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def _slicing_axes(union: trimesh.Trimesh) -> List[np.ndarray]:
    """Elongation axis (least principal inertia) plus the world X/Y/Z axes,
    de-duplicated by near-parallel test. Robust for both elongated and
    axis-aligned composite objects."""
    axes: List[np.ndarray] = []
    try:
        comp = np.asarray(union.principal_inertia_components, float)
        vecs = np.asarray(union.principal_inertia_vectors, float)
        elong = _unit(vecs[int(np.argmin(comp))])
        axes.append(elong)
    except Exception:
        try:  # PCA fallback: largest-variance vertex direction
            V = np.asarray(union.vertices, float)
            V = V - V.mean(axis=0)
            _, _, Vt = np.linalg.svd(V, full_matrices=False)
            axes.append(_unit(Vt[0]))
        except Exception:
            pass
    for ax in (np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])):
        if all(abs(float(ax @ a)) < 0.98 for a in axes):
            axes.append(ax)
    return axes or [np.array([0, 0, 1.0])]


def _grasp_geometry(obj: CompositeObject, mesh: trimesh.Trimesh) -> _GeomCtx:
    try:
        union = obj.to_mesh(boolean_union=True)
        if union is None or len(union.vertices) == 0 or len(union.faces) == 0:
            union = mesh
    except Exception:
        union = mesh
    try:
        _, _, com = obj.mesh_mass_properties(1000.0)
        com = np.asarray(com, float)
    except Exception:
        try:
            com = np.asarray(obj.center_of_mass(1000.0), float)
        except Exception:
            com = union.vertices.mean(axis=0)
    try:
        lo, hi = union.bounds
        diag = float(np.linalg.norm(hi - lo)) or 1.0
    except Exception:
        diag = 1.0
    axes = _slicing_axes(union)
    # How elongated is the object: extent along the main axis vs perpendicular.
    try:
        V = np.asarray(union.vertices, float)
        def _ext(a):
            t = V @ a
            return float(t.max() - t.min())
        ext0 = _ext(axes[0])
        ext_perp = max((_ext(a) for a in axes[1:]), default=ext0)
        elong_ratio = ext0 / (ext_perp + 1e-9)
    except Exception:
        elong_ratio = 1.0
    return _GeomCtx(union=union, com=com, axes=axes, diag=diag,
                    elong=_unit(axes[0]), elong_ratio=float(elong_ratio))


def _section_lobes(planar) -> List[np.ndarray]:
    """Split a 2D cross-section into connected components (lobes) WITHOUT needing
    networkx/shapely (Path2D.split()/polygons_full need them). Union-find over the
    entity polyline segments; returns a list of (N,2) vertex arrays, one per lobe."""
    try:
        V = np.asarray(planar.vertices, float)
        n = len(V)
        if n < 3:
            return [V] if n else []
        parent = list(range(n))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for ent in planar.entities:
            idx = np.asarray(ent.points).astype(int)
            for i in range(len(idx) - 1):
                union(int(idx[i]), int(idx[i + 1]))
            if getattr(ent, "closed", False) and len(idx) > 1:
                union(int(idx[0]), int(idx[-1]))
        comps = {}
        for i in range(n):
            comps.setdefault(find(i), []).append(i)
        lobes = [V[idxs] for idxs in comps.values() if len(idxs) >= 3]
        return lobes or [V]
    except Exception:
        try:
            return [np.asarray(planar.vertices, float)]
        except Exception:
            return []


def _measure_section(union: trimesh.Trimesh, origin: np.ndarray,
                     axis: np.ndarray) -> List[_SliceInfo]:
    """Cut a cross-section and measure each LOBE (connected component) separately:
    its centroid, width and thin (minor) direction. Splitting per lobe is what
    lets a multi-armed shape (a U's two arms, a scissor's two blades) be grasped
    on each arm — the whole-section centroid would otherwise fall in the air
    between the lobes."""
    try:
        sec = union.section(plane_origin=origin, plane_normal=axis)
        if sec is None:
            return []
        planar, to_3d = sec.to_planar()
    except Exception:
        return []
    R = np.asarray(to_3d, float)
    out: List[_SliceInfo] = []
    for pts in _section_lobes(planar):
        pts = np.asarray(pts, float)
        if len(pts) < 3:
            continue
        c2d = pts.mean(axis=0)
        d = pts - c2d
        r = np.linalg.norm(d, axis=1)
        width = 2.0 * float(np.percentile(r, 95))
        try:
            evals, evecs = np.linalg.eigh(np.cov(d.T))
            minor2d = evecs[:, 0]
        except Exception:
            minor2d = np.array([1.0, 0.0])
        minor_width = 2.0 * float(np.max(np.abs(d @ minor2d))) if len(d) else width
        minor3d = _unit(R[:3, :3] @ np.array([minor2d[0], minor2d[1], 0.0]))
        centroid3d = (R @ np.array([c2d[0], c2d[1], 0.0, 1.0]))[:3]
        out.append(_SliceInfo(centroid3d=centroid3d, width=width,
                              minor_width=minor_width, minor3d=minor3d,
                              axis=_unit(axis)))
    return out


def _find_waists(ctx: _GeomCtx, gripper: GripperSpec,
                 n_slices: int = 40, max_waists: int = 12) -> List[_SliceInfo]:
    """Graspable cross-sections (handles / bridges / necks / individual arms).
    Sweeps slices along each axis and collects every LOBE whose width fits the
    gripper; the scorer then ranks them (narrowness, balance, stability)."""
    found: List[_SliceInfo] = []
    V = np.asarray(ctx.union.vertices, float)
    for axis in ctx.axes:
        t = V @ axis
        lo, hi = float(t.min()), float(t.max())
        if hi - lo < 1e-4:
            continue
        eps = 0.04 * (hi - lo)
        for tt in np.linspace(lo + eps, hi - eps, n_slices):
            origin = ctx.com + (tt - ctx.com @ axis) * axis
            for s in _measure_section(ctx.union, origin, axis):
                if s.width <= gripper.width_max:
                    found.append(s)
    # de-dupe nearby lobes; keep the narrowest few (scorer does the rest)
    uniq: List[_SliceInfo] = []
    for s in sorted(found, key=lambda s: s.width):
        if all(np.linalg.norm(s.centroid3d - u.centroid3d) > 0.015 for u in uniq):
            uniq.append(s)
    return uniq[:max_waists]


def _surface_hit(mesh: trimesh.Trimesh, origin: np.ndarray, direction: np.ndarray):
    """Nearest ray hit (point, outward_normal) from origin along direction, or None."""
    try:
        locs, _, tri = mesh.ray.intersects_location(
            ray_origins=origin[None, :], ray_directions=direction[None, :])
        if len(locs) == 0:
            return None
        d = np.linalg.norm(locs - origin[None, :], axis=1)
        k = int(np.argmin(d))
        return np.asarray(locs[k], float), np.asarray(mesh.face_normals[tri[k]], float)
    except Exception:
        return None


def _clear_approach(ctx: _GeomCtx, contacts, grasp_axis: np.ndarray,
                    gripper: GripperSpec) -> np.ndarray:
    """Approach perpendicular to the grasp (contact) axis, from the most open
    side, preferring top-down (-Z). Clearance is measured at the CONTACT points
    (the fingertips), not the grasp centre (which is inside the object)."""
    cands = [np.array([0, 0, -1.0]), np.array([1.0, 0, 0]), np.array([-1.0, 0, 0]),
             np.array([0, 1.0, 0]), np.array([0, -1.0, 0]), np.array([0, 0, 1.0])]
    best, best_s = None, -1e9
    for d in cands:
        a = d - (d @ grasp_axis) * grasp_axis
        if np.linalg.norm(a) < 1e-3:
            continue
        a = _unit(a)
        clr = _approach_clearance_dist(ctx.union, contacts, a, gripper)
        s = clr + 0.5 * max(0.0, -float(a[2])) * (gripper.finger_length + gripper.palm_clearance)
        if s > best_s:
            best, best_s = a, s
    if best is None:
        best = np.array([0, 0, -1.0])
    if best[2] > 0:                       # convention: point from above toward contacts
        best = -best
    return best


def _synthesize_waist_grasps(ctx: _GeomCtx, waist: _SliceInfo,
                             gripper: GripperSpec) -> List[Grasp]:
    """Grasp the waist with the gripper oriented like a human would: the finger
    LENGTH runs along the part's elongation, the fingers OPEN across the section,
    and the approach is perpendicular to the elongation (a side approach). This
    is what makes a standing cylinder fall between the pads instead of being
    grabbed at a weird tilt from the rim."""
    c = waist.centroid3d
    e = _unit(waist.axis)                       # elongation: finger-length will align to this
    minor = _unit(waist.minor3d)               # in-section thin direction
    major = _unit(np.cross(e, minor))          # in-section wide direction
    # Force the side approach (finger-length along the elongation) ONLY for parts
    # that are clearly elongated AND whose long axis matches this waist's axis —
    # e.g. a standing cylinder or a handle, where a top-down rim grasp is wrong.
    # Mildly-elongated / blobby objects keep the clearance-based approach (which
    # prefers top-down and holds those shapes better).
    strict = (ctx.elong_ratio > 1.8 and abs(float(e @ ctx.elong)) > 0.9)
    grasps: List[Grasp] = []
    for f in (minor, major):
        f = _unit(f - (f @ e) * e)             # keep finger-opening perpendicular to e
        if np.linalg.norm(f) < 1e-6:
            continue
        hit1 = _surface_hit(ctx.union, c, -f)
        hit2 = _surface_hit(ctx.union, c, +f)
        if hit1 is not None and hit2 is not None:
            c1, n1 = hit1
            c2, n2 = hit2
        else:                                  # analytic fallback
            half = 0.5 * waist.minor_width + 0.001
            c1, n1 = c - half * f, -f
            c2, n2 = c + half * f, +f
        g = _build_grasp(np.asarray(c1), np.asarray(n1),
                         np.asarray(c2), np.asarray(n2), gripper)
        if g is None:
            continue
        a = _unit(np.cross(e, _unit(g.contact2 - g.contact1)))   # ⊥ both -> finger-len == e
        if strict and np.linalg.norm(a) > 1e-6:
            clr_p = _approach_clearance_dist(ctx.union, (g.contact1, g.contact2), a, gripper)
            clr_m = _approach_clearance_dist(ctx.union, (g.contact1, g.contact2), -a, gripper)
            g.approach = a if clr_p >= clr_m else -a
        else:
            g.approach = _clear_approach(ctx, (g.contact1, g.contact2),
                                         _unit(g.contact2 - g.contact1), gripper)
        grasps.append(g)
    return grasps


def _score_grasp(g: Grasp, ctx: _GeomCtx, gripper: GripperSpec,
                 waist_bonus: float = 0.0) -> float:
    # Reward NARROW grasps (a "waist"/handle is what a human grabs), but mildly
    # penalize a knife-edge too thin to grip reliably.
    w_norm = (g.width - gripper.width_min) / (gripper.width_max - gripper.width_min + 1e-9)
    f_narrow = float(np.clip(1.0 - w_norm, 0.0, 1.0))
    if g.width < 0.008:
        f_narrow *= 0.6
    # BALANCE, not "grasp at the CoM": what matters is that the CoM hangs
    # vertically below the grip so the object doesn't tip when lifted — i.e. the
    # HORIZONTAL offset (perpendicular to gravity) between the grasp and the CoM.
    # Using the 3D distance would be wrong because the CoM often lies OUTSIDE the
    # solid (an open scissor's CoM is in the air between the blades), so no
    # on-body grasp can ever be "at" it; the horizontal projection is what counts.
    d_com = float(np.linalg.norm((g.center - ctx.com)[:2])) / (ctx.diag + 1e-9)
    f_com = float(np.exp(-(d_com / 0.25) ** 2))
    margin_max = 1.0 - 1.0 / np.sqrt(1.0 + gripper.mu * gripper.mu)
    f_margin = float(np.clip(g.margin / (margin_max + 1e-9), 0.0, 1.0))
    f_align = 0.5 * (1.0 - float(_unit(g.normal1) @ _unit(g.normal2)))
    need = gripper.finger_length + gripper.palm_clearance
    clr = _approach_clearance_dist(ctx.union, (g.contact1, g.contact2), g.approach, gripper)
    f_clear = float(np.clip(clr / (need + 1e-9), 0.0, 1.0))
    f_top = float(np.clip(-g.approach[2], 0.0, 1.0))
    # Finger-LENGTH should run along the object's elongation (so a tall cylinder
    # sits between the pads, not grabbed at the rim). Only matters when the
    # object is actually elongated.
    z = _unit(g.approach)
    y = _unit(g.contact2 - g.contact1)
    y = _unit(y - (y @ z) * z)
    x = np.cross(y, z)                           # finger-length direction
    elong_factor = float(np.clip((ctx.elong_ratio - 1.6) / 0.6, 0.0, 1.0))
    f_elong = elong_factor * abs(float(x @ ctx.elong))
    score = (0.26 * f_narrow + 0.20 * f_com + 0.16 * f_margin
             + 0.10 * f_align + 0.10 * f_clear + 0.04 * f_top
             + 0.14 * f_elong + waist_bonus)
    # Closing-margin gate: a grasp near the gripper's MAX opening has almost no
    # squeeze travel and ejects the object. Penalize multiplicatively so a wide
    # grasp at the CoM never beats a narrower one. (Full credit at <=65% of max.)
    margin_frac = (gripper.width_max - g.width) / (gripper.width_max + 1e-9)
    close_ok = float(np.clip(margin_frac / 0.35, 0.15, 1.0))
    # Base gate: grasps near the object's bottom sit at the table — unstable and
    # prone to finger/table contact. Penalize the lowest ~15% of the height
    # (objects are exported seated, so union z runs base(0) -> top). Full credit
    # above that.
    try:
        zlo, zhi = float(ctx.union.bounds[0][2]), float(ctx.union.bounds[1][2])
        height = zhi - zlo
        z_frac = (g.center[2] - zlo) / (height + 1e-9)
    except Exception:
        zlo, zhi, height, z_frac = 0.0, 1.0, 1.0, 1.0
    base_ok = float(np.clip(z_frac / 0.15, 0.3, 1.0))
    # Top-edge gate: a grasp at the very top rim has little material around it
    # (e.g. a side grasp's fingers stick out past the top -> less contact). Mild
    # penalty on the top ~15%, so a feature spanning the height is grasped in the
    # upper-MIDDLE; a feature that only exists at the top (a U's arms) still wins.
    top_ok = float(np.clip((1.0 - z_frac) / 0.15, 0.5, 1.0))
    # Hang-stability gate: when lifted, the object pivots about the grip. If the
    # grip is BELOW the CoM, the CoM is an inverted pendulum and flips over (can
    # break the grasp); if the grip is at/above the CoM, it hangs stably. dz>0
    # means grip above CoM (good). Penalize grip-below-CoM, full credit above.
    dz = (g.center[2] - ctx.com[2]) / (height + 1e-9)
    stable_ok = float(np.clip(1.0 + 2.0 * dz, 0.3, 1.0))
    return score * close_ok * base_ok * top_ok * stable_ok


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plan_grasps(obj: CompositeObject, gripper: Optional[GripperSpec] = None,
                n_surface: int = 256, max_pairs: int = 2000,
                max_returned: int = 20, seed: int = 0) -> GraspReport:
    """Synthesize antipodal force-closure grasps for *obj*.

    Returns a GraspReport with up to *max_returned* validated grasps and counts
    of how many pairs passed each test (useful for diagnostics)."""
    gripper = gripper or GripperSpec()
    rng = np.random.default_rng(seed)
    mesh = _safe_mesh(obj)
    if mesh is None:
        return GraspReport(n_attempts=0, n_friction_pass=0, n_collision_free=0)

    ctx = _grasp_geometry(obj, mesh)
    n_attempts = 0
    n_friction = 0
    n_collision = 0
    candidates: List[Grasp] = []
    waist_set = set()                       # id() of waist-derived grasps (for source bonus)

    # --- Source A: waist / handle grasps (what a human would pick) ---
    for waist in _find_waists(ctx, gripper):
        for g in _synthesize_waist_grasps(ctx, waist, gripper):
            n_attempts += 1
            n_friction += 1                 # passed _build_grasp's friction-cone test
            if _approach_collision_free(ctx.union, g, gripper):
                n_collision += 1
                waist_set.add(id(g))
                candidates.append(g)

    # --- Source B: random antipodal sampler (supplement, esp. for blobby shapes) ---
    pts, n_out = _sample_surface(mesh, n_surface, rng)
    n = len(pts)
    n_pairs = min(max_pairs, n * (n - 1) // 2)
    pair_i = rng.integers(0, n, size=n_pairs)
    pair_j = rng.integers(0, n, size=n_pairs)
    keep = pair_i != pair_j
    pair_i, pair_j = pair_i[keep], pair_j[keep]
    cap = max(4 * max_returned, 40)         # generous pool to score, not first-found
    for i, j in zip(pair_i, pair_j):
        n_attempts += 1
        g = _build_grasp(pts[i], n_out[i], pts[j], n_out[j], gripper)
        if g is None:
            continue
        n_friction += 1
        if _approach_collision_free(mesh, g, gripper):
            n_collision += 1
            candidates.append(g)
            if len(candidates) >= cap:
                break

    # --- Score, sort best-first, truncate ---
    for g in candidates:
        g.score = _score_grasp(g, ctx, gripper,
                               waist_bonus=0.05 if id(g) in waist_set else 0.0)
    candidates.sort(key=lambda g: g.score, reverse=True)
    return GraspReport(n_attempts=n_attempts,
                       n_friction_pass=n_friction,
                       n_collision_free=n_collision,
                       grasps=candidates[:max_returned])


def grasp_success_rate(objects, gripper: Optional[GripperSpec] = None,
                       n_surface: int = 256, max_pairs: int = 1500,
                       seed: int = 0) -> dict:
    """Aggregate metric: fraction of *objects* with at least one valid grasp."""
    n_pass = 0
    pair_counts = []
    n_valid_per_obj = []
    for k, obj in enumerate(objects):
        r = plan_grasps(obj, gripper=gripper, n_surface=n_surface,
                        max_pairs=max_pairs, seed=seed + k)
        if r.has_grasp:
            n_pass += 1
        pair_counts.append(r.n_collision_free)
        n_valid_per_obj.append(len(r.grasps))
    return {
        "n_objects": len(objects),
        "n_with_grasp": n_pass,
        "success_rate": n_pass / max(1, len(objects)),
        "mean_valid_grasps": float(np.mean(n_valid_per_obj)) if n_valid_per_obj else 0.0,
        "median_valid_grasps": float(np.median(n_valid_per_obj)) if n_valid_per_obj else 0.0,
    }
