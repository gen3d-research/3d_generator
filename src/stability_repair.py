"""Point ⑥ — stability repair (projection onto the stable-rest manifold).

A post-hoc step that re-orients any object onto its **most-stable resting pose** (the
convex-hull facet on which it rests with the largest tip-over margin). After repair the
object is, by construction, in a pose that settles upright on a flat table — turning the
generator's "optimizes stability" into "rests stably by construction". Pure geometry
(scipy convex hull); no physics engine, no networkx.
"""
import numpy as np
from scipy.spatial import ConvexHull


def _align(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation matrix mapping unit vector a -> unit vector b (Rodrigues)."""
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = np.linalg.norm(v)
    if s < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))


def _margin_in_poly(pt: np.ndarray, poly: np.ndarray) -> float:
    """Signed distance from pt to a CCW polygon boundary (+inside, -outside)."""
    d = 1e9
    inside = True
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        ab = b - a
        t = np.clip(np.dot(pt - a, ab) / (ab @ ab + 1e-12), 0.0, 1.0)
        d = min(d, np.linalg.norm(pt - (a + t * ab)))
        if np.cross(ab, pt - a) < 0:
            inside = False
    return d if inside else -d


def tip_angle(obj, mesh=None) -> float:
    """Critical tip-over angle (deg) of the object's *current* pose; 0 if unstable.
    Pass a precomputed union mesh to avoid the (expensive) CSG rebuild."""
    if mesh is None:
        mesh = obj.to_mesh(boolean_union=True)
    v = np.asarray(mesh.vertices, float)
    zmin = v[:, 2].min()
    band = zmin + max(0.002, 0.03 * (v[:, 2].max() - zmin))
    g = v[v[:, 2] < band][:, :2]
    if len(np.unique(g, axis=0)) < 3:
        return 0.0
    try:
        hp = g[ConvexHull(g).vertices]
    except Exception:
        return 0.0
    com = np.asarray(obj.center_of_mass(1000.0), float)
    m = _margin_in_poly(com[:2], hp)
    if m <= 0:
        return 0.0
    return float(np.degrees(np.arctan2(m, max(com[2] - zmin, 1e-4))))


def best_stable_pose(obj, mesh=None):
    """Return (tip_angle_deg, R, rest_zmin_rel_com) for the most-stable resting
    orientation, or None. Each convex-hull facet is a candidate resting face; rotating
    its normal to point down lays the object on that face. The rest is stable iff the
    COM projects inside the contact polygon; stable rests rank by tip-over margin."""
    if mesh is None:
        mesh = obj.to_mesh(boolean_union=True)
    com = np.asarray(obj.center_of_mass(1000.0), float)
    try:
        hull = mesh.convex_hull
    except Exception:
        return None
    hv = np.asarray(hull.vertices, float) - com
    best, seen = None, []
    for n in np.asarray(hull.face_normals, float):
        if any(np.dot(n, s) > 0.999 for s in seen):   # same downward direction already tried
            continue
        seen.append(n)
        R = _align(n, np.array([0.0, 0.0, -1.0]))
        rv = hv @ R.T
        zmin = rv[:, 2].min()
        contact = rv[rv[:, 2] < zmin + 5e-4][:, :2]
        if len(np.unique(contact, axis=0)) < 3:
            continue
        try:
            hp = contact[ConvexHull(contact).vertices]
        except Exception:
            continue
        m = _margin_in_poly(np.zeros(2), hp)   # COM is at xy origin (centered on com)
        if m <= 0:
            continue
        tip = float(np.degrees(np.arctan2(m, max(-zmin, 1e-4))))
        if best is None or tip > best[0]:
            best = (tip, R, float(zmin))
    return best


def repair_stability(obj, min_tip_deg: float = 20.0):
    """Re-orient `obj` onto its most-stable resting pose if its current pose is tippier
    than `min_tip_deg`. Mutates + returns the object. No-op if already stable enough or
    no stable pose is found.

    Builds the (expensive) CSG union mesh exactly ONCE: the current tip angle, the
    candidate rests, and the re-seat offset all derive from it — the re-seat reuses the
    chosen pose's rotated hull minimum (hulls preserve coordinate extrema), so no
    second union is needed after the rotation."""
    mesh = obj.to_mesh(boolean_union=True)
    current = tip_angle(obj, mesh=mesh)
    if current >= min_tip_deg:
        return obj
    bp = best_stable_pose(obj, mesh=mesh)
    if bp is None or bp[0] <= current:
        return obj
    tip, R, rest_zmin = bp
    com = np.asarray(obj.center_of_mass(1000.0), float)
    for p in obj.primitives:
        t = np.asarray(p.transform.translation, float)
        p.transform.translation = R @ (t - com) + com
        p.transform.rotation = R @ np.asarray(p.transform.rotation, float)
    # Re-seat so the lowest point rests on z=0. After the rotation the COM is unmoved
    # and the lowest point sits rest_zmin BELOW it (computed from the rotated hull).
    low = float(com[2] + rest_zmin)
    for p in obj.primitives:
        p.transform.translation[2] -= low
    return obj
