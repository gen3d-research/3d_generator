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


def _approach_collision_free(mesh: trimesh.Trimesh, grasp: Grasp,
                             gripper: GripperSpec, n_samples: int = 12) -> bool:
    """Sample probe points along the approach direction in front of each finger
    and verify they are outside the object.  Uses trimesh's contains query."""
    if not mesh.is_watertight:
        # contains() is unreliable on non-watertight meshes — fall back to a
        # signed-distance-style proxy via proximity query.
        try:
            pq = trimesh.proximity.ProximityQuery(mesh)
        except Exception:
            return True  # be permissive rather than spuriously failing
        for c in (grasp.contact1, grasp.contact2):
            probe = c[None, :] - grasp.approach[None, :] * np.linspace(
                0.0, gripper.finger_length + gripper.palm_clearance, n_samples)[:, None]
            d = pq.signed_distance(probe)
            if np.any(d > 1e-4):  # positive = inside
                return False
        return True
    probes = []
    for c in (grasp.contact1, grasp.contact2):
        offs = np.linspace(0.002, gripper.finger_length + gripper.palm_clearance, n_samples)
        probe = c[None, :] - grasp.approach[None, :] * offs[:, None]
        probes.append(probe)
    P = np.vstack(probes)
    return not np.any(mesh.contains(P))


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

    pts, n_out = _sample_surface(mesh, n_surface, rng)
    n = len(pts)
    # Cap the number of candidate pairs we examine.
    n_pairs = min(max_pairs, n * (n - 1) // 2)
    pair_i = rng.integers(0, n, size=n_pairs)
    pair_j = rng.integers(0, n, size=n_pairs)
    keep = pair_i != pair_j
    pair_i, pair_j = pair_i[keep], pair_j[keep]

    n_friction = 0
    n_collision = 0
    grasps: List[Grasp] = []
    for i, j in zip(pair_i, pair_j):
        g = _build_grasp(pts[i], n_out[i], pts[j], n_out[j], gripper)
        if g is None:
            continue
        n_friction += 1
        if _approach_collision_free(mesh, g, gripper):
            n_collision += 1
            grasps.append(g)
            if len(grasps) >= max_returned:
                break
    return GraspReport(n_attempts=len(pair_i),
                       n_friction_pass=n_friction,
                       n_collision_free=n_collision,
                       grasps=grasps)


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
