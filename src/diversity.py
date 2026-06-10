"""
Diversity and failure-case analysis for generated object sets.

Addresses R2-Q5 / R2-Q6 / AE-comment-4: show the optimizer does not collapse to
a narrow set of high-scoring shapes, and quantify which archetypes are harder.

Two complementary diversity proxies are reported:

    feature_diversity    - mean pairwise distance over a hand-picked 10-D
                           shape-descriptor vector (extents, volume, primitive
                           counts, height/width ratios).  Cheap and interpretable.

    chamfer_diversity    - mean Chamfer distance over a small subsample of
                           pairs, computed on point clouds resampled from the
                           composite meshes.  Slower but captures actual
                           geometric variability.

Per-archetype failure rate is the fraction of generated objects scoring below a
configurable threshold (default 0.5).
"""

from __future__ import annotations

import numpy as np
import trimesh
from dataclasses import dataclass
from typing import List, Optional

from primitives import CompositeObject, Box, Cylinder, Sphere, Capsule


# ---------------------------------------------------------------------------

def shape_descriptor(obj: CompositeObject) -> np.ndarray:
    """Return a fixed-length descriptor vector for the object.

    Layout (10-D):
        [extent_x, extent_y, extent_z,                    (AABB extents)
         total_volume, height_over_max_xy,                (gross shape ratios)
         n_box, n_cyl, n_sph, n_cap,                      (primitive counts)
         friction]
    """
    extents = obj.aabb_extents()
    counts = {"box": 0, "cylinder": 0, "sphere": 0, "capsule": 0}
    vol = 0.0
    for p in obj.primitives:
        if isinstance(p, Box):
            counts["box"] += 1
        elif isinstance(p, Cylinder):
            counts["cylinder"] += 1
        elif isinstance(p, Sphere):
            counts["sphere"] += 1
        elif isinstance(p, Capsule):
            counts["capsule"] += 1
        try:
            vol += float(p.volume())
        except Exception:
            pass
    max_xy = max(extents[0], extents[1], 1e-6)
    return np.array([
        extents[0], extents[1], extents[2],
        vol, extents[2] / max_xy,
        counts["box"], counts["cylinder"], counts["sphere"], counts["capsule"],
        getattr(obj, "friction", 0.0),
    ], dtype=float)


def feature_diversity(objects: List[CompositeObject], normalize: bool = True
                      ) -> float:
    """Mean pairwise Euclidean distance over the shape-descriptor vectors."""
    if len(objects) < 2:
        return 0.0
    F = np.stack([shape_descriptor(o) for o in objects], axis=0)
    if normalize:
        s = F.std(axis=0, keepdims=True) + 1e-9
        F = (F - F.mean(axis=0, keepdims=True)) / s
    # mean pairwise distance via the trick ||a-b||^2 expansion
    G = F @ F.T
    sq = np.diag(G)[:, None] + np.diag(G)[None, :] - 2 * G
    sq = np.maximum(sq, 0.0)
    d = np.sqrt(sq)
    n = len(objects)
    return float(d.sum() / (n * (n - 1)))


def chamfer_diversity(objects: List[CompositeObject], n_points: int = 512,
                      max_pairs: int = 150, seed: int = 0) -> float:
    """Mean Chamfer distance over a random sample of object pairs."""
    if len(objects) < 2:
        return 0.0
    rng = np.random.default_rng(seed)
    clouds = []
    for o in objects:
        try:
            m = o.to_mesh(boolean_union=False)
            pts, _ = trimesh.sample.sample_surface(m, n_points)
            # zero-mean each cloud so we measure shape diff, not pose diff
            pts = pts - pts.mean(axis=0, keepdims=True)
            clouds.append(np.asarray(pts))
        except Exception:
            clouds.append(None)
    n = len(clouds)
    pair_dists = []
    pairs_drawn = 0
    while pairs_drawn < max_pairs:
        i, j = rng.integers(0, n, size=2)
        if i == j or clouds[i] is None or clouds[j] is None:
            continue
        a, b = clouds[i], clouds[j]
        # subsample for speed
        ai = a[rng.choice(len(a), size=min(128, len(a)), replace=False)]
        bj = b[rng.choice(len(b), size=min(128, len(b)), replace=False)]
        d_ab = np.linalg.norm(ai[:, None, :] - bj[None, :, :], axis=-1)
        chamfer = d_ab.min(axis=1).mean() + d_ab.min(axis=0).mean()
        pair_dists.append(chamfer / 2.0)
        pairs_drawn += 1
    return float(np.mean(pair_dists)) if pair_dists else 0.0


# ---------------------------------------------------------------------------

@dataclass
class FailureCaseSummary:
    archetype: str
    n_objects: int
    mean_score: float
    fail_rate: float           # fraction with score < threshold
    worst_score: float


def per_archetype_failure(scores_by_archetype: dict, threshold: float = 0.5
                          ) -> List[FailureCaseSummary]:
    out = []
    for name, scores in scores_by_archetype.items():
        s = np.asarray(scores, dtype=float)
        if len(s) == 0:
            continue
        out.append(FailureCaseSummary(
            archetype=name,
            n_objects=len(s),
            mean_score=float(s.mean()),
            fail_rate=float((s < threshold).mean()),
            worst_score=float(s.min()),
        ))
    out.sort(key=lambda r: r.fail_rate, reverse=True)
    return out


def summarize_diversity(objects: List[CompositeObject],
                        do_chamfer: bool = True, seed: int = 0) -> dict:
    out = {
        "n_objects": len(objects),
        "feature_diversity": feature_diversity(objects, normalize=True),
    }
    if do_chamfer:
        # Thread the run seed: chamfer subsamples point clouds, so an unseeded
        # call makes the reported diversity unreproducible across runs.
        out["chamfer_diversity"] = chamfer_diversity(objects, seed=seed)
    return out
