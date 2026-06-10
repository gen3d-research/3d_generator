"""Point ⑩ — Pareto front over manipulation objectives.

The scorer collapses stability / graspability / size / … into one weighted sum, which
bakes in a trade-off. This returns the **non-dominated set** instead, so you can see and
choose the trade-off (a very-stable-but-less-graspable object vs the reverse) rather than
accept the baked-in weights.
"""
import numpy as np


def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """Does objective vector a dominate b? (a >= b in every objective, a > b in one).
    All objectives are MAXIMIZED."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    return bool(np.all(a >= b) and np.any(a > b))


def pareto_front(items, objective_vectors):
    """Return (front_items, front_vectors): the non-dominated subset (objectives maximized)."""
    V = [np.asarray(v, float) for v in objective_vectors]
    keep = []
    for i, vi in enumerate(V):
        if not any(j != i and _dominates(V[j], vi) for j in range(len(V))):
            keep.append(i)
    return [items[i] for i in keep], [V[i] for i in keep]


def score_objectives(objs, scorer, keys=("stability_score", "graspability_score")):
    """Objective vectors (to maximize) for objects, via an ObjectScorer."""
    out = []
    for o in objs:
        s = scorer.score(o)
        out.append([float(getattr(s, k)) for k in keys])
    return out


def pareto_objects(objs, scorer, keys=("stability_score", "graspability_score")):
    """Convenience: the Pareto-optimal objects + their objective vectors over `keys`."""
    V = score_objectives(objs, scorer, keys)
    return pareto_front(objs, V)
