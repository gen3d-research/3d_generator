"""Learned grasp-success predictor: features and numpy inference.

The model (a small PointNet + MLP) is trained with torch by
``scripts/train_grasp_predictor.py`` on MuJoCo pick-and-lift labels
(``scripts/build_grasp_dataset.py``), and exported as ``model.npz`` with the
BatchNorm layers folded into the weights. This module needs only numpy, so the
generator venv stays torch-free (torch pulls in networkx, which changes trimesh
code paths the generator relies on).

The feature code lives here and is imported by the training script, so the two
cannot drift apart.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

SCALE = 0.1          # metres -> network units
N_POINTS = 256       # points per sample at inference (training subsamples too)


def grasp_frames(approach: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """[n, 3, 3] rotations whose columns are x = pad width, y = antipodal axis,
    z = approach."""
    z = approach / np.linalg.norm(approach, axis=1, keepdims=True)
    y = axis - np.sum(axis * z, axis=1, keepdims=True) * z
    y = y / (np.linalg.norm(y, axis=1, keepdims=True) + 1e-12)
    x = np.cross(y, z)
    return np.stack([x, y, z], axis=2)


def scalar_features(R: np.ndarray, center: np.ndarray, width: np.ndarray,
                    score: np.ndarray, mass: np.ndarray, com: np.ndarray,
                    extents: np.ndarray) -> np.ndarray:
    """Per-grasp scalars (13): width, log mass, COM and gravity direction in
    the grasp frame, grasp height, planner score, object extents."""
    com_g = np.einsum("nji,nj->ni", R, com - center) / SCALE
    up_g = np.einsum("nji,j->ni", R, np.array([0.0, 0.0, 1.0]))
    return np.column_stack([
        width / 0.08, np.log(mass), com_g, up_g,
        center[:, 2] / SCALE, score, extents / SCALE,
    ]).astype(np.float32)


def local_points(points: np.ndarray, R: np.ndarray, center: np.ndarray) -> np.ndarray:
    """Object points [n, P, 3] (object frame) -> grasp frames, network units."""
    return np.einsum("bji,bpj->bpi", R, points - center[:, None, :]) / SCALE


def even_subsample(points: np.ndarray, n: int = N_POINTS) -> np.ndarray:
    idx = np.linspace(0, len(points) - 1, n).astype(int)
    return points[idx]


class GraspPredictor:
    """Numpy forward pass of the exported model."""

    def __init__(self, path: str | Path):
        w = np.load(path)
        self.point = [(w[f"point{i}_W"], w[f"point{i}_b"]) for i in range(3)]
        self.head = [(w[f"head{i}_W"], w[f"head{i}_b"]) for i in range(3)]
        self.mu, self.sd = w["mu"], w["sd"]

    def logits(self, pts_local: np.ndarray, scalars: np.ndarray) -> np.ndarray:
        h = pts_local.astype(np.float32)                    # [B, P, 3]
        for W, b in self.point:
            h = np.maximum(h @ W.T + b, 0.0)
        f = h.max(axis=1)                                   # [B, 256]
        x = np.concatenate([f, (scalars - self.mu) / self.sd], axis=1)
        for i, (W, b) in enumerate(self.head):
            x = x @ W.T + b
            if i < len(self.head) - 1:
                x = np.maximum(x, 0.0)
        return x[:, 0]

    def predict(self, points: np.ndarray, mass: float, com, extents,
                grasps: list[dict]) -> np.ndarray:
        """Success probability for each grasp of one object.

        points: [P, 3] surface samples (object frame); grasps: dicts with
        center / approach / axis / width / score, as in the eval manifest."""
        if not grasps:
            return np.zeros(0)
        c = np.array([g["center"] for g in grasps], float)
        R = grasp_frames(np.array([g["approach"] for g in grasps], float),
                         np.array([g["axis"] for g in grasps], float))
        n = len(grasps)
        S = scalar_features(R, c, np.array([g["width"] for g in grasps], float),
                            np.array([g.get("score", 0.0) for g in grasps], float),
                            np.full(n, mass), np.tile(np.asarray(com, float), (n, 1)),
                            np.tile(np.asarray(extents, float), (n, 1)))
        P = np.broadcast_to(even_subsample(np.asarray(points, float)), (n, N_POINTS, 3))
        return 1.0 / (1.0 + np.exp(-self.logits(local_points(P, R, c), S)))


def predict_object(model: GraspPredictor, obj, grasps: list[dict], seed: int = 0) -> np.ndarray:
    """Success probability of each grasp (dicts as above) of a CompositeObject."""
    import trimesh
    mesh = obj.to_mesh()
    pts, _ = trimesh.sample.sample_surface(mesh, 512, seed=seed)
    try:
        mass, _, com = obj.mesh_mass_properties(1000.0)
    except Exception:
        mass, com = obj.total_volume() * 1000.0, obj.center_of_mass(1000.0)
    return model.predict(pts, float(mass), com, np.ptp(mesh.vertices, axis=0), grasps)


def object_lift_probability(model: GraspPredictor, obj, seed: int = 0, n_surface: int = 128,
                            max_pairs: int = 600, max_grasps: int = 12) -> float:
    """Predicted lift success of the object's best table-clearing grasp; 0 if it
    has none or planning fails. This is the CEM's physics-aware signal."""
    from grasp_planner import GripperSpec, grasps_as_dicts, plan_grasps
    try:
        report = plan_grasps(obj, GripperSpec(), n_surface=n_surface, max_pairs=max_pairs,
                             max_returned=max_grasps, seed=seed, table_check=True)
        grasps = grasps_as_dicts(report.grasps, max_grasps)
        return float(predict_object(model, obj, grasps, seed).max()) if grasps else 0.0
    except Exception:
        return 0.0
