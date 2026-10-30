"""Numpy inference of the learned grasp-success predictor (no torch needed)."""
import numpy as np

from grasp_predictor import N_POINTS, GraspPredictor


def _random_model(path, n_scalar=13, seed=0):
    rng = np.random.default_rng(seed)
    w = {"mu": np.zeros(n_scalar, np.float32), "sd": np.ones(n_scalar, np.float32)}
    for i, (o, n) in enumerate([(64, 3), (128, 64), (256, 128)]):
        w[f"point{i}_W"] = rng.normal(0, 0.3, (o, n)).astype(np.float32)
        w[f"point{i}_b"] = np.zeros(o, np.float32)
    for i, (o, n) in enumerate([(128, 256 + n_scalar), (64, 128), (1, 64)]):
        w[f"head{i}_W"] = rng.normal(0, 0.1, (o, n)).astype(np.float32)
        w[f"head{i}_b"] = np.zeros(o, np.float32)
    np.savez(path, **w)
    return path


def test_predict_shapes_and_range(tmp_path):
    gp = GraspPredictor(_random_model(tmp_path / "m.npz"))
    rng = np.random.default_rng(1)
    pts = rng.uniform(-0.03, 0.03, (512, 3)) + [0, 0, 0.03]
    grasps = [{"center": [0, 0, 0.03], "approach": [0, 0, -1], "axis": [1, 0, 0],
               "width": 0.04, "score": 0.5},
              {"center": [0, 0, 0.02], "approach": [1, 0, 0], "axis": [0, 1, 0],
               "width": 0.05, "score": 0.3}]
    p = gp.predict(pts, 0.1, [0, 0, 0.03], [0.06, 0.06, 0.06], grasps)
    assert p.shape == (2,) and np.all((p > 0) & (p < 1))
    assert gp.predict(pts, 0.1, [0, 0, 0.03], [0.06] * 3, []).shape == (0,)


def test_pointnet_is_order_invariant(tmp_path):
    gp = GraspPredictor(_random_model(tmp_path / "m.npz"))
    rng = np.random.default_rng(2)
    local = rng.normal(size=(3, N_POINTS, 3))
    S = rng.normal(size=(3, 13)).astype(np.float32)
    perm = rng.permutation(N_POINTS)
    assert np.allclose(gp.logits(local, S), gp.logits(local[:, perm], S), atol=1e-5)
