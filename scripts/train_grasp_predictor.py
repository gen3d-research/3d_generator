#!/usr/bin/env python3
"""Learned grasp-success predictor trained on MuJoCo pick-and-lift labels.

The ICARM paper's stated future work: replace the geometric grasp proxy with a
learned predictor. Input = the object's surface points expressed in the grasp
frame (x = finger-plane normal, y = antipodal axis, z = approach), plus scalars
(width, mass, COM and gravity direction in the grasp frame, grasp height,
planner score, extents). Model = a small PointNet. Label = physics lift success
of a *feasible* grasp (infeasible grasps are filtered analytically before
execution, so they are not part of the learning problem).

Reported on held-out data (whole seeds or whole archetypes, never single
grasps of a training object):
  * AUC of the predictor vs the planner score and a torque-arm heuristic
  * lift rate when each object executes the predictor's top feasible grasp,
    vs the planner's first feasible grasp, vs the oracle (any feasible grasp)

Runs in a torch venv (not the generator venv); the dataset comes from
scripts/build_grasp_dataset.py.

    ~/venv/gen3d_learn/bin/python scripts/train_grasp_predictor.py \\
        --data output/grasp_dataset/journal.npz output/grasp_dataset/archetypes.npz \\
        --split archetype --out output/grasp_predictor/archetype_split
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

SCALE = 0.1  # metres -> network units


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def load(paths: list[Path]) -> dict:
    """Concatenate datasets; keep feasible grasps only."""
    parts = []
    off = 0
    for p in paths:
        d = dict(np.load(p, allow_pickle=False))
        d["g_obj"] = d["g_obj"] + off
        off += len(d["obj_points"])
        parts.append(d)
    keys = [k for k in parts[0] if k != "config"]
    D = {k: np.concatenate([q[k] for q in parts]) for k in keys}
    keep = D["g_feasible"].astype(bool)
    for k in list(D):
        if k.startswith("g_"):
            D[k] = D[k][keep]
    return D


def grasp_frames(D: dict) -> np.ndarray:
    z = D["g_approach"] / np.linalg.norm(D["g_approach"], axis=1, keepdims=True)
    y = D["g_axis"] - np.sum(D["g_axis"] * z, axis=1, keepdims=True) * z
    y /= np.linalg.norm(y, axis=1, keepdims=True) + 1e-12
    x = np.cross(y, z)
    return np.stack([x, y, z], axis=2)          # [n, 3, 3], columns = axes


def features(D: dict) -> tuple[np.ndarray, np.ndarray]:
    """Per-grasp scalar features (float32) and frames."""
    R = grasp_frames(D)
    c = D["g_center"]
    o = D["g_obj"]
    com = np.einsum("nji,nj->ni", R, D["obj_com"][o] - c) / SCALE
    up = np.einsum("nji,j->ni", R, np.array([0.0, 0.0, 1.0]))
    S = np.column_stack([
        D["g_width"] / 0.08,
        np.log(D["obj_mass"][o]),
        com, up,
        c[:, 2] / SCALE,
        D["g_score"],
        D["obj_extents"][o] / SCALE,
    ]).astype(np.float32)
    return S, R.astype(np.float32)


def torque_arm(D: dict, R: np.ndarray) -> np.ndarray:
    """Heuristic: horizontal distance from COM to the grasp axis (gravity
    torque arm about the pinch line). Smaller = better, so return -dist."""
    o = D["g_obj"]
    v = D["obj_com"][o] - D["g_center"]
    ax = R[:, :, 1]
    perp = v - np.sum(v * ax, axis=1, keepdims=True) * ax
    perp[:, 2] = 0.0
    return -np.linalg.norm(perp, axis=1)


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------

class GraspNet(nn.Module):
    def __init__(self, n_scalar: int, width: int = 128):
        super().__init__()
        self.point = nn.Sequential(
            nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, width, 1), nn.BatchNorm1d(width), nn.ReLU(),
            nn.Conv1d(width, 2 * width, 1), nn.BatchNorm1d(2 * width), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(2 * width + n_scalar, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, pts, scal):              # pts [B, 3, P]
        f = self.point(pts).max(dim=2).values
        return self.head(torch.cat([f, scal], dim=1)).squeeze(1)


def batch_points(D, idx, R, rng=None, n_pts=256, device="cpu"):
    o = D["g_obj"][idx]
    P = D["obj_points"][o]                                   # [B, Pfull, 3]
    if rng is not None:
        sel = rng.integers(0, P.shape[1], size=(len(idx), n_pts))
    else:
        sel = np.broadcast_to(np.linspace(0, P.shape[1] - 1, n_pts).astype(int),
                              (len(idx), n_pts))
    P = np.take_along_axis(P, sel[..., None], axis=1)
    local = np.einsum("bji,bpj->bpi", R[idx], P - D["g_center"][idx][:, None, :]) / SCALE
    if rng is not None:                                      # finger-swap symmetry
        flip = rng.random(len(idx)) < 0.5
        local[flip, :, :2] *= -1
    return torch.from_numpy(np.ascontiguousarray(local.transpose(0, 2, 1),
                                                 dtype=np.float32)).to(device)


def auc(score: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import rankdata
    y = y.astype(bool)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def rank1_rates(D, idx, score) -> dict:
    """Per object: success of the top-scored feasible grasp, of the planner's
    first feasible grasp, and of any feasible grasp (oracle)."""
    o = D["g_obj"][idx]
    y = D["g_success"][idx].astype(bool)
    rank = D["g_rank"][idx]
    pick, first, anyok = [], [], []
    for obj in np.unique(o):
        m = np.where(o == obj)[0]
        pick.append(y[m[np.argmax(score[m])]])
        first.append(y[m[np.argmin(rank[m])]])
        anyok.append(y[m].any())
    return {"n_objects": len(pick), "model_top1": float(np.mean(pick)),
            "planner_first": float(np.mean(first)), "oracle_any": float(np.mean(anyok))}


# ---------------------------------------------------------------------------
# splits
# ---------------------------------------------------------------------------

def make_split(D, mode: str, seed: int, test_frac: float = 0.2):
    rng = np.random.default_rng(seed)
    o = D["g_obj"]
    if mode == "archetype":
        key = D["obj_group"][o]
    elif mode == "source":
        key = D["obj_source"][o]
    elif mode == "object":
        key = o.astype(str)
    else:
        raise ValueError(mode)
    uniq = np.unique(key)
    rng.shuffle(uniq)
    n_test = max(1, int(round(test_frac * len(uniq))))
    n_val = max(1, int(round(0.1 * len(uniq))))
    test_k, val_k = set(uniq[:n_test]), set(uniq[n_test:n_test + n_val])
    test = np.array([k in test_k for k in key])
    val = np.array([k in val_k for k in key])
    return np.where(~test & ~val)[0], np.where(val)[0], np.where(test)[0]


# ---------------------------------------------------------------------------

def predict(model, D, idx, S, R, device, bs=4096):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]
            pts = batch_points(D, b, R, device=device)
            out.append(torch.sigmoid(model(pts, torch.from_numpy(S[b]).to(device))).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, nargs="+", required=True)
    ap.add_argument("--split", default="archetype", choices=["archetype", "source", "object"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    D = load(args.data)
    S, R = features(D)
    y = D["g_success"].astype(np.float32)
    tr, va, te = make_split(D, args.split, args.seed)
    mu, sd = S[tr].mean(0), S[tr].std(0) + 1e-6
    Sn = ((S - mu) / sd).astype(np.float32)
    print(f"{len(y)} feasible grasps ({y.mean():.1%} lift) on {len(D['obj_points'])} objects; "
          f"split={args.split}: train {len(tr)} / val {len(va)} / test {len(te)}; device={device}")

    model = GraspNet(S.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    pos_w = torch.tensor((1 - y[tr].mean()) / max(y[tr].mean(), 1e-3), device=device)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    rng = np.random.default_rng(args.seed)
    best, best_state, t0 = -1.0, None, time.time()
    for ep in range(args.epochs):
        model.train()
        perm = rng.permutation(tr)
        for i in range(0, len(perm), args.batch):
            b = perm[i:i + args.batch]
            pts = batch_points(D, b, R, rng=rng, device=device)
            logit = model(pts, torch.from_numpy(Sn[b]).to(device))
            loss = lossf(logit, torch.from_numpy(y[b]).to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        va_auc = auc(predict(model, D, va, Sn, R, device), y[va])
        if va_auc > best:
            best = va_auc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"  epoch {ep:3d}  loss {loss.item():.3f}  val AUC {va_auc:.3f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
    model.load_state_dict(best_state)

    p_te = predict(model, D, te, Sn, R, device)
    res = {
        "split": args.split, "seed": args.seed, "data": [str(p) for p in args.data],
        "n": {"train": int(len(tr)), "val": int(len(va)), "test": int(len(te))},
        "test_lift_rate": float(y[te].mean()),
        "val_auc_best": best,
        "test_auc": {"model": auc(p_te, y[te]),
                     "planner_score": auc(D["g_score"][te], y[te]),
                     "torque_arm": auc(torque_arm(D, R)[te], y[te])},
        "test_rank1": {"model": rank1_rates(D, te, p_te),
                       "planner_score": rank1_rates(D, te, D["g_score"][te])},
        "test_groups": sorted(set(D["obj_group"][D["g_obj"][te]].tolist()))[:200],
        "train_seconds": time.time() - t0,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "mu": mu, "sd": sd,
                "n_scalar": S.shape[1]}, args.out / "model.pt")
    (args.out / "results.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("test_lift_rate", "test_auc", "test_rank1")}, indent=1))


if __name__ == "__main__":
    main()
