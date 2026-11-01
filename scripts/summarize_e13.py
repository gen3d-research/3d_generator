#!/usr/bin/env python3
"""E13: both arms on real shapes, out of all objects listed, with Wilson intervals. An object that failed to build
or has no feasible grasp counts as a failure. Arm difference by exact McNemar on first-grasp lift in Gazebo.

    python scripts/summarize_e13.py --root output/e13/graded --out output/e13/graded/summary.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats


def wilson(k: int, n: int) -> dict:
    if n == 0:
        return {"k": 0, "n": 0, "rate": None, "lo": None, "hi": None}
    z = stats.norm.ppf(0.975)
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return {"k": int(k), "n": int(n), "rate": float(p), "lo": float(c - h), "hi": float(c + h)}


def auc(scores, labels) -> float | None:
    y = np.array(labels, bool)
    if y.all() or not y.any():
        return None
    r = stats.rankdata(scores)
    return float((r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


def arm(d: Path, total: int) -> tuple[dict, dict]:
    objs = json.load(open(d / "objects.json"))
    by_name = {o["name"]: o for o in objs}
    mj = {r["name"]: r for r in json.load(open(d / "mujoco_eval_coacd.json"))["results"]}
    gz = {r["name"]: r for r in json.load(open(d / "gazebo_lift.json"))["results"]}
    drop = {r["name"]: r for r in json.load(open(d / "gazebo_stability.json"))["results"]}
    per = {}
    for k, o in enumerate(objs):
        n = o["name"]
        m, g, s = mj.get(n, {}), gz.get(n, {}), drop.get(n, {})
        per[k] = {"gz_lift_first": bool(g.get("lift_first")), "gz_lift_any": bool(g.get("lift_any")),
                  "mj_lift_first": bool(m.get("lift", {}).get("lift_first")),
                  "mj_lift_any": bool(m.get("lift", {}).get("lift_any")),
                  "gz_stable": bool(s.get("stable")) and not s.get("unsettled"),
                  "mj_stable": bool(m.get("drop", {}).get("stable")),
                  "has_feasible": bool(g.get("has_feasible")), "has_grasp": bool(o.get("grasps"))}
    keys = ("gz_lift_first", "gz_lift_any", "mj_lift_first", "mj_lift_any", "gz_stable", "mj_stable",
            "has_feasible", "has_grasp")
    feas = [k for k in per if per[k]["has_feasible"]]
    trials = [t for r in gz.values() for t in r.get("trials", [])]
    # A trial's rank is its grasp's position in the manifest, which holds the predictor's score in the ranked arm.
    scored = [(by_name[n]["grasps"][t["rank"]]["p_success"], t["success"]) for n, r in gz.items()
              for t in r.get("trials", []) if t["reason"] != "crashed" and "p_success" in by_name[n]["grasps"][t["rank"]]]
    reasons: dict = {}
    for t in trials:
        reasons[t["reason"]] = reasons.get(t["reason"], 0) + 1
    out = {"all": {x: wilson(sum(per[k][x] for k in per), total) for x in keys},
           "feasible": {x: wilson(sum(per[k][x] for k in feas), len(feas)) for x in keys[:6]},
           "built": len(mj), "failed_build": sum("failed" in o for o in objs),
           "trials": len(trials), "reasons": reasons, "off_path": sum(bool(t.get("off_path")) for t in trials),
           "unexplained": sum(bool(t.get("unexplained_off_path")) for t in trials),
           "per_grasp_auc_gazebo": auc(*zip(*scored)) if scored else None, "n_scored": len(scored),
           "objects": len(by_name)}
    return out, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("output/e13/graded"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    total = len(json.load(open(args.root / "prompts" / "seed_0.json")))
    plain, a = arm(args.root / "gso_plain" / "seed_0", total)
    ranked, b = arm(args.root / "gso_ranked" / "seed_0", total)
    only_r = sum(b[k]["gz_lift_first"] and not a[k]["gz_lift_first"] for k in a)
    only_p = sum(a[k]["gz_lift_first"] and not b[k]["gz_lift_first"] for k in a)
    p = stats.binomtest(only_r, only_r + only_p, 0.5).pvalue if only_r + only_p else 1.0
    res = {"total": total, "gso_plain": plain, "gso_ranked": ranked,
           "mcnemar_gz_first": {"only_ranked": only_r, "only_plain": only_p, "p": float(p)}}
    args.out.write_text(json.dumps(res, indent=1))
    for name, r in (("plain", plain), ("ranked", ranked)):
        print(name, "built", r["built"], "failed", r["failed_build"], "trials", r["trials"], r["reasons"],
              "off_path", r["off_path"], "unexplained", r["unexplained"], "auc", r["per_grasp_auc_gazebo"])
        for scope in ("all", "feasible"):
            print(" ", scope, {k: f"{100 * v['rate']:.1f} [{100 * v['lo']:.1f},{100 * v['hi']:.1f}] n={v['n']}"
                               for k, v in r[scope].items() if v["n"]})
    print("mcnemar gazebo first", res["mcnemar_gz_first"])


if __name__ == "__main__":
    main()
