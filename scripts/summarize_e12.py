#!/usr/bin/env python3
"""E12: one model's physics results per seed against the E10 arms on the same seeds. An object that failed to
build counts as a failure of every test, out of the 25 prompted. Each model writes its own file, so a model's
results can be left out cleanly.

    python scripts/summarize_e12.py --model shap_e --seeds 65 ... 74 --out output/e12/summary_shap_e.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

ARMS = {"cem_plain": ("plain", "cem"), "cem_aware": ("select", "cem"),
        "fixed_plain": ("plain", "fixed_cad"), "fixed_ranked": ("ranked", "fixed_cad")}


def rates(d: Path, method, total=25) -> dict:
    keep = (lambda r: r.get("method") == method) if method else (lambda r: True)
    mj = [r for r in json.load(open(d / "mujoco_eval_coacd.json"))["results"] if keep(r)]
    lift = [r for r in json.load(open(d / "gazebo_lift.json"))["results"] if keep(r)]
    drop = [r for r in json.load(open(d / "gazebo_stability.json"))["results"] if keep(r)]
    return {"gz_lift_first": sum(r["lift_first"] for r in lift) / total,
            "gz_lift_any": sum(r["lift_any"] for r in lift) / total,
            "mj_lift_first": sum(r.get("lift", {}).get("lift_first", False) for r in mj) / total,
            "mj_lift_any": sum(r.get("lift", {}).get("lift_any", False) for r in mj) / total,
            "gz_stable": sum(bool(r.get("stable")) and not r.get("unsettled") for r in drop) / total,
            "mj_stable": sum(bool(r.get("drop", {}).get("stable")) for r in mj) / total,
            "has_feasible": sum(r["has_feasible"] for r in lift) / total,
            "n_built": len(mj), "n_unsettled": sum(bool(r.get("unsettled")) for r in drop),
            "n_crashed": sum(t["reason"] == "crashed" for r in lift for t in r["trials"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--e10", type=Path, default=Path("output/e10"))
    ap.add_argument("--root", type=Path, default=Path("output/e12"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rows = {args.model: [rates(args.root / args.model / f"seed_{s}", None) for s in args.seeds]}
    for arm, (cfg, method) in ARMS.items():
        rows[arm] = [rates(args.e10 / cfg / f"seed_{s}", method) for s in args.seeds]
    keys = ("gz_lift_first", "gz_lift_any", "mj_lift_first", "mj_lift_any", "gz_stable", "mj_stable", "has_feasible")
    n = len(args.seeds)
    table = {arm: {k: {"mean": float(np.mean([x[k] for x in r])),
                       "ci95": float(stats.t.ppf(0.975, n - 1) * np.std([x[k] for x in r], ddof=1) / np.sqrt(n))}
                   for k in keys} for arm, r in rows.items()}
    comp = {}
    x = rows[args.model]
    for arm in ARMS:
        comp[arm] = {}
        for k in keys[:6]:
            a, b = np.array([q[k] for q in x]), np.array([q[k] for q in rows[arm]])
            d = a - b
            ci = stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n)
            comp[arm][k] = {"diff": float(d.mean()), "ci95": float(ci),
                            "p": float(stats.wilcoxon(a, b).pvalue) if np.any(d != 0) else 1.0}
    args.out.write_text(json.dumps({"model": args.model, "seeds": args.seeds, "table": table, "vs": comp,
                                    "per_seed": rows[args.model]}, indent=1))
    for arm, t in table.items():
        print(arm, {k: f"{100 * v['mean']:.1f}±{100 * v['ci95']:.1f}" for k, v in t.items()})
    for arm in ARMS:
        print("vs", arm, {k: f"{100 * v['diff']:+.1f} p={v['p']:.3f}" for k, v in comp[arm].items()})
    print("built", sum(q["n_built"] for q in x), "of", 25 * n, "unsettled", sum(q["n_unsettled"] for q in x),
          "crashed", sum(q["n_crashed"] for q in x))


if __name__ == "__main__":
    main()
