#!/usr/bin/env python3
"""Aggregate per-seed evaluation JSONs into a mean ± std table.

Reads ``3d_generator/output/seed_<seed>/{unified_eval,moveit_results,gazebo_stability}.json``
for every seed passed on the command line and writes a combined
``output/aggregated.json`` plus a markdown table to stdout.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METHOD_ORDER = ["cem", "fixed_cad", "cmaes", "random_search", "ga"]


def _moveit_per_method(blob):
    agg = defaultdict(lambda: {"n": 0, "any": 0, "succ": 0, "tot": 0})
    for e in blob["results"]:
        m = e["method"]
        agg[m]["n"] += 1
        agg[m]["any"] += int(e["any_success"])
        agg[m]["succ"] += e["n_success"]
        agg[m]["tot"] += e["n_grasps"]
    return {m: s["any"] / max(1, s["n"]) for m, s in agg.items()}


def _gazebo_per_method(blob):
    agg = defaultdict(lambda: {"n": 0, "stable": 0, "spawn": 0})
    for e in blob["results"]:
        m = e["method"]
        agg[m]["n"] += 1
        agg[m]["spawn"] += int(e.get("spawn_ok", False))
        agg[m]["stable"] += int(e.get("stable", False))
    return {m: s["stable"] / max(1, s["spawn"]) for m, s in agg.items()}


def _unified_per_method(blob):
    out = {}
    for r in blob["results"]:
        out[r["method"]] = {
            "score_mean": r["score_mean"],
            "grasp_success_rate": r["grasp_success_rate"],
            "feature_diversity": r["feature_diversity"],
            "chamfer_diversity": r["chamfer_diversity"],
        }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("seeds", nargs="+", type=int)
    p.add_argument("--out", type=Path,
                   default=ROOT / "output" / "aggregated.json")
    args = p.parse_args()

    per_seed = {}
    for seed in args.seeds:
        d = ROOT / "output" / f"seed_{seed}"
        unified = json.loads((d / "unified_eval.json").read_text())
        moveit = json.loads((d / "moveit_results.json").read_text())
        gazebo = json.loads((d / "gazebo_stability.json").read_text())
        per_seed[seed] = {
            "unified": _unified_per_method(unified),
            "moveit": _moveit_per_method(moveit),
            "gazebo": _gazebo_per_method(gazebo),
        }

    # collect per-method per-metric across seeds
    metrics_by_method = defaultdict(lambda: defaultdict(list))
    for seed, blob in per_seed.items():
        for m, vals in blob["unified"].items():
            for k, v in vals.items():
                metrics_by_method[m][k].append(v)
        for m, v in blob["moveit"].items():
            metrics_by_method[m]["moveit_plan_any"].append(v)
        for m, v in blob["gazebo"].items():
            metrics_by_method[m]["gazebo_stable"].append(v)

    summary = {}
    for m, kvs in metrics_by_method.items():
        summary[m] = {k: {"mean": float(np.mean(vs)),
                          "std": float(np.std(vs, ddof=0)),
                          "values": [float(x) for x in vs]}
                      for k, vs in kvs.items()}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"seeds": args.seeds, "summary": summary},
                                   indent=2))

    # markdown table
    print(f"\n## Multi-seed (n={len(args.seeds)}) — mean ± std\n")
    print("| Method | Suitability | Grasp synth. | MoveIt 2 plan | Gazebo stable | Feature div. |")
    print("|---|---|---|---|---|---|")
    for m in METHOD_ORDER:
        if m not in summary:
            continue
        s = summary[m]

        def cell(key, pct=True):
            mn = s[key]["mean"]
            sd = s[key]["std"]
            if pct:
                return f"{mn*100:.1f}% ± {sd*100:.1f}"
            return f"{mn:.3f} ± {sd:.3f}"

        print(f"| {m} | {cell('score_mean', pct=False)} | "
              f"{cell('grasp_success_rate')} | "
              f"{cell('moveit_plan_any')} | "
              f"{cell('gazebo_stable')} | "
              f"{cell('feature_diversity', pct=False)} |")

    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
