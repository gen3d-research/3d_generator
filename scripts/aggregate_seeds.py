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

# Metrics that are genuinely independent of the scorer CEM optimizes against.
# Suitability ("score_mean") is deliberately excluded from significance testing.
INDEPENDENT_METRICS = [
    "grasp_success_rate", "feature_diversity", "chamfer_diversity",
    "moveit_plan_any", "gazebo_stable",
]


def _t95_half_width(sem: float, n: int) -> float:
    """Half-width of a two-sided 95% t confidence interval for the mean."""
    if n < 2 or sem == 0.0:
        return 0.0
    try:
        from scipy import stats
        return float(stats.t.ppf(0.975, n - 1) * sem)
    except Exception:
        # Normal-approx fallback if scipy is unavailable.
        return float(1.96 * sem)


def _significance_vs_best(summary: dict) -> dict:
    """Paired CEM-vs-best-alternative test per independent metric.

    Uses the Wilcoxon signed-rank test when it is applicable (the
    distribution-free choice for small paired samples) and falls back to a
    paired t-test otherwise. With only a handful of seeds these tests are
    badly underpowered — the result is reported with that caveat, not as a
    licence to claim significance from n=3.
    """
    out = {}
    if "cem" not in summary:
        return out
    for metric in INDEPENDENT_METRICS:
        if metric not in summary["cem"]:
            continue
        cem_vals = np.asarray(summary["cem"][metric]["values"], dtype=float)
        # pick the best non-cem method by mean on this metric
        rivals = [(m, summary[m][metric]["mean"]) for m in summary
                  if m != "cem" and metric in summary[m]]
        if not rivals:
            continue
        best_m = max(rivals, key=lambda t: t[1])[0]
        rival_vals = np.asarray(summary[best_m][metric]["values"], dtype=float)
        if len(cem_vals) != len(rival_vals) or len(cem_vals) < 2:
            out[metric] = {"vs": best_m, "test": "none",
                           "p_value": None, "n": int(len(cem_vals)),
                           "note": "too few paired samples"}
            continue
        diff = cem_vals - rival_vals
        test, p = "none", None
        try:
            from scipy import stats
            if np.allclose(diff, 0.0):
                test, p = "degenerate", 1.0
            else:
                try:
                    test, p = "wilcoxon", float(stats.wilcoxon(cem_vals, rival_vals).pvalue)
                except Exception:
                    test, p = "paired_t", float(stats.ttest_rel(cem_vals, rival_vals).pvalue)
        except Exception:
            pass
        out[metric] = {
            "vs": best_m, "test": test, "p_value": p, "n": int(len(cem_vals)),
            "cem_mean": float(cem_vals.mean()), "rival_mean": float(rival_vals.mean()),
            "note": "UNDERPOWERED at small n — interpret with caution",
        }
    return out


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
        entry = {}
        for k, vs in kvs.items():
            arr = np.asarray(vs, dtype=float)
            n = len(arr)
            mean = float(arr.mean())
            # NOTE: "std" keeps the original population std (ddof=0) so previously
            # reported mean±std numbers reproduce exactly. The additive fields
            # below (sample std, SEM, 95% CI) are the statistically appropriate
            # ones for an n-seed comparison and should be used in new reporting.
            std_pop = float(np.std(arr, ddof=0))
            std_sample = float(np.std(arr, ddof=1)) if n > 1 else 0.0
            sem = std_sample / np.sqrt(n) if n > 1 else 0.0
            half = _t95_half_width(sem, n)
            entry[k] = {
                "mean": mean,
                "std": std_pop,            # legacy (ddof=0) — do not change
                "std_sample": std_sample,  # ddof=1
                "sem": float(sem),
                "ci95_lo": mean - half,
                "ci95_hi": mean + half,
                "n": n,
                "values": [float(x) for x in arr],
            }
        summary[m] = entry

    # Paired significance of CEM vs the best alternative on the INDEPENDENT
    # metrics (suitability is excluded — CEM trains on that scorer). Reported
    # alongside, not in place of, the descriptive table.
    significance = _significance_vs_best(summary)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"seeds": args.seeds, "summary": summary,
                                    "significance": significance},
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

    # additive: mean with 95% CI (statistically appropriate for n seeds)
    print(f"\n## Multi-seed (n={len(args.seeds)}) — mean [95% CI]\n")
    print("| Method | Grasp synth. | MoveIt 2 plan | Gazebo stable |")
    print("|---|---|---|---|")
    for m in METHOD_ORDER:
        if m not in summary:
            continue
        s = summary[m]

        def ci_cell(key):
            if key not in s:
                return "—"
            e = s[key]
            return f"{e['mean']*100:.1f}% [{e['ci95_lo']*100:.1f}, {e['ci95_hi']*100:.1f}]"

        print(f"| {m} | {ci_cell('grasp_success_rate')} | "
              f"{ci_cell('moveit_plan_any')} | {ci_cell('gazebo_stable')} |")

    # additive: paired significance of CEM vs best alternative
    if significance:
        print("\n## CEM vs best alternative — paired test (independent metrics only)\n")
        print("| Metric | vs | test | p-value | n | note |")
        print("|---|---|---|---|---|---|")
        for metric, info in significance.items():
            p = info.get("p_value")
            p_str = "n/a" if p is None else f"{p:.3f}"
            print(f"| {metric} | {info.get('vs','—')} | {info.get('test','—')} | "
                  f"{p_str} | {info.get('n','—')} | {info.get('note','')} |")

    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
