#!/usr/bin/env python3
"""E18: arms A (E12 mesh, planner order), B (E12 mesh, predictor order with the table check) and C (best of k
candidates by predicted lift, predictor order) per seed, with paired comparisons over seeds. Every prompt counts:
an object not built fails every test, out of 25 per seed. Arm B's Gazebo drop is arm A's (same mesh).

    python scripts/summarize_e18.py --model shap_e --out output/e18/summary_shap_e.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

from summarize_e12 import rates

KEYS = ("gz_lift_first", "gz_lift_any", "mj_lift_first", "mj_lift_any", "gz_stable", "has_feasible")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(65, 75)))
    ap.add_argument("--e12", type=Path, default=Path("output/e12"))
    ap.add_argument("--root", type=Path, default=Path("output/e18"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    m, n = args.model, len(args.seeds)
    rows = {"A": [], "B": [], "C": []}
    for s in args.seeds:
        a_dir = args.e12 / m / f"seed_{s}"
        rows["A"].append(rates(a_dir, None))
        b_dir = args.root / f"{m}_c0" / f"seed_{s}"
        if not (b_dir / "gazebo_stability.json").exists():
            (b_dir / "gazebo_stability.json").symlink_to((a_dir / "gazebo_stability.json").resolve())
        rows["B"].append(rates(b_dir, None))
        rows["C"].append(rates(args.root / f"{m}_sel" / f"seed_{s}", None))
    t = stats.t.ppf(0.975, n - 1)
    table = {arm: {k: {"mean": float(np.mean([x[k] for x in r])),
                       "ci95": float(t * np.std([x[k] for x in r], ddof=1) / np.sqrt(n))} for k in KEYS}
             for arm, r in rows.items()}
    comp = {}
    for x, y in (("C", "A"), ("C", "B"), ("B", "A")):
        comp[f"{x}-{y}"] = {}
        for k in KEYS:
            a, b = np.array([q[k] for q in rows[x]]), np.array([q[k] for q in rows[y]])
            d = a - b
            comp[f"{x}-{y}"][k] = {"diff": float(d.mean()), "ci95": float(t * d.std(ddof=1) / np.sqrt(n)),
                                   "p": float(stats.wilcoxon(a, b).pvalue) if np.any(d != 0) else 1.0,
                                   "seeds_up": int((d > 0).sum()), "seeds_down": int((d < 0).sum())}
    picks, chosen_score, c0_score = [], [], []
    for s in args.seeds:
        sel = json.loads((args.root / f"{m}_sel" / f"seed_{s}" / "selection.json").read_text())
        for r in sel:
            picks.append(r["chosen"])
            chosen_score.append(r["scores"][r["chosen"]])
            c0_score.append(r["scores"][0])
    k = max(picks) + 1 if picks else 0
    sel_info = {"chosen_counts": {j: picks.count(j) for j in range(k)},
                "mean_score_chosen": float(np.mean([v for v in chosen_score if v >= 0])),
                "mean_score_c0": float(np.mean([v for v in c0_score if v >= 0])),
                "c0_no_grasp": int(sum(v < 0 for v in c0_score)),
                "chosen_no_grasp": int(sum(v < 0 for v in chosen_score))}
    args.out.write_text(json.dumps({"model": m, "seeds": args.seeds, "table": table, "vs": comp,
                                    "selection": sel_info, "per_seed": rows}, indent=1))
    for arm, tb in table.items():
        print(arm, {q: f"{100 * v['mean']:.1f}±{100 * v['ci95']:.1f}" for q, v in tb.items()})
    for c, v in comp.items():
        print(c, {q: f"{100 * w['diff']:+.1f} p={w['p']:.3f}" for q, w in v.items()})
    print("selection", sel_info)
    for arm, r in rows.items():
        print(arm, "built", sum(q["n_built"] for q in r), "crashed", sum(q["n_crashed"] for q in r),
              "unsettled", sum(q["n_unsettled"] for q in r))


if __name__ == "__main__":
    main()
