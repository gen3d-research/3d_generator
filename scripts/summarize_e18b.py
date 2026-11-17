#!/usr/bin/env python3
"""E18b: arms B' (E12 mesh, clearing grasps first then the rest) and C' (best of k by the E18b score) against
E18's A, B and C, paired over seeds. Every prompt counts, out of 25 per seed. B' reuses A's Gazebo drop (same mesh).

    python scripts/summarize_e18b.py --model shap_e --out output/e18/summary_fb_shap_e.json
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
    dirs = {"A": args.e12 / m, "B": args.root / f"{m}_c0", "C": args.root / f"{m}_sel",
            "B'": args.root / f"{m}_fb_c0", "C'": args.root / f"{m}_fbsel"}
    rows = {arm: [] for arm in dirs}
    for s in args.seeds:
        a_drop = (dirs["A"] / f"seed_{s}" / "gazebo_stability.json").resolve()
        for arm, d in dirs.items():
            sd = d / f"seed_{s}"
            if arm in ("B", "B'") and not (sd / "gazebo_stability.json").exists():
                (sd / "gazebo_stability.json").symlink_to(a_drop)
            rows[arm].append(rates(sd, None))
    t = stats.t.ppf(0.975, n - 1)
    table = {arm: {k: {"mean": float(np.mean([x[k] for x in r])),
                       "ci95": float(t * np.std([x[k] for x in r], ddof=1) / np.sqrt(n))} for k in KEYS}
             for arm, r in rows.items()}
    comp = {}
    for x, y in (("B'", "B"), ("C'", "C"), ("B'", "A"), ("C'", "A"), ("C'", "B'")):
        comp[f"{x}-{y}"] = {}
        for k in KEYS:
            a, b = np.array([q[k] for q in rows[x]]), np.array([q[k] for q in rows[y]])
            d = a - b
            comp[f"{x}-{y}"][k] = {"diff": float(d.mean()), "ci95": float(t * d.std(ddof=1) / np.sqrt(n)),
                                   "p": float(stats.wilcoxon(a, b).pvalue) if np.any(d != 0) else 1.0,
                                   "seeds_up": int((d > 0).sum()), "seeds_down": int((d < 0).sum())}
    picks, only_other, none = [], 0, 0
    for s in args.seeds:
        for r in json.loads((dirs["C'"] / f"seed_{s}" / "selection.json").read_text()):
            picks.append(r["chosen"])
            sc = r["scores"][r["chosen"]]
            only_other += -3.0 < sc < -0.5
            none += sc <= -3.0
    k = max(picks) + 1
    b_only_other = sum(not any(g["clears_table"] for g in e["grasps"]) and bool(e["grasps"])
                       for s in args.seeds
                       for e in json.loads((dirs["B'"] / f"seed_{s}" / "eval_manifest.json").read_text()))
    sel_info = {"chosen_counts": {j: picks.count(j) for j in range(k)}, "chosen_only_nonclearing": int(only_other),
                "chosen_no_grasp": int(none), "b_prime_only_nonclearing": int(b_only_other)}
    args.out.write_text(json.dumps({"model": m, "seeds": args.seeds, "table": table, "vs": comp,
                                    "selection": sel_info, "per_seed": rows}, indent=1))
    for arm, tb in table.items():
        print(arm, {q: f"{100 * v['mean']:.1f}±{100 * v['ci95']:.1f}" for q, v in tb.items()})
    for c, v in comp.items():
        print(c, {q: f"{100 * w['diff']:+.1f} p={w['p']:.3f}" for q, w in v.items()})
    print("selection", sel_info)


if __name__ == "__main__":
    main()
