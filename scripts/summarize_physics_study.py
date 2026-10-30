#!/usr/bin/env python3
"""Summarize run_physics_aware_study.sh: lift success per config x method under
nominal and perturbed MuJoCo physics, with paired Wilcoxon tests over seeds.

    python scripts/summarize_physics_study.py --root output/physics_aware
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

CONFIGS = ["plain", "grasps", "select"]
TAGS = [("coacd", "nominal"), ("coacd_mu0.3", "mu 0.3"), ("coacd_mu1.0", "mu 1.0"),
        ("coacd_grip10.0", "10 N"), ("coacd_grip40.0", "40 N")]


def per_seed(root: Path, cfg: str, tag: str, method: str, metric: str) -> dict:
    out = {}
    for f in sorted((root / cfg).glob(f"seed_*/mujoco_eval_{tag}.json")):
        s = json.loads(f.read_text())["summary"].get(method)
        if s and metric in s:
            out[int(f.parent.name.split("_")[1])] = s[metric]
    return out


def paired(a: dict, b: dict) -> tuple[float, float, int]:
    k = sorted(set(a) & set(b))
    if not k:
        return float("nan"), float("nan"), 0
    x, y = np.array([a[i] for i in k]), np.array([b[i] for i in k])
    p = float(stats.wilcoxon(x, y).pvalue) if len(k) >= 6 and np.any(x != y) else float("nan")
    return float(np.mean(x - y)), p, len(k)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("output/physics_aware"))
    ap.add_argument("--metric", default="mj_lift_first")
    args = ap.parse_args()
    res = {}
    print(f"\n{args.metric} (mean % over seeds)\n")
    print("| config | method | " + " | ".join(t[1] for t in TAGS) + " |")
    print("|---|---|" + "---|" * len(TAGS))
    for cfg in CONFIGS:
        for m in ("cem", "fixed_cad"):
            cells = []
            for tag, _ in TAGS:
                d = per_seed(args.root, cfg, tag, m, args.metric)
                res[(cfg, m, tag)] = d
                cells.append(f"{100 * np.mean(list(d.values())):.1f}" if d else "—")
            print(f"| {cfg} | {m} | " + " | ".join(cells) + " |")
    print("\n| comparison | physics | mean diff (pts) | Wilcoxon p | n |")
    print("|---|---|---|---|---|")
    comps = [("CEM select vs CEM plain", ("select", "cem"), ("plain", "cem")),
             ("CEM grasps vs CEM plain", ("grasps", "cem"), ("plain", "cem")),
             ("CEM select vs Fixed CAD plain", ("select", "cem"), ("plain", "fixed_cad")),
             ("CEM select vs Fixed CAD grasps", ("select", "cem"), ("grasps", "fixed_cad"))]
    summary = []
    for name, a, b in comps:
        for tag, label in TAGS:
            diff, p, n = paired(res[(*a, tag)], res[(*b, tag)])
            summary.append({"comparison": name, "physics": label, "diff": diff, "p": p, "n": n})
            print(f"| {name} | {label} | {100 * diff:+.1f} | {p:.4f} | {n} |")
    (args.root / f"summary_{args.metric}.json").write_text(json.dumps(
        {"table": {f"{c}|{m}|{t}": v for (c, m, t), v in res.items()}, "tests": summary},
        indent=1))


if __name__ == "__main__":
    main()
