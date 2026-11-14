#!/usr/bin/env python3
"""Summarize run_predictor_scaling.sh (ROADMAP §3): AUC and top-1 lift per setting.

    python scripts/summarize_predictor_scaling.py --runs output/predictor_scaling/runs \\
        --out output/predictor_scaling/summary.json

Settings: A_frac<f> (learning curve on archetype objects), B_k<K> (all archetype data
plus the first K CEM seeds; K = 0 is A_frac1.0), C_k40 (CEM data alone). Each over
training seeds 0-2: mean and s.d.; no hypothesis tests (three seeds). Test sets: the
archetype split's held-out archetypes, and the 10-seed journal set per generator.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

GROUPS = ["cem", "cmaes", "ga", "random_search", "fixed_cad"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=Path("output/predictor_scaling/runs"))
    ap.add_argument("--out", type=Path, default=Path("output/predictor_scaling/summary.json"))
    args = ap.parse_args()

    by_setting: dict[str, list[dict]] = defaultdict(list)
    for f in sorted(args.runs.glob("*/results.json")):
        m = re.match(r"(.+)_s(\d+)$", f.parent.name)
        if not m:
            continue
        r = json.loads(f.read_text())
        by_setting[m.group(1)].append(r)
    if "A_frac1.0" in by_setting:
        by_setting["B_k0"] = by_setting["A_frac1.0"]

    def ms(vals):
        a = np.asarray(vals, float)
        return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)) if len(a) > 1 else 0.0, "n": len(a)}

    summary = {}
    for name, runs in sorted(by_setting.items()):
        s = {"n_train_grasps": ms([r["n"]["train"] for r in runs]),
             "n_train_objects": ms([r.get("n_train_objects", 0) for r in runs]),
             "n_train_extra": ms([r.get("n_train_extra", 0) for r in runs])}
        if not name.startswith("C_"):
            s["heldout_archetypes"] = {
                "auc_model": ms([r["test_auc"]["model"] for r in runs]),
                "auc_planner": ms([r["test_auc"]["planner_score"] for r in runs]),
                "top1_model": ms([r["test_rank1"]["model"]["model_top1"] for r in runs]),
                "planner_first": ms([r["test_rank1"]["model"]["planner_first"] for r in runs]),
                "oracle": ms([r["test_rank1"]["model"]["oracle_any"] for r in runs])}
        ext = {}
        for g in GROUPS:
            rows = [r["external"]["groups"][g] for r in runs if "external" in r]
            if not rows:
                continue
            ext[g] = {"n_objects": rows[0]["n_objects"], "n_grasps": rows[0]["n_grasps"],
                      "auc_model": ms([x["auc_model"] for x in rows]),
                      "auc_planner": ms([x["auc_planner"] for x in rows]),
                      "top1_model": ms([x["rank1"]["model_top1"] for x in rows]),
                      "planner_first": ms([x["rank1"]["planner_first"] for x in rows]),
                      "oracle": ms([x["rank1"]["oracle_any"] for x in rows])}
        s["journal"] = ext
        summary[name] = s
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=1))

    def cell(d):
        return f"{d['mean']:.3f}±{d['sd']:.3f}" if d["n"] > 1 else f"{d['mean']:.3f}"
    print(f"{'setting':14s} {'train g':>8s} {'arch AUC':>12s} " + " ".join(f"{g[:6]:>12s}" for g in GROUPS))
    for name, s in summary.items():
        arch = cell(s["heldout_archetypes"]["auc_model"]) if "heldout_archetypes" in s else "--"
        cols = " ".join(f"{cell(s['journal'][g]['auc_model']) if g in s['journal'] else '--':>12s}" for g in GROUPS)
        print(f"{name:14s} {s['n_train_grasps']['mean']:8.0f} {arch:>12s} {cols}")
    print("planner AUC on the journal set:", {g: round(summary[next(iter(summary))]["journal"][g]["auc_planner"]["mean"], 3)
                                            for g in GROUPS if g in summary[next(iter(summary))]["journal"]})


if __name__ == "__main__":
    main()
