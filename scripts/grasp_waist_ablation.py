#!/usr/bin/env python3
"""How much of the v2 planner's grasp-synthesis rate comes from its waist source?

Plans grasps for the same objects with the current planner and with its waist
source (`_find_waists`) disabled, leaving only the random antipodal sampler.
A grasp-synthesis rate says a grasp passes the planner's tests, not that a
robot can lift the object.

    python scripts/grasp_waist_ablation.py --seeds 42 43 44 --out output/waist_ablation.json
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import grasp_planner
from build_eval_manifest import _baseline_objects, _cem_objects


def run_seed(job) -> dict:
    method, seed, top_k, budget, n_draws = job
    if method == "cem":
        objs = _cem_objects(top_k, budget, seed)
    else:
        objs = _baseline_objects(method, top_k, budget, seed)
    full_waists = grasp_planner._find_waists
    rates = {}
    for label, waists in (("full", full_waists), ("no_waist", lambda *a, **k: [])):
        grasp_planner._find_waists = waists
        hits = []
        for d in range(n_draws):
            hits.append(np.mean([
                grasp_planner.plan_grasps(o, n_surface=200, max_pairs=1000,
                                          seed=1000 * d + k).has_grasp
                for k, o in enumerate(objs)]))
        rates[label] = float(np.mean(hits))
    grasp_planner._find_waists = full_waists
    return {"method": method, "seed": seed, **rates}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 52)))
    ap.add_argument("--methods", nargs="+", default=["cem", "fixed_cad"])
    ap.add_argument("--top-k", type=int, default=25)
    ap.add_argument("--budget", type=int, default=1500)
    ap.add_argument("--draws", type=int, default=3)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", type=Path, default=ROOT / "output" / "waist_ablation.json")
    args = ap.parse_args()
    jobs = [(m, s, args.top_k, args.budget, args.draws)
            for m in args.methods for s in args.seeds]
    with ProcessPoolExecutor(args.workers) as ex:
        rows = list(ex.map(run_seed, jobs))
    summary = {}
    for m in args.methods:
        r = sorted((x for x in rows if x["method"] == m), key=lambda x: x["seed"])
        full, none = [np.array([x[k] for x in r]) for k in ("full", "no_waist")]
        p = (float(stats.wilcoxon(full, none).pvalue)
             if len(r) >= 6 and np.any(full != none) else None)
        summary[m] = {"full": float(full.mean()), "no_waist": float(none.mean()),
                      "wilcoxon_p": p, "n": len(r)}
        print(f"{m}: full {100 * full.mean():.1f}  no waist {100 * none.mean():.1f}  p={p}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
