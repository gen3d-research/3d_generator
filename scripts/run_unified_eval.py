#!/usr/bin/env python3
"""
Unified evaluation: CEM (Ours) vs CMA-ES, GA, RandomSearch, FixedCAD.

For each method we
    1. run a fixed evaluation budget,
    2. report convergence (best-so-far) and final top-K suitability scores,
    3. compute the *independent* downstream metrics:
         * force-closure grasp success rate (grasp_planner.py),
         * shape diversity (feature_diversity, chamfer_diversity),
    4. write everything to a single JSON for downstream plotting.

This is the data source for the headline figures added during the revision.
"""

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from generator import RoboticObjectGenerator, GeneratorConfig                  # noqa: E402
from scoring import ObjectScorer                                               # noqa: E402
from baselines import run_baseline                                             # noqa: E402
from grasp_planner import grasp_success_rate, GripperSpec                      # noqa: E402
from diversity import summarize_diversity                                      # noqa: E402


def _eval_objects(name, objs, scorer):
    scores = np.array([float(scorer.score(o).total_score) for o in objs])
    grasp = grasp_success_rate(objs, GripperSpec(), n_surface=200, max_pairs=1000)
    div = summarize_diversity(objs, do_chamfer=True)
    return {
        "method": name,
        "n_objects": len(objs),
        "score_mean": float(scores.mean()),
        "score_median": float(np.median(scores)),
        "score_std": float(scores.std()),
        "score_min": float(scores.min()),
        "score_max": float(scores.max()),
        "score_distribution": scores.tolist(),
        "grasp_success_rate": grasp["success_rate"],
        "grasp_mean_valid": grasp["mean_valid_grasps"],
        "grasp_median_valid": grasp["median_valid_grasps"],
        "feature_diversity": div["feature_diversity"],
        "chamfer_diversity": div.get("chamfer_diversity", 0.0),
    }


def _run_cem(budget, n_top, seed, iterations=None, n_samples=None):
    """Wrap RoboticObjectGenerator into the same budget contract."""
    iters = iterations or 30
    samples = n_samples or max(20, budget // iters)
    cfg = GeneratorConfig(cem_iterations=iters, cem_samples=samples, seed=seed)
    gen = RoboticObjectGenerator(cfg)
    t0 = perf_counter()
    gen.train(verbose=False)
    train_time = perf_counter() - t0
    objs = gen.generate(n_top, ensure_quality=False)
    history = [h["mean_elite_score"] for h in gen.training_history]
    best_so_far = []
    cum = -np.inf
    for h in history:
        cum = max(cum, h)
        best_so_far.extend([cum] * samples)
    return objs, np.array(best_so_far[:budget]), train_time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--budget", type=int, default=600,
                   help="Function evaluations per method")
    p.add_argument("--top-k", type=int, default=100,
                   help="Number of returned candidates per method")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str,
                   default=str(ROOT / "output" / "unified_eval.json"))
    args = p.parse_args()

    scorer = ObjectScorer()
    results = []
    convergence = {}

    print(f"=== Unified evaluation: budget={args.budget}, top_k={args.top_k} ===")
    # ---- CEM (ours) ----
    print("[CEM ] running ...")
    t0 = perf_counter()
    objs, conv, _ = _run_cem(args.budget, args.top_k, args.seed)
    print(f"  done ({perf_counter() - t0:.1f}s)")
    results.append(_eval_objects("cem", objs, scorer))
    convergence["cem"] = conv.tolist()

    # ---- Other baselines ----
    for name in ("cmaes", "ga", "random_search", "fixed_cad"):
        print(f"[{name:13s}] running ...")
        t0 = perf_counter()
        r = run_baseline(name, budget=args.budget, seed=args.seed,
                         top_k=args.top_k)
        print(f"  done ({perf_counter() - t0:.1f}s)")
        results.append(_eval_objects(name, r.objects, scorer))
        convergence[name] = r.history_best.tolist()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "config": {"budget": args.budget, "top_k": args.top_k, "seed": args.seed},
        "results": results,
        "convergence": convergence,
    }
    with open(out_path, "w") as f:
        json.dump(blob, f, indent=2)
    print(f"\nSaved to {out_path}")

    # ---- Summary table to stdout ----
    print("\n{:<14} {:>8} {:>8} {:>8} {:>9} {:>9} {:>9}".format(
        "method", "mean", "median", "std", "grasp%", "feat_div", "chamfer"
    ))
    for r in results:
        print("{:<14} {:>8.3f} {:>8.3f} {:>8.3f} {:>8.1%} {:>9.3f} {:>9.4f}".format(
            r["method"], r["score_mean"], r["score_median"], r["score_std"],
            r["grasp_success_rate"], r["feature_diversity"], r["chamfer_diversity"],
        ))


if __name__ == "__main__":
    main()
