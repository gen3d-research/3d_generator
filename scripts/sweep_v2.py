#!/usr/bin/env python3
"""
Multi-seed sweep producing paper-grade v1-vs-v2 numbers on the INDEPENDENT
downstream metrics (force-closure grasp success, feature/chamfer diversity),
plus suitability scored with the v1 scorer (comparable across methods).

For each seed it trains and generates from three configurations:
    v1        - paper_repro (4 types, <=4 prims, no assembly reward / post-proc)
    v2-raw    - full v2 distribution, grasp re-rank OFF
    v2-final  - full v2 pipeline (connectivity filter + grasp re-rank)

Aggregates across seeds as mean +/- 95% t-CI and runs a paired Wilcoxon test
(v2-final vs v1, v2-raw vs v1) per independent metric. Results are written
incrementally so partial progress survives interruption.

Run from 3d_generator/:
    python scripts/sweep_v2.py --seeds 10 --n 25 --iterations 18
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from generator import (RoboticObjectGenerator, GeneratorConfig,            # noqa: E402
                       paper_repro_generator)
from grasp_planner import grasp_success_rate, GripperSpec                  # noqa: E402
from diversity import summarize_diversity                                  # noqa: E402
from scoring import ObjectScorer, ScoringConfig                            # noqa: E402

METHODS = ["v1", "v2-raw", "v2-final"]
INDEP = ["grasp_success_rate", "feature_diversity", "chamfer_diversity"]


def _metrics(objs):
    v1_scorer = ObjectScorer(ScoringConfig(assembly_weight=0.0))
    suit = float(np.mean([v1_scorer.score(o).total_score for o in objs]))
    grasp = grasp_success_rate(objs, GripperSpec(), n_surface=200, max_pairs=1000)
    div = summarize_diversity(objs, do_chamfer=True)
    counts = [len(o.primitives) for o in objs]
    return {
        "suitability_v1scorer": suit,
        "grasp_success_rate": grasp["success_rate"],
        "mean_valid_grasps": grasp["mean_valid_grasps"],
        "feature_diversity": div["feature_diversity"],
        "chamfer_diversity": div["chamfer_diversity"],
        "mean_primitives": float(np.mean(counts)),
    }


def _run_seed(seed, n, iters, samples, max_prims):
    out = {}
    v1 = paper_repro_generator(seed=seed)
    v1.config.cem_iterations = iters
    v1.config.cem_samples = samples
    v1.train(verbose=False)
    out["v1"] = _metrics(v1.generate(n))

    raw = RoboticObjectGenerator(GeneratorConfig(
        seed=seed, max_primitives=max_prims, cem_iterations=iters,
        cem_samples=samples, rerank_by_grasp=False))
    raw.train(verbose=False)
    out["v2-raw"] = _metrics(raw.generate(n))

    fin = RoboticObjectGenerator(GeneratorConfig(
        seed=seed, max_primitives=max_prims, cem_iterations=iters,
        cem_samples=samples, rerank_by_grasp=True))
    fin.train(verbose=False)
    out["v2-final"] = _metrics(fin.generate(n))
    return out


def _ci95(vals):
    a = np.asarray(vals, float)
    n = len(a)
    mean = float(a.mean())
    if n < 2:
        return mean, 0.0
    sem = float(a.std(ddof=1) / np.sqrt(n))
    try:
        from scipy import stats
        h = float(stats.t.ppf(0.975, n - 1) * sem)
    except Exception:
        h = 1.96 * sem
    return mean, h


def _aggregate(per_seed):
    metrics = list(next(iter(per_seed.values()))["v1"].keys())
    agg = {m: {} for m in METHODS}
    for m in METHODS:
        for k in metrics:
            vals = [per_seed[s][m][k] for s in per_seed]
            mean, h = _ci95(vals)
            agg[m][k] = {"mean": mean, "ci95": h, "values": vals}
    # paired significance vs v1 on independent metrics
    sig = {}
    for k in INDEP:
        sig[k] = {}
        v1v = np.array([per_seed[s]["v1"][k] for s in per_seed], float)
        for m in ("v2-raw", "v2-final"):
            mv = np.array([per_seed[s][m][k] for s in per_seed], float)
            entry = {"n": int(len(v1v)), "mean_diff": float((mv - v1v).mean())}
            try:
                from scipy import stats
                if len(v1v) >= 2 and not np.allclose(mv, v1v):
                    entry["wilcoxon_p"] = float(stats.wilcoxon(mv, v1v).pvalue)
                else:
                    entry["wilcoxon_p"] = None
            except Exception:
                entry["wilcoxon_p"] = None
            sig[k][f"{m}_vs_v1"] = entry
    return agg, sig, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10, help="number of seeds (42..)")
    ap.add_argument("--seed-start", type=int, default=42)
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--iterations", type=int, default=18)
    ap.add_argument("--samples", type=int, default=50)
    ap.add_argument("--max-primitives", type=int, default=8)
    ap.add_argument("--out", type=str, default=str(ROOT / "output" / "sweep_v2.json"))
    args = ap.parse_args()

    seeds = [args.seed_start + i for i in range(args.seeds)]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    per_seed = {}
    for i, s in enumerate(seeds):
        print(f"=== seed {s} ({i+1}/{len(seeds)}) ===", flush=True)
        per_seed[str(s)] = _run_seed(s, args.n, args.iterations,
                                     args.samples, args.max_primitives)
        agg, sig, metrics = _aggregate(per_seed)   # incremental
        out_path.write_text(json.dumps(
            {"seeds": seeds[:i + 1], "per_seed": per_seed,
             "aggregate": agg, "significance": sig,
             "config": vars(args)}, indent=2))

    # final table
    cols = [("suitability_v1scorer", "suit(v1)", 1.0, "{:.3f}"),
            ("grasp_success_rate", "grasp%", 100.0, "{:.1f}"),
            ("feature_diversity", "feat_div", 1.0, "{:.3f}"),
            ("chamfer_diversity", "chamfer", 1.0, "{:.4f}"),
            ("mean_primitives", "mean_p", 1.0, "{:.2f}")]
    print(f"\n=== Multi-seed (n_seeds={len(seeds)}, n_obj={args.n}) mean +/- 95% CI ===")
    print(f"{'method':<9}" + "".join(f"{h:>20}" for _, h, _, _ in cols))
    for m in METHODS:
        row = f"{m:<9}"
        for k, _, sc, fmt in cols:
            e = agg[m][k]
            row += f"{(fmt.format(e['mean']*sc) + ' ±' + fmt.format(e['ci95']*sc)):>20}"
        print(row)
    print("\nPaired Wilcoxon vs v1 (independent metrics):")
    for k in INDEP:
        for comp, e in sig[k].items():
            p = e["wilcoxon_p"]
            ps = "n/a" if p is None else f"{p:.3f}"
            print(f"  {k:<20} {comp:<16} mean_diff={e['mean_diff']:+.4f}  p={ps}")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
