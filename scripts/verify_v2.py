#!/usr/bin/env python3
"""
v1-vs-v2 verification: does the v2 generative expansion (more primitives, more
types, assembly reward) actually improve the INDEPENDENT downstream metrics
(force-closure grasp success, shape diversity) — not just the proxy it trains
on?

Compares three generators on the same budget/seed:
    v1        : paper_repro_generator (4 types, <=4 prims, no assembly reward,
                no connectivity/grasp post-processing)
    v2-raw    : v2 distribution, but grasp re-rank DISABLED (so we see whether
                the learned distribution itself improves, independent of the
                selection step)
    v2-final  : full v2 pipeline (connectivity filter + grasp re-rank). NOTE the
                re-rank selects for grasp success, so this is the deployed
                pipeline, not a like-for-like distribution comparison.

Run from 3d_generator/:  python scripts/verify_v2.py --n 30 --iterations 20
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from generator import RoboticObjectGenerator, GeneratorConfig, paper_repro_generator  # noqa: E402
from grasp_planner import grasp_success_rate, GripperSpec                              # noqa: E402
from diversity import summarize_diversity                                              # noqa: E402
from scoring import ObjectScorer, ScoringConfig                                        # noqa: E402


def _metrics(objs):
    # Independent metrics are computed identically for every method. Suitability
    # uses the v1 scorer (assembly_weight=0) so it is comparable across methods
    # and not biased toward the v2 objective.
    v1_scorer = ObjectScorer(ScoringConfig(assembly_weight=0.0))
    suit = float(np.mean([v1_scorer.score(o).total_score for o in objs]))
    grasp = grasp_success_rate(objs, GripperSpec(), n_surface=200, max_pairs=1000)
    div = summarize_diversity(objs, do_chamfer=True)
    counts = [len(o.primitives) for o in objs]
    types = sorted({type(p).__name__ for o in objs for p in o.primitives})
    return {
        "n_objects": len(objs),
        "suitability_v1scorer": suit,
        "grasp_success_rate": grasp["success_rate"],
        "mean_valid_grasps": grasp["mean_valid_grasps"],
        "feature_diversity": div["feature_diversity"],
        "chamfer_diversity": div["chamfer_diversity"],
        "mean_primitives": float(np.mean(counts)),
        "max_primitives": int(np.max(counts)),
        "distinct_types": len(types),
        "types": types,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--iterations", type=int, default=20)
    ap.add_argument("--samples", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-primitives", type=int, default=8)
    ap.add_argument("--out", type=str, default=str(ROOT / "output" / "verify_v2.json"))
    args = ap.parse_args()

    results = {}

    print("[v1] training paper_repro generator ...")
    v1 = paper_repro_generator(seed=args.seed)
    v1.config.cem_iterations = args.iterations
    v1.config.cem_samples = args.samples
    v1.train(verbose=False)
    results["v1"] = _metrics(v1.generate(args.n))

    print("[v2-raw] training v2 generator (grasp re-rank off) ...")
    cfg = GeneratorConfig(seed=args.seed, max_primitives=args.max_primitives,
                          cem_iterations=args.iterations, cem_samples=args.samples,
                          rerank_by_grasp=False)
    v2raw = RoboticObjectGenerator(cfg)
    v2raw.train(verbose=False)
    results["v2-raw"] = _metrics(v2raw.generate(args.n))

    print("[v2-final] full v2 pipeline (connectivity + grasp re-rank) ...")
    cfg2 = GeneratorConfig(seed=args.seed, max_primitives=args.max_primitives,
                           cem_iterations=args.iterations, cem_samples=args.samples,
                           rerank_by_grasp=True)
    v2 = RoboticObjectGenerator(cfg2)
    v2.train(verbose=False)
    results["v2-final"] = _metrics(v2.generate(args.n))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))

    # table
    cols = [("suitability_v1scorer", "suit(v1)", "{:.3f}"),
            ("grasp_success_rate", "grasp%", "{:.0%}"),
            ("mean_valid_grasps", "grasps", "{:.1f}"),
            ("feature_diversity", "feat_div", "{:.3f}"),
            ("chamfer_diversity", "chamfer", "{:.4f}"),
            ("mean_primitives", "mean_p", "{:.2f}"),
            ("distinct_types", "types", "{:d}")]
    print(f"\n{'method':<9}" + "".join(f"{h:>10}" for _, h, _ in cols))
    for m in ("v1", "v2-raw", "v2-final"):
        r = results[m]
        print(f"{m:<9}" + "".join(f"{fmt.format(r[k]):>10}" for k, _, fmt in cols))
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
