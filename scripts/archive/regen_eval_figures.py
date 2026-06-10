#!/usr/bin/env python3
"""Regenerate the two stale comparison figures used by the paper.

Fig 5 / Fig 6 / Fig 7 family — three boxplots, each comparing the five
generators (CEM, CMA-ES, GA, Random Search, Fixed CAD) on one metric:

    images/evaluation_total_score.png     — total suitability score
    images/evaluation_stability.png       — stability margin (m)
    images/evaluation_graspability.png    — antipodal grasp pair count

The originals in images/ were emitted by ``scripts/evaluate_methods.py``
back when the paper only compared Random vs. CEM.  Section VI of the
revised manuscript discusses all five methods, so the figures need to
match.

Usage::

    python3 scripts/regen_eval_figures.py --budget 500 --top-k 100 --seed 42

The budget is per-method (5x total work) and the top-K is the number
of returned candidates scored for the figure.  Defaults produce the
figure under a minute on a CPU.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baselines import (CMAESBaseline, FixedCADBaseline, GABaseline,  # noqa: E402
                       RandomSearchBaseline)
from cem import CEMConfig, CEMOptimizer  # noqa: E402
from primitives import CompositeObject  # noqa: E402
from scoring import ObjectScorer, ScoringConfig  # noqa: E402

IMG_DIR = ROOT / "images"

# Ordered exactly as Table I of the paper so the box positions are
# consistent across all three figures.
METHODS = [
    ("CEM (Ours)",      "cem"),
    ("Fixed CAD",       "fixed_cad"),
    ("CMA-ES",          "cmaes"),
    ("Random Search",   "random_search"),
    ("Genetic Alg.",    "ga"),
]


def _run_cem(budget: int, top_k: int, seed: int) -> List[CompositeObject]:
    # The CEM runs ``n_iterations`` rounds of ``n_samples``, so the total
    # evaluation count is the product; match the per-baseline budget.
    n_samples = 100
    n_iterations = max(1, budget // n_samples)
    cfg = CEMConfig(n_samples=n_samples, n_iterations=n_iterations,
                    elite_fraction=0.2, learning_rate=0.7, seed=seed)
    opt = CEMOptimizer(cfg)
    opt.optimize()
    # Sample a top-K population from the trained distribution by drawing
    # ``top_k`` fresh objects and ranking them (CEMOptimizer does not
    # cache the per-iteration candidates so we re-sample here).
    scorer = ObjectScorer()
    rng = np.random.default_rng(seed + 99991)
    pool = []
    for i in range(top_k * 3):
        obj = opt.distribution.sample_object(rng, name=f"cem_{i:04d}")
        try:
            s = float(scorer.score(obj).total_score)
        except Exception:
            s = -1.0
        pool.append((s, obj))
    pool.sort(key=lambda t: t[0], reverse=True)
    return [obj for _, obj in pool[:top_k]]


def _run_baseline(cls, budget: int, top_k: int, seed: int):
    bl = cls(budget=budget, seed=seed)
    return bl.run(top_k=top_k).objects


def collect_per_object_scores(budget: int, top_k: int, seed: int
                              ) -> Dict[str, Dict[str, List[float]]]:
    """For each method, score every returned object and harvest the
    three discriminating metrics (total score, stability margin,
    antipodal pair count)."""
    scorer = ObjectScorer()
    out: Dict[str, Dict[str, List[float]]] = {}
    runners = {
        "cem":            lambda: _run_cem(budget, top_k, seed),
        "fixed_cad":      lambda: _run_baseline(FixedCADBaseline, budget, top_k, seed),
        "cmaes":          lambda: _run_baseline(CMAESBaseline, budget, top_k, seed),
        "random_search":  lambda: _run_baseline(RandomSearchBaseline, budget, top_k, seed),
        "ga":             lambda: _run_baseline(GABaseline, budget, top_k, seed),
    }
    for _, key in METHODS:
        print(f"[regen] running {key} (budget={budget}, top_k={top_k}) ...",
              flush=True)
        objs = runners[key]()
        scores, margins, pairs = [], [], []
        for obj in objs:
            try:
                b = scorer.score(obj)
            except Exception:
                continue
            scores.append(float(b.total_score))
            margins.append(float(b.stability_margin))
            pairs.append(float(b.n_antipodal_pairs))
        out[key] = {"score": scores, "stability": margins, "grasp": pairs}
    return out


def boxplot(data_per_method: Dict[str, List[float]], *,
            title: str, ylabel: str, out_path: Path,
            ylim: tuple | None = None) -> None:
    plt.figure(figsize=(8.5, 4.5), dpi=150)
    labels, columns = [], []
    for display, key in METHODS:
        arr = np.array(data_per_method.get(key, []), dtype=float)
        arr = arr[np.isfinite(arr)]
        labels.append(display)
        columns.append(arr)
    bp = plt.boxplot(columns, tick_labels=labels, patch_artist=True,
                     medianprops=dict(color="black", linewidth=1.2),
                     boxprops=dict(facecolor="#A6CEE3", linewidth=0.8),
                     whiskerprops=dict(linewidth=0.8),
                     capprops=dict(linewidth=0.8),
                     flierprops=dict(marker="o", markersize=3,
                                     markerfacecolor="none",
                                     markeredgecolor="0.3"))
    # Highlight CEM by recolouring its box.
    if bp["boxes"]:
        bp["boxes"][0].set_facecolor("#1F77B4")
        bp["boxes"][0].set_edgecolor("#1F77B4")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.tight_layout()
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[regen] wrote {out_path}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--budget", type=int, default=500,
                   help="Per-method evaluation budget (objects scored).")
    p.add_argument("--top-k", type=int, default=100,
                   help="Number of top-scoring objects per method to "
                        "include in the box plot.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, default=IMG_DIR,
                   help="Output directory (overrides project images/).")
    args = p.parse_args()

    data = collect_per_object_scores(args.budget, args.top_k, args.seed)

    boxplot({k: v["score"]     for k, v in data.items()},
            title="Total suitability score",
            ylabel="Total score",
            out_path=args.out_dir / "evaluation_total_score.png")

    boxplot({k: v["stability"] for k, v in data.items()},
            title="Stability margin (signed distance, m)",
            ylabel="Stability margin (m)",
            out_path=args.out_dir / "evaluation_stability.png",
            ylim=(-0.06, 0.04))

    boxplot({k: v["grasp"]     for k, v in data.items()},
            title="Antipodal grasp pair count",
            ylabel="Antipodal pairs",
            out_path=args.out_dir / "evaluation_graspability.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
