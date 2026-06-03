#!/usr/bin/env python3
"""Per-archetype CEM training + score-distribution figure.

Trains an ``ArchetypeTrainer`` for each of the 20 archetype factories
defined in ``primitives.py``, scores ``-n`` objects per archetype with
the trained distribution, then emits the score-distribution box plot
used as Fig. 9 of the paper.

Usage::

    python3 scripts/run_scale_experiment.py -n 10000 --iterations 30 --train

For a quick regeneration of the paper figure (faster than 10k each):

    python3 scripts/run_scale_experiment.py -n 2000 --iterations 20 --train
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from archetype_cem import ArchetypeTrainer  # noqa: E402
from export import URDFExporter  # noqa: E402
from archetypes import ARCHETYPE_REGISTRY  # noqa: E402
from scoring import ObjectScorer  # noqa: E402

OUTPUT_DIR = ROOT / "output" / "scale_experiment"
IMG_DIR = ROOT / "images"

# All registered archetypes (single source of truth: archetypes.ARCHETYPE_REGISTRY).
ARCHETYPE_FUNCS = list(ARCHETYPE_REGISTRY.values())


def run_experiment(n_objects: int, iterations: int, train: bool,
                   export_samples: int = 0):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    all_scores = {}
    failure_rates = {}
    scorer = ObjectScorer()

    for func in ARCHETYPE_FUNCS:
        name = func.__name__.replace("create_", "")
        print(f"\n[scale] {name}: ", end="", flush=True)
        trainer = ArchetypeTrainer(func)
        if train:
            print(f"train {iterations}it; ", end="", flush=True)
            trainer.train(iterations=iterations, samples_per_iter=100)
        print(f"sample {n_objects}; ", end="", flush=True)
        objects = trainer.generate(n_objects)
        print("score; ", end="", flush=True)
        scores = [scorer.score(obj).total_score for obj in objects]
        all_scores[name] = scores
        failure_rates[name] = sum(1 for s in scores if s < 0.5) / max(1, len(scores))

        if export_samples > 0:
            arch_dir = OUTPUT_DIR / name
            arch_dir.mkdir(exist_ok=True)
            exporter = URDFExporter()
            for i in range(min(export_samples, n_objects)):
                exporter.export(objects[i], arch_dir / objects[i].name,
                                objects[i].name)
            print(f"export {min(export_samples, n_objects)}; ",
                  end="", flush=True)
        print(f"failure-rate {failure_rates[name]:.2%}", flush=True)

    # Box plot.
    print("\n[scale] writing figures ...")
    names = list(all_scores.keys())
    data: List[List[float]] = list(all_scores.values())

    fig, ax = plt.subplots(figsize=(14, 5.5), dpi=150)
    bp = ax.boxplot(data, tick_labels=names, patch_artist=True,
                    medianprops=dict(color="black", linewidth=1.0),
                    boxprops=dict(facecolor="#1F77B4", linewidth=0.8),
                    flierprops=dict(marker="o", markersize=2.5,
                                    markerfacecolor="none",
                                    markeredgecolor="0.3"))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_title(f"Per-archetype suitability score distribution "
                 f"(N={n_objects}, trained={train})")
    ax.set_ylabel("Suitability score")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    plot_path = IMG_DIR / f"archetype_comparison_N{n_objects}.png"
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[scale] wrote {plot_path}")

    # Companion bar chart of failure rates so the "below 7%" prose claim
    # is backed by a directly readable plot.
    fig, ax = plt.subplots(figsize=(14, 4.0), dpi=150)
    rates = [failure_rates[n] for n in names]
    bars = ax.bar(range(len(names)), rates,
                  color=["#D62728" if r > 0.07 else "#1F77B4" for r in rates])
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.axhline(0.07, color="0.4", linestyle="--", linewidth=1,
               label="7 % reference")
    ax.set_ylabel("Fraction with S < 0.5")
    ax.set_title("Per-archetype failure rate "
                 f"(score below 0.5, N={n_objects})")
    ax.legend(loc="upper right")
    ax.set_ylim(0, max(0.10, max(rates) * 1.2))
    fig.tight_layout()
    fail_path = IMG_DIR / f"archetype_failure_rate_N{n_objects}.png"
    fig.savefig(fail_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[scale] wrote {fail_path}")
    print("\n[scale] failure rates:")
    for n in names:
        print(f"    {n:<14s} {failure_rates[n]:6.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-n", type=int, default=2000,
                        help="Objects per archetype")
    parser.add_argument("--iterations", type=int, default=20,
                        help="Training iterations")
    parser.add_argument("--train", action="store_true",
                        help="Train the per-archetype CEM")
    parser.add_argument("--export-samples", type=int, default=0,
                        help="Export this many sample URDFs per archetype "
                             "(0 = none).")
    args = parser.parse_args()
    run_experiment(args.n, args.iterations, args.train, args.export_samples)
