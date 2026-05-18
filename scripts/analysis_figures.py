#!/usr/bin/env python3
"""Generate a *menu* of analysis-figure variants for the ICARM paper.

The existing paper figures (`evaluation_total_score.png`,
`evaluation_stability.png`, `evaluation_graspability.png`,
`archetype_comparison_N10000.png`, `archetype_failure_rate.png`) are box
plots / bar charts.  This script produces alternative views from exactly
the same data so the author can pick which fits the paper best:

5-method comparison (input: re-runs ``regen_eval_figures`` mechanics):

    01_radar_table1.png         radar of the three Table I metrics
    02_radar_full.png           radar of all six aggregated metrics
    03_grouped_bar_table1.png   grouped bar chart with error bars
    04_heatmap_methods.png      heatmap of methods x metrics
    05_cem_advantage.png        delta bar chart: CEM minus baseline
    06_score_violin.png         total score violins per method
    07_score_ecdf.png           ECDF curves of total score per method
    08_stability_kde.png        overlaid stability-margin densities
    09_pareto_scatter.png       stability vs. grasp pairs (Pareto view)
    10_component_stack.png      five score components stacked per method

20-archetype comparison (input: re-runs ``run_scale_experiment``
mechanics, default ``-n 2000`` so it finishes inside a couple of
minutes):

    11_archetype_ranked_q1.png      archetypes ranked by Q1 of total score
    12_archetype_quantile_heatmap.png 20 archetypes x [Q1, median, Q3, mean]
    13_archetype_small_multiples.png 4x5 grid of per-archetype histograms
    14_archetype_lollipop.png        lollipop (median + IQR whisker)

Raw per-object data is cached to ``output/analysis_cache.json`` so
re-runs are instant.

Usage:

    python3 scripts/analysis_figures.py                # default N=2000
    python3 scripts/analysis_figures.py --n 10000      # full scale
    python3 scripts/analysis_figures.py --refresh      # ignore cache
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from baselines import (CMAESBaseline, FixedCADBaseline, GABaseline,  # noqa: E402
                       RandomSearchBaseline)
from cem import CEMConfig, CEMOptimizer  # noqa: E402
from scoring import ObjectScorer  # noqa: E402

IMG_DIR = ROOT / "images" / "analysis"
CACHE_PATH = ROOT / "output" / "analysis_cache.json"
AGG_PATH = ROOT / "output" / "aggregated.json"

# Method order = Table I order.
METHODS: List[Tuple[str, str]] = [
    ("CEM (Ours)", "cem"),
    ("Fixed CAD", "fixed_cad"),
    ("CMA-ES", "cmaes"),
    ("Random", "random_search"),
    ("GA", "ga"),
]
METHOD_COLORS = {
    "cem": "#1F77B4",
    "fixed_cad": "#FF7F0E",
    "cmaes": "#2CA02C",
    "random_search": "#D62728",
    "ga": "#9467BD",
}
CEM_HIGHLIGHT = "#1F77B4"

# Archetype factories (matches run_scale_experiment.py).
from primitives import (create_barbell, create_bottle, create_camera,  # noqa: E402
                        create_dumbbell, create_flashlight, create_flat_box,
                        create_frying_pan, create_hammer, create_joystick,
                        create_l_shape, create_monitor, create_mug_like,
                        create_remote, create_small_box, create_snowman,
                        create_spatula, create_t_shape, create_tall_box,
                        create_u_shape, create_v_shape)
from archetype_cem import ArchetypeTrainer  # noqa: E402

ARCHETYPE_FUNCS = [
    create_small_box, create_tall_box, create_flat_box,
    create_mug_like, create_l_shape, create_dumbbell, create_hammer,
    create_bottle, create_t_shape, create_u_shape, create_v_shape,
    create_monitor, create_barbell, create_snowman, create_camera,
    create_frying_pan, create_flashlight, create_spatula, create_remote,
    create_joystick,
]


# ---------------------------------------------------------------------------
# Data collection (with on-disk cache)
# ---------------------------------------------------------------------------

def _run_cem(budget: int, top_k: int, seed: int):
    n_samples = 100
    n_iterations = max(1, budget // n_samples)
    cfg = CEMConfig(n_samples=n_samples, n_iterations=n_iterations,
                    elite_fraction=0.2, learning_rate=0.7, seed=seed)
    opt = CEMOptimizer(cfg)
    opt.optimize()
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


def collect_method_data(budget: int, top_k: int, seed: int) -> Dict:
    """Per-method, per-object metrics.  Each method emits five score
    components plus stability margin and antipodal-pair count."""
    scorer = ObjectScorer()
    runners = {
        "cem":            lambda: _run_cem(budget, top_k, seed),
        "fixed_cad":      lambda: _run_baseline(FixedCADBaseline, budget, top_k, seed),
        "cmaes":          lambda: _run_baseline(CMAESBaseline, budget, top_k, seed),
        "random_search":  lambda: _run_baseline(RandomSearchBaseline, budget, top_k, seed),
        "ga":             lambda: _run_baseline(GABaseline, budget, top_k, seed),
    }
    out = {}
    for _, key in METHODS:
        print(f"[analysis] running method={key} ...", flush=True)
        objs = runners[key]()
        rec = {"score": [], "stability": [], "grasp": [],
               "size_score": [], "stability_score": [], "grasp_score": [],
               "complexity_score": [], "validity_score": []}
        for obj in objs:
            try:
                b = scorer.score(obj)
            except Exception:
                continue
            rec["score"].append(float(b.total_score))
            rec["stability"].append(float(b.stability_margin))
            rec["grasp"].append(float(b.n_antipodal_pairs))
            rec["size_score"].append(float(b.size_score))
            rec["stability_score"].append(float(b.stability_score))
            rec["grasp_score"].append(float(b.graspability_score))
            rec["complexity_score"].append(float(b.complexity_score))
            rec["validity_score"].append(float(b.validity_score))
        out[key] = rec
    return out


def collect_archetype_data(n: int, iterations: int) -> Dict[str, List[float]]:
    scorer = ObjectScorer()
    out = {}
    for func in ARCHETYPE_FUNCS:
        name = func.__name__.replace("create_", "")
        print(f"[analysis] archetype {name} N={n} ...", flush=True)
        trainer = ArchetypeTrainer(func)
        trainer.train(iterations=iterations, samples_per_iter=100)
        objs = trainer.generate(n)
        out[name] = [float(scorer.score(o).total_score) for o in objs]
    return out


def load_cache_or_collect(refresh: bool, budget: int, top_k: int, seed: int,
                          arch_n: int, arch_iters: int) -> Dict:
    if CACHE_PATH.exists() and not refresh:
        print(f"[analysis] loading cache {CACHE_PATH}")
        with open(CACHE_PATH) as f:
            cache = json.load(f)
        if cache.get("arch_n") == arch_n:
            return cache
        print("[analysis] cache N mismatch; refreshing archetype data")
    print("[analysis] collecting fresh data ...")
    cache = {
        "budget": budget, "top_k": top_k, "seed": seed,
        "arch_n": arch_n, "arch_iters": arch_iters,
        "methods": collect_method_data(budget, top_k, seed),
        "archetypes": collect_archetype_data(arch_n, arch_iters),
    }
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f)
    print(f"[analysis] wrote cache {CACHE_PATH}")
    return cache


# ---------------------------------------------------------------------------
# Aggregated.json helpers (for Table I-style figures)
# ---------------------------------------------------------------------------

def load_table1_summary() -> Dict:
    with open(AGG_PATH) as f:
        d = json.load(f)
    return d["summary"]


# ---------------------------------------------------------------------------
# Figure 01 — radar / spider plot of the three headline Table I metrics
# ---------------------------------------------------------------------------

def _radar_axes(metrics: List[str], display: List[str]):
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    fig = plt.figure(figsize=(6.0, 5.5), dpi=150)
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(display, fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_rlabel_position(0)
    ax.grid(True, alpha=0.4)
    return fig, ax, angles


def fig_radar(summary: Dict, metrics: List[str], display: List[str],
              out_path: Path, title: str):
    fig, ax, angles = _radar_axes(metrics, display)
    for label, key in METHODS:
        vals = [float(summary[key][m]["mean"]) for m in metrics]
        # Some are already in [0,1]; chamfer_diversity is small, scale it.
        vals_plot = [min(v, 1.0) for v in vals]
        vals_plot += vals_plot[:1]
        ax.plot(angles, vals_plot, linewidth=2.0,
                color=METHOD_COLORS[key], label=label,
                marker="o" if key == "cem" else "")
        ax.fill(angles, vals_plot, alpha=0.10, color=METHOD_COLORS[key])
    ax.set_title(title, pad=18)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.10), fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[fig]  wrote {out_path}")


# ---------------------------------------------------------------------------
# Figure 03 — grouped bar chart with error bars
# ---------------------------------------------------------------------------

def fig_grouped_bar(summary: Dict, out_path: Path):
    metrics = ["grasp_success_rate", "moveit_plan_any", "gazebo_stable"]
    display = ["Grasp synth.", "MoveIt 2 plan", "Gazebo stability"]
    n_methods = len(METHODS)
    n_metrics = len(metrics)
    x = np.arange(n_metrics)
    width = 0.16
    fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=150)
    for i, (label, key) in enumerate(METHODS):
        means = [100 * summary[key][m]["mean"] for m in metrics]
        stds = [100 * summary[key][m]["std"] for m in metrics]
        offset = (i - (n_methods - 1) / 2) * width
        ax.bar(x + offset, means, width, yerr=stds, capsize=3,
               label=label, color=METHOD_COLORS[key],
               edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(display)
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 110)
    ax.set_title("Downstream success per method (3 seeds, mean $\\pm$ std)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(ncol=3, fontsize=8, loc="lower center")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[fig]  wrote {out_path}")


# ---------------------------------------------------------------------------
# Figure 04 — heatmap of methods x metrics
# ---------------------------------------------------------------------------

def fig_heatmap_methods(summary: Dict, out_path: Path):
    metrics = ["score_mean", "grasp_success_rate", "moveit_plan_any",
               "gazebo_stable", "feature_diversity", "chamfer_diversity"]
    display = ["Score", "Grasp", "Plan", "Stable", "Feat.div", "Chamfer"]
    grid = np.array([[float(summary[k][m]["mean"]) for m in metrics]
                     for _, k in METHODS])
    # Normalize each metric column to [0,1] so the heatmap is comparable.
    norm = grid.copy()
    for j in range(grid.shape[1]):
        col = grid[:, j]
        lo, hi = col.min(), col.max()
        norm[:, j] = (col - lo) / (hi - lo) if hi > lo else 0.5
    fig, ax = plt.subplots(figsize=(7.5, 4.0), dpi=150)
    im = ax.imshow(norm, cmap="viridis", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(display)
    ax.set_yticks(range(len(METHODS)))
    ax.set_yticklabels([m[0] for m in METHODS])
    for i in range(len(METHODS)):
        for j in range(len(metrics)):
            txt = f"{grid[i, j]:.2f}" if grid[i, j] < 10 else f"{grid[i, j]:.1f}"
            colour = "white" if norm[i, j] < 0.5 else "black"
            ax.text(j, i, txt, ha="center", va="center",
                    color=colour, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02,
                 label="per-column normalised")
    ax.set_title("Cross-method, cross-metric performance (3-seed means)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[fig]  wrote {out_path}")


# ---------------------------------------------------------------------------
# Figure 05 — CEM advantage (delta) bar chart
# ---------------------------------------------------------------------------

def fig_cem_advantage(summary: Dict, out_path: Path):
    metrics = ["grasp_success_rate", "moveit_plan_any", "gazebo_stable"]
    display = ["Grasp synth.", "MoveIt 2 plan", "Gazebo stability"]
    baselines = [m for m in METHODS if m[1] != "cem"]
    fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=150)
    x = np.arange(len(baselines))
    width = 0.27
    for j, m in enumerate(metrics):
        cem_mean = summary["cem"][m]["mean"]
        deltas = [100 * (cem_mean - summary[k][m]["mean"]) for _, k in baselines]
        offset = (j - 1) * width
        ax.bar(x + offset, deltas, width, label=display[j])
        for xi, d in zip(x + offset, deltas):
            ax.text(xi, d + (1 if d >= 0 else -3), f"{d:+.1f}",
                    ha="center", fontsize=7)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([b[0] for b in baselines])
    ax.set_ylabel("CEM minus baseline (pp)")
    ax.set_title("CEM's advantage on Table I metrics (positive = CEM better)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[fig]  wrote {out_path}")


# ---------------------------------------------------------------------------
# Figure 06 — violin plots of total score per method
# ---------------------------------------------------------------------------

def fig_violin(methods_data: Dict, out_path: Path):
    fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=150)
    data = [methods_data[key]["score"] for _, key in METHODS]
    parts = ax.violinplot(data, showmedians=True, widths=0.85)
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(METHOD_COLORS[METHODS[i][1]])
        body.set_alpha(0.55)
        body.set_edgecolor("black")
    for key in ("cbars", "cmins", "cmaxes", "cmedians"):
        parts[key].set_color("black")
        parts[key].set_linewidth(0.8)
    ax.set_xticks(range(1, len(METHODS) + 1))
    ax.set_xticklabels([m[0] for m in METHODS])
    ax.set_ylabel("Total suitability score")
    ax.set_title("Score distribution per method (violin)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[fig]  wrote {out_path}")


# ---------------------------------------------------------------------------
# Figure 07 — ECDF of total score per method
# ---------------------------------------------------------------------------

def fig_ecdf(methods_data: Dict, out_path: Path):
    fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=150)
    for label, key in METHODS:
        arr = np.sort(np.asarray(methods_data[key]["score"]))
        y = np.arange(1, len(arr) + 1) / len(arr)
        ax.step(arr, y, where="post", label=label,
                color=METHOD_COLORS[key], linewidth=2.0)
    ax.set_xlabel("Total suitability score")
    ax.set_ylabel("Cumulative fraction")
    ax.set_title("Empirical CDF of suitability score by method")
    ax.grid(linestyle="--", alpha=0.4)
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[fig]  wrote {out_path}")


# ---------------------------------------------------------------------------
# Figure 08 — overlaid stability-margin kernel densities
# ---------------------------------------------------------------------------

def _kde(arr, x_grid, bw=None):
    arr = np.asarray(arr, dtype=float)
    if bw is None:
        sigma = max(arr.std(), 1e-6)
        bw = 1.06 * sigma * len(arr) ** (-0.2)
    diff = (x_grid[:, None] - arr[None, :]) / bw
    kern = np.exp(-0.5 * diff ** 2) / np.sqrt(2 * np.pi)
    return kern.sum(axis=1) / (len(arr) * bw)


def fig_stability_kde(methods_data: Dict, out_path: Path):
    fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=150)
    all_vals = np.concatenate([np.asarray(methods_data[k]["stability"])
                               for _, k in METHODS])
    lo, hi = np.quantile(all_vals, [0.01, 0.99])
    x = np.linspace(lo - 0.005, hi + 0.005, 400)
    for label, key in METHODS:
        arr = methods_data[key]["stability"]
        y = _kde(arr, x)
        ax.plot(x, y, label=label,
                color=METHOD_COLORS[key], linewidth=2.0)
        ax.fill_between(x, y, 0, alpha=0.10, color=METHOD_COLORS[key])
    ax.axvline(0.0, linestyle="--", color="0.4", linewidth=1,
               label="zero margin")
    ax.set_xlabel("Stability margin (m)")
    ax.set_ylabel("Density")
    ax.set_title("Stability margin density per method")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[fig]  wrote {out_path}")


# ---------------------------------------------------------------------------
# Figure 09 — Pareto scatter (stability vs grasp count)
# ---------------------------------------------------------------------------

def fig_pareto(methods_data: Dict, out_path: Path):
    fig, ax = plt.subplots(figsize=(7.0, 5.2), dpi=150)
    for label, key in METHODS:
        x = methods_data[key]["stability"]
        y = methods_data[key]["grasp"]
        ax.scatter(x, y, alpha=0.55, s=22,
                   color=METHOD_COLORS[key],
                   edgecolor="black", linewidth=0.3, label=label)
    ax.axvline(0.0, linestyle="--", color="0.4", linewidth=1)
    ax.set_xlabel("Stability margin (m, higher = more stable)")
    ax.set_ylabel("Antipodal grasp pair count")
    ax.set_title("Stability vs. graspability Pareto view")
    ax.grid(linestyle="--", alpha=0.4)
    ax.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[fig]  wrote {out_path}")


# ---------------------------------------------------------------------------
# Figure 10 — score-component stacked bar
# ---------------------------------------------------------------------------

def fig_component_stack(methods_data: Dict, out_path: Path):
    comps = ["size_score", "stability_score", "grasp_score",
             "complexity_score", "validity_score"]
    display = ["Size", "Stability", "Grasp", "Complexity", "Validity"]
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]
    means = np.array([[np.mean(methods_data[k][c]) for c in comps]
                      for _, k in METHODS])
    fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=150)
    bottom = np.zeros(len(METHODS))
    for j, comp in enumerate(comps):
        ax.bar(range(len(METHODS)), means[:, j], bottom=bottom,
               label=display[j], color=colors[j],
               edgecolor="white", linewidth=0.5)
        bottom += means[:, j]
    ax.set_xticks(range(len(METHODS)))
    ax.set_xticklabels([m[0] for m in METHODS])
    ax.set_ylabel("Mean per-component score (sum = total)")
    ax.set_title("Where each method spends its score budget")
    ax.legend(fontsize=8, ncol=5, loc="lower center",
              bbox_to_anchor=(0.5, -0.20))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[fig]  wrote {out_path}")


# ---------------------------------------------------------------------------
# Figure 11 — archetype ranked by Q1
# ---------------------------------------------------------------------------

def _archetype_summary(arch_data: Dict[str, List[float]]):
    rows = []
    for name, scores in arch_data.items():
        arr = np.asarray(scores)
        rows.append((name, float(np.quantile(arr, 0.25)),
                     float(np.median(arr)),
                     float(np.quantile(arr, 0.75)),
                     float(arr.mean())))
    return rows


def fig_archetype_ranked_q1(arch_data: Dict, out_path: Path):
    rows = sorted(_archetype_summary(arch_data), key=lambda r: r[1])
    names = [r[0] for r in rows]
    q1 = [r[1] for r in rows]
    medians = [r[2] for r in rows]
    q3 = [r[3] for r in rows]
    fig, ax = plt.subplots(figsize=(12.0, 5.0), dpi=150)
    bar_y = q1
    ax.bar(range(len(names)), bar_y, color="#1F77B4",
           edgecolor="black", linewidth=0.4, label="Q1")
    # Overlay median markers and Q3 whiskers.
    ax.errorbar(range(len(names)), medians,
                yerr=[np.array(medians) - np.array(q1),
                      np.array(q3) - np.array(medians)],
                fmt="o", color="black", markersize=3.5, linewidth=0.8,
                label="median + IQR")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Total suitability score")
    ax.set_ylim(min(q1) - 0.03, 1.01)
    ax.set_title("20 archetypes ranked by Q1 (hardest on left)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[fig]  wrote {out_path}")


# ---------------------------------------------------------------------------
# Figure 12 — archetype quantile heatmap
# ---------------------------------------------------------------------------

def fig_archetype_quantile_heatmap(arch_data: Dict, out_path: Path):
    rows = _archetype_summary(arch_data)
    names = [r[0] for r in rows]
    grid = np.array([[r[1], r[2], r[3], r[4]] for r in rows])
    fig, ax = plt.subplots(figsize=(7.5, 8.0), dpi=150)
    im = ax.imshow(grid, cmap="viridis", aspect="auto",
                   vmin=max(0.5, grid.min()), vmax=1.0)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["Q1", "Median", "Q3", "Mean"])
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    for i in range(len(names)):
        for j in range(4):
            v = grid[i, j]
            colour = "white" if v < 0.85 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color=colour, fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02,
                 label="suitability score")
    ax.set_title("Per-archetype score summary")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[fig]  wrote {out_path}")


# ---------------------------------------------------------------------------
# Figure 13 — small-multiples grid of histograms
# ---------------------------------------------------------------------------

def fig_archetype_small_multiples(arch_data: Dict, out_path: Path):
    names = list(arch_data.keys())
    n = len(names)
    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, 2.4 * rows), dpi=140,
                             sharex=True, sharey=True)
    axes = axes.ravel()
    bins = np.linspace(0.7, 1.0, 41)
    for i, name in enumerate(names):
        ax = axes[i]
        scores = np.clip(np.asarray(arch_data[name]), 0.7, 1.0)
        ax.hist(scores, bins=bins, color="#1F77B4",
                edgecolor="white", linewidth=0.3)
        ax.set_title(name, fontsize=9)
        ax.grid(linestyle="--", alpha=0.3)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle("Per-archetype score histograms", y=1.02)
    fig.text(0.5, -0.02, "Total suitability score", ha="center")
    fig.text(-0.01, 0.5, "Count", va="center", rotation=90)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[fig]  wrote {out_path}")


# ---------------------------------------------------------------------------
# Figure 14 — lollipop plot
# ---------------------------------------------------------------------------

def fig_archetype_lollipop(arch_data: Dict, out_path: Path):
    rows = sorted(_archetype_summary(arch_data), key=lambda r: r[2])
    names = [r[0] for r in rows]
    q1 = np.array([r[1] for r in rows])
    median = np.array([r[2] for r in rows])
    q3 = np.array([r[3] for r in rows])
    fig, ax = plt.subplots(figsize=(7.5, 8.5), dpi=150)
    y = np.arange(len(names))
    ax.hlines(y, q1, q3, color="0.6", linewidth=2.2)
    ax.scatter(median, y, color="#1F77B4", s=60, zorder=3,
               edgecolor="black", linewidth=0.6, label="median")
    ax.scatter(q1, y, color="white", s=22, zorder=4,
               edgecolor="0.4", linewidth=0.6, label="Q1")
    ax.scatter(q3, y, color="white", s=22, zorder=4,
               edgecolor="0.4", linewidth=0.6, label="Q3")
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlim(min(q1.min(), 0.7) - 0.02, 1.005)
    ax.set_xlabel("Total suitability score")
    ax.set_title("Per-archetype median and IQR (ranked by median)")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.legend(fontsize=8, loc="lower left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[fig]  wrote {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--budget", type=int, default=500,
                   help="Per-method evaluation budget")
    p.add_argument("--top-k", type=int, default=100,
                   help="Top-K returned objects per method")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n", "--arch-n", dest="arch_n", type=int, default=2000,
                   help="Objects per archetype")
    p.add_argument("--arch-iters", type=int, default=20,
                   help="Per-archetype CEM iterations")
    p.add_argument("--refresh", action="store_true",
                   help="Ignore the on-disk cache")
    args = p.parse_args()

    IMG_DIR.mkdir(parents=True, exist_ok=True)

    cache = load_cache_or_collect(args.refresh, args.budget, args.top_k,
                                  args.seed, args.arch_n, args.arch_iters)
    methods_data = cache["methods"]
    arch_data = cache["archetypes"]
    summary = load_table1_summary()

    # 5-method, summary-only:
    fig_radar(summary,
              metrics=["grasp_success_rate", "moveit_plan_any", "gazebo_stable"],
              display=["Grasp", "Plan", "Stability"],
              out_path=IMG_DIR / "01_radar_table1.png",
              title="Three downstream metrics (3-seed mean)")
    fig_radar(summary,
              metrics=["score_mean", "grasp_success_rate", "moveit_plan_any",
                       "gazebo_stable", "feature_diversity", "chamfer_diversity"],
              display=["Score", "Grasp", "Plan", "Stable", "Feat.div", "Chamfer"],
              out_path=IMG_DIR / "02_radar_full.png",
              title="All aggregated metrics (chamfer rescaled to [0,1])")
    fig_grouped_bar(summary, IMG_DIR / "03_grouped_bar_table1.png")
    fig_heatmap_methods(summary, IMG_DIR / "04_heatmap_methods.png")
    fig_cem_advantage(summary, IMG_DIR / "05_cem_advantage.png")

    # 5-method, per-object distribution:
    fig_violin(methods_data, IMG_DIR / "06_score_violin.png")
    fig_ecdf(methods_data, IMG_DIR / "07_score_ecdf.png")
    fig_stability_kde(methods_data, IMG_DIR / "08_stability_kde.png")
    fig_pareto(methods_data, IMG_DIR / "09_pareto_scatter.png")
    fig_component_stack(methods_data, IMG_DIR / "10_component_stack.png")

    # 20-archetype views:
    fig_archetype_ranked_q1(arch_data, IMG_DIR / "11_archetype_ranked_q1.png")
    fig_archetype_quantile_heatmap(arch_data,
                                   IMG_DIR / "12_archetype_quantile_heatmap.png")
    fig_archetype_small_multiples(arch_data,
                                  IMG_DIR / "13_archetype_small_multiples.png")
    fig_archetype_lollipop(arch_data, IMG_DIR / "14_archetype_lollipop.png")

    print(f"\n[analysis] all figures written to {IMG_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
