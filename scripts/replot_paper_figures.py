#!/usr/bin/env python3
"""Re-render the paper's data figures at IEEE-compliant label sizes.

The ICARM template requires 8-point Times for figure labels.  The original
figures were authored at ~6 in wide and then scaled down to a 3.4 in column,
which pushed their labels to 3.7-5.5 pt.  This script re-plots them at exactly
one column wide with 8 pt Nimbus Roman (the URW Times clone that ieeeconf.cls
already uses for body text), so the LaTeX scale factor is 1.0 and the labels
land at a true 8 pt.

Nothing is recomputed.  Every value is read from the cached result files that
produced the published numbers:

    output/aggregated.json     -> per-method downstream rates (Table I)
    output/analysis_cache.json -> per-object stability margins, archetype scores

Usage (from 3d_generator/):

    python3 scripts/replot_paper_figures.py --out /path/to/paper/images
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

ROOT = Path(__file__).resolve().parents[1]

# One ieeeconf column, measured with \the\linewidth in this document: 245.718 pt.
# Generating at exactly this width makes the LaTeX scale factor 1.0, so an 8 pt
# matplotlib label renders at a true 8 pt with \includegraphics[width=\linewidth].
COL_IN = 245.718 / 72.0
import os
HS = float(os.environ.get('FIGHS','0.84'))   # height scale knob for fitting the page budget
DPI = 400

# Method order and colours as published.
METHODS = [
    ("cem",           "CEM (Ours)",   "#1f77b4"),
    ("fixed_cad",     "Fixed CAD",    "#ff7f0e"),
    ("cmaes",         "CMA-ES",       "#2ca02c"),
    ("random_search", "Random",       "#d62728"),
    ("ga",            "GA",           "#9467bd"),
]


def style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Nimbus Roman", "Times New Roman", "Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "axes.linewidth": 0.6,
        "grid.linewidth": 0.4,
        "lines.linewidth": 1.0,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
    })


def save(fig, out: Path, name: str, tight: bool = True) -> None:
    p = out / name
    fig.savefig(p, bbox_inches="tight" if tight else None, pad_inches=0.02)
    plt.close(fig)
    print(f"  wrote {p}")


# Literature counts as published in the original Fig. 2 (bar labels of
# images_orig_backup/competitive_landscape.png).  These are counts of surveyed
# text-to-3D methods per release year, not experimental results.
LANDSCAPE = [(2021, 1), (2022, 3), (2023, 11), (2024, 16), (2025, 3)]


def fig_landscape(out: Path) -> None:
    """Fig. 2 - surveyed text-to-3D generation methods per release year."""
    years = [str(y) for y, _ in LANDSCAPE]
    counts = [c for _, c in LANDSCAPE]
    fig, ax = plt.subplots(figsize=(COL_IN, 1.45 * HS))
    bars = ax.bar(years, counts, color="#708EBF", edgecolor="black", linewidth=0.3, width=0.68)
    ax.bar_label(bars, padding=1.5)
    ax.set_xlabel("Release year")
    ax.set_ylabel("Methods")
    ax.set_ylim(0, max(counts) * 1.22)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    save(fig, out, "competitive_landscape.png")


def fig_comparison(agg: dict, out: Path) -> None:
    """Fig. 6 - grouped bars, three downstream metrics per method."""
    groups = [("grasp_success_rate", "Grasp synth."),
              ("moveit_plan_any", "MoveIt 2 plan"),
              ("gazebo_stable", "Gazebo stability")]
    fig, ax = plt.subplots(figsize=(COL_IN, 1.70*HS))
    n = len(METHODS)
    width = 0.15
    x = np.arange(len(groups))
    for i, (key, label, colour) in enumerate(METHODS):
        means = [agg["summary"][key][g]["mean"] * 100 for g, _ in groups]
        stds = [agg["summary"][key][g]["std"] * 100 for g, _ in groups]
        ax.bar(x + (i - (n - 1) / 2) * width, means, width, yerr=stds, capsize=1.5,
               label=label, color=colour, edgecolor="black", linewidth=0.3,
               error_kw={"linewidth": 0.6, "capthick": 0.6})
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in groups])
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 162)          # headroom so the legend never covers a bar
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(ncol=3, loc="upper center", frameon=True, handlelength=1.2,
              columnspacing=1.0, borderpad=0.3, handletextpad=0.4)
    save(fig, out, "comparison_grouped_bar.png")


def fig_stability_density(cache: dict, out: Path) -> None:
    """Fig. 7 - per-object stability-margin density per method."""
    fig, ax = plt.subplots(figsize=(COL_IN, 1.50*HS))
    series = {}
    for key, _, _ in METHODS:
        v = np.asarray(cache["methods"][key]["stability"], dtype=float)
        v = v[np.isfinite(v)]
        if v.size >= 5 and not np.allclose(v, v[0]):
            series[key] = v
    lo = min(v.min() for v in series.values()) - 0.010
    hi = max(v.max() for v in series.values()) + 0.010
    grid = np.linspace(lo, hi, 500)          # shared support for every method
    peak = 0.0
    for key, label, colour in METHODS:
        if key not in series:
            continue
        dens = gaussian_kde(series[key])(grid)
        peak = max(peak, dens.max())
        ax.plot(grid, dens, color=colour, label=label)
        ax.fill_between(grid, dens, color=colour, alpha=0.12)
    ax.set_xlim(lo, hi)
    ax.set_ylim(0, peak * 1.55)              # headroom for the legend
    ax.axvline(0.0, color="black", linestyle="--", linewidth=0.8, label="zero margin")
    ax.set_xlabel("Stability margin (m)")
    ax.set_ylabel("Density")
    ax.grid(linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(ncol=2, frameon=True, handlelength=1.2, columnspacing=1.0,
              borderpad=0.3, handletextpad=0.4, loc="upper right")
    save(fig, out, "stability_density.png")


def fig_archetypes(cache: dict, out: Path) -> None:
    """Fig. 10 - 20 archetypes ranked by first-quartile suitability score."""
    arch = cache["archetypes"]
    stats = []
    for name, scores in arch.items():
        s = np.asarray(scores, dtype=float)
        stats.append((name, np.percentile(s, 25), np.median(s), np.percentile(s, 75)))
    stats.sort(key=lambda r: r[1])
    names = [r[0].replace("_", " ") for r in stats]
    q1 = [r[1] for r in stats]
    med = [r[2] for r in stats]
    q3 = [r[3] for r in stats]

    fig, ax = plt.subplots(figsize=(COL_IN, 1.85*HS))
    x = np.arange(len(stats))
    ax.bar(x, q1, color="#1f77b4", edgecolor="black", linewidth=0.3, label="Q1")
    lo = np.clip(np.asarray(med) - np.asarray(q1), 0, None)
    hi = np.clip(np.asarray(q3) - np.asarray(med), 0, None)
    ax.errorbar(x, med, yerr=[lo, hi], fmt="o", ms=1.6, color="black",
                elinewidth=0.5, capsize=1.2, label="median + IQR")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=60, ha="right")
    ax.set_ylabel("Total suitability score")
    ax.set_ylim(0.74, 1.02)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=True, handlelength=1.2,
              borderpad=0.3, handletextpad=0.4)
    save(fig, out, "archetype_ranked_q1.png")


def fig_radar(agg: dict, out: Path) -> None:
    """Fig. 11 - three downstream metrics on one radar."""
    axes_keys = [("grasp_success_rate", "Grasp"),
                 ("moveit_plan_any", "Plan"),
                 ("gazebo_stable", "Stability")]
    ang = np.linspace(0, 2 * np.pi, len(axes_keys), endpoint=False).tolist()
    ang += ang[:1]
    # Polar axes are square, so a full-width radar is very tall.  Put the plot
    # on the left and the legend beside it to keep the figure short.
    fig = plt.figure(figsize=(COL_IN * 0.62, COL_IN * 0.62 * HS))
    ax = fig.add_subplot(projection="polar")
    for key, label, colour in METHODS:
        vals = [agg["summary"][key][k]["mean"] for k, _ in axes_keys]
        vals += vals[:1]
        ax.plot(ang, vals, color=colour, label=label, linewidth=1.0)
        ax.fill(ang, vals, color=colour, alpha=0.07)
    ax.set_xticks(ang[:-1])
    ax.set_xticklabels([lab for _, lab in axes_keys])
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.5, 1.0])
    ax.set_yticklabels(["0.5", "1.0"])
    ax.set_rlabel_position(160)      # keep radial labels off the spoke labels
    ax.tick_params(pad=1.0)
    ax.grid(linewidth=0.4, alpha=0.6)
    ax.legend(loc="center left", bbox_to_anchor=(1.20, 0.5), frameon=False,
              handlelength=1.1, borderpad=0.2, handletextpad=0.4, labelspacing=0.5)
    save(fig, out, "discussion_radar.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path, help="paper images directory")
    ap.add_argument("--data", type=Path, default=ROOT / "output")
    ap.add_argument("--with-density", action="store_true",
                    help="also re-plot Fig. 7 (kept as published by default)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    agg = json.loads((args.data / "aggregated.json").read_text())
    cache = json.loads((args.data / "analysis_cache.json").read_text())

    style()
    print("re-plotting at one column wide, 8 pt labels:")
    fig_landscape(args.out)
    fig_comparison(agg, args.out)
    # Fig. 7 (stability_density) is deliberately NOT regenerated: the authors
    # preferred the original rendering, so images/stability_density.png is kept
    # as published.  Pass --with-density to re-plot it anyway.
    if args.with_density:
        fig_stability_density(cache, args.out)
    fig_archetypes(cache, args.out)
    fig_radar(agg, args.out)


if __name__ == "__main__":
    main()
