#!/usr/bin/env python3
"""Emit a LaTeX snippet for Table II from output/aggregated.json.

Produces the rows in the expected method order and marks the best entry
per column with \textbf{...}.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METHOD_LABEL = {
    "cem": "CEM (Ours)",
    "fixed_cad": "Fixed CAD",
    "cmaes": "CMA-ES",
    "random_search": "Random Search",
    "ga": "Genetic Alg.",
}
METHOD_ORDER = ["cem", "fixed_cad", "cmaes", "random_search", "ga"]
COLUMNS = [
    ("grasp_success_rate", "Grasp synth.", True),
    ("moveit_plan_any", "MoveIt~2 plan", True),
    ("gazebo_stable", "Gazebo stable", True),
]


def fmt(val, std, pct):
    if pct:
        m = val * 100
        s = std * 100
        if s < 0.05:
            return f"{m:.1f}\\%"
        return f"{m:.1f} $\\pm$ {s:.1f}\\%"
    if std < 1e-6:
        return f"{val:.3f}"
    return f"{val:.3f} $\\pm$ {std:.3f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("aggregated", type=Path,
                   default=ROOT / "output" / "aggregated.json", nargs="?")
    args = p.parse_args()
    summary = json.loads(args.aggregated.read_text())["summary"]
    seeds = json.loads(args.aggregated.read_text())["seeds"]

    best = {col: None for col, _, _ in COLUMNS}
    for col, _, _ in COLUMNS:
        winning_method, winning_value = None, -float("inf")
        for m in METHOD_ORDER:
            v = summary.get(m, {}).get(col, {}).get("mean")
            if v is not None and v > winning_value:
                winning_value, winning_method = v, m
        best[col] = winning_method

    print(r"\begin{table}[h]")
    print(r"\centering")
    print(rf"\caption{{Independent downstream metrics on the top-25 objects per method, averaged over {len(seeds)} seeds ({', '.join(map(str, seeds))}). Grasp synthesis = at least one Coulomb-friction-cone force-closure grasp passes a collision-free top-down approach test; MoveIt~2 planning = at least one pre-grasp pose reachable by RRTConnect from the Panda home configuration; Gazebo stability = drift $<5$\\,cm and tilt $<25^\\circ$ after a 2.5\\,s gravity settle.}}")
    print(r"\label{tab:downstream}")
    print(r"\begin{tabular}{l" + "c" * len(COLUMNS) + "}")
    print(r"\toprule")
    header = r"\textbf{Method}"
    for _, label, _ in COLUMNS:
        header += f" & \\textbf{{{label}}}"
    print(header + r" \\")
    print(r"\midrule")
    for m in METHOD_ORDER:
        if m not in summary:
            continue
        row = METHOD_LABEL[m]
        for col, _, pct in COLUMNS:
            entry = summary[m].get(col, {})
            cell = fmt(entry.get("mean", 0.0), entry.get("std", 0.0), pct)
            if best[col] == m:
                cell = r"\textbf{" + cell + "}"
            row += " & " + cell
        print(row + r" \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


if __name__ == "__main__":
    main()
