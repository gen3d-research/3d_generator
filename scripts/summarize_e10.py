#!/usr/bin/env python3
"""E10: one seed's metrics, or the study's locked comparisons over seeds.

    python scripts/summarize_e10.py --root output/e10 --seed 65 --out metrics_seed_65.json
    python scripts/summarize_e10.py --root output/e10 --seeds 65 66 ... --out summary.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

ARMS = {"cem_plain": ("plain", "cem"), "cem_aware": ("select", "cem"),
        "fixed_plain": ("plain", "fixed_cad"), "fixed_ranked": ("ranked", "fixed_cad")}


def _load(root, cfg, seed, name):
    return json.load(open(Path(root) / cfg / f"seed_{seed}" / name))["results"]


def seed_metrics(root, seed, gz_file="gazebo_lift.json") -> dict:
    out = {"seed": seed, "arms": {}}
    pairs = []
    for arm, (cfg, method) in ARMS.items():
        gz = [r for r in _load(root, cfg, seed, gz_file) if r["method"] == method]
        mj = {r["name"]: r for r in _load(root, cfg, seed, "mujoco_eval_coacd.json")}
        trials = [t for r in gz for t in r["trials"]]
        kept = [r for r in gz if not (r["trials"] and r["trials"][0].get("off_path"))]
        for r in gz:
            by_rank = {t["rank"]: t for t in mj[r["name"]].get("lift", {}).get("trials", [])}
            pairs += [(bool(t["success"]), bool(by_rank[t["rank"]].get("success", False)))
                      for t in r["trials"] if t["reason"] != "crashed"]
        mj_first = [mj[r["name"]].get("lift", {}).get("lift_first", False) for r in gz]
        mj_any = [mj[r["name"]].get("lift", {}).get("lift_any", False) for r in gz]
        out["arms"][arm] = {
            "n_objects": len(gz), "has_feasible": float(np.mean([r["has_feasible"] for r in gz])),
            "gz_lift_first": float(np.mean([r["lift_first"] for r in gz])),
            "gz_lift_any": float(np.mean([r["lift_any"] for r in gz])),
            "mj_lift_first": float(np.mean(mj_first)), "mj_lift_any": float(np.mean(mj_any)),
            "gz_lift_first_excl_off_path": float(np.mean([r["lift_first"] for r in kept])),
            "n_objects_excl_off_path": len(kept),
            "n_trials": len(trials), "n_crashed": sum(t["reason"] == "crashed" for t in trials),
            "n_off_path": sum(bool(t.get("off_path")) for t in trials),
            "n_unexplained_off_path": sum(bool(t.get("unexplained_off_path")) for t in trials)}
    a = out["arms"]
    n = sum(x["n_trials"] for x in a.values())
    out["n_trials"] = n
    out["frac_invalid"] = sum(x["n_crashed"] + x["n_unexplained_off_path"] for x in a.values()) / max(n, 1)
    out["grasp_pairs"] = {"gz_mj_both": sum(g and m for g, m in pairs),
                          "gz_only": sum(g and not m for g, m in pairs),
                          "mj_only": sum(m and not g for g, m in pairs),
                          "neither": sum(not g and not m for g, m in pairs)}
    return out


def kappa(c) -> float:
    n = sum(c.values())
    if n == 0:
        return float("nan")
    po = (c["gz_mj_both"] + c["neither"]) / n
    pg = (c["gz_mj_both"] + c["gz_only"]) / n
    pm = (c["gz_mj_both"] + c["mj_only"]) / n
    pe = pg * pm + (1 - pg) * (1 - pm)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def _p(a, b):
    return float(stats.wilcoxon(a, b).pvalue) if np.any(a != b) else 1.0


def _primary(get, key) -> list[dict]:
    """The locked comparisons on one metric, Holm-corrected over the two."""
    comps = []
    for label, a, b in (("A: cem_aware vs cem_plain", "cem_aware", "cem_plain"),
                        ("B: cem_aware vs fixed_ranked", "cem_aware", "fixed_ranked")):
        x, y = get(a, key), get(b, key)
        comps.append({"comparison": label, "diff": float((x - y).mean()), "p": _p(x, y)})
    order = sorted(range(2), key=lambda i: comps[i]["p"])
    adj = 0.0
    for rank, i in enumerate(order):
        adj = max(adj, min(1.0, (2 - rank) * comps[i]["p"]))
        comps[i]["p_holm"] = adj
    return comps


def study(root, seeds) -> dict:
    rows = [seed_metrics(root, s) for s in seeds]
    get = lambda arm, key: np.array([r["arms"][arm][key] for r in rows])   # noqa: E731
    table = {arm: {k: {"mean": float(get(arm, k).mean()),
                       "ci95": float(stats.t.ppf(0.975, len(rows) - 1) * get(arm, k).std(ddof=1) / np.sqrt(len(rows)))}
                   for k in ("gz_lift_first", "gz_lift_any", "mj_lift_first", "mj_lift_any", "has_feasible",
                             "gz_lift_first_excl_off_path")}
             for arm in ARMS}
    pairs = {k: sum(r["grasp_pairs"][k] for r in rows) for k in rows[0]["grasp_pairs"]}
    n = sum(pairs.values())
    return {"seeds": list(seeds), "arms": table, "primary": _primary(get, "gz_lift_first"),
            "sensitivity_excl_off_path": _primary(get, "gz_lift_first_excl_off_path"),
            "off_path_share": {arm: float(np.mean([r["arms"][arm]["n_off_path"] / max(r["arms"][arm]["n_trials"], 1)
                                                   for r in rows])) for arm in ARMS},
            "grasp_pairs": pairs,
            "agreement": (pairs["gz_mj_both"] + pairs["neither"]) / max(n, 1), "kappa": kappa(pairs),
            "frac_invalid_per_seed": {r["seed"]: r["frac_invalid"] for r in rows}, "per_seed": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="output/e10")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--seeds", type=int, nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    res = seed_metrics(args.root, args.seed) if args.seed is not None else study(args.root, args.seeds)
    Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
