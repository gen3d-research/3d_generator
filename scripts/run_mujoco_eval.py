#!/usr/bin/env python3
"""Cross-simulator evaluation in MuJoCo: drop test + physics pick-and-lift.

Runs ``src/mujoco_eval.py`` over the same eval manifests the ROS 2 / Gazebo
evaluators use, in parallel over CPU cores, then aggregates per method across
seeds (mean, 95 % t-CI, paired Wilcoxon CEM vs best rival) and, where a
``gazebo_stability.json`` sits next to the manifest, reports the per-object
agreement between the MuJoCo and Gazebo drop verdicts.

Usage (from 3d_generator/):

    # every seed dir of a run_multi_seed.sh result set (Gazebo-parity collision)
    python scripts/run_mujoco_eval.py --root output/journal_10seed --seeds 42 43 44

    # same, true-geometry collision (CoACD decomposition, cached per mesh)
    python scripts/run_mujoco_eval.py --root output/journal_10seed --seeds 42 43 44 \
        --collision coacd

    # one manifest
    python scripts/run_mujoco_eval.py --manifest output/seed_42/eval_manifest.json \\
        --out output/seed_42/mujoco_eval.json

Needs ``pip install mujoco`` (+ ``coacd`` for --collision coacd); no ROS, no GPU.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import fields
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mujoco_eval import MujocoEvalConfig, config_dict, evaluate_entry  # noqa: E402

METHOD_ORDER = ["cem", "fixed_cad", "cmaes", "random_search", "ga"]
METRICS = ["mj_drop_stable", "mj_has_feasible", "mj_lift_first", "mj_lift_any"]

_CFG = None
_LIFT = True


def _init(cfg_dict: dict, lift: bool) -> None:
    global _CFG, _LIFT
    _CFG, _LIFT = MujocoEvalConfig(**cfg_dict), lift


def _work(entry: dict) -> dict:
    return evaluate_entry(entry, _CFG, lift=_LIFT)


def run_manifest(manifest: Path, out: Path, cfg: MujocoEvalConfig, lift: bool,
                 workers: int, gazebo: Path | None = None) -> dict:
    entries = json.loads(manifest.read_text())
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers, initializer=_init,
                             initargs=(config_dict(cfg), lift)) as ex:
        results = list(ex.map(_work, entries, chunksize=1))
    doc = {"manifest": str(manifest), "config": config_dict(cfg), "lift": lift,
           "wall_s": time.time() - t0, "summary": summarize(results),
           "results": results}
    attach_gazebo(doc, gazebo or manifest.parent / "gazebo_stability.json")
    out.write_text(json.dumps(doc, indent=1))
    return doc


def attach_gazebo(doc: dict, gz_path: Path) -> None:
    """(Re)compute the MuJoCo-vs-Gazebo drop agreement from a Gazebo result
    file, if it exists (it may land after the MuJoCo run)."""
    if gz_path.exists():
        doc["summary"]["gazebo_agreement"] = gazebo_agreement(
            doc["results"], json.loads(gz_path.read_text()).get("results", []))


def summarize(results: list[dict]) -> dict:
    by = defaultdict(lambda: defaultdict(list))
    for r in results:
        m = r.get("method") or "?"
        if "error" in r:
            by[m]["errors"].append(1)
            continue
        by[m]["mj_drop_stable"].append(float(r["drop"]["stable"]))
        if "lift" in r:
            by[m]["mj_has_feasible"].append(float(r["lift"]["has_feasible"]))
            by[m]["mj_lift_first"].append(float(r["lift"]["lift_first"]))
            by[m]["mj_lift_any"].append(float(r["lift"]["lift_any"]))
    return {m: {k: (float(np.mean(v)) if k != "errors" else len(v))
                for k, v in d.items()} | {"n": len(d.get("mj_drop_stable", []))}
            for m, d in by.items()}


def gazebo_agreement(results: list[dict], gz_results: list[dict]) -> dict:
    gz = {r["name"]: r for r in gz_results if r.get("spawn_ok")}
    pairs = [(bool(r["drop"]["stable"]), bool(gz[r["name"]]["stable"]))
             for r in results if "drop" in r and r["name"] in gz]
    if not pairs:
        return {}
    a = np.array(pairs, dtype=float)
    agree = float(np.mean(a[:, 0] == a[:, 1]))
    p_mj, p_gz = a[:, 0].mean(), a[:, 1].mean()
    p_e = p_mj * p_gz + (1 - p_mj) * (1 - p_gz)
    kappa = float((agree - p_e) / (1 - p_e)) if p_e < 1 else float("nan")
    return {"n": len(pairs), "agreement": agree, "cohen_kappa": kappa,
            "mj_stable_rate": float(p_mj), "gz_stable_rate": float(p_gz),
            "mj_only_stable": int(np.sum((a[:, 0] == 1) & (a[:, 1] == 0))),
            "gz_only_stable": int(np.sum((a[:, 0] == 0) & (a[:, 1] == 1)))}


def _ci(values: list[float]) -> dict:
    v = np.asarray(values, float)
    n = len(v)
    sem = float(v.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    half = 0.0
    if n > 1 and sem > 0:
        from scipy import stats
        half = float(stats.t.ppf(0.975, n - 1) * sem)
    return {"mean": float(v.mean()), "std_sample": float(v.std(ddof=1)) if n > 1 else 0.0,
            "ci95_lo": float(v.mean() - half), "ci95_hi": float(v.mean() + half),
            "n": n, "values": v.tolist()}


def aggregate(per_seed: dict[int, dict]) -> dict:
    vals = defaultdict(lambda: defaultdict(list))
    agree = []
    for seed in sorted(per_seed):
        s = per_seed[seed]["summary"]
        for m, d in s.items():
            if m == "gazebo_agreement":
                continue
            for k in METRICS:
                if k in d:
                    vals[m][k].append(d[k])
        if s.get("gazebo_agreement"):
            agree.append(s["gazebo_agreement"])
    summary = {m: {k: _ci(v) for k, v in d.items()} for m, d in vals.items()}
    sig = {}
    if "cem" in summary:
        from scipy import stats
        for k in METRICS:
            rivals = [(m, summary[m][k]["mean"]) for m in summary
                      if m != "cem" and k in summary[m]]
            if not rivals or k not in summary["cem"]:
                continue
            best = max(rivals, key=lambda t: t[1])[0]
            a = np.asarray(summary["cem"][k]["values"])
            b = np.asarray(summary[best][k]["values"])
            n = min(len(a), len(b))
            p = float("nan")
            if n >= 6 and np.any(a[:n] != b[:n]):
                p = float(stats.wilcoxon(a[:n], b[:n]).pvalue)
            sig[k] = {"vs": best, "mean_diff": float(np.mean(a[:n] - b[:n])),
                      "wilcoxon_p": p, "n": n}
    out = {"seeds": sorted(per_seed), "summary": summary, "significance": sig}
    if agree:
        out["gazebo_agreement"] = {
            "n_objects": int(sum(g["n"] for g in agree)),
            "agreement": float(np.average([g["agreement"] for g in agree],
                                          weights=[g["n"] for g in agree])),
            "mean_cohen_kappa": float(np.nanmean([g["cohen_kappa"] for g in agree])),
            "mj_only_stable": int(sum(g["mj_only_stable"] for g in agree)),
            "gz_only_stable": int(sum(g["gz_only_stable"] for g in agree)),
        }
    return out


def print_table(agg: dict) -> None:
    def cell(m, k):
        c = agg["summary"].get(m, {}).get(k)
        if not c:
            return "—"
        return (f"{100 * c['mean']:.1f} [{100 * c['ci95_lo']:.1f}, "
                f"{100 * c['ci95_hi']:.1f}]")
    print(f"\nMuJoCo cross-simulator metrics, seeds {agg['seeds']} "
          f"(mean % [95% t-CI over seeds])\n")
    print("| Method | Drop-stable | Feasible grasp | Lift (first feasible) | Lift (any executed) |")
    print("|---|---|---|---|---|")
    for m in METHOD_ORDER + sorted(set(agg["summary"]) - set(METHOD_ORDER)):
        if m in agg["summary"]:
            print(f"| {m} | {cell(m, 'mj_drop_stable')} | {cell(m, 'mj_has_feasible')} | "
                  f"{cell(m, 'mj_lift_first')} | {cell(m, 'mj_lift_any')} |")
    if agg.get("significance"):
        print("\n| Metric | CEM vs | mean diff | Wilcoxon p | n |")
        print("|---|---|---|---|---|")
        for k, s in agg["significance"].items():
            print(f"| {k} | {s['vs']} | {100 * s['mean_diff']:+.1f} pts | "
                  f"{s['wilcoxon_p']:.4f} | {s['n']} |")
    if agg.get("gazebo_agreement"):
        g = agg["gazebo_agreement"]
        print(f"\nMuJoCo vs Gazebo drop verdicts: {100 * g['agreement']:.1f}% agree over "
              f"{g['n_objects']} objects (mean Cohen kappa {g['mean_cohen_kappa']:.2f}; "
              f"MuJoCo-only stable {g['mj_only_stable']}, Gazebo-only stable "
              f"{g['gz_only_stable']}).")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, help="dir holding seed_<seed>/eval_manifest.json")
    ap.add_argument("--seeds", type=int, nargs="+")
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--gazebo", type=Path, help="Gazebo drop results to compare against "
                    "(with --manifest; default <manifest dir>/gazebo_stability.json)")
    ap.add_argument("--out", type=Path, help="per-manifest output (with --manifest), "
                    "or the aggregate JSON (with --root; default "
                    "<root>/mujoco_aggregated[_<tag>].json)")
    ap.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 4))
    ap.add_argument("--no-lift", action="store_true", help="drop test only")
    ap.add_argument("--rerun", action="store_true", help="ignore existing per-seed results")
    ap.add_argument("--tag", default=None,
                    help="result-file suffix: mujoco_eval_<tag>.json / mujoco_aggregated_<tag>.json "
                         "(default: the collision model when it is not 'sdf')")
    for f in fields(MujocoEvalConfig):
        ap.add_argument("--" + f.name.replace("_", "-"), type=type(f.default),
                        default=f.default)
    args = ap.parse_args()
    cfg = MujocoEvalConfig(**{f.name: getattr(args, f.name) for f in fields(MujocoEvalConfig)})
    lift = not args.no_lift
    tag = args.tag if args.tag is not None else ("" if cfg.collision == "sdf" else cfg.collision)
    sfx = f"_{tag}" if tag else ""

    if args.manifest:
        out = args.out or args.manifest.parent / f"mujoco_eval{sfx}.json"
        doc = run_manifest(args.manifest, out, cfg, lift, args.workers, args.gazebo)
        print(json.dumps(doc["summary"], indent=1))
        print(f"wrote {out} ({doc['wall_s']:.0f}s)")
        return

    if not (args.root and args.seeds):
        ap.error("give --manifest, or --root with --seeds")
    per_seed = {}
    for seed in args.seeds:
        d = args.root / f"seed_{seed}"
        out = d / f"mujoco_eval{sfx}.json"
        if out.exists() and not args.rerun:
            doc = json.loads(out.read_text())
            attach_gazebo(doc, d / "gazebo_stability.json")
            per_seed[seed] = doc
            print(f"[seed {seed}] cached {out}")
            continue
        doc = run_manifest(d / "eval_manifest.json", out, cfg, lift, args.workers)
        per_seed[seed] = doc
        print(f"[seed {seed}] {len(doc['results'])} objects in {doc['wall_s']:.0f}s")
    agg = aggregate(per_seed)
    agg["config"] = config_dict(cfg)
    agg_out = args.out or args.root / f"mujoco_aggregated{sfx}.json"
    agg_out.write_text(json.dumps(agg, indent=1))
    print_table(agg)
    print(f"\nwrote {agg_out}")


if __name__ == "__main__":
    main()
