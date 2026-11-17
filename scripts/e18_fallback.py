#!/usr/bin/env python3
"""E18b: the table check as a preference, not a filter (design: ROADMAP §6, 17 Nov 2026).

For every E18 candidate mesh, plan grasps with E18's settings and seed (build_neural_manifest.py) twice: with the
table check, which gives E18's list, and without it, which gives E12's. The new list is the first (grasps that clear
the table), then the second's grasps that do not clear it, each group by predicted lift; 12 kept. Objects reuse the
E18 exports (SDF, CoACD parts): only the grasp lists change.

Writes, under --root:
  <model>_fb_c<j>/seed_<s>/eval_manifest.json   every candidate with the new order (j = 0 is arm B')
  <model>_fbsel/seed_<s>/{eval_manifest.json, selection.json}   arm C'
Selection score: the highest predicted lift among table-clearing grasps; a candidate with only non-clearing grasps
scores that value minus 2 (below any clearing one, above none), one with no grasp -3, one not built -4; ties go to
the lower j.

    python scripts/e18_fallback.py --model shap_e --seeds 65 66 --root output/e18
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

MODEL_NPZ = ROOT / "output/grasp_predictor/arch_object_s0/model.npz"
KEEP = 12


def replan(job):
    """One built object: new grasp list, each grasp with p_success and clears_table."""
    row, mesh_path, seed, k = job
    import trimesh
    from build_neural_manifest import MeshObject, _predictor
    from grasp_planner import GripperSpec, grasps_as_dicts, hand_clears_table, plan_grasps
    from grasp_predictor import predict_object
    m = trimesh.load(mesh_path, force="mesh", process=True)
    obj = MeshObject(m, row["name"])
    gs = GripperSpec()
    # The planner's candidate pool depends on max_returned, so both calls use E18's settings: the first gives
    # B's list exactly, the second E12's (arm A's) list, whose non-clearing grasps top B's up.
    kw = dict(n_surface=256, max_pairs=1500, max_returned=KEEP, seed=seed + k)
    clear = plan_grasps(obj, gs, table_check=True, **kw).grasps
    report = plan_grasps(obj, gs, table_check=False, **kw)
    other = [g for g in report.grasps if not hand_clears_table(g, gs)]
    out = []
    for group, flag in ((clear, True), (other, False)):
        dicts = grasps_as_dicts(group, KEEP)
        if dicts:
            p = predict_object(_predictor(str(MODEL_NPZ)), obj, dicts, seed + k)
            for d, pi in zip(dicts, p):
                d["p_success"], d["clears_table"] = float(pi), flag
            dicts = [dicts[i] for i in np.argsort(-p, kind="stable")]
        out += dicts
    out = out[:KEEP]
    new = dict(row, grasps=out, n_grasps_synth=len(clear),
               n_clear=len(clear), n_other=len(other))
    return new


def score(row) -> float:
    if row is None or "sdf" not in row:
        return -4.0
    gs = row.get("grasps") or []
    clear = [g["p_success"] for g in gs if g["clears_table"]]
    if clear:
        return max(clear)
    return max(g["p_success"] for g in gs) - 2.0 if gs else -3.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--root", type=Path, default=ROOT / "output/e18")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    M = args.model
    with ProcessPoolExecutor(args.workers) as ex:
        for s in args.seeds:
            cands = []
            for j in range(args.k):
                src = args.root / f"{M}_c{j}" / f"seed_{s}"
                dst = args.root / f"{M}_fb_c{j}" / f"seed_{s}"
                man = dst / "eval_manifest.json"
                if man.exists():
                    rows = json.loads(man.read_text())
                else:
                    clean = json.loads((src / "clean.json").read_text())
                    built = json.loads((src / "eval_manifest.json").read_text())
                    jobs = []
                    for r in built:
                        k = int(re.search(r"_(\d{4})$", r["name"]).group(1))
                        rec = clean[f"{k:03d}_{r['archetype']}"]
                        jobs.append((dict(r, method=f"{M}_fb_c{j}"), rec["path"], s, k))
                    rows = list(ex.map(replan, jobs, chunksize=2))
                    dst.mkdir(parents=True, exist_ok=True)
                    man.write_text(json.dumps(rows, indent=1))
                    print(f"{M}_fb_c{j} seed {s}: {len(rows)} objects, "
                          f"{sum(not r['grasps'] for r in rows)} without a grasp, "
                          f"{sum(r['n_clear'] == 0 and r['n_other'] > 0 for r in rows)} with only non-clearing",
                          flush=True)
                cands.append({int(re.search(r"_(\d{4})$", r["name"]).group(1)): r for r in rows})
            n = 25
            chosen, record = [], []
            for i in range(n):
                sc = [score(c.get(i)) for c in cands]
                j = max(range(args.k), key=lambda q: (sc[q], -q))
                record.append({"index": i, "scores": sc, "chosen": j})
                if sc[j] > -4.0:
                    chosen.append(dict(cands[j][i], method=f"{M}_fbsel", candidate=j))
            out = args.root / f"{M}_fbsel" / f"seed_{s}"
            out.mkdir(parents=True, exist_ok=True)
            (out / "eval_manifest.json").write_text(json.dumps(chosen, indent=1))
            (out / "selection.json").write_text(json.dumps(record, indent=1))
            picks = [r["chosen"] for r in record]
            print(f"{M}_fbsel seed {s}: {len(chosen)} of {n}; picks", {q: picks.count(q) for q in range(args.k)},
                  flush=True)


if __name__ == "__main__":
    main()
