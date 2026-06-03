#!/usr/bin/env python3
"""
Multi-seed headless Gazebo dynamic-stability sweep (paper-grade).

For each seed: train v1 (paper_repro) and v2 generators, generate N objects
each, export (union visual mesh + overlap-aware inertia), box-patch the SDF
collisions, then spawn every object into ONE running headless gz-sim world,
let it settle, and record stable/unstable. Aggregates per-seed stable-rate per
method as mean +/- 95% CI with a paired Wilcoxon test (v2 vs v1).

Run from 3d_generator/ (gz sim must be installed):
    python scripts/sweep_sim_stability.py --seeds 8 --n 5 --iterations 12
"""
import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PKG = ROOT / "ros2_ws" / "src" / "generated_objects_eval"
sys.path.insert(0, str(PKG / "generated_objects_eval"))

from generator import (RoboticObjectGenerator, GeneratorConfig,            # noqa: E402
                       paper_repro_generator)
from export import URDFExporter, ExportConfig                             # noqa: E402
import yaml                                                                # noqa: E402

WORLD = PKG / "worlds" / "panda_eval_world.sdf"
CFG = PKG / "config" / "eval_config.yaml"


def build_manifest(seeds, n, iters, out_dir):
    exporter = URDFExporter(ExportConfig(use_mesh_inertia=True, union_visual_mesh=True))
    entries = []
    for s in seeds:
        print(f"  [seed {s}] training v1 + v2 ...", flush=True)
        v1 = paper_repro_generator(seed=s)
        v1.config.cem_iterations = iters
        v1.train(verbose=False)
        v2 = RoboticObjectGenerator(GeneratorConfig(
            seed=s, max_primitives=8, cem_iterations=iters))
        v2.train(verbose=False)
        for method, gen in (("v1", v1), ("v2", v2)):
            for i, o in enumerate(gen.generate(n)):
                name = f"{method}_s{s}_{i:02d}"
                o.name = name
                p = exporter.export(o, out_dir / name, name)
                entries.append({
                    "name": name, "method": method, "seed": int(s),
                    "sdf": str(Path(p["sdf"]).resolve()),
                    "visual_mesh": str(Path(p["visual_mesh"]).resolve()),
                    "collision_mesh": str(Path(p["collision_mesh"]).resolve()),
                    "n_primitives": len(o.primitives)})
    mpath = out_dir / "manifest.json"
    mpath.write_text(json.dumps(entries, indent=2))
    return mpath, entries


def launch_world():
    subprocess.run(["pkill", "-f", "gz sim"], capture_output=True)
    proc = subprocess.Popen(
        ["gz", "sim", "-s", "-r", "-v", "1", str(WORLD)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(60):
        out = subprocess.run(["gz", "service", "-l"], capture_output=True,
                             timeout=10).stdout.decode(errors="ignore")
        if "panda_eval_world/create" in out:
            return proc
        time.sleep(1.0)
    proc.terminate()
    raise RuntimeError("gz sim world did not come up")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--seed-start", type=int, default=42)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--iterations", type=int, default=12)
    ap.add_argument("--out", type=str, default=str(ROOT / "output" / "sim_sweep"))
    args = ap.parse_args()

    seeds = [args.seed_start + i for i in range(args.seeds)]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Stage A: build + export objects ...")
    mpath, entries = build_manifest(seeds, args.n, args.iterations, out_dir)

    print("Stage B: box-patch SDF collisions ...")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "patch_sdf_collision.py"),
                    "--manifest", str(mpath)], check=True, capture_output=True)

    print("Stage C: launch headless gz world + settle test ...")
    cfg = yaml.safe_load(CFG.read_text())
    import gazebo_stability_eval as gse
    proc = launch_world()
    results = []
    try:
        for k, e in enumerate(entries):
            print(f"  [{k+1}/{len(entries)}] {e['name']}", flush=True)
            r = gse.evaluate_one(e, cfg)
            r["seed"] = e["seed"]
            results.append(r)
    finally:
        proc.terminate()
        subprocess.run(["pkill", "-f", "gz sim"], capture_output=True)

    # per (method, seed) stable-rate
    by = defaultdict(lambda: {"spawn": 0, "stable": 0})
    for r in results:
        key = (r["method"], r["seed"])
        by[key]["spawn"] += int(r.get("spawn_ok", False))
        by[key]["stable"] += int(r.get("stable", False))
    per_seed_rate = defaultdict(dict)
    for (m, s), c in by.items():
        per_seed_rate[m][s] = c["stable"] / max(1, c["spawn"])

    def ci95(vals):
        a = np.asarray(vals, float); nn = len(a); mean = float(a.mean())
        if nn < 2:
            return mean, 0.0
        sem = float(a.std(ddof=1) / np.sqrt(nn))
        try:
            from scipy import stats
            return mean, float(stats.t.ppf(0.975, nn - 1) * sem)
        except Exception:
            return mean, 1.96 * sem

    agg = {}
    for m in ("v1", "v2"):
        rates = [per_seed_rate[m][s] for s in seeds if s in per_seed_rate[m]]
        mean, h = ci95(rates)
        agg[m] = {"mean": mean, "ci95": h, "per_seed": rates}

    sig = {}
    try:
        from scipy import stats
        v1r = np.array([per_seed_rate["v1"][s] for s in seeds], float)
        v2r = np.array([per_seed_rate["v2"][s] for s in seeds], float)
        sig = {"mean_diff": float((v2r - v1r).mean()),
               "wilcoxon_p": (None if np.allclose(v1r, v2r)
                              else float(stats.wilcoxon(v2r, v1r).pvalue))}
    except Exception:
        pass

    out = {"seeds": seeds, "n_per_method_per_seed": args.n,
           "aggregate": agg, "significance": sig, "results": results}
    (out_dir / "sim_sweep.json").write_text(json.dumps(out, indent=2))

    print(f"\n=== Gazebo stability, {len(seeds)} seeds x {args.n} obj/method ===")
    for m in ("v1", "v2"):
        e = agg[m]
        print(f"  {m:<4} stable rate = {e['mean']*100:.1f}% +/- {e['ci95']*100:.1f} "
              f"(per-seed: {[round(x*100) for x in e['per_seed']]})")
    if sig:
        ps = "n/a" if sig.get("wilcoxon_p") is None else f"{sig['wilcoxon_p']:.3f}"
        print(f"  v2 - v1 mean diff = {sig['mean_diff']*100:+.1f}%  Wilcoxon p = {ps}")
    print(f"\nSaved to {out_dir/'sim_sweep.json'}")


if __name__ == "__main__":
    main()
