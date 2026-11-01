#!/usr/bin/env python3
"""Throughput of the generator: CEM training on 1..N cores, with the proxy objective
and with the learned predictor in the loop, and the per-object cost of each later
stage. Run on an idle machine.

    OMP_NUM_THREADS=1 python scripts/benchmark_throughput.py \\
        --model output/grasp_predictor/arch_object_s0/model.npz --out output/throughput.json
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from export import ExportConfig, URDFExporter
from generator import GeneratorConfig, RoboticObjectGenerator
from grasp_planner import plan_grasps


def train_rate(workers: int, samples: int, model) -> dict:
    cfg = GeneratorConfig(cem_iterations=2, cem_samples=samples // 2, seed=42,
                          cem_workers=workers, grasp_model=model)
    t = time.perf_counter()
    RoboticObjectGenerator(cfg).train(verbose=False)
    dt = time.perf_counter() - t
    return {"workers": workers, "samples": samples, "seconds": dt, "samples_per_s": samples / dt}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--samples-per-worker", type=int, default=32)
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", type=Path, default=ROOT / "output" / "throughput.json")
    args = ap.parse_args()
    res = {"cpu": platform.processor(), "cores": os.cpu_count(),
           "omp_threads": os.environ.get("OMP_NUM_THREADS"), "train": {}, "stages": {}}
    for label, model in (("proxy", None), ("predictor", args.model)):
        if label == "predictor" and not model:
            continue
        res["train"][label] = []
        for w in args.workers:
            r = train_rate(w, max(64, w * args.samples_per_worker), model)
            res["train"][label].append(r)
            print(f"{label:9s} {w:3d} workers {r['samples_per_s']:8.1f} samples/s", flush=True)

    gen = RoboticObjectGenerator(GeneratorConfig(seed=42, rerank_by_grasp=False))
    t = time.perf_counter()
    objs = gen.generate(50, ensure_quality=False)
    res["stages"]["sample_s_per_object"] = (time.perf_counter() - t) / len(objs)
    t = time.perf_counter()
    for k, o in enumerate(objs):
        plan_grasps(o, n_surface=256, max_pairs=1500, seed=k)
    res["stages"]["grasp_plan_s_per_object"] = (time.perf_counter() - t) / len(objs)
    exporter = URDFExporter(ExportConfig())
    with tempfile.TemporaryDirectory() as d:
        t = time.perf_counter()
        for k, o in enumerate(objs):
            exporter.export(o, Path(d) / str(k), f"o{k}")
        res["stages"]["export_s_per_object"] = (time.perf_counter() - t) / len(objs)
    print(json.dumps(res["stages"], indent=1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
