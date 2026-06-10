#!/usr/bin/env python3
"""Spawn a sequence of objects from different methods into the running
panda_eval_world, letting each settle for a few seconds. Used to record the
archetype-tour video on the project page.

Prerequisite: a gz_sim instance already running the panda_eval_world (e.g.
launched via ``stability_world_gui.launch.py``).
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import os
WORLD_NAME = os.environ.get("GZ_EVAL_WORLD", "panda_eval_world")  # keep in sync with generated_objects_eval.constants


def _spawn(sdf: Path, name: str, x: float, y: float, z: float) -> bool:
    req = (
        f'sdf_filename: "{sdf}", name: "{name}", '
        f'pose: {{position: {{x: {x}, y: {y}, z: {z}}}, '
        f'orientation: {{w: 1.0}}}}'
    )
    proc = subprocess.run(
        ["gz", "service", "-s", f"/world/{WORLD_NAME}/create",
         "--reqtype", "gz.msgs.EntityFactory",
         "--reptype", "gz.msgs.Boolean",
         "--timeout", "3000", "--req", req],
        capture_output=True, timeout=8.0,
    )
    return proc.returncode == 0 and b"data: true" in proc.stdout


def _despawn(name: str) -> None:
    subprocess.run(
        ["gz", "service", "-s", f"/world/{WORLD_NAME}/remove",
         "--reqtype", "gz.msgs.Entity", "--reptype", "gz.msgs.Boolean",
         "--timeout", "3000", "--req", f'name: "{name}", type: MODEL'],
        capture_output=True, timeout=6.0,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--methods", nargs="+",
                   default=["cem", "fixed_cad", "cmaes"])
    p.add_argument("--per-method", type=int, default=4)
    p.add_argument("--settle-s", type=float, default=3.0)
    args = p.parse_args()

    manifest = json.loads(args.manifest.read_text())
    plan = []
    for method in args.methods:
        bucket = [e for e in manifest if e["method"] == method][:args.per_method]
        plan.extend(bucket)

    for entry in plan:
        print(f"[tour] spawning {entry['name']} ({entry['method']})", flush=True)
        ok = _spawn(Path(entry["sdf"]), entry["name"], 0.5, 0.0, 0.45)
        if not ok:
            print(f"[tour] WARN: spawn failed", flush=True)
            continue
        time.sleep(args.settle_s)
        _despawn(entry["name"])
        time.sleep(0.3)
    print("[tour] done", flush=True)


if __name__ == "__main__":
    main()
