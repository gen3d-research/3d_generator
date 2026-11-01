#!/usr/bin/env python3
"""E12 prompts: 25 archetype names per seed, the same for every model, with each archetype's size."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from archetypes import ARCHETYPE_REGISTRY

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, nargs="+", default=list(range(65, 75)))
ap.add_argument("--n", type=int, default=25)
ap.add_argument("--out", type=Path, default=ROOT / "output/e12/prompts")
args = ap.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
names = sorted(ARCHETYPE_REGISTRY)
for s in args.seeds:
    rng = np.random.default_rng(s)
    rows = []
    for name in rng.choice(names, args.n, replace=False):
        ext = ARCHETYPE_REGISTRY[name]().aabb_extents()
        rows.append({"name": str(name), "prompt": "a " + str(name).replace("_", " "),
                     "extents_m": [float(x) for x in ext]})
    (args.out / f"seed_{s}.json").write_text(json.dumps(rows, indent=1))
print("prompts", len(names), "archetypes;", rows[:3])
