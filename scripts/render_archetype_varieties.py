#!/usr/bin/env python3
"""
Render CEM-trained parameter VARIETIES of representative archetypes (headless).
For each archetype, the per-archetype CEM (ArchetypeTrainer) tunes its factory
parameters, then we sample k variations -> docs/gallery/varieties_<name>.png, and
an optional combined grid (rows = archetypes) for the website.

Run from 3d_generator/:
    python scripts/render_archetype_varieties.py --subset --combined --copy-web
    python scripts/render_archetype_varieties.py --archetypes cup,jar --k 12
"""
import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from archetypes import ARCHETYPE_REGISTRY                     # noqa: E402
from archetype_cem import ArchetypeTrainer                    # noqa: E402
from scoring import ObjectScorer                              # noqa: E402
from _render_common import grid, verdict_color                # noqa: E402

OUT = ROOT / "docs" / "gallery"
WEB = ROOT / "docs" / "static" / "images"
SCORER = ObjectScorer()
BLUE = verdict_color("optimized")

# One per group + the new-primitive containers.
SUBSET = ["mallet", "cup", "mug_like", "teapot", "bowl", "pot",
          "jar", "nut", "kettlebell", "mouse"]


def varieties(name, k, iters, samples, train):
    """Return (cells, trainer) — k variety meshes of archetype `name`."""
    factory = ARCHETYPE_REGISTRY[name]
    tr = ArchetypeTrainer(factory)
    if train:
        tr.train(iterations=iters, samples_per_iter=samples)
    cells = []
    tries = 0
    while len(cells) < k and tries < k * 4:
        tries += 1
        params = tr.dist.sample(tr.rng)
        try:
            obj = factory(**params)
            mesh = obj.to_mesh(boolean_union=True)
            s = SCORER.score(obj).total_score
        except Exception:
            continue
        cells.append((f"{name} s={s:.2f}", mesh, BLUE))
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archetypes", default="")
    ap.add_argument("--subset", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--samples", type=int, default=50)
    ap.add_argument("--no-train", action="store_true",
                    help="raw sampled varieties instead of CEM-trained")
    ap.add_argument("--combined", action="store_true")
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--copy-web", action="store_true")
    args = ap.parse_args()

    if args.all:
        names = list(ARCHETYPE_REGISTRY)
    elif args.archetypes:
        names = [n.strip() for n in args.archetypes.split(",") if n.strip()]
    else:
        names = SUBSET
    names = [n for n in names if n in ARCHETYPE_REGISTRY]

    combined = []
    per_arch_k = 4 if args.combined else args.k
    for name in names:
        k = max(args.k, per_arch_k)
        cells = varieties(name, k, args.iters, args.samples, not args.no_train)
        if not cells:
            print(f"skip {name}: no valid varieties")
            continue
        if not args.combined:
            path = OUT / f"varieties_{name}.png"
            grid(cells, path, cols=min(args.cols, len(cells)),
                 title=f"{name} — {len(cells)} CEM-trained varieties")
            if args.copy_web:
                WEB.mkdir(parents=True, exist_ok=True)
                shutil.copy(path, WEB / path.name)
        combined.extend(cells[:per_arch_k])

    if args.combined and combined:
        path = OUT / "varieties_combined.png"
        grid(combined, path, cols=per_arch_k,
             title="Archetype varieties (CEM-trained) — rows are archetypes")
        if args.copy_web:
            WEB.mkdir(parents=True, exist_ok=True)
            shutil.copy(path, WEB / path.name)


if __name__ == "__main__":
    main()
