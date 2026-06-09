#!/usr/bin/env python3
"""
Render a VARIANT showcase for the archetype library: for every archetype, a grid of
parameter variants (the same per-archetype distribution build_archetype_manifest.py
samples), plus paginated "index" contact sheets covering all archetypes at a glance.

Outputs (headless):
  docs/gallery/variants/<name>.png        # per-archetype grid of K variants
  docs/gallery/variants_index_<NN>.png    # all archetypes, 1 variant each, paginated
  docs/static/images/variants_index_<NN>.png   # copies for the website

Run from 3d_generator/:
    python scripts/render_variant_gallery.py            # all 105 archetypes, K=9 each
    python scripts/render_variant_gallery.py --k 12 --archetypes mug_like,wine_bottle
"""
import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from archetypes import ARCHETYPE_REGISTRY                    # noqa: E402
from archetype_cem import ArchetypeDistribution              # noqa: E402
from _render_common import grid, verdict_color               # noqa: E402

OUT = ROOT / "docs" / "gallery"
VAR = OUT / "variants"
WEB = ROOT / "docs" / "static" / "images"
BLUE = verdict_color("optimized")
GREEN = verdict_color("accepted")


def variants_of(name, k, rng_seed, tries_mult=4):
    """Return up to k (label, mesh) variant cells for one archetype."""
    factory = ARCHETYPE_REGISTRY[name]
    dist = ArchetypeDistribution(factory)
    rng = np.random.default_rng(rng_seed)
    cells, tries = [], 0
    while len(cells) < k and tries < k * tries_mult:
        tries += 1
        try:
            obj = factory(**dist.sample(rng))
            if not obj.is_connected():
                continue
            cells.append((f"{name}", obj.to_mesh(boolean_union=True)))
        except Exception:
            continue
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archetypes", default="", help="comma list (default: all)")
    ap.add_argument("--k", type=int, default=9, help="variants per archetype grid")
    ap.add_argument("--cols", type=int, default=3, help="cols in a per-archetype grid")
    ap.add_argument("--per-page", type=int, default=30, help="archetypes per index page")
    ap.add_argument("--index-cols", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-web", action="store_true")
    args = ap.parse_args()

    names = ([n.strip() for n in args.archetypes.split(",") if n.strip()]
             or list(ARCHETYPE_REGISTRY))
    names = [n for n in names if n in ARCHETYPE_REGISTRY]
    VAR.mkdir(parents=True, exist_ok=True)
    WEB.mkdir(parents=True, exist_ok=True)

    index_cells = []
    for i, name in enumerate(names):
        cells = variants_of(name, args.k, args.seed + i)
        if not cells:
            print(f"skip {name}: no valid variants")
            continue
        grid([(lbl, m, BLUE) for lbl, m in cells], VAR / f"{name}.png",
             cols=min(args.cols, len(cells)),
             title=f"{name} — {len(cells)} variants")
        index_cells.append((name, cells[0][1], GREEN))   # 1 representative for the index
    print(f"wrote {len(index_cells)} per-archetype grids to {VAR}")

    # Paginated index sheets (all archetypes, one variant each)
    pages = [index_cells[p:p + args.per_page]
             for p in range(0, len(index_cells), args.per_page)]
    for pi, page in enumerate(pages, 1):
        path = OUT / f"variants_index_{pi:02d}.png"
        grid(page, path, cols=args.index_cols,
             title=f"Archetype variant library ({pi}/{len(pages)}) — "
                   f"{len(index_cells)} archetypes × ~100 variants each")
        if not args.no_web:
            shutil.copy(path, WEB / path.name)
    print(f"wrote {len(pages)} index pages")


if __name__ == "__main__":
    main()
