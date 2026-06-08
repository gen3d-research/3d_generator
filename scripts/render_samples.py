#!/usr/bin/env python3
"""
Render generated-sample galleries (headless matplotlib 3D):

  * docs/gallery/samples_verdicts.png  — RAW samples from an UNTRAINED
    distribution, each tagged accepted (green) / rejected (red, with reason).
    Shows how often random primitive assemblies fail the quality gate — i.e.
    how the limited primitive set hurts generation quality.

  * docs/gallery/samples_optimized.png — objects from a TRAINED CEM generator
    (the "optimized" set that survives scoring + grasp re-ranking).

Both are copied into docs/static/images/ for the website.

Run from 3d_generator/:  python scripts/render_samples.py
"""
import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from generator import RoboticObjectGenerator, GeneratorConfig    # noqa: E402
from _render_common import grid, verdict_color                   # noqa: E402

OUT = ROOT / "docs" / "gallery"
WEB = ROOT / "docs" / "static" / "images"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-verdicts", type=int, default=24)
    ap.add_argument("--n-optimized", type=int, default=16)
    ap.add_argument("--iterations", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    WEB.mkdir(parents=True, exist_ok=True)

    # 1) Raw accepted/rejected verdicts from an UNTRAINED distribution.
    print("sampling untrained distribution for accept/reject verdicts ...")
    gen = RoboticObjectGenerator(GeneratorConfig(seed=args.seed, max_primitives=8))
    cells = []
    n_acc = 0
    for obj, verdict in gen.sample_with_verdicts(args.n_verdicts):
        n_acc += verdict == "accepted"
        label = f"{len(obj.primitives)}p  {verdict}"
        try:
            mesh = obj.to_mesh(boolean_union=True)
        except Exception as e:                                   # pragma: no cover
            print(f"skip sample: {e}"); continue
        cells.append((label, mesh, verdict_color(verdict)))
    grid(cells, OUT / "samples_verdicts.png", cols=6,
         title=f"Raw samples — quality gate: {n_acc}/{len(cells)} accepted "
               f"(green) vs rejected (red)")
    shutil.copy(OUT / "samples_verdicts.png", WEB / "samples_verdicts.png")

    # 2) Trained-CEM "optimized" set.
    print("training CEM generator for the optimized set ...")
    gen2 = RoboticObjectGenerator(GeneratorConfig(
        seed=args.seed, max_primitives=8,
        cem_iterations=args.iterations, cem_samples=60))
    gen2.train(verbose=False)
    objs = gen2.generate(args.n_optimized)
    opt = [(f"{len(o.primitives)}p", o.to_mesh(boolean_union=True), verdict_color("optimized"))
           for o in objs]
    grid(opt, OUT / "samples_optimized.png", cols=min(8, args.n_optimized),
         title="CEM-optimized samples (trained, score + grasp re-ranked)")
    shutil.copy(OUT / "samples_optimized.png", WEB / "samples_optimized.png")
    print(f"copied galleries -> {WEB}")


if __name__ == "__main__":
    main()
