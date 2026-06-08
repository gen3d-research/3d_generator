#!/usr/bin/env python3
"""
Strategy-based generation galleries (headless). Explore what the 14-primitive
generator produces under systematic CONSTRAINTS, each rendered to
docs/gallery/strategy_<name>.png:

  default     mixed trained/untrained batch (the baseline)
  curved      only curved-surface primitive types
  faceted     only flat/faceted types (box, pyramid, wedge, hex_prism)
  single      14 types x N=2..10 copies each (one union per cell)  [+ symmetric on prime N]
  pairs       every unordered type-PAIR (one of each)              [+ symmetric]
  oneofeach   one union of all 14 types (a single connected body)
  symmetric   a mixed bilaterally-symmetric batch

A symmetric object is built by mirroring parts across the YZ (x->-x) plane around
a central straddling part (default placements use identity rotation, so the
mirror is exact). Constrained palettes are SAMPLED, not retrained (CEM training
re-introduces excluded types via epsilon-smoothing).

Run from 3d_generator/:
    python scripts/gen_strategies.py --strategy all
    python scripts/gen_strategies.py --strategy single
    python scripts/gen_strategies.py --strategy pairs --copy-web
"""
import argparse
import copy
import itertools
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cem import PRIMITIVE_SPECS, ParameterDistribution        # noqa: E402
from primitives import CompositeObject, half_extents, Transform  # noqa: E402
from scoring import ObjectScorer                              # noqa: E402
from generator import RoboticObjectGenerator, GeneratorConfig  # noqa: E402
from _render_common import grid, verdict_color                # noqa: E402

OUT = ROOT / "docs" / "gallery"
WEB = ROOT / "docs" / "static" / "images"

TYPE_INDEX = {s.key: i for i, s in enumerate(PRIMITIVE_SPECS)}
TYPES = [s.key for s in PRIMITIVE_SPECS]                       # spec order
CURVED = {"cylinder", "sphere", "capsule", "cone", "torus", "ellipsoid",
          "hollow_shell", "handle", "frustum", "hemisphere"}
FACETED = {"box", "pyramid", "wedge", "hex_prism"}
SCORER = ObjectScorer()
DIST = ParameterDistribution(max_primitives=16)               # for _attach + size priors
COL = {"accepted": verdict_color("accepted"), "opt": verdict_color("optimized")}


def is_prime(n):
    return n >= 2 and all(n % i for i in range(2, int(n ** 0.5) + 1))


def _mask_probs(keys):
    v = np.zeros(len(PRIMITIVE_SPECS))
    for k in keys:
        v[TYPE_INDEX[k]] = 1.0
    return v / v.sum()


def _build_prim(type_key, rng):
    """One primitive of type_key at a sampled size, identity transform."""
    spec = PRIMITIVE_SPECS[TYPE_INDEX[type_key]]
    log_p = rng.normal(DIST.type_log_means[spec.key], DIST.type_log_stds[spec.key])
    params = np.clip(np.exp(log_p), spec.clamp_lo, spec.clamp_hi)
    return spec.build(params, Transform.identity())


def _reseat(parts):
    low = min(float(p.transform.translation[2]) - float(half_extents(p)[2]) for p in parts)
    for p in parts:
        p.transform.translation[2] -= low
    return parts


def _assemble(prims, rng):
    """Seat prims[0], attach the rest to a random existing part, re-seat on z=0
    (mirrors ParameterDistribution.sample_object's structured placement)."""
    prims = list(prims)
    prims[0].transform.translation = np.array([0.0, 0.0, 0.0])
    placed = [prims[0]]
    for p in prims[1:]:
        anchor = placed[int(rng.integers(0, len(placed)))]
        DIST._attach(p, anchor, rng)
        placed.append(p)
    return CompositeObject(primitives=_reseat(placed), name="strat")


def _attach_px(new, anchor):
    """Seat *new* against the +x face of *anchor* with overlap (keeps it at x>0)."""
    ha, hn = half_extents(anchor), half_extents(new)
    d = ha[0] + hn[0] - DIST.attach_overlap * min(ha[0], hn[0])
    c = np.asarray(anchor.transform.translation, float).copy()
    c[0] += d
    new.transform.translation = c


def _mirror(prim):
    m = copy.deepcopy(prim)
    t = np.asarray(m.transform.translation, float).copy()
    t[0] = -t[0]
    m.transform.translation = t
    return m


def build_connected(prim_keys, rng, tries=14):
    """Union of len(prim_keys) primitives (given types), retried until connected."""
    obj = None
    for _ in range(tries):
        obj = _assemble([_build_prim(k, rng) for k in prim_keys], rng)
        if obj.is_connected():
            return obj
    return obj


def build_symmetric(prim_keys, rng, tries=14):
    """Bilaterally-symmetric union: a central straddling part + a +x chain of
    parts and their YZ mirrors. ``prim_keys[0]`` is the centre; the rest form the
    right half (mirrored to the left). Total parts = 1 + 2*(len-1)."""
    obj = None
    for _ in range(tries):
        center = _build_prim(prim_keys[0], rng)
        center.transform.translation = np.array([0.0, 0.0, 0.0])
        right = []
        for k in prim_keys[1:]:
            p = _build_prim(k, rng)
            _attach_px(p, right[-1] if right else center)
            right.append(p)
        parts = [center] + right + [_mirror(p) for p in right]
        obj = CompositeObject(primitives=_reseat(parts), name="sym")
        if obj.is_connected():
            return obj
    return obj


def _label(obj, extra=""):
    try:
        s = SCORER.score(obj).total_score
        return f"{extra}{len(obj.primitives)}p s={s:.2f}"
    except Exception:
        return f"{extra}{len(obj.primitives)}p"


def _render(cells, name, title, cols, copy_web):
    path = OUT / f"strategy_{name}.png"
    grid(cells, path, cols=cols, title=title)
    if copy_web:
        WEB.mkdir(parents=True, exist_ok=True)
        shutil.copy(path, WEB / path.name)


# --------------------------------------------------------------------------- #
def strat_palette(name, keys, n, seed, cols, copy_web):
    """A quality-filtered batch restricted to a subset of types (curved/faceted)."""
    gen = RoboticObjectGenerator(GeneratorConfig(seed=seed, max_primitives=8))
    gen.distribution.primitive_type_probs = _mask_probs(keys)
    objs = gen.generate(n, ensure_quality=True)
    cells = [(_label(o), o.to_mesh(boolean_union=True), COL["accepted"]) for o in objs]
    _render(cells, name, f"{name} primitives only ({len(objs)} sampled)", cols, copy_web)


def strat_default(n, seed, cols, train, copy_web):
    gen = RoboticObjectGenerator(GeneratorConfig(seed=seed, max_primitives=8))
    if train:
        gen.train(verbose=False)
    objs = gen.generate(n, ensure_quality=True)
    col = COL["opt"] if train else COL["accepted"]
    cells = [(_label(o), o.to_mesh(boolean_union=True), col) for o in objs]
    _render(cells, "default", f"default batch ({'trained' if train else 'untrained'})",
            cols, copy_web)


def strat_single(seed, copy_web, n_lo=2, n_hi=10):
    rng = np.random.default_rng(seed)
    Ns = list(range(n_lo, n_hi + 1))
    cells, sym_cells = [], []
    primes = [n for n in Ns if is_prime(n)]
    for t in TYPES:
        for N in Ns:
            o = build_connected([t] * N, rng)
            cells.append((_label(o, f"{t[:4]}x{N} "),
                          o.to_mesh(boolean_union=True), COL["accepted"]))
        for N in primes:
            o = build_symmetric([t] * (1 + (N - 1) // 2 if N > 2 else 2), rng)
            sym_cells.append((_label(o, f"{t[:4]}x{N}sym "),
                              o.to_mesh(boolean_union=True), COL["opt"]))
    _render(cells, "single_sweep",
            f"Single-type unions — 14 types x N={n_lo}..{n_hi}", len(Ns), copy_web)
    _render(sym_cells, "single_sweep_sym",
            f"Single-type SYMMETRIC — prime N {primes}", len(primes), copy_web)


def strat_pairs(seed, copy_web):
    rng = np.random.default_rng(seed)
    cells, sym = [], []
    for a, b in itertools.combinations(TYPES, 2):                 # 91 pairs
        o = build_connected([a, b], rng)
        cells.append((f"{a[:4]}+{b[:4]}", o.to_mesh(boolean_union=True), COL["accepted"]))
        so = build_symmetric([a, b], rng)                         # A center + B,B' mirror
        sym.append((f"{a[:4]}+{b[:4]}", so.to_mesh(boolean_union=True), COL["opt"]))
    _render(cells, "pairs", "All type-pairs (one of each) — 91 combinations", 13, copy_web)
    _render(sym, "pairs_sym", "Type-pairs, bilaterally SYMMETRIC", 13, copy_web)


def strat_oneofeach(seed, copy_web, variants=4):
    cells = []
    for v in range(variants):
        rng = np.random.default_rng(seed + v)
        o = build_connected(list(TYPES), rng, tries=30)
        cells.append((_label(o, "all14 "), o.to_mesh(boolean_union=True), COL["opt"]))
    _render(cells, "oneofeach", "One of each of the 14 types, unioned", variants, copy_web)


def strat_symmetric(n, seed, cols, copy_web):
    rng = np.random.default_rng(seed)
    cells = []
    for i in range(n):
        keys = [rng.choice(TYPES) for _ in range(int(rng.integers(2, 5)))]
        o = build_symmetric(keys, rng)
        cells.append((_label(o, "sym "), o.to_mesh(boolean_union=True), COL["opt"]))
    _render(cells, "symmetric", f"Bilaterally-symmetric mixed batch ({n})", cols, copy_web)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="all")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--copy-web", action="store_true")
    args = ap.parse_args()
    s = args.strategy

    def want(name):
        return s == "all" or s == name

    if want("default"):
        strat_default(args.n, args.seed, args.cols, args.train, args.copy_web)
    if want("curved"):
        strat_palette("curved", CURVED, args.n, args.seed, args.cols, args.copy_web)
    if want("faceted"):
        strat_palette("faceted", FACETED, args.n, args.seed, args.cols, args.copy_web)
    if want("single"):
        strat_single(args.seed, args.copy_web)
    if want("pairs"):
        strat_pairs(args.seed, args.copy_web)
    if want("oneofeach"):
        strat_oneofeach(args.seed, args.copy_web)
    if want("symmetric"):
        strat_symmetric(args.n, args.seed, args.cols, args.copy_web)


if __name__ == "__main__":
    main()
