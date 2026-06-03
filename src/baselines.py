"""
Baseline optimizers for the Generative Suitability task.

Each baseline operates under a fixed evaluation budget (number of object
scorings) and returns the top-K candidates seen during optimization.

Baselines implemented:
    RandomSearchBaseline   - sample uniformly from the initial distribution
    FixedCADBaseline       - cycle through deterministic archetype factories
    CMAESBaseline          - covariance matrix adaptation evolution strategy
    GABaseline             - simple (mu+lambda) genetic algorithm

CAVEAT ON ENCODING PARITY (see DISCREPANCIES.md, item 4):
RandomSearch / CMAES / GA all search the SAME fixed 13-D, two-Box encoding
(`decode`, below). FixedCAD instead perturbs the full archetype factories.
"Ours" (cem.ParameterDistribution) searches a strictly richer space — 1-4
primitives drawn from 4 types (box/cyl/sphere/capsule). So the optimizer
comparison among RandomSearch/CMAES/GA is apples-to-apples, but Ours has a
larger representation, which by itself inflates the diversity metric (the shape
descriptor literally counts primitive types). Treat the Ours-vs-baseline
diversity gap as confounded by representation, not purely optimizer quality.
Set encoding="multitype" on the baselines (opt-in, default "twobox") to give the
gradient-free baselines the full primitive-type palette with counts in {1..4}.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Callable, Optional

from primitives import (
    CompositeObject, Box, Cylinder, Sphere, Capsule, Transform, seat_height,
)
from archetypes import ARCHETYPE_REGISTRY
from cem import PRIMITIVE_SPECS
from scoring import ObjectScorer, ScoringConfig


# ---------------------------------------------------------------------------
# Shared encoding
#
# Each candidate is a flat vector x in R^D with two primitives stacked:
#     [ log_dx1, log_dy1, log_dz1,
#       log_dx2, log_dy2, log_dz2,
#       ox, oy, oz,         (offset of primitive 2 relative to primitive 1)
#       rx, ry, rz,         (XYZ Euler angles for primitive 2)
#       friction ]
# This is a clean 13-D space that all gradient-free optimizers can search.
# ---------------------------------------------------------------------------

ENC_DIM = 13
ENC_LO = np.array([
    np.log(0.015), np.log(0.015), np.log(0.015),  # primitive 1 log-dims
    np.log(0.015), np.log(0.015), np.log(0.015),  # primitive 2 log-dims
    -0.05, -0.05, 0.0,                            # offset XYZ
    -np.pi / 2, -np.pi / 2, -np.pi / 2,           # rotation XYZ
    0.1,                                          # friction
])
ENC_HI = np.array([
    np.log(0.14), np.log(0.14), np.log(0.14),
    np.log(0.14), np.log(0.14), np.log(0.14),
    0.05, 0.05, 0.10,
    np.pi / 2, np.pi / 2, np.pi / 2,
    2.0,
])
ENC_MID = 0.5 * (ENC_LO + ENC_HI)
ENC_RANGE = (ENC_HI - ENC_LO)


def decode(x: np.ndarray, name: str = "candidate") -> CompositeObject:
    """Decode a flat parameter vector into a 2-box CompositeObject."""
    x = np.clip(x, ENC_LO, ENC_HI)
    dims1 = np.exp(x[0:3])
    dims2 = np.exp(x[3:6])
    offset = x[6:9].copy()
    euler = x[9:12]
    friction = float(x[12])

    base = Box(dimensions=dims1,
               transform=Transform(translation=np.array([0.0, 0.0, dims1[2] / 2])))
    offset[2] = max(dims1[2] / 2 + dims2[2] / 2, offset[2] + dims1[2] / 2)
    secondary = Box(dimensions=dims2,
                    transform=Transform.from_euler(translation=offset, euler_xyz=euler))
    return CompositeObject(primitives=[base, secondary], name=name, friction=friction)


# ---------------------------------------------------------------------------
# Optional "multitype" encoding (opt-in, for representation-parity studies).
#
# Fixed-dim vector with K primitive slots. Each slot carries a T-way type
# selector (argmax over ALL registered primitive types), three log-size params
# (the first len(spec.param_names) are used), and — for non-base slots — a
# presence gate plus an offset and rotation. So the gradient-free baselines can
# search the same type palette as Ours, with counts in {1..K}. The residual gap
# to Ours (unbounded count) is documented in DISCREPANCIES.md item 4.
# ---------------------------------------------------------------------------

_T_TYPES = len(PRIMITIVE_SPECS)
_MAX_PARAMS = 3
_LOG_S_LO, _LOG_S_HI = np.log(0.005), np.log(0.14)
_BASE_BLK = _T_TYPES + _MAX_PARAMS
_SEC_BLK = _T_TYPES + _MAX_PARAMS + 1 + 3 + 3   # +presence +offset +rotation
_MT_SLOTS = 4                                    # counts in {1..4}


def _multitype_bounds(n_slots: int):
    lo = [-1.0] * _T_TYPES + [_LOG_S_LO] * _MAX_PARAMS
    hi = [1.0] * _T_TYPES + [_LOG_S_HI] * _MAX_PARAMS
    for _ in range(n_slots - 1):
        lo += ([-1.0] * _T_TYPES + [_LOG_S_LO] * _MAX_PARAMS
               + [-1.0] + [-0.05, -0.05, 0.0] + [-np.pi / 2] * 3)
        hi += ([1.0] * _T_TYPES + [_LOG_S_HI] * _MAX_PARAMS
               + [1.0] + [0.05, 0.05, 0.10] + [np.pi / 2] * 3)
    lo += [0.1]   # friction
    hi += [2.0]
    return np.array(lo), np.array(hi)


ENC_MT_LO, ENC_MT_HI = _multitype_bounds(_MT_SLOTS)
ENC_MT_MID = 0.5 * (ENC_MT_LO + ENC_MT_HI)
ENC_MT_RANGE = ENC_MT_HI - ENC_MT_LO
ENC_MT_DIM = len(ENC_MT_LO)


def _build_typed(type_logits, size_logs, transform):
    spec = PRIMITIVE_SPECS[int(np.argmax(type_logits))]
    k = len(spec.param_names)
    params = np.clip(np.exp(size_logs[:k]), spec.clamp_lo, spec.clamp_hi)
    return spec.build(params, transform)


def decode_multitype(x: np.ndarray, name: str = "candidate") -> CompositeObject:
    """Decode the multitype vector into a 1..K-primitive CompositeObject."""
    x = np.clip(x, ENC_MT_LO, ENC_MT_HI)
    c = 0
    base = _build_typed(x[c:c + _T_TYPES], x[c + _T_TYPES:c + _BASE_BLK],
                        Transform.identity())
    c += _BASE_BLK
    base.transform.translation[2] = seat_height(base)
    primitives = [base]
    for _ in range(_MT_SLOTS - 1):
        blk = x[c:c + _SEC_BLK]
        c += _SEC_BLK
        logits = blk[:_T_TYPES]
        sizes = blk[_T_TYPES:_T_TYPES + _MAX_PARAMS]
        gate = blk[_T_TYPES + _MAX_PARAMS]
        off = blk[_T_TYPES + _MAX_PARAMS + 1:_T_TYPES + _MAX_PARAMS + 4]
        rot = blk[_T_TYPES + _MAX_PARAMS + 4:_T_TYPES + _MAX_PARAMS + 7]
        if gate > 0.0:
            sec = _build_typed(logits, sizes, Transform.from_euler(off, rot))
            sec.transform.translation = sec.transform.translation + base.transform.translation
            sec.transform.translation[2] = max(0.01, float(sec.transform.translation[2]))
            primitives.append(sec)
    friction = float(x[c])
    return CompositeObject(primitives=primitives, name=name, friction=friction)


# Encoding registry: name -> (decode_fn, lo, hi, mid, range, dim)
ENCODINGS = {
    "twobox": (decode, ENC_LO, ENC_HI, ENC_MID, ENC_RANGE, ENC_DIM),
    "multitype": (decode_multitype, ENC_MT_LO, ENC_MT_HI, ENC_MT_MID,
                  ENC_MT_RANGE, ENC_MT_DIM),
}


def sample_initial(rng: np.random.Generator, n: int,
                   lo: np.ndarray = ENC_LO, hi: np.ndarray = ENC_HI) -> np.ndarray:
    """Uniform sample of n points from the encoded box (defaults to twobox)."""
    return rng.uniform(lo, hi, size=(n, len(lo)))


# ---------------------------------------------------------------------------
# Baseline base class
# ---------------------------------------------------------------------------

@dataclass
class BaselineResult:
    """Top-K candidates plus the full evaluation trace."""
    objects: List[CompositeObject]
    scores: np.ndarray              # scores of the returned objects
    history_best: np.ndarray        # best-score-so-far per evaluation
    history_mean: np.ndarray        # mean of the most recent population per evaluation
    name: str = ""


class _Baseline:
    """Shared bookkeeping; subclasses implement _step."""

    def __init__(self, budget: int, seed: int = 42,
                 scoring_config: Optional[ScoringConfig] = None,
                 encoding: str = "twobox"):
        self.budget = budget
        self.rng = np.random.default_rng(seed)
        self.scorer = ObjectScorer(scoring_config)
        if encoding not in ENCODINGS:
            raise ValueError(f"Unknown encoding {encoding!r}; choose from {list(ENCODINGS)}")
        self.encoding = encoding
        (self._decode, self._lo, self._hi,
         self._mid, self._range, self._dim) = ENCODINGS[encoding]
        self._best_score = -np.inf
        self._history_best: List[float] = []
        self._history_mean: List[float] = []
        self._all_x: List[np.ndarray] = []
        self._all_scores: List[float] = []

    def _evaluate(self, x: np.ndarray, name: str = "cand") -> float:
        obj = self._decode(x, name=name)
        try:
            s = float(self.scorer.score(obj).total_score)
        except Exception:
            s = 0.0
        self._best_score = max(self._best_score, s)
        self._history_best.append(self._best_score)
        self._all_x.append(np.asarray(x).copy())
        self._all_scores.append(s)
        return s

    def _finalize(self, top_k: int, name: str) -> BaselineResult:
        scores = np.array(self._all_scores)
        order = np.argsort(scores)[::-1][:top_k]
        objs = [self._decode(self._all_x[i], name=f"{name}_{rank:04d}")
                for rank, i in enumerate(order)]
        return BaselineResult(
            objects=objs,
            scores=scores[order],
            history_best=np.array(self._history_best),
            history_mean=np.array(self._history_mean) if self._history_mean else np.array([]),
            name=name,
        )

    def run(self, top_k: int = 100) -> BaselineResult:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Random search
# ---------------------------------------------------------------------------

class RandomSearchBaseline(_Baseline):
    """Uniform sampling within the encoded box."""

    def run(self, top_k: int = 100) -> BaselineResult:
        X = sample_initial(self.rng, self.budget, self._lo, self._hi)
        gen_size = 50
        for start in range(0, self.budget, gen_size):
            batch = X[start:start + gen_size]
            for xi in batch:
                self._evaluate(xi)
            self._history_mean.extend([float(np.mean(self._all_scores[-len(batch):]))]
                                      * len(batch))
        return self._finalize(top_k, "random_search")


# ---------------------------------------------------------------------------
# Fixed-CAD baseline (no learning, samples from deterministic factories)
# ---------------------------------------------------------------------------

# The Fixed-CAD baseline cycles through every registered archetype (single
# source of truth — adding an archetype to ARCHETYPE_REGISTRY includes it here).
_FACTORY_FNS: List[Callable[[], CompositeObject]] = list(ARCHETYPE_REGISTRY.values())


class FixedCADBaseline(_Baseline):
    """Emulates a fixed CAD library: each candidate is one archetype with
    small Gaussian perturbations on its default parameters (no optimization)."""

    def __init__(self, budget: int, seed: int = 42,
                 scoring_config: Optional[ScoringConfig] = None,
                 jitter: float = 0.10):
        super().__init__(budget, seed, scoring_config)
        self.jitter = jitter
        # Record (object, score) here directly since decode() doesn't apply.
        self._fixed_pool: List[Tuple[CompositeObject, float]] = []

    def _perturb(self, obj: CompositeObject) -> CompositeObject:
        for p in obj.primitives:
            if isinstance(p, Box):
                p.dimensions = np.maximum(0.01, p.dimensions * (1 + self.jitter * self.rng.standard_normal(3)))
            elif isinstance(p, Cylinder):
                p.radius = max(0.005, p.radius * (1 + self.jitter * self.rng.standard_normal()))
                p.height = max(0.01, p.height * (1 + self.jitter * self.rng.standard_normal()))
            elif isinstance(p, Sphere):
                p.radius = max(0.005, p.radius * (1 + self.jitter * self.rng.standard_normal()))
            elif isinstance(p, Capsule):
                p.radius = max(0.005, p.radius * (1 + self.jitter * self.rng.standard_normal()))
                p.height = max(0.01, p.height * (1 + self.jitter * self.rng.standard_normal()))
        obj.friction = float(np.clip(0.8 + 0.2 * self.rng.standard_normal(), 0.1, 2.0))
        return obj

    def run(self, top_k: int = 100) -> BaselineResult:
        for k in range(self.budget):
            factory = _FACTORY_FNS[k % len(_FACTORY_FNS)]
            obj = factory()
            obj = self._perturb(obj)
            obj.name = f"fixed_cad_{k:04d}"
            try:
                s = float(self.scorer.score(obj).total_score)
            except Exception:
                s = 0.0
            self._best_score = max(self._best_score, s)
            self._history_best.append(self._best_score)
            self._fixed_pool.append((obj, s))
        scores = np.array([s for _, s in self._fixed_pool])
        order = np.argsort(scores)[::-1][:top_k]
        objs = [self._fixed_pool[i][0] for i in order]
        for rank, o in enumerate(objs):
            o.name = f"fixed_cad_top{rank:04d}"
        return BaselineResult(
            objects=objs,
            scores=scores[order],
            history_best=np.array(self._history_best),
            history_mean=np.array([]),
            name="fixed_cad",
        )


# ---------------------------------------------------------------------------
# CMA-ES (uses pycma)
# ---------------------------------------------------------------------------

class CMAESBaseline(_Baseline):
    """CMA-ES on the shared 13-D encoding. Maximizes score by minimizing -score."""

    def __init__(self, budget: int, seed: int = 42,
                 scoring_config: Optional[ScoringConfig] = None,
                 popsize: int = 16, sigma0: float = 0.5,
                 encoding: str = "twobox"):
        super().__init__(budget, seed, scoring_config, encoding=encoding)
        self.popsize = popsize
        self.sigma0 = sigma0

    def run(self, top_k: int = 100) -> BaselineResult:
        import cma
        # Optimization is performed in a normalized cube; rescale before decoding.
        scale = 0.5 * self._range
        offset = self._mid
        es = cma.CMAEvolutionStrategy(
            np.zeros(self._dim), self.sigma0,
            {"popsize": self.popsize, "seed": int(self.rng.integers(1, 2**31)),
             "bounds": [[-1.0] * self._dim, [1.0] * self._dim],
             "verbose": -9},
        )
        n_evals = 0
        while n_evals < self.budget and not es.stop():
            asks = es.ask()
            losses = []
            pop_scores = []
            for u in asks:
                if n_evals >= self.budget:
                    losses.append(1.0)
                    continue
                x = np.asarray(u) * scale + offset
                s = self._evaluate(x, name=f"cmaes_{n_evals:04d}")
                pop_scores.append(s)
                losses.append(-s)
                n_evals += 1
            if pop_scores:
                self._history_mean.extend([float(np.mean(pop_scores))] * len(pop_scores))
            es.tell(asks, losses)
        return self._finalize(top_k, "cmaes")


# ---------------------------------------------------------------------------
# Genetic algorithm (mu + lambda)
# ---------------------------------------------------------------------------

class GABaseline(_Baseline):
    """Simple (mu+lambda) GA with tournament selection, BLX-alpha crossover,
    and Gaussian mutation in the encoded space."""

    def __init__(self, budget: int, seed: int = 42,
                 scoring_config: Optional[ScoringConfig] = None,
                 popsize: int = 30, mutation_std: float = 0.10,
                 tournament_k: int = 3, crossover_p: float = 0.8,
                 encoding: str = "twobox"):
        super().__init__(budget, seed, scoring_config, encoding=encoding)
        self.popsize = popsize
        self.mutation_std = mutation_std
        self.tournament_k = tournament_k
        self.crossover_p = crossover_p

    def _tournament(self, pop_x, pop_s):
        idx = self.rng.integers(0, len(pop_x), size=self.tournament_k)
        winner = idx[np.argmax(pop_s[idx])]
        return pop_x[winner].copy()

    def _crossover(self, a, b):
        if self.rng.random() > self.crossover_p:
            return a.copy()
        alpha = 0.5
        lo = np.minimum(a, b) - alpha * np.abs(a - b)
        hi = np.maximum(a, b) + alpha * np.abs(a - b)
        return self.rng.uniform(lo, hi)

    def _mutate(self, x):
        noise = self.rng.standard_normal(self._dim) * self.mutation_std * self._range
        return np.clip(x + noise, self._lo, self._hi)

    def run(self, top_k: int = 100) -> BaselineResult:
        pop_x = sample_initial(self.rng, self.popsize, self._lo, self._hi)
        pop_s = np.array([self._evaluate(x, name=f"ga_{i:04d}") for i, x in enumerate(pop_x)])
        self._history_mean.extend([float(pop_s.mean())] * self.popsize)
        n_evals = self.popsize
        gen = 0
        while n_evals < self.budget:
            offspring = []
            for _ in range(self.popsize):
                a = self._tournament(pop_x, pop_s)
                b = self._tournament(pop_x, pop_s)
                child = self._mutate(self._crossover(a, b))
                offspring.append(child)
            offspring = np.array(offspring)
            off_scores = []
            for i, x in enumerate(offspring):
                if n_evals >= self.budget:
                    break
                off_scores.append(self._evaluate(x, name=f"ga_g{gen}_{i:03d}"))
                n_evals += 1
            off_scores = np.array(off_scores)
            offspring = offspring[: len(off_scores)]
            self._history_mean.extend([float(off_scores.mean() if len(off_scores) else 0.0)]
                                      * len(off_scores))
            # mu+lambda: keep best popsize among parents + offspring
            combined_x = np.vstack([pop_x, offspring])
            combined_s = np.concatenate([pop_s, off_scores])
            order = np.argsort(combined_s)[::-1][:self.popsize]
            pop_x = combined_x[order]
            pop_s = combined_s[order]
            gen += 1
        return self._finalize(top_k, "ga")


# ---------------------------------------------------------------------------
# Registry helper
# ---------------------------------------------------------------------------

BASELINES = {
    "random_search": RandomSearchBaseline,
    "fixed_cad": FixedCADBaseline,
    "cmaes": CMAESBaseline,
    "ga": GABaseline,
}


def run_baseline(name: str, budget: int, seed: int = 42,
                 top_k: int = 100, **kwargs) -> BaselineResult:
    if name not in BASELINES:
        raise ValueError(f"Unknown baseline: {name}. Choose from {list(BASELINES)}")
    return BASELINES[name](budget=budget, seed=seed, **kwargs).run(top_k=top_k)
