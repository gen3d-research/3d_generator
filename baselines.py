"""
Baseline optimizers for the Generative Suitability task.

Each baseline operates under a fixed evaluation budget (number of object
scorings) and returns the top-K candidates seen during optimization.

Baselines implemented:
    RandomSearchBaseline   - sample uniformly from the initial distribution
    FixedCADBaseline       - cycle through deterministic archetype factories
    CMAESBaseline          - covariance matrix adaptation evolution strategy
    GABaseline             - simple (mu+lambda) genetic algorithm

All baselines share the same parameter encoding used by ParameterDistribution
so the comparison is apples-to-apples (R2-Q4, AE).
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Callable, Optional

from primitives import (
    CompositeObject, Box, Cylinder, Sphere, Capsule, Transform,
    create_simple_box, create_mug_like, create_l_shape,
    create_dumbbell, create_hammer, create_bottle,
    create_t_shape, create_u_shape, create_v_shape, create_monitor,
    create_barbell, create_snowman, create_camera, create_frying_pan,
    create_flashlight, create_spatula, create_remote, create_joystick,
)
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


def sample_initial(rng: np.random.Generator, n: int) -> np.ndarray:
    """Uniform sample of n points from the encoded box."""
    return rng.uniform(ENC_LO, ENC_HI, size=(n, ENC_DIM))


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
                 scoring_config: Optional[ScoringConfig] = None):
        self.budget = budget
        self.rng = np.random.default_rng(seed)
        self.scorer = ObjectScorer(scoring_config)
        self._best_score = -np.inf
        self._history_best: List[float] = []
        self._history_mean: List[float] = []
        self._all_x: List[np.ndarray] = []
        self._all_scores: List[float] = []

    def _evaluate(self, x: np.ndarray, name: str = "cand") -> float:
        obj = decode(x, name=name)
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
        objs = [decode(self._all_x[i], name=f"{name}_{rank:04d}")
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
        X = sample_initial(self.rng, self.budget)
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

_FACTORY_FNS: List[Callable[[], CompositeObject]] = [
    create_simple_box.__wrapped__ if hasattr(create_simple_box, "__wrapped__") else None,
]
# Build a list of zero-arg-callable factories that yield a CompositeObject.
def _wrap_simple_box() -> CompositeObject:
    return create_simple_box(np.array([0.05, 0.05, 0.06]))

_FACTORY_FNS = [
    _wrap_simple_box, create_mug_like, create_l_shape,
    create_dumbbell, create_hammer, create_bottle,
    create_t_shape, create_u_shape, create_v_shape, create_monitor,
    create_barbell, create_snowman, create_camera, create_frying_pan,
    create_flashlight, create_spatula, create_remote, create_joystick,
]


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
                 popsize: int = 16, sigma0: float = 0.5):
        super().__init__(budget, seed, scoring_config)
        self.popsize = popsize
        self.sigma0 = sigma0

    def run(self, top_k: int = 100) -> BaselineResult:
        import cma
        x0 = ENC_MID.copy()
        # Optimization is performed in a normalized cube; rescale before decoding.
        scale = 0.5 * ENC_RANGE
        offset = ENC_MID
        es = cma.CMAEvolutionStrategy(
            np.zeros(ENC_DIM), self.sigma0,
            {"popsize": self.popsize, "seed": int(self.rng.integers(1, 2**31)),
             "bounds": [(-1.0).repeat(ENC_DIM).tolist() if False else [-1.0] * ENC_DIM,
                        [1.0] * ENC_DIM],
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
                 tournament_k: int = 3, crossover_p: float = 0.8):
        super().__init__(budget, seed, scoring_config)
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
        noise = self.rng.standard_normal(ENC_DIM) * self.mutation_std * ENC_RANGE
        return np.clip(x + noise, ENC_LO, ENC_HI)

    def run(self, top_k: int = 100) -> BaselineResult:
        pop_x = sample_initial(self.rng, self.popsize)
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
