"""
Cross-Entropy Method (CEM) for Lightweight Generative Learning.

Adapts the object generation distribution toward high-scoring objects
without requiring any external dataset. This is the "learning" component.

The CEM iteratively:
1. Samples candidates from current distribution
2. Evaluates each candidate
3. Selects top-k% (elite) samples
4. Updates distribution parameters to maximize likelihood of elites

v2: the distribution is driven by a primitive SPEC TABLE (PRIMITIVE_SPECS), so
the set of primitive types and the maximum primitive count are configurable
rather than hardcoded. See plan-to-make-all-* / DISCREPANCIES.md.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass, field
import json
from pathlib import Path

from primitives import (
    CompositeObject, Primitive, Box, Cylinder, Sphere, Capsule,
    Cone, Pyramid, Torus, Ellipsoid, Wedge, HollowShell, Handle,
    Transform, PrimitiveType, seat_height, half_extents,
)
from scoring import ObjectScorer, ScoreBreakdown, ScoringConfig


# ---------------------------------------------------------------------------
# Primitive spec table — single source of truth for the CEM's per-type
# parameterization. Adding a new primitive type is a one-row addition here
# (plus the Primitive subclass in primitives.py). All sampling / updating /
# (de)serialization iterates this table, so nothing downstream hardcodes the
# set of types or their parameter counts.
#
# All size params are sampled in LOG space (so they stay positive) and clamped
# to [clamp_lo, clamp_hi] (linear, meters) after exponentiation.
# ---------------------------------------------------------------------------

@dataclass
class PrimitiveSpec:
    ptype: PrimitiveType
    param_names: List[str]
    init_log_mean: np.ndarray
    init_std: np.ndarray
    clamp_lo: np.ndarray
    clamp_hi: np.ndarray
    build: Callable                 # (linear_params: np.ndarray, Transform) -> Primitive
    extract: Callable               # (Primitive) -> linear_params: np.ndarray

    @property
    def key(self) -> str:
        return self.ptype.value


def _log(*xs) -> np.ndarray:
    return np.log(np.array(xs, dtype=float))


PRIMITIVE_SPECS: List[PrimitiveSpec] = [
    PrimitiveSpec(PrimitiveType.BOX, ['dx', 'dy', 'dz'],
                  _log(0.05, 0.05, 0.06), np.array([0.4, 0.4, 0.4]),
                  np.array([0.01, 0.01, 0.01]), np.array([0.15, 0.15, 0.15]),
                  build=lambda p, t: Box(dimensions=p, transform=t),
                  extract=lambda pr: np.asarray(pr.dimensions, dtype=float)),
    PrimitiveSpec(PrimitiveType.CYLINDER, ['radius', 'height'],
                  _log(0.025, 0.06), np.array([0.3, 0.4]),
                  np.array([0.005, 0.01]), np.array([0.08, 0.15]),
                  build=lambda p, t: Cylinder(radius=p[0], height=p[1], transform=t),
                  extract=lambda pr: np.array([pr.radius, pr.height])),
    PrimitiveSpec(PrimitiveType.SPHERE, ['radius'],
                  _log(0.03), np.array([0.3]),
                  np.array([0.01]), np.array([0.08]),
                  build=lambda p, t: Sphere(radius=p[0], transform=t),
                  extract=lambda pr: np.array([pr.radius])),
    PrimitiveSpec(PrimitiveType.CAPSULE, ['radius', 'height'],
                  _log(0.015, 0.04), np.array([0.3, 0.4]),
                  np.array([0.005, 0.01]), np.array([0.05, 0.12]),
                  build=lambda p, t: Capsule(radius=p[0], height=p[1], transform=t),
                  extract=lambda pr: np.array([pr.radius, pr.height])),
    PrimitiveSpec(PrimitiveType.CONE, ['radius', 'height'],
                  _log(0.025, 0.06), np.array([0.3, 0.4]),
                  np.array([0.008, 0.02]), np.array([0.07, 0.14]),
                  build=lambda p, t: Cone(radius=p[0], height=p[1], transform=t),
                  extract=lambda pr: np.array([pr.radius, pr.height])),
    PrimitiveSpec(PrimitiveType.PYRAMID, ['radius', 'height'],
                  _log(0.03, 0.05), np.array([0.3, 0.4]),
                  np.array([0.01, 0.02]), np.array([0.08, 0.14]),
                  build=lambda p, t: Pyramid(radius=p[0], height=p[1], transform=t),
                  extract=lambda pr: np.array([pr.radius, pr.height])),
    PrimitiveSpec(PrimitiveType.TORUS, ['major', 'minor'],
                  _log(0.04, 0.012), np.array([0.3, 0.3]),
                  np.array([0.025, 0.005]), np.array([0.08, 0.02]),
                  build=lambda p, t: Torus(major_radius=p[0], minor_radius=p[1], transform=t),
                  extract=lambda pr: np.array([pr.major_radius, pr.minor_radius])),
    PrimitiveSpec(PrimitiveType.ELLIPSOID, ['rx', 'ry', 'rz'],
                  _log(0.04, 0.03, 0.02), np.array([0.3, 0.3, 0.3]),
                  np.array([0.01, 0.01, 0.01]), np.array([0.08, 0.08, 0.08]),
                  build=lambda p, t: Ellipsoid(radii=p, transform=t),
                  extract=lambda pr: np.asarray(pr.radii, dtype=float)),
    PrimitiveSpec(PrimitiveType.WEDGE, ['width', 'depth', 'height'],
                  _log(0.05, 0.04, 0.04), np.array([0.3, 0.3, 0.3]),
                  np.array([0.015, 0.015, 0.015]), np.array([0.14, 0.14, 0.14]),
                  build=lambda p, t: Wedge(width=p[0], depth=p[1], height=p[2], transform=t),
                  extract=lambda pr: np.array([pr.width, pr.depth, pr.height])),
    PrimitiveSpec(PrimitiveType.HOLLOW_SHELL, ['outer', 'wall', 'height', 'floor'],
                  _log(0.035, 0.004, 0.07, 0.005), np.array([0.3, 0.25, 0.3, 0.25]),
                  np.array([0.012, 0.002, 0.02, 0.002]), np.array([0.06, 0.01, 0.14, 0.012]),
                  build=lambda p, t: HollowShell(outer_radius=p[0], wall_thickness=p[1],
                                                 height=p[2], floor_thickness=p[3], transform=t),
                  extract=lambda pr: np.array([pr.outer_radius, pr.wall_thickness,
                                               pr.height, pr.floor_thickness])),
    PrimitiveSpec(PrimitiveType.HANDLE, ['major', 'tube_a', 'tube_b', 'arc'],
                  _log(0.02, 0.006, 0.005, 1.5 * np.pi), np.array([0.3, 0.25, 0.25, 0.2]),
                  np.array([0.01, 0.003, 0.003, 0.6 * np.pi]),
                  np.array([0.05, 0.012, 0.012, 1.9 * np.pi]),
                  build=lambda p, t: Handle(major_radius=p[0], tube_a=p[1], tube_b=p[2],
                                            arc_angle=p[3], transform=t),
                  extract=lambda pr: np.array([pr.major_radius, pr.tube_a, pr.tube_b,
                                               pr.arc_angle])),
]

# Default type bias: favor box/cylinder slightly, rest uniform.
_DEFAULT_TYPE_WEIGHTS = np.ones(len(PRIMITIVE_SPECS))
_DEFAULT_TYPE_WEIGHTS[0] = 2.0   # box
_DEFAULT_TYPE_WEIGHTS[1] = 1.5   # cylinder
_SPEC_INDEX = {s.ptype: i for i, s in enumerate(PRIMITIVE_SPECS)}

# Floors so learned spread never collapses to zero.
_OFFSET_STD_FLOOR = 0.005
_ROT_STD_FLOOR = 0.02


def _default_n_primitives_probs(max_primitives: int) -> np.ndarray:
    """Soft prior over primitive count favoring small objects, length = cap."""
    head = np.array([0.3, 0.4, 0.2, 0.1])
    if max_primitives <= 4:
        w = head[:max_primitives]
    else:
        w = np.concatenate([head, np.full(max_primitives - 4, 0.02)])
    return w / w.sum()


@dataclass
class ParameterDistribution:
    """
    Factorized distribution over composite-object parameters, driven by
    PRIMITIVE_SPECS so the type set and primitive-count cap are configurable.

    Size parameters per type are stored in LOG space (positivity). Placement
    offsets/rotations are linear-space Gaussians. JSON-serializable.
    """
    max_primitives: int = 16
    n_primitives_probs: Optional[np.ndarray] = None      # length == max_primitives
    primitive_type_probs: Optional[np.ndarray] = None    # length == len(PRIMITIVE_SPECS)
    type_log_means: Optional[Dict[str, np.ndarray]] = None
    type_log_stds: Optional[Dict[str, np.ndarray]] = None
    offset_std: float = 0.03
    rotation_std: float = 0.3
    friction_mean: float = 0.8
    friction_std: float = 0.2
    # v2 default: place each non-base primitive against a FACE of an existing
    # one, axis-aligned (structured, "assembled"-looking objects). Set False for
    # the v1 free random-offset placement.
    structured_placement: bool = True
    # Overlap fraction (of the smaller half-extent) when face-attaching, so the
    # contact is a solid intersection rather than a tangential touch. Generous
    # enough to stay robust to the approximate half_extents of asymmetric
    # primitives (cone/pyramid/wedge).
    attach_overlap: float = 0.4

    def __post_init__(self):
        if self.n_primitives_probs is None:
            self.n_primitives_probs = _default_n_primitives_probs(self.max_primitives)
        else:
            self.n_primitives_probs = np.asarray(self.n_primitives_probs, dtype=float)
        if self.primitive_type_probs is None:
            self.primitive_type_probs = _DEFAULT_TYPE_WEIGHTS / _DEFAULT_TYPE_WEIGHTS.sum()
        else:
            self.primitive_type_probs = np.asarray(self.primitive_type_probs, dtype=float)
        if self.type_log_means is None:
            self.type_log_means = {s.key: s.init_log_mean.copy() for s in PRIMITIVE_SPECS}
        if self.type_log_stds is None:
            self.type_log_stds = {s.key: s.init_std.copy() for s in PRIMITIVE_SPECS}

    # -- sampling -----------------------------------------------------------

    def sample_n_primitives(self, rng: np.random.Generator) -> int:
        """Sample number of primitives (1..len(n_primitives_probs))."""
        return int(rng.choice(len(self.n_primitives_probs), p=self.n_primitives_probs) + 1)

    def sample_primitive(self, rng: np.random.Generator, is_base: bool = False) -> Primitive:
        """Sample one primitive of a CEM-chosen type. In structured mode the
        transform is left at identity (placement is done by sample_object); in
        legacy mode a random offset+rotation is applied here."""
        ti = int(rng.choice(len(PRIMITIVE_SPECS), p=self.primitive_type_probs))
        spec = PRIMITIVE_SPECS[ti]
        log_p = rng.normal(self.type_log_means[spec.key], self.type_log_stds[spec.key])
        params = np.clip(np.exp(log_p), spec.clamp_lo, spec.clamp_hi)

        offset = np.zeros(3)
        euler = np.zeros(3)
        if is_base or self.structured_placement:
            transform = Transform.identity()
        else:
            offset = rng.normal(0.0, self.offset_std, size=3)
            euler = rng.normal(0.0, self.rotation_std, size=3)
            transform = Transform.from_euler(offset, euler)

        prim = spec.build(params, transform)
        if is_base:
            prim.transform.translation[2] = seat_height(prim)
        prim._cem_offset = offset
        prim._cem_euler = euler
        return prim

    def _attach(self, new: Primitive, anchor: Primitive, rng: np.random.Generator):
        """Seat *new* against a face of *anchor*, axis-aligned, with overlap so
        the union is a single solid. Any of the 6 faces — the whole object is
        re-seated on the ground afterwards, so going below the anchor is fine."""
        ha = half_extents(anchor)
        hn = half_extents(new)
        axis = int(rng.integers(0, 3))
        sign = float(rng.choice([-1.0, 1.0]))
        d = ha[axis] + hn[axis] - self.attach_overlap * min(ha[axis], hn[axis])
        center = np.asarray(anchor.transform.translation, float).copy()
        center[axis] += sign * d
        new.transform.translation = center
        new._cem_offset = center - np.asarray(anchor.transform.translation, float)

    def sample_object(self, rng: np.random.Generator,
                      name: str = "sampled_object") -> CompositeObject:
        """Sample a composite object. Each secondary primitive attaches to a
        random existing primitive — by face contact (structured, default) or by
        a random offset (legacy) — biasing toward a single connected body."""
        n_prims = self.sample_n_primitives(rng)
        base = self.sample_primitive(rng, is_base=True)
        primitives = [base]

        if self.structured_placement:
            base.transform.translation[2] = 0.0      # re-seated globally below
            for _ in range(1, n_prims):
                prim = self.sample_primitive(rng, is_base=False)
                anchor = primitives[int(rng.integers(0, len(primitives)))]
                self._attach(prim, anchor, rng)
                primitives.append(prim)
            # Shift the whole assembly so its lowest point rests on z=0.
            low = min(float(p.transform.translation[2]) - float(half_extents(p)[2])
                      for p in primitives)
            for p in primitives:
                p.transform.translation[2] -= low
        else:
            for _ in range(1, n_prims):
                prim = self.sample_primitive(rng, is_base=False)
                anchor = primitives[int(rng.integers(0, len(primitives)))]
                prim.transform.translation = prim.transform.translation + anchor.transform.translation
                prim.transform.translation[2] = max(0.005, float(prim.transform.translation[2]))
                primitives.append(prim)

        friction = float(np.clip(rng.normal(self.friction_mean, self.friction_std), 0.1, 2.0))
        return CompositeObject(primitives=primitives, name=name, friction=friction)

    # -- serialization ------------------------------------------------------

    def to_dict(self) -> Dict:
        return {
            'max_primitives': int(self.max_primitives),
            'n_primitives_probs': self.n_primitives_probs.tolist(),
            'primitive_type_probs': self.primitive_type_probs.tolist(),
            'type_log_means': {k: np.asarray(v).tolist() for k, v in self.type_log_means.items()},
            'type_log_stds': {k: np.asarray(v).tolist() for k, v in self.type_log_stds.items()},
            'offset_std': float(self.offset_std),
            'rotation_std': float(self.rotation_std),
            'friction_mean': float(self.friction_mean),
            'friction_std': float(self.friction_std),
            'structured_placement': bool(self.structured_placement),
            'attach_overlap': float(self.attach_overlap),
        }

    @classmethod
    def from_dict(cls, d: Dict) -> 'ParameterDistribution':
        """Deserialize. Tolerant of missing keys (and of legacy v1 dicts, which
        simply fall back to the spec-table defaults for per-type params)."""
        obj = cls(max_primitives=int(d.get('max_primitives', 16)))
        if 'n_primitives_probs' in d:
            obj.n_primitives_probs = np.array(d['n_primitives_probs'], dtype=float)
        if 'primitive_type_probs' in d and len(d['primitive_type_probs']) == len(PRIMITIVE_SPECS):
            obj.primitive_type_probs = np.array(d['primitive_type_probs'], dtype=float)
        for k, v in (d.get('type_log_means') or {}).items():
            obj.type_log_means[k] = np.array(v, dtype=float)
        for k, v in (d.get('type_log_stds') or {}).items():
            obj.type_log_stds[k] = np.array(v, dtype=float)
        obj.offset_std = float(d.get('offset_std', obj.offset_std))
        obj.rotation_std = float(d.get('rotation_std', obj.rotation_std))
        obj.friction_mean = float(d.get('friction_mean', obj.friction_mean))
        obj.friction_std = float(d.get('friction_std', obj.friction_std))
        obj.structured_placement = bool(d.get('structured_placement', obj.structured_placement))
        obj.attach_overlap = float(d.get('attach_overlap', obj.attach_overlap))
        return obj


@dataclass
class CEMConfig:
    """Configuration for Cross-Entropy Method."""
    n_samples: int = 100          # Samples per iteration
    elite_fraction: float = 0.2   # Top fraction to keep
    n_iterations: int = 50        # Number of CEM iterations
    learning_rate: float = 0.7    # Interpolation with old distribution
    min_std: float = 0.1          # Minimum standard deviation
    seed: int = 42


class CEMOptimizer:
    """
    Cross-Entropy Method optimizer for object generation.

    Iteratively improves the parameter distribution to generate
    objects with high manipulation scores.
    """

    def __init__(self, config: CEMConfig = None, scoring_config: ScoringConfig = None,
                 initial_distribution: ParameterDistribution = None):
        self.config = config or CEMConfig()
        self.scorer = ObjectScorer(scoring_config)
        self.rng = np.random.default_rng(self.config.seed)

        # Current distribution
        self.distribution = initial_distribution or ParameterDistribution()

        # History for analysis
        self.history: List[Dict] = []

    def optimize(self,
                 callback: Optional[Callable[[int, float, ParameterDistribution], None]] = None
                 ) -> ParameterDistribution:
        """
        Run CEM optimization.

        Args:
            callback: Optional function called each iteration with
                     (iteration, mean_elite_score, current_distribution)

        Returns:
            Optimized parameter distribution
        """
        n_elite = max(1, int(self.config.n_samples * self.config.elite_fraction))

        for iteration in range(self.config.n_iterations):
            # Sample candidates
            candidates = []
            scores = []
            score_breakdowns = []

            for i in range(self.config.n_samples):
                obj = self.distribution.sample_object(
                    self.rng,
                    name=f"iter{iteration}_sample{i}"
                )
                breakdown = self.scorer.score(obj)

                candidates.append(obj)
                scores.append(breakdown.total_score)
                score_breakdowns.append(breakdown)

            scores = np.array(scores)

            # Select elite samples
            elite_indices = np.argsort(scores)[-n_elite:]
            elite_scores = scores[elite_indices]
            elite_objects = [candidates[i] for i in elite_indices]

            # Extract component scores for elites
            elite_breakdowns = [score_breakdowns[i] for i in elite_indices]
            elite_stability = [b.stability_score for b in elite_breakdowns]
            elite_graspability = [b.graspability_score for b in elite_breakdowns]
            elite_size = [b.size_score for b in elite_breakdowns]

            # Update distribution based on elites
            self._update_distribution(elite_objects)

            # Record history
            self.history.append({
                'iteration': iteration,
                'mean_score': float(np.mean(scores)),
                'max_score': float(np.max(scores)),
                'mean_elite_score': float(np.mean(elite_scores)),
                'mean_stability': float(np.mean(elite_stability)),
                'mean_graspability': float(np.mean(elite_graspability)),
                'mean_size': float(np.mean(elite_size)),
                'std_score': float(np.std(scores))
            })

            if callback:
                callback(iteration, float(np.mean(elite_scores)), self.distribution)

        return self.distribution

    def _update_distribution(self, elite_objects: List[CompositeObject]):
        """Moment-match the distribution to the elite samples, generalized over
        the primitive spec table and over learned placement spread."""
        if not elite_objects:
            return

        lr = self.config.learning_rate
        min_std = self.config.min_std
        dist = self.distribution
        maxp = dist.max_primitives

        n_prims_counts = np.zeros(maxp)
        ptype_counts = np.zeros(len(PRIMITIVE_SPECS))
        log_params: Dict[int, List[np.ndarray]] = {i: [] for i in range(len(PRIMITIVE_SPECS))}
        offsets: List[np.ndarray] = []
        eulers: List[np.ndarray] = []

        for obj in elite_objects:
            nn = min(len(obj.primitives), maxp)
            n_prims_counts[nn - 1] += 1
            for j, prim in enumerate(obj.primitives):
                ti = _SPEC_INDEX.get(prim.ptype)
                if ti is None:
                    continue
                spec = PRIMITIVE_SPECS[ti]
                ptype_counts[ti] += 1
                vals = np.clip(spec.extract(prim), spec.clamp_lo, spec.clamp_hi)
                log_params[ti].append(np.log(vals))
                if j > 0:
                    off = getattr(prim, '_cem_offset', None)
                    eu = getattr(prim, '_cem_euler', None)
                    if off is not None:
                        offsets.append(np.asarray(off, dtype=float))
                    if eu is not None:
                        eulers.append(np.asarray(eu, dtype=float))

        # Number of primitives
        if n_prims_counts.sum() > 0:
            new_probs = n_prims_counts / n_prims_counts.sum()
            dist.n_primitives_probs = lr * new_probs + (1 - lr) * dist.n_primitives_probs

        # Primitive type probabilities (epsilon-smoothed to avoid zero mass)
        if ptype_counts.sum() > 0:
            new_probs = ptype_counts / ptype_counts.sum()
            new_probs = (new_probs + 0.01) / (new_probs + 0.01).sum()
            dist.primitive_type_probs = lr * new_probs + (1 - lr) * dist.primitive_type_probs

        # Per-type size parameters (log-space)
        for ti, spec in enumerate(PRIMITIVE_SPECS):
            if log_params[ti]:
                arr = np.array(log_params[ti])
                new_mean = arr.mean(axis=0)
                new_std = np.maximum(arr.std(axis=0), min_std)
                dist.type_log_means[spec.key] = lr * new_mean + (1 - lr) * dist.type_log_means[spec.key]
                dist.type_log_stds[spec.key] = lr * new_std + (1 - lr) * dist.type_log_stds[spec.key]

        # Learned placement spread (offset / rotation)
        if offsets:
            new_off = float(np.std(np.array(offsets)))
            dist.offset_std = max(_OFFSET_STD_FLOOR, lr * new_off + (1 - lr) * dist.offset_std)
        if eulers:
            new_rot = float(np.std(np.array(eulers)))
            dist.rotation_std = max(_ROT_STD_FLOOR, lr * new_rot + (1 - lr) * dist.rotation_std)

        # Friction
        frictions = [obj.friction for obj in elite_objects if hasattr(obj, 'friction')]
        if frictions:
            fa = np.array(frictions)
            dist.friction_mean = lr * float(np.mean(fa)) + (1 - lr) * dist.friction_mean
            dist.friction_std = max(min_std, lr * float(np.std(fa)) + (1 - lr) * dist.friction_std)

    def save(self, path: Path):
        """Save optimizer state to file."""
        state = {
            'config': {
                'n_samples': self.config.n_samples,
                'elite_fraction': self.config.elite_fraction,
                'n_iterations': self.config.n_iterations,
                'learning_rate': self.config.learning_rate,
                'min_std': self.config.min_std,
                'seed': self.config.seed
            },
            'distribution': self.distribution.to_dict(),
            'history': self.history
        }
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> 'CEMOptimizer':
        """Load optimizer state from file."""
        with open(path, 'r') as f:
            state = json.load(f)

        config = CEMConfig(**state['config'])
        optimizer = cls(config)
        optimizer.distribution = ParameterDistribution.from_dict(state['distribution'])
        optimizer.history = state['history']
        return optimizer


def train_generator(n_iterations: int = 50,
                    n_samples: int = 100,
                    verbose: bool = True) -> CEMOptimizer:
    """
    Convenience function to train a generator from scratch.

    Returns:
        Trained CEMOptimizer with optimized distribution
    """
    config = CEMConfig(
        n_iterations=n_iterations,
        n_samples=n_samples
    )

    optimizer = CEMOptimizer(config)

    def progress_callback(iteration, mean_score, dist):
        if verbose:
            print(f"Iteration {iteration:3d}: mean elite score = {mean_score:.4f}")

    optimizer.optimize(callback=progress_callback)

    return optimizer
