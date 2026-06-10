"""
Main Object Generator Class.

High-level API for generating robotic manipulation objects.
Combines primitives, scoring, CEM optimization, and export.
"""

import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Generator
from dataclasses import dataclass
import json

from primitives import (
    CompositeObject, create_simple_box, create_mug_like, create_l_shape, 
    create_dumbbell, create_hammer, create_bottle,
    create_t_shape, create_u_shape, create_v_shape, create_monitor,
    create_barbell, create_snowman, create_camera, create_frying_pan,
    create_flashlight, create_spatula, create_remote, create_joystick
)
from scoring import ObjectScorer, ScoreBreakdown, ScoringConfig
from cem import CEMOptimizer, CEMConfig, ParameterDistribution
from export import URDFExporter, BatchExporter, ExportConfig


@dataclass
class GeneratorConfig:
    """Master configuration for the object generator."""
    # CEM parameters
    cem_iterations: int = 50
    cem_samples: int = 100
    elite_fraction: float = 0.2
    learning_rate: float = 0.7

    # Maximum primitives per object (configurable; no hard cap). The CEM learns
    # a distribution over counts 1..max_primitives.
    max_primitives: int = 16
    
    # Scoring thresholds
    min_extent: float = 0.02
    max_extent: float = 0.15
    gripper_width_max: float = 0.08
    # Assembly reward (v2): nudges the CEM toward multi-part composites instead
    # of single primitives. Saturates at target_primitives. assembly_weight=0
    # reproduces v1 scoring. See DISCREPANCIES.md item 8.
    assembly_weight: float = 1.0
    target_primitives: int = 3
    # Low-graspability gate (v2.4): penalize the total by the graspable-material
    # fraction so the CEM stops favoring ungraspable cone/pyramid parts. On by
    # default for v2; paper_repro turns it off to preserve v1 scoring.
    low_grasp_gate: bool = True
    # Point ⑦: score the dynamic tip-over margin (atan(margin/COM-height)) so the CEM
    # avoids static-stable-but-tippy objects (screwdrivers). Off by default (preserves
    # v1/paper scoring); turn on for drop-stability-aware generation.
    dynamic_stability: bool = False
    # ⑦ as a HARD gate: multiply the total by the tip-over stability (floored), so the
    # CEM actually suppresses the tippy tail instead of diluting it as a soft term.
    dynamic_stability_gate: bool = False
    # Point ⑥: post-hoc re-orient each generated object onto its most-stable resting pose
    # (a guarantee it settles upright), for objects below repair_min_tip_deg.
    repair_stability: bool = False
    repair_min_tip_deg: float = 20.0

    # Export settings
    density: float = 1000.0
    mesh_format: str = "obj"
    use_mesh_inertia: bool = True     # overlap-aware inertia at export (v2 default)
    
    # Generation
    seed: int = 42
    min_score_threshold: float = 0.4  # Reject objects below this score
    # v2 default (True): reject objects whose primitives are not a single
    # connected body (floating parts). See DISCREPANCIES.md item 6. Set False
    # for the v1 paper-repro behavior.
    require_connected: bool = True
    # v2 default (True): oversample candidates and keep the top-n by independent
    # force-closure graspability (grasp_planner), decoupling final selection
    # from the loose scorer proxy the CEM optimizes. See DISCREPANCIES.md item 8.
    rerank_by_grasp: bool = True
    rerank_oversample: int = 2        # candidate pool = rerank_oversample * n


class RoboticObjectGenerator:
    """
    Main class for generating diverse, manipulation-suitable 3D objects.
    
    Usage:
        # Quick generation with default settings
        gen = RoboticObjectGenerator()
        objects = gen.generate(n=10)
        
        # Train optimized generator first
        gen = RoboticObjectGenerator()
        gen.train(n_iterations=30)
        objects = gen.generate(n=100)
        gen.export_all(objects, "output/objects")
        
        # Load pre-trained generator
        gen = RoboticObjectGenerator.load("trained_generator.json")
        objects = gen.generate(n=50)
    """
    
    def __init__(self, config: GeneratorConfig = None):
        self.config = config or GeneratorConfig()
        self.rng = np.random.default_rng(self.config.seed)
        
        # Initialize components
        self.scoring_config = ScoringConfig(
            min_extent=self.config.min_extent,
            max_extent=self.config.max_extent,
            gripper_width_max=self.config.gripper_width_max,
            density=self.config.density,
            assembly_weight=self.config.assembly_weight,
            target_primitives=self.config.target_primitives,
            low_grasp_gate=self.config.low_grasp_gate,
            dynamic_stability=self.config.dynamic_stability,
            dynamic_stability_gate=self.config.dynamic_stability_gate,
        )
        self.scorer = ObjectScorer(self.scoring_config)
        
        # Distribution (can be trained)
        self.distribution = ParameterDistribution(max_primitives=self.config.max_primitives)
        self.is_trained = False
        self.training_history = []

    def seed_from(self, obj_or_name, concentration: float = 0.7):
        """Point ⑧: warm-start the free distribution from an archetype (by name) or a
        seed object, then `train()` lets structure evolve from there. Combine with
        dynamic_stability_gate / repair_stability to evolve a tippy seed into a stable one."""
        if isinstance(obj_or_name, str):
            from archetypes import ARCHETYPE_REGISTRY
            obj_or_name = ARCHETYPE_REGISTRY[obj_or_name]()
        self.distribution.seed_from_object(obj_or_name, concentration)
        return self

    def train(self,
              n_iterations: int = None, 
              n_samples: int = None,
              verbose: bool = True) -> List[Dict]:
        """
        Train the generator using Cross-Entropy Method.
        
        This optimizes the parameter distribution to produce
        objects with higher manipulation scores.
        
        Returns:
            Training history (list of dicts with scores per iteration)
        """
        cem_config = CEMConfig(
            n_iterations=n_iterations or self.config.cem_iterations,
            n_samples=n_samples or self.config.cem_samples,
            elite_fraction=self.config.elite_fraction,
            learning_rate=self.config.learning_rate,
            seed=self.config.seed
        )
        
        optimizer = CEMOptimizer(cem_config, self.scoring_config, 
                               initial_distribution=self.distribution)
        
        def callback(iteration, mean_score, dist):
            if verbose:
                print(f"  Iteration {iteration:3d}: elite mean = {mean_score:.4f}")
        
        if verbose:
            print("Training generator...")
        
        self.distribution = optimizer.optimize(callback=callback)
        self.is_trained = True
        self.training_history = optimizer.history
        
        if verbose:
            final_score = self.training_history[-1]['mean_elite_score']
            print(f"Training complete. Final elite score: {final_score:.4f}")
        
        return self.training_history
    
    def generate(self,
                 n: int = 1,
                 ensure_quality: bool = True,
                 max_attempts_per_object: int = 10) -> List[CompositeObject]:
        """
        Generate n objects from the current distribution.

        When ``rerank_by_grasp`` is set (v2 default) a larger candidate pool is
        produced and the top-n by independent force-closure graspability are
        kept — so final selection does not depend solely on the scorer proxy
        the CEM optimizes against.

        Args:
            n: Number of objects to generate
            ensure_quality: If True, reject objects below score threshold
            max_attempts_per_object: Max sampling attempts per object

        Returns:
            List of CompositeObjects
        """
        pool_n = n
        if ensure_quality and self.config.rerank_by_grasp and n >= 1:
            pool_n = max(n, n * int(self.config.rerank_oversample))

        objects = []
        for i in range(pool_n):
            for attempt in range(max_attempts_per_object):
                obj = self.distribution.sample_object(
                    self.rng,
                    name=f"generated_{i:04d}"
                )

                if not ensure_quality:
                    objects.append(obj)
                    break

                # Connectivity gate (v2 default on): reject floating parts.
                if self.config.require_connected and not obj.is_connected():
                    continue

                # Check quality
                score = self.scorer.score(obj)
                if score.total_score >= self.config.min_score_threshold:
                    objects.append(obj)
                    break
            else:
                # All attempts failed, use last one anyway
                objects.append(obj)

        # Re-rank the oversampled pool by independent force-closure graspability
        # and keep the best n (renaming to a stable generated_#### sequence).
        if ensure_quality and self.config.rerank_by_grasp and len(objects) > n:
            from grasp_planner import plan_grasps
            def _grasp_key(o):
                try:
                    r = plan_grasps(o, n_surface=128, max_pairs=600, seed=self.config.seed)
                    return (r.n_collision_free, r.n_friction_pass)
                except Exception:
                    return (0, 0)
            objects = sorted(objects, key=_grasp_key, reverse=True)[:n]
            for i, o in enumerate(objects):
                o.name = f"generated_{i:04d}"

        # Point ⑥: project each object onto its most-stable resting pose, so it settles
        # upright on the table by construction (a guarantee, not just an optimized proxy).
        if self.config.repair_stability:
            from stability_repair import repair_stability
            for o in objects:
                repair_stability(o, min_tip_deg=self.config.repair_min_tip_deg)

        return objects

    def sample_with_verdicts(self, n: int = 12):
        """Sample ``n`` objects from the current distribution and tag each with
        the accept/reject verdict the quality gate would give it — WITHOUT the
        retry loop and WITHOUT reranking. Used only for documentation (showing
        accepted vs rejected samples). ``generate()`` is unaffected.

        Returns a list of ``(CompositeObject, verdict)`` where verdict is one of
        ``'accepted'`` / ``'disconnected'`` / ``f'low_score:{total:.2f}'``.
        (The ``'disconnected'`` verdict requires a CSG backend; without one,
        ``is_connected()`` is permissive and only ``'low_score'`` appears.)
        """
        out = []
        for i in range(n):
            obj = self.distribution.sample_object(self.rng, name=f"sample_{i:04d}")
            if self.config.require_connected and not obj.is_connected():
                out.append((obj, "disconnected"))
                continue
            score = self.scorer.score(obj)
            if score.total_score >= self.config.min_score_threshold:
                out.append((obj, "accepted"))
            else:
                out.append((obj, f"low_score:{score.total_score:.2f}"))
        return out

    def generate_stream(self,
                        ensure_quality: bool = True) -> Generator[CompositeObject, None, None]:
        """
        Generate objects as an infinite stream.
        
        Yields:
            CompositeObjects one at a time
        """
        i = 0
        while True:
            objs = self.generate(n=1, ensure_quality=ensure_quality)
            if objs:
                obj = objs[0]
                obj.name = f"generated_{i:06d}"
                yield obj
                i += 1
    
    def score_object(self, obj: CompositeObject) -> ScoreBreakdown:
        """Get detailed score breakdown for an object."""
        return self.scorer.score(obj)
    
    def export_all(self, 
                   objects: List[CompositeObject],
                   output_dir: str,
                   name_prefix: str = "object") -> Dict[str, list]:
        """
        Export all objects to URDF/SDF format.
        
        Args:
            objects: List of objects to export
            output_dir: Output directory
            name_prefix: Prefix for object names
        
        Returns:
            Dictionary with lists of generated file paths
        """
        export_config = ExportConfig(
            density=self.config.density,
            mesh_format=self.config.mesh_format,
            use_mesh_inertia=self.config.use_mesh_inertia
        )
        
        exporter = BatchExporter(export_config)
        return exporter.export_batch(objects, Path(output_dir), name_prefix)
    
    def save(self, path: str):
        """Save generator state to file."""
        state = {
            'config': {
                'cem_iterations': self.config.cem_iterations,
                'cem_samples': self.config.cem_samples,
                'elite_fraction': self.config.elite_fraction,
                'min_extent': self.config.min_extent,
                'max_extent': self.config.max_extent,
                'gripper_width_max': self.config.gripper_width_max,
                'density': self.config.density,
                'mesh_format': self.config.mesh_format,
                'seed': self.config.seed,
                'min_score_threshold': self.config.min_score_threshold
            },
            'distribution': self.distribution.to_dict(),
            'is_trained': self.is_trained,
            'training_history': self.training_history
        }
        
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'RoboticObjectGenerator':
        """Load generator from file."""
        with open(path, 'r') as f:
            state = json.load(f)
        
        config = GeneratorConfig(**state['config'])
        gen = cls(config)
        gen.distribution = ParameterDistribution.from_dict(state['distribution'])
        gen.is_trained = state['is_trained']
        gen.training_history = state['training_history']
        
        return gen


# Convenience functions

def quick_generate(n: int = 10, train_first: bool = True) -> List[CompositeObject]:
    """
    Quick generation with sensible defaults.
    
    Args:
        n: Number of objects to generate
        train_first: If True, run brief training first
    
    Returns:
        List of generated objects
    """
    gen = RoboticObjectGenerator()
    
    if train_first:
        gen.train(n_iterations=20, n_samples=50, verbose=False)
    
    return gen.generate(n)


def generate_and_export(n: int, 
                        output_dir: str,
                        train_iterations: int = 30) -> Dict[str, list]:
    """
    Generate objects and export to URDF/SDF in one call.
    
    Args:
        n: Number of objects
        output_dir: Output directory
        train_iterations: CEM training iterations
    
    Returns:
        Dictionary of exported file paths
    """
    gen = RoboticObjectGenerator()
    gen.train(n_iterations=train_iterations, verbose=True)
    objects = gen.generate(n)
    return gen.export_all(objects, output_dir)


def paper_repro_generator(seed: int = 42) -> 'RoboticObjectGenerator':
    """Build a generator configured to reproduce the v1 (ICARM-paper) behavior
    from the v2 code: 4 primitives max, only the original 4 primitive types,
    analytic (overlap-summing) inertia, and no connectivity / grasp-rerank
    post-processing. See DISCREPANCIES.md. (v1 is also preserved on the
    legacy/v1-icarm git branch.)"""
    from cem import PRIMITIVE_SPECS
    config = GeneratorConfig(
        max_primitives=4,
        use_mesh_inertia=False,
        require_connected=False,
        rerank_by_grasp=False,
        assembly_weight=0.0,       # v1 scoring had no assembly reward
        low_grasp_gate=False,      # v1 scoring had no graspability gate
        seed=seed,
    )
    gen = RoboticObjectGenerator(config)
    # Restrict to the original box/cyl/sphere/capsule palette and v1 count prior.
    type_probs = np.zeros(len(PRIMITIVE_SPECS))
    type_probs[:4] = [0.4, 0.35, 0.15, 0.1]
    gen.distribution.primitive_type_probs = type_probs
    gen.distribution.n_primitives_probs = np.array([0.3, 0.4, 0.2, 0.1])
    gen.distribution.structured_placement = False   # v1 used free random offsets
    return gen


def create_archetype_set(output_dir: str) -> Dict[str, Path]:
    """
    Export every archetype in the central registry (archetypes.ARCHETYPE_REGISTRY)
    to URDF/SDF. Adding an archetype to the registry automatically includes it
    here — there is no per-shape list to maintain.
    """
    from export import URDFExporter
    from archetypes import ARCHETYPE_REGISTRY

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exporter = URDFExporter()
    results: Dict[str, Path] = {}
    for name, factory in ARCHETYPE_REGISTRY.items():
        obj = factory()
        obj.name = name
        paths = exporter.export(obj, output_dir / name, name)
        results[name] = paths['urdf']
    return results
