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

from primitives import CompositeObject, create_simple_box, create_mug_like, create_l_shape
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
    
    # Scoring thresholds
    min_extent: float = 0.02
    max_extent: float = 0.15
    gripper_width_max: float = 0.08
    
    # Export settings
    density: float = 1000.0
    mesh_format: str = "obj"
    
    # Generation
    seed: int = 42
    min_score_threshold: float = 0.4  # Reject objects below this score


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
            density=self.config.density
        )
        self.scorer = ObjectScorer(self.scoring_config)
        
        # Distribution (can be trained)
        self.distribution = ParameterDistribution()
        self.is_trained = False
        self.training_history = []
    
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
            seed=self.config.seed
        )
        
        optimizer = CEMOptimizer(cem_config, self.scoring_config)
        
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
        
        Args:
            n: Number of objects to generate
            ensure_quality: If True, reject objects below score threshold
            max_attempts_per_object: Max sampling attempts per object
        
        Returns:
            List of CompositeObjects
        """
        objects = []
        
        for i in range(n):
            for attempt in range(max_attempts_per_object):
                obj = self.distribution.sample_object(
                    self.rng, 
                    name=f"generated_{i:04d}"
                )
                
                if not ensure_quality:
                    objects.append(obj)
                    break
                
                # Check quality
                score = self.scorer.score(obj)
                if score.total_score >= self.config.min_score_threshold:
                    objects.append(obj)
                    break
            else:
                # All attempts failed, use last one anyway
                objects.append(obj)
        
        return objects
    
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
            mesh_format=self.config.mesh_format
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


def create_archetype_set(output_dir: str) -> Dict[str, Path]:
    """
    Create a small set of archetype objects for testing.
    
    Returns common manipulation object types:
    - Simple boxes (various sizes)
    - Cylinders
    - Mug-like objects
    - L-shapes
    """
    from export import URDFExporter
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    exporter = URDFExporter()
    results = {}
    
    # Small box
    obj = create_simple_box(np.array([0.04, 0.04, 0.04]))
    obj.name = "small_box"
    paths = exporter.export(obj, output_dir / "small_box", "small_box")
    results['small_box'] = paths['urdf']
    
    # Tall box
    obj = create_simple_box(np.array([0.03, 0.03, 0.10]))
    obj.name = "tall_box"
    paths = exporter.export(obj, output_dir / "tall_box", "tall_box")
    results['tall_box'] = paths['urdf']
    
    # Flat box
    obj = create_simple_box(np.array([0.08, 0.06, 0.02]))
    obj.name = "flat_box"
    paths = exporter.export(obj, output_dir / "flat_box", "flat_box")
    results['flat_box'] = paths['urdf']
    
    # Mug-like
    obj = create_mug_like()
    obj.name = "mug"
    paths = exporter.export(obj, output_dir / "mug", "mug")
    results['mug'] = paths['urdf']
    
    # L-shape
    obj = create_l_shape()
    obj.name = "l_shape"
    paths = exporter.export(obj, output_dir / "l_shape", "l_shape")
    results['l_shape'] = paths['urdf']
    
    return results
