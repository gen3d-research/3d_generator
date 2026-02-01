"""
Cross-Entropy Method (CEM) for Lightweight Generative Learning.

Adapts the object generation distribution toward high-scoring objects
without requiring any external dataset. This is the "learning" component.

The CEM iteratively:
1. Samples candidates from current distribution
2. Evaluates each candidate
3. Selects top-k% (elite) samples
4. Updates distribution parameters to maximize likelihood of elites
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass, field
import json
from pathlib import Path

from primitives import (
    CompositeObject, Primitive, Box, Cylinder, Sphere, Capsule,
    Transform, PrimitiveType
)
from scoring import ObjectScorer, ScoreBreakdown, ScoringConfig


@dataclass
class ParameterDistribution:
    """
    Factorized Gaussian distribution over object parameters.
    
    Parameters are stored in log-space for dimensions to ensure positivity.
    Each parameter has a mean and standard deviation.
    """
    # Number of primitives (discrete, 1-4)
    n_primitives_probs: np.ndarray = field(
        default_factory=lambda: np.array([0.3, 0.4, 0.2, 0.1])  # P(1), P(2), P(3), P(4)
    )
    
    # Primitive type probabilities
    primitive_type_probs: np.ndarray = field(
        default_factory=lambda: np.array([0.4, 0.35, 0.15, 0.1])  # box, cyl, sphere, capsule
    )
    
    # Box dimensions (log-space): mean and std
    box_dims_mean: np.ndarray = field(default_factory=lambda: np.log(np.array([0.05, 0.05, 0.06])))
    box_dims_std: np.ndarray = field(default_factory=lambda: np.array([0.4, 0.4, 0.4]))
    
    # Cylinder parameters (log-space)
    cyl_radius_mean: float = np.log(0.025)
    cyl_radius_std: float = 0.3
    cyl_height_mean: float = np.log(0.06)
    cyl_height_std: float = 0.4
    
    # Sphere radius (log-space)
    sphere_radius_mean: float = np.log(0.03)
    sphere_radius_std: float = 0.3
    
    # Capsule parameters (log-space)
    capsule_radius_mean: float = np.log(0.015)
    capsule_radius_std: float = 0.3
    capsule_height_mean: float = np.log(0.04)
    capsule_height_std: float = 0.4
    
    # Position offsets for secondary primitives (linear space)
    offset_std: float = 0.03  # Standard deviation for XYZ offsets
    
    # Rotation (euler angles, radians)
    rotation_std: float = 0.3  # Std for rotation angles

    # Friction coefficient (LogUniform-ish via normal in log space? Or just trunc normal)
    # Using normal distribution centered around common friction values
    friction_mean: float = 0.8
    friction_std: float = 0.2
    
    def sample_n_primitives(self, rng: np.random.Generator) -> int:
        """Sample number of primitives."""
        return rng.choice(len(self.n_primitives_probs), p=self.n_primitives_probs) + 1
    
    def sample_primitive_type(self, rng: np.random.Generator) -> PrimitiveType:
        """Sample primitive type."""
        idx = rng.choice(len(self.primitive_type_probs), p=self.primitive_type_probs)
        return [PrimitiveType.BOX, PrimitiveType.CYLINDER, 
                PrimitiveType.SPHERE, PrimitiveType.CAPSULE][idx]
    
    def sample_box(self, rng: np.random.Generator, transform: Transform) -> Box:
        """Sample a box primitive."""
        log_dims = rng.normal(self.box_dims_mean, self.box_dims_std)
        dims = np.exp(log_dims)
        dims = np.clip(dims, 0.01, 0.15)  # Clamp to reasonable range
        return Box(dimensions=dims, transform=transform)
    
    def sample_cylinder(self, rng: np.random.Generator, transform: Transform) -> Cylinder:
        """Sample a cylinder primitive."""
        log_r = rng.normal(self.cyl_radius_mean, self.cyl_radius_std)
        log_h = rng.normal(self.cyl_height_mean, self.cyl_height_std)
        return Cylinder(
            radius=np.clip(np.exp(log_r), 0.005, 0.08),
            height=np.clip(np.exp(log_h), 0.01, 0.15),
            transform=transform
        )
    
    def sample_sphere(self, rng: np.random.Generator, transform: Transform) -> Sphere:
        """Sample a sphere primitive."""
        log_r = rng.normal(self.sphere_radius_mean, self.sphere_radius_std)
        return Sphere(
            radius=np.clip(np.exp(log_r), 0.01, 0.08),
            transform=transform
        )
    
    def sample_capsule(self, rng: np.random.Generator, transform: Transform) -> Capsule:
        """Sample a capsule primitive."""
        log_r = rng.normal(self.capsule_radius_mean, self.capsule_radius_std)
        log_h = rng.normal(self.capsule_height_mean, self.capsule_height_std)
        return Capsule(
            radius=np.clip(np.exp(log_r), 0.005, 0.05),
            height=np.clip(np.exp(log_h), 0.01, 0.12),
            transform=transform
        )
    
    def sample_primitive(self, rng: np.random.Generator, 
                         is_base: bool = False) -> Primitive:
        """Sample a primitive with appropriate transform."""
        ptype = self.sample_primitive_type(rng)
        
        if is_base:
            # Base primitive sits on ground
            # We'll set Z position after creating based on primitive height
            transform = Transform.identity()
        else:
            # Secondary primitives have random offset and rotation
            offset = rng.normal(0, self.offset_std, size=3)
            euler = rng.normal(0, self.rotation_std, size=3)
            transform = Transform.from_euler(offset, euler)
        
        if ptype == PrimitiveType.BOX:
            prim = self.sample_box(rng, transform)
            if is_base:
                # Adjust Z to sit on ground
                prim.transform.translation[2] = prim.dimensions[2] / 2
        elif ptype == PrimitiveType.CYLINDER:
            prim = self.sample_cylinder(rng, transform)
            if is_base:
                prim.transform.translation[2] = prim.height / 2
        elif ptype == PrimitiveType.SPHERE:
            prim = self.sample_sphere(rng, transform)
            if is_base:
                prim.transform.translation[2] = prim.radius
        else:  # CAPSULE
            prim = self.sample_capsule(rng, transform)
            if is_base:
                prim.transform.translation[2] = prim.height / 2 + prim.radius
        
        return prim
    
    def sample_object(self, rng: np.random.Generator, 
                      name: str = "sampled_object") -> CompositeObject:
        """Sample a complete composite object."""
        n_prims = self.sample_n_primitives(rng)
        primitives = []
        
        # First primitive is the base
        base = self.sample_primitive(rng, is_base=True)
        primitives.append(base)
        
        # Additional primitives attach to base
        for i in range(1, n_prims):
            prim = self.sample_primitive(rng, is_base=False)
            # Offset relative to base centroid
            prim.transform.translation += base.transform.translation
            # Ensure it's above ground
            prim.transform.translation[2] = max(0.01, prim.transform.translation[2])
            primitives.append(prim)
        
        # Sample friction
        friction = np.clip(rng.normal(self.friction_mean, self.friction_std), 0.1, 2.0)
        
        return CompositeObject(primitives=primitives, name=name, friction=friction)
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            'n_primitives_probs': self.n_primitives_probs.tolist(),
            'primitive_type_probs': self.primitive_type_probs.tolist(),
            'box_dims_mean': self.box_dims_mean.tolist(),
            'box_dims_std': self.box_dims_std.tolist(),
            'cyl_radius_mean': float(self.cyl_radius_mean),
            'cyl_radius_std': float(self.cyl_radius_std),
            'cyl_height_mean': float(self.cyl_height_mean),
            'cyl_height_std': float(self.cyl_height_std),
            'sphere_radius_mean': float(self.sphere_radius_mean),
            'sphere_radius_std': float(self.sphere_radius_std),
            'capsule_radius_mean': float(self.capsule_radius_mean),
            'capsule_radius_std': float(self.capsule_radius_std),
            'capsule_height_mean': float(self.capsule_height_mean),
            'capsule_height_std': float(self.capsule_height_std),
            'offset_std': float(self.offset_std),
            'rotation_std': float(self.rotation_std),
            'friction_mean': float(self.friction_mean),
            'friction_std': float(self.friction_std)
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'ParameterDistribution':
        """Deserialize from dictionary."""
        return cls(
            n_primitives_probs=np.array(d['n_primitives_probs']),
            primitive_type_probs=np.array(d['primitive_type_probs']),
            box_dims_mean=np.array(d['box_dims_mean']),
            box_dims_std=np.array(d['box_dims_std']),
            cyl_radius_mean=d['cyl_radius_mean'],
            cyl_radius_std=d['cyl_radius_std'],
            cyl_height_mean=d['cyl_height_mean'],
            cyl_height_std=d['cyl_height_std'],
            sphere_radius_mean=d['sphere_radius_mean'],
            sphere_radius_std=d['sphere_radius_std'],
            capsule_radius_mean=d['capsule_radius_mean'],
            capsule_radius_std=d['capsule_radius_std'],
            capsule_height_mean=d['capsule_height_mean'],
            capsule_height_std=d['capsule_height_std'],
            offset_std=d['offset_std'],
            rotation_std=d['rotation_std'],
            friction_mean=d.get('friction_mean', 0.8),
            friction_std=d.get('friction_std', 0.2)
        )


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
            
            # Early stopping if converged
            # if len(self.history) > 5:
            #     recent_means = [h['mean_elite_score'] for h in self.history[-5:]]
            #     if np.std(recent_means) < 0.01:
            #         print(f"Converged at iteration {iteration}")
            #         break
        
        return self.distribution
    
    def _update_distribution(self, elite_objects: List[CompositeObject]):
        """Update distribution parameters based on elite samples."""
        if not elite_objects:
            return
        
        lr = self.config.learning_rate
        min_std = self.config.min_std
        
        # Collect statistics from elite objects
        n_prims_counts = np.zeros(4)
        ptype_counts = np.zeros(4)
        
        box_dims_list = []
        cyl_params_list = []
        sphere_params_list = []
        capsule_params_list = []
        
        for obj in elite_objects:
            n = min(len(obj.primitives), 4)
            n_prims_counts[n - 1] += 1
            
            for prim in obj.primitives:
                if isinstance(prim, Box):
                    ptype_counts[0] += 1
                    box_dims_list.append(np.log(prim.dimensions))
                elif isinstance(prim, Cylinder):
                    ptype_counts[1] += 1
                    cyl_params_list.append([np.log(prim.radius), np.log(prim.height)])
                elif isinstance(prim, Sphere):
                    ptype_counts[2] += 1
                    sphere_params_list.append(np.log(prim.radius))
                elif isinstance(prim, Capsule):
                    ptype_counts[3] += 1
                    capsule_params_list.append([np.log(prim.radius), np.log(prim.height)])
        
        # Update n_primitives distribution
        if n_prims_counts.sum() > 0:
            new_probs = n_prims_counts / n_prims_counts.sum()
            self.distribution.n_primitives_probs = (
                lr * new_probs + (1 - lr) * self.distribution.n_primitives_probs
            )
        
        # Update primitive type distribution
        if ptype_counts.sum() > 0:
            new_probs = ptype_counts / ptype_counts.sum()
            # Add small epsilon to avoid zero probabilities
            new_probs = (new_probs + 0.01) / (new_probs + 0.01).sum()
            self.distribution.primitive_type_probs = (
                lr * new_probs + (1 - lr) * self.distribution.primitive_type_probs
            )
        
        # Update box parameters
        if box_dims_list:
            box_dims = np.array(box_dims_list)
            new_mean = np.mean(box_dims, axis=0)
            new_std = np.maximum(np.std(box_dims, axis=0), min_std)
            self.distribution.box_dims_mean = lr * new_mean + (1 - lr) * self.distribution.box_dims_mean
            self.distribution.box_dims_std = lr * new_std + (1 - lr) * self.distribution.box_dims_std
        
        # Update cylinder parameters
        if cyl_params_list:
            cyl_params = np.array(cyl_params_list)
            self.distribution.cyl_radius_mean = lr * np.mean(cyl_params[:, 0]) + (1 - lr) * self.distribution.cyl_radius_mean
            self.distribution.cyl_radius_std = max(min_std, lr * np.std(cyl_params[:, 0]) + (1 - lr) * self.distribution.cyl_radius_std)
            self.distribution.cyl_height_mean = lr * np.mean(cyl_params[:, 1]) + (1 - lr) * self.distribution.cyl_height_mean
            self.distribution.cyl_height_std = max(min_std, lr * np.std(cyl_params[:, 1]) + (1 - lr) * self.distribution.cyl_height_std)
        
        # Update sphere parameters
        if sphere_params_list:
            sphere_params = np.array(sphere_params_list)
            self.distribution.sphere_radius_mean = lr * np.mean(sphere_params) + (1 - lr) * self.distribution.sphere_radius_mean
            self.distribution.sphere_radius_std = max(min_std, lr * np.std(sphere_params) + (1 - lr) * self.distribution.sphere_radius_std)
        
        # Update capsule parameters
        if capsule_params_list:
            capsule_params = np.array(capsule_params_list)
            self.distribution.capsule_radius_mean = lr * np.mean(capsule_params[:, 0]) + (1 - lr) * self.distribution.capsule_radius_mean
            self.distribution.capsule_radius_std = max(min_std, lr * np.std(capsule_params[:, 0]) + (1 - lr) * self.distribution.capsule_radius_std)
            self.distribution.capsule_height_mean = lr * np.mean(capsule_params[:, 1]) + (1 - lr) * self.distribution.capsule_height_mean
            self.distribution.capsule_height_std = max(min_std, lr * np.std(capsule_params[:, 1]) + (1 - lr) * self.distribution.capsule_height_std)

        # Update friction parameters
        frictions = [obj.friction for obj in elite_objects if hasattr(obj, 'friction')]
        if frictions:
            friction_arr = np.array(frictions)
            self.distribution.friction_mean = lr * np.mean(friction_arr) + (1 - lr) * self.distribution.friction_mean
            self.distribution.friction_std = max(min_std, lr * np.std(friction_arr) + (1 - lr) * self.distribution.friction_std)
    
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
