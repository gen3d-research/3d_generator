
import numpy as np
from typing import Dict, List, Callable, Any
from dataclasses import dataclass
import inspect
from primitives import *
from scoring import ObjectScorer, ScoringConfig

@dataclass
class LearnableParam:
    name: str
    mean: float
    std: float
    min_val: float
    max_val: float
    is_vector: bool = False
    vector_dim: int = 3
    prototype: Any = None   # the archetype's canonical default — anchor for `update`

class ArchetypeDistribution:
    """Maintains distributions for a specific archetype's parameters."""
    def __init__(self, factory_fn: Callable):
        self.factory_fn = factory_fn
        self.params: Dict[str, LearnableParam] = {}
        self._introspect_params()

    def _introspect_params(self):
        """Guess parameters from function signature."""
        sig = inspect.signature(self.factory_fn)
        for name, param in sig.parameters.items():
            # Check for required parameters (no default).  Use `is`
            # rather than `==` here because some defaults are numpy
            # arrays, and ``np.array(...) == sentinel`` returns an
            # array (not a bool) which raises in the truthiness check.
            if param.default is inspect.Parameter.empty:
                # Heuristic for required params
                if "dims" in name or "size" in name:
                     # vector 3
                     val = np.array([0.05, 0.05, 0.05])
                     self.params[name] = LearnableParam(
                         name=name, mean=val, std=val*0.5, 
                         min_val=val*0.1, max_val=val*4.0, 
                         is_vector=True, vector_dim=3
                     )
                else:
                    # Assume scalar float
                    val = 0.05
                    self.params[name] = LearnableParam(
                        name=name, mean=val, std=val*0.5, 
                        min_val=val*0.1, max_val=val*4.0
                    )
                continue

            # Heuristics to set initial distribution based on default values
            # Variants should stay *recognizable* members of the archetype family, so
            # keep the spread moderate (±12% std, clamped to 0.6x..1.6x of the default)
            # and remember the default as the prototype anchor (see `update`). Wide
            # ranges (the old 0.2x..3.0x) drift far from the shape and let the optimizer
            # walk off into a high-scoring non-screwdriver.
            if isinstance(param.default, (int, float)):
                val = float(param.default)
                self.params[name] = LearnableParam(
                    name=name, mean=val, std=val * 0.12,
                    min_val=val * 0.6, max_val=val * 1.6, prototype=val,
                )
            elif isinstance(param.default, np.ndarray):
                 val = param.default
                 self.params[name] = LearnableParam(
                     name=name, mean=val, std=val * 0.12,
                     min_val=val * 0.6, max_val=val * 1.6,
                     is_vector=True, vector_dim=len(val), prototype=np.array(val),
                 )

    def sample(self, rng: np.random.Generator) -> Dict[str, Any]:
        """Sample a parameter set."""
        result = {}
        for name, p in self.params.items():
            if p.is_vector:
                # Independent normals for each component
                # Handle array mean/std
                val = rng.normal(p.mean, p.std)
                val = np.maximum(val, p.min_val)
                val = np.minimum(val, p.max_val)
                result[name] = val
            else:
                val = rng.normal(p.mean, p.std)
                val = float(np.clip(val, p.min_val, p.max_val))
                result[name] = val
        return result

    def update(self, elites_params: List[Dict[str, Any]], lr: float = 0.7):
        """Update distributions based on elites."""
        if not elites_params:
            return

        for name, p in self.params.items():
            # Gather values for this param
            values = [e[name] for e in elites_params]
            if p.is_vector:
                vals_arr = np.array(values) # Shape (N, 3)
                new_mean = np.mean(vals_arr, axis=0)
                new_std = np.std(vals_arr, axis=0)
            else:
                vals_arr = np.array(values)
                new_mean = np.mean(vals_arr)
                new_std = np.std(vals_arr)
            
            # Update with learning rate
            p.mean = lr * new_mean + (1 - lr) * p.mean
            # Anchor toward the prototype so the optimum stays a *recognizable* variant
            # of the archetype (a tuned screwdriver, not a high-scoring blob) rather than
            # walking to the edge of the parameter box.
            if p.prototype is not None:
                anchor = 0.3
                p.mean = (1 - anchor) * p.mean + anchor * p.prototype
            p.mean = np.clip(p.mean, p.min_val, p.max_val)
            # Ensure std doesn't collapse to 0
            min_std = p.mean * 0.05 # Keep 5% noise minimum
            p.std = np.maximum(lr * new_std + (1 - lr) * p.std, min_std)


class ArchetypeTrainer:
    def __init__(self, factory_fn: Callable):
        self.factory_fn = factory_fn
        self.dist = ArchetypeDistribution(factory_fn)
        self.scorer = ObjectScorer()
        self.rng = np.random.default_rng(42)

    def train(self, iterations: int = 20, samples_per_iter: int = 50) -> List[float]:
        history = []
        elite_k = max(1, int(samples_per_iter * 0.2))
        
        print(f"Training {self.factory_fn.__name__}...", end='', flush=True)
        
        for i in range(iterations):
            # 1. Sample
            batch_params = []
            batch_scores = []
            
            for _ in range(samples_per_iter):
                params = self.dist.sample(self.rng)
                try:
                    obj = self.factory_fn(**params)
                    s = self.scorer.score(obj).total_score
                except Exception:
                    s = 0.0 # Penalty for invalid params
                
                batch_params.append(params)
                batch_scores.append(s)
            
            # 2. Select Elites
            indices = np.argsort(batch_scores)[-elite_k:]
            elites = [batch_params[i] for i in indices]
            elite_mean_score = np.mean([batch_scores[i] for i in indices])
            
            # 3. Update
            self.dist.update(elites)
            history.append(elite_mean_score)
            print(".", end='', flush=True)
            
        print(f" Done. Final: {history[-1]:.4f}")
        return history

    def generate(self, n: int) -> List[CompositeObject]:
        """Generate N objects using trained distribution."""
        objects = []
        for i in range(n):
            params = self.dist.sample(self.rng)
            obj = self.factory_fn(**params)
            obj.name = f"{self.factory_fn.__name__}_{i:05d}"
            objects.append(obj)
        return objects
