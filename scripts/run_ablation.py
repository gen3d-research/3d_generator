#!/usr/bin/env python3
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add 3d_generator to path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

try:
    from d3_generator.generator import RoboticObjectGenerator, GeneratorConfig
    from d3_generator.scoring import ObjectScorer, ScoringConfig
    from d3_generator.cem import CEMOptimizer, CEMConfig
except ImportError:
    try:
        from generator import RoboticObjectGenerator, GeneratorConfig
        from scoring import ObjectScorer, ScoringConfig
        from cem import CEMOptimizer, CEMConfig
    except ImportError:
        sys.exit(1)

OUTPUT_DIR = Path("images")
OUTPUT_DIR.mkdir(exist_ok=True)

class AblatedObjectScorer(ObjectScorer):
    """Scorer with weights modified for ablation."""
    def __init__(self, config=None, ignore_stability=False):
        super().__init__(config)
        self.ignore_stability = ignore_stability
        
    def score(self, obj):
        breakdown = super().score(obj)
        
        # Reconstruct total score with specific weights
        weights = {
            'validity': 2.0,
            'stability': 0.0 if self.ignore_stability else 1.5,
            'graspability': 1.5,
            'size': 1.0,
            'complexity': 0.5
        }
        
        total_weight = sum(weights.values())
        if total_weight == 0: return breakdown
        
        breakdown.total_score = (
            weights['validity'] * breakdown.validity_score +
            weights['stability'] * breakdown.stability_score +
            weights['graspability'] * breakdown.graspability_score +
            weights['size'] * breakdown.size_score +
            weights['complexity'] * breakdown.complexity_score
        ) / total_weight
        
        return breakdown

def run_experiment(ignore_stability=False, label="Full Method"):
    print(f"Running Experiment: {label}")
    
    # Custom training loop to inject ablated scorer
    cem_config = CEMConfig(n_iterations=20, n_samples=50)
    scoring_config = ScoringConfig()
    
    # Create optimizer with ablated scorer
    optimizer = CEMOptimizer(cem_config, scoring_config)
    optimizer.scorer = AblatedObjectScorer(scoring_config, ignore_stability=ignore_stability)
    
    optimizer.optimize(callback=None)
    
    # Generate objects from trained distribution
    dist = optimizer.distribution
    rng = np.random.default_rng(42)
    
    margins = []
    
    # Generate validation set
    for i in range(100):
        obj = dist.sample_object(rng)
        # Always score with full scorer to see "true" stability metric
        true_scorer = ObjectScorer(scoring_config)
        bd = true_scorer.score(obj)
        margins.append(bd.stability_margin)
        
    return margins

def main():
    # 1. Full Method (Control)
    margins_full = run_experiment(ignore_stability=False, label="Full Method")
    
    # 2. Ablated (No Stability)
    margins_ablated = run_experiment(ignore_stability=True, label="No Stability Constraint")
    
    # Plot
    plt.figure(figsize=(7, 5))
    
    plt.hist(margins_full, bins=20, alpha=0.5, label='Full Method', density=True, color='blue')
    plt.hist(margins_ablated, bins=20, alpha=0.5, label='No Stability Constraint', density=True, color='red')
    
    plt.axvline(0, color='k', linestyle='--', alpha=0.5)
    plt.xlabel("Stability Margin (m)")
    plt.ylabel("Frequency Density")
    plt.title("Ablation Study: Impact of Stability Constraint")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "evaluation_ablation_stability.png", dpi=300)
    plt.close()
    
    print("Ablation complete. Figure saved.")

if __name__ == "__main__":
    main()
