#!/usr/bin/env python3
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

# Add 3d_generator to path
# Use absolute path to ensure venv works correctly
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

# Import relative if directly running from scripts/ doesn't work as expected with module layout
try:
    from d3_generator.generator import RoboticObjectGenerator, GeneratorConfig
    from d3_generator.scoring import ObjectScorer
except ImportError:
    try:
        from generator import RoboticObjectGenerator, GeneratorConfig
        from scoring import ObjectScorer
    except ImportError:
        # Last resort for module resolution
        sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
        from generator import RoboticObjectGenerator, GeneratorConfig
        from scoring import ObjectScorer

OUTPUT_DIR = Path("images")
OUTPUT_DIR.mkdir(exist_ok=True)

def evaluate_random_baseline(n=100, seed=42):
    """Generate objects from initial random distribution."""
    print("Evaluating Random Baseline...")
    config = GeneratorConfig(seed=seed)
    gen = RoboticObjectGenerator(config)
    # No training
    objects = gen.generate(n, ensure_quality=False)
    
    scores = []
    margins = []
    pairs = []
    
    for obj in objects:
        breakdown = gen.score_object(obj)
        scores.append(breakdown.total_score)
        margins.append(breakdown.stability_margin)
        pairs.append(breakdown.n_antipodal_pairs)
        
    return scores, margins, pairs

def evaluate_cem_method(n=100, seed=42, iterations=30):
    """Generate objects from trained distribution."""
    print(f"Evaluating CEM Method ({iterations} iters)...")
    config = GeneratorConfig(seed=seed, cem_iterations=iterations)
    gen = RoboticObjectGenerator(config)
    gen.train(verbose=False)
    
    objects = gen.generate(n, ensure_quality=False)
    
    scores = []
    margins = []
    pairs = []
    
    for obj in objects:
        breakdown = gen.score_object(obj)
        scores.append(breakdown.total_score)
        margins.append(breakdown.stability_margin)
        pairs.append(breakdown.n_antipodal_pairs)
        
    return scores, margins, pairs

def plot_comparison(baseline_data, cem_data, metric_name, filename):
    """Create boxplot comparison."""
    plt.figure(figsize=(6, 5))
    
    data = [baseline_data, cem_data]
    labels = ['Random Baseline', 'CEM (Ours)']
    
    # Remove NaNs or Infs if any
    clean_data = []
    for d in data:
        arr = np.array(d)
        clean_data.append(arr[np.isfinite(arr)])
    
    plt.boxplot(clean_data, labels=labels, patch_artist=True,
                boxprops=dict(facecolor='#A6CEE3'),
                medianprops=dict(color='black'))
    
    plt.title(f"comparison: {metric_name}")
    plt.ylabel(metric_name)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300)
    plt.close()

def main():
    n_samples = 100
    
    # Run evaluations
    rand_scores, rand_margins, rand_pairs = evaluate_random_baseline(n=n_samples)
    cem_scores, cem_margins, cem_pairs = evaluate_cem_method(n=n_samples)
    
    # Plot comparisons
    plot_comparison(rand_scores, cem_scores, "Total Score", "evaluation_total_score.png")
    plot_comparison(rand_margins, cem_margins, "Stability Margin (m)", "evaluation_stability.png")
    plot_comparison(rand_pairs, cem_pairs, "Antipodal Grasp Pairs", "evaluation_graspability.png")
    
    print("Evaluation complete. Figures saved to images/")

if __name__ == "__main__":
    main()
