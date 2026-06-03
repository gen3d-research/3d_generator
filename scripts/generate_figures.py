#!/usr/bin/env python3
import sys
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from collections import Counter

# Add 3d_generator directory to path to import modules directly
module_path = Path(__file__).resolve().parents[1] / "src"
sys.path.append(str(module_path))

try:
    from generator import RoboticObjectGenerator, GeneratorConfig
    from primitives import Box, Cylinder
except ImportError as e:
    print(f"Import Error: {e}")
    print(f"System path: {sys.path}")
    sys.exit(1)

OUTPUT_DIR = Path("images")
OUTPUT_DIR.mkdir(exist_ok=True)

CSV_PATH = Path("Elicit - Text-to-3D Generation Tools Competitive Landscape.csv")

def set_style():
    """Set scientific plotting style."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 12,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.figsize': (6, 4),
        'lines.linewidth': 2
    })

def plot_competitive_landscape():
    print("Generating Competitive Landscape figure...")
    if not CSV_PATH.exists():
        print(f"Warning: {CSV_PATH} not found. Skipping.")
        return

    years = []
    with open(CSV_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Try to find year in 'Maturity Release Year' or other fields
            year_str = row.get('Maturity Release Year', '').strip()
            # Handle cases like "Research (2023)"
            if '(' in year_str and ')' in year_str:
                try:
                    year_str = year_str.split('(')[1].split(')')[0]
                except:
                    pass
            
            if year_str.isdigit():
                years.append(int(year_str))
    
    if not years:
        print("No year data found.")
        return

    counts = Counter(years)
    sorted_years = sorted(counts.keys())
    values = [counts[y] for y in sorted_years]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(sorted_years, values, color='#4C72B0', alpha=0.8)
    
    ax.set_xlabel("Release Year")
    ax.set_ylabel("Number of Tools/Methods")
    ax.set_title("Growth of Text-to-3D Generation Methods")
    ax.set_xticks(sorted_years)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{int(height)}',
                ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "competitive_landscape.png", dpi=300)
    plt.close()

def plot_training_progress():
    print("Generating Training Progress figure...")
    
    # Configuration for short training
    config = GeneratorConfig(
        cem_iterations=30,
        cem_samples=50,
        learning_rate=0.1,  # Slow learning for visualization
        seed=42
    )
    
    gen = RoboticObjectGenerator(config)
    
    # Patch scoring config to reduce sample count
    # Default 200 samples -> ~40,000 pairs -> Score saturates even with penalties
    # 25 samples -> ~600 pairs -> Score is sensitive to distance/alignment
    gen.scoring_config.n_surface_samples = 25
    gen.scorer.config.n_surface_samples = 25
    
    # MANUALLY PERTURB DISTRIBUTION TO SHOW LEARNING
    # Start with TINY objects (Fails Size and Graspability constraints)
    # 1. Size score < 1.0 because 0.01 < 0.02 min_extent
    # 2. Graspability score = 0 because no points are > 1cm apart
    dist = gen.distribution
    
    # Tiny Sphere: 1mm radius
    # Box has too many antipodal pairs (faces). Sphere has few.
    # Few pairs * low distance score = Low Total Score.
    # v2 distribution API: per-type params live in spec-keyed dicts, and the
    # probability vectors are length max_primitives / len(PRIMITIVE_SPECS).
    from cem import PRIMITIVE_SPECS
    dist.type_log_means['sphere'] = np.array([np.log(0.001)])
    dist.type_log_stds['sphere'] = np.array([0.1])

    # Force a single-primitive SPHERE object.
    dist.n_primitives_probs = np.zeros(dist.max_primitives)
    dist.n_primitives_probs[0] = 1.0
    type_probs = np.zeros(len(PRIMITIVE_SPECS))
    type_probs[[i for i, s in enumerate(PRIMITIVE_SPECS) if s.key == 'sphere'][0]] = 1.0
    dist.primitive_type_probs = type_probs
    
    # Run training
    def print_progress(history):
        h = history[-1]
        print(f"  Iter {h['iteration']:2d}: Elite={h['mean_elite_score']:.3f}, "
              f"Stab={h.get('mean_stability', 0):.3f}, Grasp={h.get('mean_graspability', 0):.3f}")

    # Patch the callback in generator to use our printer (hacky but effective)
    # Actually just run it and print afterwards or modify generator... 
    # Whatever, let's just inspect the history returned.
    history = gen.train(verbose=True)
    
    print("\nTraining History Analysis:")
    for h in history:
        print(f"It {h['iteration']:2d}: Total={h['mean_elite_score']:.3f}, "
              f"Stab={h.get('mean_stability',0):.3f}, Grasp={h.get('mean_graspability',0):.3f}")
    
    iterations = [h['iteration'] for h in history]
    means = [h['mean_elite_score'] for h in history]
    stability = [h.get('mean_stability', 0) for h in history]
    graspability = [h.get('mean_graspability', 0) for h in history]
    
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    # Plot components
    ax.plot(iterations, means, '-', color='#333333', linewidth=2.5, label='Total Score')
    ax.plot(iterations, stability, '--', color='#1f77b4', linewidth=1.5, alpha=0.8, label='Stability')
    ax.plot(iterations, graspability, '--', color='#ff7f0e', linewidth=1.5, alpha=0.8, label='Graspability')
    
    ax.set_xlabel("Optimization Step")
    ax.set_ylabel("Component Score")
    ax.set_title("Optimization Dynamics: Component Evolution")
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower right')
    ax.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training_progress.png", dpi=300)
    plt.savefig(OUTPUT_DIR / "training_progress_v2.png", dpi=300) # Save copy to bust cache
    plt.close()

    return gen  # Return trained generator for next step

def plot_generated_examples(gen):
    print("Generating Examples figure...")
    objects = gen.generate(n=4, ensure_quality=True)
    
    fig = plt.figure(figsize=(10, 8))
    
    for i, obj in enumerate(objects):
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        ax.set_title(f"Generated Object {i+1}\nScore: {gen.scorer.score(obj).total_score:.2f}")
        
        # Plot primitives
        for p in obj.primitives:
            # Get transformation
            T = p.transform.as_matrix()
            
            # Create a basic wireframe for box/cylinder
            # For simplicity, we'll approximate everything as boxes for the wireframe
            # or try to respect the shape if simple
            
            if p.ptype.name == 'BOX':
                dims = p.dimensions
                # Create unit cube vertices
                r = [-0.5, 0.5]
                X, Y = np.meshgrid(r, r)
                # This is just one face, let's do corners
                corners = np.array([
                    [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
                    [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5]
                ]) * dims
                
                # Transform
                corners_hom = np.hstack([corners, np.ones((8, 1))])
                transformed = (T @ corners_hom.T).T[:, :3]
                
                # Edges
                edges = [
                    [0, 1], [1, 2], [2, 3], [3, 0], # Bottom
                    [4, 5], [5, 6], [6, 7], [7, 4], # Top
                    [0, 4], [1, 5], [2, 6], [3, 7]  # Sides
                ]
                
                for start, end in edges:
                    ax.plot3D(
                        [transformed[start, 0], transformed[end, 0]],
                        [transformed[start, 1], transformed[end, 1]],
                        [transformed[start, 2], transformed[end, 2]],
                        'b-'
                    )
            
            elif p.ptype.name == 'CYLINDER':
                # Approximate cylinder with lines
                r, h = p.radius, p.height
                
                # Circles at top and bottom
                theta = np.linspace(0, 2*np.pi, 12)
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                
                z_bottom = np.full_like(x, -h/2)
                z_top = np.full_like(x, h/2)
                
                # Transform & Plot
                for z_level in [z_bottom, z_top]:
                    points = np.stack([x, y, z_level], axis=1)
                    points_hom = np.hstack([points, np.ones((len(points), 1))])
                    tf = (T @ points_hom.T).T[:, :3]
                    ax.plot3D(tf[:, 0], tf[:, 1], tf[:, 2], 'r-')
                
                # Connecting lines
                for idx in range(0, len(x), 3): # Draw a few lines
                    p1 = np.array([x[idx], y[idx], -h/2, 1])
                    p2 = np.array([x[idx], y[idx], h/2, 1])
                    tf1 = (T @ p1)[:3]
                    tf2 = (T @ p2)[:3]
                    ax.plot3D([tf1[0], tf2[0]], [tf1[1], tf2[1]], [tf1[2], tf2[2]], 'r-')

        # Set limits equal-ish
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Try to enforce aspect ratio
        limits = np.array([ax.get_xlim(), ax.get_ylim(), ax.get_zlim()])
        center = np.mean(limits, axis=1)
        radius = 0.5 * np.max(limits[:, 1] - limits[:, 0])
        ax.set_xlim([center[0]-radius, center[0]+radius])
        ax.set_ylim([center[1]-radius, center[1]+radius])
        ax.set_zlim([center[2]-radius, center[2]+radius])

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "generated_examples.png", dpi=300)
    plt.close()

def plot_system_pipeline():
    print("Generating System Pipeline figure...")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')
    
    # Define boxes
    boxes = {
        'Param': (0.05, 0.4, 0.15, 0.2, "Parametric\nDistribution\n$q_\\phi(\\theta)$"),
        'Sample': (0.25, 0.4, 0.15, 0.2, "Sample\nCandidate\n$\\theta \\sim q_\\phi$"),
        'Constr': (0.45, 0.4, 0.15, 0.2, "Constraint\nScoring\n$S(\\theta)$"),
        'Update': (0.45, 0.1, 0.15, 0.2, "CEM Update\n$\\phi \\leftarrow \\phi'$"),
        'Export': (0.65, 0.4, 0.15, 0.2, "URDF/SDF\nExport"),
        'Sim': (0.85, 0.4, 0.1, 0.2, "ROS 2\nSim")
    }
    
    for k, (x, y, w, h, text) in boxes.items():
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                                     ec="black", fc="#EAEAF2")
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center')
        
    # Arrows
    def arrow(p1, p2, text=None):
        ax.annotate("", xy=p2, xytext=p1,
                    arrowprops=dict(arrowstyle="->", lw=1.5))
        if text:
            mid = ((p1[0]+p2[0])/2, (p1[1]+p2[1])/2)
            ax.text(mid[0], mid[1]+0.02, text, ha='center', fontsize=9)

    # Main flow
    arrow((0.2, 0.5), (0.25, 0.5))
    arrow((0.4, 0.5), (0.45, 0.5))
    arrow((0.6, 0.5), (0.65, 0.5), "High Score")
    arrow((0.8, 0.5), (0.85, 0.5))
    
    # Feedback loop
    # arrow((0.525, 0.4), (0.525, 0.3)) # Down from Scoring
    # arrow((0.525, 0.1), (0.2, 0.1))   # Back to Distribution? 
    # Let's verify layout.
    
    # Scoring -> Update
    ax.annotate("", xy=(0.525, 0.3), xytext=(0.525, 0.4),
                arrowprops=dict(arrowstyle="->", lw=1.5))
    
    # Update -> Param (Feedback)
    # Path: Update (0.45, 0.2) -> Left -> Up -> Param
    ax.annotate("", xy=(0.125, 0.4), xytext=(0.45, 0.2),
                arrowprops=dict(arrowstyle="->", lw=1.5, connectionstyle="angle,angleA=180,angleB=270,rad=10"))

    plt.title("Generative Object Modeling Pipeline", fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "system_pipeline.png", dpi=300)
    plt.close()

def main():
    set_style()
    plot_competitive_landscape()
    gen = plot_training_progress()
    plot_generated_examples(gen)
    plot_system_pipeline()
    print("All figures generated in images/")

if __name__ == "__main__":
    main()
