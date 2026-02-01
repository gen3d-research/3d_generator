
import sys
import os
sys.path.append(os.getcwd())

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict

sys.path.append(str(Path.cwd() / "3d_generator"))

from primitives import (
    create_simple_box, create_mug_like, create_l_shape, 
    create_dumbbell, create_hammer, create_bottle,
    create_t_shape, create_u_shape, create_v_shape, create_monitor,
    create_barbell, create_snowman, create_camera, create_frying_pan,
    create_flashlight, create_spatula, create_remote, create_joystick
)

from archetype_cem import ArchetypeTrainer
from scoring import ObjectScorer
from export import URDFExporter

OUTPUT_DIR = Path("3d_generator/output/scale_experiment")
IMG_DIR = Path("images")

ARCHETYPE_FUNCS = [
    create_simple_box, create_mug_like, create_l_shape, 
    create_dumbbell, create_hammer, create_bottle,
    create_t_shape, create_u_shape, create_v_shape, create_monitor,
    create_barbell, create_snowman, create_camera, create_frying_pan,
    create_flashlight, create_spatula, create_remote, create_joystick
]

def run_experiment(n_objects: int, iterations: int, train: bool):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    
    all_scores = {}
    
    for func in ARCHETYPE_FUNCS:
        name = func.__name__.replace('create_', '')
        print(f"\nProcessing Archetype: {name}")
        
        trainer = ArchetypeTrainer(func)
        
        if train:
            trainer.train(iterations=iterations, samples_per_iter=100)
        
        print(f"Generating {n_objects} objects...")
        objects = trainer.generate(n_objects)
        
        # Evaluate
        print("Scoring...")
        scorer = ObjectScorer()
        scores = [scorer.score(obj).total_score for obj in objects]
        all_scores[name] = scores
        
        # Determine archetype specific subfolder
        arch_dir = OUTPUT_DIR / name
        arch_dir.mkdir(exist_ok=True)
        
        # Export (Optional? User said generate. 10k is a lot. Maybe export first 100?)
        # Exporting 10k takes time.
        # Let's export 100 sample URDFs and just save stats for the rest?
        # User prompt: "generate 10000 ... objects"
        # I'll randomly sub-sample 50 to export to save time/disk, unless forced.
        # Check an arg?
        num_export = min(n_objects, 50) 
        print(f"Exporting sample of {num_export} objects to {arch_dir}...")
        
        # from export import URDFExporter (already imported at top)

        exporter = URDFExporter()
        for i in range(num_export):
            exporter.export(objects[i], arch_dir / objects[i].name, objects[i].name)

    # Plotting
    print("\nGenerating Diagrams...")
    plt.figure(figsize=(15, 8))
    
    names = list(all_scores.keys())
    data = list(all_scores.values())
    
    plt.boxplot(data, labels=names, patch_artist=True)
    plt.xticks(rotation=45, ha='right')
    plt.title(f'Score Distribution by Archetype (N={n_objects}, Trained={train})')
    plt.ylabel('Suitability Score')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plot_path = IMG_DIR / f"archetype_comparison_N{n_objects}.png"
    plt.savefig(plot_path)
    print(f"Saved comparison plot to {plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=100, help="Number of objects per archetype")
    parser.add_argument("--iterations", type=int, default=20, help="Training iterations")
    parser.add_argument("--train", action="store_true", help="Enable training")
    args = parser.parse_args()
    
    run_experiment(args.n, args.iterations, args.train)
