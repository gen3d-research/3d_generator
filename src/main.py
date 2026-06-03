#!/usr/bin/env python3
"""
Command-line interface for robotic object generation.

Examples:
    # Generate 10 objects with default settings
    python main.py generate -n 10 -o output/objects
    
    # Train generator first, then generate
    python main.py generate -n 100 -o output/objects --train --iterations 50
    
    # Generate archetype set for testing
    python main.py archetypes -o output/archetypes
    
    # Score an existing mesh
    python main.py score path/to/mesh.obj
"""

import argparse
import sys
from pathlib import Path


def cmd_generate(args):
    """Generate objects command."""
    from generator import RoboticObjectGenerator, GeneratorConfig
    
    config = GeneratorConfig(
        cem_iterations=args.iterations,
        cem_samples=args.samples,
        seed=args.seed
    )
    
    gen = RoboticObjectGenerator(config)
    
    if args.train:
        print(f"Training generator ({args.iterations} iterations, {args.samples} samples/iter)...")
        gen.train(verbose=True)
        print()
    
    print(f"Generating {args.n} objects...")
    objects = gen.generate(args.n, ensure_quality=not args.no_filter)
    print(f"Generated {len(objects)} objects")
    
    if args.output:
        print(f"\nExporting to {args.output}...")
        results = gen.export_all(objects, args.output, args.prefix)
        print(f"Exported {len(results['urdf'])} URDF files")
    
    if args.save_generator:
        gen.save(args.save_generator)
        print(f"Saved generator to {args.save_generator}")
    
    # Print summary stats
    from scoring import ObjectScorer
    scorer = ObjectScorer()
    scores = [scorer.score(obj).total_score for obj in objects]
    
    print(f"\nScore statistics:")
    print(f"  Mean:  {sum(scores)/len(scores):.4f}")
    print(f"  Min:   {min(scores):.4f}")
    print(f"  Max:   {max(scores):.4f}")


def cmd_archetypes(args):
    """Generate archetype objects command."""
    from generator import create_archetype_set
    
    print(f"Creating archetype set in {args.output}...")
    results = create_archetype_set(args.output)
    
    print(f"Created {len(results)} archetypes:")
    for name, path in results.items():
        print(f"  {name}: {path}")


def cmd_score(args):
    """Score a mesh file command."""
    import trimesh
    from primitives import CompositeObject, Box, Transform
    from scoring import ObjectScorer
    
    print(f"Loading mesh: {args.mesh}")
    mesh = trimesh.load(args.mesh)
    
    # Wrap mesh in a dummy CompositeObject
    # (This is a hack - ideally we'd have a mesh-based primitive)
    # For now, approximate with bounding box
    extents = mesh.bounding_box.extents
    obj = CompositeObject(
        primitives=[Box(dimensions=extents, transform=Transform())],
        name=Path(args.mesh).stem
    )
    
    scorer = ObjectScorer()
    breakdown = scorer.score(obj)
    
    print(f"\nScore breakdown:")
    print(f"  Size:        {breakdown.size_score:.4f}")
    print(f"  Stability:   {breakdown.stability_score:.4f}")
    print(f"  Graspability:{breakdown.graspability_score:.4f}")
    print(f"  Complexity:  {breakdown.complexity_score:.4f}")
    print(f"  Validity:    {breakdown.validity_score:.4f}")
    print(f"  Assembly:    {breakdown.assembly_score:.4f}")
    print(f"  ─────────────────────")
    print(f"  TOTAL:       {breakdown.total_score:.4f}")
    print(f"\nDiagnostics:")
    print(f"  AABB extents: {breakdown.aabb_extents}")
    print(f"  Stability margin: {breakdown.stability_margin:.4f}")
    print(f"  Antipodal pairs: {breakdown.n_antipodal_pairs}")
    print(f"  Triangles: {breakdown.n_triangles}")
    print(f"  Watertight: {breakdown.is_watertight}")


def cmd_demo(args):
    """Quick demo command."""
    from generator import quick_generate
    from scoring import ObjectScorer
    
    print("Running quick demo...")
    print("Generating 5 objects with brief training...\n")
    
    objects = quick_generate(n=5, train_first=True)
    
    scorer = ObjectScorer()
    for obj in objects:
        score = scorer.score(obj)
        print(f"{obj.name}:")
        print(f"  Primitives: {len(obj.primitives)}")
        print(f"  Score: {score.total_score:.4f}")
        print(f"  Extents: {score.aabb_extents}")
        print()
    
    print("Demo complete! Use 'generate' command for full generation.")


def main():
    parser = argparse.ArgumentParser(
        description="Generate 3D objects for robotic manipulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate objects')
    gen_parser.add_argument('-n', type=int, default=10, help='Number of objects')
    gen_parser.add_argument('-o', '--output', type=str, help='Output directory')
    gen_parser.add_argument('--prefix', type=str, default='object', help='Name prefix')
    gen_parser.add_argument('--train', action='store_true', help='Train generator first')
    gen_parser.add_argument('--iterations', type=int, default=30, help='CEM iterations')
    gen_parser.add_argument('--samples', type=int, default=100, help='Samples per iteration')
    gen_parser.add_argument('--seed', type=int, default=42, help='Random seed')
    gen_parser.add_argument('--no-filter', action='store_true', help='Disable quality filtering')
    gen_parser.add_argument('--save-generator', type=str, help='Save trained generator to file')
    gen_parser.set_defaults(func=cmd_generate)
    
    # Archetypes command
    arch_parser = subparsers.add_parser('archetypes', help='Generate archetype set')
    arch_parser.add_argument('-o', '--output', type=str, required=True, help='Output directory')
    arch_parser.set_defaults(func=cmd_archetypes)
    
    # Score command
    score_parser = subparsers.add_parser('score', help='Score a mesh file')
    score_parser.add_argument('mesh', type=str, help='Path to mesh file')
    score_parser.set_defaults(func=cmd_score)
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Quick demo')
    demo_parser.set_defaults(func=cmd_demo)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == '__main__':
    main()
