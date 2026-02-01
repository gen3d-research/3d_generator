# Robotic Object Generator

**Generative 3D Object Modeling for Robust Robot Manipulation in ROS 2**

A lightweight, dataset-free framework for generating diverse 3D objects suitable for robotic manipulation. Objects are scored on robotics-relevant criteria (stability, graspability, size) and the generator learns to produce high-quality objects using the Cross-Entropy Method.

## Key Features

- **No external datasets required** — self-contained training loop
- **No neural networks/LLMs** — pure optimization (fast, interpretable)
- **Robotics-first scoring** — stability, graspability, feasible size
- **Direct ROS 2 integration** — exports URDF/SDF with physical properties
- **Lightweight** — runs on CPU, no GPU needed

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Python API

```python
from generator import RoboticObjectGenerator

# Create generator and train
gen = RoboticObjectGenerator()
gen.train(n_iterations=30, verbose=True)

# Generate objects
objects = gen.generate(n=50)

# Export to URDF/SDF for Gazebo/MoveIt
gen.export_all(objects, "output/objects")

# Save trained generator for reuse
gen.save("trained_generator.json")
```

### Command Line

```bash
# Generate 100 objects with training
python main.py generate -n 100 -o output/objects --train --iterations 50

# Quick demo
python main.py demo

# Generate standard archetype objects
python main.py archetypes -o output/archetypes
```

### Reproducing Paper Results

To generate the figures and evaluations presented in the paper:

```bash
# Generate figures and examples
python scripts/generate_figures.py

# Run baseline comparison
python scripts/evaluate_methods.py

# Run large-scale archetype experiment (20 archetypes, 10k objects each)
python scripts/run_scale_experiment.py -n 10000 --iterations 30 --train

# Run baseline comparison (CEM vs Random)
python scripts/evaluate_methods.py

# Run ablation study on stability
python scripts/run_ablation.py

# OR simply run the unified reproduction script:
./scripts/reproduce_paper.sh

```

## How It Works

### 1. Parametric Object Representation
Objects are compositions of simple primitives (boxes, cylinders, spheres, capsules) with rigid transforms. This keeps generation fast and interpretable.

### 2. Constraint-Based Scoring
Each object is scored on manipulation-relevant criteria:

| Criterion | Description |
|-----------|-------------|
| **Size feasibility** | Fits gripper workspace (2-15 cm) |
| **Static stability** | COM within support polygon |
| **Graspability** | Has antipodal surface pairs for parallel-jaw grasping |
| **Mesh validity** | Watertight, no degeneracies |
| **Complexity** | Bounded triangle count for fast collision checking |

### 3. Cross-Entropy Method (CEM)
The generator learns without external data:
1. Sample N candidate objects from current distribution
2. Score each on manipulation criteria
3. Keep top 20% (elite samples)
4. Update distribution to maximize likelihood of elites
5. Repeat

After ~30-50 iterations, the generator produces consistently high-quality objects.

### 4. URDF/SDF Export
Generated objects are exported with:
- Visual mesh (OBJ/STL)
- Simplified collision mesh
- Mass and inertia tensor (computed from shape + density)
- Surface properties (friction, restitution) - Randomized friction coefficients ($0.1$ to $2.0$)
- Ready for Gazebo simulation and MoveIt planning

## Project Structure

```
robotic_object_generator/
├── primitives.py      # Geometric primitives (Box, Cylinder, etc.)
├── scoring.py         # Constraint-based scoring functions
├── cem.py             # Cross-Entropy Method optimizer
├── export.py          # URDF/SDF export
├── generator.py       # Main generator class
├── main.py            # CLI entry point
└── requirements.txt
```

## Output Format

Each generated object produces:
```
object_0000/
├── object_0000.urdf           # Robot description format
├── object_0000.sdf            # Gazebo model format
├── object_0000_metadata.yaml  # Properties and scores
└── meshes/
    ├── object_0000_visual.obj
    └── object_0000_collision.obj
```

## Integration with ROS 2

### Loading in Gazebo

```xml
<!-- In your launch file -->
<include file="$(find gazebo_ros)/launch/gazebo.launch.py">
  <arg name="world" value="empty.world"/>
</include>

<node pkg="gazebo_ros" type="spawn_entity.py" name="spawn_object"
      args="-file $(find your_pkg)/objects/object_0000/object_0000.sdf 
            -entity object_0000 -x 0.5 -y 0 -z 0.8"/>
```

### Using with MoveIt 2

```python
# Add to planning scene
from moveit_msgs.msg import CollisionObject
from shape_msgs.msg import Mesh

# Load mesh and create collision object
collision_object = CollisionObject()
collision_object.id = "generated_object"
collision_object.meshes = [load_mesh("object_0000_collision.obj")]
collision_object.mesh_poses = [pose]

planning_scene.add_collision_object(collision_object)
```

## Scoring Details

### Graspability Proxy
We estimate parallel-jaw graspability without a grasp database by counting antipodal surface pairs:
- Sample points on surface with normals
- Find pairs where: normals nearly opposite (n₁·n₂ < -0.9) AND distance within gripper range
- More valid pairs → higher graspability score

### Stability Proxy  
- Compute support polygon (convex hull of ground-contact vertices)
- Project center of mass onto ground plane
- Score = distance from COM to polygon boundary (larger = more stable)

## Citation

If you use this in research, please cite:
```bibtex
@inproceedings{generative3d2025,
  title={Generative 3D Object Modeling for Robust Robot Manipulation in ROS 2},
  author={...},
  year={2025}
}
```

## License

MIT License
