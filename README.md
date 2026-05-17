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

`REPRODUCE.md` walks through every cell of Table I end-to-end (≈ 30 minutes on a
single CPU). Quick reference:

```bash
# 1. Per-seed manifest (5 methods × top-25 objects each)
python3 scripts/build_eval_manifest.py --budget 1500 --top-k 25 --seed 42 \
    --out output/seed_42/eval_manifest.json
python3 scripts/patch_sdf_collision.py --manifest output/seed_42/eval_manifest.json

# 2. All three seeds, all three downstream metrics:
bash scripts/run_multi_seed.sh 42 43 44

# 3. Aggregate into the paper's table:
python3 scripts/aggregate_seeds.py 42 43 44 --out output/aggregated.json
cp output/aggregated.json docs/data/results.json    # refresh project page

# 4. 20-archetype large-scale study (long: ~1 h):
python3 scripts/run_scale_experiment.py -n 10000 --iterations 30 --train

# 5. The pick-and-place demo (gz_sim + RViz, full physics):
ros2 launch generated_objects_eval visual_demo.launch.py use_gz_control:=true
```

Two knobs control how many objects flow through the pipeline:

- `--top-k K` on `build_eval_manifest.py` — number of objects per method in
  the manifest (and therefore the demo + the downstream evaluators).
- `--budget N` on `build_eval_manifest.py` — number of candidate objects each
  method evaluates before selecting elites.

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
3d_generator/
├── src/                      # Core Python sources
│   ├── primitives.py            geometric primitives (Box, Cylinder, ...)
│   ├── scoring.py               constraint-based scoring functions
│   ├── cem.py                   Cross-Entropy Method optimizer
│   ├── export.py                URDF/SDF exporter
│   ├── generator.py             top-level generator class
│   ├── baselines.py             CMA-ES / GA / Random / Fixed-CAD baselines
│   ├── grasp_planner.py         force-closure grasp synthesiser
│   ├── diversity.py             diversity proxies
│   ├── archetype_cem.py         per-archetype CEM track
│   └── main.py                  CLI entry point
├── scripts/                  # Reproduction + analysis scripts
│   ├── build_eval_manifest.py
│   ├── run_unified_eval.py
│   ├── run_multi_seed.sh
│   ├── aggregate_seeds.py
│   ├── patch_sdf_collision.py
│   ├── emit_latex_table.py
│   ├── make_clean_diff.py
│   └── archetype_tour.py
├── ros2_ws/src/generated_objects_eval/   # ROS 2 evaluation package
├── docs/                                 # Project page (GitHub Pages)
├── papers/conferences/ICARM/...          # Manuscript (new.tex, diff_new.tex, refs)
├── README.md
├── REPRODUCE.md
├── DEMO.md
├── CHANGELOG.md
├── requirements.txt
└── reproduce_paper.sh
```

All Python scripts add `src/` to `sys.path` automatically; you can also
`pip install -e .` once a `pyproject.toml` is added.

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

The demo driver (`ros2_ws/src/generated_objects_eval/generated_objects_eval/demo_plan_driver.py`)
shows the recommended pattern under `moveit_py` 2.12.x: publish a
`moveit_msgs/PlanningScene` *diff* on the `/planning_scene` topic instead of
calling `moveit_py.PlanningSceneInterface.apply_collision_object` (which
segfaults in 2.12.x on Jazzy). The driver uses this to add the table as a
CollisionObject so RRTConnect plans around it.

```python
from moveit_msgs.msg import CollisionObject, PlanningScene, PlanningSceneWorld
from shape_msgs.msg import SolidPrimitive

co = CollisionObject()
co.header.frame_id = "panda_link0"
co.id = "demo_table"
co.operation = CollisionObject.ADD
prim = SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[0.8, 0.8, 0.4])
co.primitives = [prim]
co.primitive_poses = [table_pose]   # geometry_msgs/Pose at (0.5, 0, 0.2)

scene = PlanningScene(is_diff=True,
                      world=PlanningSceneWorld(collision_objects=[co]))
planning_scene_pub.publish(scene)   # TRANSIENT_LOCAL latched publisher
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
