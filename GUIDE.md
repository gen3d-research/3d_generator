# Guide — generate, visualize, and test parts

End-to-end walkthrough: **train the generator → generate parts → visualize them →
drop-test and pick-and-place them in ROS 2 / Gazebo.** For the full paper-number
reproduction see [`REPRODUCE.md`](REPRODUCE.md); for the recorded demo videos see
[`DEMO.md`](DEMO.md).

The library has **19 primitive types** and **105 hand-written archetypes**
(see [`docs/PRIMITIVES.md`](docs/PRIMITIVES.md) and [`docs/ROADMAP.md`](docs/ROADMAP.md)).

---

## 0. Setup

```bash
python3 -m venv ~/venv/3d_cem
source ~/venv/3d_cem/bin/activate
pip install -r requirements.txt          # numpy, scipy, trimesh, pyyaml, cma, matplotlib, manifold3d
```

Run every command below from the repo root (`3d_generator/`). The scripts add
`src/` to `sys.path` themselves. If ROS 2 is sourced in your shell, prefix the
pure-Python commands with `env -u PYTHONPATH` so its pytest/plugin paths don't leak in.

The ROS 2 parts (§5–§6) additionally need **ROS 2 Jazzy + Gazebo (gz)** and a one-time
`colcon build` (see §5).

---

## 1. Generate / train parts

The pipeline is **sample → score → CEM update → export**. No datasets, no GPU.

```bash
# Quick sanity check (brief training, 5 objects)
python src/main.py demo

# Train the CEM and generate 100 quality-filtered objects -> URDF/SDF/meshes/metadata
python src/main.py generate -n 100 -o output/objects \
    --train --iterations 30 --samples 100 --seed 42 \
    --save-generator output/trained_generator.json

# Export the whole hand-written archetype set
python src/main.py archetypes -o output/archetypes

# Score a single mesh (size / stability / graspability / complexity / validity)
python src/main.py score output/objects/object_0000/meshes/object_0000_visual.obj
```

Each generated object writes `object_XXXX.urdf`, `object_XXXX.sdf`,
`meshes/*_visual.obj`, `meshes/*_collision.obj`, and `*_metadata.yaml`
(mass + inertia from the union solid, randomized friction).

**Python API:**

```python
from generator import RoboticObjectGenerator
gen = RoboticObjectGenerator()
gen.train(verbose=True)
objects = gen.generate(n=50)              # connected, grasp-reranked, quality-filtered
gen.export_all(objects, "output/objects")
gen.save("output/trained_generator.json")
```

Knobs that matter (`GeneratorConfig`): `max_primitives` (default 16),
`require_connected`, `rerank_by_grasp`, `low_grasp_gate` (penalize ungraspable
cones/pyramids), `use_mesh_inertia`. `generator.paper_repro_generator()` returns a
v1/paper-configured generator (4 primitive types, no gate).

---

## 2. Visualize parts

All renderers are **headless** (matplotlib Agg) and write PNGs under
`docs/gallery/` (and copy a few to `docs/static/images/` for the website).

```bash
# (a) Primitive library — the 19 types, each with 3 variants showing its DOF
python scripts/render_primitives.py                 # -> docs/gallery/primitives.png
python scripts/render_primitives.py --emit-doc      # markdown spec table for the docs

# (b) Archetype gallery (all 105) + a 16-object trained-CEM sample sheet
python scripts/render_gallery.py                    # -> archetypes.png, v2_samples.png

# (c) Accepted / rejected / optimized sample galleries
python scripts/render_samples.py                    # -> samples_verdicts.png, samples_optimized.png

# (d) Generation STRATEGIES — systematic constrained sweeps
python scripts/gen_strategies.py --strategy all     # -> strategy_*.png
#   single  : 19 types x N=2..10 copies unioned (+ symmetric on prime N)
#   pairs   : every two-type pair (+ symmetric)
#   curved / faceted : palette-restricted batches
#   oneofeach : one connected union of all types
#   symmetric : bilaterally-symmetric batch
python scripts/gen_strategies.py --strategy single --copy-web   # just one strategy

# (e) CEM-trained parameter VARIETIES of representative archetypes
python scripts/render_archetype_varieties.py --subset --combined   # -> varieties_combined.png
python scripts/render_archetype_varieties.py --archetypes cup,jar,nut --k 12
```

Open the PNGs directly, or open `docs/index.html` — the **Shapes** and **Strategies**
sections embed the galleries.

---

## 3. Build a manifest (objects + grasps) for the sim tests

The ROS 2 drop/pick tests consume an **eval manifest**: per-method objects exported to
URDF/SDF with synthesized force-closure grasp candidates. This trains all five methods
(CEM + CMA-ES / GA / random / fixed-CAD baselines), exports the top-K of each, and
synthesizes grasps.

```bash
python scripts/build_eval_manifest.py \
    --budget 1500 --top-k 25 --seed 42 \
    --out output/seed_42/eval_manifest.json \
    --export-root output/seed_42/manifest_objects
python scripts/patch_sdf_collision.py \
    --manifest output/seed_42/eval_manifest.json     # stiff contact + high friction for grasping
```

`--top-k 25` → 5 methods × 25 = 125 objects. Raise `--budget` for a longer search,
`--top-k` for more objects per method. The pure-CPU **grasp success / diversity**
metrics come from:

```bash
python scripts/run_unified_eval.py --budget 1500 --top-k 100 --seed 42 \
    --out output/seed_42/unified_eval.json
```

---

## 4. Build the ROS 2 package (once)

```bash
cd ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select generated_objects_eval
source install/setup.bash
cd ..
```

Needs `ros-jazzy-moveit`, `ros-jazzy-moveit-py`, the Panda MoveIt config/description,
`ros-jazzy-ros-gz-sim`, and `gz_ros2_control` (see [`REPRODUCE.md`](REPRODUCE.md) §0).

---

## 5. Drop test (Gazebo dynamic stability)

Spawns each manifest object 5 cm above the table in `panda_eval_world`, lets physics
settle, then measures **vertical drift** and **tilt** — "stable" iff drift < 5 cm and
tilt < 25° (`config/eval_config.yaml`).

```bash
# Terminal A — the world (headless server). Use *_gui for a visible window.
ros2 launch generated_objects_eval stability_world.launch.py

# Terminal B — drop every object and score it
ros2 run generated_objects_eval gazebo_stability_eval \
    --manifest $(pwd)/output/seed_42/eval_manifest.json \
    --out $(pwd)/output/seed_42/gazebo_stability.json \
    --max-objects 25
```

It prints a per-method summary (`spawn_ok`, `stable`, `rate=%`) and writes per-object
`{spawn_ok, stable, tilt_deg, drift_m, final_pose}` to the JSON. To drop a **specific**
generated part, point `--manifest` at any manifest containing it (or hand-write a
one-entry manifest with its `sdf` path).

> Kill stale sims between runs: `pkill -KILL -f "gz sim"`. The gz transport routes
> `/world/.../create` non-deterministically if two same-world servers are alive.

---

## 6. Pick-and-place test (MoveIt 2 + Gazebo, with physics)

The Panda runs an **8-stage** pick-and-place under `gz_ros2_control`:
*ready → pre-grasp → grasp → lift → transport → place → retract*. The object is held by
**genuine finger friction** (effort-controlled gripper), the table is published as a
MoveIt `CollisionObject` so RRTConnect plans around it, and the object's **real
collision mesh** is added to the planning scene.

```bash
source ros2_ws/install/setup.bash
ros2 launch generated_objects_eval visual_demo.launch.py \
    manifest:=$(pwd)/output/seed_42/eval_manifest.json \
    method:=cem use_gz_control:=true loop:=true
```

- `use_gz_control:=true` → the arm actually moves in physics (omit for a faster
  RViz-only animation).
- `method:=cem|cmaes|ga|random_search|fixed_cad` → which method's objects to cycle.
- A different random object spawns each cycle; pin one with the driver's
  `--object-index N` / `--spawn-x/-y/-z`.
- `headless:=true` runs gz server-only; `render_engine:=ogre` for hybrid GPUs.

**Headless / batch MoveIt-plan success** (no window) — plans to every grasp pose and
records success rate:

```bash
ros2 launch generated_objects_eval moveit_planning_eval.launch.py \
    manifest:=$(pwd)/output/seed_42/eval_manifest.json \
    out:=$(pwd)/output/seed_42/moveit_results.json max_objects:=0
```

See [`DEMO.md`](DEMO.md) for recording the videos, RViz tips, and GPU/`libEGL` gotchas.

---

## 7. Full evaluation (all metrics, all seeds)

```bash
bash scripts/run_multi_seed.sh 42 43 44          # manifest + grasp + MoveIt + Gazebo per seed
python scripts/aggregate_seeds.py 42 43 44 --out output/aggregated.json
cp output/aggregated.json docs/data/results.json # refresh the project page table
```

`aggregate_seeds.py` prints a markdown table (force-closure grasp %, MoveIt plan %,
Gazebo stable %, diversity) with 95% CIs and a CEM-vs-best significance test. Full
details + tolerances in [`REPRODUCE.md`](REPRODUCE.md).
