# Reproducing the numbers

End-to-end, in under thirty minutes on a single CPU. Three independent stages
each produce part of Table II in the paper.

## 0. Prerequisites

- Ubuntu 24.04 (Noble) with **ROS 2 Jazzy** installed (Debian binaries from
  `packages.ros.org` work).
- The following Debian packages:

```bash
sudo apt install -y \
    ros-jazzy-moveit ros-jazzy-moveit-py \
    ros-jazzy-moveit-resources-panda-description \
    ros-jazzy-moveit-resources-panda-moveit-config \
    ros-jazzy-ros-gz-sim ros-jazzy-robot-state-publisher \
    ros-jazzy-tf2-ros latexdiff
```

- Python 3.12 packages:

```bash
pip install --user --break-system-packages trimesh pyyaml cma matplotlib
```

- ~1 GB of free disk for the per-seed manifest directories.

## 1. Clone and build

```bash
git clone https://github.com/xya22er/3d_generator.git
cd 3d_generator
cd ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select generated_objects_eval
source install/setup.bash
cd ..
```

## 2. Build a per-method manifest of objects + grasps

For each method (CEM, CMA-ES, GA, random search, fixed CAD), this trains under a
1500-evaluation budget, exports the top-K objects to URDF + SDF + meshes under
`output/seed_42/manifest_objects/`, and synthesises grasp candidates for each.
The SDF collision geometry is then rewritten to the visual mesh's AABB.

```bash
python3 scripts/build_eval_manifest.py \
    --budget 1500 --top-k 25 --seed 42 \
    --out output/seed_42/eval_manifest.json
python3 scripts/patch_sdf_collision.py \
    --manifest output/seed_42/eval_manifest.json
```

Repeat with `--seed 43` and `--seed 44` for the three-seed table.

**Scaling the number of generated objects.** Two knobs control this:

- `--budget N` — how many candidate objects each method *evaluates* before
  selecting elites. 1500 is the paper setting; raise it to 5000+ for a
  longer search at the cost of linear runtime.
- `--top-k K` — how many objects per method end up in the manifest (and
  therefore in the demo rotation, the MoveIt 2 evaluator, and the Gazebo
  stability evaluator). 25 is the paper setting (5 methods × 25 = 125
  objects per seed); use `--top-k 100` to evaluate the full top-100
  population per method (5 × 100 = 500 objects per seed).

For a completely *fresh* set of objects (no top-K filtering, no force-closure
sampler), use the standalone generator CLI:

```bash
# 10,000 CEM-generated objects in URDF+SDF
python3 src/main.py generate -n 10000 -o output/cem_10k --train --iterations 30
```

## 3. Python-level evaluation

Produces the suitability score, force-closure grasp success rate, feature
diversity, and Chamfer diversity per method.

```bash
python3 scripts/run_unified_eval.py \
    --budget 1500 --top-k 100 --seed 42 \
    --out output/seed_42/unified_eval.json
```

## 4. MoveIt 2 motion-planning success

Brings up `robot_state_publisher`, `home_joint_state_publisher`, a static
`world -> panda_link0` TF, and the `moveit_planning_eval` node.

```bash
source ros2_ws/install/setup.bash
ros2 launch generated_objects_eval moveit_planning_eval.launch.py \
    manifest:=$(pwd)/output/seed_42/eval_manifest.json \
    out:=$(pwd)/output/seed_42/moveit_results.json \
    max_objects:=0
```

> **Heads-up.** The `moveit_py` 2.12.x shutdown segfaults **after** the JSON
> is written. `scripts/run_multi_seed.sh` polls for the JSON and kills the
> launch immediately so the next stage can start; if you invoke the launch by
> hand, watch for `"Wrote 125 results to ..."` and Ctrl-C.

## 5. Gazebo dynamic-stability rate

```bash
# Terminal A: world
ros2 launch generated_objects_eval stability_world.launch.py

# Terminal B: eval
ros2 run generated_objects_eval gazebo_stability_eval \
    --manifest $(pwd)/output/seed_42/eval_manifest.json \
    --out $(pwd)/output/seed_42/gazebo_stability.json
```

## 6. All seeds in one command

```bash
bash scripts/run_multi_seed.sh 42 43 44
```

About 20 minutes per seed. Logs land in
`output/seed_<N>/{moveit,gazebo,gz_world}.log`. The script kills stale
`gz_sim` instances before every seed (the gz transport routes service calls
non-deterministically across same-world instances otherwise).

## 7. Aggregate

```bash
python3 scripts/aggregate_seeds.py 42 43 44 --out output/aggregated.json
cp output/aggregated.json docs/data/results.json   # refresh the project page
```

The aggregator prints a markdown table identical to Table II in the paper.

## 8. Rebuild the paper

```bash
cd papers/conferences/ICARM/_IEEE_ARM__*
pdflatex new.tex && bibtex new && pdflatex new.tex && pdflatex new.tex

# Camera-ready diff (manual \textcolor{blue}{} highlights on semantically new prose):
pdflatex diff_new.tex && bibtex diff_new && pdflatex diff_new.tex && pdflatex diff_new.tex

# Section IV workflow figure (Mermaid -> PNG via mermaid.ink):
bash ../../../../scripts/render_sec_iv_workflow.sh

# Twelve-archetype examples figure (matplotlib + trimesh):
python3 ../../../../scripts/render_examples_grid.py --label \
    --out images/generated_examples.png
```

## 9. Run the pick-and-place demo

See `DEMO.md` for the full walkthrough. Quick start:

```bash
source ros2_ws/install/setup.bash
ros2 launch generated_objects_eval visual_demo.launch.py use_gz_control:=true
```

This animates a random object from each method in turn, with the table
published as a CollisionObject so the planner avoids it.

## Reproduction tolerance

Reproduction tolerance is ±3% on every cell of Table II across seeds
42/43/44. If a number drifts by more than that, the most likely cause is one
of:

- **Stale `gz sim` instances** from a previous run stealing
  `/world/.../create` service calls. Run `pkill -KILL -f "gz sim"` and re-launch.
- **`CheckStartStateCollision' failed`** in the MoveIt log: the Panda is
  being evaluated at all-zero joints (self-collision). The launch uses
  `home_joint_state_publisher` + SRDF `set_start_state("ready")` to avoid
  this; if those got bypassed, restore them.
- **Mesh normals missing** in the exported collision OBJ. Run
  `patch_sdf_collision.py` to rewrite collisions as AABB primitives.
