# generated_objects_eval

Downstream evaluation of objects produced by the **3d_generator** pipeline
on a Franka Panda using MoveIt 2 and Gazebo (gz_sim). Built to answer the
independent-downstream-metric requests from the ICARM 2026 reviewers
(R1, R2-Q1 / Q2 / Q3, and the Associate Editor).

## What this package adds to the paper

| Metric                              | Source                          |
|-------------------------------------|---------------------------------|
| Force-closure grasp success rate    | `3d_generator/grasp_planner.py` |
| MoveIt 2 motion-planning success    | `moveit_planning_eval` node     |
| Dynamic stability under physics     | `gazebo_stability_eval` node    |
| Shape diversity (feature + Chamfer) | `3d_generator/diversity.py`     |

All four metrics share a single manifest produced by
`3d_generator/scripts/build_eval_manifest.py`, which exports each generator
method's top-K objects to URDF + SDF + meshes and synthesises grasp
candidates per object up-front. The ROS 2 nodes consume that manifest.

## End-to-end run

From the workspace root:

```bash
# 1. Train the generator and produce the per-method object manifest.
python3 3d_generator/scripts/build_eval_manifest.py \
    --budget 1000 --top-k 25 --seed 42 \
    --out 3d_generator/output/eval_manifest.json

# 2. Build the ROS 2 workspace.
cd ros2_ws && colcon build --packages-select generated_objects_eval

# 3. MoveIt 2 motion-planning success rate.
source install/setup.bash
ros2 launch generated_objects_eval moveit_planning_eval.launch.py \
    manifest:=$PWD/../3d_generator/output/eval_manifest.json \
    out:=$PWD/../3d_generator/output/moveit_results.json \
    max_objects:=25

# 4. Gazebo dynamic-stability rate.
ros2 launch generated_objects_eval stability_world.launch.py &
ros2 run generated_objects_eval gazebo_stability_eval \
    --manifest $PWD/../3d_generator/output/eval_manifest.json \
    --out $PWD/../3d_generator/output/gazebo_stability.json
```

The MoveIt launch file starts `robot_state_publisher` and
`joint_state_publisher` alongside the evaluator so that the MoveIt 2
planning-scene monitor receives `/joint_states` at startup — without those
auxiliaries the monitor blocks waiting for an external robot driver.

## Physical pick-and-place demo (the arm grasps spawned objects)

The Panda physically picks objects that spawn in front of it, in gz_sim with
`gz_ros2_control`. The grasp is **genuine** — the fingers squeeze the object with
a constant force and hold it by **friction** (no weld/cheat). Each loop cycle
spawns a different generated object on the table and runs the full pick:
pre-grasp → grasp → close (squeeze) → lift → transport → place → release →
retract. A closed-loop **lift check** confirms the object actually rose; if the
friction grasp slips, the driver opens and retries the next ranked candidate.

```bash
# 1. Generated objects + grasp candidates (mesh collision + firm friction).
python3 3d_generator/scripts/build_eval_manifest.py \
    --methods cem --top-k 6 --out 3d_generator/output/demo_manifest.json
python3 3d_generator/scripts/patch_sdf_collision.py \
    --manifest 3d_generator/output/demo_manifest.json

# 2. Build + source.
cd ros2_ws && colcon build --packages-select generated_objects_eval
source install/setup.bash

# 3. Run the demo: gz_sim (Panda + object) + RViz + MoveIt, looping picks.
ros2 launch generated_objects_eval visual_demo.launch.py \
    manifest:=$PWD/../3d_generator/output/demo_manifest.json \
    method:=cem use_gz_control:=true loop:=true
```

`use_gz_control:=true` is what makes the arm move in physics (spawns the Panda
with `gz_ros2_control` + `joint_state_broadcaster` + `panda_arm_controller` +
`panda_hand_controller`, and the driver runs with `--execute`). Without it the
launch is an RViz-only animation. Verified headless end-to-end: the gripper
gently stalls on the object and "lift check GRASPED (held by friction):
z 0.40 -> 0.51" — it rises ~11 cm and completes the full pick→place→release with
no weld.

What makes the genuine grasp work (all handled by the launch/driver now):

* **Effort (force) finger control.** The fingers use an `effort` command
  interface driven by a `forward_command_controller/ForwardCommandController`;
  the driver publishes a constant **negative** force (`-50 N`, the no-eject
  sweet spot) to `/panda_hand_controller/commands` to squeeze, held through
  lift/transport so friction retains the object. Position control could only
  slip or eject. (mu=30 finger+object friction; finger effort limit 150 N;
  close velocity 0.04 m/s for symmetric contact.)
* **Correct grasp pose.** Plan `panda_link8` with `TIP_OFFSET=0.1034` (fingertip
  TCP, not the wrist), orient the finger axis to the antipodal grasp line, and
  anchor targets to the object's **settled** pose (it falls ~5 cm before
  resting).
* **Robust execution.** Plan from the **current** state, loosened
  `trajectory_execution.allowed_start_tolerance` (0.1 rad), a controller
  warm-up, **absolute** manifest paths, and the driver not publishing
  `/joint_states` in execute mode (the broadcaster owns it).
* **RViz reflects reality.** The object marker is driven by the object's real gz
  pose; the arm reflects the broadcaster `/joint_states`.

Grasp success is **shape-dependent** (the inherent reality of friction grasping
in sim): graspable objects are picked reliably; some shapes slip and the driver
retries / moves on. There is no weld — failures are honest.

## Reproducing manuscript numbers

To reproduce the headline numbers in the revised paper (Section V-D) use
`--top-k 50` and `--budget 3000`; results are averaged over 3 seeds:
`--seed 42`, `--seed 43`, `--seed 44`.

## Caveats

* The Panda URDF is loaded with `ros2_control_hardware_type=mock_components`
  so it can be parsed by the standard `urdf_xml_parser` (the alternative
  hardware values inject elements that the parser silently rejects).
* The MoveIt 2 evaluator uses `moveit_py` headless. If the planning-scene
  monitor still hangs at startup, verify that
  `/joint_states` is actually being published (`ros2 topic echo /joint_states
  --once`); a frozen joint_state_publisher will keep the monitor blocked.
