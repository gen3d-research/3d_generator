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
