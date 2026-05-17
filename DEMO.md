# Running the pick-and-place demo

Visualises the full pipeline on a Franka Panda in RViz 2 and Gazebo, with the
same generated object that was scored in Table II of the paper.

## What you will see

- **Gazebo (left half of screen).** The generated object rests on a table in
  the `panda_eval_world`. The Panda itself is *not* in Gazebo — only the
  object — so the gz window shows the object physics.
- **RViz 2 (right half of screen).** The Panda's `RobotModel` display
  animates through a full pick-and-place sequence:
  *ready → pre-grasp → grasp → lift → transport → place → retract → ready*.
  While the object is "grasped" a marker travels with the gripper. Each
  stage's joint trajectory is computed by MoveIt 2 (RRTConnect via OMPL).

## Prerequisites

Same as `REPRODUCE.md` plus a display server. Build the workspace first:

```bash
cd ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select generated_objects_eval
source install/setup.bash
cd ..
```

Ensure a manifest exists (or build one):

```bash
python3 scripts/build_eval_manifest.py \
    --budget 1500 --top-k 25 --seed 42 \
    --out output/seed_42/eval_manifest.json
python3 scripts/patch_sdf_collision.py \
    --manifest output/seed_42/eval_manifest.json
```

## 1. Side-by-side demo (gz_sim + RViz + pick-and-place loop)

```bash
ros2 launch generated_objects_eval visual_demo.launch.py \
    manifest:=$(pwd)/output/seed_42/eval_manifest.json \
    method:=cem loop:=true
```

This brings up five processes:

1. `gz sim -r` on the `panda_eval_world` (windowed).
2. `robot_state_publisher` with the Panda URDF.
3. A static `world -> panda_link0` TF (the SRDF declares a `virtual_joint`;
   without this transform, RViz can't place the Panda and you would see an
   empty viewport).
4. `rviz2` with the bundled `demo.rviz` (RobotModel + scene markers + the
   grasped-object marker).
5. `demo_plan_driver`: spawns the CEM object in Gazebo, plans the 8-stage
   pick-and-place sequence, and replays each plan on `/joint_states` so the
   `RobotModel` in RViz animates end-to-end.

The driver loops indefinitely with `loop:=true`; stop with Ctrl-C.

## 2. Just Gazebo (drop & settle)

```bash
# Terminal A: world
ros2 launch generated_objects_eval stability_world_gui.launch.py

# Terminal B: stability eval
ros2 run generated_objects_eval gazebo_stability_eval \
    --manifest $(pwd)/output/seed_42/eval_manifest.json \
    --out /tmp/gz_video.json --max-objects 6
```

## 3. Archetype tour

```bash
ros2 launch generated_objects_eval stability_world_gui.launch.py

python3 scripts/archetype_tour.py \
    --manifest output/seed_42/eval_manifest.json \
    --methods cem fixed_cad cmaes --per-method 4 --settle-s 3.0
```

## Recording videos

### Option A — ffmpeg + x11grab

```bash
WIN=$(xdotool search --name "Gazebo" | head -1)
xdotool getwindowgeometry $WIN

ffmpeg -f x11grab -framerate 30 -video_size 1280x720 -i :0.0+100,80 \
       -c:v libx264 -pix_fmt yuv420p -crf 23 \
       docs/videos/gz_drop_settle.mp4
```

### Option B — OBS Studio

Add one *Window Capture* source per pane, arrange into a side-by-side scene,
hit Start Recording, then run the launch file.

### Option C — Gazebo's built-in recorder

`Settings → Video Recording → Start` in the `gz sim` window. Saves to
`~/.gz/sim/videos/`.

Drop the resulting mp4 files into `docs/videos/` with the names:

- `side_by_side.mp4`
- `rviz_motion_plan.mp4`
- `gz_drop_settle.mp4`
- `archetype_tour.mp4`

The project-page `index.html` wires those filenames in.

## Poster frames

```bash
cd docs && bash make_posters.sh
```

Extracts a t=1s frame from each mp4 into `static/images/poster_*.png`.

## Common gotchas

### RViz starts but the Panda doesn't render

- `ros2 topic echo /joint_states --once` — should print 9 joint positions.
- `ros2 topic echo /tf_static --once` — should include `world -> panda_link0`.
- If RViz Fixed Frame complains, set it to `panda_link0` (Displays → Global Options).

### NVIDIA `libEGL` warnings — `failed to create dri2 screen`

On hybrid-graphics laptops, gz_sim picks the wrong renderer:

```bash
# a) Force gz to render through the NVIDIA GPU
__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia \
ros2 launch generated_objects_eval visual_demo.launch.py manifest:=...

# b) Fall back to OGRE 1.x (no Vulkan/EGL needed)
#   edit launch/visual_demo.launch.py and replace
#     "--render-engine", "ogre2"  ->  "--render-engine", "ogre"

# c) Software rendering (always works, slow)
LIBGL_ALWAYS_SOFTWARE=1 ros2 launch generated_objects_eval visual_demo.launch.py manifest:=...
```

### "Missing virtual_joint" warning every second

The Panda SRDF declares a virtual_joint linking `world` to `panda_link0`;
until the static TF is published, MoveIt's planning_scene_monitor logs this.
Our launch starts `static_transform_publisher` automatically; if you see the
warning, that node failed to start (check
`~/.ros/log/*/static_transform_publisher*.log`).

### `CheckStartStateCollision' failed` in the planning log

The Panda is being evaluated at all-zero joints (self-collision). The driver
pins the start state to the SRDF `ready` group state; if you see this,
rebuild with `colcon build --packages-select generated_objects_eval`.

### Stale `gz sim` instances

```bash
pkill -KILL -f "gz sim"
pkill -KILL -f "gazebo_stability_eval"
pkill -KILL -f "moveit_planning_eval"
```
