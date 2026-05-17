# Recording the project-page videos

The four videos referenced by `3d_generator/docs/index.html` are:

| File                          | What it shows                          | How to capture                                          |
|-------------------------------|----------------------------------------|---------------------------------------------------------|
| `videos/gz_drop_settle.mp4`   | CEM objects falling onto the table     | `stability_world_gui.launch.py` + `archetype_tour.py`   |
| `videos/rviz_motion_plan.mp4` | MoveIt 2 trajectory animation in RViz  | `visual_demo.launch.py` (RViz half)                     |
| `videos/side_by_side.mp4`     | gz + RViz running together              | `visual_demo.launch.py` (full)                          |
| `videos/archetype_tour.mp4`   | Cycle through CEM/CMA-ES/CAD samples   | `archetype_tour.py`                                     |

All four scripts assume `eval_manifest.json` is on disk (produced by
`build_eval_manifest.py`).

## Recording option A — ffmpeg + x11grab (headful)

```bash
# Pick the gz_sim window or the RViz window
WIN=$(xdotool search --name "Gazebo" | head -1)
xdotool getwindowgeometry $WIN          # note position + size

ffmpeg -f x11grab -framerate 30 \
       -video_size 1280x720 -i :0.0+100,80 \
       -c:v libx264 -pix_fmt yuv420p -crf 23 \
       videos/gz_drop_settle.mp4
```

Stop with Ctrl-C when the take is done.

## Recording option B — OBS Studio

OBS gives per-source captures (gz window, RViz window, and a combined scene)
and a built-in record button. Add a *Window Capture* source per pane, set the
output to mp4, hit Start Recording, then run the launch file.

## Recording option C — Gazebo's built-in recorder

In the gz_sim window: Settings → Video Recording → Start. Saves to
`~/.gz/sim/videos/`. Useful only for the gz half; doesn't cover RViz.

## Full reproduction commands

```bash
cd ros2_ws && colcon build --packages-select generated_objects_eval
source install/setup.bash

# Make sure a fresh manifest exists.
python3 ../3d_generator/scripts/build_eval_manifest.py \
    --budget 1500 --top-k 25 --seed 42 \
    --out ../3d_generator/output/eval_manifest.json
python3 ../3d_generator/scripts/patch_sdf_collision.py \
    --manifest ../3d_generator/output/eval_manifest.json

# Video 1 — gz drop + settle (no RViz)
ros2 launch generated_objects_eval stability_world_gui.launch.py &
sleep 6
install/generated_objects_eval/bin/gazebo_stability_eval \
    --manifest ../3d_generator/output/eval_manifest.json \
    --out /tmp/gazebo_video.json --max-objects 6
# Stop recording, kill the launch when satisfied.

# Video 2 + 3 — side-by-side gz + RViz with MoveIt 2 plan loop
ros2 launch generated_objects_eval visual_demo.launch.py \
    manifest:=../3d_generator/output/eval_manifest.json \
    method:=cem loop:=true

# Video 4 — archetype tour (spawn 6 objects from 3 methods in sequence)
python3 ../3d_generator/scripts/archetype_tour.py \
    --manifest ../3d_generator/output/eval_manifest.json
```

After recording, drop the mp4 files into `3d_generator/docs/videos/` with the
filenames listed above; `index.html` already wires them up.
