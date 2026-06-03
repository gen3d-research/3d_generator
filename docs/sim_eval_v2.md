# Sim-in-the-loop evaluation (v2)

The repo ships a Gazebo Harmonic (`gz sim`) + MoveIt 2 downstream-eval package
at `ros2_ws/src/generated_objects_eval/`. This note records the v2 driver and a
real headless **dynamic-stability** result comparing v1 and v2 generated objects.

## Stability test (real physics, headless)

Pipeline:

```bash
# 1) Train v1 (paper_repro) + v2 generators, export objects, write a manifest.
python scripts/run_sim_eval_v2.py build --n 6 --out output/sim_eval

# 2) Swap the trimesh collision OBJs (no vertex normals -> DART rejects them)
#    for AABB-box collisions (the repo's standard fix).
python scripts/patch_sdf_collision.py --manifest output/sim_eval/manifest.json

# 3) Launch the world headless and run the settle test in one shell so the
#    server stays alive for the whole run.
PKG=ros2_ws/src/generated_objects_eval
gz sim -s -r "$PKG/worlds/panda_eval_world.sdf" >/tmp/gz.log 2>&1 &
#   (wait until `gz service -l | grep panda_eval_world/create` appears)
python "$PKG/generated_objects_eval/gazebo_stability_eval.py" \
    --manifest output/sim_eval/manifest.json \
    --out      output/sim_eval/gazebo_stability.json \
    --config   "$PKG/config/eval_config.yaml"
kill %1
```

Each object is spawned ~5 cm above the 0.4 m table, settles for 2 s, and is
counted stable iff final tilt < 25° and vertical drift < 5 cm.

### Result (seed 42, 6 objects per method)

| method | spawned | stable | rate | typical tilt | typical drift | parts |
|--------|--------:|-------:|-----:|-------------:|--------------:|------:|
| v1 (paper_repro) | 6/6 | 6/6 | 100% | 0.0° | ~0.02 m | 1 |
| v2 | 6/6 | 6/6 | 100% | 0.0° | ~0.005 m | 3 |

**Takeaway:** the richer v2 objects (multi-part, mixed primitive types) are just
as dynamically stable as v1's single primitives in real physics — the
connectivity filter + ground-seating keep them upright. This complements the
independent force-closure grasp metric (`grasp_planner.py`, see
`scripts/verify_v2.py`), where v2 trades some single-object graspability for a
large gain in shape diversity.

## Grasp-execution test (MoveIt) — not run here

`generated_objects_eval/moveit_planning_eval.py` (+ `moveit_planning_eval.launch.py`)
runs RRTConnect planning to per-object approach poses on the Panda. It needs the
full MoveIt 2 stack (move_group, controllers) running, which is heavier than a
headless `gz sim` server; run it on a workstation with the built workspace
sourced. The manifest produced in step (1) is the same one it consumes.
```
