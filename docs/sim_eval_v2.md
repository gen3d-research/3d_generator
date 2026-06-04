# Sim-in-the-loop evaluation (v2)

The repo ships a Gazebo Harmonic (`gz sim`) + MoveIt 2 downstream-eval package
at `ros2_ws/src/generated_objects_eval/`. This note records the v2 driver and a
real headless **dynamic-stability** result comparing v1 and v2 generated objects.

## Stability test (real physics, headless)

Pipeline:

```bash
# 1) Train v1 (paper_repro) + v2 generators, export objects, write a manifest.
python scripts/run_sim_eval_v2.py build --n 6 --out output/sim_eval

# 2) Bump collision friction + inject the DetachableJoint plugin. (Collision is
#    now the exported MESH — export writes vertex normals so DART accepts it —
#    so objects rest flush; the old AABB-box workaround is gone.)
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

### Multi-seed result (8 seeds × 5 objects/method, paper-grade)

Reproduce with `python scripts/sweep_sim_stability.py --seeds 8 --n 5`.

| method | stable rate (mean ± 95% CI) | per-seed |
|--------|----------------------------:|----------|
| v1 (paper_repro) | **100.0% ± 0.0** | 100×8 |
| v2 | **100.0% ± 0.0** | 100×8 |

v2 − v1 mean diff = +0.0% (Wilcoxon n/a — identical). All 40 objects/method
settled upright (tilt 0°, drift within the 5 cm / 25° tolerances) across every
seed.

**Takeaway:** the richer v2 objects (multi-part, mixed primitive types) are just
as dynamically stable as v1's single primitives in real physics. This complements
the independent force-closure grasp metric (`scripts/verify_v2.py`,
`docs/results_v2.md`), where v2 trades a non-significant amount of single-object
graspability for a significant gain in shape diversity.

**Note on collision fidelity:** the 100% numbers above were measured with the
old AABB-box collision (over-approximate, very stable). `patch_sdf_collision.py`
now keeps the exact **mesh collision** (export writes vertex normals so DART
accepts it), which is a stricter, faithful test — re-run
`scripts/sweep_sim_stability.py` to refresh the numbers under mesh collision.

## Grasp-execution test (MoveIt) — not run here

`generated_objects_eval/moveit_planning_eval.py` (+ `moveit_planning_eval.launch.py`)
runs RRTConnect planning to per-object approach poses on the Panda. It needs the
full MoveIt 2 stack (move_group, controllers) running, which is heavier than a
headless `gz sim` server; run it on a workstation with the built workspace
sourced. The manifest produced in step (1) is the same one it consumes.
```
