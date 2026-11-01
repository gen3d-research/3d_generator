# Physics evaluation in MuJoCo and the learned grasp-success predictor

The ICARM paper's three downstream metrics are force-closure grasp synthesis (an antipodal grasp
*exists*), MoveIt 2 planning and a Gazebo drop test. None of them executes a grasp. This document
describes the second physics engine added after the paper: a MuJoCo drop test, a MuJoCo
pick-and-lift, the physics-labelled grasp dataset built from it, the learned grasp-success
predictor trained on that dataset, and the planner's hand-versus-table check that the physics
test motivated.

```
eval_manifest.json ──► mujoco_eval.parse_sdf ──► MJCF (object + table [+ gripper])
  (objects, SDFs,         per-primitive collision      │
   ranked grasps)         or CoACD of the true mesh    ├─► drop_test      ─► run_mujoco_eval.py ─► per-seed + aggregate
                                                       └─► pick_and_lift  ─► build_grasp_dataset.py ─► *.npz
                                                                                      │
                                                    train_grasp_predictor.py ◄────────┘  (torch, separate venv)
```

## Object model

`src/mujoco_eval.py` rebuilds each exported object from its SDF: mass, COM and the inertia tensor
from `<inertial>`, and one MuJoCo geom per `<collision>` (native box / cylinder / sphere, convex
mesh pieces). Two collision models:

| `--collision` | Geometry | Use |
|---|---|---|
| `sdf` (default) | the exported per-primitive set, exactly what Gazebo sees; concave primitives (hollow shell, open tube, torus, handle, gear, L/U/T/I profile) are convex-hulled | cross-simulator parity with Gazebo |
| `coacd` | a CoACD convex decomposition of the true visual mesh (threshold 0.05), cached as `meshes/coacd_0.05/part_*.obj` | grasping on the real shape, cavities included |

CoACD sometimes aborts in C++, so each decomposition runs in a child process (single-threaded),
retrying with other preprocessing modes; an object whose decomposition fails falls back to the
SDF geometry and is recorded as `collision: sdf_fallback`.

Contact model: MuJoCo's default soft contact for object and table (a stiff contact makes a 5 cm
drop rebound about 2 cm and topples stable objects); the gripper pads alone get a stiff contact
(time constant 2·dt) through geom `priority`, so a 20 N pinch does not sink centimetres into a light
object. Friction μ = 0.5 everywhere (the value the force-closure synthesiser assumes), torsional
0.005, elliptic cones, `impratio` 10, dt = 1 ms.

## Drop test

Mirrors `gazebo_stability_eval`: spawn the base-seated object 5 cm above the table with a 5° tilt
about x, simulate 2 s, and call it stable if the tilt stays under 25° and the base drifts less than
5 cm from the table top. `run_mujoco_eval.py` reports, per seed, the Cohen κ between the MuJoCo
and Gazebo verdicts when a `gazebo_stability.json` is next to the manifest (or given with
`--gazebo`).

Validation: on the 10,405 archetype variants of the v2.9 study the two engines agree on 90.2 % of
objects (κ = 0.70), and per-archetype stable rates correlate at r = 0.88. Disagreements concentrate
near the tipping margin (Gazebo's stiffer, bouncier ODE contact, kp = 1e6 and restitution 0.1,
topples more).

## Pick-and-lift

A free-floating parallel-jaw gripper driven by a mocap weld: Panda-like, 8 cm maximum opening, 2 ×
4 cm pads, a palm 3 cm behind the grasp centre spanning the finger travel, 20 N per finger. On the
settled object, the manifest's grasps are walked in the planner's rank order:

1. **Feasibility** (skipped, not executed): wider than the gripper; approach from below; or the
   hand, pre-shaped to the grasp width plus 1.5 cm per finger, already penetrates the table or
   object by more than 2 mm at the pre-grasp 10 cm back along the approach.
2. **Execution** of up to three feasible grasps: approach 0.6 s, close 0.6 s, lift 10 cm in 1 s,
   hold 0.5 s.
3. **Success**: the object rose at least 7 cm and both pads still touch it. Failures are labelled
   `missed` (closed on air), `dropped` (pinched but not lifted, usually pivoting about the pinch
   line) or `slipped` (lost during the lift).

Per object: `has_feasible`, `lift_first` (first feasible grasp holds) and `lift_any`.

```bash
python scripts/run_mujoco_eval.py --root output/journal_10seed --seeds 42 43 44 45 46 47 48 49 50 51
python scripts/run_mujoco_eval.py --root output/journal_10seed --seeds 42 43 44 45 46 47 48 49 50 51 --collision coacd
python scripts/run_mujoco_eval.py --manifest output/arch_variants/manifest.json \
    --gazebo output/arch_variants/gazebo_stability_v29.json --no-lift
```

A seed (125 objects) takes a few seconds on 12 cores; the first `coacd` run on a set of objects
adds about 5 s of CoACD per object, after which the decompositions are cached.

## Grasp dataset

`scripts/build_grasp_dataset.py` executes *every* feasible planner grasp of every object
(`grasps_per_object` raised to 64) and stores, per object, 512 surface points of the true mesh,
mass, COM and extents, and per grasp its frame, width, planner score, feasibility, outcome and
reason. All arrays are in the object frame (`.npz`; field list in the script's docstring).

## Learned grasp-success predictor

`scripts/train_grasp_predictor.py` (torch; run it from a separate venv, see below) expresses the
object's points in each grasp frame (x = pad width, y = antipodal axis, z = approach; 256 points
per sample, finger-swap augmentation) and feeds a small PointNet (3→64→128→256, max-pool) plus 13
scalars (width, log mass, COM and gravity direction in the grasp frame, grasp height, planner score,
extents) to an MLP. It is trained on feasible grasps only, with BCE.

Splits hold out whole groups, never single grasps of a training object: `--split source` (whole
seeds / manifests), `archetype` (whole archetypes or generator methods) or `object`. Reported on
the test split: AUC of the model, of the planner score and of a torque-arm heuristic, and the lift
rate when each object executes the model's top grasp vs the planner's first vs the oracle.
`model.pt` and `results.json` go to `--out`.

**Why a separate venv.** torch depends on `networkx`; with networkx installed, trimesh takes
different code paths (`is_volume`, `split`, winding repair) that the generator's geometry relies on.
Keep the generator venv torch-free:

```bash
python3 -m venv ~/venv/gen3d_learn
~/venv/gen3d_learn/bin/pip install torch numpy scipy --extra-index-url https://download.pytorch.org/whl/cu130
~/venv/gen3d_learn/bin/python scripts/train_grasp_predictor.py \
    --data output/grasp_dataset/journal.npz --split source --out output/grasp_predictor/run0
```

## Predictor in the generator

The numpy predictor (`src/grasp_predictor.py`, weights exported as `model.npz`) can steer generation
in three places; `scripts/run_physics_aware_study.sh` runs them as configs (`CONFIGS`, `METHODS`).

| Config | What changes | Option |
|---|---|---|
| `grasps` | each object's grasps are re-ranked, with the table check | `--table-check --grasp-model M` |
| `select` | `grasps`, and the CEM keeps the top-k of a pool by best predicted grasp | `--cem-pool POOL` |
| `train` | `grasps`, and the CEM selects elites on score x predicted lift | `--train-with-model` |
| `train_select` | `train` and `select` together | both |

In code, `GeneratorConfig(grasp_model=path)` sets the training objective. Train the predictor on
objects disjoint from the graded seeds, and grade under perturbed physics as the script does, so
the result does not rest on the physics the labels came from.

`GeneratorConfig(cem_workers=n)` / `CEMConfig(n_workers=n)` sample and score in `n` forked workers,
each sample with its own seed stream, so the result does not depend on `n`. It differs from the
serial path (`n_workers=1`), which draws from one stream and is the published path.
`scripts/benchmark_throughput.py` measures both objectives on 1 to 32 cores.

## Planner hand-versus-table check

`grasp_planner.gripper_min_z()` sweeps the same gripper geometry from the pre-grasp to the grasp
pose and returns its lowest point in the object frame (the table is z = 0 for base-seated objects).
`plan_grasps(table_check=True)` and `build_eval_manifest.py --table-check` drop grasps that reach
below −2 mm. It is off by default so published manifests reproduce. On the 10-seed grasp dataset it
catches 94.6 % of MuJoCo's pre-grasp table collisions, and the additional grasps it rejects lift
only 39 % when executed, against 81 % for the rest.
