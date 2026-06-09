# Guide — generate, visualize, and test parts

End-to-end walkthrough: **train the generator → generate parts → visualize them →
drop-test and pick-and-place them in ROS 2 / Gazebo.** Every command below lists what
each flag does. For the full paper-number reproduction see [`REPRODUCE.md`](REPRODUCE.md);
for the recorded demo videos see [`DEMO.md`](DEMO.md).

The library has **19 primitive types** and **105 hand-written archetypes**
(see [`docs/PRIMITIVES.md`](docs/PRIMITIVES.md) and [`docs/ROADMAP.md`](docs/ROADMAP.md)).

---

## 0. Setup

```bash
python3 -m venv ~/venv/3d_cem
source ~/venv/3d_cem/bin/activate
pip install -r requirements.txt
```

- `python3 -m venv ~/venv/3d_cem` — create an isolated virtual environment at that path
  (kept outside the repo so build artefacts don't churn it).
- `source ~/venv/3d_cem/bin/activate` — activate it in the current shell (run once per shell).
- `pip install -r requirements.txt` — install the deps: numpy, scipy, trimesh, pyyaml, cma,
  matplotlib, manifold3d (the CSG backend used by hollow/rounded/tube primitives).

Run every command below from the repo root (`3d_generator/`). The scripts add `src/` to
`sys.path` themselves. If ROS 2 is sourced in your shell, prefix the pure-Python commands
with **`env -u PYTHONPATH`** so its pytest/plugin paths don't leak in. The ROS 2 parts
(§5–§6) need **ROS 2 Jazzy + Gazebo (gz)** and a one-time `colcon build` (§4).

---

## 1. Generate / train parts

The pipeline is **sample → score → CEM update → export**. No datasets, no GPU.

```bash
python src/main.py demo
```
- `demo` — subcommand: brief training + 5 objects, a quick sanity check (no flags).

```bash
python src/main.py generate -n 100 -o output/objects \
    --train --iterations 30 --samples 100 --seed 42 \
    --save-generator output/trained_generator.json
```
- `generate` — subcommand: produce objects and (optionally) export them.
- `-n 100` — **number of objects** to generate after training (default 10).
- `-o output/objects` — **output directory**; each object becomes
  `object_XXXX/{*.urdf,*.sdf,meshes/*.obj,*_metadata.yaml}`. Omit to score in-memory
  without writing files.
- `--train` — **train the CEM first** (otherwise it samples from the untrained prior).
- `--iterations 30` — **CEM iterations** while training (more = better-converged
  distribution, linear cost; default 30).
- `--samples 100` — **candidate objects scored per iteration** (the CEM keeps the top 20%
  as elites each round; default 100).
- `--seed 42` — **RNG seed** for reproducible objects (default 42).
- `--save-generator output/trained_generator.json` — **persist the trained distribution**
  to JSON so you can reload it later instead of retraining.
- *(other flags)* `--prefix NAME` sets the object filename prefix (default `object`);
  `--no-filter` disables the connectivity/score/grasp quality gate (keeps every sample).

```bash
python src/main.py archetypes -o output/archetypes
```
- `archetypes` — subcommand: export the whole hand-written archetype set.
- `-o output/archetypes` — **output directory** (required for this subcommand).

```bash
python src/main.py score output/objects/object_0000/meshes/object_0000_visual.obj
```
- `score` — subcommand: print the 6-component score (size / stability / graspability /
  complexity / validity / assembly) for one mesh file (positional path argument).

**Python API:**

```python
from generator import RoboticObjectGenerator
gen = RoboticObjectGenerator()
gen.train(verbose=True)
objects = gen.generate(n=50)              # connected, grasp-reranked, quality-filtered
gen.export_all(objects, "output/objects")
gen.save("output/trained_generator.json")
```

`GeneratorConfig` knobs: `max_primitives` (default 16), `require_connected`,
`rerank_by_grasp`, `low_grasp_gate` (penalize ungraspable cones/pyramids),
`use_mesh_inertia`. `generator.paper_repro_generator()` returns the v1/paper config.

---

## 2. Visualize parts

All renderers are **headless** (matplotlib Agg) and write PNGs under `docs/gallery/`.

```bash
python scripts/render_primitives.py
python scripts/render_primitives.py --emit-doc
```
- *(no flags)* — render the 19 primitive types × 3 variants → `docs/gallery/primitives.png`.
- `--emit-doc` — instead of rendering, print the markdown parameter/clamp table (used to keep
  the docs in sync with `PRIMITIVE_SPECS`).

```bash
python scripts/render_gallery.py
```
- *(no flags)* — render all 105 archetypes → `archetypes.png` and a 16-object trained-CEM
  sample sheet → `v2_samples.png`. (`--cols N` sets grid width.)

```bash
python scripts/gen_strategies.py --strategy all
python scripts/gen_strategies.py --strategy single --copy-web
```
- `--strategy all` — run every constrained-generation strategy. Other values:
  `single` (each type × N=2..10 copies unioned, + symmetric on prime N), `pairs`
  (every two-type pair, + symmetric), `curved` / `faceted` (palette-restricted batches),
  `oneofeach` (one union of all types), `symmetric` (bilaterally-symmetric batch),
  `default` (mixed baseline). Each writes `docs/gallery/strategy_<name>.png`.
- `--copy-web` — also copy the PNGs into `docs/static/images/` for the website.
- *(other flags)* `--n 12` objects per batch; `--cols 6` grid width; `--seed 42`;
  `--train` trains the baseline first; `--max-primitives` caps parts.

```bash
python scripts/render_archetype_varieties.py --subset --combined
python scripts/render_archetype_varieties.py --archetypes cup,jar,nut --k 12
```
- `--subset` — use the curated ~10-archetype representative set (one per group + containers).
- `--combined` — emit one grid (rows = archetypes) instead of one PNG per archetype.
- `--archetypes cup,jar,nut` — explicit comma-separated archetype names instead of the subset.
- `--k 12` — **varieties per archetype** to sample (default 10).
- *(other flags)* `--all` (every archetype, big); `--iters 20` / `--samples 50` (per-archetype
  CEM training budget); `--no-train` (raw sampled varieties, skip CEM tuning); `--cols`,
  `--seed`, `--copy-web`.

Open the PNGs directly, or open `docs/index.html` (Shapes / Strategies sections).

---

## 3. Build a manifest (objects + grasps) for the sim tests

The ROS 2 drop/pick tests consume an **eval manifest**: per-method objects exported to
URDF/SDF with synthesized force-closure grasp candidates.

```bash
python scripts/build_eval_manifest.py \
    --budget 1500 --top-k 25 --seed 42 \
    --out output/seed_42/eval_manifest.json \
    --export-root output/seed_42/manifest_objects
```
- `--budget 1500` — **candidates each method evaluates** before selecting elites (bigger =
  longer search, linear cost; paper setting 1500, default 600).
- `--top-k 25` — **objects kept per method** in the manifest (5 methods × 25 = 125 objects).
- `--seed 42` — RNG seed.
- `--out PATH` — where to write the manifest JSON.
- `--export-root DIR` — directory for the exported URDF/SDF/mesh files the manifest points at.
- *(other flags)* `--methods cem cmaes ga random_search fixed_cad` (which methods to include);
  `--n-grasps 12` (max grasp candidates synthesized per object).

```bash
python scripts/patch_sdf_collision.py --manifest output/seed_42/eval_manifest.json
```
- `--manifest PATH` — rewrite every SDF in this manifest with **stiff contact + high friction**
  (so the gripper can actually hold objects) and remove any legacy weld joint. Idempotent.

```bash
python scripts/run_unified_eval.py --budget 1500 --top-k 100 --seed 42 \
    --out output/seed_42/unified_eval.json
```
- `--budget` / `--seed` — as above.
- `--top-k 100` — evaluate the full top-100 population per method (CPU grasp-success +
  diversity; default 100, independent of the manifest's top-k).
- `--out PATH` — output JSON (per-method force-closure grasp rate, feature/Chamfer diversity,
  suitability score).

---

## 4. Build the ROS 2 package (once)

```bash
cd ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select generated_objects_eval
source install/setup.bash
cd ..
```
- `source /opt/ros/jazzy/setup.bash` — put ROS 2 Jazzy on the path.
- `colcon build --packages-select generated_objects_eval` — build **only** this package
  (faster than building the whole workspace).
- `source install/setup.bash` — overlay the freshly-built package onto the environment.

Needs `ros-jazzy-moveit`, `ros-jazzy-moveit-py`, the Panda MoveIt config/description,
`ros-jazzy-ros-gz-sim`, and `gz_ros2_control` (see [`REPRODUCE.md`](REPRODUCE.md) §0).

---

## 5. Drop test (Gazebo dynamic stability)

Spawns each manifest object ~5 cm above the table in `panda_eval_world`, lets physics settle,
then measures **vertical drift** (from the table top, the expected rest height) and **tilt** —
"stable" iff drift < 5 cm and tilt < 25° (`config/eval_config.yaml`).

```bash
# Terminal A — the world (headless server). Use stability_world_gui.launch.py for a window.
ros2 launch generated_objects_eval stability_world.launch.py
```
- `ros2 launch <pkg> <file>` — start the Gazebo world that the evaluator spawns objects into.
  No arguments needed; `_gui` variant opens the gz window (for recording).

```bash
# Terminal B — drop every object and score it
ros2_ws/install/generated_objects_eval/bin/gazebo_stability_eval \
    --manifest $(pwd)/output/seed_42/eval_manifest.json \
    --out $(pwd)/output/seed_42/gazebo_stability.json \
    --max-objects 25
```
- We call the **binary directly** (this node is registered as a console script, not via
  `ros2 run`). `$(pwd)/...` makes the paths absolute so gz finds the SDFs.
- `--manifest PATH` — the manifest whose objects to drop (any manifest with `sdf` paths;
  hand-write a one-entry manifest to drop a single specific part).
- `--out PATH` — per-object results JSON (`spawn_ok`, `stable`, `tilt_deg`, `drift_m`,
  `final_pose`).
- `--max-objects 25` — cap how many to test (**0 = all**, the default). Use a small number
  for a quick check or a short video.
- *(other flags)* `--config PATH` — override `eval_config.yaml` (spawn pose, settle time,
  drift/tilt tolerances).

It prints a per-method summary (`spawn_ok`, `stable`, `rate=%`). Kill stale sims between runs
with a **precise** pattern — `pkill -KILL -f "gz sim -s -r"` (two same-world servers fight over
the spawn service). Avoid the broad `pkill -f "gz sim"`: it matches any process whose command
line merely contains "gz sim", which can SIGKILL an unrelated shell or a live world.

---

## 6. Pick-and-place test (MoveIt 2 + Gazebo, with physics)

The Panda runs an **8-stage** pick-and-place under `gz_ros2_control`:
*ready → pre-grasp → grasp → lift → transport → place → retract*. The object is held by genuine
finger friction (effort-controlled gripper), the table is a MoveIt `CollisionObject`, and the
object's real collision mesh is added to the planning scene.

```bash
source ros2_ws/install/setup.bash
ros2 launch generated_objects_eval visual_demo.launch.py \
    manifest:=$(pwd)/output/seed_42/eval_manifest.json \
    method:=cem use_gz_control:=true loop:=true
```
- `manifest:=PATH` — which manifest to pull objects from (auto-detects `output/seed_42/...`
  if omitted).
- `method:=cem` — which generator's objects to cycle: `cem | cmaes | ga | random_search |
  fixed_cad` (default `cem`).
- `use_gz_control:=true` — **the arm actually moves in physics** under `gz_ros2_control`. Set
  `false` (default) for a faster RViz-only animation (no Gazebo controllers).
- `loop:=true` — keep cycling objects indefinitely (good for screen-recording; default true).
- *(other args)* `headless:=true` runs gz server-only (no window); `render_engine:=ogre`
  falls back from `ogre2` for hybrid-GPU laptops.

To pin a **specific** object / spawn pose, run the driver directly with
`--object-index N`, `--spawn-x/-y/-z`, `--place-dx/-dy/-dz` (see `DEMO.md`).

**Headless / batch MoveIt-plan success** (no window) — plans to every grasp pose, records the
success rate:

```bash
ros2 launch generated_objects_eval moveit_planning_eval.launch.py \
    manifest:=$(pwd)/output/seed_42/eval_manifest.json \
    out:=$(pwd)/output/seed_42/moveit_results.json \
    max_objects:=30
```
- `manifest:=PATH` — objects to plan grasps for.
- `out:=PATH` — results JSON (per object: `n_grasps`, `n_success`, `any_success`).
- `max_objects:=30` — cap objects (**0 = all**). `moveit_py` 2.12 segfaults on shutdown
  *after* writing the JSON — wait for `"Wrote N results"` then Ctrl-C (or use
  `scripts/run_multi_seed.sh`, which polls + kills automatically).

See [`DEMO.md`](DEMO.md) for recording the videos, RViz tips, and GPU/`libEGL` gotchas.

---

## 7b. Test ARCHETYPE VARIANTS in ROS 2 (N variants per archetype)

To drop-test / pick-test many parameter **variants of each archetype** (e.g. 100 mugs,
100 bottles, …), build a variant manifest and feed it to the same §5–§6 evaluators.
`build_archetype_manifest.py` generates the variants (per-archetype CEM), exports each to
URDF/SDF, synthesizes grasps, and tags each with `method=<archetype name>` so results group
per archetype.

```bash
python scripts/build_archetype_manifest.py --variants 8 \
    --archetypes mug_like,wine_bottle,nut,i_beam,gear_like,bucket \
    --out output/arch_demo/manifest.json --export-root output/arch_demo/objects
python scripts/patch_sdf_collision.py --manifest output/arch_demo/manifest.json
```
- `--variants 8` — **parameter variants to generate per archetype** (set 100 for the full ask).
- `--archetypes a,b,c` — **comma-separated archetype names** to include (omit for **all 105**).
- `--out PATH` / `--export-root DIR` — manifest JSON and the exported URDF/SDF/mesh tree.
- *(other flags)* `--train` CEM-tunes each archetype first (optimized-but-narrower variants;
  default is raw sampled variants, more diverse); `--iters` / `--samples` are that CEM's
  budget; `--n-grasps 12`; `--seed 42`.

Then run **exactly the §5 drop test** and **§6 pick-and-place** with
`--manifest .../arch_demo/manifest.json`. The drop-test / MoveIt summaries print one row per
archetype; `visual_demo.launch.py method:=mug_like` cycles just that archetype's variants.

### Full run — all 105 archetypes × 100 variants, then drop-test in one session

Copy-paste (use the venv's python; `env -u PYTHONPATH` keeps a sourced ROS 2 off `PYTHONPATH`):

```bash
# 1. Generate 10,500 variants (all 105 archetypes × 100) -> URDF/SDF + grasps. ~hours of CPU.
env -u PYTHONPATH ~/venv/3d_cem/bin/python scripts/build_archetype_manifest.py \
    --variants 100 --out output/arch_variants/manifest.json \
    --export-root output/arch_variants/objects

# 2. Stiffen contact + raise friction on every SDF so objects rest flush.
env -u PYTHONPATH ~/venv/3d_cem/bin/python scripts/patch_sdf_collision.py \
    --manifest output/arch_variants/manifest.json

# 3. Drop-test all of them in ONE gz session (no restart needed).
source ros2_ws/install/setup.bash
ros2 launch generated_objects_eval stability_world.launch.py &      # headless world, one session
ros2_ws/install/generated_objects_eval/bin/gazebo_stability_eval \
    --manifest $(pwd)/output/arch_variants/manifest.json \
    --out $(pwd)/output/arch_variants/gazebo_stability.json --max-objects 0
```

- Step 3 is sequential (~3–5 s/object → 10,500 objects ≈ 10+ h). Trim with `--max-objects N`
  for a sample, or `--variants`/`--archetypes` in step 1 for a smaller set.
- **`& ` on the launch** backgrounds the world so the next line (the evaluator) runs in the
  same shell; the evaluator spawns objects into that running world via `gz service`.

**Watch the drops in a Gazebo window** — swap the launch for the GUI world:

```bash
ros2 launch generated_objects_eval stability_world_gui.launch.py &   # opens the gz window
ros2_ws/install/generated_objects_eval/bin/gazebo_stability_eval \
    --manifest $(pwd)/output/arch_variants/manifest.json \
    --out $(pwd)/output/arch_variants/gazebo_stability.json --max-objects 30
```

(Use a small `--max-objects` with the GUI so you can actually watch each object fall and
settle.) When done, stop the world: `kill %1` (or `pkill -f "gz sim -s -r"` — a *precise*
pattern; never `pkill -f "gz sim"`, which can match unrelated shells).

> **One gz session handles the whole manifest** — the evaluator spawns → settles → *despawns*
> each object in turn, so models don't accumulate (verified: 45 variants spawned + settled
> 45/45 in a single session). You do **not** need to restart Gazebo between objects.
> (If you ever see everything after the first ~20 objects report `spawn_ok=False`, it's an
> external kill of the gz process — most often an over-broad `pkill -f "gz sim"` matching the
> wrong process. Kill precisely, e.g. `pkill -f "gz sim -s -r"`, or just the launch PID.)
>
> **The only real constraint is wall-clock.** Generation is cheap, but per-object export +
> grasp synthesis is ~0.5–2 s and the sim is sequential (~3–5 s/object). 105 archetypes ×
> 100 = 10,500 objects is a few hours of CPU and ~10+ hours of sim **in one run** — fine to
> leave running, but scope it down (fewer `--variants` or a subset of `--archetypes`) if you
> want a quick pass. The pure-CPU grasp/diversity eval can always run on the full set.

---

## 7. Full evaluation (all metrics, all seeds)

```bash
bash scripts/run_multi_seed.sh 42 43 44
```
- Positional args `42 43 44` — **the seeds** to run. For each, it builds the manifest, patches
  SDFs, and runs the grasp + MoveIt + Gazebo metrics into `output/seed_<N>/`. ~20 min/seed.

```bash
python scripts/aggregate_seeds.py 42 43 44 --out output/aggregated.json
cp output/aggregated.json docs/data/results.json
```
- Positional args `42 43 44` — **the seeds to aggregate** (must already have been run).
- `--out PATH` — combined JSON (per-method means, 95% CIs, CEM-vs-best significance test).
  It also prints a markdown table. The `cp` refreshes the project-page numbers.

Full details + reproduction tolerances in [`REPRODUCE.md`](REPRODUCE.md).
