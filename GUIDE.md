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

> **Want a specific generation path?** All **12 paths**, each as a copy-paste command, are in
> [`docs/GENERATION_PATHS.md`](docs/GENERATION_PATHS.md). This section details the most-used path
> (④ free CEM) and the `main.py` subcommands; **§3.5** below turns *any* path's output into a
> drop-/pick-testable manifest.

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

## 3.5 Build a manifest from a custom generator (paths ⑥–⑫)

`build_eval_manifest.py` only knows the five paper methods. `scripts/build_path_manifest.py` is its
counterpart for the Python-API paths (⑥–⑫ in [`docs/GENERATION_PATHS.md`](docs/GENERATION_PATHS.md))
— or any composition of them — and writes a manifest **identical in shape**, so it drops straight
into §5/§6.

```bash
# one path = one command (run --help for the full preset table)
python scripts/build_path_manifest.py --path 6 \
    --out output/path6/manifest.json --export-root output/path6/objects
python scripts/build_path_manifest.py --path 12 \
    --prompt "a small stable graspable curved bottle" \
    --out output/path12/manifest.json --export-root output/path12/objects

# or compose the knobs by hand — they stack (--method-tag sets the §6 method:= label)
python scripts/build_path_manifest.py --seed-archetype mug_like --gate --palette curved \
    --method-tag mug_seeded \
    --out output/mug_seeded/manifest.json --export-root output/mug_seeded/objects

# then stiffen contact/friction so the gripper can hold objects:
python scripts/patch_sdf_collision.py --manifest output/path6/manifest.json
```

Knobs: `--path {4,6,7,8,9,10,11,12}` preset, or stack `--repair` ⑥ · `--gate` ⑦ ·
`--seed-archetype NAME` ⑧ · `--palette curved|faceted|<keys>` ⑨ · `--pareto` ⑩ ·
`--target-size M` ⑪ · `--prompt "..."` ⑫. Budget: `--n` / `--iterations` / `--samples` / `--seed`.
When it finishes it prints the exact `--manifest` (drop test) and `manifest:= method:=` (pick test)
lines to paste.

Now `--manifest output/path6/manifest.json` works in the **§5 drop test** as-is, and
`manifest:=…/path6/manifest.json method:=path6` works in the **§6 pick test** (the demo filters by
`method`, so pass the `method:=` tag the runner reports).

### Under the hood — the equivalent inline snippet

`build_path_manifest.py` is just this loop; copy it into your own code to customize the export or
grasp settings. Edit **only** the `CONFIGURE` block; everything below it is the same wiring
`build_eval_manifest.py` uses.

```bash
env -u PYTHONPATH ~/venv/3d_cem/bin/python - <<'PY'
import sys, json; sys.path.insert(0, "src")
from pathlib import Path
import numpy as np, trimesh
from generator import RoboticObjectGenerator, GeneratorConfig
from export import URDFExporter, ExportConfig
from grasp_planner import plan_grasps, GripperSpec
from cem import CURVED_TYPES, FACETED_TYPES

# ===== CONFIGURE THE PATH (uncomment the line(s) for the path you want) =====
cfg = GeneratorConfig(seed=42)              # base = ④ free CEM
# cfg.repair_stability = True               # ⑥ re-orient onto a stable resting pose
# cfg.dynamic_stability_gate = True         # ⑦ suppress the tippy tail (hard gate)
gen = RoboticObjectGenerator(cfg)
# gen.seed_from("screwdriver")              # ⑧ warm-start from an archetype, let structure evolve
# gen.constrain_types(CURVED_TYPES)         # ⑨ train using only curved (or FACETED_TYPES) primitives
# gen.target_size(0.05)                     # ⑪ bias toward a 5 cm longest extent
gen.train(verbose=False)
objs = gen.generate(15)
# --- ⑫ text2geometry: replace the four lines above with -------------------
# from text2gen import generate_from_text
# objs, intent = generate_from_text("a small stable graspable curved bottle", n=15)
# --- ⑩ Pareto: keep only the non-dominated stability/graspability set ------
# from pareto import pareto_objects
# objs, _ = pareto_objects(objs, gen.scorer, keys=("stability_score", "graspability_score"))

# ===== EXPORT + WRITE A DROP/PICK-TESTABLE MANIFEST (don't edit below) =====
method, root = "path_demo", Path("output/path_demo/objects")
exporter = URDFExporter(ExportConfig(use_convex_hull=False, simplify_collision=False))
manifest = []
for k, obj in enumerate(objs):
    obj.name = f"{method}_{k:04d}"
    paths = exporter.export(obj, root / obj.name, obj.name)
    rep = plan_grasps(obj, GripperSpec(), n_surface=256, max_pairs=1500, max_returned=12, seed=42 + k)
    vm = trimesh.load(paths["visual_mesh"], force="mesh")
    mass, _, com = obj.mesh_mass_properties(1000.0)
    g = []
    for gr in rep.grasps[:12]:
        ax = np.asarray(gr.contact2) - np.asarray(gr.contact1); n = float(np.linalg.norm(ax))
        ax = (ax / n).tolist() if n > 1e-9 else [0.0, 1.0, 0.0]
        g.append({"center": gr.center.tolist(), "approach": gr.approach.tolist(), "axis": ax,
                  "width": float(gr.width), "margin": float(gr.margin),
                  "score": float(getattr(gr, "score", 0.0))})
    manifest.append({"name": obj.name, "method": method,
        "urdf": str(Path(paths["urdf"]).resolve()), "sdf": str(Path(paths["sdf"]).resolve()),
        "visual_mesh": str(Path(paths["visual_mesh"]).resolve()),
        "collision_mesh": str(Path(paths["collision_mesh"]).resolve()),
        "mass": float(mass), "com": list(map(float, com)),
        "aabb": [vm.bounds[0].tolist(), vm.bounds[1].tolist()],
        "extents": (vm.bounds[1] - vm.bounds[0]).tolist(),
        "grasps": g, "n_grasps_synth": len(rep.grasps)})
out = Path("output/path_demo/manifest.json"); out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(manifest, indent=2)); print("wrote", len(manifest), "->", out)
PY

# stiffen contact/friction (same as the paper manifests) so the gripper can hold objects:
python scripts/patch_sdf_collision.py --manifest output/path_demo/manifest.json
```

Now `--manifest output/path_demo/manifest.json` works in the **§5 drop test** as-is, and
`manifest:=…/path_demo/manifest.json method:=path_demo` works in the **§6 pick test** (the pick
demo filters by `method`, so pass `method:=path_demo` to match the tag this snippet writes).

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

## 5. Drop test (Gazebo dynamic stability) — watch it, or score it

Spawns each manifest object ~5 cm above the table in `panda_eval_world`, lets physics settle,
then measures **vertical drift** (from the table top, the expected rest height) and **tilt** —
"stable" iff drift < 5 cm and tilt < 25° (`config/eval_config.yaml`). Build the ROS 2 package
first (§4) and `source ros2_ws/install/setup.bash`. The two modes below use the **same evaluator**;
they differ only in which world launch you start (windowed vs. headless).

### 5a. Visualize it — watch objects fall in a Gazebo window

```bash
# Terminal A — open the world in a gz window
ros2 launch generated_objects_eval stability_world_gui.launch.py
```

```bash
# Terminal B — drop the first few objects so you can watch each one settle
ros2_ws/install/generated_objects_eval/bin/gazebo_stability_eval \
    --manifest $(pwd)/output/seed_42/eval_manifest.json \
    --out $(pwd)/output/seed_42/gazebo_stability.json \
    --max-objects 6
```
Keep `--max-objects` small (≈6) so you can actually see each drop. Swap the manifest for **any**
path's `manifest.json` (e.g. `output/path_demo/manifest.json` from §3.5) to watch that path's parts.
To record a clip, use Gazebo's `Settings → Video Recording`, or see [`DEMO.md`](DEMO.md) §3.

### 5b. Score it — headless, every object

```bash
# Terminal A — headless server (no window)
ros2 launch generated_objects_eval stability_world.launch.py
```

```bash
# Terminal B — drop every object and score it
ros2_ws/install/generated_objects_eval/bin/gazebo_stability_eval \
    --manifest $(pwd)/output/seed_42/eval_manifest.json \
    --out $(pwd)/output/seed_42/gazebo_stability.json \
    --max-objects 0          # 0 = all (the default)
```

Flags / notes (both modes):
- We call the **binary directly** (this node is registered as a console script, not via
  `ros2 run`). `$(pwd)/...` makes the paths absolute so gz finds the SDFs.
- `--manifest PATH` — the manifest whose objects to drop (any manifest with `sdf` paths;
  hand-write a one-entry manifest to drop a single specific part).
- `--out PATH` — per-object results JSON (`spawn_ok`, `stable`, `tilt_deg`, `drift_m`,
  `final_pose`); the node also prints a per-method `spawn_ok / stable / rate=%` summary.
- `--max-objects N` — cap how many to test (**0 = all**, the default).
- `--config PATH` — override `eval_config.yaml` (spawn pose, settle time, drift/tilt tolerances).
- **At scale** (thousands of objects) use the crash-resilient chunked runner — see §7b.

Kill stale sims between runs with a **precise** pattern — `pkill -KILL -f "gz sim -s -r"` (two
same-world servers fight over the spawn service). Avoid the broad `pkill -f "gz sim"`: it matches
any process whose command line merely contains "gz sim", which can SIGKILL an unrelated shell or a
live world.

---

## 6. Pick-and-place test (MoveIt 2 + Gazebo) — watch it, or score it

The Panda runs an **8-stage** pick-and-place under `gz_ros2_control`:
*ready → pre-grasp → grasp → lift → transport → place → retract*. The object is held by genuine
finger friction (effort-controlled gripper), the table is a MoveIt `CollisionObject`, and the
object's real collision mesh is added to the planning scene. Build + `source` the package (§4) first.

### 6a. Visualize it — Gazebo + RViz, the arm actually picks (full physics)

One launch brings up the gz window (object on the table), an RViz window (the Panda planning), and
cycles a new object each pass:

```bash
source ros2_ws/install/setup.bash
ros2 launch generated_objects_eval visual_demo.launch.py \
    manifest:=$(pwd)/output/seed_42/eval_manifest.json \
    method:=cem use_gz_control:=true loop:=true
```
- `manifest:=PATH` — which manifest to pull objects from (auto-detects `output/seed_42/...`
  if omitted). For a **custom path** (§3.5), point at its `manifest.json` **and** pass
  `method:=path_demo` so the demo finds those entries.
- `method:=cem` — which generator's objects to cycle: `cem | cmaes | ga | random_search |
  fixed_cad` (or your own method tag). Default `cem`.
- `use_gz_control:=true` — **the arm actually moves in physics** under `gz_ros2_control`. Set
  `false` (default) for a faster **RViz-only animation** (no Gazebo controllers) — useful on a
  slow GPU.
- `loop:=true` — keep cycling objects indefinitely (good for screen-recording; default true).
- *(other args)* `headless:=true` runs gz server-only (no window); `render_engine:=ogre`
  falls back from `ogre2` for hybrid-GPU laptops.

To pin a **specific** object / spawn pose, run the driver directly with
`--object-index N`, `--spawn-x/-y/-z`, `--place-dx/-dy/-dz` (see [`DEMO.md`](DEMO.md)).
See [`DEMO.md`](DEMO.md) for recording the videos, RViz tips, and GPU/`libEGL` gotchas.

### 6b. Score it — headless MoveIt-plan success (no window)

Plans to every grasp pose in the manifest and records the success rate:

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

> **At scale, use the chunked runner — `scripts/run_drop_test_chunked.sh`.** A single gz
> session is fine for a few dozen objects, but the gz physics engine (ODE) crashes on a small
> fraction (~0.1%) of meshes — `ODE INTERNAL ERROR … collision_trimesh_trimesh.cpp` — which
> kills the world, and since `gazebo_stability_eval` writes its JSON only at the end, a single
> long run can lose everything when a world dies mid-way. The chunked runner processes the
> manifest in small **fresh-world chunks** across N isolated parallel workers, so a crash
> loses only one chunk, every chunk is checkpointed, and crashed chunks are retried:
>
> ```bash
> bash scripts/run_drop_test_chunked.sh \
>     output/arch_variants/manifest.json \
>     output/arch_variants/gazebo_stability.json 100 4    # chunk=100, 4 workers
> ```
> Results group per archetype (`method`), same as the single-world `gazebo_stability_eval`.
> Tunables via env: `GZ_SVC_TIMEOUT_MS` (gz service wait, default 15000), `GZ_SVC_SUBPROC_S`
> (subprocess kill, 20), `GZ_ABORT_AFTER_FAILS` (consecutive spawn fails → world-died abort, 5).
> Measured: 4 workers × ~100-object chunks → ~99% spawn, ~1 chunk/worker every few minutes;
> 3,150 objects (30/archetype) in ~30 min.
>
> Never use the broad `pkill -f "gz sim"` to clean up — it matches any process whose command
> line merely contains the string; kill precisely (`pkill -f "gz sim -s -r"`) or by PID/group.

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
