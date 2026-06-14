# Generation paths

The repo can produce objects several different ways. They differ along **two axes**:

- **Structure** — is the object's *topology* (which primitives, how many, how connected)
  **fixed** by a hand-written template, or **discovered** by the search?
- **Optimization** — are the shape parameters **raw-sampled**, or **optimized** against the
  manipulation score (`ObjectScorer`: stability + graspability + size + complexity + validity)?

|                         | **raw-sampled**                              | **score-optimized**                          |
|-------------------------|----------------------------------------------|----------------------------------------------|
| **fixed structure** (archetype) | ① archetype variants (raw)            | ② archetype variants, **CEM-tuned**          |
| **free structure** (composed)   | ③ random search (a baseline)          | ④ **free CEM** (+ CMA-ES / GA baselines)     |

This is the key to reading any stability/graspability number: **only the score-optimized,
free-structure path (④) is "the generator" the headline claims describe.** The others are
baselines, ablations, or library tooling.

---

## ▶ Generate with each path — exact commands

Run everything from `3d_generator/` with the venv active (`source ~/venv/3d_cem/bin/activate`).
If ROS 2 is sourced in your shell, run Python via `env -u PYTHONPATH ~/venv/3d_cem/bin/python`
instead of bare `python`. All outputs land under `output/`.

A path that writes a **manifest** (`…/manifest.json`) plugs straight into the **drop test**
(GUIDE.md §5) and **pick-and-place test** (GUIDE.md §6). The `main.py` paths write a folder of
URDF/SDF instead; wrap those (and the Python-API paths ⑥–⑫) into a manifest with the one snippet
in **GUIDE.md §3.5**.

### ① Archetype variants — raw (no optimization)
```bash
# every archetype once, at its hand-set defaults -> URDF/SDF per shape
python src/main.py archetypes -o output/archetypes

# OR: N parameter variants per archetype, as a drop/pick-testable manifest
python scripts/build_archetype_manifest.py --variants 8 \
    --archetypes mug_like,wine_bottle,nut \
    --out output/path1_raw/manifest.json --export-root output/path1_raw/objects
```
Omit `--archetypes` to cover all 105; use `--variants 100` for the full per-archetype set.

### ② Archetype variants — CEM-tuned (same structure, optimized params)
```bash
python scripts/build_archetype_manifest.py --variants 8 --train \
    --archetypes mug_like,wine_bottle,nut \
    --out output/path2_cem/manifest.json --export-root output/path2_cem/objects
```
`--train` adds the per-archetype CEM (`--iters` / `--samples` set its budget).

### ③ Random search — baseline (free structure, no optimization)
```bash
python scripts/build_eval_manifest.py --methods random_search \
    --budget 1500 --top-k 25 --seed 42 \
    --out output/path3_random/manifest.json --export-root output/path3_random/objects
```

### ④ Free CEM — *the* generator (free structure, optimized)
```bash
# (a) CLI: train, generate, export a folder of URDF/SDF
python src/main.py generate -n 25 -o output/path4_cem/objects \
    --train --iterations 30 --samples 100 --seed 42

# (b) the paper manifest: CEM + CMA-ES + GA + random + fixed_cad, all in one
python scripts/build_eval_manifest.py --budget 1500 --top-k 25 --seed 42 \
    --out output/seed_42/eval_manifest.json --export-root output/seed_42/manifest_objects
python scripts/patch_sdf_collision.py --manifest output/seed_42/eval_manifest.json
```
In (b): `cem` = ④, `cmaes`/`ga` = the other optimizers over the same free space, `random_search`
= ③, `fixed_cad` = ①. Pick a subset with `--methods cem cmaes ga`.

### ⑤ Constrained strategies — visualization galleries (PNG only)
```bash
python scripts/gen_strategies.py --strategy all
# strategies: single | pairs | curved | faceted | oneofeach | symmetric | default | all
```
Writes `docs/gallery/strategy_<name>.png`. (To *train* a palette-constrained generator and then
drop/pick-test it, use path ⑨ below.)

### ⑥–⑫ Python-API paths — one command each via `build_path_manifest.py`

Each of these is path ④ with one switch flipped. `scripts/build_path_manifest.py --path N` builds a
drop/pick-testable manifest for any of them (same shape as `build_eval_manifest.py`):

```bash
python scripts/build_path_manifest.py --path 6  --out output/path6/manifest.json  --export-root output/path6/objects    # ⑥ stability-repair
python scripts/build_path_manifest.py --path 7  --out output/path7/manifest.json  --export-root output/path7/objects    # ⑦ dynamic-stability gate
python scripts/build_path_manifest.py --path 8  --out output/path8/manifest.json  --export-root output/path8/objects    # ⑧ seed (screwdriver) + gate
python scripts/build_path_manifest.py --path 9  --palette curved      --out output/path9/manifest.json  --export-root output/path9/objects     # ⑨ constrained-optimized
python scripts/build_path_manifest.py --path 10 --out output/path10/manifest.json --export-root output/path10/objects   # ⑩ Pareto front
python scripts/build_path_manifest.py --path 11 --target-size 0.05    --out output/path11/manifest.json --export-root output/path11/objects    # ⑪ targeted size
python scripts/build_path_manifest.py --path 12 --prompt "a small stable graspable curved bottle" --out output/path12/manifest.json --export-root output/path12/objects   # ⑫ text2geometry
```

`--path N` is a preset over orthogonal knobs that also **stack standalone** (`--repair`, `--gate`,
`--seed-archetype NAME`, `--palette curved|faceted|<keys>`, `--target-size M`, `--pareto`,
`--prompt "..."`, plus `--n` / `--iterations` / `--samples` / `--seed`), so you can compose them
exactly like the Python recipes:

| # | Path | Runner knob | Python equivalent |
|---|------|-------------|-------------------|
| ⑥ | stability-repair — settle upright **by construction** | `--repair` | `GeneratorConfig(repair_stability=True)` |
| ⑦ | dynamic-stability **gate** — kill the tippy tail | `--gate` | `GeneratorConfig(dynamic_stability_gate=True)` |
| ⑧ | archetype-seeded free CEM — warm-start, evolve | `--seed-archetype screwdriver --gate` | `gen.seed_from("screwdriver")` |
| ⑨ | constrained-**optimized** — train on a palette | `--palette curved` | `gen.constrain_types(CURVED_TYPES)` |
| ⑩ | Pareto front — the trade-off set, not the weighted sum | `--pareto` | `pareto_objects(objs, gen.scorer, keys=(...))` |
| ⑪ | targeted size | `--target-size 0.05` | `gen.target_size(0.05)` |
| ⑫ | text2geometry — prompt → config → CEM | `--prompt "..."` | `generate_from_text("...")` |

They compose: ⑧ = ⑦ + a seed; ⑫ composes ⑦/⑧/⑨/⑪ from one sentence. For full control (or to embed
in your own code), the equivalent inline snippet is in **GUIDE.md §3.5**.

---

## What exists today

### ④ Free CEM generation — *the* generator
`generator.RoboticObjectGenerator` + `cem.ParameterDistribution`/`CEMOptimizer`.
Samples a composite object (base primitive + parts attached to random existing parts →
connected-by-construction), scores it, keeps the top-20% elites, and updates the learned
distribution (`primitive_type_probs`, per-type `type_log_means`, `n_primitives_probs`,
placement spreads). **Structure and parameters are both optimized**, against the multi-objective
score. It does not target any specific object — it *discovers* shapes that are stable + graspable,
and **rejects ones that aren't**. Entry: `main.py generate --train`; method `cem` in
`build_eval_manifest.py`.

### ④ (baselines) — alternative searchers over the same free space
`baselines.run_baseline`: **CMA-ES**, **GA**, **random search** optimize the *same* composite
space with different search strategies (CMA-ES/GA = optimized; random = path ③). **fixed_cad**
exports raw archetypes (= path ①). These exist so the paper can attribute results to the *search*,
not just the representation. Methods `cmaes / ga / random_search / fixed_cad` in
`build_eval_manifest.py`.

### ① Archetype variants, raw
`build_archetype_manifest.py` (default) → `archetype_cem.ArchetypeDistribution`. Introspects a
hand-written factory's signature (`archetypes.ARCHETYPE_REGISTRY`, 105 templates) and samples
parameter variations **around the hand-set defaults — no optimization**. A `screwdriver` stays a
thin rod. Use: library coverage, data augmentation, "what does the archetype family look like."

### ② Archetype variants, CEM-tuned  ← the proposed "third path" (it's already here)
`build_archetype_manifest.py --train` → `archetype_cem.ArchetypeTrainer`. Runs a **per-archetype
CEM over that template's parameters**, maximizing the same `ObjectScorer` total. **Structure is
fixed (still a screwdriver); only its parameters are optimized.** This is exactly *"generate the
archetype variants through the optimizer."*

**Measured lift (15 raw vs 15 CEM-tuned variants, stability proxy → total):**

| archetype | stability raw → tuned | total raw → tuned |
|---|---|---|
| mug_like | 0.99 → 1.00 | 0.93 → 0.93 |
| allen_key | **0.05 → 0.64** | 0.71 → 0.84 |
| flashlight | **0.15 → 0.70** | 0.76 → 0.87 |
| screwdriver | 0.86 → 1.00 | 0.96 → 1.00 |

Two things this path makes visible:
1. **The optimizer genuinely lifts stability within a fixed design** (allen_key 0.05→0.64,
   flashlight 0.15→0.70) — it finds proportions (lower CoM, wider base) that the hand defaults missed.
2. **It exposes the proxy-vs-dynamics gap.** The score is a *static* stability proxy (CoM inside the
   support polygon). The optimizer can drive it to 1.0 while the object still **tips in the dynamic
   Gazebo drop** — e.g. screwdriver proxy 0.86→1.00 but 0% drop-stable. This is the single most
   useful diagnostic for the stability claim: it separates "the optimizer is working" from "the
   proxy predicts dynamics."

### ⑤ Constrained / strategy generation
`gen_strategies.py`: path ④ run under **constraints** — palette-restricted (curved-only,
faceted-only), single-type ×N, all type-pairs, one-of-each, bilateral symmetry. Currently these
**sample** from a masked distribution rather than retrain (CEM training re-introduces excluded
types via epsilon-smoothing), so they're "constrained raw" until path ⑦ below is added.

---

## What could be added (same lens)

Ordered by how directly each addresses the stability claim. **⑥–⑧ are now built.**

- **⑥ Stability-repair / projection.** ✅ `src/stability_repair.py` re-orients any object onto
  its most-stable resting pose (the convex-hull facet with the largest tip-over margin), so it
  settles upright **by construction** — the bridge from "optimizes a proxy" to "guarantees it".
  `GeneratorConfig.repair_stability`. Validated: screwdriver 11°→27°, allen_key 0°→64°, pencil
  4°→38° (re-oriented to lie flat); funnel (no better pose) left alone.
- **⑦ Dynamic-stability-in-the-loop.** ✅ A tip-over metric `atan(margin / COM-height)`
  (`ScoringConfig.dynamic_stability`) — a fast geometric surrogate (pybullet is unusable here:
  NumPy-1-vs-2 ABI), corr 0.70 with the Gazebo drop. As a *soft* term it's diluted (no population
  shift); as a **gate** (`dynamic_stability_gate`, total ×= clip(tip-stability, 0.2, 1)) it works:
  free-CEM tippy tail 5/24 → **0/24**, mean tip-angle 28° → **49°**.
- **⑧ Archetype-seeded free CEM (warm-start).** ✅ `ParameterDistribution.seed_from_object` /
  `RoboticObjectGenerator.seed_from(name)` bias the *free* distribution's type/count/size priors
  toward a seed, then `train()` lets **structure evolve**. Seeded from the screwdriver + gate, the
  population reaches mean tip-angle 38.8°, 0/12 tippy (vs the screwdriver's ~11°) — it *deviates*
  toward stability instead of being trapped, cleanly separating "tune the design" (②) from
  "redesign it" (⑧).
- **⑨ Constrained *optimized* generation.** ✅ `ParameterDistribution.type_mask` +
  `RoboticObjectGenerator.constrain_types(keys)` make the CEM update mask-aware (the
  epsilon-smoothing no longer leaks excluded types), so a palette-constrained generator can be
  *trained*, not just sampled. Validated: training a curved-only generator keeps faceted mass at
  0.0000 and emits only curved objects.
- **⑩ Pareto / multi-objective front.** ✅ `src/pareto.py` (`pareto_front` / `pareto_objects`)
  returns the non-dominated trade-off set over chosen objectives (e.g. stability vs graspability)
  instead of the baked-in weighted sum, so you pick the operating point.
- **⑪ Conditional / targeted generation.** ✅ `ScoringConfig.target_extent` /
  `generator.target_size(m)` peak the size score at a target overall size. *Caveat:* it's a soft
  ~12%-weighted term (biases, doesn't tightly hit the target in a short run — up-weight / size-gate
  / longer train to tighten, same lesson as ⑦).
- **⑫ Text-conditioned (text2geometry).** ✅ `src/text2gen.py` (`generate_from_text(prompt)`) maps a
  prompt to a config by composing ⑦/⑧/⑨/⑪ — archetype seed, palette, target size, stability/grasp
  gates — then hands off to the free CEM. Keyword rules, no neural model: "a small stable graspable
  curved bottle" → seed=bottle + curved palette + 4 cm + stable+grasp gates. The optimizer is the
  *decoder* of a text interface.

**All twelve paths are now built.**

---

## Practical takeaway for the claims

- Report numbers **per path**. "The generator is X% stable" should mean **path ④**, not ① or ②.
- The honest, defensible framing is **"optimizes" / "substantially improves"**, not "guarantees":
  path ④ is multi-objective (trades some stability for graspability) and optimizes a *static proxy*.
  Validation = the dynamic Gazebo drop (CEM 88% vs un-optimized CAD 30%).
- Adding **② as a reported third arm** is worthwhile *precisely because* it isolates optimizer-lift
  within a fixed structure and surfaces the proxy-vs-dynamics gap — and **⑥/⑦** are the changes that
  would let you legitimately strengthen "optimizes" toward "guarantees."
