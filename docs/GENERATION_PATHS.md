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

Ordered by how directly each addresses the stability claim.

- **⑥ Stability-repair / projection.** A post-hoc local optimizer that nudges *any* object's
  placement/params until a **hard** stability (and grasp) constraint holds — projecting onto the
  feasible manifold. Turns "optimizes a proxy" into "guarantees by construction," for archetypes
  *and* free generations. Closest fix to the "guarantee" wording.
- **⑦ Dynamic-stability-in-the-loop.** Replace/augment the static proxy with a fast settle check
  (a few-step physics rollout, or a learned surrogate) so the optimizer sees what the drop test
  sees. Closes the gap path ② exposes; would raise CEM's 88% dynamic-stable toward the proxy's ~100%.
- **⑧ Archetype-seeded free CEM (warm-start).** Initialize the *free* distribution from an
  archetype's primitives, then let **structure evolve**. Bridges fixed↔free: the optimizer can start
  from a screwdriver and *deviate* toward stability (grow the handle into a base) instead of being
  trapped in screwdriver-space. Cleanly separates "tune the design" (②) from "redesign it" (⑧).
- **⑨ Constrained *optimized* generation.** Combine ⑤'s constraints with training (mask-aware CEM
  update, skipping the epsilon-smoothing leak): "optimize a stable+graspable object using only
  curved primitives."
- **⑩ Pareto / multi-objective front.** Replace the weighted-sum score with an NSGA-II-style front
  over (stability, graspability, …) so you can *choose* the trade-off instead of baking in weights —
  and report where on the front the generator sits.
- **⑪ Conditional / targeted generation.** Condition on a target property vector (size class, grasp
  width, min stability margin) and optimize toward it — "make me a graspable, very-stable, 6 cm object."
- **⑫ Text-conditioned (text2geometry).** The workspace is literally `text2geometry_ws`, yet no
  text→shape path exists. The natural front-end: map a prompt to an **archetype seed + objective
  weights + constraints**, then hand off to path ④/⑧. This makes the existing optimizer the
  *decoder* of a text interface — no neural shape model required.

---

## Practical takeaway for the claims

- Report numbers **per path**. "The generator is X% stable" should mean **path ④**, not ① or ②.
- The honest, defensible framing is **"optimizes" / "substantially improves"**, not "guarantees":
  path ④ is multi-objective (trades some stability for graspability) and optimizes a *static proxy*.
  Validation = the dynamic Gazebo drop (CEM 88% vs un-optimized CAD 30%).
- Adding **② as a reported third arm** is worthwhile *precisely because* it isolates optimizer-lift
  within a fixed structure and surfaces the proxy-vs-dynamics gap — and **⑥/⑦** are the changes that
  would let you legitimately strengthen "optimizes" toward "guarantees."
