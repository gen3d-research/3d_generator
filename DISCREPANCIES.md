# Known code ↔ paper discrepancies

> **v2 UPDATE.** The codebase has moved to **v2**: an unbounded/configurable
> primitive count, 5 new primitive types (cone, pyramid, torus, ellipsoid,
> wedge → 9 total), ~80 archetypes via a central registry, and the corrections
> below folded in **as defaults**. The original ICARM-paper code is preserved on
> the git branch **`legacy/v1-icarm`** (tag `v1-icarm`). To reproduce the v1
> behavior from v2 code, use `generator.paper_repro_generator()` (4 primitives,
> original 4 types, analytic inertia, no connectivity/grasp post-processing).
>
> Status of the items below under v2 defaults:
> - **Item 2 (overlap-aware inertia):** now **ON** by default (`use_mesh_inertia=True`).
> - **Item 5 (sphere stability):** **fixed** (adaptive ground band).
> - **Item 6 (floating parts):** now rejected by default (`require_connected=True`),
>   and sampling is connected-by-construction.
> - **Item 7 (learned placement):** the CEM now learns `offset_std`/`rotation_std`.
> - **Item 8 (grasp proxy):** generation now re-ranks by the independent planner
>   (`rerank_by_grasp=True`).
> - **Item 4 (encoding parity):** baselines gained a `multitype` encoding over all
>   types with counts {1..4}.
> - **Item 3 (stability COM source):** still analytic COM in the CEM hot path
>   (kept for speed); the support polygon is mesh-based. Minor; documented.
>
> The historical record below describes the situation **at v1 / paper submission**.

---

## 1. `reproduce_paper.sh` / docs pointed at a non-existent entrypoint *(fixed, no number change)*

The script invoked `python 3d_generator/main.py`, but the CLI lives at
`src/main.py`; `CLAUDE.md` documented `python main.py` from `3d_generator/`.
Neither path exists, so the script died at step 2 under `set -e`. Fixed: the
script now `cd`s to its own directory and calls `python src/main.py`; the
LaTeX build is gated behind `PAPER_DIR`. This is a pure path fix — it does not
change any computed result.

## 2. Mass / COM / inertia double-count overlapping primitives *(opt-in fix)*

`CompositeObject.center_of_mass` / `combined_inertia` sum **per-primitive**
volumes (`total_volume()` explicitly "ignores overlap"). For objects whose
primitives overlap (dumbbell, snowman, hammer, most 2-primitive CEM samples)
this over-estimates mass and biases COM/inertia — yet `export.py` advertises
"physically consistent inertial properties." Measured example: an overlapping
dumbbell's summed mass is **15% higher** than the true union mass.

- **Locked behavior:** analytic summed inertia (what the exported URDFs and any
  mass-derived numbers in the paper used).
- **Opt-in correction:** `ExportConfig(use_mesh_inertia=True)` computes
  overlap-aware mass/COM/inertia from the boolean-union solid
  (`CompositeObject.mesh_mass_properties`). Falls back to analytic if the union
  is not watertight. Requires a CSG backend (`manifold3d`, now in
  requirements.txt).

## 3. Stability uses two inconsistent geometry models *(documented, not changed)*

`scoring._score_stability` builds the support polygon from **mesh** vertices but
takes the COM from the **analytic** (overlap-ignoring) `center_of_mass`
(`scoring.py:191`). The margin therefore mixes two geometry models. Left as-is
to preserve the locked stability scores; flagged for future correction (use the
union-mesh COM on both sides).

## 4. Baseline encoding is NOT "apples-to-apples" *(docstring fixed; opt-in parity added)*

`baselines.py` previously claimed all baselines shared one encoding. In reality:
RandomSearch / CMA-ES / GA search a fixed **2-box** 13-D space; FixedCAD perturbs
the archetype factories; and **Ours** (`cem.ParameterDistribution`) searches a
strictly richer space (1–4 primitives across 4 types). The diversity metric
counts primitive types directly (`diversity.py:60-65`), so part of the
Ours-vs-baseline **diversity gap is a representation artifact, not optimizer
quality**. Grasp-success-rate is the cleanest independent comparison.

- **Locked behavior:** baselines use `encoding="twobox"` (the published config;
  bit-for-bit unchanged).
- **Opt-in parity:** pass `encoding="multitype"` to the gradient-free baselines
  for a fixed-dim 22-D space with 1–2 primitives across 4 types. This narrows
  the gap; the residual (counts 3–4) cannot be expressed in a fixed-dim
  continuous vector for CMA-ES/GA. For a fully controlled study, instead
  restrict Ours to the two-box encoding.

## 5. Statistics: population std, no CIs, no significance test *(additive, point estimates unchanged)*

`aggregate_seeds.py` reported mean ± **population** std (`ddof=0`) over a handful
of seeds, with no confidence interval or significance test. The legacy `mean`
and `std` fields are **unchanged** (locked numbers reproduce). Added alongside:
sample std (`ddof=1`), SEM, a t-based 95% CI, and a paired Wilcoxon/t test of
CEM vs the best alternative on the **independent** metrics only (suitability is
excluded as circular). At n≈3 these tests are badly underpowered and are
reported with that caveat — they do not license a significance claim from the
locked seed count. **Recommendation for any new run: ≥10 seeds.**

## 6. Generated objects can have floating/disconnected parts *(opt-in fix)*

`cem.sample_object` only clamps a secondary primitive's `z ≥ 0.01`; nothing
enforces contact with the base, so physically unrealizable floating geometry can
pass the watertight validity check.

- **Locked behavior:** no connectivity filter.
- **Opt-in correction:** `GeneratorConfig(require_connected=True)` rejects
  objects whose boolean union has more than one body
  (`CompositeObject.is_connected`). Requires a CSG backend. NOTE: some hand-authored
  archetypes (e.g. the mug, whose handle only touches the body tangentially)
  read as disconnected under a strict union — verify archetypes before enabling
  this in an archetype-based run.

## 7. CEM never learns spatial arrangement *(documented, not changed)*

`cem._update_distribution` updates sizes/types/counts/friction but never
`offset_std` / `rotation_std`. Part placement stays at the prior — the "learned"
distribution does not adapt the arrangement of secondary primitives. Changing
this would alter the locked CEM outputs, so it is left for future work.

## 8. Scorer graspability is a loose antipodal proxy *(documented, not changed)*

`scoring._score_graspability` checks opposing normals + distance but not that the
contact line aligns with the normals (true antipodal geometry), and normalizes
by a magic constant `expected_quality = 30.0`. The CEM optimizes this loose
proxy; the rigorous force-closure check lives in `grasp_planner.py` and is used
only at evaluation. Suitability scores saturate near 1.0 as a result — see the
near-degenerate `score_mean` column in `unified_eval`.

## 9. `run_unified_eval.py` default budget *(fixed to match paper)*

The standalone default was 600 while the paper pipeline (`run_multi_seed.sh`)
passes 1500. The default is now 1500 so a standalone run matches the published
configuration.

---

### Environment dependencies discovered

`manifold3d` (CSG backend) and `rtree` (trimesh spatial index, required by the
grasp planner) were missing and are now pinned in `requirements.txt`. Without
`rtree` the unified evaluation crashes in `grasp_planner`; without a CSG backend
the items 2 and 6 corrections silently fall back to their analytic/permissive
defaults.
