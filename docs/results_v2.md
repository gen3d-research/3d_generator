# v2 results (multi-seed, paper-grade)

Independent-metric comparison of the generated distributions, 10 seeds (42–51),
25 objects/method/seed. Suitability is scored with the **v1 scorer**
(`assembly_weight=0`) so it is comparable across methods; grasp success and
diversity are independent of the scorer the CEM optimizes. Reproduce with:

```bash
python scripts/sweep_v2.py --seeds 10 --n 25 --iterations 18 --samples 50
```

Numbers below are for the **v2.1 generator (structured placement)** — the
current default. (The earlier v2.0 free-offset numbers are kept at the bottom
for comparison.)

| method | suitability (v1 scorer) | grasp success | feature diversity | chamfer diversity | mean parts |
|--------|------------------------:|--------------:|------------------:|------------------:|-----------:|
| v1 (paper_repro) | 0.995 ± 0.004 | 88.8 ± 7.5 % | 3.741 ± 0.100 | 0.0118 ± 0.0021 | 1.00 |
| v2-raw (no re-rank) | 0.943 ± 0.012 | 80.0 ± 7.1 % | 4.212 ± 0.098 | 0.0144 ± 0.0009 | 3.02 ± 0.05 |
| v2-final (full pipeline) | 0.943 ± 0.017 | 86.4 ± 5.8 % | 4.219 ± 0.092 | 0.0142 ± 0.0011 | 3.04 ± 0.05 |

(± = 95% t confidence interval over seeds.)

Paired Wilcoxon vs v1 (independent metrics):

| metric | comparison | mean diff | p |
|--------|------------|----------:|--:|
| grasp success | v2-raw vs v1 | −0.088 | 0.148 |
| grasp success | v2-final vs v1 | −0.024 | 0.562 |
| feature diversity | v2-raw vs v1 | +0.471 | **0.002** |
| feature diversity | v2-final vs v1 | +0.478 | **0.002** |
| chamfer diversity | v2-raw vs v1 | +0.0026 | 0.049 |
| chamfer diversity | v2-final vs v1 | +0.0025 | 0.322 |

## Findings

1. **Feature diversity is significantly higher under v2** (+0.47, p = 0.002):
   v2 spans the full 9-type palette with ~3 parts/object vs v1's lone primitive.
   The robust headline.

2. **Graspability is statistically indistinguishable from v1** after re-rank
   (86.4 % vs 88.8 %, p = 0.56). The distribution-level effect before re-ranking
   (v2-raw, −8.8 %) is not significant either (p = 0.15) but trends lower —
   multi-part objects are somewhat harder to grasp, and the re-rank recovers it.

3. **Chamfer (geometric) diversity is at best marginal** — v2-raw +0.0026 is
   borderline (p = 0.049) and v2-final is not significant (p = 0.32). Do **not**
   claim a strong chamfer-diversity gain; the robust diversity result is the
   feature/descriptor one.

4. **Suitability** drops only modestly (0.995 → 0.943) and is *higher* than the
   v2.0 free-offset generator (0.88): structured, axis-aligned assemblies are
   more regular/stable and so score better than random blobs.

**Net (defensible claims):** v2 produces significantly more
shape-descriptor-diverse, multi-part composites (~3 parts, all 9 types) at no
significant cost to graspability — without overclaiming geometric (chamfer)
diversity.

## v2.0 (free random-offset placement) — superseded

Kept for the record; structured placement (v2.1, above) replaced it. v2.0 had
lower suitability (≈0.88) and flat chamfer diversity (p = 0.56). Across seeds the
two share the same headline: significant feature-diversity gain, graspability ≈
v1 after re-rank.

## Dynamic stability (Gazebo Harmonic, single run)

See `docs/sim_eval_v2.md`: v1 and v2 both settled 100 % stable (tilt 0°, drift
within tolerance) on 6 objects/method. A multi-seed sim sweep is left as future
work (it needs `gz sim` orchestration per seed).
