# v2 results (multi-seed, paper-grade)

Independent-metric comparison of the generated distributions, 10 seeds (42–51),
25 objects/method/seed. Suitability is scored with the **v1 scorer**
(`assembly_weight=0`) so it is comparable across methods; grasp success and
diversity are independent of the scorer the CEM optimizes. Reproduce with:

```bash
python scripts/sweep_v2.py --seeds 10 --n 25 --iterations 18 --samples 50
```

| method | suitability (v1 scorer) | grasp success | feature diversity | chamfer diversity | mean parts |
|--------|------------------------:|--------------:|------------------:|------------------:|-----------:|
| v1 (paper_repro) | 0.995 ± 0.003 | 88.0 ± 6.9 % | 3.703 ± 0.160 | 0.0115 ± 0.0024 | 1.00 |
| v2-raw (no re-rank) | 0.878 ± 0.023 | 82.0 ± 7.7 % | 4.168 ± 0.077 | 0.0115 ± 0.0016 | 2.70 ± 0.35 |
| v2-final (full pipeline) | 0.876 ± 0.020 | 92.4 ± 3.1 % | 4.168 ± 0.131 | 0.0111 ± 0.0013 | 2.70 ± 0.35 |

(± = 95% t confidence interval over seeds.)

Paired Wilcoxon vs v1 (independent metrics):

| metric | comparison | mean diff | p |
|--------|------------|----------:|--:|
| grasp success | v2-raw vs v1 | −0.060 | 0.055 |
| grasp success | v2-final vs v1 | +0.044 | 0.164 |
| feature diversity | v2-raw vs v1 | +0.465 | **0.002** |
| feature diversity | v2-final vs v1 | +0.464 | **0.002** |
| chamfer diversity | v2-raw vs v1 | +0.000 | 0.557 |
| chamfer diversity | v2-final vs v1 | −0.000 | 0.557 |

## Findings (and corrections to the earlier single-seed snapshot)

1. **Feature diversity is significantly higher under v2** (+0.46, p = 0.002):
   v2 spans the full 9-type palette with ~2.7 parts/object vs v1's lone
   primitive. This is the robust headline.

2. **Chamfer (point-cloud geometric) diversity is NOT significantly changed**
   (p = 0.56). The single-seed run had suggested v2 ~doubled chamfer diversity;
   across 10 seeds that was a seed artifact. v2's diversity gain is in the
   shape-descriptor sense (type/count/ratio variety), not raw point-cloud
   spread.

3. **v2-final graspability is statistically indistinguishable from v1**
   (92.4 % vs 88.0 %, p = 0.16), with a *tighter* CI (±3.1 vs ±6.9). The
   grasp re-rank fully compensates for multi-part objects being harder to grasp;
   the earlier single-seed "drop to 83 %" was noise. The distribution-level
   effect before re-ranking (v2-raw, −6 %) is only borderline (p = 0.055).

4. **Suitability on the v1 scorer drops** (0.995 → 0.88) as expected: v2 trains
   on the assembly-augmented objective, so its objects no longer trivially max
   the v1 proxy.

**Net:** v2 delivers significantly more shape-descriptor diversity and richer
composites at no significant cost to graspability (after re-rank) — but it does
not increase raw geometric (chamfer) diversity. Report (1) and (3) as the
defensible claims; do not claim a chamfer-diversity gain.

## Dynamic stability (Gazebo Harmonic, single run)

See `docs/sim_eval_v2.md`: v1 and v2 both settled 100 % stable (tilt 0°, drift
within tolerance) on 6 objects/method. A multi-seed sim sweep is left as future
work (it needs `gz sim` orchestration per seed).
