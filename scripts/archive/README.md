# Archived (dead) code

Kept for reference only — nothing imports or invokes these:

- `archetype_config.py` — superseded by `archetype_cem.ArchetypeDistribution`'s
  signature introspection (the path that actually runs; see CLAUDE.md).
- `analysis_figures.py` + `regen_eval_figures.py` — a self-contained pair of stale
  figure generators superseded by `generate_figures.py` / `render_*.py`; only
  referenced each other.

Verified unreferenced (grep over *.py, *.sh, *.md) before archiving. The audit's other
"dead" candidates were NOT archived: `evaluate_methods.py` and `generate_figures.py`
are used by `reproduce_paper.sh`; `verify_v2.py`, `sweep_v2.py`, `run_sim_eval_v2.py`,
and `sweep_sim_stability.py` are documented workflows (docs/sim_eval_v2.md,
docs/results_v2.md).
