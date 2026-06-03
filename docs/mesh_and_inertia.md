# Mesh generation & URDF inertia — analysis and v2.1 fixes

## URDF mass / inertia accuracy (audited)

| check | result |
|---|---|
| single box vs exact analytic | mass & Ixx/Iyy/Izz match to **0.00%** — frame (tensor about COM, link axes, rpy=0), formula, and units are correct |
| inertia validity, all 80 archetypes + 60 sampled objects | **100%** positive-definite, symmetric, satisfy the principal-moment triangle inequality; COM always inside the AABB |
| overlap handling | v2 union mass is correct; the v1/analytic path **overestimates overlapping objects by ~11%** (double-counts overlap) — fixed by `use_mesh_inertia=True` |
| non-watertight union fallback | **0/80** archetypes, **0%** of sampled objects fall back to the analytic path |

**The one real error was faceting** (curved primitives integrated from coarse
polyhedra): sphere/ellipsoid were ~−3.4% volume / ~−5.6% inertia at
`icosphere(subdivisions=2)`.

## Fixes applied (v2.1)

1. **Exact analytic inertia for single primitives** — when an object is one
   box / cylinder / sphere / ellipsoid, `mesh_mass_properties` returns the exact
   closed-form tensor (and `Ellipsoid` got an exact `m/5·diag(...)` tensor).
   Single sphere & ellipsoid mass/inertia error is now **0.00%**.
2. **Higher tessellation** — `icosphere` subdivisions 2→3 (sphere/ellipsoid
   faceting < ~1%), cone given 32 sections. Cylinders were already < 1%.
3. **Union visual/collision mesh** (`ExportConfig.union_visual_mesh=True`) — the
   exported visual/collision mesh is now the watertight boolean **union**, not a
   concatenation of overlapping primitives. Fixes the non-manifold "soup"
   (internal faces, self-intersections) seen in RViz/MoveIt and makes the
   rendered/collision geometry consistent with the mass/inertia mesh.
4. **Structured placement** (`ParameterDistribution.structured_placement=True`)
   — each non-base primitive is seated against a **face** of an existing one,
   axis-aligned, with an overlap margin, and the whole assembly is re-seated on
   the ground. Generated objects now look like coherent multi-part assemblies
   instead of randomly offset blobs. Legacy free-offset placement remains
   available (and is what `paper_repro_generator` uses).

Remaining notes (not bugs): the Gazebo SDF collision is still replaced by an
AABB box (`patch_sdf_collision.py`) as a coarse sim-contact proxy; density is a
uniform 1000 kg/m³; structured-placement connectivity is ~85% at sample time
(the `require_connected` filter keeps only connected objects at generation).

## Sweep re-run for v2.1

`docs/results_v2.md` has been re-run for the v2.1 structured-placement
generator. Summary: suitability rose 0.88 → 0.94 (cleaner assemblies), parts
≈ 3.0, the significant feature-diversity gain (p = 0.002) and graspability
parity after re-rank (p = 0.56 vs v1) both hold; chamfer diversity is at best
marginal. The v2.0 free-offset numbers are retained at the bottom of that file.
