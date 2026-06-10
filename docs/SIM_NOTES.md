# Simulation notes — shortcuts, semantics, and knobs

Everything in the Gazebo evaluation that is a deliberate simulation choice rather than a
physical fact. Read this before interpreting (or comparing) stability / grasp numbers.

## Collision representation (v2.9: per-primitive)

Exported SDFs/URDFs carry **one collision element per primitive**
(`ExportConfig.per_primitive_collision`, default on):

- `box / cylinder / sphere` → native analytic shapes, posed by the primitive transform.
- `capsule` → cylinder + 2 spheres (native everywhere; SDF < 1.8 has no `<capsule>`).
- everything else → that primitive's own **convex-hull** mesh piece (world frame).

Why: native/convex pairs bypass ODE's trimesh-trimesh collider, whose internal assert
(`collision_trimesh_trimesh.cpp`) killed gz worlds at scale whenever a failed despawn left
two mesh objects overlapping; analytic contacts are also far cheaper and cleaner.

Consequences to be aware of:
- **Concave types (torus, handle, hollow_shell, open_tube, gear, profiles) are hulled in
  gz collision** — a mug's cavity is sealed *for physics contact*. The MoveIt planning
  scene still loads the true mesh (`collision_mesh` in the manifest), so grasp planning
  into concavities is unaffected.
- **Mesh-contact noise is gone.** Tessellated contacts used to act as an accidental
  perturbation (toppling metastable objects, stopping spheres from rolling). Native
  contacts behave like ideal rigid bodies — see the spawn tilt below.
- Legacy single-mesh collision: `ExportConfig(per_primitive_collision=False)`.

## Drop-test (dynamic stability) semantics

- **Spawn**: `(0.5, 0, table_top + GZ_DROP_HEIGHT_M)`, default drop height 0.05 m.
- **Spawn tilt** (`stability.spawn_tilt_deg`, default **5°**; env `GZ_SPAWN_TILT_DEG`):
  an explicit, reproducible perturbation. With clean per-primitive contacts a perfectly
  vertical drop lets metastable objects (a screwdriver on its handle base) stand; mesh
  noise used to topple them *by accident*. The tilt restores discrimination on principled
  grounds. Validated: with 5° tilt, mesh- and per-primitive-collision verdicts agree
  11/12 (the disagreement is a kettlebell whose native sphere correctly *rolls*).
- **Settle**: 2.0 s of physics (`settle_time_s`) — slow tippers near the threshold may
  still be falling; raise it if you study marginal objects.
- **Stable iff** tilt < 25° (`upright_tolerance_deg`) AND |final z − table_top| < 5 cm
  (`drift_tolerance_m`, measured from the **expected rest height**, not the spawn).
  ~3% of objects land within ±5 mm of the drift threshold — prefer the tilt criterion
  when in doubt.

## Grasp-hold physics (the pick-and-place demo)

- `patch_sdf_collision.py` sets **μ = 30** and stiff contact (`kp=1e6, kd=1e2`,
  `min_depth=1mm`) on every object collision. **This is a sim shortcut, not a material
  property** (real surface friction is ~0.5–1.5): it lets the effort-controlled finger
  squeeze hold objects under gravity without a weld. If objects slip in sim, suspect
  this before blaming the grasp.
- Restitution 0.1; object friction values sampled at export (`μ ~ N(0.8, 0.2)` clipped
  to [0.1, 2.0]) are *overwritten* by the patch for the grasp evaluations.

## Comparability of result sets

Numbers are only comparable within one configuration of
{collision mode, spawn tilt, drift reference, settle time}:

| result set | collision | tilt | drift ref | settle |
|---|---|---|---|---|
| submitted paper (3 seeds) | single mesh | 0° | spawn point (pre-fix) | wall clock |
| corrected re-run 2026-06-10 (`aggregated_corrected.json`) | single mesh | 0° | table top | wall clock |
| 10,433-variant drop study | single mesh | 0° | table top | wall clock |
| v2.9 re-run (`aggregated_v29.json`) | per-primitive | 5° | table top | **sim time** |
| v2.9 10k study (`gazebo_stability_v29.json`) | per-primitive | 5° | table top | **sim time** |

Note on the two 10k studies: the v2.9 study (10,494 objects, 76% stable) additionally uses
the **tightened variant distribution** (recognizable ±12% variants, prototype-anchored) —
archetype-level comparable to the meshcol study (85% stable), not per-object. The band
structure sharpens under clean physics + explicit tilt: flat objects that mesh-contact
noise used to topple now rest at 100% (frying_pan, spatula, teapot, tall_box), while
rollers and thin rods are correctly punished to 0% (dumbbell, kettlebell, baseball_bat,
screwdriver, chisel) — "rolls or tips" is no longer masked by tessellation artifacts.

The settle column matters: the wall-clock `time.sleep` silently under-settles when the
sim's real-time factor drops under machine load (objects get queried frozen mid-air at
the spawn pose). `settle()` now waits on `/world/<W>/stats` **simulation time** — the
wall-clock result sets above were produced on an idle machine and spot-verified, but
any future run is load-independent by construction.

Cross-config robustness: the method ranking (CEM the only top-tier-on-all-three) is
identical under the corrected mesh-collision config (CEM 97.7/98.7/92.0) and the v2.9
config (CEM 98.0/97.3/88.0).
