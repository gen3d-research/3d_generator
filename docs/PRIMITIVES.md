# Primitive & Archetype Library — gallery, math, and fidelity audit

This document (a) catalogs the **18 primitive types** the generator builds objects
from — with a picture, parameters, degrees of freedom (DOF), clamp ranges, math,
and **limitations**; (b) audits the hand-written **archetypes** to show which
real-world shapes are currently *faked* because the primitive set is too small;
and (c) **proposes new primitives** to close those gaps. It also shows how the
limited primitive set hurts CEM **generation quality** (accepted vs rejected vs
optimized samples).

Regenerate every image here from `3d_generator/`:

```bash
python scripts/render_primitives.py          # docs/gallery/primitives.png
python scripts/render_gallery.py             # docs/gallery/archetypes.png, v2_samples.png
python scripts/render_samples.py             # docs/gallery/samples_{verdicts,optimized}.png
python scripts/render_primitives.py --emit-doc   # re-emit the spec table below
```

---

## 1. Primitive library

Each row varies one primitive type to illustrate its **degrees of freedom** — the
number of independent shape parameters, i.e. how many *different ways* a type can
vary. A `sphere` is 1-DOF (only its radius changes); a `box` is 3-DOF.

![Primitive gallery](gallery/primitives.png)

### Parameters, DOF, and clamp ranges

Pulled directly from the CEM spec table `PRIMITIVE_SPECS` in `src/cem.py`
(sizes in metres; parameters are sampled in **log-space** then exponentiated and
clamped to `[min, max]`):

| Primitive | DOF | Parameters | default (m) | clamp min | clamp max |
|---|---|---|---|---|---|
| box | 3 | dx, dy, dz | 0.05, 0.05, 0.06 | 0.01, 0.01, 0.01 | 0.15, 0.15, 0.15 |
| cylinder | 2 | radius, height | 0.025, 0.06 | 0.005, 0.01 | 0.08, 0.15 |
| sphere | 1 | radius | 0.03 | 0.01 | 0.08 |
| capsule | 2 | radius, height | 0.015, 0.04 | 0.005, 0.01 | 0.05, 0.12 |
| cone | 2 | radius, height | 0.025, 0.06 | 0.008, 0.02 | 0.07, 0.14 |
| pyramid | 2 | radius, height | 0.03, 0.05 | 0.01, 0.02 | 0.08, 0.14 |
| torus | 2 | major, minor | 0.04, 0.012 | 0.025, 0.005 | 0.08, 0.02 |
| ellipsoid | 3 | rx, ry, rz | 0.04, 0.03, 0.02 | 0.01, 0.01, 0.01 | 0.08, 0.08, 0.08 |
| wedge | 3 | width, depth, height | 0.05, 0.04, 0.04 | 0.015, 0.015, 0.015 | 0.14, 0.14, 0.14 |
| hollow_shell ✨ | 4 | outer, wall, height, floor | 0.035, 0.004, 0.07, 0.005 | 0.012, 0.002, 0.02, 0.002 | 0.06, 0.01, 0.14, 0.012 |
| handle ✨ | 4 | major, tube_a, tube_b, arc | 0.02, 0.006, 0.005, 4.71 | 0.01, 0.003, 0.003, 1.88 | 0.05, 0.012, 0.012, 5.97 |
| frustum ✨ | 3 | r_bot, r_top, height | 0.04, 0.025, 0.06 | 0.008, 0.005, 0.02 | 0.08, 0.08, 0.14 |
| hemisphere ✨ | 1 | radius | 0.03 | 0.012 | 0.08 |
| hex_prism ✨ | 2 | radius, height | 0.018, 0.012 | 0.006, 0.004 | 0.04, 0.05 |
| open_tube ✨ | 3 | outer, wall, height | 0.02, 0.005, 0.05 | 0.008, 0.002, 0.015 | 0.05, 0.012, 0.14 |
| ngon_prism ✨ | 3 | n_sides, radius, height | 5, 0.02, 0.03 | 3, 0.008, 0.01 | 8, 0.05, 0.08 |
| rounded_box ✨ | 4 | dx, dy, dz, fillet | 0.06, 0.04, 0.03, 0.008 | 0.02, 0.02, 0.015, 0.003 | 0.12, 0.12, 0.1, 0.015 |
| gear_prism ✨ | 4 | n_teeth, r_outer, r_inner, height | 8, 0.03, 0.022, 0.015 | 5, 0.015, 0.01, 0.006 | 12, 0.05, 0.04, 0.04 |

**Total: 48 sampled shape parameters across 18 types** (✨ = the audit-driven
additions: v2.2 hollow_shell + handle, v2.3 frustum + hemisphere, v2.5 hex_prism,
v2.6 open_tube + ngon_prism, v2.7 rounded_box + gear_prism). Each primitive also
carries a 6-DOF `Transform` (position + orientation) set during composition.

### Math & construction (per type)

`to_mesh()` and inertia live in `src/primitives.py`. Inertia is either an
**exact analytic tensor** or **derived from the mesh** (`_mesh_inertia`).

| Primitive | Volume | Mesh construction | Inertia |
|---|---|---|---|
| **Box** | `dx·dy·dz` | `trimesh.creation.box` | analytic `m/12·diag(b²+c², a²+c², a²+b²)` |
| **Cylinder** | `π r² h` | `trimesh.creation.cylinder` | analytic (`Ixx=Iyy=m/12(3r²+h²)`, `Izz=m/2 r²`) |
| **Sphere** | `4/3 π r³` | `icosphere(subdiv=3)` | analytic `2/5 m r²` (isotropic) |
| **Capsule** | cyl+2 hemi | `trimesh.creation.capsule` | analytic (cylinder approximation) |
| **Cone** | `1/3 π r² h` | `trimesh.creation.cone(sections=32)`, recentred | `_mesh_inertia` |
| **Pyramid** | `1/3 base·h` | `cone(sections=4)`, recentred | `_mesh_inertia` |
| **Torus** | `2π² R r²` | `trimesh.creation.torus` | `_mesh_inertia` |
| **Ellipsoid** | `4/3 π rx·ry·rz` | unit `icosphere` scaled by `radii` | analytic `m/5·diag(b²+c², …)` |
| **Wedge** | `½ w·d·h` | manual triangular-prism vertices/faces | `_mesh_inertia` |
| **HollowShell** ✨ | `π(R²H − Rᵢ²(H−floor))` | outer cylinder **minus** inner cavity (CSG / manifold3d) | `_mesh_inertia` |
| **Handle** ✨ | `π·a·b·R·arc` | manual elliptical-tube sweep along a circular arc, fan-capped | `_mesh_inertia` |
| **Frustum** ✨ | `π h/3 (r₀²+r₀r₁+r₁²)` | cone **intersected** with a clip box at z=H (CSG) | `_mesh_inertia` |
| **Hemisphere** ✨ | `2/3 π r³` | `icosphere` **intersected** with the z≥0 half-space (CSG) | `_mesh_inertia` |
| **HexPrism** ✨ | `3√3/2 · r² h` | `cylinder(sections=6)` | `_mesh_inertia` |
| **OpenTube** ✨ | `π(R²−Rᵢ²) h` | cylinder **minus** cylinder (open both ends, CSG) | `_mesh_inertia` |
| **NGonPrism** ✨ | `n/2 · r² sin(2π/n) · h` | `cylinder(sections=n)`, n∈3..8 | `_mesh_inertia` |
| **RoundedBox** ✨ | `lwh + 2r(lw+lh+wh) + πr²(l+w+h) + 4/3πr³` | **convex hull** of 8 corner spheres (= box⊕ball) | `_mesh_inertia` |
| **GearPrism** ✨ | `n·r₀·rᵢ·sin(π/n)·h` | manual alternating-radius star prism (fan-capped) | `_mesh_inertia` |

### Limitations (what each type **cannot** represent)

This is the crux of the audit — every limitation below forces an archetype to
*fake* a real feature:

- **Box / Wedge** — flat faces only; no curves or fillets.
- **Cylinder** — straight walls, **solid** (no bore/cavity), constant radius (no taper).
- **Sphere / Ellipsoid** — closed convex blobs; **solid**, no opening or shell.
- **Capsule** — fixed hemispherical caps; can't make a flat-ended rounded bar or a cavity.
- **Cone / Pyramid** — always taper to a **point** (apex); cannot make a *truncated* (flat-top) taper such as a flowerpot, bucket, or flared cup.
- **Torus** — a **full closed ring** only; cannot make a C-shaped handle, a hook, or a headband (a partial arc), and its cross-section is **circular** only.
- *(none)* a **hollow / shelled** body of any kind, a **partial arc**, a **truncated cone**, or a **dome/hemisphere** — these capabilities are simply absent.

---

## 2. Archetype library & fidelity audit

The 96 hand-written archetypes (`src/archetypes.py` + 20 v1 factories in
`src/primitives.py`) are the "ground-truth" shapes the generator is meant to
approximate. Multi-part archetypes deliberately **overlap** their parts so the
union is one connected body.

![Archetype gallery](gallery/archetypes.png)

Most archetypes are **faithful** — boxes/bars/plates/posts and their assemblies
(tools, electronics, fasteners, toys) are genuinely box/cylinder/capsule shapes
and look right. But every **container, handle, taper, and dome is faked** with a
solid or full-ring stand-in. The audit clusters the gaps:

### Faked shapes → missing primitive

| Archetype(s) | Real object | How it's faked now | Missing primitive |
|---|---|---|---|
| `mug_like`, `cup`, `pot`, `jar`, `bowl` | open-top vessel with walls | **solid cylinder** body (no cavity) | **Hollow shell / open container** |
| `teapot`, `wine_glass` | hollow round body | **solid ellipsoid** | Hollow shell (round) |
| `funnel` | open cone | **solid cone** | Hollow cone / open container |
| `mug_like` handle | C-shaped handle | **horizontal cylinder** (straight bar!) | **Handle / partial arc** |
| `cup`, `pot`, `teapot` handles | C-shaped handle | **full torus** (closed ring) | Handle / partial arc |
| `kettlebell` handle, `hook` | bail / J-hook | **full torus** | Handle / partial arc |
| `headphones` band | headband | **full torus** | Handle / partial arc (half) |
| `plunger` cup, `trophy` cup | flared bell / open cup | **solid cone** (apex) | **Truncated cone / frustum** |
| `ladle` scoop, `teapot` lid | dome / scoop | **solid sphere / capsule** | **Hemisphere / dome** |
| `nut`, `bolt` head | hex fastener | **round cylinder / 4-gon pyramid** | **n-gon prism** (hex) |

Rings and through-holes that genuinely *are* tori — `washer`, `tape_roll`,
`ring_stack`, `cutting_board` hole, `binder_ring` — are **not** faked; the torus
is the correct primitive there.

---

## 3. How the limited set hurts generation quality

Random assemblies of the current primitives are blobby, and the score/connectivity
gate rejects a fraction outright. Raw samples from an **untrained** distribution
(green = accepted by the quality gate, red = rejected with reason):

![Sample verdicts](gallery/samples_verdicts.png)

After CEM training, the surviving **optimized** set is cleaner, but still limited
to what 9 solid primitives can express — note the absence of any believable cup,
mug, or handled object:

![Optimized samples](gallery/samples_optimized.png)

The takeaway: the generator can only ever assemble *solid* primitives, so it
cannot discover hollow or handled objects no matter how long it trains. Closing
that gap requires **new primitives**, not more CEM iterations.

### Why the optimized set is full of cones, pyramids, and wedges

The CEM *does* learn a primitive-type distribution (`primitive_type_probs`,
updated each iteration from the elite frequencies in `cem.py`), so a natural
question is why it keeps picking pointed primitives that are, by design, hard to
grasp. Measuring the scorer per single primitive:

| primitive (alone) | graspability | total score | accepted (≥0.40)? |
|---|---|---|---|
| cone, pyramid | **0.00** | 0.67 | **yes** |
| wedge | 1.00 | 0.87 | yes |
| box / cylinder / sphere / torus / ellipsoid | 1.00 | 0.87 | yes |

Three things conspire:

1. **Graspability is only ~18 % of the total** (weight `1.5 / 8.5`). A cone with
   *zero* graspability still scores **0.67** — comfortably above the 0.40 accept
   threshold — because size, stability, validity and assembly carry it.
2. **Inside a composite, a pointed part is "free".** Graspability is measured
   over the whole object's surface, so `cylinder + cone-tip` scores
   graspability **1.00** — the cylinder supplies the antipodal pair and the cone
   costs nothing. The *assembly* bonus actively rewards adding such parts (noses,
   spikes, feet).
3. **Wedges aren't penalized at all** — the proxy treats the two parallel
   triangular faces as a valid antipodal pair (1.00). (A real parallel-jaw
   gripper on the *sloped* faces would struggle, but the proxy finds the flat
   ones.)

So cone/pyramid/wedge-bearing composites readily become elites, and the learned
`primitive_type_probs` never suppresses those types. This is a **scoring** issue,
separate from the missing-primitives issue above.

**✅ Fixed (v2.4) — the low-graspability gate.** The total score is now multiplied
by a **part-aware graspability gate**: `total ×= clip(g_whole · g_parts, 0.25, 1)`,
where `g_parts` is the **volume-weighted** per-type graspability (cone/pyramid ≈ 0.2,
everything else 1.0). So an ungraspable part is no longer "free": a *lone* cone is
gated below the accept threshold (0.67 → 0.17), and a composite is penalized in
proportion to how much of its **volume** is pointy (a small decorative tip barely
matters; a big cone drops the score a lot). Fully-graspable objects (a box, a mug)
are unchanged (gate = 1.0), so v1 / `paper_repro` scoring is preserved — the gate is
opt-in (off in `ScoringConfig` by default, on in the v2 generator). **Measured
effect:** trained-CEM cone+pyramid usage fell from ~4% → ~1% of parts, and the
optimized type mix shifted to all-graspable shapes (box, hemisphere, frustum,
sphere, handle).

---

## 4. Proposed new primitives (design sketches)

Five proposals, ordered by impact, derived from the audit clusters. Each was
**feasibility-probed** (read-only) in the project venv: a candidate mesh was
built and checked for watertightness, single body, and finite inertia.

> **Construction constraint (important).** The project venv has `manifold3d`
> (boolean union works) but **no `shapely` / `networkx` / triangulation engine**,
> so `trimesh.creation.revolve`, `Trimesh.slice_plane`, `Trimesh.fix_normals`,
> and `trimesh.creation.sweep_polygon` are all unavailable. The dependency-safe
> pattern — already used by `Wedge` — is a **manual surface-of-revolution**:
> build vertices/faces by hand with correct winding and `Trimesh(..., process=True)`.
> All five probes below pass with that pattern (numpy + trimesh only).

Adding any of these is the standard two-step: a `@dataclass` subclass in
`primitives.py` (set `ptype`, `to_mesh`, `volume`, `inertia_tensor`) + one
`PRIMITIVE_SPECS` row in `cem.py` + a `PrimitiveType` enum entry, then a `_helper`
in `archetypes.py` and a rewrite of the affected archetypes.

> **Status: ALL FIVE proposals are now IMPLEMENTED** — hollow shell + handle
> (v2.2), frustum + hemisphere (v2.3), hex prism (v2.5) — see `primitives.py`, the
> `PRIMITIVE_SPECS` rows in `cem.py`, and the rewired archetypes
> (`mug_like`/`cup`/`pot`/`teapot`/`jar`/`bowl` + `plunger`/`trophy`/`ladle` +
> `nut`/`bolt`). The primitive set went 9 → 14.

### 4.1 Hollow shell / open container  ★ ✅ IMPLEMENTED
- **Replaces:** solid-cylinder bodies of `mug_like`, `cup`, `pot`, `jar`, `bowl`; extends to `teapot`/`wine_glass` (round variant) and `funnel` (open cone).
- **Params (4 DOF):** `outer_radius`, `wall_thickness`, `height`, `floor_thickness`.
- **Math:** `V = π·height·outer_r² − π·(height−floor)·(outer_r−wall)²`; inertia via `_mesh_inertia`.
- **Construction (as built):** outer cylinder **minus** an inner cavity cylinder via CSG (`manifold3d`). A *manual* surface-of-revolution was tried first but its inner/outer walls wind oppositely, corrupting `center_mass` (no `networkx` here to repair winding); the CSG path yields a clean, consistently-wound manifold. Volume error **−0.3%**.
- **⚠️ Connectivity gotcha (found during integration):** **don't SEAL the cavity.** Capping a hollow shell with a solid lid traps the interior as an enclosed void, so the union mesh has two disconnected surfaces (outer + inner) and `is_connected()` reports **False** (it counts surface components). Mugs/cups/pots stay open → fine. The `jar` uses an **open ring/neck band** (torus) instead of a sealing lid for exactly this reason. A handle/spout that *pierces the wall* connects normally.
- **Spec row (as built):** `['outer','wall','height','floor']`, defaults `_log(0.035,0.004,0.07,0.005)`, clamp `[0.012,0.002,0.02,0.002]…[0.06,0.01,0.14,0.012]`.

### 4.2 Handle / partial arc  ★ ✅ IMPLEMENTED (elliptical tube, 4 DOF)
- **Replaces:** the faked handles of `mug_like` (straight cylinder!), `cup`, `pot`, `teapot`, `kettlebell`, `hook`, `headphones`.
- **Params:** circular tube **3 DOF** `major_radius, tube_radius, arc_angle` (recommended) — or elliptical tube **4 DOF** `+tube_b` (matches the "swept ellipse" idea; flatter, more realistic, one extra CEM param).
- **Math:** `V ≈ π·tube_r²·major_r·arc_angle` (full torus = `arc=2π`); inertia via `_mesh_inertia`.
- **Construction:** manual revolve of the tube cross-section over `φ ∈ [−arc/2, +arc/2]`, fan-capping both open ends; then recenter on the mesh centroid (project convention).
- **Feasibility probe:** watertight ✓, 1 body, finite inertia ✓ (winding must be made outward-consistent — the probe's signed volume was negative until faces are wound correctly; magnitude error ~1.5%).
- **Spec row template:** `['major','tube','arc']`, defaults `_log(0.02,0.006,1.5π)`, clamp `[0.01,0.003,0.6π]…[0.05,0.012,1.9π]` (never a full ring → avoids degenerate self-touching caps).
- **Risk:** must overlap the body for connectivity (inner edge `major−tube` penetrates the wall by ~`tube_radius`); cap `arc < 1.9π`.
- **Recommendation:** start with the **circular tube (3 DOF)**; add the elliptical variant later if handle realism warrants it.

### 4.3 Truncated cone / frustum  ◆ ✅ IMPLEMENTED
- **Replaces:** `plunger` cup, `trophy` cup; enables flared cups, buckets, flowerpots, lampshades (none expressible today — cone only tapers to a point).
- **Params (3 DOF):** `radius_bottom`, `radius_top`, `height`.
- **Math:** `V = π·h/3·(r0² + r0·r1 + r1²)`; inertia via `_mesh_inertia`.
- **Construction:** revolve the trapezoid `(0,0)→(r0,0)→(r1,h)→(0,h)`.
- **Feasibility probe:** watertight ✓, 1 body, finite inertia ✓, volume error **−0.3%**.
- **Spec row template:** `['r_bot','r_top','height']`, defaults `_log(0.04,0.025,0.06)`, clamp `[0.008,0.005,0.02]…[0.08,0.08,0.14]`.
- **Risk:** when `r_top→0` it degenerates to a cone (fine); keep both radii `≥ 5 mm`.

### 4.4 Hemisphere / dome  ◆ ✅ IMPLEMENTED
- **Replaces:** `ladle` scoop, `teapot` lid, dome lids; an alternative hollow-`bowl` body.
- **Params:** `radius` (1 DOF) solid dome, or `radius, thickness` (2 DOF) hollow bowl-cap.
- **Math:** solid `V = 2/3 π r³`; inertia via `_mesh_inertia`.
- **Construction:** revolve a quarter-circle profile `(0,0)→arc→(0,r)` around Z (a flat-bottomed half-sphere).
- **Feasibility probe:** watertight ✓, 1 body, finite inertia ✓, volume error **−0.7%**.
- **Spec row template:** `['radius']`, defaults `_log(0.04)`, clamp `[0.012]…[0.08]`.
- **Risk:** minimal; it's a clipped sphere of revolution.

### 4.5 n-gon prism (hex)  ▫ ✅ IMPLEMENTED
- **Replaces:** `nut` body, `bolt` head (currently round cylinder / 4-gon pyramid).
- **Params (2 DOF + fixed n):** `radius`, `height` (n=6 hex by default).
- **Math:** regular-polygon prism `V = ½ n r² sin(2π/n)·h`; analytic or `_mesh_inertia`.
- **Construction:** `trimesh.creation.cylinder(radius, height, sections=6)` — trivially watertight.
- **Feasibility probe:** watertight ✓, 1 body, finite inertia ✓.
- **Spec row template:** `['radius','height']`, defaults `_log(0.018,0.012)`, clamp `[0.006,0.004]…[0.04,0.05]`.
- **Risk:** none; lowest priority (cosmetic for fasteners).

### Summary

| Proposal | DOF | Fixes | Construction | Status |
|---|---|---|---|---|
| Hollow shell | 4 | mug, cup, pot, jar, bowl | CSG cylinder − cavity | ✅ **built (v2.2)** |
| Handle / arc | 4 | mug, cup, pot, teapot | manual elliptical-tube sweep | ✅ **built (v2.2)** |
| Frustum | 3 | plunger, trophy, buckets, flowerpots | cone ∩ clip box (CSG) | ✅ **built (v2.3)** |
| Hemisphere | 1 | ladle, lids, domes | sphere ∩ half-space (CSG) | ✅ **built (v2.3)** |
| Hex prism | 2 | nut, bolt | `cylinder(sections=6)` | ✅ **built (v2.5)** |

**Done:** v2.2 hollow shell + handle (mug/cup/pot/teapot/jar/bowl) and v2.3
frustum + hemisphere (plunger/trophy/ladle) — all rebuilt and connected; see the
archetype gallery. **All five proposals are now built** — the audit is fully closed.
