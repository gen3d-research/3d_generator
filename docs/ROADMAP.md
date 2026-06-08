# Roadmap — what to add next

> **Update (v2.6):** the top two primitives (**open tube**, **n-gon prism**) and a
> first batch of **Group-G archetypes** (bucket, flower_pot, vase, wine_bottle,
> goblet, lamp, salt_shaker, pencil_cup, mortar, pipe, napkin_ring, octagon_nut,
> funnel_v2) are now **built** — 16 primitives, 93 archetypes. The remaining
> primitives (rounded box, gear/star, extrusion, helix) and archetypes below are
> still open.

The fidelity audit ([`PRIMITIVES.md`](PRIMITIVES.md)) took the primitive set from 9 → 14
(hollow_shell, handle, frustum, hemisphere, hex_prism) and added a graspability gate + a
per-type cap. This doc is the **menu of what's still worth adding**: (A) new *archetypes*
the 14-primitive set already unlocks, and (B) new *primitives* for the shapes that remain
un-expressible. Nothing here is implemented yet — it's a backlog to pick from.

Construction constraints for any new primitive (this venv): **manifold3d ✓** (CSG works),
**shapely / networkx ✗** (no 2-D polygon ops, no `trimesh.creation.revolve` / `slice_plane` /
`fix_normals` / `sweep_polygon`). So build via **CSG of existing solids** or a **manual
surface-of-revolution / triangle mesh** (the `HollowShell`/`Frustum`/`Hemisphere`/`Handle`
pattern). Keep containers **open** — a sealed cavity reads as two surfaces and fails
`is_connected()`.

---

## A. New archetypes unlocked by the current 14 primitives

Each is a multi-part composite (parts overlap → one connected body). "Leans on" = the
audit-added primitive that makes it look right.

### Containers & kitchen
| Archetype | Composition | Leans on |
|---|---|---|
| **bucket** | open `frustum` shell + `handle` (arc bail) | frustum + handle |
| **flower pot** | wide-top `frustum` + thin `torus` rim | frustum |
| **vase** | stacked `frustum` + `hollow_shell` neck | frustum / hollow_shell |
| **watering can** | `hollow_shell` body + `frustum` spout + `handle` + thin spout `cylinder` | hollow_shell + handle + frustum |
| **measuring cup** | `frustum` `hollow_shell` body + `handle` + `wedge` spout | hollow_shell + handle |
| **goblet / chalice** | `hemisphere` cup + `cylinder` stem + `frustum` foot | hemisphere + frustum |
| **mortar** | thick `hemisphere` / short wide `hollow_shell` | hemisphere |
| **coffee/French press** | `hollow_shell` carafe + `cylinder` plunger rod + `handle` | hollow_shell + handle |
| **salt / pepper shaker** | `hex_prism` or `cylinder` body + perforated `hemisphere` top | hemisphere + hex_prism |
| **wine bottle** | `cylinder` body + `frustum` shoulder/neck + small `cylinder`/`hemisphere` cap | frustum |

### Tools, desk, household
| Archetype | Composition | Leans on |
|---|---|---|
| **lamp** | `frustum` shade + `cylinder` stem + `hemisphere`/`cylinder` base | frustum + hemisphere |
| **pencil cup / desk caddy** | `hex_prism` `hollow_shell` (open) | hex_prism + hollow_shell |
| **nut driver / socket** | `hex_prism` `hollow_shell` socket + `cylinder` shaft | hex_prism + hollow_shell |
| **funnel (rebuild)** | `frustum` + `cylinder` neck (vs the current solid cone) | frustum |
| **trophy (rebuild)** | `hemisphere`/`frustum` cup + two `handle`s + `cylinder` stem + `box` base | hemisphere + handle |
| **birdhouse / canister** | `hex_prism` body + `pyramid`/`frustum` roof | hex_prism + frustum |

These reuse the existing `_shell`/`_handle`/`_frustum`/`_hemi`/`_hex` helpers in
`src/archetypes.py` — each is a ~5-line registered factory.

---

## B. New primitives still missing

Ordered roughly by value × effort. "Unlocks" lists archetypes/features it would enable.

| Primitive | Params | Construction | Effort | Unlocks |
|---|---|---|---|---|
| **open tube / pipe** ✅ | outer_r, wall, height | CSG: cylinder − cylinder (both ends open) | low | pipe, straw, napkin ring (built v2.6) |
| **n-gon prism** ✅ | n_sides, radius, height | `cylinder(sections=n)`, n∈3..8 | low | octagon nut, faceted bodies (built v2.6) |
| **rounded / fillet box** | dims, fillet_r | CSG (box ∪ edge cylinders ∪ corner spheres) or minkowski | medium | phone, tablet, controller, soap bar, key fob — much more realistic |
| **gear / star prism** | n_teeth, r_outer, r_inner, height | manual alternating-radius mesh | medium | real gears/sprockets (the current `gear_like` fakes teeth with boxes), star knobs |
| **dish / plate** | radius, depth, wall | flattened `hollow_shell` / shallow SoR cap | low | plate, saucer, shallow bowl, lid (or just clamp `hollow_shell` thin) |
| **arbitrary-polygon extrusion** | profile (L/U/T), height | needs a small triangulator (no shapely) | high | brackets, channels, extrusions, I-beams |
| **helix / spring** | coil_r, wire_r, pitch, turns | sweep a circle along a helix polyline (manual) | high | spring, coil, threaded look |
| **saddle (hyperbolic patch)** | width, depth, curvature | parametric grid mesh | high (niche) | ergonomic grips, seats |

**Already covered, do not add:** partial arcs/handles → `handle`; elbows → `pipe_elbow`;
tees/crosses → `cross_joint` (composite). A torus is the right primitive for washers/rings.

### Suggested order
1. **open tube** + **n-gon prism** (both low-effort, immediately useful, pure-numpy/CSG).
2. **rounded box** (high realism payoff for the many box-bodied electronics archetypes).
3. **gear/star prism** (fixes the faked `gear_like`).
4. The rest (extrusion / helix / saddle) only if a target archetype needs them.

---

*Generated alongside the strategy + variety galleries — see
[`gallery/`](gallery/) (`strategy_*.png`, `varieties_*.png`).*
