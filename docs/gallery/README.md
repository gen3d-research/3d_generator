# Object gallery

Regenerate with: `python scripts/render_gallery.py` (from `3d_generator/`).

## Archetype library (`archetypes.png`)

All 80 hand-written archetypes in `ARCHETYPE_REGISTRY` (`src/archetypes.py`),
built from the 9 primitive types (box, cylinder, sphere, capsule, cone, pyramid,
torus, ellipsoid, wedge). Each is a single connected body.

![archetypes](archetypes.png)

## v2 generated samples (`v2_samples.png`)

Objects sampled from a trained v2 generator (`max_primitives=8`). Titles show
the primitive count per object — note the multi-part composites mixing several
primitive types, in contrast to v1's single-primitive output.

![v2 samples](v2_samples.png)
