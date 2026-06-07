#!/usr/bin/env python3
"""
Prepare each generated object's SDF for the GENUINE (physics-only) pick demo:

* Keep the **mesh collision** (matches the visual mesh exactly, so the object
  rests flush on the table — no clearance). The old workaround replaced it with
  an AABB box because the trimesh OBJs had no vertex normals and DART rejected
  them; ``export.py`` now writes normals (``include_normals=True``), so the mesh
  loads fine and the box (which, lacking a <pose>, floated seated objects ~half
  their height) is gone.

* Bump the contact friction to mu=30 so the gripper's finger-squeeze grasp holds
  the object by friction under gravity.

* Add stiff/damped contact (<contact><ode><kp><kd>) so the fingers press on the
  mesh instead of sinking into / skating off it — firms up marginal holds.

* Strip any legacy DetachableJoint "weld" plugin — the grasp is now genuine
  (finger friction only), not a kinematic cheat.

Idempotent: re-running re-applies the same friction, adds contact only if absent,
and leaves no weld.
"""

import argparse
import json
import re
from pathlib import Path


_FRICTION_MU = 30.0

# Stiff, damped contact so the fingers PRESS on the mesh instead of sinking into
# it or skating off — firms up marginal friction holds. High kp (stiffness),
# moderate kd (damping); min_depth avoids jitter at shallow penetration.
_CONTACT_XML = (
    "<contact>\n"
    "            <ode><kp>1e6</kp><kd>1e2</kd>"
    "<max_vel>0.0</max_vel><min_depth>0.001</min_depth></ode>\n"
    "          </contact>\n          ")


def patch_sdf(sdf_path: Path, visual_mesh: Path = None) -> bool:
    sdf_text = sdf_path.read_text()
    orig = sdf_text

    # Firm up the contact friction so the genuine (physics-only) finger grasp
    # holds. The grasp is real: the gripper squeezes the object and high
    # friction holds it under gravity — there is no DetachableJoint weld.
    sdf_text = re.sub(r"<mu>[^<]*</mu>", f"<mu>{_FRICTION_MU}</mu>", sdf_text)
    sdf_text = re.sub(r"<mu2>[^<]*</mu2>", f"<mu2>{_FRICTION_MU}</mu2>", sdf_text)

    # Add stiff contact to each collision surface (idempotent: only if absent).
    if "<contact>" not in sdf_text:
        sdf_text = sdf_text.replace("<friction>", _CONTACT_XML + "<friction>")

    # Strip any previously-injected DetachableJoint weld (legacy cheat).
    sdf_text = re.sub(
        r"\s*<plugin[^>]*detachable-joint-system.*?</plugin>",
        "", sdf_text, flags=re.DOTALL)

    if sdf_text == orig:
        return False
    sdf_path.write_text(sdf_text)
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    args = p.parse_args()

    manifest = json.loads(args.manifest.read_text())
    n_patched = 0
    for e in manifest:
        sdf = Path(e["sdf"])
        vis = Path(e["visual_mesh"])
        if patch_sdf(sdf, vis):
            n_patched += 1
    print(f"patched {n_patched}/{len(manifest)} SDFs")


if __name__ == "__main__":
    main()
