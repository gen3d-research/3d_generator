#!/usr/bin/env python3
"""
Prepare each generated object's SDF for the physical pick demo:

1. Keep the **mesh collision** (it now matches the visual mesh exactly, so the
   object rests flush on the table — no clearance). The old workaround replaced
   the collision with an AABB box because the trimesh OBJs had no vertex normals
   and DART rejected them; ``export.py`` now writes normals
   (``include_normals=True``), so the mesh loads fine as a collision shape and
   the box (which, lacking a <pose>, sat centered on the link origin and made
   seated objects float ~half their height above the table) is no longer used.
   We only bump the collision friction to mu=30 for firm, stable contact.

2. Inject a gz-sim DetachableJoint plugin (libgz-sim8-detachable-joint-system)
   so the visual_demo driver can weld the object to the gripper on attach and
   release it on detach. NOTE the child link must be ``panda_leftfinger``:
   ``panda_hand``/``panda_link8`` carry no <inertial> and are dropped by
   urdf2sdf, so a ``panda_hand`` weld silently never forms.

Both edits are idempotent: re-running won't duplicate the plugin and re-bumping
friction is a no-op.
"""

import argparse
import json
import re
from pathlib import Path


# Higher mu pads the friction budget so a partially-misaligned grasp
# still benefits from contact even if the DetachableJoint hasn't fired.
_FRICTION_MU = 30.0

# gz-sim DetachableJoint plugin block.  ``{name}`` is substituted with
# the object's SDF model name; the link inside is ``<name>_link`` by
# convention used by ``export.py``.  The parent_link is the OBJECT's
# link (so the joint creation does not need to reference panda's link
# tree from the object's side), and child_model / child_link point at
# panda_hand.  The driver toggles via the per-object attach/detach
# topics.  ``suppress_child_warning`` silences the plugin's warning
# when panda is not yet in the world at SDF load time.
_DETACHABLE_PLUGIN_TPL = """  <plugin filename="gz-sim-detachable-joint-system"
          name="gz::sim::systems::DetachableJoint">
    <parent_link>{name}_link</parent_link>
    <child_model>panda</child_model>
    <!-- panda_hand / panda_link8 carry no <inertial> in the MoveIt panda
         description, so urdf2sdf DROPS them and they do not exist in the gz
         model. Attach to panda_leftfinger, which survives and moves with the
         gripper. (Was panda_hand -> the joint silently never formed.) -->
    <child_link>panda_leftfinger</child_link>
    <detach_topic>/{name}/detach</detach_topic>
    <attach_topic>/{name}/attach</attach_topic>
    <output_topic>/{name}/state</output_topic>
    <suppress_child_warning>true</suppress_child_warning>
  </plugin>
"""


def patch_sdf(sdf_path: Path, visual_mesh: Path = None) -> bool:
    sdf_text = sdf_path.read_text()
    orig = sdf_text

    # Keep the mesh collision; just firm up the contact friction.
    sdf_text = re.sub(r"<mu>[^<]*</mu>", f"<mu>{_FRICTION_MU}</mu>", sdf_text)
    sdf_text = re.sub(r"<mu2>[^<]*</mu2>", f"<mu2>{_FRICTION_MU}</mu2>", sdf_text)

    # Inject the DetachableJoint plugin block before </model>, unless
    # this SDF already carries it (idempotent re-run).
    name_match = re.search(r'<model\s+name="([^"]+)"', sdf_text)
    if name_match and "gz-sim-detachable-joint-system" not in sdf_text:
        plugin_xml = _DETACHABLE_PLUGIN_TPL.format(name=name_match.group(1))
        sdf_text = sdf_text.replace("</model>", plugin_xml + "</model>", 1)

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
