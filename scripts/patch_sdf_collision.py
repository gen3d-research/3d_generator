#!/usr/bin/env python3
"""
Rewrite every SDF in the manifest so its <collision> uses a <box> primitive
matched to the visual mesh's AABB, and inject a gz-sim DetachableJoint
plugin so the visual_demo driver can weld the object to panda_hand on
gripper-close and release it on gripper-open.

Two reasons for the two transformations to live in the same script:

1. Gazebo / DART chokes on the trimesh-exported collision OBJs because
   they lack vertex normals (see gz_v4.log: "submesh ... does not have
   a normal count that matches its vertex count").  Replacing the
   collision geometry with a box of the right extents is the surgical
   fix used by every static manipulation benchmark we have looked at.
   The friction is set to mu=30 (matching the Robotiq cube in the
   ROS 2 Manipulation Masterclass course, 13/.../ros2_online_workshop.world)
   so the new firm contact backs up the rigid attach.

2. gz-sim Harmonic does not ship a libgazebo_grasp_fix.so analogue (the
   course's Gazebo Classic plugin).  The closest substitute is
   libgz-sim8-detachable-joint-system.so, which creates a fixed joint
   between two named links and exposes attach/detach topics.  Baking
   the plugin into each spawned object's SDF lets the driver call
   ``gz topic -t /<obj>/attach`` when the gripper closes and have the
   object follow panda_hand rigidly through lift/transport/place.

Both edits are idempotent: re-running on an already-patched SDF will
not duplicate the plugin and will keep the existing box collision.
"""

import argparse
import json
import re
from pathlib import Path

import trimesh


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
    <child_link>panda_hand</child_link>
    <detach_topic>/{name}/detach</detach_topic>
    <attach_topic>/{name}/attach</attach_topic>
    <output_topic>/{name}/state</output_topic>
    <suppress_child_warning>true</suppress_child_warning>
  </plugin>
"""


def aabb_of(mesh_path: Path) -> tuple[float, float, float]:
    m = trimesh.load(str(mesh_path), force="mesh")
    e = m.bounding_box.extents
    return float(e[0]), float(e[1]), float(e[2])


def patch_sdf(sdf_path: Path, visual_mesh: Path) -> bool:
    sdf_text = sdf_path.read_text()
    orig = sdf_text
    ex, ey, ez = aabb_of(visual_mesh)
    new_collision = (
        f"<collision name=\"collision\">"
        f"<geometry><box><size>{ex:.6f} {ey:.6f} {ez:.6f}</size></box></geometry>"
        f"<surface><friction><ode>"
        f"<mu>{_FRICTION_MU}</mu><mu2>{_FRICTION_MU}</mu2>"
        f"</ode></friction></surface>"
        f"</collision>"
    )
    sdf_text = re.sub(
        r"<collision\b[^>]*>.*?</collision>",
        new_collision,
        sdf_text,
        flags=re.DOTALL,
    )

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
