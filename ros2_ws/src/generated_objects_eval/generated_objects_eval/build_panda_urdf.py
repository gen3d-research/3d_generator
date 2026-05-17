"""Assemble a Panda URDF augmented with the gz_ros2_control plugin.

Reads ``urdf/panda_base.urdf`` and ``urdf/panda_ros2_control.xml`` from the
installed package share, splices the latter in front of the closing
``</robot>`` tag, expands the ``$(GENERATED_OBJECTS_EVAL_CONFIG)`` placeholder
to the package's installed ``config/`` directory, and returns the resulting
URDF string.  Used by ``arm_gz_sim.launch.py``.
"""

from __future__ import annotations

from pathlib import Path

from ament_index_python.packages import get_package_share_directory


def build_panda_urdf(extra_xml: str | None = None) -> str:
    share = Path(get_package_share_directory("generated_objects_eval"))
    base = (share / "urdf" / "panda_base.urdf").read_text()
    ctrl = (share / "urdf" / "panda_ros2_control.xml").read_text()
    cfg_dir = str(share / "config")
    ctrl = ctrl.replace("$(GENERATED_OBJECTS_EVAL_CONFIG)", cfg_dir)

    if "</robot>" not in base:
        raise RuntimeError("panda_base.urdf has no </robot> closing tag")
    full = base.replace("</robot>", ctrl + ("" if extra_xml is None else extra_xml) + "\n</robot>")
    return full


if __name__ == "__main__":
    print(build_panda_urdf())
