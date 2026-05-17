"""Assemble a Panda URDF augmented with the gz_ros2_control plugin.

Reads ``urdf/panda_base.urdf`` and ``urdf/panda_ros2_control.xml`` from the
installed package share and produces a single URDF string suitable for
``ros_gz_sim create -topic /robot_description``.

Four transformations on top of the base URDF:

1. Splice the ``ros2_control`` + ``<gazebo>`` plugin block in front of the
   closing ``</robot>`` tag and expand ``$(GENERATED_OBJECTS_EVAL_CONFIG)``
   to the package's installed ``config/`` directory.
2. Anchor the floating base by prepending a ``<link name="world"/>`` and a
   fixed joint ``world -> panda_link0``.  Without this the Panda topples
   under gravity the instant gz_sim's physics steps.
3. Strip ``<mimic joint="panda_finger_joint1"/>`` from
   ``panda_finger_joint2``.  DART (gz_sim's physics) does not support mimic
   constraints; leaving the tag in produces a warning and an unconstrained
   second finger that falls under gravity.
4. Inject a stub ``<inertial>`` block into every link that lacks one.
   ``moveit_resources_panda_description`` ships an inertialess URDF; gz's
   urdf2sdf converter silently strips inertialess links, leaving the model
   with zero links and refusing to spawn (``A model must have at least one
   link``).  Since the base is welded to the world frame the exact values
   do not matter for the kinematic demo --- a unit-mass placeholder is
   enough to keep the link present in the SDF.
"""

from __future__ import annotations

import re
from pathlib import Path

from ament_index_python.packages import get_package_share_directory

_WORLD_ANCHOR = """  <link name="world"/>
  <joint name="virtual_joint" type="fixed">
    <parent link="world"/>
    <child link="panda_link0"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
  </joint>
"""

_INERTIAL_STUB = """    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
"""

# High-friction surface for the Panda finger pads.  Without this DART
# uses its default ``mu`` (~1.0) for the finger-object contact and the
# generated objects slip out the moment the arm starts to lift.
# mu=30 matches the Robotiq 85 finger-tip values used in the ROS 2
# Manipulation Masterclass course
# (moveit_course_humble/13/.../robotiq_85_gripper_macro_gz.urdf.xacro
# lines 56-57, 76-77).  Combined with the DetachableJoint plugin baked
# into each object's SDF, this gives a firm contact backing the rigid
# attach the driver triggers when the gripper closes.
_FINGER_FRICTION = """
<gazebo reference="panda_leftfinger">
  <mu1>30.0</mu1>
  <mu2>30.0</mu2>
  <kp>1000000.0</kp>
  <kd>100.0</kd>
  <minDepth>0.001</minDepth>
  <maxVel>0.1</maxVel>
</gazebo>
<gazebo reference="panda_rightfinger">
  <mu1>30.0</mu1>
  <mu2>30.0</mu2>
  <kp>1000000.0</kp>
  <kd>100.0</kd>
  <minDepth>0.001</minDepth>
  <maxVel>0.1</maxVel>
</gazebo>
"""


def _strip_finger_mimic(urdf: str) -> str:
    return re.sub(
        r'<mimic\s+joint="panda_finger_joint1"\s*/>\s*',
        "",
        urdf,
    )


def _inject_world_anchor(urdf: str) -> str:
    return re.sub(
        r'(<link\s+name="panda_link0">)',
        _WORLD_ANCHOR + r"\1",
        urdf,
        count=1,
    )


def _inject_inertials(urdf: str) -> str:
    """Add a stub <inertial> to every <link> that doesn't already have one.

    Handles two URDF forms:
      <link name="X" />           (self-closing, frame-only)
      <link name="X"> ... </link> (with visual/collision children)

    The ``world`` link is left alone --- gz_sim treats it as a static frame.
    """
    # 1. Expand self-closing <link name="X" /> to <link name="X"><inertial>...
    def expand_self_closing(m: re.Match) -> str:
        name = m.group(1)
        if name == "world":
            return m.group(0)
        return f'<link name="{name}">\n{_INERTIAL_STUB}  </link>'

    urdf = re.sub(
        r'<link\s+name="([^"]+)"\s*/>',
        expand_self_closing,
        urdf,
    )

    # 2. Inject an inertial right after the opening tag for any link that
    #    doesn't already have one in its body.
    def inject(m: re.Match) -> str:
        name = m.group(1)
        body = m.group(2)
        if name == "world" or "<inertial>" in body:
            return m.group(0)
        return f'<link name="{name}">\n{_INERTIAL_STUB}{body}</link>'

    return re.sub(
        r'<link\s+name="([^"]+)"\s*>(.*?)</link>',
        inject,
        urdf,
        flags=re.DOTALL,
    )


def build_panda_urdf(extra_xml: str | None = None) -> str:
    share = Path(get_package_share_directory("generated_objects_eval"))
    base = (share / "urdf" / "panda_base.urdf").read_text()
    ctrl = (share / "urdf" / "panda_ros2_control.xml").read_text()
    cfg_dir = str(share / "config")
    ctrl = ctrl.replace("$(GENERATED_OBJECTS_EVAL_CONFIG)", cfg_dir)

    base = _inject_world_anchor(base)
    base = _strip_finger_mimic(base)
    base = _inject_inertials(base)

    if "</robot>" not in base:
        raise RuntimeError("panda_base.urdf has no </robot> closing tag")
    return base.replace(
        "</robot>",
        ctrl + _FINGER_FRICTION +
        ("" if extra_xml is None else extra_xml) + "\n</robot>",
    )


if __name__ == "__main__":
    print(build_panda_urdf())
