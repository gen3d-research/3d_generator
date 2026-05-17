"""Publishes a fixed Panda home joint configuration on /joint_states.

The standard joint_state_publisher's publish_default_positions feature yields
all-zeros, which puts the Panda in self-collision (panda_hand against
panda_link5 / panda_link7) and causes MoveIt's CheckStartStateCollision
adapter to abort every planning request.  We publish the canonical home
configuration instead so the start state is always valid.
"""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

PANDA_JOINT_NAMES = [
    "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
    "panda_joint5", "panda_joint6", "panda_joint7",
    "panda_finger_joint1", "panda_finger_joint2",
]
PANDA_HOME = [
    0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785,
    0.035, 0.035,
]


class HomeJointStatePublisher(Node):
    def __init__(self):
        super().__init__("home_joint_state_publisher")
        self.pub = self.create_publisher(JointState, "/joint_states", 10)
        self.create_timer(1.0 / 30.0, self._publish)

    def _publish(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = PANDA_JOINT_NAMES
        msg.position = PANDA_HOME
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = HomeJointStatePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
