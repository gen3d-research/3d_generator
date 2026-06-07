"""End-to-end pick-and-place demonstration for the project-page videos.

For one generated object the driver:

    1. Spawns the SDF into the running gz_sim world (so the camera sees it).
    2. Renders the table / spawn marker / object marker in RViz.
    3. Plans and *animates* a multi-stage sequence on the Panda:

           ready -> pre-grasp -> grasp -> lift -> transport -> place -> retract -> ready

       Each plan's joint trajectory is replayed on ``/joint_states`` so
       RViz's RobotModel display shows the arm moving end-to-end.
    4. While the object is "grasped" (between grasp and place) a marker is
       attached to the end-effector so the viewer can see it being carried.

Loops indefinitely (``--loop``) so screen recording is straightforward.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import rclpy
from control_msgs.action import ParallelGripperCommand
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion, Vector3
from moveit_msgs.msg import CollisionObject, PlanningScene, PlanningSceneWorld
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import ColorRGBA, Header, Float64MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from visualization_msgs.msg import Marker, MarkerArray

WORLD_NAME = "panda_eval_world"
PANDA_JOINT_NAMES = [
    "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
    "panda_joint5", "panda_joint6", "panda_joint7",
    "panda_finger_joint1", "panda_finger_joint2",
]
PANDA_READY = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]


# ---------------------------------------------------------------------------

def look_at_quaternion(approach: np.ndarray) -> Quaternion:
    z_des = -approach / (np.linalg.norm(approach) + 1e-12)
    ref = np.array([0.0, 0.0, 1.0]) if abs(z_des[2]) < 0.95 else np.array([1.0, 0.0, 0.0])
    x_des = np.cross(ref, z_des)
    x_des /= np.linalg.norm(x_des) + 1e-12
    y_des = np.cross(z_des, x_des)
    R = np.stack([x_des, y_des, z_des], axis=1)
    qw = 0.5 * math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2]))
    qx = (R[2, 1] - R[1, 2]) / (4 * qw + 1e-12)
    qy = (R[0, 2] - R[2, 0]) / (4 * qw + 1e-12)
    qz = (R[1, 0] - R[0, 1]) / (4 * qw + 1e-12)
    return Quaternion(x=float(qx), y=float(qy), z=float(qz), w=float(qw))


def _rotate(quat, v):
    """Rotate vector v by quaternion quat=(x,y,z,w)."""
    q = np.asarray(quat, float)
    v = np.asarray(v, float)
    qv = q[:3]
    t = 2.0 * np.cross(qv, v)
    return v + q[3] * t + np.cross(qv, t)


def _mat_to_quat(R) -> Quaternion:
    qw = 0.5 * math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2]))
    qx = (R[2, 1] - R[1, 2]) / (4 * qw + 1e-12)
    qy = (R[0, 2] - R[2, 0]) / (4 * qw + 1e-12)
    qz = (R[1, 0] - R[0, 1]) / (4 * qw + 1e-12)
    return Quaternion(x=float(qx), y=float(qy), z=float(qz), w=float(qw))


def grasp_quaternion(approach: np.ndarray, axis: np.ndarray) -> Quaternion:
    """Full grasp orientation for panda_link8.

    panda_link8 +z points toward the object, so it aligns with the (downward)
    approach. The gripper's fingers open along +y, so y aligns with the
    antipodal grasp line (contact1->contact2). Without this the fingers close
    along an arbitrary axis and miss the object entirely.
    """
    z = np.asarray(approach, float)
    z = z / (np.linalg.norm(z) + 1e-12)
    a = np.asarray(axis, float)
    y = a - (a @ z) * z                      # project the grasp line off z
    if np.linalg.norm(y) < 1e-6:             # axis ~parallel to approach: pick any perp
        ref = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        y = ref - (ref @ z) * z
    y /= np.linalg.norm(y) + 1e-12
    x = np.cross(y, z)
    R = np.stack([x, y, z], axis=1)
    return _mat_to_quat(R)


def spawn_in_gazebo(sdf: Path, name: str, x: float, y: float, z: float) -> bool:
    req = (
        f'sdf_filename: "{sdf}", name: "{name}", '
        f'pose: {{position: {{x: {x}, y: {y}, z: {z}}}, '
        f'orientation: {{w: 1.0}}}}'
    )
    proc = subprocess.run(
        ["gz", "service", "-s", f"/world/{WORLD_NAME}/create",
         "--reqtype", "gz.msgs.EntityFactory",
         "--reptype", "gz.msgs.Boolean",
         "--timeout", "3000", "--req", req],
        capture_output=True, timeout=8.0,
    )
    return proc.returncode == 0 and b"data: true" in proc.stdout


def despawn_in_gazebo(name: str) -> None:
    """Remove a previously spawned model from the running gz_sim world.

    Failures are non-fatal: if the entity is already gone the next spawn
    call will still succeed under a new (or same) name.
    """
    subprocess.run(
        ["gz", "service", "-s", f"/world/{WORLD_NAME}/remove",
         "--reqtype", "gz.msgs.Entity", "--reptype", "gz.msgs.Boolean",
         "--timeout", "3000", "--req", f'name: "{name}", type: MODEL'],
        capture_output=True, timeout=6.0,
    )


def gz_topic_publish(topic: str,
                     msg_type: str = "gz.msgs.Empty",
                     payload: str = "unused: true") -> bool:
    """Fire one ``gz topic`` publish.

    Used to toggle each object's DetachableJoint plugin on
    ``/<name>/attach`` and ``/<name>/detach``.  The plugin's input is
    ``gz.msgs.Empty``; the payload string is irrelevant but a non-empty
    body is required for ``gz topic -p`` to actually publish.
    """
    proc = subprocess.run(
        ["gz", "topic", "-t", topic, "-m", msg_type, "-p", payload],
        capture_output=True, timeout=4.0,
    )
    return proc.returncode == 0


# ---------------------------------------------------------------------------

class DemoNode(Node):
    """Holds the publishers for joint_states + scene markers + grasped-object
    marker, and exposes helpers to animate joint trajectories."""

    def __init__(self, hand_open: float = 0.04, hand_closed: float = 0.0,
                 publish_joints: bool = True):
        super().__init__("demo_plan_driver_pub")
        # In gz_ros2_control (execute) mode the joint_state_broadcaster is the
        # sole /joint_states source; if this node ALSO publishes, the two
        # streams race and MoveIt's current-state monitor sees a state that
        # doesn't match the real robot -> "start point deviates" rejects every
        # trajectory. So suppress our publisher in execute mode and keep it only
        # for RViz-only animation.
        self._publish_joints = publish_joints
        # Force wall-clock — keep our stamps aligned with
        # static_transform_publisher's wall-clock-stamped /tf_static.
        try:
            from rclpy.parameter import Parameter
            self.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, False)])
        except Exception:
            pass
        latched = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        # /joint_states: best-effort + KEEP_LAST(1) so RViz always renders the
        # newest joint vector (rival 30 Hz publishers cannot stall us).
        js_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self.joint_pub = self.create_publisher(JointState, "/joint_states", js_qos)
        self.scene_pub = self.create_publisher(Marker, "/demo_scene_markers", latched)
        self.grasped_pub = self.create_publisher(
            MarkerArray, "/demo_grasped_object", latched)
        # /planning_scene with TRANSIENT_LOCAL so late-joining planning_scene_monitors
        # (e.g. MoveItPy starting after this publisher) still see the table.
        # This bypasses moveit_py.apply_collision_object, whose 2.12.x binding
        # segfaults; the equivalent topic-based path is stable.
        self.planning_scene_pub = self.create_publisher(
            PlanningScene, "/planning_scene", latched)
        self.hand_open = hand_open
        self.hand_closed = hand_closed
        self._current_q = list(PANDA_READY)
        self._current_grip = hand_open
        self._lock = threading.Lock()
        self._stop = False
        # 60 Hz — high enough to dominate any 30 Hz rival publisher in RViz.
        self.create_timer(1.0 / 60.0, self._publish_current)
        # Action client for the parallel_gripper_action_controller spawned by
        # gz_ros2_control.  Only used when --execute is set; the
        # send_gripper_goal() helper short-circuits when the action
        # server isn't available (RViz-only mode).
        self._gripper_client = ActionClient(
            self, ParallelGripperCommand,
            "/panda_hand_controller/gripper_cmd")

        # Object tracking: a background thread reads the object's ACTUAL pose
        # from gz and republishes its RViz mesh marker there, so RViz reflects
        # the real (physics-driven) object position/movement — not a fake pose.
        self._track = None  # (name, mesh_url, extents) or None
        threading.Thread(target=self._track_loop, daemon=True).start()

        # Track the live finger opening (from the broadcaster) so we can tell
        # whether the fingers stalled against the object (contact) or closed all
        # the way to 0 (missed it).
        self._finger_pos = None
        self.create_subscription(JointState, "/joint_states", self._js_cb, 10)

        # Finger EFFORT command (forward_command_controller): [f1, f2].
        # Negative = close/squeeze, positive = open. The ForwardCommandController
        # holds the last command, so a single close publish keeps squeezing
        # through lift/transport -> genuine friction hold.
        self._finger_cmd = self.create_publisher(
            Float64MultiArray, "/panda_hand_controller/commands", 10)
        # Direct arm-trajectory command (bypasses MoveIt) — recovers the arm to
        # home even when a joint drifts out of bounds and MoveIt's
        # CheckStartStateBounds would otherwise abort every plan.
        self._arm_traj = self.create_publisher(
            JointTrajectory, "/panda_arm_controller/joint_trajectory", 10)
        # Closing squeeze force (N); friction holds the object. ~-50 is the
        # sweet spot: enough to hold without ejecting smaller objects (higher,
        # e.g. -75, slams light objects out of the gripper). The ideal force is
        # object-dependent — the inherent reality of friction grasping.
        self.close_eff = -50.0
        self.open_eff = 20.0

    def set_finger_effort(self, eff: float):
        self._finger_cmd.publish(Float64MultiArray(data=[float(eff), float(eff)]))

    def command_arm(self, positions, duration: float = 3.0):
        """Send the arm straight to a joint configuration via the controller
        (no MoveIt) — works even from an out-of-bounds state."""
        msg = JointTrajectory()
        msg.joint_names = PANDA_JOINT_NAMES[:7]
        pt = JointTrajectoryPoint()
        pt.positions = [float(p) for p in positions]
        pt.time_from_start.sec = int(duration)
        pt.time_from_start.nanosec = int((duration - int(duration)) * 1e9)
        msg.points = [pt]
        self._arm_traj.publish(msg)

    def _js_cb(self, msg):
        if "panda_finger_joint1" in msg.name:
            self._finger_pos = float(msg.position[msg.name.index("panda_finger_joint1")])

    def track_object(self, name, mesh_url, extents):
        self._track = (name, mesh_url, np.asarray(extents, dtype=float))

    def _track_loop(self):
        while not self._stop:
            tr = self._track
            if tr is not None:
                name, mesh_url, extents = tr
                p = query_object_pose(name)
                if p is not None:
                    pos, quat = p
                    pose = Pose(
                        position=Point(x=pos[0], y=pos[1], z=pos[2]),
                        orientation=Quaternion(x=quat[0], y=quat[1],
                                               z=quat[2], w=quat[3]))
                    self.publish_object(pose, extents,
                                        attached_to_link="panda_link0",
                                        mesh_resource=mesh_url)
            time.sleep(0.25)

    def send_gripper_goal(self, position: float, max_effort: float = 30.0,
                          timeout_s: float = 4.0) -> bool:
        """Drive the parallel gripper to ``position`` (joint opening, m).

        ``position`` is the per-finger opening (0 = closed, 0.04 = fully open
        for the Panda).  Returns True on success / acceptance, False if the
        action server is not reachable within ``timeout_s`` (which happens
        silently in RViz-only mode).
        """
        if not self._gripper_client.wait_for_server(timeout_sec=2.0):
            return False
        goal = ParallelGripperCommand.Goal()
        # ParallelGripperCommand uses sensor_msgs/JointState.  The controller
        # config drives joint1 as the primary; joint2 is mirrored to match.
        goal.command.name = ["panda_finger_joint1"]
        goal.command.position = [float(position)]
        goal.command.effort = [float(max_effort)]
        future = self._gripper_client.send_goal_async(goal)
        deadline = time.time() + timeout_s
        while not future.done() and time.time() < deadline:
            time.sleep(0.05)
        if not future.done():
            return False
        handle = future.result()
        if not handle or not handle.accepted:
            return False
        result_future = handle.get_result_async()
        deadline = time.time() + timeout_s
        while not result_future.done() and time.time() < deadline:
            time.sleep(0.05)
        return result_future.done()

    def _publish_current(self):
        if not self._publish_joints:
            return
        with self._lock:
            q = list(self._current_q)[:7]  # 7 arm joints only
            g = self._current_grip
        # Pad with zeros if for any reason fewer than 7 joints are stored.
        while len(q) < 7:
            q.append(0.0)
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = PANDA_JOINT_NAMES
        msg.position = q + [g, g]
        self.joint_pub.publish(msg)

    def set_state(self, q, grip=None):
        # Keep only the 7 arm joints in _current_q; the two finger joints
        # are driven separately via _current_grip.  Without this, the
        # current_state_monitor logs "number of joint names does not match
        # number of positions" when animate() writes a 9-element vector.
        with self._lock:
            self._current_q = list(q)[:7]
            if grip is not None:
                self._current_grip = grip

    def _extract_waypoints(self, trajectory):
        """Return (points, joint_names) from a moveit_py RobotTrajectory.
        The wrapper's surface area changed across 2.12.x releases, so try
        every documented path."""
        accessors = [
            lambda t: (t.joint_trajectory.points,
                       t.joint_trajectory.joint_names),
            lambda t: (t.get_robot_trajectory_msg().joint_trajectory.points,
                       t.get_robot_trajectory_msg().joint_trajectory.joint_names),
            lambda t: (t.robot_trajectory.joint_trajectory.points,
                       t.robot_trajectory.joint_trajectory.joint_names),
        ]
        for acc in accessors:
            try:
                p, n = acc(trajectory)
                if p:
                    return list(p), list(n)
            except Exception:
                continue
        return None, None

    def animate(self, trajectory, goal_q=None, time_scale: float = 1.0,
                min_stage_seconds: float = 2.5):
        """Replay a moveit_py trajectory on /joint_states.  If the waypoints
        can't be extracted from the wrapper (API drift), fall back to
        interpolating linearly from the current pose to ``goal_q`` over
        ``min_stage_seconds``.  Always blocks at least ``min_stage_seconds``
        so each stage is visible on camera."""
        t_stage_start = time.time()
        pts, names = self._extract_waypoints(trajectory) if trajectory is not None else (None, None)

        if pts:
            self.get_logger().info(f"    animate: {len(pts)} waypoints")
            if len(pts) < 30:
                pts = self._densify(pts, target=40)
            idx = [names.index(n) if n in names else None for n in PANDA_JOINT_NAMES]
            t0 = time.time()
            for p in pts:
                t_target = (p.time_from_start.sec
                            + p.time_from_start.nanosec * 1e-9) * time_scale
                dt = t_target - (time.time() - t0)
                if dt > 0:
                    time.sleep(dt)
                q = [(float(p.positions[j]) if j is not None else 0.0) for j in idx]
                self.set_state(q)
        elif goal_q is not None:
            # Linear interpolation fallback.
            self.get_logger().warn(
                "    animate: no waypoints from wrapper, interpolating to goal"
            )
            start = list(self._current_q)
            n_steps = 40
            for k in range(1, n_steps + 1):
                a = k / n_steps
                q = [(1 - a) * s + a * g for s, g in zip(start, goal_q)]
                self.set_state(q)
                time.sleep(min_stage_seconds / n_steps)

        elapsed = time.time() - t_stage_start
        if elapsed < min_stage_seconds:
            time.sleep(min_stage_seconds - elapsed)

    def _densify(self, points, target=40):
        """Linearly interpolate the existing waypoints so the motion is
        visually smooth in RViz."""
        if len(points) >= target or len(points) < 2:
            return points
        ts = [p.time_from_start.sec + p.time_from_start.nanosec * 1e-9 for p in points]
        total = ts[-1] if ts[-1] > 0 else max(1.0, len(points) * 0.05)
        out = []
        for k in range(target):
            t = total * k / (target - 1)
            i = 0
            while i + 1 < len(ts) and ts[i + 1] < t:
                i += 1
            t0, t1 = ts[i], ts[min(i + 1, len(ts) - 1)]
            alpha = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
            p0 = points[i].positions
            p1 = points[min(i + 1, len(points) - 1)].positions
            pos = [(1 - alpha) * a + alpha * b for a, b in zip(p0, p1)]

            class _Stamp:
                pass
            stamp = _Stamp()
            stamp.sec = int(t)
            stamp.nanosec = int((t - int(t)) * 1e9)

            class _Wp:
                pass
            w = _Wp()
            w.positions = pos
            w.time_from_start = stamp
            out.append(w)
        return out

    def publish_scene(self, table_pose=(0.5, 0.0, 0.2), table_size=(0.8, 0.8, 0.4),
                      collision_top_clearance: float = 0.04):
        """Render the table marker AND publish its MoveIt CollisionObject.

        ``collision_top_clearance`` shrinks the CollisionObject from the
        top by that many metres so the planner has room to bring the
        fingertip down to the object on the table without considering the
        finger-tip-vs-table-top contact a collision.  The Gazebo table is
        unaffected, so the physical gripper still rests on the real top.
        Set to 0 to publish the exact-size collision box.
        """
        # Visual marker for the table in RViz's Marker display (full size).
        m = Marker()
        m.header = Header(frame_id="panda_link0",
                          stamp=self.get_clock().now().to_msg())
        m.ns = "scene"
        m.id = 0
        m.type = Marker.CUBE
        m.action = Marker.ADD
        m.pose.position.x = table_pose[0]
        m.pose.position.y = table_pose[1]
        m.pose.position.z = table_pose[2]
        m.pose.orientation.w = 1.0
        m.scale.x = table_size[0]
        m.scale.y = table_size[1]
        m.scale.z = table_size[2]
        m.color = ColorRGBA(r=0.55, g=0.40, b=0.25, a=1.0)
        self.scene_pub.publish(m)
        # Collision twin (top shrunken by `collision_top_clearance`) for
        # MoveIt 2.  Published as a /planning_scene diff so MoveItPy's
        # planning_scene_monitor picks it up; the frame is panda_link0 to
        # match the rest of the demo's TF chain.
        c = float(collision_top_clearance)
        co_height = float(table_size[2]) - c
        co_center_z = float(table_pose[2]) - c / 2.0
        co = CollisionObject()
        co.header.frame_id = "panda_link0"
        co.header.stamp = self.get_clock().now().to_msg()
        co.id = "demo_table"
        co.operation = CollisionObject.ADD
        prim = SolidPrimitive()
        prim.type = SolidPrimitive.BOX
        prim.dimensions = [float(table_size[0]), float(table_size[1]),
                           co_height]
        co.primitives = [prim]
        pose = Pose()
        pose.position.x = float(table_pose[0])
        pose.position.y = float(table_pose[1])
        pose.position.z = co_center_z
        pose.orientation.w = 1.0
        co.primitive_poses = [pose]
        scene = PlanningScene()
        scene.is_diff = True
        scene.world = PlanningSceneWorld()
        scene.world.collision_objects = [co]
        self.planning_scene_pub.publish(scene)

    def set_object_collision(self, present: bool, base=None, quat=None,
                             extents=None, aabb=None):
        """Add/remove the spawned object as a MoveIt collision obstacle (id
        'demo_object') so the arm plans AROUND it during approach instead of
        knocking it. A box from the object's AABB at its settled pose; shrunk
        slightly so the local pre-grasp near the surface isn't over-blocked."""
        co = CollisionObject()
        co.header.frame_id = "panda_link0"
        co.header.stamp = self.get_clock().now().to_msg()
        co.id = "demo_object"
        if not present:
            co.operation = CollisionObject.REMOVE
        else:
            co.operation = CollisionObject.ADD
            if aabb is not None:
                amin = np.asarray(aabb[0], float)
                amax = np.asarray(aabb[1], float)
                size = amax - amin
                ctr_local = 0.5 * (amin + amax)
            else:
                size = np.asarray(extents if extents is not None else [0.05] * 3, float)
                ctr_local = np.zeros(3)
            size = np.maximum(size * 0.9, 0.005)
            base = np.zeros(3) if base is None else np.asarray(base, float)
            quat = np.array([0.0, 0.0, 0.0, 1.0]) if quat is None else np.asarray(quat, float)
            ctr_world = base + _rotate(quat, ctr_local)
            prim = SolidPrimitive()
            prim.type = SolidPrimitive.BOX
            prim.dimensions = [float(size[0]), float(size[1]), float(size[2])]
            co.primitives = [prim]
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = map(float, ctr_world)
            pose.orientation = Quaternion(x=float(quat[0]), y=float(quat[1]),
                                          z=float(quat[2]), w=float(quat[3]))
            co.primitive_poses = [pose]
        scene = PlanningScene()
        scene.is_diff = True
        scene.world = PlanningSceneWorld()
        scene.world.collision_objects = [co]
        self.planning_scene_pub.publish(scene)
        time.sleep(0.15)   # let the planning-scene monitor apply the diff before planning

    def publish_object(self, pose: Pose, extents: np.ndarray,
                       attached_to_link: Optional[str] = None,
                       color=(0.10, 0.45, 0.85, 1.0),
                       mesh_resource: Optional[str] = None):
        """Render the object marker in RViz.

        When ``mesh_resource`` is given (a ``file://...obj`` URL) we emit a
        ``Marker.MESH_RESOURCE`` so the RViz silhouette matches the model
        gz_sim spawned.  Otherwise we fall back to the old box marker,
        which is what the gripper-attached and place-position markers use
        (they are not real objects in the world)."""
        ma = MarkerArray()
        m = Marker()
        m.header = Header(stamp=self.get_clock().now().to_msg())
        m.header.frame_id = attached_to_link or "panda_link0"
        m.ns = "object"
        m.id = 1
        if mesh_resource:
            m.type = Marker.MESH_RESOURCE
            m.mesh_resource = mesh_resource
            m.mesh_use_embedded_materials = False
            # The OBJ exporter writes meshes at object-local scale, so a
            # unit scale matches what gz_sim renders.
            m.scale = Vector3(x=1.0, y=1.0, z=1.0)
        else:
            m.type = Marker.CUBE
            m.scale = Vector3(x=float(extents[0]), y=float(extents[1]),
                              z=float(extents[2]))
        m.action = Marker.ADD
        m.pose = pose
        m.color = ColorRGBA(r=color[0], g=color[1], b=color[2], a=color[3])
        ma.markers.append(m)
        self.grasped_pub.publish(ma)


# ---------------------------------------------------------------------------

def plan_to_pose(arm, target: PoseStamped, from_current: bool = True,
                 attempts: int = 2) -> Optional[object]:
    # Plan from the robot's ACTUAL current state (not a fixed "ready" config),
    # otherwise moveit_py.execute() rejects every trajectory with "start point
    # deviates from current robot state" — the planned start (ready) never
    # matches where gz_ros2_control actually has the arm. In RViz-only mode the
    # current state comes from the driver's own /joint_states publisher; in
    # gz_ros2_control mode it comes from joint_state_broadcaster.
    #
    # RRTConnect is randomized and the near-table grasp pose is tight, so retry
    # a few times before declaring the pose unreachable.
    for _ in range(max(1, attempts)):
        if from_current:
            arm.set_start_state_to_current_state()
        else:
            arm.set_start_state(configuration_name="ready")
        arm.set_goal_state(pose_stamped_msg=target, pose_link="panda_link8")
        result = arm.plan()
        if result and result.trajectory is not None:
            return result.trajectory
    return None


def pose_stamped(xyz: np.ndarray, approach: np.ndarray, axis=None) -> PoseStamped:
    ps = PoseStamped()
    ps.header.frame_id = "panda_link0"
    ps.pose.position.x = float(xyz[0])
    ps.pose.position.y = float(xyz[1])
    ps.pose.position.z = float(xyz[2])
    # With the antipodal grasp axis we can orient the fingers to straddle the
    # object; otherwise fall back to approach-only (arbitrary finger axis).
    ps.pose.orientation = (grasp_quaternion(approach, axis) if axis is not None
                           else look_at_quaternion(approach))
    return ps


def pose_xyz(xyz, ori=(0.0, 0.0, 0.0, 1.0)) -> Pose:
    return Pose(position=Point(x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2])),
                orientation=Quaternion(x=ori[0], y=ori[1], z=ori[2], w=ori[3]))


def rank_grasps(grasps, com=None):
    """Order grasp candidates best-first. The grasp_planner now SCORES grasps
    (waist/handle near the CoM, clear approach, deep friction margin) and the
    manifest is written sorted, so trust that order: rank by the planner's
    `score` descending, falling back to CoM proximity if no score is present."""
    com = np.asarray(com, float) if com is not None else None

    def key(g):
        if "score" in g:
            return -float(g["score"])
        d_com = (float(np.linalg.norm(np.asarray(g["center"], float) - com))
                 if com is not None else 0.0)
        return d_com
    return sorted(grasps, key=key)


def go_home(demo, arm, execute, moveit_py):
    """Return the arm to the 'ready' home config. Done between cycles so the arm
    is parked clear of the spawn zone when the next object drops, and so it
    approaches the new object from above instead of sweeping through it.

    Commands the controller DIRECTLY (not MoveIt): this also recovers the arm
    when a grasp left a joint at/just past its limit — MoveIt would refuse to
    plan from there ('Start state out of bounds') and the demo would wedge."""
    demo.get_logger().info("  -> home")
    if execute:
        demo.command_arm(PANDA_READY, duration=3.0)
        time.sleep(3.3)
        return
    # RViz-only: animate via MoveIt.
    arm.set_start_state_to_current_state()
    arm.set_goal_state(configuration_name="ready")
    result = arm.plan()
    if result and result.trajectory is not None:
        demo.animate(result.trajectory, time_scale=1.5, min_stage_seconds=2.0)


def query_object_z(name: str, timeout: float = 4.0):
    """World-frame z of a spawned gz model from /world/<world>/pose/info, or
    None if it can't be read. Used to confirm the object actually rose after
    the lift (closed-loop grasp-success check)."""
    try:
        out = subprocess.run(
            ["gz", "topic", "-e", "-t", f"/world/{WORLD_NAME}/pose/info", "-n", "8"],
            capture_output=True, timeout=timeout).stdout.decode(errors="ignore")
    except Exception:
        return None
    import re
    best = None
    for m in re.finditer(
            r'pose\s*\{[^{}]*?name:\s*"' + re.escape(name)
            + r'"[^{}]*?position\s*\{([^{}]*?)\}', out, re.DOTALL):
        zz = re.search(r'z\s*:\s*(-?\d+\.?\d*(?:e[+\-]?\d+)?)', m.group(1))
        if zz:
            best = float(zz.group(1))
    return best


_NUM = r'(-?\d+\.?\d*(?:e[+\-]?\d+)?)'


def query_object_pose(name: str, timeout: float = 4.0):
    """Full world pose (xyz + quat xyzw) of a spawned gz model, or None.
    Used to render the object in RViz at its ACTUAL physics pose."""
    try:
        out = subprocess.run(
            ["gz", "topic", "-e", "-t", f"/world/{WORLD_NAME}/pose/info", "-n", "1"],
            capture_output=True, timeout=timeout).stdout.decode(errors="ignore")
    except Exception:
        return None
    import re
    best = None
    for m in re.finditer(
            r'pose\s*\{[^{}]*?name:\s*"' + re.escape(name)
            + r'"[^{}]*?position\s*\{([^{}]*?)\}[^{}]*?orientation\s*\{([^{}]*?)\}',
            out, re.DOTALL):
        pos = {k: float(v) for k, v in re.findall(r'([xyz])\s*:\s*' + _NUM, m.group(1))}
        ori = {k: float(v) for k, v in re.findall(r'([xyzw])\s*:\s*' + _NUM, m.group(2))}
        best = ([pos.get('x', 0.0), pos.get('y', 0.0), pos.get('z', 0.0)],
                [ori.get('x', 0.0), ori.get('y', 0.0), ori.get('z', 0.0), ori.get('w', 1.0)])
    return best


def _move(demo, arm, xyz, approach, execute, moveit_py, label, min_s=2.5,
          axis=None) -> bool:
    """Plan to a pose, execute it in physics (if executing), and animate it.
    Returns True iff the plan (and execution, when executing) succeeded."""
    demo.get_logger().info(f"  -> {label}")
    traj = plan_to_pose(arm, pose_stamped(xyz, approach, axis))
    if traj is None:
        demo.get_logger().warn(f"     plan failed at {label}")
        return False
    if execute and moveit_py is not None:
        try:
            moveit_py.execute(traj, controllers=["panda_arm_controller"])
        except Exception as exc:
            demo.get_logger().warn(f"     execute failed at {label}: {exc}")
            return False
    demo.animate(traj, time_scale=1.5, min_stage_seconds=min_s)
    return True


def pick_and_place_once(demo: DemoNode, arm, entry, spawn_cfg, place_offset,
                        execute: bool = False, moveit_py=None,
                        max_grasp_tries: int = 3):
    """One pick-and-place cycle: try ranked grasp candidates until one actually
    lifts the object (verified by its z rising in gz), then place it."""
    spawn = np.array([spawn_cfg["x"], spawn_cfg["y"], spawn_cfg["z"]])
    extents = np.array([0.05, 0.05, 0.05])
    if entry.get("extents") is not None:
        extents = np.asarray(entry["extents"])

    # Static table for MoveIt + RViz.
    demo.publish_scene()
    mesh_url = None
    visual_mesh = entry.get("visual_mesh")
    if visual_mesh and Path(visual_mesh).is_file():
        mesh_url = f"file://{visual_mesh}"
    name = entry["name"]
    # RViz shows the object at its REAL gz pose (tracked in a background thread),
    # so it reflects the actual physics — resting, grasped, lifted, dropped.
    demo.track_object(name, mesh_url, extents)

    # The grasp_planner now SCORES grasp points (narrow waist/handle near the CoM,
    # clear approach, deep friction margin), so the manifest's grasps are already
    # good choices. Use them best-first — no more synthetic vertical grasp (it
    # ignored the object's real graspable faces and ejected odd shapes).
    grasps = rank_grasps(entry["grasps"], entry.get("com")) if entry.get("grasps") else []
    if not grasps:
        demo.get_logger().warn(f"{name}: no grasp candidates")
        return False
    com = np.asarray(entry.get("com", [0.0, 0.0, 0.025]), float)
    aabb = entry.get("aabb")

    # Auto-size the closing force from the object's mass: heavier objects need a
    # firmer squeeze, lighter ones get ejected by too much force. This is the
    # initial guess; it's then adapted per attempt from the contact diagnostic.
    # Start near the empirical sweet spot (~48 N held a ~0.4 kg object) with a
    # mild mass scaling; the adaptive loop refines it. Starting low avoids
    # wasting attempts on ejects (the high-force failure mode).
    mass = float(entry.get("mass", 0.2))
    force_base = float(np.clip(38.0 + 30.0 * mass, 35.0, 60.0))
    force = force_base
    # panda_hand sits at panda_link8 (hand_joint origin = 0); the fingertip TCP
    # is ~0.1034 m further along the gripper +z. We plan panda_link8, so offset
    # targets by TIP_OFFSET (NOT 0.207, which double-counted the 0.107 link7->
    # link8 segment and left the WRIST — not the fingers — at the object).
    TIP_OFFSET = 0.1034

    # Grasp centers are in the object's local frame. The object FALLS from its
    # spawn height and settles ~5 cm lower, so anchor the grasp targets to its
    # actual settled origin (not the spawn pose) — otherwise every grasp aims
    # above the real object and the fingers close in empty air.
    base = spawn.copy()
    base_quat = np.array([0.0, 0.0, 0.0, 1.0])   # settled orientation (for tipped objects)
    if execute:
        p = query_object_pose(name)
        if p is not None:
            base = np.asarray(p[0], float)
            base_quat = np.asarray(p[1], float)
            demo.get_logger().info(
                f"  settled object origin {base.round(3).tolist()} "
                f"quat {base_quat.round(2).tolist()} (spawn was {spawn.round(3).tolist()})")

    picked = False
    grasp_pose = None
    attempts = 0          # real grasp ATTEMPTS (squeezes); plan-failures don't count
    PER_GRASP_TRIES = 3   # retry the SAME grasp at adjusted force before moving on
    gi = 0
    per_grasp = 0
    while attempts < max_grasp_tries and gi < len(grasps):
        g = grasps[gi]
        # The planner's center/approach/axis are in the OBJECT frame; rotate by
        # the settled orientation so a tipped object is still grasped correctly.
        center = base + _rotate(base_quat, np.asarray(g["center"]))
        approach = _rotate(base_quat, np.asarray(g["approach"], float))
        approach = approach / (np.linalg.norm(approach) + 1e-12)
        axis = (_rotate(base_quat, np.asarray(g["axis"])).tolist()
                if g.get("axis") is not None else None)
        grasp = center - approach * TIP_OFFSET
        pre = center - approach * (TIP_OFFSET + 0.08)
        lift = grasp + np.array([0.0, 0.0, 0.12])
        demo.get_logger().info(
            f"  grasp candidate (attempt {attempts + 1}/{max_grasp_tries}, "
            f"score={g.get('score', 0.0):.2f}, approach_z={approach[2]:+.2f}, "
            f"width={g.get('width', 0.0) * 1000:.0f}mm)")

        if execute:
            demo.set_finger_effort(demo.open_eff)
        # Collision-aware approach: make the object an obstacle while moving to
        # the pre-grasp (so the arm routes AROUND it instead of knocking it),
        # then remove it for the short local descent into finger contact.
        demo.set_object_collision(True, base, base_quat, extents, aabb)
        approached = _move(demo, arm, pre, approach, execute, moveit_py, "pre-grasp", axis=axis)
        demo.set_object_collision(False)
        if not approached:
            gi += 1; per_grasp = 0; continue   # plan failure: next grasp, no attempt burned
        if not _move(demo, arm, grasp, approach, execute, moveit_py, "grasp", axis=axis):
            gi += 1; per_grasp = 0; continue
        attempts += 1   # both poses planned -> this is a real grasp attempt
        per_grasp += 1

        # GENUINE grasp: squeeze the fingers onto the object with force; the
        # high finger+object friction (mu=30) holds it under gravity. There is
        # NO DetachableJoint weld — the hold is physics only. Commanding the
        # fingers to ~0 with allow_stalling makes them keep pressing against the
        # object (they stall at its width) rather than stopping at a gap.
        demo.get_logger().info(f"  -> close (squeeze, force={force:.0f} N)")
        z_before = None
        finger_after = None
        if execute:
            z_before = query_object_z(name)
            demo.set_finger_effort(-force)
            time.sleep(1.2)   # let the contact forces build up
            # Contact diagnostic: finger opening after squeeze. Stalled (>~0.004)
            # = the fingers are pressing on the object; ~0 = they closed through
            # empty space / shoved a light object out (ejected).
            finger_after = demo._finger_pos
            demo.get_logger().info(
                f"     finger opening after squeeze = {finger_after} "
                f"(0=closed, 0.04=open)")

        # Lift slowly so the object's inertia doesn't break the friction grip.
        _move(demo, arm, lift, approach, execute, moveit_py, "lift", min_s=3.5, axis=axis)

        if execute:
            z_after = query_object_z(name)
            # Genuine success: the object actually rose with the gripper (held
            # by friction). If it stayed on the table the grasp slipped.
            if z_before is not None and z_after is not None and (z_after - z_before) <= 0.05:
                # AUTO-TUNE the force from the failure mode (contact diagnostic),
                # in small ADDITIVE steps so it walks through the (often narrow)
                # holding window instead of multiplicatively jumping over it:
                #   fingers ~closed (<0.004) -> ejected/over-squeezed -> LESS force
                #   fingers stalled on object  -> grip too weak        -> MORE force
                ejected = finger_after is not None and finger_after < 0.004
                if ejected:
                    force = max(20.0, force - 8.0)
                    why = "ejected -> reduce force"
                else:
                    force = min(120.0, force + 8.0)
                    why = "weak grip -> increase force"
                demo.get_logger().warn(
                    f"     lift check SLIPPED: z {z_before:.3f} -> {z_after:.3f} "
                    f"({why}, next force={force:.0f} N)")
                demo.set_finger_effort(demo.open_eff)
                # The good grasp often just needs the right force, so RETRY THE
                # SAME grasp at the adjusted force (down after an eject, up after
                # a weak grip) for a few steps before moving to the next grasp.
                if per_grasp < PER_GRASP_TRIES:
                    continue                      # same gi -> retry this grasp
                gi += 1
                per_grasp = 0
                continue
            demo.get_logger().info(
                f"     lift check GRASPED (held by friction, {force:.0f} N): "
                f"z {z_before} -> {z_after}")
        picked = True
        grasp_pose = (grasp, approach, lift, axis)
        break

    if not picked:
        demo.get_logger().warn(f"  {name}: no successful grasp in {attempts} attempts")
        if execute:
            demo.set_finger_effort(demo.open_eff)
        go_home(demo, arm, execute, moveit_py)   # park clear of the spawn zone
        return False

    # ---- place the (physically held) object, then release it ----
    grasp, approach, lift, axis = grasp_pose
    place_pre = lift + np.asarray(place_offset)
    place = place_pre - np.array([0.0, 0.0, 0.06])
    retract = place + np.array([0.0, 0.0, 0.12])

    _move(demo, arm, place_pre, approach, execute, moveit_py, "transport", min_s=3.5, axis=axis)
    _move(demo, arm, place, approach, execute, moveit_py, "place", min_s=3.0, axis=axis)

    # Release: open the fingers; the object drops onto the table under gravity
    # (genuine physics — the RViz marker, tracked from the real pose, follows).
    demo.get_logger().info("  -> open (release)")
    if execute:
        demo.set_finger_effort(demo.open_eff)
        time.sleep(0.8)
    _move(demo, arm, retract, approach, execute, moveit_py, "retract", axis=axis)
    # Park at home so the next object spawns + settles with the arm clear, and
    # the next approach comes from above instead of sweeping the new object off.
    go_home(demo, arm, execute, moveit_py)
    return True


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--method", default="cem")
    parser.add_argument("--object-index", type=int, default=None,
                        help="If set, lock the demo to this one manifest "
                             "entry. Default: shuffle the method's pool and "
                             "spawn a different random object each cycle.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for the shuffle (default: clock).")
    parser.add_argument("--spawn-x", type=float, default=0.5)
    parser.add_argument("--spawn-y", type=float, default=0.0)
    parser.add_argument("--spawn-z", type=float, default=0.45,
                        help="Drop height in metres above the world origin. "
                             "Default 0.45 spawns the object 5 cm above the "
                             "table top (z=0.40) so it falls under gravity "
                             "and physically lands on the table.")
    parser.add_argument("--place-dx", type=float, default=0.0)
    parser.add_argument("--place-dy", type=float, default=0.15)
    parser.add_argument("--place-dz", type=float, default=0.0)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--no-gazebo-spawn", action="store_true")
    parser.add_argument("--max-grasp-tries", type=int, default=8,
                        help="Grasp ATTEMPTS (squeezes) per object before giving "
                             "up (each verified by the lift check); plan-failures "
                             "don't count. Grasps are tried best-first (planner "
                             "score) and the same grasp is retried at adjusted "
                             "force before moving to the next candidate.")
    parser.add_argument("--execute", action="store_true",
                        help="Also send each plan to panda_arm_controller via "
                             "moveit_py.execute() so the Panda physically moves "
                             "inside gz_sim. Requires use_gz_control:=true.")
    args = parser.parse_args(rclpy.utilities.remove_ros_args(sys.argv)[1:])

    manifest = json.loads(args.manifest.read_text())
    pool = [e for e in manifest if e["method"] == args.method]
    if not pool:
        sys.exit(f"no objects with method '{args.method}'")

    rng = random.Random(args.seed)
    # Pre-shuffle so the first object is not always cem_0000 / box-shaped.
    if args.object_index is None:
        rng.shuffle(pool)

    def _pick(cycle_idx: int):
        if args.object_index is not None:
            return pool[args.object_index % len(pool)]
        # Cycle through the shuffled pool; reshuffle once we've exhausted it
        # so each loop sees a different sampling order.
        if cycle_idx > 0 and cycle_idx % len(pool) == 0:
            rng.shuffle(pool)
        return pool[cycle_idx % len(pool)]

    entry = _pick(0)

    if not args.no_gazebo_spawn:
        if spawn_in_gazebo(Path(entry["sdf"]), entry["name"],
                           args.spawn_x, args.spawn_y, args.spawn_z):
            print(f"[demo] spawned {entry['name']} in gz_sim", flush=True)
            # Let it fall and settle on the table under gravity before MoveIt
            # plans against its resting pose. (No DetachableJoint anymore — the
            # grasp is genuine, held by finger friction.)
            time.sleep(1.2)
        else:
            print(f"[demo] WARN: spawn failed for {entry['name']}", flush=True)

    print("", flush=True)
    print("=" * 70, flush=True)
    print(" WHERE TO LOOK", flush=True)
    print("=" * 70, flush=True)
    print(" * RViz window: the Franka Panda's animated pick-and-place loop.", flush=True)
    print("                Wait ~3 s for the model to subscribe to /robot_description,", flush=True)
    print("                then the arm will move through 8 stages per cycle.", flush=True)
    print(" * gz_sim window: the generated object resting on the table.", flush=True)
    print("                  (The Panda is NOT spawned in Gazebo --- only the object", flush=True)
    print("                   is, so the gz physics scene shows the object alone.)", flush=True)
    print(" * If RViz is blank, set Fixed Frame to panda_link0 (Displays panel) and", flush=True)
    print("   verify /joint_states is publishing: 'ros2 topic hz /joint_states'.", flush=True)
    print("=" * 70, flush=True)
    print("", flush=True)

    # Order is load-bearing: create the rclpy DemoNode *before* MoveItPy
    # because MoveItPy's ctor calls rclcpp::init() which has, in some
    # moveit_py 2.12.x builds, clobbered the rclpy graph (DemoNode then
    # silently does not register, and nothing reaches /joint_states).
    rclpy.init()
    # In execute mode the joint_state_broadcaster owns /joint_states; don't
    # compete with it (see DemoNode docstring).
    demo = DemoNode(publish_joints=not args.execute)
    spinner = threading.Thread(target=rclpy.spin, args=(demo,), daemon=True)
    spinner.start()
    # Give the publisher time to advertise before MoveItPy seizes rclcpp.
    time.sleep(0.5)

    from moveit.planning import MoveItPy
    moveit_py = MoveItPy(node_name="demo_plan_driver")
    arm = moveit_py.get_planning_component("panda_arm")

    # Warm-up: MoveIt's trajectory_execution_manager connects its
    # FollowJointTrajectory action client lazily, so the very first execute()
    # races the connection ("Action client not connected"). When executing in
    # physics, give the client a few seconds to find panda_arm_controller
    # before the first real pick stage.
    if args.execute:
        demo.get_logger().info("waiting for controller action clients to connect ...")
        time.sleep(5.0)

    spawn_cfg = {"x": args.spawn_x, "y": args.spawn_y, "z": args.spawn_z}
    place_offset = (args.place_dx, args.place_dy, args.place_dz)

    # Start every session from the home config: gz spawns the Panda in a
    # different pose, and planning the first approach from there produced a large
    # swing that knocked the object. From home the approach comes from above.
    if args.execute:
        demo.set_finger_effort(demo.open_eff)
        go_home(demo, arm, execute=True, moveit_py=moveit_py)

    cycle = 0
    try:
        while True:
            pick_and_place_once(demo, arm, entry, spawn_cfg, place_offset,
                                execute=args.execute, moveit_py=moveit_py,
                                max_grasp_tries=args.max_grasp_tries)
            if not args.loop:
                break
            cycle += 1
            next_entry = _pick(cycle)
            # Despawn the current object before spawning the next so gz_sim
            # doesn't accumulate a pile of objects at the same xyz.  When
            # object_index is fixed (no shuffle) we keep the same entry, so
            # the despawn/respawn is skipped.
            if next_entry["name"] != entry["name"] and not args.no_gazebo_spawn:
                despawn_in_gazebo(entry["name"])
                if spawn_in_gazebo(Path(next_entry["sdf"]), next_entry["name"],
                                   args.spawn_x, args.spawn_y, args.spawn_z):
                    print(f"[demo] spawned {next_entry['name']} "
                          f"(cycle {cycle})", flush=True)
                    # Let gravity land + settle the object before the next plan.
                    time.sleep(1.2)
                else:
                    print(f"[demo] WARN: spawn failed for "
                          f"{next_entry['name']}", flush=True)
            entry = next_entry
    except KeyboardInterrupt:
        pass
    finally:
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
