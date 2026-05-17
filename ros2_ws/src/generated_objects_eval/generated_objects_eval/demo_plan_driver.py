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
from std_msgs.msg import ColorRGBA, Header
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


# ---------------------------------------------------------------------------

class DemoNode(Node):
    """Holds the publishers for joint_states + scene markers + grasped-object
    marker, and exposes helpers to animate joint trajectories."""

    def __init__(self, hand_open: float = 0.04, hand_closed: float = 0.0):
        super().__init__("demo_plan_driver_pub")
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

def plan_to_pose(arm, target: PoseStamped) -> Optional[object]:
    arm.set_start_state(configuration_name="ready")
    arm.set_goal_state(pose_stamped_msg=target, pose_link="panda_link8")
    result = arm.plan()
    if result and result.trajectory is not None:
        return result.trajectory
    return None


def pose_stamped(xyz: np.ndarray, approach: np.ndarray) -> PoseStamped:
    ps = PoseStamped()
    ps.header.frame_id = "panda_link0"
    ps.pose.position.x = float(xyz[0])
    ps.pose.position.y = float(xyz[1])
    ps.pose.position.z = float(xyz[2])
    ps.pose.orientation = look_at_quaternion(approach)
    return ps


def pose_xyz(xyz, ori=(0.0, 0.0, 0.0, 1.0)) -> Pose:
    return Pose(position=Point(x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2])),
                orientation=Quaternion(x=ori[0], y=ori[1], z=ori[2], w=ori[3]))


def pick_and_place_once(demo: DemoNode, arm, entry, spawn_cfg, place_offset,
                        execute: bool = False, moveit_py=None):
    """One full pick-and-place cycle for *entry*."""
    spawn = np.array([spawn_cfg["x"], spawn_cfg["y"], spawn_cfg["z"]])
    extents = np.array([0.05, 0.05, 0.05])
    if entry.get("extents") is not None:
        extents = np.asarray(entry["extents"])

    # Render the static table + the resting object once.  Show the real
    # visual mesh in RViz so the silhouette matches what gz_sim is
    # showing (hammer, mug, dumbbell, …) rather than a generic blue box.
    demo.publish_scene()
    mesh_url = None
    visual_mesh = entry.get("visual_mesh")
    if visual_mesh and Path(visual_mesh).is_file():
        mesh_url = f"file://{visual_mesh}"
    demo.publish_object(pose_xyz(spawn), extents,
                        attached_to_link="panda_link0",
                        mesh_resource=mesh_url)

    grasps = entry["grasps"]
    if not grasps:
        demo.get_logger().warn(f"{entry['name']}: no grasp candidates")
        return
    g = grasps[0]
    center = spawn + np.asarray(g["center"])
    approach = np.asarray(g["approach"])
    # The grasp planner reports ``center`` as the world-frame point where
    # the gripper fingertips should converge.  We plan against panda_link8,
    # which sits ~21 cm behind the fingertip along the gripper's +z axis
    # (panda_link7 -> joint8 -> hand -> fingers), so every target needs an
    # offset of TIP_OFFSET in the *negative* approach direction.  Without
    # this, link8 ends up at the grasp center and the fingertip would be
    # inside the table -> RRTConnect rejects the pose.
    TIP_OFFSET = 0.207
    pre = center - approach * (TIP_OFFSET + 0.08)
    grasp = center - approach * TIP_OFFSET
    lift = grasp + np.array([0.0, 0.0, 0.10])
    place_pre = lift + np.asarray(place_offset)
    place = place_pre - np.array([0.0, 0.0, 0.08])
    retract = place + np.array([0.0, 0.0, 0.10])

    sequence = [
        ("pre-grasp", pre, approach, False),
        ("grasp",     grasp, approach, False),
        ("close",     None,  None,     True),
        ("lift",      lift,  approach, True),
        ("transport", place_pre, approach, True),
        ("place",     place, approach, True),
        ("open",      None,  None,     False),
        ("retract",   retract, approach, False),
    ]
    for stage_name, xyz, app, gripper_closed in sequence:
        demo.get_logger().info(f"  -> {stage_name}")
        # Gripper width state.  In RViz-only mode this drives the
        # animated finger positions on /joint_states; in gz_ros2_control
        # mode it is overridden by the broadcaster's stream and the real
        # close/open happens via the action client below.
        demo.set_state(demo._current_q,
                       demo.hand_closed if gripper_closed else demo.hand_open)
        # When executing in physics, command the real gripper.  The
        # GripperActionController's "position" command is the *opening*
        # of the parallel jaw (0 = fully closed, 0.04 = fully open).
        if execute and stage_name in ("close", "open"):
            target = demo.hand_closed if stage_name == "close" else demo.hand_open
            # Higher max_effort during close lets the finger squeeze the
            # object firmly enough for friction to hold it during lift.
            effort = 40.0 if stage_name == "close" else 10.0
            ok = demo.send_gripper_goal(target, max_effort=effort)
            demo.get_logger().info(
                f"     gripper {stage_name} -> {'ok' if ok else 'no server'}")
        # Attach / detach the object marker.
        if stage_name in ("lift", "transport", "place"):
            # marker attached to the gripper hand, slightly below it
            demo.publish_object(pose_xyz([0.0, 0.0, 0.05]), extents,
                                attached_to_link="panda_hand",
                                color=(0.95, 0.55, 0.10, 1.0),
                                mesh_resource=mesh_url)
        elif stage_name == "open":
            # marker drops at the place location
            demo.publish_object(pose_xyz(place + np.array([0.0, 0.0, -0.04])),
                                extents,
                                attached_to_link="panda_link0",
                                color=(0.10, 0.55, 0.20, 1.0),
                                mesh_resource=mesh_url)
        if xyz is None:
            # Gripper-only stage: hold pose for a beat so the open/close is visible.
            time.sleep(1.0)
            continue
        traj = plan_to_pose(arm, pose_stamped(xyz, app))
        if traj is None:
            demo.get_logger().warn(f"     plan failed at {stage_name}")
            time.sleep(1.5)
            continue
        # In gz_ros2_control mode we also send the trajectory to the
        # FollowJointTrajectory controller so the Panda physically moves
        # inside gz_sim.  In RViz-only mode (default) we just animate
        # /joint_states locally.
        if execute and moveit_py is not None:
            try:
                moveit_py.execute(traj, controllers=["panda_arm_controller"])
            except Exception as exc:
                demo.get_logger().warn(f"     execute failed: {exc}")
        demo.animate(traj, time_scale=1.5, min_stage_seconds=2.5)
    # Return arm to ready.
    demo.set_state(PANDA_READY, demo.hand_open)
    time.sleep(0.8)


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
    parser.add_argument("--spawn-z", type=float, default=0.42)
    parser.add_argument("--place-dx", type=float, default=0.0)
    parser.add_argument("--place-dy", type=float, default=0.15)
    parser.add_argument("--place-dz", type=float, default=0.0)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--no-gazebo-spawn", action="store_true")
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
    demo = DemoNode()
    spinner = threading.Thread(target=rclpy.spin, args=(demo,), daemon=True)
    spinner.start()
    # Give the publisher time to advertise before MoveItPy seizes rclcpp.
    time.sleep(0.5)

    from moveit.planning import MoveItPy
    moveit_py = MoveItPy(node_name="demo_plan_driver")
    arm = moveit_py.get_planning_component("panda_arm")

    spawn_cfg = {"x": args.spawn_x, "y": args.spawn_y, "z": args.spawn_z}
    place_offset = (args.place_dx, args.place_dy, args.place_dz)

    cycle = 0
    try:
        while True:
            pick_and_place_once(demo, arm, entry, spawn_cfg, place_offset,
                                execute=args.execute, moveit_py=moveit_py)
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
