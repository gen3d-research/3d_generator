"""
MoveIt 2 motion-planning evaluator over a batch of generated objects.

For every object in the input manifest the node loads its visual mesh as a
collision object at the configured spawn pose, asks MoveIt 2 to plan a
trajectory from a fixed home configuration to each pre-grasp pose derived from
the pre-computed grasp candidates, and logs whether the plan succeeded.

This is the metric used in the revised manuscript to address R2-Q1 / Q2 / Q3
(circular evaluation) and the Associate Editor's request for an independent
downstream signal.

The node is intended to be launched via the bundled launch file so that the
MoveItPy parameters are wired through the ROS parameter server::

    ros2 launch generated_objects_eval moveit_planning_eval.launch.py \\
        manifest:=/abs/path/manifest.json out:=/abs/path/results.json \\
        max_objects:=10

Direct invocation works too once the required parameters are already on the
parameter server.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import List

import numpy as np
import rclpy
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion
from moveit.core.robot_state import RobotState
from moveit.planning import MoveItPy, PlanRequestParameters
from moveit_msgs.msg import CollisionObject, PlanningScene, PlanningSceneWorld
from rclpy.logging import get_logger
from shape_msgs.msg import Mesh, MeshTriangle, SolidPrimitive
from std_msgs.msg import Header

try:
    import trimesh
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("trimesh is required: pip install trimesh") from exc

LOGGER = get_logger("moveit_planning_eval")
DEFAULT_HOME = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]


# ---------------------------------------------------------------------------

def make_quaternion(x: float, y: float, z: float, w: float) -> Quaternion:
    return Quaternion(x=float(x), y=float(y), z=float(z), w=float(w))


def trimesh_to_mesh_msg(mesh: trimesh.Trimesh) -> Mesh:
    msg = Mesh()
    msg.vertices = [Point(x=float(v[0]), y=float(v[1]), z=float(v[2]))
                    for v in mesh.vertices]
    msg.triangles = [MeshTriangle(vertex_indices=[int(f[0]), int(f[1]), int(f[2])])
                     for f in mesh.faces]
    return msg


def look_at_quaternion(approach: np.ndarray) -> Quaternion:
    """End-effector pointing along +approach direction (so its tool axis aligns
    with the gripper closure direction towards the object)."""
    z_des = -approach / (np.linalg.norm(approach) + 1e-12)
    ref = np.array([0.0, 0.0, 1.0]) if abs(z_des[2]) < 0.95 else np.array([1.0, 0.0, 0.0])
    x_des = np.cross(ref, z_des)
    x_des = x_des / (np.linalg.norm(x_des) + 1e-12)
    y_des = np.cross(z_des, x_des)
    R = np.stack([x_des, y_des, z_des], axis=1)
    qw = 0.5 * math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2]))
    qx = (R[2, 1] - R[1, 2]) / (4 * qw + 1e-12)
    qy = (R[0, 2] - R[2, 0]) / (4 * qw + 1e-12)
    qz = (R[1, 0] - R[0, 1]) / (4 * qw + 1e-12)
    return make_quaternion(qx, qy, qz, qw)


# ---------------------------------------------------------------------------

class PlanningEvaluator:
    def __init__(self, planner_id: str, planning_time: float,
                 planning_group: str = "panda_arm", collision_aware: bool = True):
        self.moveit_py = MoveItPy(node_name="moveit_planning_eval")
        self.arm = self.moveit_py.get_planning_component(planning_group)
        self.planning_scene = self.moveit_py.get_planning_scene_monitor()
        self.robot_model = self.moveit_py.get_robot_model()
        self.planning_group = planning_group
        self.planner_id = planner_id
        self.planning_time = planning_time
        self.collision_aware = collision_aware

        # Collision objects reach MoveIt via /planning_scene DIFFS — the
        # apply_collision_object python binding segfaults in moveit_py 2.12.x,
        # but the latched planning-scene-diff topic is the pattern the demo
        # driver has used successfully all along (see demo_plan_driver /
        # README): the planning_scene_monitor ingests the diff asynchronously.
        self._helper_node = rclpy.create_node("moveit_planning_eval_helper")
        from rclpy.qos import (QoSProfile, ReliabilityPolicy, HistoryPolicy,
                               DurabilityPolicy)
        latched = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             history=HistoryPolicy.KEEP_LAST, depth=1)
        self._planning_scene_pub = self._helper_node.create_publisher(
            PlanningScene, "/planning_scene", latched)
        if self.collision_aware:
            self._publish_table()

    def _publish_diff(self, co: CollisionObject, settle_s: float = 0.25):
        scene = PlanningScene()
        scene.is_diff = True
        scene.world = PlanningSceneWorld()
        scene.world.collision_objects = [co]
        self._planning_scene_pub.publish(scene)
        import time
        time.sleep(settle_s)   # let the planning_scene_monitor ingest the diff

    def _publish_table(self, table_pose=(0.5, 0.0, 0.2), table_size=(0.8, 0.8, 0.4),
                       top_clearance: float = 0.015):
        """The table the objects rest on (matches panda_eval_world + the demo).
        Top shrunk by `top_clearance` so a fingertip touching the tabletop is not
        itself a collision (same convention as demo_plan_driver.publish_scene)."""
        co = CollisionObject()
        co.header.frame_id = "panda_link0"
        co.id = "eval_table"
        co.operation = CollisionObject.ADD
        prim = SolidPrimitive()
        prim.type = SolidPrimitive.BOX
        prim.dimensions = [float(table_size[0]), float(table_size[1]),
                           float(table_size[2]) - top_clearance]
        pose = Pose()
        pose.position.x = float(table_pose[0])
        pose.position.y = float(table_pose[1])
        pose.position.z = float(table_pose[2]) - top_clearance / 2.0
        pose.orientation.w = 1.0
        co.primitives = [prim]
        co.primitive_poses = [pose]
        self._publish_diff(co, settle_s=0.4)
        LOGGER.info("collision-aware: table published to the planning scene")

    def reset_to_home(self):
        # ``set_start_state_to_current_state`` is racy when the
        # current_state_monitor latches an early all-zero snapshot before
        # ``home_joint_state_publisher`` has started broadcasting; the
        # CheckStartStateCollision adapter then rejects 70-90% of plans on
        # subsequent seeds.  Pin the start state to the SRDF "ready" group
        # state (= the canonical Panda home configuration) instead.
        self.arm.set_start_state(configuration_name="ready")

    def set_collision_mesh(self, name: str, mesh_path: Path, pose: Pose):
        """Add the object's real collision mesh to the planning scene (diff topic).
        No-op in kinematic-only mode (collision_aware=False reproduces the
        as-submitted metric)."""
        if not self.collision_aware:
            return
        tm = trimesh.load(str(mesh_path), force="mesh")
        co = CollisionObject()
        co.header.frame_id = "panda_link0"
        co.id = name
        co.operation = CollisionObject.ADD
        co.meshes = [trimesh_to_mesh_msg(tm)]
        co.mesh_poses = [pose]
        self._publish_diff(co)

    def clear_scene(self, name: str):
        if not self.collision_aware:
            return
        co = CollisionObject()
        co.header.frame_id = "panda_link0"
        co.id = name
        co.operation = CollisionObject.REMOVE
        self._publish_diff(co, settle_s=0.1)

    def plan_to(self, target: PoseStamped, attempts: int = 5) -> dict:
        self.reset_to_home()
        self.arm.set_goal_state(pose_stamped_msg=target, pose_link="panda_link8")
        result = self.arm.plan()
        if result and result.trajectory is not None:
            traj = result.trajectory
            try:
                duration = float(traj.get_duration())
                n_points = len(traj.joint_trajectory.points)
            except Exception:
                duration, n_points = 0.0, 0
            return {"success": True, "duration_s": duration, "n_points": n_points}
        return {"success": False, "duration_s": 0.0, "n_points": 0}


# ---------------------------------------------------------------------------

# panda_hand sits at panda_link8; the fingertip TCP is ~0.1034 m further along the
# gripper +z (see demo_plan_driver.TIP_OFFSET — and its warning about 0.207). We plan
# panda_link8 poses, so the link8 goal must sit TIP_OFFSET behind the desired
# fingertip point. Without this, every "pre-grasp" placed the FINGERS ~5 cm PAST the
# grasp center — inside the object/table — which kinematic-only planning silently
# accepted and collision-aware planning correctly rejected at a rate of 100%.
TIP_OFFSET = 0.1034


def grasps_to_targets(grasps, object_spawn, pre_grasp_offset=0.05) -> List[PoseStamped]:
    spawn = np.array([object_spawn["x"], object_spawn["y"], object_spawn["z"]])
    out = []
    for g in grasps:
        center = spawn + np.asarray(g["center"])
        approach = np.asarray(g["approach"])
        pre = center - approach * (TIP_OFFSET + pre_grasp_offset)
        ps = PoseStamped()
        ps.header.frame_id = "panda_link0"
        ps.pose.position.x = float(pre[0])
        ps.pose.position.y = float(pre[1])
        ps.pose.position.z = float(pre[2])
        ps.pose.orientation = look_at_quaternion(approach)
        out.append(ps)
    return out


def run(manifest_path: Path, out_path: Path, config: dict, max_objects: int = 0,
        collision_aware: bool = True):
    with manifest_path.open() as f:
        manifest = json.load(f)
    if max_objects and max_objects > 0:
        manifest = manifest[:max_objects]
    plan_cfg = config["planning"]
    spawn_cfg = config["object_spawn"]

    evaluator = PlanningEvaluator(planner_id=plan_cfg["planner_id"],
                                  planning_time=plan_cfg["planning_time_s"],
                                  planning_group=plan_cfg["planning_group"],
                                  collision_aware=collision_aware)
    results = []
    for k, entry in enumerate(manifest):
        LOGGER.info(f"[{k+1}/{len(manifest)}] {entry['name']} ({entry['method']})")
        spawn_pose = Pose()
        spawn_pose.position.x = float(spawn_cfg["x"])
        spawn_pose.position.y = float(spawn_cfg["y"])
        spawn_pose.position.z = float(spawn_cfg["z"])
        spawn_pose.orientation.w = 1.0
        try:
            evaluator.set_collision_mesh(
                entry["name"],
                Path(entry.get("collision_mesh") or entry["visual_mesh"]),
                spawn_pose)
        except Exception as exc:
            LOGGER.error(f"Failed to set collision mesh: {exc}")
            results.append({"name": entry["name"], "method": entry["method"],
                            "error": str(exc), "any_success": False,
                            "n_success": 0, "n_grasps": 0})
            continue
        targets = grasps_to_targets(entry["grasps"], spawn_cfg,
                                    plan_cfg["pre_grasp_offset_m"])
        grasp_results = []
        for i, ps in enumerate(targets):
            r = evaluator.plan_to(ps, plan_cfg["num_planning_attempts"])
            r["grasp_idx"] = i
            grasp_results.append(r)
        n_success = sum(1 for r in grasp_results if r["success"])
        results.append({
            "name": entry["name"],
            "method": entry["method"],
            "n_grasps": len(grasp_results),
            "n_success": n_success,
            "any_success": n_success > 0,
            "per_grasp": grasp_results,
        })
        evaluator.clear_scene(entry["name"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write: the harness polls for this file and then HARD-KILLS the launch
    # (moveit_py 2.12 segfaults on shutdown), so a plain write_text can be observed
    # half-written (we have a moveit_results.json.race-corrupted to prove it).
    tmp_path = out_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps({"results": results, "config": config}, indent=2))
    tmp_path.replace(out_path)
    LOGGER.info(f"Wrote {len(results)} results to {out_path}")

    summary: dict = {}
    for r in results:
        m = r["method"]
        s = summary.setdefault(m, {"n": 0, "any": 0, "succ": 0, "tot": 0})
        s["n"] += 1
        s["any"] += int(r["any_success"])
        s["succ"] += r["n_success"]
        s["tot"] += r["n_grasps"]
    print("\n=== MoveIt 2 planning summary ===")
    for m, s in summary.items():
        any_rate = s["any"] / s["n"] if s["n"] else 0
        per_rate = s["succ"] / max(1, s["tot"])
        print(f"  {m:14s} n={s['n']:3d}  any_grasp_planned={any_rate:.1%}  "
              f"per_grasp_plan={per_rate:.1%}")


def main():
    argv = rclpy.utilities.remove_ros_args(args=sys.argv)
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--max-objects", type=int, default=0)
    parser.add_argument("--kinematic-only", action="store_true",
                        help="skip planning-scene collision objects (reproduces the "
                             "as-submitted kinematic-only metric)")
    args = parser.parse_args(argv[1:])

    import yaml
    cfg = yaml.safe_load(args.config.read_text())

    rclpy.init()
    try:
        run(args.manifest, args.out, cfg, args.max_objects,
            collision_aware=not args.kinematic_only)
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
