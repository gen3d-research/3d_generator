#!/usr/bin/env python3
"""Physics pick-and-lift in headless Gazebo, the counterpart of mujoco_eval.pick_and_lift.

A free-floating parallel-jaw gripper of the same geometry and masses, friction only, no weld and
no arm, so the physics engine is the difference. The world stays paused and advances in exact step
counts, so timing does not depend on machine load. Feasibility comes from the MuJoCo run on the
same manifest (only which grasps it executed, not their outcome). Each trial re-settles the object.

Run with ROS 2 sourced, under the system Python (gz bindings):

    python3 scripts/gazebo_lift_eval.py --manifest M.json --mujoco mujoco_eval_coacd.json \\
        --out gazebo_lift.json --workers 8
    python3 scripts/gazebo_lift_eval.py --control    # harness check on a box
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
from gz.msgs10.boolean_pb2 import Boolean
from gz.msgs10.contacts_pb2 import Contacts
from gz.msgs10.double_pb2 import Double
from gz.msgs10.entity_factory_pb2 import EntityFactory
from gz.msgs10.entity_pb2 import Entity
from gz.msgs10.pose_v_pb2 import Pose_V
from gz.msgs10.world_control_pb2 import WorldControl
from gz.msgs10.world_stats_pb2 import WorldStatistics
from gz.transport13 import Node

WORLD = "lift_world"
TABLE_TOP = 0.4
MU = 0.5
GRIP_N = 20.0
MAX_OPENING = 0.08
PREGRASP = 0.10
LIFT = 0.10
LIFT_FRAC = 0.7
PRESHAPE = 0.015
HAND_TOL = 0.005
HOLD_OPEN_N = 5.0
XY = (0.5, 0.0)

WORLD_SDF = f"""<?xml version="1.0"?>
<sdf version="1.10"><world name="{WORLD}">
  <physics type="dart"><max_step_size>0.001</max_step_size><real_time_factor>0</real_time_factor></physics>
  <gravity>0 0 -9.81</gravity>
  <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
  <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
  <plugin filename="gz-sim-contact-system" name="gz::sim::systems::Contact"/>
  <model name="table"><static>true</static><pose>0.5 0 0.2 0 0 0</pose>
    <link name="table_link"><collision name="collision">
      <geometry><box><size>0.8 0.8 0.4</size></box></geometry>
      <surface><friction><ode><mu>{MU}</mu><mu2>{MU}</mu2></ode></friction></surface>
    </collision></link></model>
</world></sdf>
"""

POSE_PLUGIN = ('<plugin filename="gz-sim-pose-publisher-system" name="gz::sim::systems::PosePublisher">'
               '<publish_link_pose>true</publish_link_pose><publish_model_pose>true</publish_model_pose>'
               '<publish_collision_pose>false</publish_collision_pose><publish_visual_pose>false</publish_visual_pose>'
               '<publish_nested_model_pose>false</publish_nested_model_pose>'
               '<use_pose_vector_msg>true</use_pose_vector_msg><update_frequency>200</update_frequency></plugin>')


class WorldDead(RuntimeError):
    """The world stopped answering; the chunk restarts in a fresh one."""


def _rot_from_quat(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _rpy_from_rot(R):
    p = np.arcsin(np.clip(-R[2, 0], -1, 1))
    if abs(np.cos(p)) > 1e-6:
        return np.arctan2(R[2, 1], R[2, 2]), p, np.arctan2(R[1, 0], R[0, 0])
    return np.arctan2(-R[1, 2], R[1, 1]), p, 0.0


def hand_frame(approach, axis):
    z = np.asarray(approach, float)
    z = z / (np.linalg.norm(z) + 1e-12)
    y = np.asarray(axis, float)
    y = y - np.dot(y, z) * z
    ny = np.linalg.norm(y)
    if ny < 1e-6:
        return None
    y /= ny
    return np.column_stack([np.cross(y, z), y, z])


def _box(name, size, pos):
    s = " ".join(f"{x:.6f}" for x in size)
    return (f'<collision name="{name}"><pose>{pos[0]} {pos[1]} {pos[2]} 0 0 0</pose>'
            f'<geometry><box><size>{s}</size></box></geometry>'
            f'<surface><friction><ode><mu>{MU}</mu><mu2>{MU}</mu2></ode></friction></surface></collision>')


def _inertial(mass, pos, i):
    return (f'<inertial><mass>{mass}</mass><pose>{pos[0]} {pos[1]} {pos[2]} 0 0 0</pose><inertia>'
            f'<ixx>{i}</ixx><iyy>{i}</iyy><izz>{i}</izz></inertia></inertial>')


def _finger(link, sign, open_q):
    """Finger: joint q = 0 at the pre-shape, positive towards the centre."""
    y = sign * open_q
    joint = link + "_j"
    return f"""
    <link name="{link}"><gravity>false</gravity><pose>0 {y:.6f} 0 0 0 0</pose>
      {_inertial(0.05, (0, sign * 0.005, -0.01), 1e-5)}
      {_box(link + '_pad', (0.02, 0.01, 0.04), (0, sign * 0.005, -0.01))}
      <sensor name="{link}_contact" type="contact">
        <contact><collision>{link}_pad</collision></contact>
      </sensor>
    </link>
    <joint name="{joint}" type="prismatic"><parent>hand</parent><child>{link}</child>
      <axis><xyz>0 {-sign} 0</xyz><dynamics><damping>5</damping></dynamics>
        <limit><lower>0</lower><upper>{open_q:.6f}</upper><effort>200</effort></limit></axis>
    </joint>
    <plugin filename="gz-sim-apply-joint-force-system" name="gz::sim::systems::ApplyJointForce">
      <joint_name>{joint}</joint_name></plugin>"""


def hand_sdf(name, pre, Rh, open_q):
    rpy = _rpy_from_rot(Rh)
    up = Rh.T @ np.array([0.0, 0.0, 1.0])
    palm_half = MAX_OPENING / 2 + 0.005
    vel = ('<plugin filename="gz-sim-joint-controller-system" name="gz::sim::systems::JointController">'
           '<joint_name>{}</joint_name><initial_velocity>0</initial_velocity></plugin>')
    return f"""<?xml version="1.0"?>
<sdf version="1.10"><model name="{name}">
  <pose>{pre[0]:.6f} {pre[1]:.6f} {pre[2]:.6f} {rpy[0]:.6f} {rpy[1]:.6f} {rpy[2]:.6f}</pose>
  <link name="carriage"><gravity>false</gravity>{_inertial(0.1, (0, 0, 0), 1e-3)}</link>
  <joint name="lift" type="prismatic"><parent>world</parent><child>carriage</child>
    <axis><xyz expressed_in="__model__">{up[0]:.9f} {up[1]:.9f} {up[2]:.9f}</xyz>
      <limit><lower>-1</lower><upper>1</upper><effort>1e6</effort></limit></axis></joint>
  <link name="hand"><gravity>false</gravity>{_inertial(0.6, (0, 0, -0.04), 1e-3)}
    {_box('palm', (0.024, 2 * palm_half, 0.02), (0, 0, -0.04))}</link>
  <joint name="approach" type="prismatic"><parent>carriage</parent><child>hand</child>
    <axis><xyz>0 0 1</xyz><limit><lower>-1</lower><upper>1</upper><effort>1e6</effort></limit></axis></joint>
  {_finger('lf', 1, open_q)}
  {_finger('rf', -1, open_q)}
  {vel.format('lift')}
  {vel.format('approach')}
  {POSE_PLUGIN}
</model></sdf>"""


def object_sdf(sdf_path: Path) -> tuple[str, np.ndarray]:
    """A copy of the object's SDF beside the original (relative mesh paths), friction set to MU,
    publishing its link pose. Returns the path and the centre of mass in the link frame."""
    t = sdf_path.read_text()
    m = re.search(r"<inertial>.*?<pose>([^<]+)</pose>", t, re.S)
    com = np.array(m.group(1).split()[:3], float) if m else np.zeros(3)
    t = re.sub(r"<mu>[^<]*</mu>", f"<mu>{MU}</mu>", t)
    t = re.sub(r"<mu2>[^<]*</mu2>", f"<mu2>{MU}</mu2>", t)
    t = t.replace("</model>", POSE_PLUGIN + "</model>")
    out = sdf_path.with_name(sdf_path.stem + ".lift.sdf")
    out.write_text(t)
    return str(out), com


class World:
    """One paused gz server on this process's partition, advanced in exact step counts."""

    def __init__(self, tmpdir: str):
        path = Path(tmpdir) / f"lift_world_{os.getpid()}.sdf"
        path.write_text(WORLD_SDF)
        self.log = open(Path(tmpdir) / f"gz_{os.getpid()}.log", "ab")
        self.proc = subprocess.Popen(["gz", "sim", "-s", str(path)], stdout=self.log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        self.node = Node()
        self.iterations = -1
        self.poses = {}
        self.contacts = {"lf": set(), "rf": set()}
        self.lock = threading.Lock()
        self.node.subscribe(WorldStatistics, f"/world/{WORLD}/stats", self._stats)
        deadline = time.time() + 60
        while self.iterations < 0:
            if time.time() > deadline or self.proc.poll() is not None:
                raise WorldDead("world did not start")
            time.sleep(0.05)

    def _stats(self, msg):
        self.iterations = msg.iterations

    def _contact(self, pad, msg):
        with self.lock:
            for c in msg.contact:
                self.contacts[pad].update((c.collision1.name, c.collision2.name))

    def watch_pose(self, model):
        def cb(msg):
            with self.lock:
                for p in msg.pose:
                    self.poses[p.name] = (np.array([p.position.x, p.position.y, p.position.z]),
                                          np.array([p.orientation.x, p.orientation.y,
                                                    p.orientation.z, p.orientation.w]))
        self.node.subscribe(Pose_V, f"/model/{model}/pose", cb)

    def watch_contacts(self, hand):
        for pad in ("lf", "rf"):
            self.node.subscribe(Contacts,
                                f"/world/{WORLD}/model/{hand}/link/{pad}/sensor/{pad}_contact/contact",
                                lambda m, pad=pad: self._contact(pad, m))

    def pose(self, name):
        """A model's pose is in the world frame, a link's (model::link) in its model's frame."""
        with self.lock:
            return self.poses.get(name)

    def step(self, n):
        if self.proc.poll() is not None:
            raise WorldDead("world exited")
        target = self.iterations + n
        req = WorldControl(multi_step=n, pause=True)
        ok, rep = self.node.request(f"/world/{WORLD}/control", req, WorldControl, Boolean, 10000)
        if not ok or not rep.data:
            raise WorldDead("control request failed")
        deadline = time.time() + max(60.0, n * 0.1)
        while self.iterations < target:
            if time.time() > deadline or self.proc.poll() is not None:
                raise WorldDead("world stopped stepping")
            time.sleep(0.005)
        time.sleep(0.03)        # the pose and contact messages of the last steps

    def spawn(self, name, xyz=(0, 0, 0), sdf=None, filename=None):
        req = EntityFactory(name=name, allow_renaming=False)
        if sdf is not None:
            req.sdf = sdf
        else:
            req.sdf_filename = filename
        req.pose.position.x, req.pose.position.y, req.pose.position.z = xyz if sdf is None else (0, 0, 0)
        if sdf is not None:         # the hand's pose is in its own SDF
            req.ClearField("pose")
        ok, rep = self.node.request(f"/world/{WORLD}/create", req, EntityFactory, Boolean, 15000)
        if not ok or not rep.data:
            raise WorldDead(f"spawn of {name} failed")
        self.step(1)

    def remove(self, name):
        req = Entity(name=name, type=Entity.MODEL)
        ok, rep = self.node.request(f"/world/{WORLD}/remove", req, Entity, Boolean, 15000)
        if not ok or not rep.data:
            raise WorldDead(f"remove of {name} failed")
        self.step(1)

    def publisher(self, topic, wait=10.0):
        pub = self.node.advertise(topic, Double)
        deadline = time.time() + wait
        while not pub.has_connections():
            if time.time() > deadline:
                raise WorldDead(f"no subscriber on {topic}")
            self.step(1)
        return pub

    def close(self):
        try:
            os.killpg(self.proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        self.proc.wait()
        self.log.close()


def trial(w: World, sdf_file, com, grasp, tag) -> dict:
    obj, hand = f"obj_{tag}", f"hand_{tag}"
    w.poses.clear()
    w.watch_pose(obj)
    w.spawn(obj, (XY[0], XY[1], TABLE_TOP + 0.002), filename=sdf_file)
    w.step(1000)
    pose = w.pose(obj)
    if pose is None:
        raise WorldDead("no object pose")
    p_o, q_o = pose
    R_o = _rot_from_quat(q_o)
    Rh = hand_frame(R_o @ np.asarray(grasp["approach"], float), R_o @ np.asarray(grasp["axis"], float))
    if Rh is None:
        w.remove(obj)
        return {"success": False, "reason": "degenerate_frame"}
    c = p_o + R_o @ np.asarray(grasp["center"], float)
    a = Rh[:, 2]
    pre = c - PREGRASP * a
    open_q = min(MAX_OPENING / 2, float(grasp.get("width", 0.0)) / 2 + PRESHAPE)

    w.watch_pose(hand)
    w.watch_contacts(hand)
    w.spawn(hand, sdf=hand_sdf(hand, pre, Rh, open_q))
    base = f"/model/{hand}/joint"
    v_lift = w.publisher(f"{base}/lift/cmd_vel")
    v_app = w.publisher(f"{base}/approach/cmd_vel")
    fingers = [w.publisher(f"{base}/{f}_j/cmd_force") for f in ("lf", "rf")]

    def send(pub, value):
        pub.publish(Double(data=float(value)))

    def forces(value):
        for f in fingers:
            send(f, value)
        time.sleep(0.05)

    def com_z():
        p, q = w.pose(obj)
        return float(p[2] + (_rot_from_quat(q) @ com)[2])

    forces(-HOLD_OPEN_N)
    send(v_app, PREGRASP / 0.6)
    time.sleep(0.05)
    w.step(600)
    send(v_app, 0.0)
    time.sleep(0.05)
    w.step(100)
    forces(GRIP_N)
    w.step(580)
    with w.lock:
        w.contacts["lf"].clear()
        w.contacts["rf"].clear()
    w.step(20)
    with w.lock:
        closed_on = [any(obj in n for n in w.contacts[f]) for f in ("lf", "rf")]
    z0 = com_z()
    send(v_lift, LIFT / 1.0)
    time.sleep(0.05)
    w.step(1000)
    send(v_lift, 0.0)
    time.sleep(0.05)
    w.step(480)
    with w.lock:
        w.contacts["lf"].clear()
        w.contacts["rf"].clear()
    w.step(20)
    with w.lock:
        held = [any(obj in n for n in w.contacts[f]) for f in ("lf", "rf")]
    rise = com_z() - z0
    # The hand's origin should be at the grasp centre after the approach, LIFT higher after the lift.
    hand_end = w.pose(hand)[0] + Rh @ w.pose(f"{hand}::hand")[0]   # the carriage is canonical
    hand_err = float(np.linalg.norm(hand_end - (c + [0.0, 0.0, LIFT])))
    success = bool(rise >= LIFT_FRAC * LIFT and all(held))
    if success:
        reason = "ok"
    elif not any(closed_on):
        reason = "missed"
    elif rise < 0.02:
        reason = "dropped"
    else:
        reason = "slipped"
    w.remove(hand)
    w.remove(obj)
    return {"success": success, "reason": reason, "rise_m": rise, "closed_on": closed_on,
            "held": held, "hand_err_m": hand_err}


def eval_chunk(chunk, tmpdir, tag_base=0) -> list[dict]:
    """Objects in order, restarting the world after a crash; a trial crashed twice is recorded."""
    results, w = [], None
    try:
        for i, e in enumerate(chunk):
            sdf_file, com = object_sdf(Path(e["sdf"]))
            trials = []
            for rank in e["ranks"]:
                grasp = e["grasps"][rank]
                rec = None
                for attempt in range(2):
                    try:
                        if w is None:
                            w = World(tmpdir)
                        rec = trial(w, sdf_file, com, grasp, f"{tag_base + i}_{rank}_{attempt}")
                        break
                    except WorldDead as err:
                        rec = {"success": False, "reason": "crashed", "error": str(err)}
                        if w is not None:
                            w.close()
                        w = None
                rec["rank"] = rank
                trials.append(rec)
            succ = [t["success"] for t in trials]
            results.append({"name": e["name"], "method": e["method"], "has_feasible": bool(trials),
                            "lift_first": bool(succ[0]) if succ else False,
                            "lift_any": bool(any(succ)), "trials": trials})
    finally:
        if w is not None:
            w.close()
    return results


def _worker(args):
    chunk = json.load(open(args.chunk_in))
    res = eval_chunk(chunk, str(Path(args.chunk_out).parent))
    json.dump({"results": res}, open(args.chunk_out, "w"))


def _run_all(args):
    man = json.load(open(args.manifest))
    if args.methods:
        man = [e for e in man if e["method"] in args.methods]
    mj = {r["name"]: r for r in json.load(open(args.mujoco))["results"]}
    entries = []
    for e in man:
        trials = mj[e["name"]].get("lift", {}).get("trials", [])
        entries.append({"name": e["name"], "method": e["method"], "sdf": e["sdf"],
                        "grasps": e.get("grasps", []), "ranks": [t["rank"] for t in trials]})
    out = Path(args.out)
    chunks_dir = out.parent / f"_lift_{out.stem}"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    chunks = [entries[i:i + args.chunk] for i in range(0, len(entries), args.chunk)]
    paths = []
    for j, c in enumerate(chunks):
        json.dump(c, open(chunks_dir / f"in_{j:04d}.json", "w"))
        paths.append((chunks_dir / f"in_{j:04d}.json", chunks_dir / f"out_{j:04d}.json"))
    pending = [(i, o) for i, o in paths if not o.exists()]
    base = os.environ.get("GZ_PARTITION", "gen3d")
    slots = {}
    free = list(range(args.workers))
    while pending or slots:
        while pending and free:
            i, o = pending.pop(0)
            s = free.pop()
            env = dict(os.environ, GZ_PARTITION=f"{base}_lift{os.getpid()}_{s}")
            log = open(o.with_suffix(".log"), "ab")
            p = subprocess.Popen([sys.executable, __file__, "--chunk-in", str(i), "--chunk-out", str(o)],
                                 env=env, stdout=log, stderr=subprocess.STDOUT)
            slots[p] = (s, o, log, time.time())
        for p in list(slots):
            s, o, log, t0 = slots[p]
            if p.poll() is not None or time.time() - t0 > args.chunk_timeout:
                if p.poll() is None:
                    p.kill()
                    p.wait()
                log.close()
                free.append(s)
                del slots[p]
                print(f"[lift] {o.name} exit={p.returncode} {time.time() - t0:.0f}s", flush=True)
        time.sleep(0.5)
    merged, missing = [], 0
    for _, o in paths:
        if o.exists():
            merged += json.load(open(o))["results"]
        else:
            missing += 1
    json.dump({"results": merged, "missing_chunks": missing, "friction": MU, "grip_n": GRIP_N},
              open(out, "w"), indent=1)
    print(f"[lift] merged {len(merged)} results, {missing} chunks missing -> {out}", flush=True)


CONTROL_BOX = """<?xml version="1.0"?>
<sdf version="1.7"><model name="box"><link name="box_link">
  <inertial><mass>0.125</mass><pose>0 0 0.025 0 0 0</pose><inertia><ixx>5.2e-5</ixx><iyy>5.2e-5</iyy><izz>5.2e-5</izz></inertia></inertial>
  <collision name="c"><pose>0 0 0.025 0 0 0</pose><geometry><box><size>0.05 0.05 0.05</size></box></geometry>
    <surface><friction><ode><mu>1</mu><mu2>1</mu2></ode></friction></surface></collision>
</link></model></sdf>
"""


def _control():
    """A central side grasp must lift the box; a grasp beside it must close on air."""
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "box.sdf"
        p.write_text(CONTROL_BOX)
        sdf_file, com = object_sdf(p)
        good = {"center": [0, 0, 0.025], "approach": [1, 0, 0], "axis": [0, 1, 0], "width": 0.05}
        air = {"center": [0, 0.12, 0.025], "approach": [1, 0, 0], "axis": [0, 1, 0], "width": 0.05}
        top = {"center": [0, 0, 0.025], "approach": [0, 0, -1], "axis": [0, 1, 0], "width": 0.05}
        w = World(tmp)
        try:
            for name, g in (("side", good), ("top", top), ("air", air)):
                t0 = time.time()
                r = trial(w, sdf_file, com, g, f"c_{name}")
                print(name, f"{time.time() - t0:.1f}s", r, flush=True)
        finally:
            w.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest")
    ap.add_argument("--mujoco")
    ap.add_argument("--out")
    ap.add_argument("--methods", nargs="+")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=25)
    ap.add_argument("--chunk-timeout", type=float, default=3600)
    ap.add_argument("--chunk-in")
    ap.add_argument("--chunk-out")
    ap.add_argument("--control", action="store_true")
    args = ap.parse_args()
    if args.control:
        _control()
    elif args.chunk_in:
        _worker(args)
    else:
        _run_all(args)


if __name__ == "__main__":
    main()
