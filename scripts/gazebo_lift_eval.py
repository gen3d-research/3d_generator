#!/usr/bin/env python3
"""Physics pick-and-lift in headless Gazebo, the counterpart of mujoco_eval.pick_and_lift.

A free-floating parallel-jaw gripper of the same geometry and masses, friction only, no weld and
no arm, so the physics engine is the difference. Each object settles once; each trial is a world of
its own, started from that pose in a process of its own, run in-process (gz.sim8) in exact step
counts: nothing goes through transport discovery, which loses messages when many worlds share a
loaded machine, and timing does not depend on load. Feasibility comes from the MuJoCo run on the
same manifest (only which grasps it executed, not their outcome).

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
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import gz.math7  # noqa: F401  (registers the pose types gz.sim8 returns)
import numpy as np

WORLD = "lift_world"
TABLE_TOP = 0.4
MU = 0.5
GRIP_N = 20.0
MAX_OPENING = 0.08
PREGRASP = 0.10
LIFT = 0.10
LIFT_FRAC = 0.7
PRESHAPE = 0.015
HOLD_OPEN_N = 5.0
XY = (0.5, 0.0)
# Phases in 1 ms steps: settle; approach, hold; close; lift, hold. The last 20 steps of the close and
# of the final hold are the windows in which pad contact is read.
SETTLE = 1000
APPROACH_END = 600
HOLD_END = 700
CLOSE_END = 1300
LIFT_END = 2300
TRIAL_END = 2800
WINDOW = 20
PHASES = ("approach", "close", "lift")
OFF_PATH_M = 0.005
PHASE_DEV_M = 0.001

WORLD_SDF = """<?xml version="1.0"?>
<sdf version="1.10"><world name="{world}">
  <physics type="dart"><max_step_size>0.001</max_step_size><real_time_factor>0</real_time_factor></physics>
  <gravity>0 0 -9.81</gravity>
  <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
  <plugin filename="gz-sim-contact-system" name="gz::sim::systems::Contact"/>
  <model name="table"><static>true</static><pose>0.5 0 0.2 0 0 0</pose>
    <link name="table_link"><collision name="collision">
      <geometry><box><size>0.8 0.8 0.4</size></box></geometry>
      <surface><friction><ode><mu>{mu}</mu><mu2>{mu}</mu2></ode></friction></surface>
    </collision></link></model>
  {models}
</world></sdf>
"""


def rot_from_quat(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def rpy_from_rot(R):
    s = np.hypot(R[0, 0], R[1, 0])
    p = np.arctan2(-R[2, 0], s)
    if s > 1e-4:
        return np.arctan2(R[2, 1], R[2, 2]), p, np.arctan2(R[1, 0], R[0, 0])
    return np.arctan2(-R[1, 2], R[1, 1]), p, 0.0      # gimbal lock: all yaw goes into roll


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


def _pose(p, R):
    r = rpy_from_rot(R)
    return f"{p[0]:.9f} {p[1]:.9f} {p[2]:.9f} {r[0]:.9f} {r[1]:.9f} {r[2]:.9f}"


def _box(name, size, pos):
    s = " ".join(f"{x:.6f}" for x in size)
    return (f'<collision name="{name}"><pose>{pos[0]} {pos[1]} {pos[2]} 0 0 0</pose>'
            f'<geometry><box><size>{s}</size></box></geometry>'
            f'<surface><friction><ode><mu>{MU}</mu><mu2>{MU}</mu2></ode></friction></surface></collision>')


def _inertial(mass, pos, i):
    return (f'<inertial><mass>{mass}</mass><pose>{pos[0]} {pos[1]} {pos[2]} 0 0 0</pose><inertia>'
            f'<ixx>{i}</ixx><iyy>{i}</iyy><izz>{i}</izz></inertia></inertial>')


def unexplained_off_path(hand_err, dev, contact) -> bool:
    """Off the commanded path, and some phase in which the hand deviated had no hand contact."""
    return hand_err > OFF_PATH_M and any(d > PHASE_DEV_M and not contact[ph] for ph, d in dev.items())


def _sensor(name, collision):
    return (f'<sensor name="{name}_contact" type="contact"><contact><collision>{collision}</collision>'
            '</contact></sensor>')


def _finger(link, sign, open_q):
    """Finger: joint q = 0 at the pre-shape, positive towards the centre."""
    return f"""
    <link name="{link}"><gravity>false</gravity><pose>0 {sign * open_q:.6f} 0 0 0 0</pose>
      {_inertial(0.05, (0, sign * 0.005, -0.01), 1e-5)}
      {_box(link + '_pad', (0.02, 0.01, 0.04), (0, sign * 0.005, -0.01))}
      {_sensor(link, link + '_pad')}
    </link>
    <joint name="{link}_j" type="prismatic"><parent>hand</parent><child>{link}</child>
      <axis><xyz>0 {-sign} 0</xyz><dynamics><damping>5</damping></dynamics>
        <limit><lower>0</lower><upper>{open_q:.6f}</upper><effort>200</effort></limit></axis>
    </joint>"""


def hand_sdf(pre, Rh, open_q):
    up = Rh.T @ np.array([0.0, 0.0, 1.0])
    palm_half = MAX_OPENING / 2 + 0.005
    return f"""<model name="hand">
  <pose>{_pose(pre, Rh)}</pose>
  <link name="carriage"><gravity>false</gravity>{_inertial(0.1, (0, 0, 0), 1e-3)}</link>
  <joint name="lift" type="prismatic"><parent>world</parent><child>carriage</child>
    <axis><xyz expressed_in="__model__">{up[0]:.9f} {up[1]:.9f} {up[2]:.9f}</xyz>
      <limit><lower>-1</lower><upper>1</upper><effort>1e6</effort></limit></axis></joint>
  <link name="hand"><gravity>false</gravity>{_inertial(0.6, (0, 0, -0.04), 1e-3)}
    {_box('palm', (0.024, 2 * palm_half, 0.02), (0, 0, -0.04))}
    {_sensor('hand', 'palm')}</link>
  <joint name="approach" type="prismatic"><parent>carriage</parent><child>hand</child>
    <axis><xyz>0 0 1</xyz><limit><lower>-1</lower><upper>1</upper><effort>1e6</effort></limit></axis></joint>
  {_finger('lf', 1, open_q)}
  {_finger('rf', -1, open_q)}
</model>"""


def object_sdf(sdf_path, p, R) -> str:
    """The object's model as inline SDF: friction set to MU, absolute mesh paths, the given pose."""
    sdf_path = Path(sdf_path)
    t = sdf_path.read_text()
    t = re.sub(r"<mu>[^<]*</mu>", f"<mu>{MU}</mu>", t)
    t = re.sub(r"<mu2>[^<]*</mu2>", f"<mu2>{MU}</mu2>", t)
    t = re.sub(r"<uri>(?!/|file:)([^<]+)</uri>",
               lambda m: f"<uri>{(sdf_path.parent / m.group(1)).resolve()}</uri>", t)
    m = re.search(r"<model\b[^>]*>.*</model>", t, re.S)
    return re.sub(r"<model\b[^>]*>", f'<model name="obj"><pose>{_pose(p, R)}</pose>', m.group(0), count=1)


def _fixture(models, tmpdir, name):
    from gz.sim8 import TestFixture
    path = Path(tmpdir) / f"{name}.sdf"
    path.write_text(WORLD_SDF.format(world=WORLD, mu=MU, models=models))
    return TestFixture(str(path))


def _vec(v):
    return np.array([v.x(), v.y(), v.z()])


def _quat(q):
    return np.array([q.x(), q.y(), q.z(), q.w()])


def settle(req, tmpdir) -> dict:
    """The object's pose after SETTLE steps from just above the table, upright."""
    from gz.sim8 import Link, Model, World, world_entity
    fix = _fixture(object_sdf(req["sdf"], (XY[0], XY[1], TABLE_TOP + 0.002), np.eye(3)), tmpdir, "settle")
    out = {}

    def post(info, ecm):
        if info.iterations == SETTLE:
            model = Model(World(world_entity(ecm)).model_by_name(ecm, "obj"))
            pose = Link(model.canonical_link(ecm)).world_pose(ecm)
            out.update(p=_vec(pose.pos()).tolist(), q=_quat(pose.rot()).tolist())

    fix.on_post_update(post)
    fix.finalize()
    fix.server().run(True, SETTLE, False)
    return out


def run_trial(req, tmpdir) -> dict:
    from gz.msgs10.contacts_pb2 import Contacts
    from gz.sim8 import Joint, Link, Model, World, world_entity
    from gz.transport13 import Node
    p_o, R_o = np.array(req["p"]), rot_from_quat(req["q"])
    g = req["grasp"]
    Rh = hand_frame(R_o @ np.asarray(g["approach"], float), R_o @ np.asarray(g["axis"], float))
    if Rh is None:
        return {"success": False, "reason": "degenerate_frame"}
    c = p_o + R_o @ np.asarray(g["center"], float)
    pre = c - PREGRASP * Rh[:, 2]
    open_q = min(MAX_OPENING / 2, float(g.get("width", 0.0)) / 2 + PRESHAPE)
    fix = _fixture(object_sdf(req["sdf"], p_o, R_o) + hand_sdf(pre, Rh, open_q), tmpdir, "trial")

    s = {"it": 0, "closed": set(), "held": set(), "z0": None, "z1": None, "hand": {}, "joints": None,
         "contact": {ph: set() for ph in PHASES}}
    node = Node()

    def contact(part):
        def cb(msg):
            it = s["it"]
            phase = "approach" if it <= HOLD_END else "close" if it <= CLOSE_END else "lift"
            for k in msg.contact:
                names = (k.collision1.name, k.collision2.name)
                other = ("obj" if any(n.startswith("obj::") for n in names)
                         else "table" if any(n.startswith("table::") for n in names) else None)
                if other is None:
                    continue
                s["contact"][phase].add(f"{part}:{other}")
                if other == "obj" and CLOSE_END - WINDOW < it <= CLOSE_END:
                    s["closed"].add(part)
                elif other == "obj" and TRIAL_END - WINDOW < it <= TRIAL_END:
                    s["held"].add(part)
        return cb

    for part in ("hand", "lf", "rf"):       # "hand" is the palm's link
        topic = f"/world/{WORLD}/model/hand/link/{part}/sensor/{part}_contact/contact"
        node.subscribe(Contacts, topic, contact(part))

    def pre_update(info, ecm):
        it = info.iterations + 1          # the step about to run
        s["it"] = it
        if s["joints"] is None:
            model = Model(World(world_entity(ecm)).model_by_name(ecm, "hand"))
            s["joints"] = {n: Joint(model.joint_by_name(ecm, n)) for n in ("lift", "approach", "lf_j", "rf_j")}
        j = s["joints"]
        force = -HOLD_OPEN_N if it <= HOLD_END else GRIP_N
        j["lf_j"].set_force(ecm, [force])
        j["rf_j"].set_force(ecm, [force])
        j["approach"].set_velocity(ecm, [PREGRASP / (APPROACH_END * 0.001) if it <= APPROACH_END else 0.0])
        lifting = CLOSE_END < it <= LIFT_END
        j["lift"].set_velocity(ecm, [LIFT / ((LIFT_END - CLOSE_END) * 0.001) if lifting else 0.0])

    def post_update(info, ecm):
        if info.iterations not in (HOLD_END, CLOSE_END, TRIAL_END):
            return
        world = World(world_entity(ecm))
        hand = Model(world.model_by_name(ecm, "hand"))
        s["hand"][info.iterations] = _vec(Link(hand.link_by_name(ecm, "hand")).world_pose(ecm).pos())
        obj = Link(Model(world.model_by_name(ecm, "obj")).canonical_link(ecm))
        z = float(_vec(obj.world_inertial_pose(ecm).pos())[2])
        if info.iterations == CLOSE_END:
            s["z0"] = z
        elif info.iterations == TRIAL_END:
            s["z1"] = z

    fix.on_pre_update(pre_update)
    fix.on_post_update(post_update)
    fix.finalize()
    server = fix.server()
    server.run(True, TRIAL_END, False)
    closed_on = [p in s["closed"] for p in ("lf", "rf")]
    held = [p in s["held"] for p in ("lf", "rf")]
    rise = s["z1"] - s["z0"]
    h = s["hand"]
    hand_err = float(np.linalg.norm(h[TRIAL_END] - (c + [0.0, 0.0, LIFT])))
    # How far the hand left its commanded path in each phase, and what it touched in that phase.
    dev = {"approach": float(np.linalg.norm(h[HOLD_END] - c)),
           "close": float(np.linalg.norm(h[CLOSE_END] - h[HOLD_END])),
           "lift": float(np.linalg.norm(h[TRIAL_END] - h[CLOSE_END] - [0.0, 0.0, LIFT]))}
    contact = {ph: sorted(v) for ph, v in s["contact"].items()}
    success = bool(rise >= LIFT_FRAC * LIFT and all(held))
    reason = ("ok" if success else "missed" if not any(closed_on)
              else "dropped" if rise < 0.02 else "slipped")
    return {"success": success, "reason": reason, "rise_m": rise, "closed_on": closed_on,
            "held": held, "hand_err_m": hand_err, "hand_dev_m": dev, "hand_contact": contact,
            "off_path": hand_err > OFF_PATH_M, "unexplained_off_path": unexplained_off_path(hand_err, dev, contact)}


def _child(kind, req, partition, timeout) -> dict:
    """One settle or trial in a process of its own; a crash or hang costs only that trial."""
    env = dict(os.environ, GZ_PARTITION=partition)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(req, f)
    try:
        r = subprocess.run([sys.executable, __file__, "--child", kind, "--req", f.name], env=env,
                           capture_output=True, timeout=timeout, text=True)
    except subprocess.TimeoutExpired:
        return {"crash": "timeout"}
    finally:
        os.unlink(f.name)
    for line in reversed(r.stdout.splitlines()):
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    return {"crash": f"exit {r.returncode}: {r.stderr[-200:]}"}


def eval_object(e, timeout) -> dict:
    tag = f"{os.getpid()}_{e['method']}_{e['name']}"
    trials = []
    settled = None
    for attempt in range(3):
        settled = _child("settle", {"sdf": e["sdf"]}, f"lift_{tag}_s{attempt}", timeout)
        if "crash" not in settled:
            break
    for rank in e["ranks"]:
        if "crash" in settled:
            trials.append({"rank": rank, "success": False, "reason": "crashed", "errors": [settled["crash"]]})
            continue
        req = {"sdf": e["sdf"], "p": settled["p"], "q": settled["q"], "grasp": e["grasps"][rank]}
        errors = []
        for attempt in range(3):
            rec = _child("trial", req, f"lift_{tag}_{rank}_{attempt}", timeout)
            if "crash" not in rec:
                break
            errors.append(rec.pop("crash"))
            rec = {"success": False, "reason": "crashed"}
        rec["rank"] = rank
        if errors:
            rec["errors"] = errors
        trials.append(rec)
    succ = [t["success"] for t in trials]
    return {"name": e["name"], "method": e["method"], "has_feasible": bool(trials),
            "lift_first": bool(succ[0]) if succ else False, "lift_any": bool(any(succ)), "trials": trials}


def run_all(args):
    man = json.load(open(args.manifest))
    if args.methods:
        man = [e for e in man if e["method"] in args.methods]
    mj = {r["name"]: r for r in json.load(open(args.mujoco))["results"]}
    entries = [{"name": e["name"], "method": e["method"], "sdf": e["sdf"], "grasps": e.get("grasps", []),
                "ranks": [t["rank"] for t in mj[e["name"]].get("lift", {}).get("trials", [])]} for e in man]
    out = Path(args.out)
    ckpt = out.parent / f"_lift_{out.stem}"
    ckpt.mkdir(parents=True, exist_ok=True)

    def work(e):
        path = ckpt / f"{e['name']}.json"
        if path.exists():
            return json.load(open(path))
        res = eval_object(e, args.trial_timeout)
        path.write_text(json.dumps(res))
        n = sum(t["reason"] == "crashed" for t in res["trials"])
        print(f"[lift] {e['name']} {len(res['trials'])} trials, {n} crashed", flush=True)
        return res

    with ThreadPoolExecutor(args.workers) as ex:
        results = list(ex.map(work, entries))
    json.dump({"results": results, "friction": MU, "grip_n": GRIP_N}, open(out, "w"), indent=1)
    print(f"[lift] merged {len(results)} results -> {out}", flush=True)


CONTROL_BOX = """<?xml version="1.0"?>
<sdf version="1.7"><model name="box"><link name="box_link">
  <inertial><mass>0.125</mass><pose>0 0 0.025 0 0 0</pose><inertia><ixx>5.2e-5</ixx><iyy>5.2e-5</iyy><izz>5.2e-5</izz></inertia></inertial>
  <collision name="c"><pose>0 0 0.025 0 0 0</pose><geometry><box><size>0.05 0.05 0.05</size></box></geometry>
    <surface><friction><ode><mu>1</mu><mu2>1</mu2></ode></friction></surface></collision>
</link></model></sdf>
"""


def control():
    """A central side or top grasp must lift the box; a grasp beside it must close on air."""
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "box.sdf"
        p.write_text(CONTROL_BOX)
        base = "lift_control"
        s = _child("settle", {"sdf": str(p)}, base + "_s", 300)
        print("settled", s)
        grasps = {"side": {"center": [0, 0, 0.025], "approach": [1, 0, 0], "axis": [0, 1, 0], "width": 0.05},
                  "top": {"center": [0, 0, 0.025], "approach": [0, 0, -1], "axis": [0, 1, 0], "width": 0.05},
                  "air": {"center": [0, 0.12, 0.025], "approach": [1, 0, 0], "axis": [0, 1, 0], "width": 0.05},
                  # the palm lands on the box's top face: blocked, palm:obj
                  "palm_blocked": {"center": [0, 0, 0.005], "approach": [0, 0, -1], "axis": [0, 1, 0], "width": 0.06},
                  # the pads reach the table beside the box: blocked, pad:table
                  "pad_on_table": {"center": [0, 0.12, 0.002], "approach": [0, 0, -1], "axis": [0, 1, 0], "width": 0.05}}
        for name, g in grasps.items():
            print(name, _child("trial", {"sdf": str(p), "p": s["p"], "q": s["q"], "grasp": g}, f"{base}_{name}", 300))


def child(args):
    req = json.load(open(args.req))
    with tempfile.TemporaryDirectory() as tmp:
        res = settle(req, tmp) if args.child == "settle" else run_trial(req, tmp)
    print("RESULT " + json.dumps(res), flush=True)
    sys.stdout.flush()
    os._exit(0)     # skip gz teardown: nothing depends on it


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest")
    ap.add_argument("--mujoco")
    ap.add_argument("--out")
    ap.add_argument("--methods", nargs="+")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--trial-timeout", type=float, default=600)
    ap.add_argument("--child", choices=["settle", "trial"])
    ap.add_argument("--req")
    ap.add_argument("--control", action="store_true")
    args = ap.parse_args()
    if args.control:
        control()
    elif args.child:
        child(args)
    else:
        run_all(args)


if __name__ == "__main__":
    main()
