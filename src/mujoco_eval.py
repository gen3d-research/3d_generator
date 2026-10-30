"""MuJoCo cross-simulator evaluation: drop test + physics pick-and-lift.

A second, independent physics engine for the downstream metrics. It reads the
same eval manifest and exported SDFs that the ROS 2 / Gazebo evaluators read,
so every verdict here can be compared per object with the Gazebo one.

* **Drop test** mirrors ``gazebo_stability_eval``: spawn 5 cm above the table
  with a 5 deg tilt about x, settle 2 s of simulation time, stable iff the tilt
  stays under 25 deg and the base drifts under 5 cm from the table top.
* **Pick-and-lift** executes the manifest's ranked antipodal grasps with a
  free-floating parallel-jaw gripper (Panda-like: 8 cm max opening, 2 x 4 cm
  pads, a palm 3 cm behind the grasp centre) on the *settled* object: open,
  approach 10 cm along the grasp approach, close with a constant force, lift
  10 cm, hold. Success iff the object rose >= 7 cm with the hand and both pads
  still touch it. Friction is mu=0.5 on every contact — the value the force-closure
  synthesiser assumes — not the mu=30 that patch_sdf_collision.py writes for
  the Gazebo finger grasps.

Two collision models (``MujocoEvalConfig.collision``):

* ``"sdf"`` — the exported per-primitive set (native box / cylinder / sphere,
  convex-hull pieces for the other types), exactly what Gazebo sees. Concave
  primitives (hollow shell, open tube, torus, handle, gear, L/U/T/I profile)
  are hulled, so a grasp planned across a cup wall cannot be executed.
* ``"coacd"`` — a CoACD convex decomposition of the true visual mesh (cached on
  disk), which keeps cavities; this is the faithful model for grasping.

Pure Python + ``mujoco`` (pip); no ROS. Used by scripts/run_mujoco_eval.py.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

try:
    import mujoco
except ImportError as e:   # pragma: no cover - optional dependency
    raise ImportError("mujoco_eval needs `pip install mujoco`") from e


@dataclass
class MujocoEvalConfig:
    # drop test (gazebo_stability_eval parity)
    drop_height_m: float = 0.05
    spawn_tilt_deg: float = 5.0
    settle_time_s: float = 2.0
    upright_tolerance_deg: float = 25.0
    drift_tolerance_m: float = 0.05
    # physics
    timestep: float = 0.001
    mu: float = 0.5
    # collision model: "sdf" = the exported per-primitive set (what Gazebo
    # sees; concave primitives hulled), "coacd" = convex decomposition of the
    # true visual mesh (keeps cavities, so fingers can enter a cup or a ring)
    collision: str = "sdf"
    coacd_threshold: float = 0.05
    # pick-and-lift
    grasps_per_object: int = 3
    max_opening_m: float = 0.08        # Panda: 2 x 4 cm finger travel
    grip_force_n: float = 20.0         # per finger, constant while closed
    pregrasp_offset_m: float = 0.10
    lift_height_m: float = 0.10
    lift_success_frac: float = 0.7     # object must rise >= 70% of the lift
    pregrasp_penetration_m: float = 0.002
    preshape_clearance_m: float = 0.015  # per finger, beyond half the grasp width


# ---------------------------------------------------------------------------
# SDF -> MJCF
# ---------------------------------------------------------------------------

def _floats(text: str | None, n: int | None = None) -> list[float]:
    vals = [float(v) for v in (text or "").split()]
    if n is not None and len(vals) < n:
        vals += [0.0] * (n - len(vals))
    return vals


def _rpy_to_quat_wxyz(r: float, p: float, y: float) -> np.ndarray:
    """SDF fixed-axis roll-pitch-yaw (R = Rz Ry Rx) -> MuJoCo (w, x, y, z)."""
    cr, sr = math.cos(r / 2), math.sin(r / 2)
    cp, sp = math.cos(p / 2), math.sin(p / 2)
    cy, sy = math.cos(y / 2), math.sin(y / 2)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


def parse_sdf(sdf_path: str | Path) -> dict:
    """Read mass properties and per-primitive collision geometry from an
    exported object SDF."""
    sdf_path = Path(sdf_path)
    link = ET.parse(sdf_path).getroot().find(".//link")
    inert = link.find("inertial")
    ipose = _floats(inert.findtext("pose"), 6)
    I = inert.find("inertia")
    g = lambda k: float(I.findtext(k, "0"))           # noqa: E731
    I_in = np.array([[g("ixx"), g("ixy"), g("ixz")],
                     [g("ixy"), g("iyy"), g("iyz")],
                     [g("ixz"), g("iyz"), g("izz")]])
    # fullinertia carries no frame rotation, so express the tensor in the link
    # frame (SDF inertial poses are normally unrotated; this is a no-op then).
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, _rpy_to_quat_wxyz(*ipose[3:6]))
    R = R.reshape(3, 3)
    I_b = R @ I_in @ R.T
    out = {
        "mass": float(inert.findtext("mass")),
        "com": ipose[:3],
        # MuJoCo fullinertia order: ixx iyy izz ixy ixz iyz
        "inertia": [I_b[0, 0], I_b[1, 1], I_b[2, 2], I_b[0, 1], I_b[0, 2], I_b[1, 2]],
        "geoms": [],
    }
    for col in link.findall("collision"):
        pose = _floats(col.findtext("pose"), 6)
        geo = col.find("geometry")
        entry = {"pos": pose[:3], "quat": _rpy_to_quat_wxyz(*pose[3:6]).tolist()}
        if geo.find("box") is not None:
            entry.update(kind="box",
                         size=[s / 2 for s in _floats(geo.findtext("box/size"), 3)])
        elif geo.find("cylinder") is not None:
            entry.update(kind="cylinder",
                         size=[float(geo.findtext("cylinder/radius")),
                               float(geo.findtext("cylinder/length")) / 2])
        elif geo.find("sphere") is not None:
            entry.update(kind="sphere", size=[float(geo.findtext("sphere/radius"))])
        elif geo.find("mesh") is not None:
            uri = geo.findtext("mesh/uri").replace("model://", "")
            entry.update(kind="mesh", file=str((sdf_path.parent / uri).resolve()))
        else:
            continue
        out["geoms"].append(entry)
    return out


def coacd_geoms(visual_mesh: str | Path, threshold: float = 0.05) -> list[dict]:
    """Convex decomposition of the true (visual) mesh, cached on disk next to
    it as ``coacd_<threshold>/part_<i>.obj``. Raises if CoACD fails."""
    from convex_decomp import decompose_file
    visual_mesh = Path(visual_mesh)
    parts = decompose_file(visual_mesh, visual_mesh.parent / f"coacd_{threshold:g}",
                           threshold)
    return [{"kind": "mesh", "file": str(p.resolve()), "pos": [0.0, 0.0, 0.0],
             "quat": [1.0, 0.0, 0.0, 0.0]} for p in parts]


def _v(a) -> str:
    return " ".join(f"{float(x):.9g}" for x in a)


_GRIPPER = """
    <body name="target" mocap="true" pos="0 0 0.6"/>
    <body name="hand" pos="0 0 0.6" gravcomp="1">
      <freejoint name="hand_free"/>
      <inertial pos="0 0 -0.04" mass="0.6" diaginertia="1e-3 1e-3 1e-3"/>
      <geom name="palm" class="grip" type="box" size="0.012 {palm_half} 0.01" pos="0 0 -0.04"/>
      <body name="lf" gravcomp="1">
        <joint name="lf_j" type="slide" axis="0 1 0" range="0 {half_open}" damping="5" armature="0.01"/>
        <inertial pos="0 0.005 -0.01" mass="0.05" diaginertia="1e-5 1e-5 1e-5"/>
        <geom name="lf_pad" class="grip" type="box" size="0.01 0.005 0.02" pos="0 0.005 -0.01"/>
      </body>
      <body name="rf" gravcomp="1">
        <joint name="rf_j" type="slide" axis="0 -1 0" range="0 {half_open}" damping="5" armature="0.01"/>
        <inertial pos="0 -0.005 -0.01" mass="0.05" diaginertia="1e-5 1e-5 1e-5"/>
        <geom name="rf_pad" class="grip" type="box" size="0.01 0.005 0.02" pos="0 -0.005 -0.01"/>
      </body>
    </body>
"""

_GRIPPER_TAIL = """
  <equality>
    <weld body1="target" body2="hand" solref="0.004 1"/>
  </equality>
  <contact><exclude body1="lf" body2="rf"/></contact>
  <actuator>
    <motor name="lf_m" joint="lf_j" ctrllimited="true" ctrlrange="-100 100"/>
    <motor name="rf_m" joint="rf_j" ctrllimited="true" ctrlrange="-100 100"/>
  </actuator>
"""


def build_mjcf(obj: dict, cfg: MujocoEvalConfig, gripper: bool) -> str:
    """One object on a table plane (z=0), optionally with the gripper."""
    assets, geoms = [], []
    for i, g in enumerate(obj["geoms"]):
        if g["kind"] == "mesh":
            assets.append(f'<mesh name="m{i}" file="{g["file"]}"/>')
            geoms.append(f'<geom class="obj" type="mesh" mesh="m{i}" '
                         f'pos="{_v(g["pos"])}" quat="{_v(g["quat"])}"/>')
        else:
            geoms.append(f'<geom class="obj" type="{g["kind"]}" size="{_v(g["size"])}" '
                         f'pos="{_v(g["pos"])}" quat="{_v(g["quat"])}"/>')
    mu = cfg.mu
    return f"""
<mujoco model="gen3d_eval">
  <compiler angle="radian" balanceinertia="true" boundmass="1e-4" boundinertia="1e-9"/>
  <option timestep="{cfg.timestep}" cone="elliptic" impratio="10" integrator="implicitfast"/>
  <visual><headlight ambient="0.35 0.35 0.35" diffuse="0.65 0.65 0.65"/></visual>
  <default>
    <!-- Object and table keep MuJoCo's default contact (no rebound: a stiff
         contact bounces a 5 cm drop ~2 cm and topples stable objects). The
         pads alone are stiff, via priority: the default soft constraint lets
         a 20 N grip drive pads centimetres into a 50 g object. -->
    <geom friction="{mu} 0.005 0.0001" condim="4"/>
    <joint solreflimit="{2 * cfg.timestep} 1"/>
    <default class="obj"><geom condim="4" density="0" rgba="0.33 0.55 0.85 1"/></default>
    <default class="grip"><geom condim="4" density="0" rgba="0.95 0.55 0.2 1"
                                priority="1" solref="{2 * cfg.timestep} 1"/></default>
  </default>
  <asset>{''.join(assets)}</asset>
  <worldbody>
    <geom name="table" type="plane" size="1 1 0.01" rgba="0.8 0.8 0.78 1"/>
    <body name="obj" pos="0 0 0">
      <freejoint name="obj_free"/>
      <inertial pos="{_v(obj['com'])}" mass="{obj['mass']:.9g}"
                fullinertia="{_v(obj['inertia'])}"/>
      {''.join(geoms)}
    </body>
    {_GRIPPER.format(half_open=cfg.max_opening_m / 2,
                     palm_half=cfg.max_opening_m / 2 + 0.005) if gripper else ''}
  </worldbody>
  {_GRIPPER_TAIL if gripper else ''}
</mujoco>
"""


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _tilt_deg(quat_wxyz: np.ndarray) -> float:
    w, x, y, z = quat_wxyz
    r22 = 1 - 2 * (x * x + y * y)
    return math.degrees(math.acos(max(-1.0, min(1.0, r22))))


def _mat_to_quat(R: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R).reshape(-1))
    return q


def _finite(d) -> bool:
    return bool(np.all(np.isfinite(d.qpos)) and np.all(np.isfinite(d.qvel)))


class _Sim:
    def __init__(self, obj: dict, cfg: MujocoEvalConfig, gripper: bool):
        self.cfg = cfg
        self.m = mujoco.MjModel.from_xml_string(build_mjcf(obj, cfg, gripper))
        self.d = mujoco.MjData(self.m)
        self.obj_b = self.m.body("obj").id
        oj = self.m.joint("obj_free")
        self.oq, self.ov = oj.qposadr[0], oj.dofadr[0]
        if gripper:
            self.hand_b = self.m.body("hand").id
            hj = self.m.joint("hand_free")
            self.hq, self.hv = hj.qposadr[0], hj.dofadr[0]
            self.fq = [self.m.joint("lf_j").qposadr[0], self.m.joint("rf_j").qposadr[0]]
            self.mocap = self.m.body_mocapid[self.m.body("target").id]
            self.grip_geoms = {self.m.geom(n).id for n in ("palm", "lf_pad", "rf_pad")}
            self.pads = (self.m.geom("lf_pad").id, self.m.geom("rf_pad").id)

    def step(self, seconds: float, ctrl=None, mocap_path=None):
        n = max(1, int(round(seconds / self.cfg.timestep)))
        for k in range(n):
            if ctrl is not None:
                self.d.ctrl[:] = ctrl() if callable(ctrl) else ctrl
            if mocap_path is not None:
                self.d.mocap_pos[self.mocap] = mocap_path((k + 1) / n)
            mujoco.mj_step(self.m, self.d)
        return _finite(self.d)

    def place_obj(self, z: float, quat_wxyz) -> None:
        self.d.qpos[self.oq:self.oq + 3] = [0.0, 0.0, z]
        self.d.qpos[self.oq + 3:self.oq + 7] = quat_wxyz
        self.d.qvel[self.ov:self.ov + 6] = 0.0

    def place_hand(self, pos, quat_wxyz, opening_q: float) -> None:
        self.d.qpos[self.hq:self.hq + 3] = pos
        self.d.qpos[self.hq + 3:self.hq + 7] = quat_wxyz
        self.d.qvel[self.hv:self.hv + 6] = 0.0
        for a in self.fq:
            self.d.qpos[a] = opening_q
        self.d.mocap_pos[self.mocap] = pos
        self.d.mocap_quat[self.mocap] = quat_wxyz

    def pads_touching(self) -> tuple[bool, bool]:
        """Whether each finger pad is in contact with the object."""
        touch = [False, False]
        for c in self.d.contact[:self.d.ncon]:
            for i, pad in enumerate(self.pads):
                other = c.geom2 if c.geom1 == pad else c.geom1 if c.geom2 == pad else -1
                if other >= 0 and self.m.geom_bodyid[other] == self.obj_b:
                    touch[i] = True
        return touch[0], touch[1]

    def hand_penetration(self) -> float:
        """Deepest penetration of any gripper geom (table or object), metres."""
        worst = 0.0
        for c in self.d.contact[:self.d.ncon]:
            if c.geom1 in self.grip_geoms or c.geom2 in self.grip_geoms:
                worst = max(worst, -float(c.dist))
        return worst


# ---------------------------------------------------------------------------
# drop test
# ---------------------------------------------------------------------------

def drop_test(obj: dict, cfg: MujocoEvalConfig) -> dict:
    sim = _Sim(obj, cfg, gripper=False)
    half = math.radians(cfg.spawn_tilt_deg) / 2
    sim.place_obj(cfg.drop_height_m, [math.cos(half), math.sin(half), 0.0, 0.0])
    mujoco.mj_forward(sim.m, sim.d)
    if not sim.step(cfg.settle_time_s):
        return {"stable": False, "reason": "diverged"}
    pos = sim.d.xpos[sim.obj_b]
    tilt = _tilt_deg(sim.d.xquat[sim.obj_b])
    drift = abs(float(pos[2]))
    return {"stable": bool(tilt < cfg.upright_tolerance_deg
                           and drift < cfg.drift_tolerance_m),
            "tilt_deg": tilt, "drift_m": drift}


# ---------------------------------------------------------------------------
# pick-and-lift
# ---------------------------------------------------------------------------

def _hand_frame(approach, axis) -> np.ndarray | None:
    z = np.asarray(approach, float)
    z = z / (np.linalg.norm(z) + 1e-12)
    y = np.asarray(axis, float)
    y = y - np.dot(y, z) * z
    ny = np.linalg.norm(y)
    if ny < 1e-6:
        return None
    y /= ny
    return np.column_stack([np.cross(y, z), y, z])


def pick_and_lift(obj: dict, grasps: list[dict], cfg: MujocoEvalConfig,
                  on_phase=None) -> dict:
    """Execute ranked grasps on the settled object, as a robot would.

    Grasps are walked in the planner's rank order. An *infeasible* grasp (wider
    than the gripper, degenerate frame, approaching from below the table, or a
    hand pose that already intersects the table/object 10 cm before contact)
    is skipped, not executed; up to ``cfg.grasps_per_object`` feasible grasps
    are executed. Reported: ``has_feasible``, ``lift_first`` (the first
    feasible grasp holds) and ``lift_any`` (any executed grasp holds).

    ``on_phase(rank, phase, sim)``, if given, is called after each phase
    (pregrasp / approach / close / lift) — used for rendering figures."""
    note = on_phase or (lambda *a: None)
    sim = _Sim(obj, cfg, gripper=True)
    m, d = sim.m, sim.d
    # Settle upright with the hand parked out of reach, then snapshot.
    sim.place_obj(0.002, [1.0, 0.0, 0.0, 0.0])
    sim.place_hand([0.0, 0.0, 0.6], [1.0, 0.0, 0.0, 0.0], cfg.max_opening_m / 2)
    mujoco.mj_forward(m, d)
    if not sim.step(1.0, ctrl=[5.0, 5.0]):
        return {"has_feasible": False, "lift_first": False, "lift_any": False,
                "trials": [], "reason": "settle_diverged"}
    snap_q, snap_v = d.qpos.copy(), d.qvel.copy()
    settled_t = d.time
    p_o = d.xpos[sim.obj_b].copy()
    R_o = d.xmat[sim.obj_b].reshape(3, 3).copy()

    trials, skipped, infeasible = [], {}, []

    def skip(rec, why):            # infeasible: not executed, not a trial
        trials.remove(rec)
        skipped[why] = skipped.get(why, 0) + 1
        infeasible.append({"rank": rec["rank"], "reason": why})

    for gi, g in enumerate(grasps):
        if len(trials) >= cfg.grasps_per_object:
            break
        rec = {"rank": gi, "score": g.get("score"), "width": g.get("width")}
        trials.append(rec)
        if float(g.get("width", 0.0)) > cfg.max_opening_m - 0.004:
            skip(rec, "too_wide")
            continue
        Rh = _hand_frame(R_o @ np.asarray(g["approach"]), R_o @ np.asarray(g["axis"]))
        if Rh is None:
            skip(rec, "degenerate_frame")
            continue
        c = p_o + R_o @ np.asarray(g["center"], float)
        a = Rh[:, 2]
        if a[2] > 0.7:
            skip(rec, "from_below")
            continue
        pre = c - cfg.pregrasp_offset_m * a
        qh = _mat_to_quat(Rh)

        d.qpos[:], d.qvel[:] = snap_q, snap_v
        d.time = settled_t
        d.act[:] = 0.0
        # Pre-shape to the grasp width (a real gripper need not open fully).
        open_q = min(cfg.max_opening_m / 2,
                     float(g.get("width", 0.0)) / 2 + cfg.preshape_clearance_m)
        sim.place_hand(pre, qh, open_q)
        mujoco.mj_forward(m, d)
        note(gi, "pregrasp", sim)
        pen = sim.hand_penetration()
        if pen > cfg.pregrasp_penetration_m:
            skip(rec, "pregrasp_collision")
            continue

        hold = lambda: [200.0 * (open_q - d.qpos[a_]) for a_ in sim.fq]   # noqa: E731
        ok = sim.step(0.6, ctrl=hold, mocap_path=lambda s: pre + s * (c - pre))
        ok = ok and sim.step(0.1, ctrl=hold)
        note(gi, "approach", sim)
        f = -cfg.grip_force_n
        ok = ok and sim.step(0.6, ctrl=[f, f])
        closed_on = sim.pads_touching()
        note(gi, "close", sim)
        z0 = float(d.xipos[sim.obj_b][2])
        start = d.mocap_pos[sim.mocap].copy()
        lift = np.array([0.0, 0.0, cfg.lift_height_m])
        ok = ok and sim.step(1.0, ctrl=[f, f], mocap_path=lambda s: start + s * lift)
        ok = ok and sim.step(0.5, ctrl=[f, f])
        note(gi, "lift", sim)
        if not ok:
            rec.update(success=False, reason="diverged")
            continue
        rise = float(d.xipos[sim.obj_b][2]) - z0
        dist = float(np.linalg.norm(d.xipos[sim.obj_b] - d.xpos[sim.hand_b]))
        opening = float(sum(d.qpos[a_] for a_ in sim.fq))
        held = sim.pads_touching()
        # Held = rose with the hand and still pinched between BOTH pads (a COM
        # radius would fail long objects lifted by one end).
        success = rise >= cfg.lift_success_frac * cfg.lift_height_m and all(held)
        if success:
            reason = "ok"
        elif not any(closed_on):
            reason = "missed"          # closed on air (or object pushed away)
        elif rise < 0.02:
            reason = "dropped"         # pinched, but friction did not lift it
        else:
            reason = "slipped"         # lifted partway, then lost
        rec.update(success=bool(success), rise_m=rise, com_to_tcp_m=dist,
                   opening_m=opening, closed_on=list(closed_on), reason=reason)
    succ = [t.get("success", False) for t in trials]
    return {"has_feasible": bool(trials), "lift_first": bool(succ[0]) if succ else False,
            "lift_any": bool(any(succ)), "n_executed": len(trials),
            "n_planned": len(grasps), "skipped": skipped, "infeasible": infeasible,
            "trials": trials}


def evaluate_entry(entry: dict, cfg: MujocoEvalConfig | None = None,
                   lift: bool = True) -> dict:
    """Drop test + (optionally) pick-and-lift for one manifest entry."""
    cfg = cfg or MujocoEvalConfig()
    out = {"name": entry["name"], "method": entry.get("method")}
    try:
        obj = parse_sdf(entry["sdf"])
        out["collision"] = cfg.collision
        if cfg.collision == "coacd":
            try:
                obj["geoms"] = coacd_geoms(entry["visual_mesh"], cfg.coacd_threshold)
            except Exception as e:     # noqa: BLE001 - keep the SDF geometry
                out["collision"] = "sdf_fallback"
                out["coacd_error"] = str(e)
        elif cfg.collision != "sdf":
            raise ValueError(f"unknown collision model {cfg.collision!r}")
        out["drop"] = drop_test(obj, cfg)
        if lift:
            out["lift"] = pick_and_lift(obj, entry.get("grasps", []), cfg)
    except Exception as e:   # one bad asset must not kill a batch
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def config_dict(cfg: MujocoEvalConfig) -> dict:
    return asdict(cfg)

