"""
Gazebo dynamic-stability evaluator.

For each generated object in the manifest, the node spawns it onto the table
in the running ``panda_eval_world``, lets physics settle for the configured
period, reads the resulting pose, and despawns it. An object is counted
stable iff its vertical drift and tilt remain below the configured tolerances.

Interaction with Gazebo happens through ``gz service`` and ``gz topic``
subprocess calls so we do not depend on the gz.transport Python bindings,
which are not always packaged consistently across Jazzy releases.

Usage::

    # Terminal 1
    ros2 launch generated_objects_eval stability_world.launch.py

    # Terminal 2
    ros2 run generated_objects_eval gazebo_stability_eval \\
        --manifest /abs/path/manifest.json \\
        --out /abs/path/gazebo_stability.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

WORLD_NAME = "panda_eval_world"


# ---------------------------------------------------------------------------

def _run(cmd: list[str], timeout: float = 5.0) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, timeout=timeout)


def spawn_sdf(sdf_path: Path, name: str, x: float, y: float, z: float) -> bool:
    req = (
        f'sdf_filename: "{sdf_path}", '
        f'name: "{name}", '
        f'pose: {{position: {{x: {x}, y: {y}, z: {z}}}, '
        f'orientation: {{w: 1.0}}}}'
    )
    # Generous service timeout: under parallel gz worlds the create response can
    # take well over the old 3 s, which silently turned into spawn_ok=False. The gz
    # --timeout (ms) must stay BELOW the subprocess timeout (s) so gz returns its own
    # failure first; either way a transport glitch is caught, not crash-propagated.
    gz_ms = os.environ.get("GZ_SVC_TIMEOUT_MS", "15000")
    sub_s = float(os.environ.get("GZ_SVC_SUBPROC_S", "22"))
    cmd = [
        "gz", "service", "-s", f"/world/{WORLD_NAME}/create",
        "--reqtype", "gz.msgs.EntityFactory",
        "--reptype", "gz.msgs.Boolean",
        "--timeout", gz_ms,
        "--req", req,
    ]
    try:
        proc = _run(cmd, timeout=sub_s)
    except subprocess.SubprocessError:
        return False
    return proc.returncode == 0 and b"data: true" in proc.stdout


def despawn(name: str):
    gz_ms = os.environ.get("GZ_SVC_TIMEOUT_MS", "15000")
    sub_s = float(os.environ.get("GZ_SVC_SUBPROC_S", "22"))
    req = f'name: "{name}", type: MODEL'
    try:
        _run([
            "gz", "service", "-s", f"/world/{WORLD_NAME}/remove",
            "--reqtype", "gz.msgs.Entity", "--reptype", "gz.msgs.Boolean",
            "--timeout", gz_ms, "--req", req,
        ], timeout=sub_s)
    except subprocess.SubprocessError:
        pass   # a stale model left in the world is fine — query_pose filters by name


# Parse one snapshot of /world/<world>/pose/info text output into a name->pose dict.
_POSE_RE = re.compile(
    r'pose\s*\{[^{}]*?name:\s*"([^"]+)"[^{}]*?'
    r'position\s*\{([^{}]*?)\}[^{}]*?'
    r'orientation\s*\{([^{}]*?)\}',
    re.DOTALL,
)
_VEC_RE = re.compile(r'([xyzw])\s*:\s*(-?\d+\.?\d*(?:e[+\-]?\d+)?)')


def _parse_vec(text: str) -> dict:
    return {k: float(v) for k, v in _VEC_RE.findall(text)}


def query_pose(name: str, max_messages: int = 4, timeout: float = 12.0
               ) -> Optional[dict]:
    """Read /world/<world>/pose/info messages until *name* (or its main link)
    appears, then return its decoded pose.

    gz_sim publishes the pose for both the model and its links; the model
    itself carries the world-space pose, the link carries an identity pose
    relative to the model.  We prefer the model pose."""
    proc = subprocess.Popen(
        ["gz", "topic", "-e", "-t", f"/world/{WORLD_NAME}/pose/info",
         "-n", str(max_messages)],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    try:
        out, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        return None
    text = out.decode(errors="ignore")
    # The model pose carries the world-space transform; link/visual entries
    # repeat zeroed-out poses relative to the model and would otherwise win
    # the "last match" race.  Match the model name exclusively.
    last_match = None
    for m in _POSE_RE.finditer(text):
        if m.group(1) == name:
            last_match = m
    if last_match is None:
        return None
    pos = _parse_vec(last_match.group(2))
    ori = _parse_vec(last_match.group(3))
    return {
        "position": [pos.get("x", 0.0), pos.get("y", 0.0), pos.get("z", 0.0)],
        "orientation": [ori.get("x", 0.0), ori.get("y", 0.0),
                        ori.get("z", 0.0), ori.get("w", 1.0)],
    }


def tilt_angle_deg(quat_xyzw: np.ndarray) -> float:
    x, y, z, w = quat_xyzw
    r22 = 1 - 2 * (x * x + y * y)
    return math.degrees(math.acos(max(-1.0, min(1.0, r22))))


# ---------------------------------------------------------------------------

def evaluate_one(entry: dict, cfg: dict) -> dict:
    spawn = cfg["object_spawn"]
    stab = cfg["stability"]
    name = entry["name"]
    sdf_path = Path(entry["sdf"]).resolve()
    # Drop height above the table (GZ_DROP_HEIGHT_M, default 5 cm). Raise it for a more
    # dramatic fall — the drift metric measures from the table top (the expected rest
    # height), so a clean settle reads ~0 drift regardless of how high it was dropped.
    spawn_z = spawn["z"] + float(os.environ.get("GZ_DROP_HEIGHT_M", "0.05"))

    ok = spawn_sdf(sdf_path, name, spawn["x"], spawn["y"], spawn_z)
    if not ok:
        return {"name": name, "method": entry["method"],
                "spawn_ok": False, "stable": False}

    time.sleep(stab["settle_time_s"])
    pose = query_pose(name)
    despawn(name)
    if pose is None:
        return {"name": name, "method": entry["method"],
                "spawn_ok": True, "stable": False, "reason": "no_pose"}
    # Drift = deviation from the EXPECTED resting height (the table top), not from
    # the elevated spawn point. Objects are base-seated (model origin at z=0), so a
    # cleanly-settled object rests with its origin at the table top (spawn["z"]).
    # Measuring from spawn_z would count the intentional settle-drop as "drift" and
    # trip the tolerance for perfectly-stable, upright objects.
    drift = abs(pose["position"][2] - spawn["z"])
    tilt = tilt_angle_deg(np.asarray(pose["orientation"]))
    stable = (drift < stab["drift_tolerance_m"]) and \
             (tilt < stab["upright_tolerance_deg"])
    return {"name": name, "method": entry["method"], "spawn_ok": True,
            "stable": stable, "tilt_deg": tilt, "drift_m": drift,
            "final_pose": pose}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--max-objects", type=int, default=0)
    args = parser.parse_args()

    import yaml
    if args.config is None:
        from ament_index_python.packages import get_package_share_directory
        share = Path(get_package_share_directory("generated_objects_eval"))
        args.config = share / "config" / "eval_config.yaml"
    cfg = yaml.safe_load(args.config.read_text())

    with args.manifest.open() as f:
        manifest = json.load(f)
    if args.max_objects and args.max_objects > 0:
        manifest = manifest[:args.max_objects]

    results = []
    consec_fail = 0
    abort_after = int(os.environ.get("GZ_ABORT_AFTER_FAILS", "6"))
    for k, entry in enumerate(manifest):
        print(f"[{k+1}/{len(manifest)}] {entry['name']} ({entry['method']})",
              flush=True)
        try:
            r = evaluate_one(entry, cfg)
        except Exception as e:   # never let one object kill the whole worker
            r = {"name": entry.get("name"), "method": entry.get("method"),
                 "spawn_ok": False, "stable": False, "reason": f"error:{e}"}
        results.append(r)
        # If the gz world dies (physics crash), every subsequent spawn fails. Bail out
        # fast instead of slow-failing the tail, so the caller can restart the world.
        consec_fail = 0 if r.get("spawn_ok") else consec_fail + 1
        if consec_fail >= abort_after:
            print(f"[abort] {consec_fail} consecutive spawn failures — world likely "
                  f"dead; stopping at {k+1}/{len(manifest)}", flush=True)
            break

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"results": results, "config": cfg}, indent=2))

    summary: dict = {}
    for r in results:
        m = r["method"]
        s = summary.setdefault(m, {"n": 0, "spawn": 0, "stable": 0})
        s["n"] += 1
        s["spawn"] += int(r.get("spawn_ok", False))
        s["stable"] += int(r.get("stable", False))
    print("\n=== Gazebo dynamic-stability summary ===")
    for m, s in summary.items():
        rate = s["stable"] / max(1, s["spawn"])
        print(f"  {m:14s} n={s['n']:3d} spawn_ok={s['spawn']:3d} "
              f"stable={s['stable']:3d}  rate={rate:.1%}")


if __name__ == "__main__":
    main()
