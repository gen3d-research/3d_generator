"""MuJoCo cross-simulator evaluator: drop-test and pick-and-lift invariants on
synthetic SDFs (no manifest, no ROS). Skipped when mujoco is not installed."""
import pytest

pytest.importorskip("mujoco")

import mujoco_eval as me  # noqa: E402


def _box_sdf(tmp_path, name, size, mass):
    sx, sy, sz = size
    ixx = mass * (sy ** 2 + sz ** 2) / 12
    iyy = mass * (sx ** 2 + sz ** 2) / 12
    izz = mass * (sx ** 2 + sy ** 2) / 12
    p = tmp_path / f"{name}.sdf"
    p.write_text(f"""<?xml version="1.0" ?>
<sdf version="1.7"><model name="{name}"><link name="{name}_link">
  <inertial><mass>{mass}</mass><pose>0 0 {sz / 2} 0 0 0</pose>
    <inertia><ixx>{ixx}</ixx><iyy>{iyy}</iyy><izz>{izz}</izz>
      <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>
  <collision name="c0"><pose>0 0 {sz / 2} 0 0 0</pose>
    <geometry><box><size>{sx} {sy} {sz}</size></box></geometry></collision>
</link></model></sdf>""")
    return p


def test_parse_sdf_reads_native_box(tmp_path):
    obj = me.parse_sdf(_box_sdf(tmp_path, "cube", (0.04, 0.04, 0.04), 0.1))
    assert obj["mass"] == pytest.approx(0.1)
    assert obj["com"] == pytest.approx([0, 0, 0.02])
    (g,) = obj["geoms"]
    assert g["kind"] == "box" and g["size"] == pytest.approx([0.02, 0.02, 0.02])


def test_cube_is_drop_stable_and_liftable(tmp_path):
    entry = {"name": "cube", "method": "t",
             "sdf": str(_box_sdf(tmp_path, "cube", (0.04, 0.04, 0.04), 0.1)),
             # top-down pinch across x through the cube's centre
             "grasps": [{"center": [0, 0, 0.02], "approach": [0, 0, -1],
                         "axis": [1, 0, 0], "width": 0.04, "score": 1.0}]}
    r = me.evaluate_entry(entry)
    assert "error" not in r
    assert r["drop"]["stable"]
    assert r["lift"]["has_feasible"] and r["lift"]["lift_first"]
    assert r["lift"]["trials"][0]["rise_m"] > 0.07


def test_thin_rod_tips_under_spawn_tilt(tmp_path):
    # tipping angle atan(0.005 / 0.07) = 4.1 deg < the 5 deg spawn tilt
    entry = {"name": "rod", "method": "t",
             "sdf": str(_box_sdf(tmp_path, "rod", (0.01, 0.01, 0.14), 0.05)),
             "grasps": []}
    r = me.evaluate_entry(entry, lift=False)
    assert not r["drop"]["stable"]
    assert r["drop"]["tilt_deg"] > 25


def test_low_side_grasp_with_vertical_fingers_is_skipped(tmp_path):
    # Side approach whose finger axis is vertical: the palm reaches into the
    # table, so the grasp is infeasible and must be skipped, not executed.
    entry = {"name": "slab", "method": "t",
             "sdf": str(_box_sdf(tmp_path, "slab", (0.06, 0.06, 0.02), 0.1)),
             "grasps": [{"center": [0, 0, 0.01], "approach": [1, 0, 0],
                         "axis": [0, 0, 1], "width": 0.02, "score": 1.0}]}
    r = me.evaluate_entry(entry)
    assert not r["lift"]["has_feasible"]
    assert r["lift"]["skipped"] == {"pregrasp_collision": 1}
