"""Regression tests locking the v2 invariants.

Run from 3d_generator/:  python -m pytest -q
(conftest.py puts src/ on sys.path.)
"""
import json

import numpy as np
import pytest

import primitives as P
from primitives import (
    Box, Cylinder, Sphere, Capsule, Cone, Pyramid, Torus, Ellipsoid, Wedge,
    HollowShell, Handle, Frustum, Hemisphere, HexPrism, OpenTube, NGonPrism,
    Transform, seat_height,
)
from archetypes import ARCHETYPE_REGISTRY
from scoring import ObjectScorer, ScoringConfig
import cem
from cem import ParameterDistribution, PRIMITIVE_SPECS
import generator


# --- new primitive types ---------------------------------------------------

PRIMS = {
    "box": Box(dimensions=np.array([0.05, 0.04, 0.06])),
    "cylinder": Cylinder(radius=0.025, height=0.06),
    "sphere": Sphere(radius=0.03),
    "capsule": Capsule(radius=0.015, height=0.04),
    "cone": Cone(radius=0.03, height=0.06),
    "pyramid": Pyramid(radius=0.03, height=0.05),
    "torus": Torus(major_radius=0.04, minor_radius=0.012),
    "ellipsoid": Ellipsoid(radii=np.array([0.04, 0.03, 0.02])),
    "wedge": Wedge(width=0.05, depth=0.04, height=0.04),
    "hollow_shell": HollowShell(outer_radius=0.035, wall_thickness=0.004,
                                height=0.07, floor_thickness=0.005),
    "handle": Handle(major_radius=0.02, tube_a=0.006, tube_b=0.005,
                     arc_angle=1.5 * np.pi),
    "frustum": Frustum(radius_bottom=0.04, radius_top=0.025, height=0.06),
    "hemisphere": Hemisphere(radius=0.03),
    "hex_prism": HexPrism(radius=0.018, height=0.012),
    "open_tube": OpenTube(outer_radius=0.02, wall_thickness=0.005, height=0.05),
    "ngon_prism": NGonPrism(n_sides=5, radius=0.02, height=0.03),
}


@pytest.mark.parametrize("name", list(PRIMS))
def test_primitive_mesh_and_inertia(name):
    prim = PRIMS[name]
    mesh = prim.to_mesh()
    assert mesh.is_watertight, f"{name} mesh not watertight"
    assert prim.volume() > 0, f"{name} non-positive volume"
    I = prim.inertia_tensor(1000.0)
    assert I.shape == (3, 3) and np.all(np.isfinite(I)), f"{name} bad inertia"
    assert np.allclose(I, I.T, atol=1e-9), f"{name} inertia not symmetric"


@pytest.mark.parametrize("name", list(PRIMS))
def test_primitive_seat_on_ground(name):
    prim = PRIMS[name]
    prim.transform.translation[2] = seat_height(prim)
    low = prim.to_mesh().bounds[0][2]
    assert abs(low) < 1e-6, f"{name} not seated on z=0 (low={low})"


def test_all_primitive_types_have_specs():
    spec_types = {s.ptype for s in PRIMITIVE_SPECS}
    assert spec_types == set(P.PrimitiveType), "PRIMITIVE_SPECS must cover every PrimitiveType"


# --- archetype registry -----------------------------------------------------

def test_registry_size():
    assert len(ARCHETYPE_REGISTRY) >= 80


@pytest.mark.parametrize("name", sorted(ARCHETYPE_REGISTRY))
def test_archetype_builds_connected_and_scores(name):
    obj = ARCHETYPE_REGISTRY[name]()
    assert obj.primitives, f"{name} produced no primitives"
    assert obj.is_connected(), f"{name} is not a single connected body"
    s = ObjectScorer().score(obj).total_score
    assert s > 0.0, f"{name} scored {s}"


# --- CEM distribution -------------------------------------------------------

def test_cem_configurable_count():
    d = ParameterDistribution(max_primitives=12)
    assert len(d.n_primitives_probs) == 12
    assert len(d.primitive_type_probs) == len(PRIMITIVE_SPECS)
    rng = np.random.default_rng(0)
    counts = [len(d.sample_object(rng).primitives) for _ in range(200)]
    assert max(counts) > 4, "should sample beyond the old 4-cap"
    assert min(counts) >= 1


def test_cem_roundtrip():
    d = ParameterDistribution(max_primitives=10)
    d.offset_std = 0.0123
    d2 = ParameterDistribution.from_dict(json.loads(json.dumps(d.to_dict())))
    assert d2.max_primitives == 10
    assert set(d2.type_log_means) == set(d.type_log_means)
    assert abs(d2.offset_std - 0.0123) < 1e-9
    np.testing.assert_allclose(d2.n_primitives_probs, d.n_primitives_probs)


def test_cem_legacy_dict_loads():
    # v1-style dict (old per-type keys, 4-length type probs) must not crash.
    legacy = {
        "n_primitives_probs": [0.3, 0.4, 0.2, 0.1],
        "primitive_type_probs": [0.4, 0.35, 0.15, 0.1],
        "box_dims_mean": [-3.0, -3.0, -2.8],
        "offset_std": 0.02,
    }
    d = ParameterDistribution.from_dict(legacy)
    assert len(d.primitive_type_probs) == len(PRIMITIVE_SPECS)  # kept default


# --- scoring fixes ----------------------------------------------------------

def test_sphere_stability_nonzero():
    sph = P.CompositeObject(
        primitives=[Sphere(radius=0.03, transform=Transform(translation=np.array([0, 0, 0.03])))],
        name="s")
    assert ObjectScorer().score(sph).stability_score > 0.0


def test_assembly_reward_orders_parts():
    box = P.create_simple_box(np.array([0.05, 0.05, 0.06]))
    multi = P.create_dumbbell()  # 3 parts
    sc = ObjectScorer(ScoringConfig())  # v2 default assembly_weight=1
    assert sc.score(multi).total_score > sc.score(box).total_score


def test_assembly_weight_zero_recovers_v1():
    box = P.create_simple_box(np.array([0.05, 0.05, 0.06]))
    v1 = ObjectScorer(ScoringConfig(assembly_weight=0.0)).score(box).total_score
    # A clean single box maxes every v1 component.
    assert v1 == pytest.approx(1.0, abs=1e-6)


def test_low_grasp_gate_penalizes_ungraspable_not_graspable():
    box = P.create_simple_box(np.array([0.05, 0.05, 0.06]))
    cone = P.CompositeObject(primitives=[
        Cone(radius=0.03, height=0.07,
             transform=Transform(translation=np.array([0.0, 0.0, 0.0175])))])
    off = ObjectScorer(ScoringConfig(low_grasp_gate=False))
    on = ObjectScorer(ScoringConfig(low_grasp_gate=True))
    # A graspable box is unchanged by the gate; a lone (ungraspable) cone is
    # penalized below the acceptance threshold.
    assert on.score(box).total_score == pytest.approx(off.score(box).total_score)
    assert off.score(cone).total_score >= 0.4
    assert on.score(cone).total_score < 0.4


# --- paper-repro preset -----------------------------------------------------

def test_paper_repro_restricts_space():
    g = generator.paper_repro_generator()
    assert g.config.max_primitives == 4
    assert g.config.assembly_weight == 0.0
    assert not g.config.require_connected and not g.config.rerank_by_grasp
    # only the original 4 primitive types carry probability mass
    probs = g.distribution.primitive_type_probs
    assert np.all(probs[4:] == 0.0)
    rng = np.random.default_rng(1)
    assert max(len(g.distribution.sample_object(rng).primitives) for _ in range(50)) <= 4
