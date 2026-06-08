"""
Parametric Primitive Representations for Robotic Object Generation.

Each object is a union of simple primitives (boxes, cylinders, capsules)
with rigid transforms. This keeps generation interpretable and fast.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from enum import Enum
import trimesh
from scipy.spatial.transform import Rotation


class PrimitiveType(Enum):
    BOX = "box"
    CYLINDER = "cylinder"
    CAPSULE = "capsule"
    SPHERE = "sphere"
    # v2 additions — see archetypes/CEM spec table.
    CONE = "cone"
    PYRAMID = "pyramid"
    TORUS = "torus"
    ELLIPSOID = "ellipsoid"
    WEDGE = "wedge"
    # v2.2 additions — hollow/handled shapes for realistic containers.
    HOLLOW_SHELL = "hollow_shell"
    HANDLE = "handle"
    # v2.3 additions — tapered/domed shapes.
    FRUSTUM = "frustum"
    HEMISPHERE = "hemisphere"
    # v2.5 addition — faceted fastener.
    HEX_PRISM = "hex_prism"


@dataclass
class Transform:
    """SE(3) rigid transform: translation + rotation."""
    translation: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))
    
    @classmethod
    def from_euler(cls, translation: np.ndarray, euler_xyz: np.ndarray) -> 'Transform':
        """Create transform from translation and Euler angles (XYZ order)."""
        rot = Rotation.from_euler('xyz', euler_xyz).as_matrix()
        return cls(translation=np.array(translation), rotation=rot)
    
    @classmethod
    def identity(cls) -> 'Transform':
        return cls()
    
    def as_matrix(self) -> np.ndarray:
        """Return 4x4 homogeneous transformation matrix."""
        T = np.eye(4)
        T[:3, :3] = self.rotation
        T[:3, 3] = self.translation
        return T
    
    def apply(self, points: np.ndarray) -> np.ndarray:
        """Apply transform to Nx3 points."""
        return (self.rotation @ points.T).T + self.translation


@dataclass
class Primitive:
    """Base class for geometric primitives."""
    ptype: PrimitiveType = field(default=None)
    transform: Transform = field(default_factory=Transform.identity)
    
    def to_mesh(self) -> trimesh.Trimesh:
        raise NotImplementedError
    
    def volume(self) -> float:
        raise NotImplementedError
    
    def inertia_tensor(self, density: float = 1000.0) -> np.ndarray:
        """Inertia tensor about centroid, assuming uniform density."""
        raise NotImplementedError

    def _mesh_inertia(self, density: float = 1000.0) -> np.ndarray:
        """Inertia about the primitive's centroid, in world orientation,
        derived from its (transformed, watertight) mesh.

        Used by primitive types whose analytic inertia tensor is error-prone
        (cone, torus, wedge, …). The mesh is built centered on its centroid in
        the local frame and then transformed, so ``moment_inertia`` is taken
        about the world-space centroid — matching the contract of the analytic
        ``inertia_tensor`` (about centroid, rotated into world axes).
        """
        m = self.to_mesh()
        m.density = float(density)
        return np.asarray(m.moment_inertia, dtype=float)


@dataclass
class Box(Primitive):
    """
    Axis-aligned box primitive.
    Dimensions: [width_x, depth_y, height_z]
    """
    dimensions: np.ndarray = field(default_factory=lambda: np.array([0.05, 0.05, 0.05]))
    
    def __post_init__(self):
        object.__setattr__(self, 'ptype', PrimitiveType.BOX)
        self.dimensions = np.array(self.dimensions)
    
    def to_mesh(self) -> trimesh.Trimesh:
        mesh = trimesh.creation.box(extents=self.dimensions)
        mesh.apply_transform(self.transform.as_matrix())
        return mesh
    
    def volume(self) -> float:
        return float(np.prod(self.dimensions))
    
    def inertia_tensor(self, density: float = 1000.0) -> np.ndarray:
        """Box inertia about centroid: I = (m/12) * diag(b²+c², a²+c², a²+b²)"""
        m = self.volume() * density
        a, b, c = self.dimensions
        Ixx = (m / 12) * (b**2 + c**2)
        Iyy = (m / 12) * (a**2 + c**2)
        Izz = (m / 12) * (a**2 + b**2)
        I_local = np.diag([Ixx, Iyy, Izz])
        # Rotate inertia tensor to world frame
        R = self.transform.rotation
        return R @ I_local @ R.T


@dataclass
class Cylinder(Primitive):
    """
    Cylinder primitive aligned with Z-axis before transform.
    """
    radius: float = 0.025
    height: float = 0.05
    
    def __post_init__(self):
        object.__setattr__(self, 'ptype', PrimitiveType.CYLINDER)
    
    def to_mesh(self) -> trimesh.Trimesh:
        mesh = trimesh.creation.cylinder(radius=self.radius, height=self.height)
        mesh.apply_transform(self.transform.as_matrix())
        return mesh
    
    def volume(self) -> float:
        return np.pi * self.radius**2 * self.height
    
    def inertia_tensor(self, density: float = 1000.0) -> np.ndarray:
        """
        Cylinder inertia about centroid (axis along Z):
        Ixx = Iyy = (m/12)(3r² + h²)
        Izz = (m/2)r²
        """
        m = self.volume() * density
        r, h = self.radius, self.height
        Ixx = (m / 12) * (3 * r**2 + h**2)
        Iyy = Ixx
        Izz = (m / 2) * r**2
        I_local = np.diag([Ixx, Iyy, Izz])
        R = self.transform.rotation
        return R @ I_local @ R.T


@dataclass
class Sphere(Primitive):
    """Sphere primitive."""
    radius: float = 0.025
    
    def __post_init__(self):
        object.__setattr__(self, 'ptype', PrimitiveType.SPHERE)
    
    def to_mesh(self) -> trimesh.Trimesh:
        # subdivisions=3 keeps the mass/inertia faceting error < ~1% (was ~3-5%
        # at subdivisions=2). See DISCREPANCIES / mesh-accuracy notes.
        mesh = trimesh.creation.icosphere(radius=self.radius, subdivisions=3)
        mesh.apply_transform(self.transform.as_matrix())
        return mesh
    
    def volume(self) -> float:
        return (4/3) * np.pi * self.radius**3
    
    def inertia_tensor(self, density: float = 1000.0) -> np.ndarray:
        """Sphere inertia: I = (2/5)mr² for all axes."""
        m = self.volume() * density
        I_scalar = (2/5) * m * self.radius**2
        return np.diag([I_scalar, I_scalar, I_scalar])


@dataclass
class Capsule(Primitive):
    """
    Capsule (cylinder with hemispherical caps) aligned with Z-axis.
    Total height = height + 2*radius
    """
    radius: float = 0.02
    height: float = 0.04  # cylinder portion only
    
    def __post_init__(self):
        object.__setattr__(self, 'ptype', PrimitiveType.CAPSULE)
    
    def to_mesh(self) -> trimesh.Trimesh:
        mesh = trimesh.creation.capsule(radius=self.radius, height=self.height)
        mesh.apply_transform(self.transform.as_matrix())
        return mesh
    
    def volume(self) -> float:
        # Cylinder + sphere
        v_cyl = np.pi * self.radius**2 * self.height
        v_sphere = (4/3) * np.pi * self.radius**3
        return v_cyl + v_sphere
    
    def inertia_tensor(self, density: float = 1000.0) -> np.ndarray:
        """Approximate as cylinder (close enough for simulation)."""
        m = self.volume() * density
        r = self.radius
        h = self.height + 2 * self.radius  # total height
        Ixx = (m / 12) * (3 * r**2 + h**2)
        Iyy = Ixx
        Izz = (m / 2) * r**2
        I_local = np.diag([Ixx, Iyy, Izz])
        R = self.transform.rotation
        return R @ I_local @ R.T


# ---------------------------------------------------------------------------
# v2 primitive types. Each builds its mesh centered on its own centroid in the
# local frame (so transform.translation == world centroid, matching the
# analytic conventions in CompositeObject) and derives its inertia tensor from
# the mesh via Primitive._mesh_inertia.
# ---------------------------------------------------------------------------

@dataclass
class Cone(Primitive):
    """Cone aligned with +Z (apex up), resting on its circular base."""
    radius: float = 0.025
    height: float = 0.06

    def __post_init__(self):
        object.__setattr__(self, 'ptype', PrimitiveType.CONE)

    def to_mesh(self) -> trimesh.Trimesh:
        mesh = trimesh.creation.cone(radius=self.radius, height=self.height,
                                     sections=32)
        # trimesh cone has its base at z=0; centroid is at h/4. Recenter.
        mesh.apply_translation([0.0, 0.0, -self.height / 4.0])
        mesh.apply_transform(self.transform.as_matrix())
        return mesh

    def volume(self) -> float:
        return (1.0 / 3.0) * np.pi * self.radius ** 2 * self.height

    def inertia_tensor(self, density: float = 1000.0) -> np.ndarray:
        return self._mesh_inertia(density)


@dataclass
class Pyramid(Primitive):
    """Square-base pyramid aligned with +Z (apex up), resting on its base.

    ``radius`` is the circumradius of the square base (built as a 4-section
    cone), ``height`` the apex height.
    """
    radius: float = 0.03
    height: float = 0.05

    def __post_init__(self):
        object.__setattr__(self, 'ptype', PrimitiveType.PYRAMID)

    def to_mesh(self) -> trimesh.Trimesh:
        mesh = trimesh.creation.cone(radius=self.radius, height=self.height,
                                     sections=4)
        mesh.apply_translation([0.0, 0.0, -self.height / 4.0])
        mesh.apply_transform(self.transform.as_matrix())
        return mesh

    def volume(self) -> float:
        # square base inscribed in circle of radius r -> side r*sqrt(2),
        # area 2 r^2; volume = (1/3) * base_area * h.
        return (2.0 / 3.0) * self.radius ** 2 * self.height

    def inertia_tensor(self, density: float = 1000.0) -> np.ndarray:
        return self._mesh_inertia(density)


@dataclass
class Torus(Primitive):
    """Torus lying in the XY plane (ring axis along Z)."""
    major_radius: float = 0.04
    minor_radius: float = 0.012

    def __post_init__(self):
        object.__setattr__(self, 'ptype', PrimitiveType.TORUS)

    def to_mesh(self) -> trimesh.Trimesh:
        mesh = trimesh.creation.torus(major_radius=self.major_radius,
                                      minor_radius=self.minor_radius)
        mesh.apply_transform(self.transform.as_matrix())
        return mesh

    def volume(self) -> float:
        return 2.0 * np.pi ** 2 * self.major_radius * self.minor_radius ** 2

    def inertia_tensor(self, density: float = 1000.0) -> np.ndarray:
        return self._mesh_inertia(density)


@dataclass
class Ellipsoid(Primitive):
    """Ellipsoid with semi-axes (rx, ry, rz)."""
    radii: np.ndarray = field(default_factory=lambda: np.array([0.04, 0.03, 0.02]))

    def __post_init__(self):
        object.__setattr__(self, 'ptype', PrimitiveType.ELLIPSOID)
        self.radii = np.array(self.radii, dtype=float)

    def to_mesh(self) -> trimesh.Trimesh:
        mesh = trimesh.creation.icosphere(radius=1.0, subdivisions=3)
        mesh.apply_scale(self.radii.tolist())
        mesh.apply_transform(self.transform.as_matrix())
        return mesh

    def volume(self) -> float:
        return (4.0 / 3.0) * np.pi * float(np.prod(self.radii))

    def inertia_tensor(self, density: float = 1000.0) -> np.ndarray:
        """Exact solid-ellipsoid tensor: I = m/5 diag(b²+c², a²+c², a²+b²),
        rotated into world axes (no faceting error)."""
        m = self.volume() * density
        a, b, c = self.radii
        I_local = (m / 5.0) * np.diag([b * b + c * c, a * a + c * c, a * a + b * b])
        R = self.transform.rotation
        return R @ I_local @ R.T


@dataclass
class Wedge(Primitive):
    """Isosceles triangular prism ("tent"/ramp), extruded along Y.

    Cross-section is a triangle of base ``width`` (x) and ``height`` (z) with a
    centered apex; extruded over ``depth`` (y). Centered on its centroid.
    """
    width: float = 0.05
    depth: float = 0.04
    height: float = 0.04

    def __post_init__(self):
        object.__setattr__(self, 'ptype', PrimitiveType.WEDGE)

    def to_mesh(self) -> trimesh.Trimesh:
        w, d, h = self.width, self.depth, self.height
        # Triangle in XZ: base from -w/2..w/2 at z=0, apex at (0, h); extrude
        # along Y in [-d/2, d/2]. Triangle centroid z = h/3 -> recenter by it.
        cz = h / 3.0
        v = np.array([
            [-w / 2, -d / 2, -cz], [w / 2, -d / 2, -cz], [0.0, -d / 2, h - cz],
            [-w / 2,  d / 2, -cz], [w / 2,  d / 2, -cz], [0.0,  d / 2, h - cz],
        ])
        f = np.array([
            [0, 2, 1], [3, 4, 5],            # the two triangular faces
            [0, 1, 4], [0, 4, 3],            # bottom quad
            [1, 2, 5], [1, 5, 4],            # right slope
            [0, 3, 5], [0, 5, 2],            # left slope
        ])
        mesh = trimesh.Trimesh(vertices=v, faces=f, process=True)
        mesh.fix_normals()
        mesh.apply_transform(self.transform.as_matrix())
        return mesh

    def volume(self) -> float:
        return 0.5 * self.width * self.height * self.depth

    def inertia_tensor(self, density: float = 1000.0) -> np.ndarray:
        return self._mesh_inertia(density)


def _recenter_and_place(mesh: trimesh.Trimesh, transform) -> trimesh.Trimesh:
    """Ensure outward winding (positive volume), recenter on the centroid (so
    ``transform.translation == world centroid``, the project contract), then apply
    the transform. Gate on is_watertight (topological), NOT is_volume — the latter
    needs is_winding_consistent, which needs networkx (absent here) and returns
    False even for a perfectly good mesh. Signed volume + center_mass are valid on
    any watertight, consistently-wound mesh; invert() flips a globally-inward
    winding to outward."""
    if mesh.is_watertight:
        if mesh.volume < 0:
            mesh.invert()
        mesh.apply_translation(-mesh.center_mass)
    mesh.apply_transform(transform.as_matrix())
    return mesh


def _finish_mesh(verts, faces, transform) -> trimesh.Trimesh:
    """Build a manual mesh from vertices/faces, then recenter + place it."""
    return _recenter_and_place(
        trimesh.Trimesh(vertices=verts, faces=faces, process=True), transform)


@dataclass
class HollowShell(Primitive):
    """Open-top hollow cylinder — a cup/mug/pot/jar/bowl BODY with a real wall and
    floor (the cavity is what makes containers look right instead of solid pegs).
    Aligned with +Z, open at the top, centered on its centroid."""
    outer_radius: float = 0.035
    wall_thickness: float = 0.004
    height: float = 0.07
    floor_thickness: float = 0.005

    def __post_init__(self):
        object.__setattr__(self, 'ptype', PrimitiveType.HOLLOW_SHELL)

    def _dims(self):
        R = float(self.outer_radius)
        H = float(self.height)
        w = float(np.clip(self.wall_thickness, 0.001, R - 0.001))
        fl = float(np.clip(self.floor_thickness, 0.001, H - 0.001))
        return R, H, w, fl, R - w

    def to_mesh(self) -> trimesh.Trimesh:
        R, H, w, fl, Ri = self._dims()
        # outer solid cylinder minus an inner cavity cylinder (open top). CSG
        # (manifold3d, already required for unions) yields a clean, consistently-
        # wound manifold — unlike a manual revolve, whose inner/outer walls wind
        # oppositely and corrupt center_mass without networkx to repair them. If
        # no CSG backend is present, fall back to a solid cylinder.
        outer = trimesh.creation.cylinder(radius=R, height=H, sections=48)
        outer.apply_translation([0.0, 0.0, H / 2.0])
        pad = 0.01
        hc = (H - fl) + pad
        inner = trimesh.creation.cylinder(radius=Ri, height=hc, sections=48)
        inner.apply_translation([0.0, 0.0, fl + hc / 2.0])
        try:
            shell = outer.difference(inner)
        except Exception:
            shell = outer
        if shell.is_watertight:
            shell.apply_translation(-shell.center_mass)
        shell.apply_transform(self.transform.as_matrix())
        return shell

    def volume(self) -> float:
        R, H, w, fl, Ri = self._dims()
        return float(np.pi * (R * R * H - Ri * Ri * (H - fl)))

    def inertia_tensor(self, density: float = 1000.0) -> np.ndarray:
        return self._mesh_inertia(density)


@dataclass
class Handle(Primitive):
    """C-shaped handle: an ELLIPTICAL tube cross-section swept along a circular arc
    (a partial torus). Replaces the faked full-torus / straight-cylinder handles on
    mugs, cups, pots, teapots, kettlebells, hooks. The arc is symmetric about +X
    and opens toward -X (so it attaches against a body wall on the +X side).
    Centered on its centroid."""
    major_radius: float = 0.02
    tube_a: float = 0.006      # in-plane (radial) semi-axis of the tube
    tube_b: float = 0.005      # out-of-plane (z) semi-axis of the tube
    arc_angle: float = 1.5 * np.pi

    def __post_init__(self):
        object.__setattr__(self, 'ptype', PrimitiveType.HANDLE)

    def to_mesh(self) -> trimesh.Trimesh:
        R = float(self.major_radius)
        a = float(max(self.tube_a, 1e-3))
        b = float(max(self.tube_b, 1e-3))
        arc = float(np.clip(self.arc_angle, 0.4, 2.0 * np.pi - 0.05))
        n_major = max(8, int(48 * arc / (2.0 * np.pi)))
        n_minor = 18
        phi = np.linspace(-arc / 2.0, arc / 2.0, n_major)
        tt = np.linspace(0.0, 2.0 * np.pi, n_minor, endpoint=False)
        ct, st = np.cos(tt), np.sin(tt)
        verts = []
        for p in phi:
            cx, cy = R * np.cos(p), R * np.sin(p)
            ex, ey = np.cos(p), np.sin(p)        # in-plane radial direction
            for k in range(n_minor):
                rad = a * ct[k]
                verts.append([cx + rad * ex, cy + rad * ey, b * st[k]])
        faces = []
        for i in range(n_major - 1):
            for k in range(n_minor):
                k2 = (k + 1) % n_minor
                A = i * n_minor + k; B = i * n_minor + k2
                C = (i + 1) * n_minor + k; D = (i + 1) * n_minor + k2
                faces += [[A, B, D], [A, D, C]]
        # fan-cap both open ends to the tube-centre point
        c0 = len(verts); verts.append([R * np.cos(phi[0]), R * np.sin(phi[0]), 0.0])
        for k in range(n_minor):
            faces.append([c0, (k + 1) % n_minor, k])
        base = (n_major - 1) * n_minor
        c1 = len(verts); verts.append([R * np.cos(phi[-1]), R * np.sin(phi[-1]), 0.0])
        for k in range(n_minor):
            faces.append([c1, base + k, base + (k + 1) % n_minor])
        return _finish_mesh(np.asarray(verts, float), np.asarray(faces, int),
                            self.transform)

    def volume(self) -> float:
        # cross-section area (π·a·b) × swept centroid path (R·arc_angle)
        arc = float(np.clip(self.arc_angle, 0.4, 2.0 * np.pi - 0.05))
        return float(np.pi * self.tube_a * self.tube_b * self.major_radius * arc)

    def inertia_tensor(self, density: float = 1000.0) -> np.ndarray:
        return self._mesh_inertia(density)


@dataclass
class Frustum(Primitive):
    """Truncated cone — a flat-topped taper for flared cups, buckets, flowerpots,
    lampshades (a plain Cone can only taper to a point). +Z, resting on its
    ``radius_bottom`` base, centered on its centroid."""
    radius_bottom: float = 0.04
    radius_top: float = 0.025
    height: float = 0.06

    def __post_init__(self):
        object.__setattr__(self, 'ptype', PrimitiveType.FRUSTUM)

    def to_mesh(self) -> trimesh.Trimesh:
        rb = float(max(self.radius_bottom, 1e-3))
        rt = float(max(self.radius_top, 1e-3))
        H = float(max(self.height, 2e-3))
        if abs(rb - rt) < 1e-4:                       # degenerate -> cylinder
            m = trimesh.creation.cylinder(radius=rb, height=H, sections=48)
            m.apply_translation([0.0, 0.0, H / 2.0])
        else:
            # clip a full cone at z=H: its radius there equals the smaller radius.
            rmax = max(rb, rt)
            h_full = H * rmax / (rmax - min(rb, rt))
            cone = trimesh.creation.cone(radius=rmax, height=h_full, sections=48)
            if rt > rb:                               # flared (wider top): apex down
                cone.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))
                cone.apply_translation([0.0, 0.0, H])
            clip = trimesh.creation.box(extents=[4 * rmax, 4 * rmax, H])
            clip.apply_translation([0.0, 0.0, H / 2.0])
            try:
                m = cone.intersection(clip)
            except Exception:                          # no CSG backend -> mid cylinder
                m = trimesh.creation.cylinder(radius=0.5 * (rb + rt), height=H, sections=48)
                m.apply_translation([0.0, 0.0, H / 2.0])
        return _recenter_and_place(m, self.transform)

    def volume(self) -> float:
        rb, rt, H = self.radius_bottom, self.radius_top, self.height
        return float(np.pi * H / 3.0 * (rb * rb + rb * rt + rt * rt))

    def inertia_tensor(self, density: float = 1000.0) -> np.ndarray:
        return self._mesh_inertia(density)


@dataclass
class Hemisphere(Primitive):
    """Solid half-sphere (dome) for lids, domes, scoops — a flat-bottomed cap
    instead of a full sphere. +Z, flat face down, centered on its centroid."""
    radius: float = 0.03

    def __post_init__(self):
        object.__setattr__(self, 'ptype', PrimitiveType.HEMISPHERE)

    def to_mesh(self) -> trimesh.Trimesh:
        r = float(max(self.radius, 2e-3))
        sphere = trimesh.creation.icosphere(radius=r, subdivisions=3)
        clip = trimesh.creation.box(extents=[3 * r, 3 * r, 2 * r])
        clip.apply_translation([0.0, 0.0, r])          # keep the z>=0 half
        try:
            m = sphere.intersection(clip)
        except Exception:
            m = sphere
        return _recenter_and_place(m, self.transform)

    def volume(self) -> float:
        return float(2.0 / 3.0 * np.pi * self.radius ** 3)

    def inertia_tensor(self, density: float = 1000.0) -> np.ndarray:
        return self._mesh_inertia(density)


@dataclass
class HexPrism(Primitive):
    """Regular hexagonal prism — for hex nuts and bolt heads (a round cylinder or
    4-gon pyramid is the wrong shape for a fastener). +Z axis, ``radius`` is the
    circumradius (centre-to-corner); centered on its centroid."""
    radius: float = 0.018
    height: float = 0.012

    def __post_init__(self):
        object.__setattr__(self, 'ptype', PrimitiveType.HEX_PRISM)

    def to_mesh(self) -> trimesh.Trimesh:
        mesh = trimesh.creation.cylinder(radius=self.radius, height=self.height,
                                         sections=6)  # already centred on origin
        mesh.apply_transform(self.transform.as_matrix())
        return mesh

    def volume(self) -> float:
        return float(1.5 * np.sqrt(3.0) * self.radius ** 2 * self.height)

    def inertia_tensor(self, density: float = 1000.0) -> np.ndarray:
        return self._mesh_inertia(density)


def seat_height(prim: Primitive) -> float:
    """Distance from a primitive's centroid to its lowest point in its default
    (unrotated) orientation — i.e. the z translation that rests it on z=0.

    Single source of truth for ground-seating, used by the CEM sampler and the
    baseline decoders.
    """
    if isinstance(prim, Box):
        return float(prim.dimensions[2] / 2)
    if isinstance(prim, Cylinder):
        return float(prim.height / 2)
    if isinstance(prim, Sphere):
        return float(prim.radius)
    if isinstance(prim, Capsule):
        return float(prim.height / 2 + prim.radius)
    if isinstance(prim, (Cone, Pyramid)):
        return float(prim.height / 4)        # centroid at h/4 above the base
    if isinstance(prim, Torus):
        return float(prim.minor_radius)
    if isinstance(prim, Ellipsoid):
        return float(prim.radii[2])
    if isinstance(prim, Wedge):
        return float(prim.height / 3)        # triangle centroid at h/3
    # Fallback: derive from the mesh.
    return float(-prim.to_mesh().bounds[0][2])


def half_extents(prim: Primitive) -> np.ndarray:
    """Axis-aligned half-extents of a primitive in its LOCAL (unrotated) frame.

    Used by the structured-placement sampler to seat one primitive against a
    face of another. Approximate for cone/pyramid/wedge (treated symmetric),
    which is fine for contact placement with an overlap margin.
    """
    if isinstance(prim, Box):
        return np.asarray(prim.dimensions, float) / 2.0
    if isinstance(prim, Cylinder):
        return np.array([prim.radius, prim.radius, prim.height / 2.0])
    if isinstance(prim, Sphere):
        return np.array([prim.radius, prim.radius, prim.radius])
    if isinstance(prim, Capsule):
        h = prim.height / 2.0 + prim.radius
        return np.array([prim.radius, prim.radius, h])
    if isinstance(prim, (Cone, Pyramid)):
        return np.array([prim.radius, prim.radius, prim.height / 2.0])
    if isinstance(prim, Torus):
        rr = prim.major_radius + prim.minor_radius
        return np.array([rr, rr, prim.minor_radius])
    if isinstance(prim, Ellipsoid):
        return np.asarray(prim.radii, float)
    if isinstance(prim, Wedge):
        return np.array([prim.width / 2.0, prim.depth / 2.0, prim.height / 2.0])
    he = (prim.to_mesh().bounds[1] - prim.to_mesh().bounds[0]) / 2.0
    return np.asarray(he, float)


@dataclass
class CompositeObject:
    """
    Object composed of multiple primitives via union.
    This is our "object family" representation.
    """
    primitives: List[Primitive] = field(default_factory=list)
    name: str = "generated_object"
    friction: float = 1.0  # Coulomb friction coefficient
    
    def to_mesh(self, boolean_union: bool = True) -> trimesh.Trimesh:
        """
        Convert to single triangle mesh.
        
        Args:
            boolean_union: If True, compute boolean union (slower but cleaner).
                          If False, just concatenate (faster, may have internal faces).
        """
        if not self.primitives:
            return trimesh.Trimesh()
        
        meshes = [p.to_mesh() for p in self.primitives]
        
        if boolean_union and len(meshes) > 1:
            # Try boolean union for a clean mesh, using whatever CSG backend
            # trimesh finds (manifold3d or Blender). Falls back to concatenation
            # if no backend is installed.
            try:
                return trimesh.boolean.union(meshes)
            except Exception:
                # Fall back to concatenation if boolean fails
                pass
        
        # Concatenate meshes
        return trimesh.util.concatenate(meshes)
    
    def total_volume(self) -> float:
        """Approximate volume (sum of primitives, ignores overlap)."""
        return sum(p.volume() for p in self.primitives)
    
    def center_of_mass(self, density: float = 1000.0) -> np.ndarray:
        """Compute center of mass assuming uniform density."""
        if not self.primitives:
            return np.zeros(3)
        
        total_mass = 0.0
        weighted_pos = np.zeros(3)
        
        for p in self.primitives:
            m = p.volume() * density
            # Centroid is at the transform's translation
            weighted_pos += m * p.transform.translation
            total_mass += m
        
        return weighted_pos / total_mass if total_mass > 0 else np.zeros(3)
    
    def combined_inertia(self, density: float = 1000.0) -> Tuple[float, np.ndarray]:
        """
        Compute total mass and inertia tensor about combined COM.
        Uses parallel axis theorem.
        """
        if not self.primitives:
            return 0.0, np.zeros((3, 3))
        
        com = self.center_of_mass(density)
        total_mass = 0.0
        total_inertia = np.zeros((3, 3))
        
        for p in self.primitives:
            m = p.volume() * density
            I_p = p.inertia_tensor(density)
            
            # Vector from combined COM to primitive centroid
            r = p.transform.translation - com
            
            # Parallel axis theorem: I = I_cm + m(|r|²E - r⊗r)
            r_outer = np.outer(r, r)
            I_shifted = I_p + m * (np.dot(r, r) * np.eye(3) - r_outer)
            
            total_inertia += I_shifted
            total_mass += m
        
        return total_mass, total_inertia
    
    def _strict_union(self) -> trimesh.Trimesh:
        """Boolean union that RAISES if no CSG backend is available.

        Unlike ``to_mesh(boolean_union=True)`` (which silently falls back to
        concatenation), this guarantees the result is a true union — required
        for overlap-aware mass properties and connectivity. Needs a boolean
        backend (``manifold3d`` — see requirements.txt — or Blender on PATH).
        """
        meshes = [p.to_mesh() for p in self.primitives]
        if not meshes:
            return trimesh.Trimesh()
        if len(meshes) == 1:
            return meshes[0]
        return trimesh.boolean.union(meshes)

    def mesh_mass_properties(self, density: float = 1000.0):
        """Mass / COM / inertia from the *boolean-union* solid.

        The analytic `center_of_mass` / `combined_inertia` sum per-primitive
        volumes and so DOUBLE-COUNT any region where primitives overlap,
        over-estimating mass and biasing the COM/inertia. This computes the
        same quantities from the watertight union mesh, which is overlap-aware
        and matches the geometry actually exported.

        Returns (mass, inertia_3x3_about_com, com). Raises if the union is not
        watertight (callers should fall back to the analytic path).
        """
        # Single primitive: use its exact analytic tensor where one exists
        # (box/cylinder/sphere/ellipsoid) — zero faceting error and no CSG.
        if len(self.primitives) == 1:
            p = self.primitives[0]
            if isinstance(p, (Box, Cylinder, Sphere, Ellipsoid)):
                mass = float(p.volume() * density)
                return mass, p.inertia_tensor(density), np.asarray(p.transform.translation, float)

        mesh = self._strict_union()
        if mesh is None or len(mesh.vertices) == 0 or not mesh.is_watertight:
            raise ValueError("union mesh is not watertight; cannot integrate mass properties")
        mesh.density = float(density)
        mass = float(mesh.mass)
        com = np.asarray(mesh.center_mass, dtype=float)
        inertia = np.asarray(mesh.moment_inertia, dtype=float)  # about COM, body frame
        return mass, inertia, com

    def n_connected_components(self) -> int:
        """Number of spatially-connected solid bodies in the union.

        Uses the boolean union so that overlapping primitives merge into one
        body; physically separated (floating) primitives remain distinct. A
        value > 1 means the object is not a single connected rigid body — which
        is physically unrealizable as a single exported link.
        """
        try:
            mesh = self._strict_union()
            if mesh is None or len(mesh.vertices) == 0:
                return 0
            return int(mesh.body_count)
        except Exception:
            # If the union engine is unavailable, be permissive (treat as one).
            return 1

    def is_connected(self) -> bool:
        """True if the object forms a single connected rigid body."""
        return self.n_connected_components() <= 1

    def aabb(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute axis-aligned bounding box. Returns (min_corner, max_corner)."""
        mesh = self.to_mesh(boolean_union=False)
        if len(mesh.vertices) == 0:
            return np.zeros(3), np.zeros(3)
        return mesh.bounds[0], mesh.bounds[1]
    
    def aabb_extents(self) -> np.ndarray:
        """Return [width, depth, height] of AABB."""
        min_c, max_c = self.aabb()
        return max_c - min_c


# Factory functions for common object archetypes
def create_simple_box(dims: np.ndarray, position: np.ndarray = None) -> CompositeObject:
    """Create a simple box object."""
    if position is None:
        position = np.array([0, 0, dims[2]/2])  # Rest on ground

    box = Box(
        dimensions=dims,
        transform=Transform(translation=position)
    )
    return CompositeObject(primitives=[box], name="simple_box")


# Three concrete simple-box variants used by ``create_archetype_set`` and
# the per-archetype CEM scale experiment.  Each has a default ``dims``
# vector so ``ArchetypeDistribution._introspect_params`` learns a
# Gaussian over the dimensions centered on this archetype's footprint.
def create_small_box(dims: np.ndarray = np.array([0.04, 0.04, 0.04])
                     ) -> CompositeObject:
    obj = create_simple_box(dims)
    obj.name = "small_box"
    return obj


def create_tall_box(dims: np.ndarray = np.array([0.03, 0.03, 0.10])
                    ) -> CompositeObject:
    obj = create_simple_box(dims)
    obj.name = "tall_box"
    return obj


def create_flat_box(dims: np.ndarray = np.array([0.08, 0.06, 0.02])
                    ) -> CompositeObject:
    obj = create_simple_box(dims)
    obj.name = "flat_box"
    return obj


def create_mug_like(
    body_radius: float = 0.035,
    body_height: float = 0.08,
    handle_radius: float = 0.008,
    handle_height: float = 0.05
) -> CompositeObject:
    """Create a mug-like object (cylinder + handle)."""
    # Main body
    body = Cylinder(
        radius=body_radius,
        height=body_height,
        transform=Transform(translation=np.array([0, 0, body_height/2]))
    )
    
    # Handle (small cylinder attached to side). Offset chosen so the handle
    # OVERLAPS the body (a single connected solid), not merely touches it.
    handle_offset = body_radius
    handle = Cylinder(
        radius=handle_radius,
        height=handle_height,
        transform=Transform.from_euler(
            translation=np.array([handle_offset, 0, body_height/2]),
            euler_xyz=np.array([np.pi/2, 0, 0])  # Rotate to be horizontal
        )
    )
    
    return CompositeObject(primitives=[body, handle], name="mug_like")


def create_l_shape(
    base_dims: np.ndarray = None,
    upright_dims: np.ndarray = None
) -> CompositeObject:
    """Create an L-shaped object from two boxes."""
    if base_dims is None:
        base_dims = np.array([0.08, 0.04, 0.02])
    if upright_dims is None:
        upright_dims = np.array([0.02, 0.04, 0.06])
    
    base = Box(
        dimensions=base_dims,
        transform=Transform(translation=np.array([0, 0, base_dims[2]/2]))
    )
    
    # Upright attached at one end
    upright_x = (base_dims[0] - upright_dims[0]) / 2
    upright = Box(
        dimensions=upright_dims,
        transform=Transform(translation=np.array([
            upright_x, 0, base_dims[2] + upright_dims[2]/2
        ]))
    )
    
    return CompositeObject(primitives=[base, upright], name="l_shape")


def create_dumbbell(
    handle_length: float = 0.08,
    handle_radius: float = 0.01,
    weight_radius: float = 0.03
) -> CompositeObject:
    """Create a dumbbell shape (two spheres connected by a cylinder)."""
    # Handle
    handle = Cylinder(
        radius=handle_radius,
        height=handle_length,
        transform=Transform.from_euler(
            translation=np.array([0, 0, handle_length/2]),
            euler_xyz=np.array([0, 0, 0])
        )
    )
    
    # Weights at ends
    # Bottom weight
    weight1 = Sphere(
        radius=weight_radius,
        transform=Transform(translation=np.array([0, 0, -weight_radius + 0.01])) # Slightly embedded/offset
    )
    # Top weight
    weight2 = Sphere(
        radius=weight_radius,
        transform=Transform(translation=np.array([0, 0, handle_length + weight_radius*0.8]))
    )
    
    # Adjust handle to be centered relative to weights? 
    # Actually let's make it symmetric around Z center usually, but here we build up from Z=0
    # Let's rebuild:
    # Weight 1 at Z=R
    weight1 = Sphere(
        radius=weight_radius,
        transform=Transform(translation=np.array([0, 0, weight_radius]))
    )
    # Handle on top of weight 1
    handle = Cylinder(
        radius=handle_radius,
        height=handle_length,
        transform=Transform(translation=np.array([0, 0, weight_radius + handle_length/2]))
    )
    # Weight 2 on top of handle (overlapping the handle top, not just touching)
    weight2 = Sphere(
        radius=weight_radius,
        transform=Transform(translation=np.array([0, 0, weight_radius + handle_length]))
    )
    
    return CompositeObject(primitives=[weight1, handle, weight2], name="dumbbell")


def create_hammer(
    handle_length: float = 0.12,
    handle_radius: float = 0.012,
    head_dims: np.ndarray = None
) -> CompositeObject:
    """Create a hammer-like object (cylinder handle + box head)."""
    if head_dims is None:
        head_dims = np.array([0.08, 0.035, 0.035])
    
    # Handle (vertical)
    handle = Cylinder(
        radius=handle_radius,
        height=handle_length,
        transform=Transform(translation=np.array([0, 0, handle_length/2]))
    )
    
    # Head (horizontal box at top)
    head = Box(
        dimensions=head_dims,
        transform=Transform(translation=np.array([0, 0, handle_length + head_dims[2]/2]))
    )
    
    return CompositeObject(primitives=[handle, head], name="hammer")


def create_t_shape(
    h_dims: np.ndarray = None,
    v_dims: np.ndarray = None
) -> CompositeObject:
    """Create a T-shaped object."""
    if h_dims is None: h_dims = np.array([0.10, 0.03, 0.03]) # Top bar
    if v_dims is None: v_dims = np.array([0.03, 0.03, 0.08]) # Vertical post
    
    # Vertical post centered at origin
    post = Box(
        dimensions=v_dims,
        transform=Transform(translation=np.array([0, 0, v_dims[2]/2]))
    )
    
    # Horizontal bar on top
    bar = Box(
        dimensions=h_dims,
        transform=Transform(translation=np.array([0, 0, v_dims[2] + h_dims[2]/2]))
    )
    
    return CompositeObject(primitives=[post, bar], name="t_shape")


def create_u_shape(
    base_dims: np.ndarray = None,
    wall_dims: np.ndarray = None
) -> CompositeObject:
    """Create a U-shaped object."""
    if base_dims is None: base_dims = np.array([0.10, 0.04, 0.02])
    if wall_dims is None: wall_dims = np.array([0.02, 0.04, 0.06])
    
    # Base
    base = Box(
        dimensions=base_dims,
        transform=Transform(translation=np.array([0, 0, base_dims[2]/2]))
    )
    
    # Left Wall
    left = Box(
        dimensions=wall_dims,
        transform=Transform(translation=np.array([
            -base_dims[0]/2 + wall_dims[0]/2, 0, base_dims[2] + wall_dims[2]/2
        ]))
    )
    
    # Right Wall
    right = Box(
        dimensions=wall_dims,
        transform=Transform(translation=np.array([
            base_dims[0]/2 - wall_dims[0]/2, 0, base_dims[2] + wall_dims[2]/2
        ]))
    )
    
    return CompositeObject(primitives=[base, left, right], name="u_shape")


def create_v_shape(
    arm_length: float = 0.08,
    radius: float = 0.015,
    angle_deg: float = 45.0
) -> CompositeObject:
    """Create a V-shaped object using two capsules."""
    angle_rad = np.radians(angle_deg)
    
    # Left arm (rotated -angle)
    # Pivot at origin. Center of arm is at length/2 * [sin, 0, cos]
    # For simplicity, lie flat on table
    
    # Let's make it flat on ground (XY plane)
    # Origin is the "corner".
    
    # Left Arm
    left_tr = Transform.from_euler(
        translation=np.array([
            arm_length/2 * np.cos(np.radians(180-angle_deg)), 
            arm_length/2 * np.sin(np.radians(180-angle_deg)), 
            radius
        ]),
        euler_xyz=np.array([0, np.pi/2, np.radians(angle_deg)]) # Capsule aligns Z default, rotate to XY
    )
    left = Capsule(radius=radius, height=arm_length, transform=left_tr)

    # Right Arm
    right_tr = Transform.from_euler(
        translation=np.array([
            arm_length/2 * np.cos(np.radians(angle_deg)), 
            arm_length/2 * np.sin(np.radians(angle_deg)), 
            radius
        ]),
        euler_xyz=np.array([0, np.pi/2, np.radians(-angle_deg)])
    )
    right = Capsule(radius=radius, height=arm_length, transform=right_tr)    

    # Simpler approach: V in XZ plane standing up? No, usually manipulation objects lie flat.
    # Let's do simple orthogonal V first? 
    # Let's do standard boomerang shape.
    
    return CompositeObject(primitives=[left, right], name="v_shape")


def create_monitor(
    screen_dims: np.ndarray = None,
    stand_height: float = 0.05
) -> CompositeObject:
    """Create a monitor-like object."""
    if screen_dims is None: screen_dims = np.array([0.12, 0.01, 0.08])
    
    # Base plate
    base = Box(
        dimensions=np.array([0.06, 0.06, 0.01]),
        transform=Transform(translation=np.array([0, 0, 0.005]))
    )
    
    # Stand pole
    pole = Cylinder(
        radius=0.01,
        height=stand_height,
        transform=Transform(translation=np.array([0, 0, 0.01 + stand_height/2]))
    )
    
    # Screen
    screen = Box(
        dimensions=screen_dims,
        transform=Transform(translation=np.array([0, 0, 0.01 + stand_height + screen_dims[2]/2]))
    )
    
    return CompositeObject(primitives=[base, pole, screen], name="monitor")


def create_barbell(
    handle_len: float = 0.12,
    handle_rad: float = 0.01,
    weight_rad: float = 0.04,
    weight_width: float = 0.02
) -> CompositeObject:
    """Create a barbell with cylindrical weights."""
    # Handle
    handle = Cylinder(
        radius=handle_rad,
        height=handle_len,
        transform=Transform.from_euler(
            translation=np.array([0, 0, weight_rad]),
            euler_xyz=np.array([0, np.pi/2, 0]) # Horizontal
        )
    )
    
    # Weights (Cylinders rotated 90 deg to align with handle? No, same axis as handle)
    # Cylinder default axis is Z. We want handle along X.
    # Weights also along X.
    
    w1 = Cylinder(
        radius=weight_rad,
        height=weight_width,
        transform=Transform.from_euler(
            translation=np.array([-handle_len/2, 0, weight_rad]),
            euler_xyz=np.array([0, np.pi/2, 0])
        )
    )
    
    w2 = Cylinder(
        radius=weight_rad,
        height=weight_width,
        transform=Transform.from_euler(
            translation=np.array([handle_len/2, 0, weight_rad]),
            euler_xyz=np.array([0, np.pi/2, 0])
        )
    )
    
    return CompositeObject(primitives=[handle, w1, w2], name="barbell")


def create_snowman(
    r1: float = 0.04,
    r2: float = 0.03,
    r3: float = 0.02
) -> CompositeObject:
    """Create a stack of 3 spheres."""
    z = 0
    
    s1 = Sphere(radius=r1, transform=Transform(translation=np.array([0, 0, z + r1])))
    z += 2*r1 * 0.9 # Little overlap
    
    s2 = Sphere(radius=r2, transform=Transform(translation=np.array([0, 0, z + r2])))
    z += 2*r2 * 0.9
    
    s3 = Sphere(radius=r3, transform=Transform(translation=np.array([0, 0, z + r3])))
    
    return CompositeObject(primitives=[s1, s2, s3], name="snowman")


def create_camera(
    body_dims: np.ndarray = None,
    lens_radius: float = 0.025
) -> CompositeObject:
    """Create a camera shape."""
    if body_dims is None: body_dims = np.array([0.10, 0.04, 0.06])
    
    # Body
    body = Box(
        dimensions=body_dims,
        transform=Transform(translation=np.array([0, 0, body_dims[2]/2]))
    )
    
    # Lens (cylinder sticking out of front Y face)
    # Body Y range: -0.02 to 0.02
    lens_len = 0.03
    lens = Cylinder(
        radius=lens_radius,
        height=lens_len,
        transform=Transform.from_euler(
            translation=np.array([0, body_dims[1]/2 + lens_len/2 - 0.005, body_dims[2]/2]), # Offset Y
            euler_xyz=np.array([np.pi/2, 0, 0]) # Rotate to point Y
        )
    )
    
    return CompositeObject(primitives=[body, lens], name="camera")


def create_frying_pan(
    pan_radius: float = 0.06,
    pan_height: float = 0.02,
    handle_len: float = 0.10
) -> CompositeObject:
    """Create a frying pan."""
    # Pan (Cylinder)
    pan = Cylinder(
        radius=pan_radius,
        height=pan_height,
        transform=Transform(translation=np.array([0, 0, pan_height/2]))
    )
    
    # Handle
    handle = Cylinder(
        radius=0.008,
        height=handle_len,
        transform=Transform.from_euler(
            translation=np.array([pan_radius + handle_len/2 - 0.005, 0, pan_height/2]),
            euler_xyz=np.array([0, np.pi/2, 0])
        )
    )
    
    return CompositeObject(primitives=[pan, handle], name="frying_pan")


def create_flashlight(
    head_rad: float = 0.025,
    handle_rad: float = 0.012,
    total_len: float = 0.15
) -> CompositeObject:
    """Create a flashlight."""
    head_len = 0.04
    handle_len = total_len - head_len
    
    # Laying flat along X
    
    # Handle
    handle = Cylinder(
        radius=handle_rad,
        height=handle_len,
        transform=Transform.from_euler(
            translation=np.array([handle_len/2, 0, head_rad]), # Z=head_rad to lie flat? Or just radius
            euler_xyz=np.array([0, np.pi/2, 0])
        )
    )
    
    # Head
    head = Cylinder(
        radius=head_rad,
        height=head_len,
        transform=Transform.from_euler(
            translation=np.array([handle_len + head_len/2, 0, head_rad]),
            euler_xyz=np.array([0, np.pi/2, 0])
        )
    )
    
    return CompositeObject(primitives=[handle, head], name="flashlight")


def create_spatula(
    handle_len: float = 0.12,
    blade_dims: np.ndarray = None
) -> CompositeObject:
    """Create a spatula."""
    if blade_dims is None: blade_dims = np.array([0.06, 0.08, 0.005])
    
    # Handle
    handle = Cylinder(
        radius=0.008,
        height=handle_len,
        transform=Transform.from_euler(
            translation=np.array([0, -handle_len/2, 0.01]),
            euler_xyz=np.array([np.pi/2, 0, 0]) # Along Y
        )
    )
    
    # Blade
    blade = Box(
        dimensions=blade_dims,
        transform=Transform(translation=np.array([0, blade_dims[1]/2, 0.005]))
    )
    
    return CompositeObject(primitives=[handle, blade], name="spatula")


def create_remote(
    dims: np.ndarray = None
) -> CompositeObject:
    """Create a remote control (box with buttons?). Just a detailed box for now."""
    if dims is None: dims = np.array([0.05, 0.15, 0.015])
    
    body = Box(
        dimensions=dims,
        transform=Transform(translation=np.array([0, 0, dims[2]/2]))
    )
    
    # Add a "button" row (small box on top)
    btn = Box(
        dimensions=np.array([0.04, 0.04, 0.005]),
        transform=Transform(translation=np.array([0, dims[1]/3, dims[2] + 0.0025]))
    )
    
    return CompositeObject(primitives=[body, btn], name="remote")


def create_joystick(
    base_dims: np.ndarray = None,
    stick_height: float = 0.06
) -> CompositeObject:
    """Create a joystick arcade controller."""
    if base_dims is None: base_dims = np.array([0.10, 0.08, 0.04])
    
    # Base
    base = Box(
        dimensions=base_dims,
        transform=Transform(translation=np.array([0, 0, base_dims[2]/2]))
    )
    
    # Stick
    stick = Cylinder(
        radius=0.008,
        height=stick_height,
        transform=Transform(translation=np.array([0, 0, base_dims[2] + stick_height/2]))
    )
    
    # Ball top
    ball = Sphere(
        radius=0.02,
        transform=Transform(translation=np.array([0, 0, base_dims[2] + stick_height + 0.015]))
    )
    
    return CompositeObject(primitives=[base, stick, ball], name="joystick")

def create_bottle(
    body_radius: float = 0.03,
    body_height: float = 0.08,
    neck_radius: float = 0.01,
    neck_height: float = 0.03
) -> CompositeObject:
    """Create a bottle shape (cylinder body + cylinder neck)."""
    # Main body
    body = Cylinder(
        radius=body_radius,
        height=body_height,
        transform=Transform(translation=np.array([0, 0, body_height/2]))
    )
    
    # Neck
    neck = Cylinder(
        radius=neck_radius,
        height=neck_height,
        transform=Transform(translation=np.array([0, 0, body_height + neck_height/2]))
    )
    
    return CompositeObject(primitives=[body, neck], name="bottle")

