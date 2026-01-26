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
        mesh = trimesh.creation.icosphere(radius=self.radius, subdivisions=2)
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


@dataclass
class CompositeObject:
    """
    Object composed of multiple primitives via union.
    This is our "object family" representation.
    """
    primitives: List[Primitive] = field(default_factory=list)
    name: str = "generated_object"
    
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
            # Try boolean union for clean mesh
            try:
                result = meshes[0]
                for m in meshes[1:]:
                    result = result.union(m, engine='blender')
                return result
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
    
    # Handle (small cylinder attached to side)
    handle_offset = body_radius + handle_radius
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
