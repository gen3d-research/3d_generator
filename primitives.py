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
    # Weight 2 on top of handle
    weight2 = Sphere(
        radius=weight_radius,
        transform=Transform(translation=np.array([0, 0, weight_radius*2 + handle_length]))
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

