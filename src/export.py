"""
URDF/SDF Export for ROS 2 Integration.

Exports generated objects to standard robot description formats
with physically consistent inertial properties, ready for:
- Gazebo simulation
- MoveIt 2 planning
- RViz visualization
"""

import numpy as np
from pathlib import Path
from typing import Optional, Dict
from dataclasses import dataclass
import trimesh
import xml.etree.ElementTree as ET
from xml.dom import minidom
import yaml
import shutil

from primitives import CompositeObject, PrimitiveType


def _rpy_from_matrix(R: np.ndarray):
    """Rotation matrix -> URDF/SDF fixed-axis roll-pitch-yaw."""
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(np.asarray(R, dtype=float)).as_euler('xyz')


@dataclass
class ExportConfig:
    """Configuration for URDF/SDF export."""
    density: float = 1000.0           # kg/m³
    friction_mu1: float = 0.8         # Coulomb friction
    friction_mu2: float = 0.8         # Torsional friction
    restitution: float = 0.1          # Bounce coefficient

    # Mesh simplification
    simplify_collision: bool = True
    max_collision_faces: int = 500
    use_convex_hull: bool = False     # Use convex hull for collision

    # v2.9 default (True): emit ONE collision element PER PRIMITIVE — native
    # box/cylinder/sphere shapes where exact (capsule = cylinder + 2 spheres),
    # a small per-primitive convex-hull mesh otherwise. Native/convex pairs
    # bypass ODE's fragile trimesh-trimesh collider (the assert that killed gz
    # worlds at scale when a failed despawn left two mesh objects overlapping)
    # and are much cheaper than one big non-convex mesh. The single
    # collision-mesh OBJ is still written for MoveIt planning-scene use.
    # Set False for the legacy single-mesh collision.
    per_primitive_collision: bool = True

    # Inertia computation. v2 default (True) computes overlap-aware
    # mass/COM/inertia from the boolean-union solid (correct for overlapping
    # primitives; see DISCREPANCIES.md item 2), falling back to the analytic
    # overlap-summing path if the union is not watertight or no CSG backend is
    # available. Set False for the v1 paper-repro behavior.
    use_mesh_inertia: bool = True
    
    # Visual appearance
    color_rgba: tuple = (0.7, 0.7, 0.7, 1.0)  # Default gray
    
    # File formats
    mesh_format: str = "obj"  # "obj" or "stl"

    # v2 default (True): export the watertight boolean-UNION as the visual /
    # collision mesh instead of a raw concatenation of overlapping primitives
    # (which leaves internal faces / self-intersections — bad topology in
    # RViz/MoveIt). Falls back to concatenation if no CSG backend is available.
    union_visual_mesh: bool = True


class URDFExporter:
    """Exports CompositeObjects to URDF format with all required files."""
    
    def __init__(self, config: ExportConfig = None):
        self.config = config or ExportConfig()
    
    def export(self, 
               obj: CompositeObject,
               output_dir: Path,
               name: Optional[str] = None) -> Dict[str, Path]:
        """
        Export object to URDF with all supporting files.
        
        Args:
            obj: CompositeObject to export
            output_dir: Directory to write files
            name: Name for the object (uses obj.name if not provided)
        
        Returns:
            Dictionary mapping file types to their paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        name = name or obj.name
        
        # Create subdirectories
        meshes_dir = output_dir / "meshes"
        meshes_dir.mkdir(exist_ok=True)
        
        # Generate meshes. The union (boolean_union=True) yields a clean
        # watertight manifold; it falls back to concatenation if no CSG backend
        # is installed. Set union_visual_mesh=False for the v1 concatenation.
        visual_mesh = obj.to_mesh(boolean_union=self.config.union_visual_mesh)
        collision_mesh = self._prepare_collision_mesh(visual_mesh)
        
        # Compute physical properties
        inertia_method = "analytic_sum"
        if self.config.use_mesh_inertia:
            try:
                mass, inertia, com = obj.mesh_mass_properties(self.config.density)
                inertia_method = "union_mesh"
            except Exception:
                # Union not watertight — fall back to the analytic path. WARNING:
                # combined_inertia SUMS per-primitive volumes, double-counting any
                # overlap (up to ~15% mass error on heavily-overlapped composites).
                print(f"warning: {name}: boolean union failed — inertia falls back "
                      f"to the analytic overlap-summing path (mass may be "
                      f"over-estimated); recorded in metadata as 'analytic_sum'.")
                mass, inertia = obj.combined_inertia(self.config.density)
                com = obj.center_of_mass(self.config.density)
        else:
            mass, inertia = obj.combined_inertia(self.config.density)
            com = obj.center_of_mass(self.config.density)
        self._last_inertia_method = inertia_method
        
        # Save meshes
        ext = self.config.mesh_format
        visual_path = meshes_dir / f"{name}_visual.{ext}"
        collision_path = meshes_dir / f"{name}_collision.{ext}"
        
        # include_normals=True writes per-vertex normals (vn) — without them
        # gz/DART rejects the mesh as a collision shape ("normal count does not
        # match vertex count"), which is what forced the AABB-box workaround.
        visual_mesh.export(visual_path, include_normals=True)
        collision_mesh.export(collision_path, include_normals=True)
        
        # Collision decomposition: per-primitive (v2.9 default) or the legacy
        # single mesh. The collision-mesh OBJ above is always written — MoveIt
        # planning scenes load it from the manifest regardless of this choice.
        if self.config.per_primitive_collision:
            collisions = self._collision_entries(obj, name, meshes_dir, ext)
        else:
            collisions = [{"kind": "mesh", "uri": f"meshes/{name}_collision.{ext}",
                           "R": np.eye(3), "t": np.zeros(3)}]

        # Generate URDF
        urdf_path = output_dir / f"{name}.urdf"
        self._write_urdf(
            urdf_path, name, mass, inertia, com,
            f"meshes/{name}_visual.{ext}",
            collisions
        )

        # Generate SDF (Gazebo)
        sdf_path = output_dir / f"{name}.sdf"
        friction_val = getattr(obj, 'friction', self.config.friction_mu1)
        self._write_sdf(
            sdf_path, name, mass, inertia, com,
            f"meshes/{name}_visual.{ext}",
            collisions,
            friction_coeff=friction_val
        )
        
        # Generate metadata
        metadata_path = output_dir / f"{name}_metadata.yaml"
        self._write_metadata(
            metadata_path, obj, name, mass, inertia, com
        )
        
        return {
            'urdf': urdf_path,
            'sdf': sdf_path,
            'visual_mesh': visual_path,
            'collision_mesh': collision_path,
            'metadata': metadata_path
        }
    
    def _collision_entries(self, obj: CompositeObject, name: str,
                           meshes_dir: Path, ext: str) -> list:
        """Decompose the object into per-primitive collision entries.

        Native analytic shapes wherever they are exact:
          box / cylinder / sphere  -> native, posed by the primitive transform
          capsule                  -> cylinder + 2 spheres (native everywhere;
                                      SDF < 1.8 has no <capsule>)
        Everything else contributes its own WORLD-FRAME convex-hull mesh piece
        with identity pose — several primitive types recenter their meshes on
        the centroid, so reusing the already-placed mesh avoids any frame
        bookkeeping. Hull pieces are small, convex, and watertight. Non-convex
        types (torus, handle, hollow_shell, open_tube, gear, profile) lose their
        concavity in gz collision only; MoveIt keeps the true mesh.
        """
        entries = []
        for i, p in enumerate(obj.primitives):
            R = np.asarray(p.transform.rotation, dtype=float)
            t = np.asarray(p.transform.translation, dtype=float)
            k = p.ptype
            if k == PrimitiveType.BOX:
                entries.append({"kind": "box",
                                "size": np.asarray(p.dimensions, float),
                                "R": R, "t": t})
            elif k == PrimitiveType.CYLINDER:
                entries.append({"kind": "cylinder", "radius": float(p.radius),
                                "length": float(p.height), "R": R, "t": t})
            elif k == PrimitiveType.SPHERE:
                entries.append({"kind": "sphere", "radius": float(p.radius),
                                "R": R, "t": t})
            elif k == PrimitiveType.CAPSULE:
                entries.append({"kind": "cylinder", "radius": float(p.radius),
                                "length": float(p.height), "R": R, "t": t})
                for s in (-1.0, 1.0):
                    c = t + R @ np.array([0.0, 0.0, s * float(p.height) / 2.0])
                    entries.append({"kind": "sphere", "radius": float(p.radius),
                                    "R": np.eye(3), "t": c})
            else:
                piece = p.to_mesh()
                try:
                    piece = piece.convex_hull
                except Exception:
                    pass
                rel = f"meshes/{name}_col{i}.{ext}"
                piece.export(meshes_dir / f"{name}_col{i}.{ext}",
                             include_normals=True)
                entries.append({"kind": "mesh", "uri": rel,
                                "R": np.eye(3), "t": np.zeros(3)})
        return entries

    def _prepare_collision_mesh(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """Prepare simplified collision geometry."""
        if self.config.use_convex_hull:
            try:
                return mesh.convex_hull
            except Exception:
                pass
        
        if self.config.simplify_collision and len(mesh.faces) > self.config.max_collision_faces:
            try:
                # Simplify mesh
                simplified = mesh.simplify_quadric_decimation(
                    self.config.max_collision_faces
                )
                return simplified
            except Exception:
                pass
        
        return mesh
    
    def _write_urdf(self, path: Path, name: str, mass: float,
                    inertia: np.ndarray, com: np.ndarray,
                    visual_mesh_path: str, collisions: list):
        """Write URDF file."""
        robot = ET.Element('robot', name=name)
        
        # Single link (object is a rigid body)
        link = ET.SubElement(robot, 'link', name=f'{name}_link')
        
        # Inertial
        inertial = ET.SubElement(link, 'inertial')
        ET.SubElement(inertial, 'mass', value=f'{mass:.6f}')
        ET.SubElement(inertial, 'origin', 
                      xyz=f'{com[0]:.6f} {com[1]:.6f} {com[2]:.6f}',
                      rpy='0 0 0')
        
        # Inertia tensor (symmetric, so only 6 values needed)
        ET.SubElement(inertial, 'inertia',
                      ixx=f'{inertia[0,0]:.9f}',
                      ixy=f'{inertia[0,1]:.9f}',
                      ixz=f'{inertia[0,2]:.9f}',
                      iyy=f'{inertia[1,1]:.9f}',
                      iyz=f'{inertia[1,2]:.9f}',
                      izz=f'{inertia[2,2]:.9f}')
        
        # Visual
        visual = ET.SubElement(link, 'visual')
        vis_geom = ET.SubElement(visual, 'geometry')
        ET.SubElement(vis_geom, 'mesh', filename=visual_mesh_path)
        
        vis_material = ET.SubElement(visual, 'material', name='object_material')
        rgba = self.config.color_rgba
        ET.SubElement(vis_material, 'color', 
                      rgba=f'{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}')
        
        # Collision element(s) — one per entry (native shape or mesh piece).
        for c in collisions:
            collision = ET.SubElement(link, 'collision')
            xyz = c["t"]
            rpy = _rpy_from_matrix(c["R"])
            ET.SubElement(collision, 'origin',
                          xyz=f'{xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f}',
                          rpy=f'{rpy[0]:.6f} {rpy[1]:.6f} {rpy[2]:.6f}')
            col_geom = ET.SubElement(collision, 'geometry')
            if c["kind"] == "box":
                s = c["size"]
                ET.SubElement(col_geom, 'box', size=f'{s[0]:.6f} {s[1]:.6f} {s[2]:.6f}')
            elif c["kind"] == "cylinder":
                ET.SubElement(col_geom, 'cylinder',
                              radius=f'{c["radius"]:.6f}', length=f'{c["length"]:.6f}')
            elif c["kind"] == "sphere":
                ET.SubElement(col_geom, 'sphere', radius=f'{c["radius"]:.6f}')
            else:
                ET.SubElement(col_geom, 'mesh', filename=c["uri"])

        # Write formatted XML
        xml_str = minidom.parseString(ET.tostring(robot)).toprettyxml(indent='  ')
        # Remove extra blank lines
        xml_str = '\n'.join([line for line in xml_str.split('\n') if line.strip()])
        
        with open(path, 'w') as f:
            f.write(xml_str)
    
    def _write_sdf(self, path: Path, name: str, mass: float,
                   inertia: np.ndarray, com: np.ndarray,
                   visual_mesh_path: str, collisions: list,
                   friction_coeff: float = 0.8):
        """Write SDF file (Gazebo format)."""
        sdf = ET.Element('sdf', version='1.7')
        model = ET.SubElement(sdf, 'model', name=name)
        
        # Static = False (object can move)
        ET.SubElement(model, 'static').text = 'false'
        
        link = ET.SubElement(model, 'link', name=f'{name}_link')
        
        # Inertial
        inertial = ET.SubElement(link, 'inertial')
        ET.SubElement(inertial, 'mass').text = f'{mass:.6f}'
        
        pose = ET.SubElement(inertial, 'pose')
        pose.text = f'{com[0]:.6f} {com[1]:.6f} {com[2]:.6f} 0 0 0'
        
        inertia_elem = ET.SubElement(inertial, 'inertia')
        ET.SubElement(inertia_elem, 'ixx').text = f'{inertia[0,0]:.9f}'
        ET.SubElement(inertia_elem, 'ixy').text = f'{inertia[0,1]:.9f}'
        ET.SubElement(inertia_elem, 'ixz').text = f'{inertia[0,2]:.9f}'
        ET.SubElement(inertia_elem, 'iyy').text = f'{inertia[1,1]:.9f}'
        ET.SubElement(inertia_elem, 'iyz').text = f'{inertia[1,2]:.9f}'
        ET.SubElement(inertia_elem, 'izz').text = f'{inertia[2,2]:.9f}'
        
        # Visual
        visual = ET.SubElement(link, 'visual', name=f'{name}_visual')
        vis_geom = ET.SubElement(visual, 'geometry')
        vis_mesh = ET.SubElement(vis_geom, 'mesh')
        ET.SubElement(vis_mesh, 'uri').text = visual_mesh_path
        
        vis_material = ET.SubElement(visual, 'material')
        diffuse = ET.SubElement(vis_material, 'diffuse')
        rgba = self.config.color_rgba
        diffuse.text = f'{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}'
        
        # Collision element(s) — one per entry, each with its own surface block
        # (patch_sdf_collision substitutes every <mu> and prepends <contact> to
        # every <friction>, so the patch applies to all of them).
        for j, c in enumerate(collisions):
            collision = ET.SubElement(link, 'collision', name=f'{name}_collision_{j}')
            xyz = c["t"]
            rpy = _rpy_from_matrix(c["R"])
            ET.SubElement(collision, 'pose').text = (
                f'{xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f} '
                f'{rpy[0]:.6f} {rpy[1]:.6f} {rpy[2]:.6f}')
            col_geom = ET.SubElement(collision, 'geometry')
            if c["kind"] == "box":
                s = c["size"]
                box = ET.SubElement(col_geom, 'box')
                ET.SubElement(box, 'size').text = f'{s[0]:.6f} {s[1]:.6f} {s[2]:.6f}'
            elif c["kind"] == "cylinder":
                cyl = ET.SubElement(col_geom, 'cylinder')
                ET.SubElement(cyl, 'radius').text = f'{c["radius"]:.6f}'
                ET.SubElement(cyl, 'length').text = f'{c["length"]:.6f}'
            elif c["kind"] == "sphere":
                sph = ET.SubElement(col_geom, 'sphere')
                ET.SubElement(sph, 'radius').text = f'{c["radius"]:.6f}'
            else:
                col_mesh = ET.SubElement(col_geom, 'mesh')
                ET.SubElement(col_mesh, 'uri').text = c["uri"]

            surface = ET.SubElement(collision, 'surface')
            friction = ET.SubElement(surface, 'friction')
            ode = ET.SubElement(friction, 'ode')
            ET.SubElement(ode, 'mu').text = f'{friction_coeff}'
            ET.SubElement(ode, 'mu2').text = f'{friction_coeff}'

            bounce = ET.SubElement(surface, 'bounce')
            ET.SubElement(bounce, 'restitution_coefficient').text = f'{self.config.restitution}'
        
        # Write formatted XML
        xml_str = minidom.parseString(ET.tostring(sdf)).toprettyxml(indent='  ')
        xml_str = '\n'.join([line for line in xml_str.split('\n') if line.strip()])
        
        with open(path, 'w') as f:
            f.write(xml_str)
    
    def _write_metadata(self, path: Path, obj: CompositeObject, name: str,
                        mass: float, inertia: np.ndarray, com: np.ndarray):
        """Write metadata YAML file."""
        extents = obj.aabb_extents()
        
        metadata = {
            'name': name,
            'n_primitives': len(obj.primitives),
            'primitives': [
                {
                    'type': p.ptype.value,
                    'position': p.transform.translation.tolist()
                }
                for p in obj.primitives
            ],
            'physical_properties': {
                'mass_kg': float(mass),
                'center_of_mass': com.tolist(),
                'inertia_tensor': inertia.tolist(),
                'density_kg_m3': self.config.density,
                # 'union_mesh' = overlap-aware (correct); 'analytic_sum' = per-primitive
                # sum that double-counts overlaps (fallback when the union fails).
                'inertia_method': getattr(self, '_last_inertia_method', 'analytic_sum'),
            },
            'geometry': {
                'aabb_extents': extents.tolist(),
                'approximate_volume_m3': float(obj.total_volume())
            },
            'surface_properties': {
                'friction_mu': getattr(obj, 'friction', self.config.friction_mu1),
                'restitution': self.config.restitution
            }
        }
        
        with open(path, 'w') as f:
            yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)


class BatchExporter:
    """Export multiple objects efficiently."""
    
    def __init__(self, config: ExportConfig = None):
        self.exporter = URDFExporter(config)
    
    def export_batch(self, 
                     objects: list,
                     output_dir: Path,
                     name_prefix: str = "object") -> Dict[str, list]:
        """
        Export a batch of objects.
        
        Args:
            objects: List of CompositeObjects
            output_dir: Base directory for all exports
            name_prefix: Prefix for object names
        
        Returns:
            Dictionary with lists of generated file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {
            'urdf': [],
            'sdf': [],
            'visual_mesh': [],
            'collision_mesh': [],
            'metadata': []
        }
        
        for i, obj in enumerate(objects):
            name = f"{name_prefix}_{i:04d}"
            obj_dir = output_dir / name
            
            try:
                paths = self.exporter.export(obj, obj_dir, name)
                for key, path in paths.items():
                    results[key].append(path)
            except Exception as e:
                print(f"Warning: Failed to export {name}: {e}")
        
        # Write manifest
        manifest_path = output_dir / "manifest.yaml"
        manifest = {
            'n_objects': len(results['urdf']),
            'objects': [
                {
                    'name': f"{name_prefix}_{i:04d}",
                    'urdf': str(results['urdf'][i].relative_to(output_dir)) if i < len(results['urdf']) else None
                }
                for i in range(len(objects))
            ]
        }
        
        with open(manifest_path, 'w') as f:
            yaml.dump(manifest, f, default_flow_style=False)
        
        return results


def export_single(obj: CompositeObject, 
                  output_dir: str,
                  name: str = None) -> Dict[str, Path]:
    """Convenience function to export a single object."""
    exporter = URDFExporter()
    return exporter.export(obj, Path(output_dir), name)
