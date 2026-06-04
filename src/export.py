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

from primitives import CompositeObject


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
        if self.config.use_mesh_inertia:
            try:
                mass, inertia, com = obj.mesh_mass_properties(self.config.density)
            except Exception:
                # Union not watertight — fall back to the analytic path.
                mass, inertia = obj.combined_inertia(self.config.density)
                com = obj.center_of_mass(self.config.density)
        else:
            mass, inertia = obj.combined_inertia(self.config.density)
            com = obj.center_of_mass(self.config.density)
        
        # Save meshes
        ext = self.config.mesh_format
        visual_path = meshes_dir / f"{name}_visual.{ext}"
        collision_path = meshes_dir / f"{name}_collision.{ext}"
        
        # include_normals=True writes per-vertex normals (vn) — without them
        # gz/DART rejects the mesh as a collision shape ("normal count does not
        # match vertex count"), which is what forced the AABB-box workaround.
        visual_mesh.export(visual_path, include_normals=True)
        collision_mesh.export(collision_path, include_normals=True)
        
        # Generate URDF
        urdf_path = output_dir / f"{name}.urdf"
        self._write_urdf(
            urdf_path, name, mass, inertia, com,
            f"meshes/{name}_visual.{ext}",
            f"meshes/{name}_collision.{ext}"
        )
        
        # Generate SDF (Gazebo)
        sdf_path = output_dir / f"{name}.sdf"
        friction_val = getattr(obj, 'friction', self.config.friction_mu1)
        self._write_sdf(
            sdf_path, name, mass, inertia, com,
            f"meshes/{name}_visual.{ext}",
            f"meshes/{name}_collision.{ext}",
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
                    visual_mesh_path: str, collision_mesh_path: str):
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
        
        # Collision
        collision = ET.SubElement(link, 'collision')
        col_geom = ET.SubElement(collision, 'geometry')
        ET.SubElement(col_geom, 'mesh', filename=collision_mesh_path)
        
        # Write formatted XML
        xml_str = minidom.parseString(ET.tostring(robot)).toprettyxml(indent='  ')
        # Remove extra blank lines
        xml_str = '\n'.join([line for line in xml_str.split('\n') if line.strip()])
        
        with open(path, 'w') as f:
            f.write(xml_str)
    
    def _write_sdf(self, path: Path, name: str, mass: float,
                   inertia: np.ndarray, com: np.ndarray,
                   visual_mesh_path: str, collision_mesh_path: str,
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
        
        # Collision
        collision = ET.SubElement(link, 'collision', name=f'{name}_collision')
        col_geom = ET.SubElement(collision, 'geometry')
        col_mesh = ET.SubElement(col_geom, 'mesh')
        ET.SubElement(col_mesh, 'uri').text = collision_mesh_path
        
        # Surface properties
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
                'density_kg_m3': self.config.density
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
