"""
Constraint-Based Scoring for Robotic Manipulation.

Scores objects on robotics-relevant criteria without requiring labeled data:
- Size feasibility (fits gripper workspace)
- Static stability (won't topple)
- Graspability proxy (antipodal surface pairs)
- Mesh validity and complexity
"""

import numpy as np
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass
import trimesh
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist

from primitives import CompositeObject


@dataclass
class ScoringConfig:
    """Configuration for scoring thresholds."""
    # Size constraints (meters)
    min_extent: float = 0.02      # Minimum dimension
    max_extent: float = 0.15      # Maximum dimension
    
    # Gripper constraints
    gripper_width_min: float = 0.01   # Minimum grasp width
    gripper_width_max: float = 0.08   # Maximum gripper opening
    
    # Stability
    min_stability_margin: float = 0.005  # 5mm margin to support boundary
    
    # Graspability
    n_surface_samples: int = 200      # Points to sample on surface
    normal_alignment_thresh: float = 0.9  # cos(θ) threshold for antipodal
    
    # Mesh complexity
    max_triangles: int = 5000
    min_thickness: float = 0.005  # 5mm minimum feature thickness
    
    # Density for physics
    density: float = 1000.0  # kg/m³ (like water/plastic)


@dataclass
class ScoreBreakdown:
    """Detailed breakdown of all score components."""
    size_score: float = 0.0
    stability_score: float = 0.0
    graspability_score: float = 0.0
    complexity_score: float = 0.0
    validity_score: float = 0.0
    total_score: float = 0.0
    
    # Diagnostic info
    aabb_extents: np.ndarray = None
    stability_margin: float = 0.0
    n_antipodal_pairs: int = 0
    n_triangles: int = 0
    is_watertight: bool = False
    
    def to_dict(self) -> Dict:
        return {
            'size_score': self.size_score,
            'stability_score': self.stability_score,
            'graspability_score': self.graspability_score,
            'complexity_score': self.complexity_score,
            'validity_score': self.validity_score,
            'total_score': self.total_score,
            'aabb_extents': self.aabb_extents.tolist() if self.aabb_extents is not None else None,
            'stability_margin': self.stability_margin,
            'n_antipodal_pairs': self.n_antipodal_pairs,
            'n_triangles': self.n_triangles,
            'is_watertight': self.is_watertight
        }


class ObjectScorer:
    """
    Scores CompositeObjects for robotic manipulation suitability.
    All scores are in [0, 1] range.
    """
    
    def __init__(self, config: ScoringConfig = None):
        self.config = config or ScoringConfig()
    
    def score(self, obj: CompositeObject) -> ScoreBreakdown:
        """Compute full score breakdown for an object."""
        result = ScoreBreakdown()
        
        # Generate mesh (needed for most checks)
        try:
            mesh = obj.to_mesh(boolean_union=False)  # Fast version
        except Exception as e:
            # Mesh generation failed
            result.validity_score = 0.0
            result.total_score = 0.0
            return result
        
        if len(mesh.vertices) == 0:
            result.total_score = 0.0
            return result
        
        # Compute individual scores
        result.size_score, result.aabb_extents = self._score_size(obj)
        result.stability_score, result.stability_margin = self._score_stability(mesh, obj)
        result.graspability_score, result.n_antipodal_pairs = self._score_graspability(mesh)
        result.complexity_score, result.n_triangles = self._score_complexity(mesh)
        result.validity_score, result.is_watertight = self._score_validity(mesh)
        
        # Weighted combination
        # Weights chosen to prioritize: validity > stability > graspability > size > complexity
        weights = {
            'validity': 2.0,
            'stability': 1.5,
            'graspability': 1.5,
            'size': 1.0,
            'complexity': 0.5
        }
        
        total_weight = sum(weights.values())
        result.total_score = (
            weights['validity'] * result.validity_score +
            weights['stability'] * result.stability_score +
            weights['graspability'] * result.graspability_score +
            weights['size'] * result.size_score +
            weights['complexity'] * result.complexity_score
        ) / total_weight
        
        return result
    
    def _score_size(self, obj: CompositeObject) -> Tuple[float, np.ndarray]:
        """
        Score based on AABB fitting gripper workspace.
        
        Objects should be:
        - Not too small (hard to grasp precisely)
        - Not too large (exceeds gripper/workspace)
        """
        extents = obj.aabb_extents()
        
        # Check each dimension
        scores = []
        for ext in extents:
            if ext < self.config.min_extent:
                # Too small - linear penalty
                s = ext / self.config.min_extent
            elif ext > self.config.max_extent:
                # Too large - linear penalty
                s = self.config.max_extent / ext
            else:
                # Good range
                s = 1.0
            scores.append(s)
        
        return float(np.mean(scores)), extents
    
    def _score_stability(self, mesh: trimesh.Trimesh, obj: CompositeObject) -> Tuple[float, float]:
        """
        Score based on static stability on a flat surface.
        
        Computes:
        1. Support polygon (convex hull of ground-contact vertices)
        2. COM projection onto ground plane
        3. Margin from COM to support polygon boundary
        """
        # Find vertices near the ground plane (z ≈ z_min). The tolerance scales
        # with object height so that rounded/pointed bottoms (spheres, cones,
        # ellipsoids) still yield a contact patch of >=3 vertices instead of
        # being auto-scored 0 by a fixed 2 mm band (DISCREPANCIES.md item 5).
        vertices = mesh.vertices
        z_min = vertices[:, 2].min()
        z_max = vertices[:, 2].max()
        extent_z = max(z_max - z_min, 1e-6)
        ground_thresh = z_min + max(0.002, 0.03 * extent_z)

        ground_vertices = vertices[vertices[:, 2] < ground_thresh]
        
        if len(ground_vertices) < 3:
            # Not enough support points
            return 0.0, 0.0
        
        # Compute support polygon (2D convex hull in XY)
        try:
            support_2d = ground_vertices[:, :2]
            if len(np.unique(support_2d, axis=0)) < 3:
                return 0.0, 0.0
            hull = ConvexHull(support_2d)
            hull_points = support_2d[hull.vertices]
        except Exception:
            return 0.0, 0.0
        
        # Get COM projection
        com = obj.center_of_mass(self.config.density)
        com_2d = com[:2]
        
        # Compute distance from COM to hull boundary
        margin = self._point_to_polygon_margin(com_2d, hull_points)
        
        if margin < 0:
            # COM outside support polygon - unstable
            return 0.0, margin
        
        # Score based on margin
        # Good stability: margin > min_stability_margin
        score = min(1.0, margin / (self.config.min_stability_margin * 2))
        
        return score, margin
    
    def _point_to_polygon_margin(self, point: np.ndarray, polygon: np.ndarray) -> float:
        """
        Compute signed distance from point to polygon boundary.
        Positive = inside, Negative = outside.
        """
        # Check if point is inside polygon
        n = len(polygon)
        inside = True
        min_dist = float('inf')
        
        for i in range(n):
            p1 = polygon[i]
            p2 = polygon[(i + 1) % n]
            
            # Edge vector and normal
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            normal = normal / (np.linalg.norm(normal) + 1e-10)
            
            # Check which side of edge
            to_point = point - p1
            side = np.dot(to_point, normal)
            
            if side < 0:
                inside = False
            
            # Distance to edge
            t = np.clip(np.dot(to_point, edge) / (np.dot(edge, edge) + 1e-10), 0, 1)
            closest = p1 + t * edge
            dist = np.linalg.norm(point - closest)
            min_dist = min(min_dist, dist)
        
        return min_dist if inside else -min_dist
    
    def _score_graspability(self, mesh: trimesh.Trimesh) -> Tuple[float, int]:
        """
        Score based on antipodal grasp opportunities.
        
        Samples surface points and counts pairs that satisfy:
        1. Nearly opposite normals (dot product < -threshold)
        2. Distance within gripper width range
        
        This is a proxy for parallel-jaw graspability without needing a grasp database.
        """
        # Sample points on surface with normals
        try:
            points, face_idx = trimesh.sample.sample_surface(
                mesh, self.config.n_surface_samples
            )
            normals = mesh.face_normals[face_idx]
        except Exception:
            return 0.0, 0
        
        if len(points) < 10:
            return 0.0, 0
        
        # Compute pairwise distances
        distances = cdist(points, points)
        
        # Compute pairwise normal alignment
        # For antipodal: n1 · n2 ≈ -1
        normal_dots = normals @ normals.T
        
        # Soft alignment score: map [-1, -0.9] to [1, 0] roughly
        # We want alignment < -0.9.
        # Let's use sim = -dot. We want sim > 0.9.
        # Score = clip((sim - 0.5) / 0.4, 0, 1) ? No, let's keep it simple.
        # Continuous pair score:
        
        sim = -normal_dots
        alignment_score = np.clip((sim - 0.8) / 0.2, 0, 1) # 0.8->0, 1.0->1
        
        # Distance score
        # ideal: [min, max]
        d = distances
        d_min, d_max = self.config.gripper_width_min, self.config.gripper_width_max
        
        # Vectorized distance scoring
        # If d < min: d/min
        # If d > max: max/d
        # Else: 1.0
        dist_score = np.ones_like(d)
        mask_small = d < d_min
        mask_large = d > d_max
        
        # Avoid div by zero
        safe_d = np.maximum(d, 1e-6)
        
        dist_score[mask_small] = safe_d[mask_small] / d_min
        dist_score[mask_large] = d_max / safe_d[mask_large]
        
        # Pairwise quality
        pair_quality = alignment_score * dist_score
        
        # Zero out diagonal (self-pairs) and lower triangle
        np.fill_diagonal(pair_quality, 0)
        pair_quality = np.triu(pair_quality)
        
        # Total quality is average of top N best pairs? 
        # Or sum normalized by expected?
        # Let's stick to "sum of quality" normalized.
        
        total_quality = np.sum(pair_quality)
        expected_quality = 30.0 # Same baseline as before
        
        score = min(1.0, total_quality / expected_quality)
        n_pairs = int(np.sum(pair_quality > 0.9)) # Count high quality ones for stats
        
        return score, n_pairs
    
    def _score_complexity(self, mesh: trimesh.Trimesh) -> Tuple[float, int]:
        """
        Score based on mesh complexity.
        
        Penalizes very complex meshes that slow down collision checking.
        """
        n_triangles = len(mesh.faces)
        
        if n_triangles > self.config.max_triangles:
            score = self.config.max_triangles / n_triangles
        else:
            score = 1.0
        
        return score, n_triangles
    
    def _score_validity(self, mesh: trimesh.Trimesh) -> Tuple[float, bool]:
        """
        Score based on mesh validity.
        
        Checks:
        - Watertight (closed surface)
        - No degenerate triangles
        - Reasonable volume
        """
        score = 1.0
        
        # Check watertight
        is_watertight = mesh.is_watertight
        if not is_watertight:
            score *= 0.7  # Penalty but not fatal
        
        # Check for degenerate faces
        areas = mesh.area_faces
        n_degenerate = np.sum(areas < 1e-10)
        if n_degenerate > 0:
            score *= max(0.5, 1 - n_degenerate / len(areas))
        
        # Check volume is positive and reasonable
        try:
            if mesh.is_watertight:
                vol = mesh.volume
                if vol < 1e-9:  # Essentially zero volume
                    score *= 0.3
        except Exception:
            pass
        
        return score, is_watertight


def quick_score(obj: CompositeObject, config: ScoringConfig = None) -> float:
    """Convenience function to get just the total score."""
    scorer = ObjectScorer(config)
    return scorer.score(obj).total_score


def is_valid_for_manipulation(obj: CompositeObject, min_score: float = 0.5) -> bool:
    """Check if object meets minimum threshold for manipulation."""
    return quick_score(obj) >= min_score
