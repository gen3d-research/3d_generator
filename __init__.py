"""
Robotic Object Generator

Generative 3D object modeling for robust robot manipulation in ROS 2.
"""

from .primitives import (
    CompositeObject,
    Box,
    Cylinder,
    Sphere,
    Capsule,
    Transform,
    create_simple_box,
    create_mug_like,
    create_l_shape
)

from .scoring import (
    ObjectScorer,
    ScoreBreakdown,
    ScoringConfig,
    quick_score,
    is_valid_for_manipulation
)

from .cem import (
    CEMOptimizer,
    CEMConfig,
    ParameterDistribution,
    train_generator
)

from .export import (
    URDFExporter,
    BatchExporter,
    ExportConfig,
    export_single
)

from .generator import (
    RoboticObjectGenerator,
    GeneratorConfig,
    quick_generate,
    generate_and_export,
    create_archetype_set
)

__version__ = "0.1.0"
__all__ = [
    # Primitives
    "CompositeObject",
    "Box",
    "Cylinder", 
    "Sphere",
    "Capsule",
    "Transform",
    "create_simple_box",
    "create_mug_like",
    "create_l_shape",
    
    # Scoring
    "ObjectScorer",
    "ScoreBreakdown",
    "ScoringConfig",
    "quick_score",
    "is_valid_for_manipulation",
    
    # CEM
    "CEMOptimizer",
    "CEMConfig",
    "ParameterDistribution",
    "train_generator",
    
    # Export
    "URDFExporter",
    "BatchExporter",
    "ExportConfig",
    "export_single",
    
    # Generator
    "RoboticObjectGenerator",
    "GeneratorConfig",
    "quick_generate",
    "generate_and_export",
    "create_archetype_set",
]
