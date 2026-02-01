
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Callable

@dataclass
class ParamConfig:
    """Design space for a single parameter."""
    name: str
    mean: float
    std: float
    min_val: float
    max_val: float

@dataclass
class ArchetypeDefinition:
    """Configuration for a learnable archetype."""
    name: str
    factory_fn_name: str
    params: Dict[str, ParamConfig]

# Define the learnable parameters for each archetype
ARCHETYPE_CONFIGS = {
    "simple_box": ArchetypeDefinition(
        name="simple_box",
        factory_fn_name="create_simple_box",
        params={
            "dims": ParamConfig("dims", 0.05, 0.02, 0.02, 0.15) # Treated as scalar for cubic, or vector?
        }
    ),
    "t_shape": ArchetypeDefinition(
        name="t_shape",
        factory_fn_name="create_t_shape",
        params={
            # We'll learn independent dimensions for H and V parts
            # Parameters map to vectors in the factory, we'll unpack them
            "h_dim_x": ParamConfig("h_dim_x", 0.10, 0.03, 0.04, 0.15),
            "h_dim_y": ParamConfig("h_dim_y", 0.03, 0.01, 0.015, 0.06),
            "h_dim_z": ParamConfig("h_dim_z", 0.03, 0.01, 0.015, 0.06),
            "v_dim_x": ParamConfig("v_dim_x", 0.03, 0.01, 0.015, 0.06),
            "v_dim_y": ParamConfig("v_dim_y", 0.03, 0.01, 0.015, 0.06),
            "v_dim_z": ParamConfig("v_dim_z", 0.08, 0.03, 0.03, 0.15),
        }
    ),
    "dumbbell": ArchetypeDefinition(
        name="dumbbell",
        factory_fn_name="create_dumbbell",
        params={
            "handle_length": ParamConfig("handle_length", 0.08, 0.02, 0.04, 0.15),
            "handle_radius": ParamConfig("handle_radius", 0.01, 0.003, 0.005, 0.02),
            "weight_radius": ParamConfig("weight_radius", 0.03, 0.01, 0.015, 0.06),
        }
    ),
    # Add definitions for others... generalized for brevity in this file
}

# Helper to expand the config completely or dynamically?
# For the sake of the user request "10000 from EACH archetype", I should try to support all.
# But defining 20 configs manually is error prone.
# I will implement a Generic Wrapper that inspects the function or uses a default noise model.
