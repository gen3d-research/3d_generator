"""Point ⑫ — text-conditioned generation (the text2geometry front-end).

Map a natural-language prompt to a generation config — archetype seed (⑧), primitive
palette (⑨), target size (⑪), and stability / graspability gates (⑦) — then hand off to the
free CEM (④). No neural model: keyword rules turn the existing optimizer into a text decoder.

    objs, intent = generate_from_text("a small stable graspable curved bottle")
"""
import re

import numpy as np

from generator import RoboticObjectGenerator, GeneratorConfig
from archetypes import ARCHETYPE_REGISTRY

CURVED = ["cylinder", "sphere", "capsule", "cone", "torus", "ellipsoid",
          "hollow_shell", "handle", "frustum", "hemisphere"]
FACETED = ["box", "pyramid", "wedge", "hex_prism", "ngon_prism",
           "rounded_box", "gear_prism", "extruded_profile"]
_SIZE_WORDS = {"tiny": 0.03, "small": 0.04, "medium": 0.07,
               "large": 0.11, "big": 0.11, "huge": 0.14}


def parse_prompt(prompt: str) -> dict:
    """Parse a prompt into generation intents (seed / palette / target_extent / gates)."""
    p = prompt.lower()
    intent = {"seed": None, "palette": None, "target_extent": None,
              "stable": False, "graspable": False}

    # Archetype seed: longest registry name mentioned (prefer 'wine_bottle' over 'bottle').
    for name in sorted(ARCHETYPE_REGISTRY, key=len, reverse=True):
        if re.search(r"\b" + re.escape(name.replace("_", " ")) + r"\b", p):
            intent["seed"] = name
            break

    # Primitive palette.
    if any(w in p for w in ["curved", "round", "rounded", "smooth"]):
        intent["palette"] = CURVED
    elif any(w in p for w in ["faceted", "boxy", "angular", "flat-sided"]):
        intent["palette"] = FACETED

    # Target size: explicit "N cm/mm/m", else a size word.
    m = re.search(r"(\d+(?:\.\d+)?)\s*(cm|mm|m)\b", p)
    if m:
        intent["target_extent"] = float(m.group(1)) * {"cm": 0.01, "mm": 0.001, "m": 1.0}[m.group(2)]
    else:
        for w, v in _SIZE_WORDS.items():
            if re.search(r"\b" + w + r"\b", p):
                intent["target_extent"] = v
                break

    # Objectives.
    if any(w in p for w in ["stable", "stand", "upright", "steady", "balanced"]):
        intent["stable"] = True
    if any(w in p for w in ["graspable", "grasp", "grip", "pick", "hold"]):
        intent["graspable"] = True
    return intent


def text_to_generator(prompt: str, **cfg_overrides):
    """Build a (RoboticObjectGenerator, intent) configured from `prompt`."""
    it = parse_prompt(prompt)
    cfg = GeneratorConfig(
        dynamic_stability_gate=it["stable"],
        repair_stability=it["stable"],
        low_grasp_gate=True if it["graspable"] else cfg_overrides.pop("low_grasp_gate", True),
        target_extent=it["target_extent"],
        **cfg_overrides,
    )
    g = RoboticObjectGenerator(cfg)
    if it["seed"]:
        g.seed_from(it["seed"])
    if it["palette"]:
        g.constrain_types(it["palette"])
    return g, it


def generate_from_text(prompt: str, n: int = 8, train: bool = True, **cfg):
    """One-shot: prompt -> trained generator -> n objects (+ the parsed intent)."""
    g, it = text_to_generator(prompt, **cfg)
    if train:
        g.train(verbose=False)
    return g.generate(n), it
