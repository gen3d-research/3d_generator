"""Shared constants for the evaluation nodes (single source of truth).

WORLD_NAME was previously hardcoded in three modules; renaming the gz world silently
broke whichever copy you forgot. Override with the GZ_EVAL_WORLD environment variable.
"""
import os

WORLD_NAME = os.environ.get("GZ_EVAL_WORLD", "panda_eval_world")
