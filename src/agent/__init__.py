"""src.agent — Assertion → closed survey item generator.

Submodules:
    prompts     SYSTEM_PROMPT and model defaults
    model       Model loading (HF Hub / local path)
    generation  Single-item inference and JSON extraction
    pipeline    Batch processing over all assertions
"""

from src.agent.prompts import LOCAL_MODEL, MODEL_REGISTRY, SYSTEM_PROMPT
from src.agent.model import load_model
from src.agent.generation import generate_item
from src.agent.pipeline import run_all

__all__ = ["LOCAL_MODEL", "MODEL_REGISTRY", "SYSTEM_PROMPT", "load_model", "generate_item", "run_all"]
