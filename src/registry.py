"""Model registry — single source of truth for --size → model-name mapping.

Kept in a separate module so that the evaluator can import it without
triggering the transformers-dependent model loader in src.agent.model.
"""

MODEL_REGISTRY: dict[str, str] = {
    "small":  "Qwen/Qwen2.5-1.5B-Instruct",
    "big":    "Qwen/Qwen2.5-7B-Instruct",
    "large":  "Qwen/Qwen2.5-14B-Instruct",
}

# Default / backward-compat alias.
DEFAULT_SIZE = "small"
