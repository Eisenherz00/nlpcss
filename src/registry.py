"""Model registry — single source of truth for --size → model-name mapping.

Kept in a separate module so that the evaluator can import it without
triggering the transformers-dependent model loader in src.agent.model.
"""

MODEL_REGISTRY: dict[str, str] = {
    "small":  "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
    "big":    "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "large":  "mlx-community/Qwen2.5-32B-Instruct-4bit",
}

# Default / backward-compat alias.
DEFAULT_SIZE = "small"
