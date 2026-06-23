"""Model loading for the survey-item agent.

Handles HuggingFace Hub downloads (laptop) and local-path loading (LRZ GPU).
"""

import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_name: str):
    """Load tokenizer + model. Local path on LRZ; HF Hub download locally."""
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if sys.platform == "darwin" and torch.backends.mps.is_available():
        # Avoid device_map="auto" on macOS because it triggers slow
        # CPU/disk offloading on 8GB Macs.
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16
        ).to("mps")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto"
        )
    print(f"Model loaded on device: {model.device}")
    return tokenizer, model
