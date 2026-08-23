"""Single-item generation: inference + JSON extraction.

Contains the low-level functions that call the model for one assertion
and parse its output into a Python dict.
"""

import json
import re
import sys

from src.agent.prompts import SYSTEM_PROMPT



def _extract_json(text: str) -> dict:
    """Pull a JSON object out of the model's raw text output."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        candidate = fenced.group(1)
    else:
        brace = re.search(r"\{.*?\}", text, re.DOTALL)
        candidate = brace.group(0) if brace else text
    return json.loads(candidate)


def _run(assertion: str, tokenizer, model, suffix: str = "") -> str:
    """Run a single inference call and return the raw decoded text."""
    user_message = f'Assertion: "{assertion}"\nOutput:'
    if suffix:
        user_message += f"\n{suffix}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    # macOS (MLX 架构)
    if sys.platform == "darwin":
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        tok = getattr(tokenizer, "_tokenizer", tokenizer)
        if hasattr(tok, "apply_chat_template"):
            prompt_str = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt_str = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        sampler = make_sampler(temp=0.7, top_p=0.9)
        return generate(
            model,
            tokenizer,
            prompt=prompt_str,
            max_tokens=512,
            sampler=sampler,
            verbose=False,
        )


def generate_item(assertion: str, tokenizer, model) -> dict:
    """Generate one survey-item JSON, with a single retry on parse failure."""
    answer = _run(assertion, tokenizer, model)
    try:
        return _extract_json(answer)
    except (json.JSONDecodeError, ValueError):
        answer = _run(
            assertion,
            tokenizer,
            model,
            suffix="Output JSON only, no markdown.",
        )
        return _extract_json(answer)