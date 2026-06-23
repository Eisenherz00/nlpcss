"""Single-item generation: inference + JSON extraction.

Contains the low-level functions that call the model for one assertion
and parse its output into a Python dict.
"""

import json
import re

from src.agent.prompts import SYSTEM_PROMPT


def _extract_json(text: str) -> dict:
    """Pull a JSON object out of the model's raw text output."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        candidate = fenced.group(1)
    else:
        brace = re.search(r"\{.*\}", text, re.DOTALL)
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
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs, max_new_tokens=512, do_sample=True, temperature=0.7, top_p=0.9
    )
    input_length = inputs["input_ids"].shape[1]
    return tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)


def generate_item(assertion: str, tokenizer, model) -> dict:
    """Generate one survey-item JSON, with a single retry on parse failure."""
    answer = _run(assertion, tokenizer, model)
    try:
        return _extract_json(answer)
    except (json.JSONDecodeError, ValueError):
        answer = _run(assertion, tokenizer, model, suffix="Output JSON only, no markdown.")
        return _extract_json(answer)
