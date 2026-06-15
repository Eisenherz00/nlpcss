import json
import re
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer

# === MODEL CONFIG ===
# Local development (auto-downloads ~3GB from HuggingFace on first run)
# MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

# LRZ deployment (uncomment when running on LRZ GPU cluster):
MODEL_NAME = "/dss/dssmcmlfs01/pn25ju/pn25ju-dss-0000/models/Qwen3.5-9B"
# ====================

SYSTEM_PROMPT = """You are a survey methodology expert following Saris & Gallhofer (2014) conventions.
Given a declarative assertion, you produce a complete CLOSED survey item (a question
plus an ordered, labeled response scale) and return it as a single JSON object.

Return ONLY a JSON object with EXACTLY these keys:
- "assertion":          the original assertion string
- "basic_concept":      one of "feeling", "evaluation", "importance", "cognition", "behavior", "norm", "right"
- "question_text":      the survey question (must be a closed question, not open-ended)
- "format_type":        one of "direct_interrogative", "direct_imperative", "indirect_interrogative", "indirect_imperative"
- "scale_type":         "unipolar" or "bipolar"
- "n_points":           integer between 5 and 11 (default 5)
- "labels":             list of text labels, length must equal n_points, MONOTONICALLY ordered
                        (either from most-negative-end to most-positive-end, or the reverse — never shuffled)
- "polarity_reason":    one short sentence justifying scale_type given the basic_concept

HARD RULES (your output WILL be checked against each of these):

[Wording rules]
W1. Question format MUST be one of the four formats listed in format_type. Prefer "direct_interrogative" unless the assertion strongly suggests otherwise.
    - direct_interrogative:   "How satisfied are you with your job?"
    - direct_imperative:      "Indicate your satisfaction with your job on a scale from 0 to 10."
    - indirect_interrogative: "Do you feel that you are satisfied with your job?"
    - indirect_imperative:    "Please tell me how satisfied you are with your job."
W2. No LEADING wording. Forbidden patterns: "Don't you agree...", "Wouldn't you say...", "Isn't it true that...", "Don't you think...".
W3. No LOADED presuppositions. Do not assume facts not stated in the assertion.
W4. No VAGUE frequency terms in labels (e.g. "often", "sometimes", "rarely", "frequently", "regularly", "occasionally"). Labels must directly reflect the concept being measured.
W5. The question must be CLOSED (answerable on the provided scale). Never produce open-ended "What/Why" questions that require a free-text reply.

[Scale rules]
S1. NO agree-disagree scales. Never use labels like "Strongly agree", "Agree", "Disagree", "Strongly disagree". Use item-specific labels that name the concept directly (e.g. "Not at all satisfied" ... "Completely satisfied").
S2. Labels MUST be balanced:
    - For bipolar scales: equal number of negative and positive labels around an optional midpoint (e.g. 2 neg + 1 mid + 2 pos for 5 points; 3 neg + 1 mid + 3 pos for 7 points).
    - For unipolar scales: labels evenly graded from zero/low to maximum (e.g. "Not at all" ... "Completely").
S3. Labels MUST be monotonically ordered along the concept (not shuffled).
S4. Scale polarity must MATCH the concept:
    - Bipolar concepts (good/bad evaluations, positive/negative feelings)        -> "bipolar" scale_type
    - Unipolar concepts (satisfaction, importance, love, agreement-with-a-fact)  -> "unipolar" scale_type
S5. n_points MUST be between 5 and 11. Default to 5.
S6. Every scale point MUST have a clear text label that uses the concept word(s) from the question (e.g. for "satisfaction" use "satisfied" in labels; do not switch to a different concept like "frequency").

EXAMPLES:

Example 1 (feeling, unipolar):
Assertion: "I am satisfied with my job"
Output:
{
  "assertion": "I am satisfied with my job",
  "basic_concept": "feeling",
  "question_text": "How satisfied are you with your job?",
  "format_type": "direct_interrogative",
  "scale_type": "unipolar",
  "n_points": 5,
  "labels": ["Not at all satisfied", "Slightly satisfied", "Moderately satisfied", "Very satisfied", "Completely satisfied"],
  "polarity_reason": "Satisfaction has a clear zero point and increases in one direction, so a unipolar scale fits."
}

Example 2 (evaluation, bipolar):
Assertion: "The current government is doing a good job"
Output:
{
  "assertion": "The current government is doing a good job",
  "basic_concept": "evaluation",
  "question_text": "How would you rate the job the current government is doing?",
  "format_type": "direct_interrogative",
  "scale_type": "bipolar",
  "n_points": 5,
  "labels": ["Very bad", "Bad", "Neither good nor bad", "Good", "Very good"],
  "polarity_reason": "Good/bad evaluations are inherently bipolar with a clear neutral midpoint."
}

Example 3 (importance, unipolar):
Assertion: "Family is important to me"
Output:
{
  "assertion": "Family is important to me",
  "basic_concept": "importance",
  "question_text": "How important is family to you?",
  "format_type": "direct_interrogative",
  "scale_type": "unipolar",
  "n_points": 5,
  "labels": ["Not at all important", "Slightly important", "Moderately important", "Very important", "Extremely important"],
  "polarity_reason": "Importance ranges from none to extreme along one direction, so unipolar fits."
}"""


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        device_map="auto",
    )
    return tokenizer, model


def _extract_json(text):
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        brace = re.search(r"\{.*\}", text, re.DOTALL)
        candidate = brace.group(0) if brace else text
    return json.loads(candidate)


def _run(assertion, tokenizer, model, suffix=""):
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
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )
    input_length = inputs["input_ids"].shape[1]
    return tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)


def generate_item(assertion, tokenizer, model):
    answer = _run(assertion, tokenizer, model)
    try:
        return _extract_json(answer)
    except (json.JSONDecodeError, ValueError):
        answer = _run(assertion, tokenizer, model, suffix="Output JSON only, no markdown.")
        return _extract_json(answer)


if __name__ == "__main__":
    tokenizer, model = load_model()

    assertions_path = Path("./data/assertions.json")
    gold = json.loads(assertions_path.read_text())

    results = []
    for entry in gold:
        item = generate_item(entry["assertion"], tokenizer, model)
        results.append(item)
        print(json.dumps(item, indent=2, ensure_ascii=False))
        print()

    out_dir = Path("./outputs")
    out_dir.mkdir(exist_ok=True)
    (out_dir / "items.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"Saved {len(results)} items to {out_dir / 'items.json'}")