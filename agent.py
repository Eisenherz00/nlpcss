"""agent.py — Assertion -> closed survey item generator.

ONE file, ONE program. The only thing that changes between environments is
which model is loaded:

  * Local development (laptop, no GPU): small model that auto-downloads from
    the HuggingFace Hub. Just run `python agent.py`. Use `--limit N` to only
    process the first N assertions for a quick smoke test, since the full run
    is slow without a real GPU.

  * LRZ deployment (GPU node): the large model is read from a local path on the
    dss filesystem. run_agent.sh sets MODEL_NAME to that path before launching,
    so nothing is downloaded on the compute node.

Model selection order: --model flag > MODEL_NAME env var > LOCAL_MODEL default.

The SYSTEM_PROMPT encodes the Saris & Gallhofer (2014) basic-concept typology
(see data/protocol.md and the questionnaire-design PDF) and is the heart of the
agent: it tells the model how to turn one declarative assertion into one closed
survey item, returned as a single JSON object.
"""

import argparse
import json
import os
import re
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer

# Small model for laptop development (auto-downloads ~3 GB on first run).
LOCAL_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

SYSTEM_PROMPT = """You are a survey methodology expert following Saris & Gallhofer (2014) conventions.
Given a declarative assertion, you turn it into a single CLOSED survey item and return it as ONE JSON object.

There are TWO families of basic concepts, and they take DIFFERENT response formats:

A) SUBJECTIVE concepts -> an ordered, labeled RATING scale.
   Concepts: feeling, evaluation, importance, cognition, norm, right, values,
   causal_relationship, similarity_relationship, preference, policies,
   action_tendencies, expectations_of_future_events, evaluative_belief, behavior, quantities
   -> scale_type is "unipolar" or "bipolar"; labels are MONOTONICALLY ordered from one pole to the other.

B) OBJECTIVE / factual concepts -> a NOMINAL closed item (unordered categories, yes/no, or numeric bands).
   Concepts: events, demographics, knowledge, time, place, procedures
   -> scale_type is "nominal"; labels are MUTUALLY EXCLUSIVE, EXHAUSTIVE factual answer options
      (e.g. ["No", "Yes"] for an event; ["Rural area", "Small town", "Suburb", "Large city"] for place).
      Do NOT use agreement or intensity wording for these; just offer the factual options.

Return ONLY a JSON object with EXACTLY these keys:
- "assertion":          the original assertion string
- "basic_concept":      one of the concepts listed above
- "question_text":      the survey question (must be a CLOSED question, not open-ended)
- "format_type":        one of "direct_interrogative", "direct_imperative", "indirect_interrogative", "indirect_imperative"
- "scale_type":         "unipolar", "bipolar", or "nominal"
- "n_points":           integer. For RATING scales: 5-11 (default 5).
                        For NOMINAL items: the number of answer options (2-11; use 2 for yes/no).
- "labels":             list of text labels, length must equal n_points.
                        RATING: monotonically ordered (most-negative-end to most-positive-end, or reverse — never shuffled).
                        NOMINAL: the mutually exclusive answer categories (order is not meaningful).
- "polarity_reason":    one short sentence justifying the scale_type / response format given the basic_concept

HARD RULES (your output WILL be checked against each of these):

[Wording rules — apply to EVERY item]
W1. Question format MUST be one of the four formats listed in format_type. Prefer "direct_interrogative" unless the assertion strongly suggests otherwise.
    - direct_interrogative:   "How satisfied are you with your job?"
    - direct_imperative:      "Indicate your satisfaction with your job on a scale from 0 to 10."
    - indirect_interrogative: "Do you feel that you are satisfied with your job?"
    - indirect_imperative:    "Please tell me how satisfied you are with your job."
W2. No LEADING wording. Forbidden patterns: "Don't you agree...", "Wouldn't you say...", "Isn't it true that...", "Don't you think...".
W3. No LOADED presuppositions. Do not assume facts not stated in the assertion.
W4. No VAGUE frequency terms in labels (e.g. "often", "sometimes", "rarely", "frequently", "regularly", "occasionally"). Labels must directly reflect the concept being measured.
W5. The question must be CLOSED (answerable on the provided options). Never produce open-ended "What/Why" questions that require a free-text reply.

[Rating-scale rules — apply ONLY when scale_type is "unipolar" or "bipolar"]
S1. NO agree-disagree scales. Never use labels like "Strongly agree", "Agree", "Disagree", "Strongly disagree". Use item-specific labels that name the concept directly (e.g. "Not at all satisfied" ... "Completely satisfied").
S2. Labels MUST be balanced:
    - Bipolar: equal number of negative and positive labels around an optional midpoint (e.g. 2 neg + 1 mid + 2 pos for 5 points).
    - Unipolar: labels evenly graded from zero/low to maximum (e.g. "Not at all" ... "Completely").
S3. Labels MUST be monotonically ordered along the concept (not shuffled).
S4. Scale polarity must MATCH the concept:
    - Bipolar concepts (good/bad evaluations, positive/negative feelings, should/should-not norms) -> "bipolar"
    - Unipolar concepts (satisfaction, importance, intensity, likelihood, counts)                 -> "unipolar"
S5. n_points MUST be between 5 and 11. Default to 5.
S6. Every scale point MUST have a clear text label that uses the concept word(s) from the question.

[Nominal rules — apply ONLY when scale_type is "nominal"]
N1. Use "nominal" ONLY for objective concepts: events, demographics, knowledge, time, place, procedures.
N2. Provide MUTUALLY EXCLUSIVE and EXHAUSTIVE categories. For a plain yes/no fact use exactly ["No", "Yes"].
N3. Categories must be FACTUAL — never agreement or intensity wording, never agree/disagree.
N4. The question must still be CLOSED (the respondent picks one option, no free text).

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
}

Example 4 (demographics, nominal — yes/no fact):
Assertion: "I have completed a university degree"
Output:
{
  "assertion": "I have completed a university degree",
  "basic_concept": "demographics",
  "question_text": "Have you completed a university degree?",
  "format_type": "direct_interrogative",
  "scale_type": "nominal",
  "n_points": 2,
  "labels": ["No", "Yes"],
  "polarity_reason": "Holding a degree is an objective yes/no fact, so a nominal two-option item fits."
}

Example 5 (place, nominal — categories):
Assertion: "I grew up in a rural area"
Output:
{
  "assertion": "I grew up in a rural area",
  "basic_concept": "place",
  "question_text": "In what type of area did you grow up?",
  "format_type": "direct_interrogative",
  "scale_type": "nominal",
  "n_points": 4,
  "labels": ["Rural area", "Small town", "Suburb", "Large city"],
  "polarity_reason": "Type of area is an objective category with no inherent order, so a nominal item fits."
}"""


def load_model(model_name):
    """Load tokenizer + model. Local path on LRZ; HF Hub download locally."""
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto"
    )
    print(f"Model loaded on device: {model.device}")
    return tokenizer, model


def _extract_json(text):
    """Pull a JSON object out of the model's raw text output."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        candidate = fenced.group(1)
    else:
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
        **inputs, max_new_tokens=512, do_sample=True, temperature=0.7, top_p=0.9
    )
    input_length = inputs["input_ids"].shape[1]
    return tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)


def generate_item(assertion, tokenizer, model):
    """Generate one survey-item JSON, with a single retry on parse failure."""
    answer = _run(assertion, tokenizer, model)
    try:
        return _extract_json(answer)
    except (json.JSONDecodeError, ValueError):
        answer = _run(assertion, tokenizer, model, suffix="Output JSON only, no markdown.")
        return _extract_json(answer)


def run_all(model_name, assertions_path, out_dir, limit=None):
    """Load the model, generate an item per assertion, write outputs/items.json."""
    tokenizer, model = load_model(model_name)

    gold = json.loads(Path(assertions_path).read_text())
    if limit:
        gold = gold[:limit]

    results = []
    for i, entry in enumerate(gold, 1):
        print(f"[{i}/{len(gold)}] Processing: {entry['assertion']}")
        item = generate_item(entry["assertion"], tokenizer, model)
        results.append(item)
        print(json.dumps(item, indent=2, ensure_ascii=False))
        print()

    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    (out / "items.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"Saved {len(results)} items to {out / 'items.json'}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Turn assertions into closed survey items.")
    parser.add_argument(
        "--model",
        default=os.environ.get("MODEL_NAME", LOCAL_MODEL),
        help="Model name/path. Defaults to $MODEL_NAME, else the small local model.",
    )
    parser.add_argument("--assertions", default="./data/assertions.json")
    parser.add_argument("--out", default="./outputs")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only process the first N assertions (handy for quick local tests).",
    )
    args = parser.parse_args()
    run_all(args.model, args.assertions, args.out, limit=args.limit)


if __name__ == "__main__":
    main()
