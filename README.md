# Assertion → Survey Item Agent

An LLM agent that turns a declarative **assertion** into a single **closed survey
item** (question + response scale), plus a rule-based **evaluator** that scores
each item against survey-methodology best practice.

This implements **Step 3** of the Saris & Gallhofer 3-step procedure
("Development of requests for an answer from assertions") taught in the LMU
*Questionnaire Design* course. The agent's prompt and the evaluator's checks are
derived directly from that course material (`Questionnaire Design_LMU_1 2.pdf`)
and the project protocol (`data/protocol.md`).

---

## Pipeline

```
data/assertions.json ──▶ agent.py ──▶ outputs/items.json ──▶ evaluator.py ──▶ outputs/eval_report.json
   (assertion + gold)     (generate)     (survey items)        (score)          (pass rates)
```

1. **Generate** (`agent.py`): for each assertion, the model emits one JSON survey
   item (`question_text`, `format_type`, `scale_type`, `n_points`, `labels`, …).
2. **Evaluate** (`evaluator.py`): rule-based checks score each item; criteria that
   genuinely need an LLM judge are reported as *deferred* rather than faked.
3. **Test the rubric** (`test_evaluator.py`): confirms each evaluator check fires
   on a crafted bad item and passes a good one (`data/evaluator_test_cases.json`).

---

## Quick start

### Local (laptop, for development/testing — LRZ is slow to iterate on)

```bash
poetry install                     # create venv + install all dependencies

poetry run python agent.py --limit 5         # quick smoke test
poetry run python agent.py                   # full run with the small local model
poetry run python evaluator.py               # score outputs/items.json
poetry run python tests/test_evaluator.py    # check the evaluator rubric itself
```

`agent.py` defaults to a small model (`Qwen2.5-1.5B-Instruct`, auto-downloaded)
so it runs without a GPU. Model selection order: `--model` > `$MODEL_NAME` env >
local default.

### LRZ (GPU node — the actual deliverable)

```bash
bash scripts/run_on_lrz.sh   # rsync code -> sbatch job -> poll -> rsync results back
```

`scripts/run_on_lrz.sh` orchestrates from your laptop; it submits an inline SLURM batch
job to run on the GPU node. The large model is read from a **local dss
path** (set via `MODEL_NAME` in the script), so nothing is downloaded on the
compute node.

---

## Knowledge base (from the course PDF + protocol)

The agent does **not** improvise survey design — it encodes the rules below.

### The 3-step Saris & Gallhofer procedure

1. **Concepts**: distinguish concepts-by-postulation (constructs, multi-item)
   from concepts-by-intuition (CI, measurable with one question). All constructs
   are operationalised through CIs.
2. **Assertions**: for each CI, fix the *domain* and *basic concept*, then write a
   declarative assertion (e.g. "I am satisfied with my job"). ← *given to us as input*
3. **Requests for an answer**: turn the assertion into a closed survey item.
   ← **this is what the agent automates.**

### Basic concepts → response format

The basic concept determines whether the item is a **rating scale** or a
**nominal** item:

| Family | Basic concepts | `scale_type` | Labels |
| --- | --- | --- | --- |
| **Subjective** | feeling, evaluation, importance, cognition, norm, right, values, causal_relationship, similarity_relationship, preference, policies, action_tendencies, expectations_of_future_events, evaluative_belief, behavior, quantities | `unipolar` / `bipolar` | ordered, monotonic, item-specific |
| **Objective / factual** | events, demographics, knowledge, time, place, procedures | `nominal` | mutually exclusive, exhaustive categories |

- **Unipolar** concept (satisfaction, importance, intensity, counts): one
  direction, *no neutral midpoint* — e.g. "Not at all" → "Completely".
- **Bipolar** concept (good/bad, agree/should↔should-not): two poles with an
  optional neutral midpoint — e.g. "Very bad" … "Very good".
- Match the scale polarity to the concept's polarity.

### The four request (question) formats

| `format_type` | Example |
| --- | --- |
| `direct_interrogative` | "How satisfied are you with your job?" |
| `direct_imperative` | "Indicate your job satisfaction on a scale from 0 to 10." |
| `indirect_interrogative` | "Do you feel that you are satisfied with your job?" |
| `indirect_imperative` | "Please tell me how satisfied you are with your job." |

### Wording pitfalls to avoid (checked by the evaluator)

- **Leading** questions ("Don't you agree that…") — push a preferred answer.
- **Loaded** questions ("What is the best book you read last year?") — hidden
  presuppositions. *(needs an LLM judge)*
- **Recall error** — long recall windows respondents can't remember accurately;
  telescoping. *(needs an LLM judge)*
- **Vague/ambiguous** wording — undefined quantifiers ("often", "regularly").
- **Sensitive topics** — accusatory phrasing; should be normalised/indirect.
  *(needs an LLM judge)*
- **Double-barreled**, **negatively worded**, and **double-negative** items.

### Response-scale rules

- Prefer **closed** over open questions (unless top-of-mind data is the goal).
- **Avoid agree–disagree (AD) scales**; use item-specific (IS) scales —
  empirically higher quality (Saris et al. 2010).
- Response options must be **complete**, **mutually exclusive**, and all refer to
  the **same concept**.
- **Number of points**: 5–11 for rating scales (7 a common cap; 5 or 11 also
  used); nominal items need ≥2 options.
- **Order labels** monotonically (consistently positive→negative or reverse).
- **Label** the scale points (at least the endpoints / fixed reference points).
- Keep scales **balanced/symmetric** unless the variable is known to be skewed.

---

## Evaluator coverage

Rule-based checks (auto-scored): `format_type_valid`, `no_leading_wording`,
`no_vague_terms`, `is_closed_question`, `balanced_categories`,
`labels_ordered_monotonically`, `polarity_matches_concept`, `all_labels_present`,
`n_points_in_range`, `no_agree_disagree`.

Deferred to an LLM judge (semantic judgement required): `no_loaded_question`,
`no_recall_error`, `no_sensitive_topic_handling`, `assertion_question_alignment`.

---

## Repository layout

```
nlpcss/
├── src/                                 # Library code (importable as `src.agent`, `src.evaluator`)
│   ├── agent/
│   │   ├── prompts.py                   #   SYSTEM_PROMPT + LOCAL_MODEL constants
│   │   ├── model.py                     #   load_model() — HF Hub / local path
│   │   ├── generation.py                #   _extract_json(), _run(), generate_item()
│   │   └── pipeline.py                  #   run_all() batch processing
│   └── evaluator/
│       ├── lexicon.py                   #   Sentiment word lists + classify_label()
│       ├── checks.py                    #   15 check_xxx() functions
│       ├── scoring.py                   #   evaluate_item(), evaluate_batch()
│       └── report.py                    #   Terminal report printing
├── tests/
│   └── test_evaluator.py               # Unit test for the evaluator rubric
├── data/
│   ├── assertions.json                  # Set A: generation inputs + gold labels
│   └── evaluator_test_cases.json        # Set B: bad/good pairs per evaluation criterion
├── agent.py                             # Thin CLI entry point → src.agent
├── evaluator.py                         # Thin CLI entry point → src.evaluator
├── scripts/
│   └── run_on_lrz.sh                    # Local: sync → submit → poll → sync back
├── pyproject.toml                       # Python packaging (pip install -e .)
├── requirements.txt
└── outputs/                             # Generated items + eval reports (gitignored)
```

## Data

- **`data/assertions.json`** — 75 assertions covering every basic concept with ≥3
  examples each, annotated with `domain`, `basic_concept`, `expected_polarity`,
  and a `sensitive` flag. Used both as generation input and as gold for the
  evaluator's polarity check.
- **`data/evaluator_test_cases.json`** — one bad/good item pair per Caro
  evaluation criterion, with an `auto_testable` flag separating rule-checkable
  criteria from LLM-judge ones.

## Main references

- Saris, W. & Gallhofer, I. (2014). *Design, Evaluation, and Analysis of
  Questionnaires for Survey Research.* Wiley.
- Weber, W., Gallhofer, I. & Saris, W. (2020). *Design of Survey Questions.*
  SAGE Research Methods Foundations.
- Saris, Revilla, Krosnick & Shaeffer (2010). Comparing agree/disagree vs.
  construct-specific response options. *Survey Research Methods* 4(1).
