Slide 1 — What the agent does

One assertion in, one closed survey item out (question + scale), as structured JSON. Everything downstream scores that JSON.



Slide 2 — The agent is a prompt

It doesn't invent survey methodology, the prompt encodes it: subjective concept -> rating scale, objective concept -> nominal item.

@import "agent.py" {line_begin=36 line_end=48 class="line-numbers"}


Slide 3 — A strict output contract

Output is a fixed 8-field JSON. That contract is what makes automatic scoring possible, no free text to parse.

@import "agent.py" {line_begin=50 line_end=62 class="line-numbers"}


Slide 4 — How a single item is generated

Generate -> parse -> retry once if it fails. That's all the logic. The intelligence is in the prompt, the code just catches failures.

@import "agent.py" {line_begin=189 line_end=214 class="line-numbers"}


Slide 5 — Live example

"I feel anxious about climate change." -> unipolar 5-point scale, graded labels, no agree/disagree. Exactly what the prompt asked for.

terminal output of python agent.py, the [1/75] item


Slide 6 — The evaluator: one criterion = one function

Rule-based, transparent. Each criterion is one small function returning pass/fail + a reason.

@import "evaluator.py" {line_begin=100 line_end=115 class="line-numbers"}

(optional, the lexicon behind classify_label)

@import "evaluator.py" {line_begin=5 line_end=43 class="line-numbers"}


Slide 7 — We don't fake the hard checks

11 criteria are rule-checkable. The other 4 need real judgement (loaded, recall, sensitive, alignment), so they're marked deferred, not faked.

@import "evaluator.py" {line_begin=65 line_end=90 class="line-numbers"}


Slide 8 — Results

Runs end-to-end and self-tests. On 75 items, ~91% pass; the rubric itself is unit-tested. Weak checks (polarity, balance) are next.
