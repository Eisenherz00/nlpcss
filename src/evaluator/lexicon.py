"""Lexicon constants and label classification for the evaluator.

Provides word lists for sentiment polarity detection, vague-term detection,
agree-disagree detection, and leading-question patterns.
"""

from typing import List

NEGATIVE_HINTS: List[str] = [
    "not at all", "not", "never", "no ", "dislike", "bad", "low", "weak", "poor",
    "negative", "unimportant", "dissatisf", "disagree", "worst", "rarely", "extremely bad",
    "very bad",
]

POSITIVE_HINTS: List[str] = [
    "very", "extremely", "completely", "highly", "great", "good", "love", "like",
    "strong", "positive", "important", "satisf", "agree", "best", "always", "frequently",
]

NEUTRAL_HINTS: List[str] = [
    "neither", "neutral", "moderate", "somewhat", "slightly", "average", "midway",
]

VAGUE_TERMS: List[str] = [
    "often", "sometimes", "rarely", "frequently", "regularly", "occasionally",
]

AGREE_DISAGREE_LABELS = {
    "strongly disagree", "disagree", "somewhat disagree", "slightly disagree",
    "strongly agree", "agree", "somewhat agree", "slightly agree",
    "neither agree nor disagree",
}

LEADING_PATTERNS: List[str] = [
    r"\bdon'?t you\b",
    r"\bwouldn'?t you agree\b",
    r"\bisn'?t it true\b",
    r"\bdon'?t you think\b",
    r"\bwouldn'?t you say\b",
]


def classify_label(label: str) -> int:
    """Returns -1 (negative), 0 (neutral), +1 (positive) based on lexicon match."""
    lc = label.lower()
    if any(h in lc for h in NEUTRAL_HINTS):
        return 0
    if any(h in lc for h in NEGATIVE_HINTS):
        return -1
    if any(h in lc for h in POSITIVE_HINTS):
        return +1
    return 0
