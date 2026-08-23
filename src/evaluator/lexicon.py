"""Lexicon constants and label classification for the evaluator.

Provides word lists for sentiment polarity detection, vague-term detection,
agree-disagree detection, and leading-question patterns.
"""

from typing import List

NEGATIVE_HINTS: List[str] = [
    # Generic negation / absence
    "not at all", "never", "dislike",
    # Evaluative negatives
    "bad", "very bad", "extremely bad", "poor", "worst", "low", "weak",
    "negative", "unimportant", "dissatisf",
    # Opposition / disagreement
    "disagree", "oppose", "strongly oppose", "somewhat oppose",
    # un- / dis- prefix family (longer strings first to avoid partial shadowing)
    "unreliable", "unsafe", "unfair", "unhappy", "uncomfortable",
    "unlikely", "unpleasant", "unacceptable", "unjust", "unwell",
    # Comparative negatives for preference items
    "prefer office", "prefer in-person", "prefer driving", "prefer cash",
    # Low-intensity markers
    "not very", "hardly",
]

POSITIVE_HINTS: List[str] = [
    # Intensity / degree
    "extremely", "completely", "highly", "very",
    # Evaluative positives
    "great", "good", "excellent", "best",
    "love", "like", "strong", "positive",
    # Agreement / support
    "agree", "support", "strongly support", "somewhat support",
    # Concept-specific positives
    "important", "satisf", "reliable", "safe", "fair", "happy",
    "comfortable", "likely", "pleasant", "acceptable", "just",
    # Frequency at the high end
    "always", "frequently",
    # Comparative positives for preference items
    "prefer home", "prefer online", "prefer train", "prefer digital",
]

NEUTRAL_HINTS: List[str] = [
    # True midpoint / neither-pole markers only — do NOT add degree modifiers
    # like "somewhat" or "slightly" here, as they also modify negative/positive labels.
    "neither", "neutral", "average", "midway",
    "no preference", "indifferent",
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
    """Return -1 (negative), 0 (neutral), or +1 (positive) based on lexicon match.

    Neutral hints are checked first because midpoint labels like
    "Neither support nor oppose" must score 0 even though they contain words
    that appear in other hint lists.  Within negative/positive lists, longer
    strings are tried first to avoid partial shadowing (e.g. "not very"
    before "very").
    """
    lc = label.lower()
    # Neutral / midpoint labels take priority
    if any(h in lc for h in NEUTRAL_HINTS):
        return 0
    # Check negative hints (sorted longest-first to avoid partial shadowing)
    if any(h in lc for h in sorted(NEGATIVE_HINTS, key=len, reverse=True)):
        return -1
    # Check positive hints (sorted longest-first)
    if any(h in lc for h in sorted(POSITIVE_HINTS, key=len, reverse=True)):
        return +1
    return 0
