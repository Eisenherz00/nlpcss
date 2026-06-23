"""src.evaluator — Rule-based survey item evaluation.

Submodules:
    lexicon     Sentiment/quality word lists and label classification
    checks      Individual check functions (one per evaluation criterion)
    scoring     Item-level and batch-level scoring
    report      Terminal report printing
"""

from src.evaluator.scoring import evaluate_item, evaluate_batch
from src.evaluator.report import print_report

__all__ = ["evaluate_item", "evaluate_batch", "print_report"]
