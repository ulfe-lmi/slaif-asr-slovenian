from __future__ import annotations

import re
from dataclasses import dataclass


TOKEN_PATTERN = re.compile(r"\S+")


@dataclass(frozen=True)
class ErrorRate:
    distance: int
    denominator: int
    percent: float


def levenshtein_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for row_index, ref_item in enumerate(reference, start=1):
        current = [row_index]
        for column_index, hyp_item in enumerate(hypothesis, start=1):
            substitution = previous[column_index - 1] + (0 if ref_item == hyp_item else 1)
            insertion = current[column_index - 1] + 1
            deletion = previous[column_index] + 1
            current.append(min(substitution, insertion, deletion))
        previous = current
    return previous[-1]


def raw_wer(reference: str, hypothesis: str) -> ErrorRate:
    reference_words = TOKEN_PATTERN.findall(reference.strip())
    hypothesis_words = TOKEN_PATTERN.findall(hypothesis.strip())
    denominator = len(reference_words)
    distance = levenshtein_distance(reference_words, hypothesis_words)
    percent = 0.0 if denominator == 0 and distance == 0 else (100.0 if denominator == 0 else distance / denominator * 100.0)
    return ErrorRate(distance=distance, denominator=denominator, percent=percent)


def raw_cer(reference: str, hypothesis: str) -> ErrorRate:
    reference_chars = list(reference)
    hypothesis_chars = list(hypothesis)
    denominator = len(reference_chars)
    distance = levenshtein_distance(reference_chars, hypothesis_chars)
    percent = 0.0 if denominator == 0 and distance == 0 else (100.0 if denominator == 0 else distance / denominator * 100.0)
    return ErrorRate(distance=distance, denominator=denominator, percent=percent)


def empty_status(hypothesis: str) -> str:
    return "EMPTY_HYPOTHESIS" if not hypothesis.strip() else "NONEMPTY"


def recognition_change(reference: str, base_hypothesis: str, adapted_hypothesis: str) -> str:
    if not adapted_hypothesis.strip():
        return "EMPTY_HYPOTHESIS"
    if adapted_hypothesis == reference:
        return "EXACT_MATCH"
    base_wer = raw_wer(reference, base_hypothesis).percent
    adapted_wer = raw_wer(reference, adapted_hypothesis).percent
    if adapted_wer < base_wer:
        return "IMPROVED"
    if adapted_wer > base_wer:
        return "REGRESSED"
    if adapted_hypothesis.strip():
        return "UNCHANGED"
    return "NONEMPTY"
