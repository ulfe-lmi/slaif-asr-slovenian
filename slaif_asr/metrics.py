from __future__ import annotations

import re
from dataclasses import dataclass
from statistics import median


TOKEN_PATTERN = re.compile(r"\S+")


@dataclass(frozen=True)
class ErrorRate:
    distance: int
    denominator: int
    percent: float


@dataclass(frozen=True)
class EditCounts:
    substitutions: int
    deletions: int
    insertions: int

    @property
    def distance(self) -> int:
        return self.substitutions + self.deletions + self.insertions


@dataclass(frozen=True)
class CorpusMetricSummary:
    corpus_wer: float
    corpus_cer: float
    mean_utterance_wer: float
    mean_utterance_cer: float
    median_utterance_wer: float
    median_utterance_cer: float
    empty_hypothesis_count: int
    total_word_edits: int
    total_reference_words: int
    total_character_edits: int
    total_reference_characters: int


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


def edit_counts(reference: list[str], hypothesis: list[str]) -> EditCounts:
    rows = len(reference) + 1
    cols = len(hypothesis) + 1
    costs = [[0] * cols for _ in range(rows)]
    counts = [[EditCounts(0, 0, 0) for _ in range(cols)] for _ in range(rows)]
    for row in range(1, rows):
        costs[row][0] = row
        counts[row][0] = EditCounts(0, row, 0)
    for col in range(1, cols):
        costs[0][col] = col
        counts[0][col] = EditCounts(0, 0, col)
    for row in range(1, rows):
        for col in range(1, cols):
            if reference[row - 1] == hypothesis[col - 1]:
                choices = [(costs[row - 1][col - 1], counts[row - 1][col - 1])]
            else:
                previous = counts[row - 1][col - 1]
                choices = [
                    (
                        costs[row - 1][col - 1] + 1,
                        EditCounts(previous.substitutions + 1, previous.deletions, previous.insertions),
                    )
                ]
            deleted = counts[row - 1][col]
            inserted = counts[row][col - 1]
            choices.extend(
                [
                    (costs[row - 1][col] + 1, EditCounts(deleted.substitutions, deleted.deletions + 1, deleted.insertions)),
                    (
                        costs[row][col - 1] + 1,
                        EditCounts(inserted.substitutions, inserted.deletions, inserted.insertions + 1),
                    ),
                ]
            )
            # Prefer substitutions, then deletions, then insertions for deterministic tie-breaking.
            cost, count = min(choices, key=lambda item: (item[0], item[1].insertions, item[1].deletions))
            costs[row][col] = cost
            counts[row][col] = count
    return counts[-1][-1]


def raw_wer(reference: str, hypothesis: str) -> ErrorRate:
    reference_words = TOKEN_PATTERN.findall(reference.strip())
    hypothesis_words = TOKEN_PATTERN.findall(hypothesis.strip())
    denominator = len(reference_words)
    distance = levenshtein_distance(reference_words, hypothesis_words)
    percent = 0.0 if denominator == 0 and distance == 0 else (100.0 if denominator == 0 else distance / denominator * 100.0)
    return ErrorRate(distance=distance, denominator=denominator, percent=percent)


def raw_word_edit_counts(reference: str, hypothesis: str) -> EditCounts:
    return edit_counts(TOKEN_PATTERN.findall(reference.strip()), TOKEN_PATTERN.findall(hypothesis.strip()))


def raw_cer(reference: str, hypothesis: str) -> ErrorRate:
    reference_chars = list(reference)
    hypothesis_chars = list(hypothesis)
    denominator = len(reference_chars)
    distance = levenshtein_distance(reference_chars, hypothesis_chars)
    percent = 0.0 if denominator == 0 and distance == 0 else (100.0 if denominator == 0 else distance / denominator * 100.0)
    return ErrorRate(distance=distance, denominator=denominator, percent=percent)


def raw_character_edit_counts(reference: str, hypothesis: str) -> EditCounts:
    return edit_counts(list(reference), list(hypothesis))


def percent(distance: int, denominator: int) -> float:
    return 0.0 if denominator == 0 and distance == 0 else (100.0 if denominator == 0 else distance / denominator * 100.0)


def corpus_metric_summary(rows: list[tuple[str, str]]) -> CorpusMetricSummary:
    if not rows:
        raise ValueError("at least one reference/hypothesis row is required")
    wers = [raw_wer(reference, hypothesis).percent for reference, hypothesis in rows]
    cers = [raw_cer(reference, hypothesis).percent for reference, hypothesis in rows]
    total_word_edits = 0
    total_reference_words = 0
    total_character_edits = 0
    total_reference_characters = 0
    empty_count = 0
    for reference, hypothesis in rows:
        word_rate = raw_wer(reference, hypothesis)
        char_rate = raw_cer(reference, hypothesis)
        total_word_edits += word_rate.distance
        total_reference_words += word_rate.denominator
        total_character_edits += char_rate.distance
        total_reference_characters += char_rate.denominator
        empty_count += 1 if not hypothesis.strip() else 0
    return CorpusMetricSummary(
        corpus_wer=round(percent(total_word_edits, total_reference_words), 3),
        corpus_cer=round(percent(total_character_edits, total_reference_characters), 3),
        mean_utterance_wer=round(sum(wers) / len(wers), 3),
        mean_utterance_cer=round(sum(cers) / len(cers), 3),
        median_utterance_wer=round(float(median(wers)), 3),
        median_utterance_cer=round(float(median(cers)), 3),
        empty_hypothesis_count=empty_count,
        total_word_edits=total_word_edits,
        total_reference_words=total_reference_words,
        total_character_edits=total_character_edits,
        total_reference_characters=total_reference_characters,
    )


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
