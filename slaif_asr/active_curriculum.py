from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from slaif_asr.metrics import (
    CorpusMetricSummary,
    corpus_metric_summary,
    raw_cer,
    raw_character_edit_counts,
    raw_wer,
    raw_word_edit_counts,
)


@dataclass(frozen=True)
class ScoredCandidate:
    candidate_id: str
    reference: str
    hypothesis: str
    phenomena: tuple[str, ...]
    word_substitutions: int
    word_deletions: int
    word_insertions: int
    character_substitutions: int
    character_deletions: int
    character_insertions: int
    word_error_rate: float
    character_error_rate: float
    empty_hypothesis: bool


@dataclass(frozen=True)
class PromotionDecision:
    promoted: bool
    reasons: tuple[str, ...]
    synthetic_holdout: CorpusMetricSummary
    real_gate: CorpusMetricSummary


def score_candidate(
    *,
    candidate_id: str,
    reference: str,
    hypothesis: str,
    phenomena: tuple[str, ...],
) -> ScoredCandidate:
    word_counts = raw_word_edit_counts(reference, hypothesis)
    char_counts = raw_character_edit_counts(reference, hypothesis)
    return ScoredCandidate(
        candidate_id=candidate_id,
        reference=reference,
        hypothesis=hypothesis,
        phenomena=tuple(phenomena),
        word_substitutions=word_counts.substitutions,
        word_deletions=word_counts.deletions,
        word_insertions=word_counts.insertions,
        character_substitutions=char_counts.substitutions,
        character_deletions=char_counts.deletions,
        character_insertions=char_counts.insertions,
        word_error_rate=round(raw_wer(reference, hypothesis).percent, 3),
        character_error_rate=round(raw_cer(reference, hypothesis).percent, 3),
        empty_hypothesis=not hypothesis.strip(),
    )


def deterministic_active_selection(
    scored: list[ScoredCandidate],
    *,
    hard_count: int,
    phenomenon_quota: dict[str, int] | None = None,
) -> list[ScoredCandidate]:
    quota = dict(phenomenon_quota or {})
    selected: list[ScoredCandidate] = []
    selected_ids: set[str] = set()
    for phenomenon, count in sorted(quota.items()):
        matching = [item for item in scored if phenomenon in item.phenomena and item.candidate_id not in selected_ids]
        for item in sorted(matching, key=ranking_key)[:count]:
            selected.append(item)
            selected_ids.add(item.candidate_id)
            if len(selected) >= hard_count:
                return selected
    for item in sorted(scored, key=ranking_key):
        if item.candidate_id in selected_ids:
            continue
        selected.append(item)
        selected_ids.add(item.candidate_id)
        if len(selected) >= hard_count:
            break
    return selected


def ranking_key(item: ScoredCandidate) -> tuple[Any, ...]:
    return (
        0 if item.empty_hypothesis else 1,
        -item.character_error_rate,
        -item.word_error_rate,
        -len(set(item.phenomena)),
        item.candidate_id,
    )


def deterministic_controls(
    candidates: list[ScoredCandidate],
    *,
    exclude_ids: set[str],
    count: int,
    seed: int,
) -> list[ScoredCandidate]:
    import hashlib

    pool = [item for item in candidates if item.candidate_id not in exclude_ids]
    ordered = sorted(
        pool,
        key=lambda item: hashlib.sha256(f"{seed}:{item.candidate_id}".encode("utf-8")).hexdigest(),
    )
    return ordered[:count]


def decide_promotion(
    *,
    integrity_passed: bool,
    synthetic_base_rows: list[tuple[str, str]],
    synthetic_challenger_rows: list[tuple[str, str]],
    real_base_rows: list[tuple[str, str]],
    real_challenger_rows: list[tuple[str, str]],
) -> PromotionDecision:
    reasons: list[str] = []
    synthetic_base = corpus_metric_summary(synthetic_base_rows)
    synthetic_challenger = corpus_metric_summary(synthetic_challenger_rows)
    real_base = corpus_metric_summary(real_base_rows)
    real_challenger = corpus_metric_summary(real_challenger_rows)
    synthetic_wer_relative = relative_improvement(synthetic_base.corpus_wer, synthetic_challenger.corpus_wer)
    synthetic_cer_relative = relative_improvement(synthetic_base.corpus_cer, synthetic_challenger.corpus_cer)
    if not integrity_passed:
        reasons.append("state-dictionary integrity failed")
    if synthetic_wer_relative < 15.0 and synthetic_cer_relative < 15.0:
        reasons.append("synthetic holdout did not improve by 15% relative WER or CER")
    if real_challenger.corpus_wer - real_base.corpus_wer > 1.0:
        reasons.append("real-gate corpus WER regressed by more than 1.0 absolute point")
    if real_challenger.corpus_cer - real_base.corpus_cer > 1.5:
        reasons.append("real-gate corpus CER regressed by more than 1.5 absolute points")
    if real_challenger.empty_hypothesis_count > real_base.empty_hypothesis_count:
        reasons.append("real-gate empty-hypothesis count increased")
    return PromotionDecision(
        promoted=not reasons,
        reasons=tuple(reasons),
        synthetic_holdout=synthetic_challenger,
        real_gate=real_challenger,
    )


def relative_improvement(base: float, challenger: float) -> float:
    return 0.0 if base == 0 else (base - challenger) / base * 100.0


def build_round2_failure_brief(scored: list[ScoredCandidate], *, include_references: bool = True) -> dict[str, Any]:
    clusters: dict[str, int] = {}
    empty_examples: list[dict[str, str]] = []
    for item in scored:
        for phenomenon in item.phenomena:
            clusters[phenomenon] = clusters.get(phenomenon, 0) + 1
        if item.empty_hypothesis and include_references:
            empty_examples.append({"candidate_id": item.candidate_id, "reference": item.reference, "hypothesis": item.hypothesis})
    return {
        "schema_version": "1.0",
        "source": "synthetic_candidate_pool_only",
        "failed_phenomenon_counts": dict(sorted(clusters.items())),
        "empty_hypothesis_examples": sorted(empty_examples, key=lambda item: item["candidate_id"])[:16],
        "real_gate_reference_text_included": False,
        "synthetic_holdout_errors_included": False,
    }


def assert_training_ids_are_disjoint(
    *,
    training_ids: set[str],
    synthetic_holdout_ids: set[str],
    real_gate_ids: set[str],
) -> None:
    holdout_overlap = sorted(training_ids.intersection(synthetic_holdout_ids))
    real_overlap = sorted(training_ids.intersection(real_gate_ids))
    if holdout_overlap:
        raise ValueError(f"synthetic holdout IDs cannot enter training: {holdout_overlap}")
    if real_overlap:
        raise ValueError(f"real gate IDs cannot enter training: {real_overlap}")


def promotion_decision_json(decision: PromotionDecision) -> dict[str, Any]:
    return {
        "promoted": decision.promoted,
        "reasons": list(decision.reasons),
        "synthetic_holdout": asdict(decision.synthetic_holdout),
        "real_gate": asdict(decision.real_gate),
    }
