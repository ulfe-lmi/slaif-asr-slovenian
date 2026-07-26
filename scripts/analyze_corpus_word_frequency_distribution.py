#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from slaif_asr.real_eval import NORMALIZER_VERSION, atomic_write_json, atomic_write_text
try:
    from scripts.analyze_corpus_vocabulary_overlap import (
        ARTUR_J_DATASET_ID,
        EXPECTED_ARTUR_J_ROWS,
        EXPECTED_ARTUR_J_SHA256,
        EXPECTED_FLEURS_V2_ROWS,
        EXPECTED_FLEURS_V2_SHA256,
        EXPECTED_SCALE8000_ROWS,
        EXPECTED_SCALE8000_SHA256,
        FLEURS_V2_DATASET_ID,
        SCALE8000_DATASET_ID,
        DatasetBinding,
        RealGateLeakageError,
        _jsonl_texts,
        extract_normalized_tokens,
        load_dataset_vocabulary,
    )
except ModuleNotFoundError as error:
    if not error.name or not error.name.startswith("scripts"):
        raise
    from analyze_corpus_vocabulary_overlap import (  # type: ignore[no-redef]
        ARTUR_J_DATASET_ID,
        EXPECTED_ARTUR_J_ROWS,
        EXPECTED_ARTUR_J_SHA256,
        EXPECTED_FLEURS_V2_ROWS,
        EXPECTED_FLEURS_V2_SHA256,
        EXPECTED_SCALE8000_ROWS,
        EXPECTED_SCALE8000_SHA256,
        FLEURS_V2_DATASET_ID,
        SCALE8000_DATASET_ID,
        DatasetBinding,
        RealGateLeakageError,
        _jsonl_texts,
        extract_normalized_tokens,
        load_dataset_vocabulary,
    )


WORK_ORDER_ID = "0049"
CLASSIFICATION_COMPLETE = "WORD_FREQUENCY_DISTRIBUTION_ANALYSIS_COMPLETE"
CLASSIFICATION_BLOCKED = "WORD_FREQUENCY_DISTRIBUTION_ANALYSIS_BLOCKED_INPUT_MISSING"
CLASSIFICATION_LEAKAGE = "WORD_FREQUENCY_DISTRIBUTION_ANALYSIS_INVALID_REAL_GATE_LEAKAGE"
CLASSIFICATION_INVALID = "EXPERIMENT_INVALID"

BLOCKED_SCALE8000 = "BLOCKED_SCALE8000_TEXT_UNAVAILABLE"
BLOCKED_FLEURS = "BLOCKED_FLEURS_V2_REFERENCES_UNAVAILABLE"
BLOCKED_ARTUR = "BLOCKED_ARTUR_J_REFERENCES_UNAVAILABLE"

PRIMARY_MODE = "normalized_word_forms"
PRIMARY_TOKEN_POLICY = "unicode_alnum_tokens_requiring_letter_casefold_nfc_no_lemmatization"
BOOTSTRAP_UNIT = "row_utterance"
BOOTSTRAP_REPLICATES = 1_000
BOOTSTRAP_SEED = 480_049
PAIR_LABEL_WITHIN = "WITHIN_SAMPLING_ENVELOPE"
PAIR_LABEL_ABOVE = "ABOVE_SAMPLING_ENVELOPE"
NUMERICAL_TOLERANCE = 1e-12

DEFAULT_SCALE8000_PATH = Path(
    "runs/data-quality/sl-corpus-v5-scale8000-training-v1/fixed-combined-training-text.local.jsonl"
)
DEFAULT_FLEURS_PATH = Path("runs/evaluation-gates/fleurs-sl-si-test-full-v2/manifest.jsonl")
DEFAULT_ARTUR_PATH = Path("runs/evaluation-gates/artur-j-public-gate-v1/manifest.jsonl")
DEFAULT_JSON_OUTPUT = Path("docs/analysis/scale8000-gate-word-frequency-distribution.json")
DEFAULT_MARKDOWN_OUTPUT = Path("docs/analysis/scale8000-gate-word-frequency-distribution.md")

PAIR_SPECS = (
    ("scale8000_vs_fleurs_v2", "scale8000", "fleurs_v2"),
    ("scale8000_vs_artur_j", "scale8000", "artur_j"),
    ("fleurs_v2_vs_artur_j", "fleurs_v2", "artur_j"),
)
DATASET_STREAM_OFFSETS = {
    "scale8000": 11,
    "fleurs_v2": 23,
    "artur_j": 37,
}
TOP_K_VALUES = (10, 50, 100, 500, 1_000)
FREQUENCY_BANDS = (
    ("exactly_1", 1, 1),
    ("exactly_2", 2, 2),
    ("3_to_5", 3, 5),
    ("6_to_10", 6, 10),
    ("11_to_50", 11, 50),
    ("more_than_50", 51, None),
)

REQUIRED_REPORT_TOP_LEVEL_KEYS = {
    "schema_version",
    "work_order_id",
    "classification",
    "analysis_role",
    "normalization",
    "bootstrap_policy",
    "input_bindings",
    "datasets",
    "pairwise",
    "within_corpus_sampling",
    "interpretation_policy",
    "safety",
}
ALLOWED_CLASSIFICATIONS = {
    CLASSIFICATION_COMPLETE,
    CLASSIFICATION_BLOCKED,
    CLASSIFICATION_LEAKAGE,
    CLASSIFICATION_INVALID,
}
EXPECTED_SAFETY_FLAGS = {
    "raw_references_committed",
    "full_gate_wordlists_committed",
    "missing_gate_wordlists_committed",
    "top_or_ranked_gate_wordlists_committed",
    "individual_gate_statistics_committed",
    "local_paths_or_manifests_committed",
    "used_for_training_selection",
    "used_for_generation_prompting",
    "training_run",
    "asr_evaluation_run",
    "new_text_generated",
    "corpus_modified",
}
FORBIDDEN_REPORT_KEYS = {
    "audio_filepath",
    "full_gate_vocabularies",
    "full_gate_vocabulary",
    "full_gate_wordlists",
    "gate_vocabularies",
    "gate_vocabulary",
    "input_path",
    "input_paths",
    "lexical_content",
    "local_manifest",
    "local_manifests",
    "local_path",
    "local_paths",
    "missing_word_forms",
    "missing_words",
    "per_word_frequencies",
    "per_word_frequency",
    "per_word_probability_ratios",
    "ranked_word_lists",
    "ranked_words",
    "raw_reference",
    "raw_references",
    "reference",
    "references",
    "source_path",
    "source_paths",
    "spoken_text",
    "target_text",
    "text",
    "texts",
    "tokens",
    "top_words",
    "word_forms",
    "wordlist",
    "wordlists",
}
FORBIDDEN_KEY_FRAGMENTS = (
    "full_word",
    "missing_word",
    "per_word",
    "ranked_word",
    "raw_reference",
    "top_word",
    "word_list",
    "wordlist",
)
WINDOWS_ABSOLUTE_PATH = re.compile(r"^[A-Za-z]:[\\/]")


class InputUnavailableError(RuntimeError):
    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


@dataclass(frozen=True)
class DatasetFrequency:
    binding: DatasetBinding
    token_counts: Counter[str]
    row_token_counts: tuple[Counter[str], ...]

    @property
    def row_count(self) -> int:
        return len(self.row_token_counts)

    @property
    def total_tokens(self) -> int:
        return sum(self.token_counts.values())


@dataclass(frozen=True)
class TieInclusiveTopKSelection:
    forms: frozenset[str]
    requested_k_effective: int
    cutoff_occurrence_count: int
    tie_expansion_count: int

    @property
    def effective_k(self) -> int:
        return len(self.forms)


def dataset_from_texts(
    *,
    key: str,
    dataset_id: str,
    source_role: str,
    source_sha256: str,
    texts: Iterable[str],
) -> DatasetFrequency:
    rows: list[Counter[str]] = []
    aggregate: Counter[str] = Counter()
    for text in texts:
        row = Counter(extract_normalized_tokens(text, include_numeric_only=False))
        rows.append(row)
        aggregate.update(row)
    if not rows:
        raise ValueError(f"{dataset_id}: no rows are available")
    if not aggregate:
        raise ValueError(f"{dataset_id}: normalization produced no word tokens")
    return DatasetFrequency(
        binding=DatasetBinding(
            key=key,
            dataset_id=dataset_id,
            source_role=source_role,
            source_sha256=source_sha256,
        ),
        token_counts=aggregate,
        row_token_counts=tuple(rows),
    )


def load_dataset_frequency(
    path: Path,
    *,
    key: str,
    dataset_id: str,
    source_role: str,
    text_field: str,
    expected_rows: int,
    expected_sha256: str,
    expected_dataset_field: str | None = None,
) -> DatasetFrequency:
    validated = load_dataset_vocabulary(
        path,
        key=key,
        dataset_id=dataset_id,
        source_role=source_role,
        text_field=text_field,
        expected_rows=expected_rows,
        expected_dataset_field=expected_dataset_field,
        expected_sha256=expected_sha256,
    )
    texts = _jsonl_texts(
        path,
        dataset_id=dataset_id,
        text_field=text_field,
        expected_rows=expected_rows,
        expected_dataset_field=expected_dataset_field,
    )
    dataset = dataset_from_texts(
        key=key,
        dataset_id=dataset_id,
        source_role=source_role,
        source_sha256=validated.binding.source_sha256,
        texts=texts,
    )
    if dataset.token_counts != validated.primary.token_counts:
        raise ValueError(f"{dataset_id}: Work Order 0048 tokenization reuse check failed")
    return dataset


def normalized_probabilities(counts: Mapping[str, int]) -> dict[str, float]:
    total = sum(counts.values())
    if total <= 0:
        raise ValueError("a frequency distribution must contain at least one token")
    forms = tuple(sorted(form for form, count in counts.items() if count > 0))
    probabilities = {form: counts[form] / total for form in forms}
    if not math.isclose(
        math.fsum(probabilities[form] for form in forms),
        1.0,
        rel_tol=0.0,
        abs_tol=NUMERICAL_TOLERANCE,
    ):
        raise ValueError("normalized probabilities do not sum to one")
    return probabilities


def _distribution_metrics(
    probabilities_a: Mapping[str, float],
    probabilities_b: Mapping[str, float],
) -> dict[str, float]:
    support = tuple(sorted(set(probabilities_a) | set(probabilities_b)))
    jsd_terms: list[float] = []
    squared_hellinger_terms: list[float] = []
    absolute_difference_terms: list[float] = []
    overlap_terms: list[float] = []
    for form in support:
        probability_a = probabilities_a.get(form, 0.0)
        probability_b = probabilities_b.get(form, 0.0)
        midpoint = 0.5 * (probability_a + probability_b)
        if probability_a:
            jsd_terms.append(0.5 * probability_a * math.log2(probability_a / midpoint))
        if probability_b:
            jsd_terms.append(0.5 * probability_b * math.log2(probability_b / midpoint))
        squared_hellinger_terms.append(
            (math.sqrt(probability_a) - math.sqrt(probability_b)) ** 2
        )
        absolute_difference_terms.append(abs(probability_a - probability_b))
        overlap_terms.append(min(probability_a, probability_b))
    jsd = math.fsum(jsd_terms)
    squared_hellinger_sum = math.fsum(squared_hellinger_terms)
    absolute_difference_sum = math.fsum(absolute_difference_terms)
    overlap = math.fsum(overlap_terms)
    hellinger = math.sqrt(0.5 * squared_hellinger_sum)
    total_variation = 0.5 * absolute_difference_sum
    if not math.isclose(overlap, 1.0 - total_variation, rel_tol=0.0, abs_tol=1e-10):
        raise ValueError("probability-mass overlap does not equal one minus total variation")
    return {
        "jensen_shannon_divergence_base2": _clamp_unit(jsd),
        "hellinger_distance": _clamp_unit(hellinger),
        "total_variation_distance": _clamp_unit(total_variation),
        "probability_mass_overlap_coefficient": _clamp_unit(overlap),
    }


def full_union_distribution_metrics(
    counts_a: Mapping[str, int],
    counts_b: Mapping[str, int],
) -> dict[str, float | str]:
    probabilities_a = normalized_probabilities(counts_a)
    probabilities_b = normalized_probabilities(counts_b)
    result: dict[str, float | str] = dict(_distribution_metrics(probabilities_a, probabilities_b))
    forms_a = set(probabilities_a)
    forms_b = set(probabilities_b)
    shared_forms = tuple(sorted(forms_a & forms_b))
    forms_only_in_a = tuple(sorted(forms_a - forms_b))
    forms_only_in_b = tuple(sorted(forms_b - forms_a))
    support_component = 0.5 * math.fsum(
        (
            math.fsum(probabilities_a[form] for form in forms_only_in_a),
            math.fsum(probabilities_b[form] for form in forms_only_in_b),
        )
    )
    shared_frequency_component = 0.5 * math.fsum(
        abs(probabilities_a[form] - probabilities_b[form]) for form in shared_forms
    )
    total_variation = float(result["total_variation_distance"])
    if not math.isclose(
        math.fsum((support_component, shared_frequency_component)),
        total_variation,
        rel_tol=0.0,
        abs_tol=1e-10,
    ):
        raise ValueError("total-variation decomposition failed")
    result.update(
        {
            "support_mismatch_tv_component": support_component,
            "shared_frequency_tv_component": shared_frequency_component,
            "dominant_tv_component": (
                "support_mismatch"
                if support_component > shared_frequency_component
                else "shared_form_frequency_difference"
            ),
        }
    )
    return result


def shared_conditional_distribution_metrics(
    counts_a: Mapping[str, int],
    counts_b: Mapping[str, int],
) -> dict[str, int | float]:
    probabilities_a = normalized_probabilities(counts_a)
    probabilities_b = normalized_probabilities(counts_b)
    shared_forms = tuple(sorted(set(probabilities_a) & set(probabilities_b)))
    if not shared_forms:
        raise ValueError("shared-vocabulary conditional distribution has empty support")
    shared_mass_a = math.fsum(probabilities_a[form] for form in shared_forms)
    shared_mass_b = math.fsum(probabilities_b[form] for form in shared_forms)
    conditional_a = {
        form: probabilities_a[form] / shared_mass_a for form in shared_forms
    }
    conditional_b = {
        form: probabilities_b[form] / shared_mass_b for form in shared_forms
    }
    metrics = _distribution_metrics(conditional_a, conditional_b)
    return {
        "shared_forms": len(shared_forms),
        "shared_form_token_mass_in_a": shared_mass_a,
        "shared_form_token_mass_in_b": shared_mass_b,
        "conditional_jensen_shannon_divergence_base2": metrics["jensen_shannon_divergence_base2"],
        "conditional_hellinger_distance": metrics["hellinger_distance"],
        "conditional_probability_mass_overlap_coefficient": metrics[
            "probability_mass_overlap_coefficient"
        ],
    }


def _weighted_median(items: list[tuple[float, float]]) -> float:
    if not items:
        raise ValueError("weighted median requires at least one item")
    ordered = sorted(items)
    total_weight = math.fsum(weight for _, weight in ordered)
    if total_weight <= 0:
        raise ValueError("weighted median requires positive total weight")
    threshold = 0.5 * total_weight
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= threshold:
            return value
    return ordered[-1][0]


def shared_frequency_ratio_metrics(
    counts_a: Mapping[str, int],
    counts_b: Mapping[str, int],
) -> dict[str, int | float | str]:
    probabilities_a = normalized_probabilities(counts_a)
    probabilities_b = normalized_probabilities(counts_b)
    shared_forms = tuple(sorted(set(probabilities_a) & set(probabilities_b)))
    weighted_ratios: list[tuple[float, float]] = []
    for form in shared_forms:
        ratio = abs(math.log2(probabilities_a[form] / probabilities_b[form]))
        weighted_ratios.append((ratio, min(probabilities_a[form], probabilities_b[form])))
    total_weight = math.fsum(weight for _, weight in weighted_ratios)
    if total_weight <= 0:
        raise ValueError("shared-form ratio weighting has zero mass")

    def fraction_within(maximum_log2_ratio: float) -> float:
        return math.fsum(
            weight for ratio, weight in weighted_ratios if ratio <= maximum_log2_ratio
        ) / total_weight

    return {
        "shared_forms": len(shared_forms),
        "symmetric_mass_weighting": "minimum_empirical_probability",
        "shared_mass_weight_sum": total_weight,
        "weighted_mean_absolute_log2_ratio": math.fsum(
            ratio * weight for ratio, weight in weighted_ratios
        )
        / total_weight,
        "weighted_median_absolute_log2_ratio": _weighted_median(weighted_ratios),
        "fraction_of_shared_mass_within_factor_2": fraction_within(1.0),
        "fraction_of_shared_mass_within_factor_4": fraction_within(2.0),
        "fraction_of_shared_mass_within_factor_10": fraction_within(math.log2(10.0)),
    }


def _average_ranks(counts: Mapping[str, int], forms: tuple[str, ...]) -> dict[str, float]:
    ordered = sorted(forms, key=lambda form: (-counts[form], form))
    ranks: dict[str, float] = {}
    index = 0
    while index < len(ordered):
        end = index + 1
        count = counts[ordered[index]]
        while end < len(ordered) and counts[ordered[end]] == count:
            end += 1
        average_rank = ((index + 1) + end) / 2.0
        for position in range(index, end):
            ranks[ordered[position]] = average_rank
        index = end
    return ranks


def spearman_rank_correlation(
    counts_a: Mapping[str, int],
    counts_b: Mapping[str, int],
    *,
    minimum_count: int = 1,
) -> tuple[int, float]:
    forms = tuple(
        sorted(
            form
            for form in set(counts_a) & set(counts_b)
            if counts_a[form] >= minimum_count and counts_b[form] >= minimum_count
        )
    )
    if len(forms) < 2:
        return len(forms), 0.0
    ranks_a = _average_ranks(counts_a, forms)
    ranks_b = _average_ranks(counts_b, forms)
    mean_a = math.fsum(ranks_a[form] for form in forms) / len(forms)
    mean_b = math.fsum(ranks_b[form] for form in forms) / len(forms)
    covariance = math.fsum(
        (ranks_a[form] - mean_a) * (ranks_b[form] - mean_b) for form in forms
    )
    variance_a = math.fsum((ranks_a[form] - mean_a) ** 2 for form in forms)
    variance_b = math.fsum((ranks_b[form] - mean_b) ** 2 for form in forms)
    if variance_a == 0.0 or variance_b == 0.0:
        return len(forms), 1.0 if ranks_a == ranks_b else 0.0
    return len(forms), covariance / math.sqrt(variance_a * variance_b)


def frequency_rank_metrics(
    counts_a: Mapping[str, int],
    counts_b: Mapping[str, int],
) -> dict[str, object]:
    all_eligible, all_correlation = spearman_rank_correlation(counts_a, counts_b)
    repeated_eligible, repeated_correlation = spearman_rank_correlation(
        counts_a,
        counts_b,
        minimum_count=2,
    )
    return {
        "tie_policy": "average_ranks_count_descending_codepoint_tiebreak",
        "all_shared_forms": {
            "eligible_forms": all_eligible,
            "spearman_rank_correlation": all_correlation,
        },
        "count_at_least_2_in_both": {
            "eligible_forms": repeated_eligible,
            "spearman_rank_correlation": repeated_correlation,
        },
    }


def _tie_inclusive_top_k(
    counts: Mapping[str, int],
    requested_k: int,
) -> TieInclusiveTopKSelection:
    if requested_k <= 0:
        raise ValueError("requested top-k must be positive")
    if not counts:
        raise ValueError("top-k selection requires a nonempty frequency distribution")
    requested_k_effective = min(requested_k, len(counts))
    cutoff_occurrence_count = sorted(counts.values(), reverse=True)[requested_k_effective - 1]
    forms = frozenset(
        form for form, count in counts.items() if count >= cutoff_occurrence_count
    )
    return TieInclusiveTopKSelection(
        forms=forms,
        requested_k_effective=requested_k_effective,
        cutoff_occurrence_count=cutoff_occurrence_count,
        tie_expansion_count=len(forms) - requested_k_effective,
    )


def top_k_agreement_metrics(
    counts_a: Mapping[str, int],
    counts_b: Mapping[str, int],
    requested_k: int,
) -> dict[str, int | float]:
    top_a = _tie_inclusive_top_k(counts_a, requested_k)
    top_b = _tie_inclusive_top_k(counts_b, requested_k)
    shared = len(top_a.forms & top_b.forms)
    union = len(top_a.forms | top_b.forms)
    minimum_effective_k = min(top_a.effective_k, top_b.effective_k)
    return {
        "requested_k": requested_k,
        "effective_k_in_a": top_a.effective_k,
        "cutoff_occurrence_count_in_a": top_a.cutoff_occurrence_count,
        "tie_expansion_count_in_a": top_a.tie_expansion_count,
        "effective_k_in_b": top_b.effective_k,
        "cutoff_occurrence_count_in_b": top_b.cutoff_occurrence_count,
        "tie_expansion_count_in_b": top_b.tie_expansion_count,
        "shared_forms": shared,
        "jaccard_similarity": shared / union if union else 0.0,
        "overlap_coefficient": shared / minimum_effective_k if minimum_effective_k else 0.0,
    }


def frequency_of_frequency_profile(counts: Mapping[str, int]) -> dict[str, dict[str, int | float]]:
    unique_forms = len(counts)
    total_tokens = sum(counts.values())
    if unique_forms == 0 or total_tokens == 0:
        raise ValueError("frequency-of-frequency profile requires a nonempty distribution")
    result: dict[str, dict[str, int | float]] = {}
    for label, minimum, maximum in FREQUENCY_BANDS:
        values = [
            count
            for count in counts.values()
            if count >= minimum and (maximum is None or count <= maximum)
        ]
        forms_in_band = len(values)
        tokens_in_band = sum(values)
        result[label] = {
            "unique_forms": forms_in_band,
            "fraction_of_unique_forms": forms_in_band / unique_forms,
            "tokens_contributed": tokens_in_band,
            "fraction_of_tokens": tokens_in_band / total_tokens,
        }
    if sum(int(item["unique_forms"]) for item in result.values()) != unique_forms:
        raise ValueError("frequency bands do not partition the vocabulary")
    if sum(int(item["tokens_contributed"]) for item in result.values()) != total_tokens:
        raise ValueError("frequency bands do not partition the tokens")
    return result


def concentration_metrics(counts: Mapping[str, int]) -> dict[str, object]:
    probabilities = normalized_probabilities(counts)
    forms = tuple(probabilities)
    entropy = -math.fsum(
        probabilities[form] * math.log2(probabilities[form]) for form in forms
    )
    vocabulary_size = len(probabilities)
    normalized_entropy = entropy / math.log2(vocabulary_size) if vocabulary_size > 1 else 0.0
    total = sum(counts.values())
    ordered_counts = sorted(counts.values(), reverse=True)
    top_k_mass: dict[str, dict[str, int | float]] = {}
    for requested_k in (10, 50, 100, 500):
        effective_k = min(requested_k, vocabulary_size)
        top_k_mass[f"k_{requested_k}"] = {
            "requested_k": requested_k,
            "effective_k": effective_k,
            "token_mass": sum(ordered_counts[:effective_k]) / total,
        }
    return {
        "top_k_token_mass": top_k_mass,
        "shannon_entropy_bits": entropy,
        "normalized_shannon_entropy": normalized_entropy,
        "simpson_concentration": math.fsum(probabilities[form] ** 2 for form in forms),
        "effective_vocabulary_size_shannon": 2.0**entropy,
    }


def dataset_distribution_report(dataset: DatasetFrequency) -> dict[str, object]:
    return {
        "row_count": dataset.row_count,
        "total_word_tokens": dataset.total_tokens,
        "unique_word_forms": len(dataset.token_counts),
        "frequency_of_frequency": frequency_of_frequency_profile(dataset.token_counts),
        "concentration": concentration_metrics(dataset.token_counts),
    }


def pairwise_point_metrics(
    counts_a: Mapping[str, int],
    counts_b: Mapping[str, int],
) -> dict[str, object]:
    return {
        "full_union": full_union_distribution_metrics(counts_a, counts_b),
        "shared_vocabulary_conditional": shared_conditional_distribution_metrics(counts_a, counts_b),
        "shared_form_frequency_ratios": shared_frequency_ratio_metrics(counts_a, counts_b),
        "frequency_rank_similarity": frequency_rank_metrics(counts_a, counts_b),
        "top_k_vocabulary_agreement": {
            f"k_{requested_k}": top_k_agreement_metrics(counts_a, counts_b, requested_k)
            for requested_k in TOP_K_VALUES
        },
    }


def _bootstrap_sample(
    rows: tuple[Counter[str], ...],
    rng: random.Random,
) -> Counter[str]:
    selected_row_counts = Counter(rng.choices(range(len(rows)), k=len(rows)))
    result: Counter[str] = Counter()
    for row_index, multiplicity in selected_row_counts.items():
        for form, count in rows[row_index].items():
            result[form] += multiplicity * count
    if not result:
        raise ValueError("a bootstrap replicate produced no word tokens")
    return result


def _bootstrap_primary_metrics(
    counts_a: Mapping[str, int],
    counts_b: Mapping[str, int],
) -> dict[str, float]:
    full = full_union_distribution_metrics(counts_a, counts_b)
    shared = shared_conditional_distribution_metrics(counts_a, counts_b)
    _, spearman_repeated = spearman_rank_correlation(counts_a, counts_b, minimum_count=2)
    return {
        "full_union_jensen_shannon_divergence_base2": float(
            full["jensen_shannon_divergence_base2"]
        ),
        "shared_conditional_jensen_shannon_divergence_base2": float(
            shared["conditional_jensen_shannon_divergence_base2"]
        ),
        "probability_mass_overlap_coefficient": float(
            full["probability_mass_overlap_coefficient"]
        ),
        "spearman_rank_correlation_count_at_least_2": spearman_repeated,
    }


def _percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("a percentile requires at least one value")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def confidence_interval(values: list[float]) -> dict[str, float]:
    result = {
        "p2_5": _percentile(values, 0.025),
        "p50": _percentile(values, 0.50),
        "p97_5": _percentile(values, 0.975),
    }
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("bootstrap confidence interval contains a non-finite value")
    if not result["p2_5"] <= result["p50"] <= result["p97_5"]:
        raise ValueError("bootstrap confidence interval is not ordered")
    return result


def run_bootstrap_analysis(
    datasets: Mapping[str, DatasetFrequency],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[dict[str, dict[str, dict[str, float]]], dict[str, dict[str, dict[str, float]]]]:
    if replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    expected_keys = set(DATASET_STREAM_OFFSETS)
    if set(datasets) != expected_keys:
        raise ValueError("bootstrap datasets do not match the work-order bindings")

    primary_rngs = {
        key: random.Random(seed + 100 * offset)
        for key, offset in DATASET_STREAM_OFFSETS.items()
    }
    secondary_rngs = {
        key: random.Random(seed + 100 * offset + 1)
        for key, offset in DATASET_STREAM_OFFSETS.items()
    }
    cross_values = {
        pair_key: {
            "full_union_jensen_shannon_divergence_base2": [],
            "shared_conditional_jensen_shannon_divergence_base2": [],
            "probability_mass_overlap_coefficient": [],
            "spearman_rank_correlation_count_at_least_2": [],
        }
        for pair_key, _, _ in PAIR_SPECS
    }
    within_values = {
        key: {
            "full_union_jensen_shannon_divergence_base2": [],
            "shared_conditional_jensen_shannon_divergence_base2": [],
            "probability_mass_overlap_coefficient": [],
            "spearman_rank_correlation_count_at_least_2": [],
        }
        for key in expected_keys
    }

    for _ in range(replicates):
        primary_samples = {
            key: _bootstrap_sample(dataset.row_token_counts, primary_rngs[key])
            for key, dataset in datasets.items()
        }
        secondary_samples = {
            key: _bootstrap_sample(dataset.row_token_counts, secondary_rngs[key])
            for key, dataset in datasets.items()
        }
        for pair_key, key_a, key_b in PAIR_SPECS:
            metrics = _bootstrap_primary_metrics(primary_samples[key_a], primary_samples[key_b])
            for metric, value in metrics.items():
                cross_values[pair_key][metric].append(value)
        for key in expected_keys:
            metrics = _bootstrap_primary_metrics(primary_samples[key], secondary_samples[key])
            for metric, value in metrics.items():
                within_values[key][metric].append(value)

    cross_intervals = {
        pair_key: {
            metric: confidence_interval(values)
            for metric, values in metrics.items()
        }
        for pair_key, metrics in cross_values.items()
    }
    within_intervals = {
        key: {
            metric: confidence_interval(values)
            for metric, values in metrics.items()
        }
        for key, metrics in within_values.items()
    }
    return cross_intervals, within_intervals


def build_aggregate_report(
    scale8000: DatasetFrequency,
    fleurs_v2: DatasetFrequency,
    artur_j: DatasetFrequency,
    *,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, object]:
    datasets = {
        scale8000.binding.key: scale8000,
        fleurs_v2.binding.key: fleurs_v2,
        artur_j.binding.key: artur_j,
    }
    if set(datasets) != set(DATASET_STREAM_OFFSETS):
        raise ValueError("dataset keys do not match the work-order schema")
    cross_bootstrap, within_bootstrap = run_bootstrap_analysis(
        datasets,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )

    pairwise: dict[str, dict[str, object]] = {}
    for pair_key, key_a, key_b in PAIR_SPECS:
        point = pairwise_point_metrics(datasets[key_a].token_counts, datasets[key_b].token_counts)
        within_a_p97_5 = within_bootstrap[key_a][
            "full_union_jensen_shannon_divergence_base2"
        ]["p97_5"]
        within_b_p97_5 = within_bootstrap[key_b][
            "full_union_jensen_shannon_divergence_base2"
        ]["p97_5"]
        within_maximum = max(within_a_p97_5, within_b_p97_5)
        if within_maximum <= 0.0 or not math.isfinite(within_maximum):
            raise ValueError("within-corpus JSD envelope denominator is zero or non-finite")
        cross_median = cross_bootstrap[pair_key][
            "full_union_jensen_shannon_divergence_base2"
        ]["p50"]
        excess_ratio = cross_median / within_maximum
        if not math.isfinite(excess_ratio):
            raise ValueError("excess-divergence ratio is non-finite")
        point.update(
            {
                "dataset_a": key_a,
                "dataset_b": key_b,
                "bootstrap": cross_bootstrap[pair_key],
                "sampling_envelope": {
                    "label": (
                        PAIR_LABEL_WITHIN if cross_median <= within_maximum else PAIR_LABEL_ABOVE
                    ),
                    "cross_corpus_median_jsd": cross_median,
                    "maximum_relevant_within_corpus_jsd_p97_5": within_maximum,
                    "excess_divergence_ratio": excess_ratio,
                },
            }
        )
        pairwise[pair_key] = point

    report: dict[str, object] = {
        "schema_version": "1.0",
        "work_order_id": WORK_ORDER_ID,
        "classification": CLASSIFICATION_COMPLETE,
        "analysis_role": "read_only_aggregate_corpus_diagnostic",
        "normalization": {
            "mode": PRIMARY_MODE,
            "normalizer": NORMALIZER_VERSION,
            "unicode_normalization": "NFC",
            "case_handling": "casefold_lowercase",
            "punctuation": "split",
            "slovenian_diacritics": "preserved",
            "numeric_only_tokens": "excluded",
            "mixed_letter_number_tokens": "retained",
            "lemmatization": "none",
            "stemming": "none",
            "token_policy": PRIMARY_TOKEN_POLICY,
            "analysis_units": "normalized_surface_word_forms_not_lemmas_morphemes_or_tokenizer_units",
        },
        "bootstrap_policy": {
            "unit": BOOTSTRAP_UNIT,
            "replicates": bootstrap_replicates,
            "random_seed": bootstrap_seed,
            "lower_percentile": 2.5,
            "median_percentile": 50.0,
            "upper_percentile": 97.5,
            "sampling": "independent_rows_with_replacement_preserving_original_row_count",
            "pretokenization": "fixed_policy_applied_once_then_sampled_row_counters_reused_exactly",
        },
        "input_bindings": {
            key: {
                "dataset_id": dataset.binding.dataset_id,
                "source_role": dataset.binding.source_role,
                "row_count": dataset.row_count,
                "source_sha256": dataset.binding.source_sha256,
            }
            for key, dataset in datasets.items()
        },
        "datasets": {
            key: dataset_distribution_report(dataset) for key, dataset in datasets.items()
        },
        "pairwise": pairwise,
        "within_corpus_sampling": within_bootstrap,
        "interpretation_policy": {
            "sampling_labels_only": "WITHIN_SAMPLING_ENVELOPE_or_ABOVE_SAMPLING_ENVELOPE",
            "single_metric_equivalence_claim": False,
            "representativeness_claim": False,
            "effect_sizes_and_sampling_envelopes_required": True,
        },
        "safety": {
            "raw_references_committed": False,
            "full_gate_wordlists_committed": False,
            "missing_gate_wordlists_committed": False,
            "top_or_ranked_gate_wordlists_committed": False,
            "individual_gate_statistics_committed": False,
            "local_paths_or_manifests_committed": False,
            "used_for_training_selection": False,
            "used_for_generation_prompting": False,
            "training_run": False,
            "asr_evaluation_run": False,
            "new_text_generated": False,
            "corpus_modified": False,
        },
    }
    assert_aggregate_only_report(report)
    return report


def _clamp_unit(value: float) -> float:
    if value < -NUMERICAL_TOLERANCE or value > 1.0 + NUMERICAL_TOLERANCE:
        raise ValueError("a bounded distribution metric is outside zero to one")
    return min(1.0, max(0.0, value))


def assert_aggregate_only_report(report: Mapping[str, object]) -> None:
    def walk(value: object, *, key: str = "") -> None:
        if key not in EXPECTED_SAFETY_FLAGS and (
            key in FORBIDDEN_REPORT_KEYS
            or any(fragment in key for fragment in FORBIDDEN_KEY_FRAGMENTS)
        ):
            raise RealGateLeakageError(f"aggregate report contains forbidden field: {key}")
        if isinstance(value, Mapping):
            for child_key, child_value in value.items():
                if not isinstance(child_key, str):
                    raise ValueError("aggregate report keys must be strings")
                walk(child_value, key=child_key)
            return
        if isinstance(value, (list, tuple, set)):
            raise RealGateLeakageError("aggregate report must not contain arrays or lexical lists")
        if isinstance(value, str):
            if value.startswith(("/", "file://")) or WINDOWS_ABSOLUTE_PATH.match(value):
                raise RealGateLeakageError("aggregate report contains a local absolute path")
            if "\n" in value or "\r" in value:
                raise ValueError("aggregate report strings must be single-line metadata")
            return
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("aggregate report contains a non-finite number")
        if not isinstance(value, (bool, int, float)) and value is not None:
            raise ValueError("aggregate report contains an unsupported value type")

    walk(report)
    if set(report) != REQUIRED_REPORT_TOP_LEVEL_KEYS:
        raise ValueError("aggregate report has unexpected top-level fields")
    if report.get("classification") not in ALLOWED_CLASSIFICATIONS:
        raise ValueError("aggregate report has an invalid classification")
    safety = report.get("safety")
    if not isinstance(safety, Mapping) or set(safety) != EXPECTED_SAFETY_FLAGS:
        raise ValueError("aggregate report has invalid safety fields")
    if any(safety.get(flag) is not False for flag in EXPECTED_SAFETY_FLAGS):
        raise RealGateLeakageError("aggregate report declares leakage or a prohibited action")


def _percent(value: object) -> str:
    return f"{100.0 * float(value):.2f}%"


def _decimal(value: object, places: int = 4) -> str:
    return f"{float(value):.{places}f}"


def _interval_text(value: object, places: int = 4) -> str:
    if not isinstance(value, Mapping):
        raise ValueError("bootstrap interval is malformed")
    return (
        f"{_decimal(value['p50'], places)} "
        f"[{_decimal(value['p2_5'], places)}, {_decimal(value['p97_5'], places)}]"
    )


def render_markdown(report: Mapping[str, object]) -> str:
    assert_aggregate_only_report(report)
    bindings = _mapping(report["input_bindings"], "input bindings")
    datasets = _mapping(report["datasets"], "dataset summaries")
    pairwise = _mapping(report["pairwise"], "pairwise summaries")
    within = _mapping(report["within_corpus_sampling"], "within-corpus summaries")
    bootstrap_policy = _mapping(report["bootstrap_policy"], "bootstrap policy")

    label_by_key = {
        "scale8000": "scale-8000",
        "fleurs_v2": "FLEURS-v2",
        "artur_j": "ARTUR-J",
    }
    pair_label_by_key = {
        "scale8000_vs_fleurs_v2": "scale-8000 vs FLEURS-v2",
        "scale8000_vs_artur_j": "scale-8000 vs ARTUR-J",
        "fleurs_v2_vs_artur_j": "FLEURS-v2 vs ARTUR-J",
    }
    band_label_by_key = {
        "exactly_1": "exactly 1",
        "exactly_2": "exactly 2",
        "3_to_5": "3–5",
        "6_to_10": "6–10",
        "11_to_50": "11–50",
        "more_than_50": "more than 50",
    }

    lines = [
        "# Scale-8000 vs FLEURS/ARTUR-J Word-Frequency Distribution",
        "",
        f"Classification: `{report['classification']}`",
        "",
        "This Work Order 0049 result is a read-only, aggregate-only corpus diagnostic. "
        "It separates vocabulary support, shared-form frequency, rank, concentration, long-tail, "
        "and finite-sample effects. It is not an equivalence or representativeness claim.",
        "",
        "## Input bindings",
        "",
        "| Dataset | Role | Rows | Source SHA256 |",
        "|---|---|---:|---|",
    ]
    for key in ("scale8000", "fleurs_v2", "artur_j"):
        binding = _mapping(bindings[key], f"{key} binding")
        lines.append(
            f"| {label_by_key[key]} | {binding['source_role']} | {binding['row_count']} | "
            f"`{binding['source_sha256']}` |"
        )
    lines.extend(
        [
            "",
            "Every dataset identity, row count, and SHA256 was checked before analysis. A mismatch "
            "fails closed as `EXPERIMENT_INVALID` before either report is written.",
            "",
            "## Normalization and analysis units",
            "",
            f"- Normalizer: `{NORMALIZER_VERSION}`.",
            "- Analysis units: normalized surface-word forms, not lemmas, morphemes, or tokenizer units.",
            "- NFC normalization, casefold/lowercase, punctuation splitting, and preserved Slovenian diacritics.",
            "- Numeric-only tokens are excluded; mixed letter-number tokens are retained.",
            "- No lemmatization or stemming.",
            "",
            "## Dataset distribution summaries",
            "",
            "| Dataset | Rows | Total word tokens | Unique word forms |",
            "|---|---:|---:|---:|",
        ]
    )
    for key in ("scale8000", "fleurs_v2", "artur_j"):
        dataset = _mapping(datasets[key], f"{key} dataset summary")
        lines.append(
            f"| {label_by_key[key]} | {dataset['row_count']} | {dataset['total_word_tokens']} | "
            f"{dataset['unique_word_forms']} |"
        )

    for key in ("scale8000", "fleurs_v2", "artur_j"):
        dataset = _mapping(datasets[key], f"{key} dataset summary")
        bands = _mapping(dataset["frequency_of_frequency"], f"{key} frequency bands")
        lines.extend(
            [
                "",
                f"### {label_by_key[key]} frequency-of-frequency profile",
                "",
                "| Occurrences | Unique forms | Fraction of forms | Tokens contributed | Fraction of tokens |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for band_key, _, _ in FREQUENCY_BANDS:
            band = _mapping(bands[band_key], f"{key} {band_key} band")
            lines.append(
                f"| {band_label_by_key[band_key]} | {band['unique_forms']} | "
                f"{_percent(band['fraction_of_unique_forms'])} | {band['tokens_contributed']} | "
                f"{_percent(band['fraction_of_tokens'])} |"
            )

    lines.extend(
        [
            "",
            "### Concentration and entropy",
            "",
            "| Dataset | Top-10 mass | Top-50 mass | Top-100 mass | Top-500 mass | Shannon entropy (bits) | Normalized entropy | Simpson concentration | Effective vocabulary |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for key in ("scale8000", "fleurs_v2", "artur_j"):
        dataset = _mapping(datasets[key], f"{key} dataset summary")
        concentration = _mapping(dataset["concentration"], f"{key} concentration")
        top_k = _mapping(concentration["top_k_token_mass"], f"{key} top-k mass")
        lines.append(
            f"| {label_by_key[key]} | {_percent(_mapping(top_k['k_10'], 'top 10')['token_mass'])} | "
            f"{_percent(_mapping(top_k['k_50'], 'top 50')['token_mass'])} | "
            f"{_percent(_mapping(top_k['k_100'], 'top 100')['token_mass'])} | "
            f"{_percent(_mapping(top_k['k_500'], 'top 500')['token_mass'])} | "
            f"{_decimal(concentration['shannon_entropy_bits'], 4)} | "
            f"{_decimal(concentration['normalized_shannon_entropy'], 4)} | "
            f"{_decimal(concentration['simpson_concentration'], 6)} | "
            f"{_decimal(concentration['effective_vocabulary_size_shannon'], 2)} |"
        )

    lines.extend(
        [
            "",
            "Top-k values are aggregate token mass only. Effective k is the available vocabulary "
            "when it is smaller than the requested k.",
            "",
            "## Pairwise full-union distributions",
            "",
            "| Comparison | JSD (base 2) | Hellinger | Total variation | Probability-mass overlap | Support TV component | Shared-frequency TV component |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for pair_key, _, _ in PAIR_SPECS:
        pair = _mapping(pairwise[pair_key], f"{pair_key} pair")
        full = _mapping(pair["full_union"], f"{pair_key} full union")
        lines.append(
            f"| {pair_label_by_key[pair_key]} | "
            f"{_decimal(full['jensen_shannon_divergence_base2'], 4)} | "
            f"{_decimal(full['hellinger_distance'], 4)} | "
            f"{_decimal(full['total_variation_distance'], 4)} | "
            f"{_percent(full['probability_mass_overlap_coefficient'])} | "
            f"{_decimal(full['support_mismatch_tv_component'], 4)} | "
            f"{_decimal(full['shared_frequency_tv_component'], 4)} |"
        )

    lines.extend(
        [
            "",
            "## Pairwise shared-vocabulary conditional distributions",
            "",
            "| Comparison | Shared forms | Shared mass in A | Shared mass in B | Conditional JSD | Conditional Hellinger | Conditional overlap |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for pair_key, _, _ in PAIR_SPECS:
        pair = _mapping(pairwise[pair_key], f"{pair_key} pair")
        shared = _mapping(pair["shared_vocabulary_conditional"], f"{pair_key} shared")
        lines.append(
            f"| {pair_label_by_key[pair_key]} | {shared['shared_forms']} | "
            f"{_percent(shared['shared_form_token_mass_in_a'])} | "
            f"{_percent(shared['shared_form_token_mass_in_b'])} | "
            f"{_decimal(shared['conditional_jensen_shannon_divergence_base2'], 4)} | "
            f"{_decimal(shared['conditional_hellinger_distance'], 4)} | "
            f"{_percent(shared['conditional_probability_mass_overlap_coefficient'])} |"
        )

    lines.extend(
        [
            "",
            "## Shared-form frequency ratios",
            "",
            "Absolute log2 ratios use symmetric `min(p_A, p_B)` mass weighting.",
            "",
            "| Comparison | Weighted mean | Weighted median | Within factor 2 | Within factor 4 | Within factor 10 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for pair_key, _, _ in PAIR_SPECS:
        pair = _mapping(pairwise[pair_key], f"{pair_key} pair")
        ratios = _mapping(pair["shared_form_frequency_ratios"], f"{pair_key} ratios")
        lines.append(
            f"| {pair_label_by_key[pair_key]} | "
            f"{_decimal(ratios['weighted_mean_absolute_log2_ratio'], 4)} | "
            f"{_decimal(ratios['weighted_median_absolute_log2_ratio'], 4)} | "
            f"{_percent(ratios['fraction_of_shared_mass_within_factor_2'])} | "
            f"{_percent(ratios['fraction_of_shared_mass_within_factor_4'])} | "
            f"{_percent(ratios['fraction_of_shared_mass_within_factor_10'])} |"
        )

    lines.extend(
        [
            "",
            "## Frequency-rank similarity",
            "",
            "Spearman ranks use descending counts, average ranks for ties, and codepoint ordering only "
            "as an internal deterministic tie-break. No ranked forms are emitted.",
            "",
            "| Comparison | All shared eligible | All shared Spearman | Count≥2 eligible | Count≥2 Spearman |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for pair_key, _, _ in PAIR_SPECS:
        pair = _mapping(pairwise[pair_key], f"{pair_key} pair")
        ranks = _mapping(pair["frequency_rank_similarity"], f"{pair_key} ranks")
        all_shared = _mapping(ranks["all_shared_forms"], f"{pair_key} all-shared ranks")
        repeated = _mapping(ranks["count_at_least_2_in_both"], f"{pair_key} repeated ranks")
        lines.append(
            f"| {pair_label_by_key[pair_key]} | {all_shared['eligible_forms']} | "
            f"{_decimal(all_shared['spearman_rank_correlation'], 4)} | "
            f"{repeated['eligible_forms']} | "
            f"{_decimal(repeated['spearman_rank_correlation'], 4)} |"
        )

    lines.extend(
        [
            "",
            "## Top-k vocabulary agreement",
            "",
            "Top-k sets are tie-inclusive. For each dataset, the cutoff is the occurrence count at "
            "`min(requested k, vocabulary size)` in descending frequency order; every form at or "
            "above that cutoff is included. Effective k can therefore exceed requested k. Only "
            "aggregate set sizes, cutoffs, expansions, and agreement metrics are emitted.",
            "",
        ]
    )
    for pair_key, _, _ in PAIR_SPECS:
        pair = _mapping(pairwise[pair_key], f"{pair_key} pair")
        top_k = _mapping(pair["top_k_vocabulary_agreement"], f"{pair_key} top-k")
        lines.extend(
            [
                f"### {pair_label_by_key[pair_key]}",
                "",
                "| Requested k | Effective k in A | Cutoff in A | Tie expansion in A | Effective k in B | Cutoff in B | Tie expansion in B | Shared forms | Jaccard | Overlap coefficient |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for requested_k in TOP_K_VALUES:
            item = _mapping(top_k[f"k_{requested_k}"], f"{pair_key} top {requested_k}")
            lines.append(
                f"| {requested_k} | {item['effective_k_in_a']} | "
                f"{item['cutoff_occurrence_count_in_a']} | {item['tie_expansion_count_in_a']} | "
                f"{item['effective_k_in_b']} | {item['cutoff_occurrence_count_in_b']} | "
                f"{item['tie_expansion_count_in_b']} | {item['shared_forms']} | "
                f"{_decimal(item['jaccard_similarity'], 4)} | "
                f"{_decimal(item['overlap_coefficient'], 4)} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Sampling uncertainty",
            "",
            f"Bootstrap unit: row/utterance; replicates: {bootstrap_policy['replicates']}; "
            f"seed: {bootstrap_policy['random_seed']}; interval: 2.5th, 50th, and 97.5th percentiles. "
            "Rows are sampled independently with replacement while preserving each dataset row count.",
            "",
            "### Within-corpus sampling envelopes",
            "",
            "| Dataset | Full-union JSD median [95% interval] | Conditional JSD median [95% interval] | Overlap median [95% interval] | Count≥2 Spearman median [95% interval] |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for key in ("scale8000", "fleurs_v2", "artur_j"):
        item = _mapping(within[key], f"{key} within bootstrap")
        lines.append(
            f"| {label_by_key[key]} | "
            f"{_interval_text(item['full_union_jensen_shannon_divergence_base2'])} | "
            f"{_interval_text(item['shared_conditional_jensen_shannon_divergence_base2'])} | "
            f"{_interval_text(item['probability_mass_overlap_coefficient'])} | "
            f"{_interval_text(item['spearman_rank_correlation_count_at_least_2'])} |"
        )

    lines.extend(
        [
            "",
            "### Cross-corpus bootstrap summaries",
            "",
            "| Comparison | Full-union JSD median [95% interval] | Conditional JSD median [95% interval] | Overlap median [95% interval] | Count≥2 Spearman median [95% interval] | Envelope label | Excess-divergence ratio |",
            "|---|---:|---:|---:|---:|---|---:|",
        ]
    )
    for pair_key, _, _ in PAIR_SPECS:
        pair = _mapping(pairwise[pair_key], f"{pair_key} pair")
        bootstrap = _mapping(pair["bootstrap"], f"{pair_key} bootstrap")
        envelope = _mapping(pair["sampling_envelope"], f"{pair_key} envelope")
        lines.append(
            f"| {pair_label_by_key[pair_key]} | "
            f"{_interval_text(bootstrap['full_union_jensen_shannon_divergence_base2'])} | "
            f"{_interval_text(bootstrap['shared_conditional_jensen_shannon_divergence_base2'])} | "
            f"{_interval_text(bootstrap['probability_mass_overlap_coefficient'])} | "
            f"{_interval_text(bootstrap['spearman_rank_correlation_count_at_least_2'])} | "
            f"`{envelope['label']}` | {_decimal(envelope['excess_divergence_ratio'], 2)} |"
        )

    lines.extend(["", "## Pairwise interpretation", ""])
    for pair_key, key_a, key_b in PAIR_SPECS:
        pair = _mapping(pairwise[pair_key], f"{pair_key} pair")
        full = _mapping(pair["full_union"], f"{pair_key} full union")
        shared = _mapping(pair["shared_vocabulary_conditional"], f"{pair_key} shared")
        ratios = _mapping(pair["shared_form_frequency_ratios"], f"{pair_key} ratios")
        ranks = _mapping(pair["frequency_rank_similarity"], f"{pair_key} ranks")
        repeated = _mapping(ranks["count_at_least_2_in_both"], f"{pair_key} repeated ranks")
        envelope = _mapping(pair["sampling_envelope"], f"{pair_key} envelope")
        dominance = (
            "vocabulary-support mismatch"
            if full["dominant_tv_component"] == "support_mismatch"
            else "different frequencies among shared forms"
        )
        lines.extend(
            [
                f"### {pair_label_by_key[pair_key]}",
                "",
                f"1. Shared probability mass: {_percent(full['probability_mass_overlap_coefficient'])}.",
                f"2. Full-distribution difference: JSD {_decimal(full['jensen_shannon_divergence_base2'], 4)}, "
                f"Hellinger {_decimal(full['hellinger_distance'], 4)}, and total variation "
                f"{_decimal(full['total_variation_distance'], 4)}.",
                f"3. Shared-form frequency difference: conditional JSD "
                f"{_decimal(shared['conditional_jensen_shannon_divergence_base2'], 4)}; "
                f"{_percent(ratios['fraction_of_shared_mass_within_factor_2'])} of symmetric shared mass "
                "is within a factor of two.",
                f"4. Frequent shared-form ordering: count≥2 Spearman "
                f"{_decimal(repeated['spearman_rank_correlation'], 4)} over "
                f"{repeated['eligible_forms']} eligible forms.",
                f"5. Sampling envelope: `{envelope['label']}`; cross-corpus median JSD is "
                f"{_decimal(envelope['excess_divergence_ratio'], 2)} times the maximum relevant "
                "within-corpus JSD p97.5.",
                f"6. Total-variation decomposition is larger for {dominance}.",
                "",
            ]
        )

    lines.extend(
        [
            "These labels refer only to the predeclared bootstrap divergence envelope. Statistical "
            "separation is not an equivalence, identity, language-distribution, or representativeness claim. "
            "Effect sizes and sampling envelopes remain visible for interpretation.",
            "",
            "## Safety",
            "",
            "- No raw FLEURS or ARTUR-J references are committed.",
            "- No full, missing, top-k, or ranked real-gate word lists are committed.",
            "- No per-word frequencies, ratios, or bootstrap values are committed.",
            "- No local paths or manifests are committed.",
            "- No gate information was used for prompting, selection, curriculum construction, or training.",
            "- No text generation, corpus modification, model training, or ASR evaluation was run.",
            "",
            "## Known limitations",
            "",
            "- Surface forms only; no lemmatization.",
            "- Empirical unigram analysis only.",
            "- The small ARTUR-J sample produces wider uncertainty.",
            "- Frequency similarity does not prove acoustic or ASR similarity.",
            "- Frequency-of-frequency bands, empirical entropy, and observed support depend on corpus "
            "sample size; they are descriptive and not directly size-matched estimates.",
            "- The row bootstrap measures resampling stability conditional on these observed corpora. "
            "It does not correct corpus/domain-selection bias or establish population-level representativeness.",
            "",
        ]
    )
    return "\n".join(lines)


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} is malformed")
    return value


def assert_markdown_aggregate_only(markdown: str) -> None:
    if "\x00" in markdown:
        raise ValueError("Markdown contains a control character")
    if re.search(r"(?:^|[\s`(])(?:/(?!/)|[A-Za-z]:[\\/])", markdown):
        raise RealGateLeakageError("Markdown contains a local absolute path")
    forbidden_markers = (
        "## Raw references",
        "## Missing words",
        "## Top words",
        "## Ranked words",
        "## Per-word frequencies",
    )
    if any(marker in markdown for marker in forbidden_markers):
        raise RealGateLeakageError("Markdown contains a forbidden lexical section")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute aggregate-only word-frequency distributions for Work Order 0049."
    )
    parser.add_argument("--scale8000-text", type=Path, default=DEFAULT_SCALE8000_PATH)
    parser.add_argument("--fleurs-v2-manifest", type=Path, default=DEFAULT_FLEURS_PATH)
    parser.add_argument("--artur-j-manifest", type=Path, default=DEFAULT_ARTUR_PATH)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_MARKDOWN_OUTPUT)
    return parser.parse_args(argv)


def _require_input(path: Path, reason_code: str) -> None:
    if not path.is_file():
        raise InputUnavailableError(reason_code)


def run_analysis(args: argparse.Namespace) -> dict[str, object]:
    _require_input(args.scale8000_text, BLOCKED_SCALE8000)
    _require_input(args.fleurs_v2_manifest, BLOCKED_FLEURS)
    _require_input(args.artur_j_manifest, BLOCKED_ARTUR)
    scale8000 = load_dataset_frequency(
        args.scale8000_text,
        key="scale8000",
        dataset_id=SCALE8000_DATASET_ID,
        source_role="synthetic_training_text",
        text_field="target_text",
        expected_rows=EXPECTED_SCALE8000_ROWS,
        expected_sha256=EXPECTED_SCALE8000_SHA256,
    )
    fleurs_v2 = load_dataset_frequency(
        args.fleurs_v2_manifest,
        key="fleurs_v2",
        dataset_id=FLEURS_V2_DATASET_ID,
        source_role="immutable_real_gate",
        text_field="text",
        expected_rows=EXPECTED_FLEURS_V2_ROWS,
        expected_sha256=EXPECTED_FLEURS_V2_SHA256,
        expected_dataset_field=FLEURS_V2_DATASET_ID,
    )
    artur_j = load_dataset_frequency(
        args.artur_j_manifest,
        key="artur_j",
        dataset_id=ARTUR_J_DATASET_ID,
        source_role="immutable_real_gate",
        text_field="text",
        expected_rows=EXPECTED_ARTUR_J_ROWS,
        expected_sha256=EXPECTED_ARTUR_J_SHA256,
        expected_dataset_field=ARTUR_J_DATASET_ID,
    )
    report = build_aggregate_report(scale8000, fleurs_v2, artur_j)
    markdown = render_markdown(report)
    assert_markdown_aggregate_only(markdown)
    atomic_write_json(args.output_json, report)
    atomic_write_text(args.output_markdown, markdown)
    return report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = run_analysis(args)
    except InputUnavailableError as error:
        print(f"{CLASSIFICATION_BLOCKED}: {error.reason_code}", file=sys.stderr)
        return 2
    except RealGateLeakageError as error:
        print(f"{CLASSIFICATION_LEAKAGE}: {error}", file=sys.stderr)
        return 1
    except (AssertionError, ValueError) as error:
        print(f"{CLASSIFICATION_INVALID}: {error}", file=sys.stderr)
        return 1
    print(report["classification"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
