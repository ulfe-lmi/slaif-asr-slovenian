from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from slaif_asr.acoustic_quality import corpus_audio_spec
from slaif_asr.batched_streaming import file_sha256
from slaif_asr.config import REPO_ROOT
from slaif_asr.corpus_v2_scoring import (
    CHECKPOINT_SHA256,
    MODEL_REPOSITORY,
    MODEL_REVISION,
    NEMO_REVISION,
    SCORING_AUTHORIZATION_PATH,
    SCORING_RUN_ID,
    assert_public_scoring_payload_safe,
    bucket_counts,
    duration_bucket,
    git_revision,
    load_audio_manifest_rows,
    normalize_sl_asr_text,
    scoring_paths,
    sha256_text,
    verify_scoring_authorization,
)
from slaif_asr.data_quality import atomic_write_json, atomic_write_jsonl, load_jsonl


SELECTION_POLICY_VERSION = "corpus-v2-selected-training-policy-v1"
SELECTED_CERTIFICATE_STATUS = "SELECTED_TRAINING_MANIFEST_READY"
SELECTED_CERTIFICATE_SCHEMA_VERSION = "1.0"
TARGET_HARD = 120
TARGET_CONTROL = 40
PUBLIC_FORBIDDEN_KEYS = {
    "candidate_id",
    "candidate_ids",
    "selected_training_id",
    "sample_id",
    "sample_ids",
    "text",
    "spoken_text",
    "target_text",
    "reference",
    "hypothesis",
    "audio_filepath",
    "local_path",
}
PUBLIC_FORBIDDEN_MARKERS = (
    "gamsv2-",
    "gams9holdout-",
    "/" + "home" + "/",
    "/" + "mnt" + "/" + "data",
    "/" + "tmp" + "/",
)


@dataclass(frozen=True)
class SelectionCandidate:
    sample_id: str
    reference: str
    audio_filepath: str
    duration_seconds: float
    domain: str
    phenomena: tuple[str, ...]
    prompt_cell: str
    source_id: str
    source_family_id: str
    utterance_family_id: str
    discovered_template_family: str
    text_sha256: str
    audio_sha256: str
    normalized_wer: float
    normalized_cer: float
    empty_hypothesis: bool
    deletion_rate: float
    row: dict[str, Any]
    audio_row: dict[str, Any]

    @property
    def hard_score(self) -> float:
        return (1000.0 if self.empty_hypothesis else 0.0) + self.normalized_wer + 0.25 * self.normalized_cer


@dataclass(frozen=True)
class SelectionAttempt:
    relax_domain_cap: bool
    relax_source_family_cap: bool
    relax_cell_minimum: bool
    relax_discovered_family_cap: bool
    selected_count: int
    reason: str


def selection_root() -> Path:
    return REPO_ROOT / "runs/scoring/sl-corpus-v2-v1/selected-training"


def selection_paths() -> dict[str, Path]:
    root = selection_root()
    return {
        "root": root,
        "manifest": root / "selected-training-manifest.local.jsonl",
        "audio_manifest": root / "selected-training-audio-manifest.local.jsonl",
        "selection": root / "selected-training-selection.local.json",
        "report": root / "selected-training-report.local.json",
    }


def load_local_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def require_scoring_completed() -> dict[str, Any]:
    verify_scoring_authorization(require_status="SCORING_AUTHORIZED")
    summaries = {
        "synthetic_candidate": load_local_json(scoring_paths("synthetic_candidate")["summary"]),
        "synthetic_holdout": load_local_json(scoring_paths("synthetic_holdout")["summary"]),
    }
    if summaries["synthetic_candidate"].get("status") != "PASSED":
        raise RuntimeError("candidate-source scoring has not passed")
    if summaries["synthetic_holdout"].get("status") != "PASSED":
        raise RuntimeError("synthetic-holdout scoring has not passed")
    if int(summaries["synthetic_candidate"].get("prediction_count", -1)) != 415:
        raise RuntimeError("candidate-source scoring must have exactly 415 predictions")
    if int(summaries["synthetic_holdout"].get("prediction_count", -1)) != 96:
        raise RuntimeError("synthetic-holdout scoring must have exactly 96 predictions")
    return summaries


def _per_row_path(corpus_role: str) -> Path:
    return scoring_paths(corpus_role)["per_row"]


def load_selection_candidates() -> list[SelectionCandidate]:
    per_rows = load_jsonl(_per_row_path("synthetic_candidate"))
    audio_rows = {str(row["candidate_id"]): row for row in load_audio_manifest_rows("synthetic_candidate")}
    candidates: list[SelectionCandidate] = []
    for row in per_rows:
        sample_id = str(row["sample_id"])
        audio_row = audio_rows.get(sample_id)
        if audio_row is None:
            raise RuntimeError("candidate scoring row missing matching audio row")
        normalized_reference = normalize_sl_asr_text(str(row["reference"]))
        reference_word_count = max(1, len(normalized_reference.split()))
        deletions = int(row["normalized_word_edits"]["deletions"])
        candidates.append(
            SelectionCandidate(
                sample_id=sample_id,
                reference=str(row["reference"]),
                audio_filepath=str(audio_row["audio_filepath"]),
                duration_seconds=float(row["duration_seconds"]),
                domain=str(row["domain"]),
                phenomena=tuple(str(item) for item in row.get("phenomena", [])),
                prompt_cell=str(row["prompt_cell"]),
                source_id=str(row["source_id"]),
                source_family_id=str(row["source_family_id"]),
                utterance_family_id=str(row["utterance_family_id"]),
                discovered_template_family=str(row["discovered_template_family"]),
                text_sha256=str(row["text_sha256"]),
                audio_sha256=str(row["audio_sha256"]),
                normalized_wer=float(row["normalized_wer"]),
                normalized_cer=float(row["normalized_cer"]),
                empty_hypothesis=bool(row["empty_hypothesis"]),
                deletion_rate=deletions / reference_word_count,
                row=row,
                audio_row=audio_row,
            )
        )
    if len(candidates) != 415:
        raise RuntimeError(f"expected 415 selection candidates, found {len(candidates)}")
    return candidates


def stable_hash(value: str) -> str:
    return sha256_text(value)


def ranked_hard_candidates(candidates: Sequence[SelectionCandidate]) -> list[SelectionCandidate]:
    return sorted(
        candidates,
        key=lambda item: (
            -item.hard_score,
            -item.normalized_cer,
            -item.duration_seconds,
            stable_hash(item.sample_id),
            item.sample_id,
        ),
    )


def _can_add(
    candidate: SelectionCandidate,
    selected: Sequence[SelectionCandidate],
    *,
    target: int,
    relax_domain_cap: bool,
    relax_source_family_cap: bool,
    relax_discovered_family_cap: bool,
) -> bool:
    selected_ids = {item.sample_id for item in selected}
    if candidate.sample_id in selected_ids:
        return False
    if candidate.text_sha256 in {item.text_sha256 for item in selected}:
        return False
    if candidate.audio_sha256 in {item.audio_sha256 for item in selected}:
        return False
    if candidate.utterance_family_id in {item.utterance_family_id for item in selected}:
        return False
    if not relax_discovered_family_cap and candidate.discovered_template_family in {
        item.discovered_template_family for item in selected
    }:
        return False
    if not relax_domain_cap:
        domain_cap = max(1, math.floor(target * 0.25))
        if sum(1 for item in selected if item.domain == candidate.domain) >= domain_cap:
            return False
    if not relax_source_family_cap:
        source_cap = max(1, math.floor(target * 0.05))
        if sum(1 for item in selected if item.source_family_id == candidate.source_family_id) >= source_cap:
            return False
    return True


def _try_select_hard(
    candidates: Sequence[SelectionCandidate],
    *,
    target: int,
    relax_domain_cap: bool,
    relax_source_family_cap: bool,
    relax_cell_minimum: bool,
    relax_discovered_family_cap: bool,
) -> list[SelectionCandidate]:
    ranked = ranked_hard_candidates(candidates)
    selected: list[SelectionCandidate] = []
    if not relax_cell_minimum:
        cells = sorted({item.prompt_cell for item in ranked})
        for cell in cells:
            cell_rows = [item for item in ranked if item.prompt_cell == cell]
            for item in cell_rows:
                if len([row for row in selected if row.prompt_cell == cell]) >= 4:
                    break
                if _can_add(
                    item,
                    selected,
                    target=target,
                    relax_domain_cap=relax_domain_cap,
                    relax_source_family_cap=relax_source_family_cap,
                    relax_discovered_family_cap=relax_discovered_family_cap,
                ):
                    selected.append(item)
    for item in ranked:
        if len(selected) >= target:
            break
        if _can_add(
            item,
            selected,
            target=target,
            relax_domain_cap=relax_domain_cap,
            relax_source_family_cap=relax_source_family_cap,
            relax_discovered_family_cap=relax_discovered_family_cap,
        ):
            selected.append(item)
    return selected


def select_hard_examples(candidates: Sequence[SelectionCandidate], *, target: int) -> tuple[list[SelectionCandidate], list[dict[str, Any]]]:
    attempts: list[SelectionAttempt] = []
    plans = [
        (False, False, False, False, "strict"),
        (True, False, False, False, "domain_cap_relaxed"),
        (True, True, False, False, "source_family_cap_relaxed"),
        (True, True, True, False, "cell_minimum_relaxed"),
        (True, True, True, True, "discovered_family_cap_relaxed"),
    ]
    final: list[SelectionCandidate] = []
    for relax_domain, relax_source, relax_cell, relax_discovered, reason in plans:
        selected = _try_select_hard(
            candidates,
            target=target,
            relax_domain_cap=relax_domain,
            relax_source_family_cap=relax_source,
            relax_cell_minimum=relax_cell,
            relax_discovered_family_cap=relax_discovered,
        )
        attempts.append(
            SelectionAttempt(
                relax_domain_cap=relax_domain,
                relax_source_family_cap=relax_source,
                relax_cell_minimum=relax_cell,
                relax_discovered_family_cap=relax_discovered,
                selected_count=len(selected),
                reason=reason,
            )
        )
        final = selected
        if len(selected) >= target:
            final = selected[:target]
            break
    if len(final) != target:
        raise RuntimeError(f"could not select {target} hard examples; selected {len(final)}")
    return final, [attempt.__dict__ for attempt in attempts]


def select_controls(
    candidates: Sequence[SelectionCandidate],
    hard: Sequence[SelectionCandidate],
    *,
    target: int,
) -> list[SelectionCandidate]:
    selected_ids = {item.sample_id for item in hard}
    selected_text = {item.text_sha256 for item in hard}
    selected_audio = {item.audio_sha256 for item in hard}
    selected_utterance = {item.utterance_family_id for item in hard}
    selected_discovered = {item.discovered_template_family for item in hard}
    strata: dict[tuple[str, str], list[SelectionCandidate]] = defaultdict(list)
    for item in candidates:
        if item.sample_id in selected_ids:
            continue
        if item.text_sha256 in selected_text or item.audio_sha256 in selected_audio:
            continue
        if item.utterance_family_id in selected_utterance or item.discovered_template_family in selected_discovered:
            continue
        strata[(item.prompt_cell, item.domain)].append(item)
    for key in list(strata):
        strata[key].sort(key=lambda item: (stable_hash(item.sample_id), item.sample_id))
    controls: list[SelectionCandidate] = []
    keys = sorted(strata)
    while len(controls) < target and keys:
        progressed = False
        for key in list(keys):
            bucket = strata[key]
            while bucket:
                candidate = bucket.pop(0)
                if candidate.text_sha256 in selected_text or candidate.audio_sha256 in selected_audio:
                    continue
                if candidate.utterance_family_id in selected_utterance:
                    continue
                controls.append(candidate)
                selected_text.add(candidate.text_sha256)
                selected_audio.add(candidate.audio_sha256)
                selected_utterance.add(candidate.utterance_family_id)
                selected_discovered.add(candidate.discovered_template_family)
                progressed = True
                break
            if not bucket:
                keys.remove(key)
            if len(controls) == target:
                break
        if not progressed:
            break
    if len(controls) != target:
        raise RuntimeError(f"could not select {target} controls; selected {len(controls)}")
    return controls


def count_by(items: Sequence[SelectionCandidate], field: str) -> dict[str, int]:
    if field == "phenomena":
        counter = Counter()
        for item in items:
            counter.update(item.phenomena)
        return dict(sorted(counter.items()))
    return dict(sorted(Counter(str(getattr(item, field)) for item in items).items()))


def metric_distribution(items: Sequence[SelectionCandidate]) -> dict[str, dict[str, int]]:
    rows = [item.row for item in items]
    return {
        "normalized_wer": bucket_counts(rows, "normalized_wer"),
        "normalized_cer": bucket_counts(rows, "normalized_cer"),
        "duration": dict(sorted(Counter(duration_bucket(item.duration_seconds) for item in items).items())),
    }


def local_selected_rows(
    hard: Sequence[SelectionCandidate],
    controls: Sequence[SelectionCandidate],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected = [("hard", item) for item in hard] + [("control", item) for item in controls]
    manifest_rows: list[dict[str, Any]] = []
    audio_rows: list[dict[str, Any]] = []
    for rank, (reason, item) in enumerate(selected, start=1):
        selected_id = f"sl-corpus-v2-selected-training-v1-{rank:03d}"
        manifest_rows.append(
            {
                "schema_version": "1.0",
                "selected_training_id": selected_id,
                "source_candidate_id": item.sample_id,
                "audio_filepath": item.audio_filepath,
                "duration": item.duration_seconds,
                "text": item.reference,
                "lang": "sl-SI",
                "target_lang": "sl-SI",
                "role": "selected_training",
                "selection_reason": reason,
                "selection_rank": rank,
                "text_sha256": item.text_sha256,
                "audio_sha256": item.audio_sha256,
                "source_corpus_id": corpus_audio_spec("synthetic_candidate").corpus_id,
                "source_text_sha256": corpus_audio_spec("synthetic_candidate").expected_accepted_sha256,
                "source_audio_manifest_sha256": "c1d366e1d05b6f728af51b3350556b6d915fabf5a6b584a6aa2f9fdc0df538bc",
                "scoring_run_id": SCORING_RUN_ID,
                "parent_model": {
                    "repository": MODEL_REPOSITORY,
                    "revision": MODEL_REVISION,
                    "checkpoint_sha256": CHECKPOINT_SHA256,
                },
            }
        )
        audio_rows.append(
            {
                **item.audio_row,
                "selected_training_id": selected_id,
                "source_candidate_id": item.sample_id,
                "role": "selected_training",
                "selection_reason": reason,
                "selection_rank": rank,
                "scoring_run_id": SCORING_RUN_ID,
            }
        )
    return manifest_rows, audio_rows


def hash_aggregate(items: Sequence[SelectionCandidate]) -> str:
    lines = sorted(f"{item.text_sha256}|{item.audio_sha256}" for item in items)
    return sha256_text("\n".join(lines) + "\n")


def holdout_exclusion_counts(selected: Sequence[SelectionCandidate]) -> dict[str, int]:
    holdout_audio = load_audio_manifest_rows("synthetic_holdout")
    holdout_ids = {str(row["candidate_id"]) for row in holdout_audio}
    holdout_text = {str(row["target_text_sha256"]) for row in holdout_audio}
    holdout_audio_hashes = {str(row["audio_sha256"]) for row in holdout_audio}
    return {
        "selected_holdout_id_overlaps": len({item.sample_id for item in selected} & holdout_ids),
        "selected_holdout_text_hash_overlaps": len({item.text_sha256 for item in selected} & holdout_text),
        "selected_holdout_audio_hash_overlaps": len({item.audio_sha256 for item in selected} & holdout_audio_hashes),
    }


def diversity_summary(hard: Sequence[SelectionCandidate], controls: Sequence[SelectionCandidate], all_candidates: Sequence[SelectionCandidate]) -> dict[str, Any]:
    selected = list(hard) + list(controls)
    return {
        "selected_total": len(selected),
        "hard": len(hard),
        "control": len(controls),
        "domain_counts": count_by(selected, "domain"),
        "prompt_cell_counts": count_by(selected, "prompt_cell"),
        "phenomenon_counts": count_by(selected, "phenomena"),
        "duration_bucket_counts": dict(sorted(Counter(duration_bucket(item.duration_seconds) for item in selected).items())),
        "hard_domain_counts": count_by(hard, "domain"),
        "control_domain_counts": count_by(controls, "domain"),
        "source_domain_counts": count_by(all_candidates, "domain"),
        "selected_metric_distribution": metric_distribution(selected),
        "source_metric_distribution": metric_distribution(all_candidates),
        "unique_counts": {
            "text_hashes": len({item.text_sha256 for item in selected}),
            "audio_hashes": len({item.audio_sha256 for item in selected}),
            "utterance_families": len({item.utterance_family_id for item in selected}),
            "discovered_template_families": len({item.discovered_template_family for item in selected}),
        },
    }


def assert_public_selection_payload_safe(payload: Any) -> None:
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)

    def walk(value: Any, key: str = "") -> None:
        if key in PUBLIC_FORBIDDEN_KEYS:
            raise ValueError(f"public selection payload contains forbidden key: {key}")
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                walk(child_value, child_key)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(payload)
    if any(marker in serialized for marker in PUBLIC_FORBIDDEN_MARKERS):
        raise ValueError("public selection payload contains row IDs, local paths, or raw text markers")


def build_selected_training(*, target_hard: int = TARGET_HARD, target_control: int = TARGET_CONTROL, require_status: str | None = None) -> tuple[dict[str, Any], int]:
    scoring_summaries = require_scoring_completed()
    candidates = load_selection_candidates()
    hard, relaxation_attempts = select_hard_examples(candidates, target=target_hard)
    controls = select_controls(candidates, hard, target=target_control)
    selected = hard + controls
    exclusion = holdout_exclusion_counts(selected)
    if any(value != 0 for value in exclusion.values()):
        raise RuntimeError(f"holdout exclusion failed: {exclusion}")
    paths = selection_paths()
    paths["root"].mkdir(parents=True, exist_ok=True)
    manifest_rows, audio_rows = local_selected_rows(hard, controls)
    atomic_write_jsonl(paths["manifest"], manifest_rows)
    atomic_write_jsonl(paths["audio_manifest"], audio_rows)
    manifest_sha = file_sha256(paths["manifest"])
    audio_manifest_sha = file_sha256(paths["audio_manifest"])
    candidate_scoring_sha = file_sha256(scoring_paths("synthetic_candidate")["summary"])
    holdout_scoring_sha = file_sha256(scoring_paths("synthetic_holdout")["summary"])
    selected_hash_aggregate = hash_aggregate(selected)
    diversity = diversity_summary(hard, controls, candidates)
    status = SELECTED_CERTIFICATE_STATUS
    certificate = {
        "schema_version": SELECTED_CERTIFICATE_SCHEMA_VERSION,
        "certificate_id": "sl-corpus-v2-selected-training-v1",
        "status": status,
        "decision_date": "2026-06-24",
        "candidate_source_hashes": {
            "text_partition_sha256": corpus_audio_spec("synthetic_candidate").expected_accepted_sha256,
            "audio_manifest_sha256": "c1d366e1d05b6f728af51b3350556b6d915fabf5a6b584a6aa2f9fdc0df538bc",
            "audio_certificate_sha256": "25737f59397d5c5acdd99e6af83e1129587199cb1a184eaec43dd27139bb1692",
        },
        "holdout_hashes": {
            "text_partition_sha256": corpus_audio_spec("synthetic_holdout").expected_accepted_sha256,
            "audio_manifest_sha256": "7848f57e1fb65a2ef514815eec8092cd0a205b29819f6afeb767ea951473990d",
            "audio_certificate_sha256": "d5c1660b8b11b8b250d04034dfb2abe14a96dda33d48560875a51d7168865297",
        },
        "scoring_authorization_certificate_sha256": file_sha256(SCORING_AUTHORIZATION_PATH),
        "candidate_scoring_run_sha256": candidate_scoring_sha,
        "holdout_scoring_run_sha256": holdout_scoring_sha,
        "selected_row_count": len(selected),
        "hard_count": len(hard),
        "control_count": len(controls),
        "selection_policy_version": SELECTION_POLICY_VERSION,
        "selected_manifest_sha256": manifest_sha,
        "selected_audio_manifest_sha256": audio_manifest_sha,
        "selected_text_audio_hash_aggregate": selected_hash_aggregate,
        "candidate_holdout_exclusion_proof": exclusion,
        "diversity_constraint_results": diversity,
        "relaxation_log": relaxation_attempts,
        "parent_model": {
            "repository": MODEL_REPOSITORY,
            "revision": MODEL_REVISION,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "nemo_revision": NEMO_REVISION,
        },
        "authorized_next_actions": [
            "prepare a later bounded training work order",
            "use this selected-training manifest as a proposed single-voice synthetic training partition only after separate training authorization",
        ],
        "prohibited_actions": [
            "model training in this PR",
            "TRAINING_ELIGIBLE certification",
            "checkpoint promotion",
            "public performance claims",
            "use as real-speech evidence",
        ],
        "limitations": [
            "Selected training is single-voice synthetic Piper audio.",
            "Selected training is not real-speech evidence.",
            "Training remains unauthorized until a later training work order.",
            "The untouched Nemotron base remains the only accepted parent.",
        ],
    }
    assert_public_selection_payload_safe(certificate)
    report = {
        "schema_version": "1.0",
        "report": "corpus-v2-selected-training",
        "status": status,
        "repository_commit": git_revision(),
        "certificate": certificate,
        "source_scoring_metrics": {
            "candidate_source": scoring_summaries["synthetic_candidate"]["aggregate"]["metrics"],
            "synthetic_holdout": scoring_summaries["synthetic_holdout"]["aggregate"]["metrics"],
        },
        "selection_summary": {
            "total_selected": len(selected),
            "hard": len(hard),
            "control": len(controls),
            "metric_distribution": diversity["selected_metric_distribution"],
            "selected_vs_source_metric_distribution": {
                "selected": diversity["selected_metric_distribution"],
                "source": diversity["source_metric_distribution"],
            },
            "holdout_exclusion": exclusion,
        },
    }
    assert_public_selection_payload_safe(report)
    atomic_write_json(paths["selection"], {
        "selected_ids": [item.sample_id for item in selected],
        "hard_ids": [item.sample_id for item in hard],
        "control_ids": [item.sample_id for item in controls],
        "relaxation_attempts": relaxation_attempts,
        "holdout_exclusion": exclusion,
    })
    atomic_write_json(paths["report"], report)
    cert_path = REPO_ROOT / "docs/data-certificates/sl-corpus-v2-selected-training-v1.json"
    report_json = REPO_ROOT / "docs/data-reports/0009-corpus-v2-selected-training.json"
    report_md = REPO_ROOT / "docs/data-reports/0009-corpus-v2-selected-training.md"
    experiment_json = REPO_ROOT / "docs/experiments/0007-corpus-v2-base-scoring-and-selection.json"
    experiment_md = REPO_ROOT / "docs/experiments/0007-corpus-v2-base-scoring-and-selection.md"
    atomic_write_json(cert_path, certificate)
    atomic_write_json(report_json, report)
    write_selected_training_markdown(report_md, report)
    experiment = build_experiment_report(report, scoring_summaries)
    assert_public_selection_payload_safe(experiment)
    atomic_write_json(experiment_json, experiment)
    write_experiment_markdown(experiment_md, experiment)
    result = {
        "status": status,
        "selected_rows": len(selected),
        "hard_count": len(hard),
        "control_count": len(controls),
        "selected_manifest_sha256": manifest_sha,
        "selected_audio_manifest_sha256": audio_manifest_sha,
        "selected_certificate_sha256": file_sha256(cert_path),
        "selected_report_sha256": file_sha256(report_json),
        "experiment_report_sha256": file_sha256(experiment_json),
        "relaxation_log": relaxation_attempts,
        "holdout_exclusion": exclusion,
    }
    return result, 0 if require_status is None or status == require_status else 1


def build_experiment_report(selection_report: dict[str, Any], scoring_summaries: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "experiment_id": "corpus-v2-base-scoring-and-selection-v1",
        "status": "completed in PR; pending strategic review",
        "repository_commit": git_revision(),
        "model": {
            "repository": MODEL_REPOSITORY,
            "revision": MODEL_REVISION,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "nemo_revision": NEMO_REVISION,
        },
        "scoring": {
            "candidate_source": scoring_summaries["synthetic_candidate"]["aggregate"],
            "synthetic_holdout": scoring_summaries["synthetic_holdout"]["aggregate"],
        },
        "selection": selection_report["selection_summary"],
        "certificate": {
            "status": selection_report["certificate"]["status"],
            "selected_row_count": selection_report["certificate"]["selected_row_count"],
            "hard_count": selection_report["certificate"]["hard_count"],
            "control_count": selection_report["certificate"]["control_count"],
            "selected_manifest_sha256": selection_report["certificate"]["selected_manifest_sha256"],
            "selected_audio_manifest_sha256": selection_report["certificate"]["selected_audio_manifest_sha256"],
        },
        "limitations": selection_report["certificate"]["limitations"],
    }


def write_selected_training_markdown(path: Path, payload: dict[str, Any]) -> None:
    cert = payload["certificate"]
    lines = [
        "# Corpus-v2 Selected Training",
        "",
        f"Status: `{cert['status']}`",
        "",
        "This privacy-safe report records selected-training construction from the accepted candidate source only. It does not authorize model training and does not issue `TRAINING_ELIGIBLE`.",
        "",
        "## Selection",
        "",
        f"- Total selected rows: {cert['selected_row_count']}",
        f"- Hard examples: {cert['hard_count']}",
        f"- Controls: {cert['control_count']}",
        f"- Selected manifest SHA256: `{cert['selected_manifest_sha256']}`",
        f"- Selected audio manifest SHA256: `{cert['selected_audio_manifest_sha256']}`",
        f"- Holdout exclusion: `{json.dumps(cert['candidate_holdout_exclusion_proof'], sort_keys=True)}`",
        "",
        "## Constraints",
        "",
        f"- Selection policy: `{cert['selection_policy_version']}`",
        f"- Relaxation attempts: `{json.dumps(cert['relaxation_log'], sort_keys=True)}`",
        "",
        "## Limitations",
        "",
        *[f"- {item}" for item in cert["limitations"]],
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_experiment_markdown(path: Path, payload: dict[str, Any]) -> None:
    candidate = payload["scoring"]["candidate_source"]
    holdout = payload["scoring"]["synthetic_holdout"]
    lines = [
        "# Experiment 0007: Corpus-v2 Base Scoring and Selection",
        "",
        "Status: **completed in PR; pending strategic review**",
        "",
        "This experiment scores the accepted single-voice synthetic candidate source and independent synthetic holdout with the untouched Nemotron base, then constructs a selected-training manifest from the candidate source only. No model training occurred.",
        "",
        "## Model",
        "",
        f"- Repository: `{MODEL_REPOSITORY}`",
        f"- Revision: `{MODEL_REVISION}`",
        f"- Checkpoint SHA256: `{CHECKPOINT_SHA256}`",
        f"- NeMo revision: `{NEMO_REVISION}`",
        "",
        "## Scoring Metrics",
        "",
        "| Partition | Rows | Normalized WER | Normalized CER | Empty hypotheses |",
        "|---|---:|---:|---:|---:|",
        f"| candidate source | {candidate['rows']} | {candidate['metrics']['normalized']['corpus_wer']} | {candidate['metrics']['normalized']['corpus_cer']} | {candidate['metrics']['raw']['empty_hypothesis_count']} |",
        f"| synthetic holdout | {holdout['rows']} | {holdout['metrics']['normalized']['corpus_wer']} | {holdout['metrics']['normalized']['corpus_cer']} | {holdout['metrics']['raw']['empty_hypothesis_count']} |",
        "",
        "## Selected Training",
        "",
        f"- Rows: {payload['certificate']['selected_row_count']}",
        f"- Hard/control: {payload['certificate']['hard_count']} / {payload['certificate']['control_count']}",
        f"- Certificate status: `{payload['certificate']['status']}`",
        "",
        "## Limitations",
        "",
        *[f"- {item}" for item in payload["limitations"]],
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
