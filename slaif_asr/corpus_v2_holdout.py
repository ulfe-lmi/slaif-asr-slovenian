from __future__ import annotations

import csv
import hashlib
import json
import re
import statistics
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

from slaif_asr.corpus_v2_generation import (
    GpuMonitor,
    ParsedLine,
    Rejection,
    assert_public_summary_safe as assert_generation_summary_safe,
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_text,
    config_sha256,
    extract_utterance_lines,
    filter_records,
    fingerprint_counts,
    load_jsonl,
    rejection_counts,
    resolve_repo_path,
    summarize_gpu_monitor,
    write_rejections,
)
from slaif_asr.data_quality import (
    ALGORITHM_VERSION,
    assert_privacy_safe_report,
    build_protected_index_payload,
    canonical_json_sha256,
    entity_masked_form,
    fingerprint_hash,
    load_json,
    number_masked_form,
    sha256_file,
    surface_form,
    validate_corpus,
)


HOLDOUT_GENERATOR_VERSION = "gams-corpus-v2-independent-holdout-v1"
PUBLIC_REPORT_SCHEMA_VERSION = "1.0"
TEXT_RECORD_SCHEMA_VERSION = "2.0"
HOLDOUT_CORPUS_ID = "sl-corpus-v2-independent-synthetic-holdout-v1"
EXPECTED_CANDIDATE_SOURCE_SHA256 = "b8a5e4769ef881e90e94f45e36cb4bdbabd24feac0ebcb804fcf5fe760a301d6"
EXPECTED_CANDIDATE_SOURCE_ROWS = 415


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(path: Path) -> dict[str, Any]:
    config = load_json(path)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    if config.get("corpus_id") != HOLDOUT_CORPUS_ID:
        raise ValueError("unexpected holdout corpus_id")
    if config.get("partition_role") != "synthetic_holdout":
        raise ValueError("holdout partition_role must be synthetic_holdout")
    if int(config.get("requested_generation_rows", 0)) != 160:
        raise ValueError("requested_generation_rows must be 160")
    if int(config.get("fixed_holdout_rows", 0)) != 96:
        raise ValueError("fixed_holdout_rows must be 96")
    model = config.get("model", {})
    if model.get("repository") != "cjvt/GaMS-9B-Instruct":
        raise ValueError("holdout generator must be cjvt/GaMS-9B-Instruct")
    if model.get("revision") != "292744023fa0b7ccc7ae2c3c885a67468e49fa03":
        raise ValueError("unexpected GaMS-9B revision")
    quant = config.get("quantization", {})
    if quant.get("load_in_4bit") is not True or quant.get("quant_type") != "nf4":
        raise ValueError("GaMS holdout generation must use 4-bit NF4")
    if quant.get("double_quantization") is not True or quant.get("compute_dtype") != "bfloat16":
        raise ValueError("GaMS holdout generation must use double quantization and BF16 compute")
    device = config.get("device_policy", {})
    if device.get("visible_gpu_count") != 1 or device.get("cpu_offload") is not False or device.get("disk_offload") is not False:
        raise ValueError("holdout generation must use one GPU and forbid offload")
    generation = config.get("generation", {})
    if int(generation.get("prompt_batch_size", 0)) != 4:
        raise ValueError("prompt_batch_size must default to 4")
    cells = config.get("prompt_cells", [])
    if len(cells) != 8:
        raise ValueError("exactly eight holdout prompt cells are required")
    seen: set[str] = set()
    requested_total = 0
    selected_total = 0
    for cell in cells:
        cell_id = str(cell.get("cell_id", ""))
        if not re.fullmatch(r"holdout-cell\d{2}", cell_id):
            raise ValueError(f"unsafe holdout cell_id {cell_id!r}")
        if cell_id in seen:
            raise ValueError(f"duplicate holdout cell_id {cell_id}")
        seen.add(cell_id)
        for key in ("domain", "register", "length_target", "phenomena", "source_family_id", "prompt_revision", "seed_sequence"):
            if key not in cell:
                raise ValueError(f"{cell_id}: missing {key}")
        if int(cell.get("requested_rows", 0)) != 20:
            raise ValueError(f"{cell_id}: requested_rows must be 20")
        if int(cell.get("selected_rows", 0)) != 12:
            raise ValueError(f"{cell_id}: selected_rows must be 12")
        if int(cell.get("maximum_retries", -1)) != 2:
            raise ValueError(f"{cell_id}: maximum_retries must be 2")
        if len(cell.get("seed_sequence", [])) != 3:
            raise ValueError(f"{cell_id}: seed_sequence must include one initial attempt and two retries")
        requested_total += int(cell["requested_rows"])
        selected_total += int(cell["selected_rows"])
    if requested_total != int(config["requested_generation_rows"]):
        raise ValueError("requested_generation_rows must equal cell requested rows")
    if selected_total != int(config["fixed_holdout_rows"]):
        raise ValueError("fixed_holdout_rows must equal cell selected rows")


def run_dir(config: dict[str, Any]) -> Path:
    return resolve_repo_path(config["run_directory"])


def raw_generation_dir(config: dict[str, Any]) -> Path:
    return run_dir(config) / "raw-generation"


def generated_all_path(config: dict[str, Any]) -> Path:
    return run_dir(config) / "generated-all.local.jsonl"


def fixed_holdout_path(config: dict[str, Any]) -> Path:
    return run_dir(config) / "fixed-holdout.local.jsonl"


def rejected_path(config: dict[str, Any]) -> Path:
    return run_dir(config) / "rejected.local.jsonl"


def validation_report_path(config: dict[str, Any]) -> Path:
    return run_dir(config) / "validation.local.json"


def validation_local_review_path(config: dict[str, Any]) -> Path:
    return run_dir(config) / "validation-review.local.jsonl"


def candidate_review_for_validation_path(config: dict[str, Any]) -> Path:
    return run_dir(config) / "candidate-review-for-validation.local.jsonl"


def review_capsule_tsv_path(config: dict[str, Any]) -> Path:
    return run_dir(config) / "review-capsule.local.tsv"


def review_capsule_md_path(config: dict[str, Any]) -> Path:
    return run_dir(config) / "review-capsule.local.md"


def whole_file_command_path(config: dict[str, Any]) -> Path:
    return run_dir(config) / "whole-file-decision-command.local.txt"


def gpu_monitor_path(config: dict[str, Any]) -> Path:
    return run_dir(config) / "gpu-monitor.local.csv"


def generation_summary_path(config: dict[str, Any]) -> Path:
    return run_dir(config) / "generation-summary.local.json"


def public_json_report_path(config: dict[str, Any]) -> Path:
    return resolve_repo_path(config["public_reports"]["json"])


def public_markdown_report_path(config: dict[str, Any]) -> Path:
    return resolve_repo_path(config["public_reports"]["markdown"])


def candidate_source_path(config: dict[str, Any]) -> Path:
    return resolve_repo_path(config["candidate_source"]["accepted_candidates_path"])


def candidate_review_path(config: dict[str, Any]) -> Path:
    return resolve_repo_path(config["candidate_source"]["accepted_review_path"])


def prompt_cell_by_id(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(cell["cell_id"]): cell for cell in config["prompt_cells"]}


def build_prompt(cell: dict[str, Any], *, requested_rows: int) -> str:
    phenomena = ", ".join(str(item).replace("_", " ") for item in cell["phenomena"])
    requirements = [
        "Povedi morajo biti naravne, samostojne in slovenske.",
        "Morfologija, ujemanje, skloni in predlogi morajo biti pravilni.",
        "Pomen mora biti verjeten za vsakdanjo govorjeno rabo.",
        "Uporabi raznoliko besedišče in raznolike stavčne zgradbe.",
        "Vsako poved napiši v svojo vrstico.",
        "Ne uporabljaj oštevilčenja, alinej, JSON-a, oznak ali razlag.",
        "Ne uporabljaj umetnih ovojev, ponavljajočih se začetkov ali skupnih repov.",
        "V poved ne vstavljaj identifikatorjev korpusa, vrstic, skupin, serij, kandidatov, postaj ali vzorcev.",
        "Ne kopiraj besedila iz evalvacijskih zbirk ali drugih znanih korpusov.",
        "Kvote ne zapolni s ponavljanjem istega stavčnega okvira.",
    ]
    return "\n".join(
        [
            "Naloga: predlagaj slovenske povedi za neodvisni sintetični diagnostični holdout za ASR.",
            f"Število povedi: {requested_rows}",
            f"Področje: {cell['domain']}",
            f"Register: {cell['register']}",
            f"Ciljna dolžina: {cell['length_target']}",
            f"Pojavi, ki naj bodo naravno zastopani: {phenomena}",
            "",
            "Zahteve:",
            *[f"- {item}" for item in requirements],
            "",
            "Odgovor naj vsebuje samo povedi, po eno v vsaki vrstici.",
        ]
    )


def prompt_mentions_forbidden_source(prompt: str, config: dict[str, Any]) -> bool:
    lowered = prompt.casefold()
    forbidden = {
        "gamsv2",
        "candidate_id",
        "source_family_id",
        "fleurs",
        "artur",
        config["candidate_source"]["corpus_id"].casefold(),
    }
    forbidden.update(str(cell["cell_id"]).casefold() for cell in config["prompt_cells"])
    forbidden.update(str(cell["source_family_id"]).casefold() for cell in config["prompt_cells"])
    return any(item in lowered for item in forbidden)


def attempt_id(cell_id: str, attempt_index: int) -> str:
    return f"{cell_id}-attempt-{attempt_index:02d}"


def candidate_id(cell_id: str, attempt_index: int, output_ordinal: int) -> str:
    compact_cell = cell_id.replace("holdout-", "h")
    return f"gams9holdout-{compact_cell}-a{attempt_index:02d}-o{output_ordinal:03d}"


def source_id(cell_id: str, attempt_index: int, output_ordinal: int) -> str:
    return f"source-{candidate_id(cell_id, attempt_index, output_ordinal)}"


def build_record(
    *,
    config: dict[str, Any],
    cell: dict[str, Any],
    attempt_index: int,
    output_ordinal: int,
    text: str,
    extraction_mode: str,
) -> dict[str, Any]:
    seed = int(cell["seed_sequence"][attempt_index])
    cid = candidate_id(str(cell["cell_id"]), attempt_index, output_ordinal)
    return {
        "schema_version": TEXT_RECORD_SCHEMA_VERSION,
        "candidate_id": cid,
        "language": "sl-SI",
        "spoken_text": text,
        "target_text": text,
        "partition_role": config["partition_role"],
        "source_type": config["source_type"],
        "source_id": source_id(str(cell["cell_id"]), attempt_index, output_ordinal),
        "source_family_id": cell["source_family_id"],
        "template_family_id": None,
        "utterance_family_id": cid,
        "phenomena": list(cell["phenomena"]),
        "domain": cell["domain"],
        "license": config["model"]["license"],
        "generation": {
            "system": "project-generated",
            "method": "gams-local-text-proposal",
            "corpus_id": config["corpus_id"],
            "model_repository": config["model"]["repository"],
            "model_revision": config["model"]["revision"],
            "prompt_revision": cell["prompt_revision"],
            "corpus_prompt_revision": config["prompt_revision"],
            "seed": seed,
            "prompt_cell": cell["cell_id"],
            "generation_attempt": attempt_id(str(cell["cell_id"]), attempt_index),
            "extraction_mode": extraction_mode,
            "quantization_policy": config["quantization"]["policy"],
            "source_strategy": "independent-holdout-gams-9b",
        },
        "entities": [],
        "minimal_pair": None,
    }


def output_text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def planned_prompts_for_attempt(config: dict[str, Any], admissible_by_cell: dict[str, int], attempt_index: int) -> list[dict[str, Any]]:
    prompts: list[dict[str, Any]] = []
    for cell in config["prompt_cells"]:
        cell_id = str(cell["cell_id"])
        selected_target = int(cell["selected_rows"])
        if int(admissible_by_cell.get(cell_id, 0)) >= selected_target:
            continue
        if attempt_index > int(cell["maximum_retries"]):
            continue
        requested = int(cell["requested_rows"]) if attempt_index == 0 else max(0, selected_target - int(admissible_by_cell.get(cell_id, 0)))
        if requested <= 0:
            continue
        prompt = build_prompt(cell, requested_rows=requested)
        if prompt_mentions_forbidden_source(prompt, config):
            raise ValueError(f"{cell_id}: prompt leaks protected or corpus bookkeeping terms")
        prompts.append(
            {
                "cell_id": cell_id,
                "attempt_index": attempt_index,
                "seed": int(cell["seed_sequence"][attempt_index]),
                "prompt": prompt,
                "requested_rows": requested,
            }
        )
    return prompts


def load_candidate_source(config: dict[str, Any]) -> list[dict[str, Any]]:
    path = candidate_source_path(config)
    digest = sha256_file(path)
    expected_digest = str(config["candidate_source"]["sha256"])
    if digest != expected_digest:
        raise ValueError(f"candidate source SHA256 mismatch: {digest}")
    rows = load_jsonl(path)
    expected_rows = int(config["candidate_source"]["rows"])
    if len(rows) != expected_rows:
        raise ValueError(f"candidate source row count mismatch: {len(rows)}")
    return rows


def load_candidate_reviews(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows = load_jsonl(candidate_review_path(config))
    if len(rows) != int(config["candidate_source"]["rows"]):
        raise ValueError(f"candidate review row count mismatch: {len(rows)}")
    return rows


def candidate_overlap_rejections(records: Sequence[dict[str, Any]], candidate_rows: Sequence[dict[str, Any]]) -> list[Rejection]:
    candidate_surface = {fingerprint_hash(surface_form(str(row["spoken_text"]))) for row in candidate_rows}
    candidate_number = {fingerprint_hash(number_masked_form(str(row["spoken_text"]))) for row in candidate_rows}
    candidate_entity = {fingerprint_hash(entity_masked_form(str(row["spoken_text"]), ())) for row in candidate_rows}
    rejections: list[Rejection] = []
    for row in records:
        text = str(row["spoken_text"])
        cid = str(row["candidate_id"])
        generation = row.get("generation", {}) if isinstance(row.get("generation"), dict) else {}
        cell_id = str(generation.get("prompt_cell", "unknown"))
        attempt = str(generation.get("generation_attempt", "unknown"))
        reason: str | None = None
        if fingerprint_hash(surface_form(text)) in candidate_surface:
            reason = "candidate_source_surface_overlap"
        elif fingerprint_hash(number_masked_form(text)) in candidate_number:
            reason = "candidate_source_number_masked_overlap"
        elif fingerprint_hash(entity_masked_form(text, ())) in candidate_entity:
            reason = "candidate_source_entity_masked_overlap"
        if reason:
            rejections.append(Rejection(reason, cell_id, attempt, candidate_id=cid))
    return rejections


def filter_and_select_fixed_holdout(
    records: Sequence[dict[str, Any]],
    *,
    config: dict[str, Any],
    existing_rejections: Sequence[Rejection] | Sequence[dict[str, Any]],
    protected_indexes: Sequence[Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[Rejection], dict[str, Any]]:
    candidate_rows = load_candidate_source(config)
    retained, rejected, summary = filter_records(
        records,
        config=config,
        existing_rejections=existing_rejections,
        protected_indexes=protected_indexes,
    )
    overlap_rejections = candidate_overlap_rejections(retained, candidate_rows)
    rejected.extend(overlap_rejections)
    overlap_ids = {item.candidate_id for item in overlap_rejections}
    admissible = [row for row in retained if row["candidate_id"] not in overlap_ids]
    fixed, selection_rejections = select_fixed_by_cell(admissible, config)
    rejected.extend(selection_rejections)
    summary = {
        **summary,
        "candidate_source_overlap_rejections": rejection_counts(overlap_rejections),
        "admissible_by_cell": admissible_counts_by_cell(admissible, config),
        "fixed_by_cell": fixed_counts_by_cell(fixed),
    }
    return admissible, fixed, rejected, summary


def admissible_counts_by_cell(records: Sequence[dict[str, Any]], config: dict[str, Any]) -> dict[str, int]:
    by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        generation = row.get("generation", {})
        by_cell[str(generation.get("prompt_cell", "unknown"))].append(row)
    return {str(cell["cell_id"]): len(by_cell.get(str(cell["cell_id"]), [])) for cell in config["prompt_cells"]}


def select_fixed_by_cell(records: Sequence[dict[str, Any]], config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[Rejection]]:
    by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        generation = row.get("generation", {})
        by_cell[str(generation.get("prompt_cell", "unknown"))].append(row)
    fixed: list[dict[str, Any]] = []
    rejected: list[Rejection] = []
    shortfalls: dict[str, int] = {}
    for cell in config["prompt_cells"]:
        cell_id = str(cell["cell_id"])
        selected_rows = int(cell["selected_rows"])
        rows = sorted(
            by_cell.get(cell_id, []),
            key=lambda row: (output_text_hash(str(row["candidate_id"])), str(row["candidate_id"])),
        )
        if len(rows) < selected_rows:
            shortfalls[cell_id] = selected_rows - len(rows)
            continue
        fixed.extend(rows[:selected_rows])
        for overflow in rows[selected_rows:]:
            rejected.append(
                Rejection(
                    "deterministic_selection_overflow",
                    cell_id,
                    str(overflow.get("generation", {}).get("generation_attempt", "unknown")),
                    candidate_id=str(overflow["candidate_id"]),
                )
            )
    if shortfalls:
        raise ValueError(f"holdout cell shortfall after bounded generation: {shortfalls}")
    fixed = sorted(fixed, key=lambda row: str(row["candidate_id"]))
    return fixed, rejected


def fixed_counts_by_cell(records: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in records:
        counts[str(row.get("generation", {}).get("prompt_cell", "unknown"))] += 1
    return dict(sorted(counts.items()))


def ensure_protected_indexes(config: dict[str, Any]) -> tuple[list[Path], list[dict[str, Any]]]:
    index_paths: list[Path] = []
    identities: list[dict[str, Any]] = []
    for item in config["protected_sources"]:
        manifest = resolve_repo_path(item["manifest"])
        metadata = resolve_repo_path(item["metadata"])
        output = resolve_repo_path(item["index"])
        payload = build_protected_index_payload(manifest, metadata)
        if not output.exists() or load_json(output) != payload:
            atomic_write_json(output, payload)
        index_paths.append(output)
        identities.append(
            {
                "gate_id": payload["gate_id"],
                "manifest_sha256": payload["manifest_sha256"],
                "reference_manifest_sha256": payload.get("reference_manifest_sha256"),
                "row_count": payload["row_count"],
                "surface_hash_count": len(payload["surface_hashes"]),
                "number_masked_hash_count": len(payload["number_masked_hashes"]),
                "index_sha256": sha256_file(output),
            }
        )
    return index_paths, identities


def git_revision() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def validate_fixed_holdout(config: dict[str, Any]) -> dict[str, Any]:
    data_quality_config_path = resolve_repo_path(config["data_quality_config"])
    data_quality_config = load_json(data_quality_config_path)
    retired_registry = load_json(resolve_repo_path(config["retired_registry"]))
    protected_index_paths, _identities = ensure_protected_indexes(config)
    candidate_reviews = load_candidate_reviews(config)
    atomic_write_jsonl(candidate_review_for_validation_path(config), candidate_reviews)
    report, local_review_rows = validate_corpus(
        corpus_id=config["validation"]["joint_corpus_id"],
        config=data_quality_config,
        config_sha256=canonical_json_sha256(data_quality_config),
        retired_registry=retired_registry,
        partitions={
            "synthetic_candidate": candidate_source_path(config),
            "synthetic_holdout": fixed_holdout_path(config),
        },
        linguistic_review_path=candidate_review_for_validation_path(config),
        protected_index_paths=protected_index_paths,
        repository_revision=git_revision(),
    )
    assert_privacy_safe_report(report)
    atomic_write_json(validation_report_path(config), report)
    atomic_write_jsonl(validation_local_review_path(config), local_review_rows)
    return report


def expected_holdout_draft(report: dict[str, Any]) -> bool:
    if report.get("final_text_status") != "DRAFT":
        return False
    reasons = set(report.get("decision_reasons", []))
    return reasons == {"missing_linguistic_review"}


def write_review_capsule(config: dict[str, Any]) -> None:
    rows = load_jsonl(fixed_holdout_path(config))
    rows_by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_cell[str(row.get("generation", {}).get("prompt_cell", "unknown"))].append(row)

    review_capsule_tsv_path(config).parent.mkdir(parents=True, exist_ok=True)
    with review_capsule_tsv_path(config).open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "candidate_id",
                "spoken_text",
                "target_text",
                "domain",
                "phenomena",
                "source_family_id",
                "outcome",
                "review_revision",
                "reason_codes",
                "minimal_pair_approved",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for row in sorted(rows, key=lambda item: str(item["candidate_id"])):
            writer.writerow(
                {
                    "candidate_id": row["candidate_id"],
                    "spoken_text": row["spoken_text"],
                    "target_text": row["target_text"],
                    "domain": row["domain"],
                    "phenomena": ",".join(str(item) for item in row["phenomena"]),
                    "source_family_id": row["source_family_id"],
                    "outcome": "",
                    "review_revision": "",
                    "reason_codes": "",
                    "minimal_pair_approved": "False",
                }
            )
    lines = [
        "# Corpus-v2 Independent Synthetic Holdout Review Capsule",
        "",
        "This local file contains raw generated text and must remain uncommitted.",
        "",
        "Human decision required: ACCEPT or REJECT for the exact fixed holdout file.",
        "",
        "Review grammar, agreement, case government, prepositions, entity inflection, semantic plausibility, spoken naturalness, register/domain fit, category correctness, pronunciation plausibility, and transcription correctness.",
        "",
    ]
    for cell in config["prompt_cells"]:
        cell_id = str(cell["cell_id"])
        lines.extend([f"## {cell_id}: {cell['domain']}", ""])
        for row in sorted(rows_by_cell.get(cell_id, []), key=lambda item: str(item["candidate_id"])):
            lines.append(f"- {row['spoken_text']}")
        lines.append("")
    atomic_write_text(review_capsule_md_path(config), "\n".join(lines))
    digest = sha256_file(fixed_holdout_path(config))
    command = "\n".join(
        [
            ".venv/bin/python scripts/admit_reviewed_corpus_v2.py \\",
            f"  --config {config['config_path_for_command']} \\",
            "  --whole-file-outcome <ACCEPT_OR_REJECT> \\",
            "  --review-revision human-holdout-review-v1 \\",
            "  --decision-id human-holdout-decision-v1 \\",
            f"  --expected-corpus-sha256 {digest} \\",
            "  --expected-rows 96 \\",
            "  --require-status TEXT_ACCEPTED",
            "",
        ]
    )
    atomic_write_text(whole_file_command_path(config), command)


def build_public_payload(config: dict[str, Any]) -> dict[str, Any]:
    generated = load_jsonl(generated_all_path(config)) if generated_all_path(config).exists() else []
    fixed = load_jsonl(fixed_holdout_path(config)) if fixed_holdout_path(config).exists() else []
    rejected = load_jsonl(rejected_path(config)) if rejected_path(config).exists() else []
    validation = load_json(validation_report_path(config)) if validation_report_path(config).exists() else None
    protected_indexes = validation.get("protected_indexes", []) if isinstance(validation, dict) else []
    protected_counts = validation.get("protected_overlap_counts", {}) if isinstance(validation, dict) else {}
    fixed_hash = sha256_file(fixed_holdout_path(config)) if fixed_holdout_path(config).exists() else None
    generated_hash = sha256_file(generated_all_path(config)) if generated_all_path(config).exists() else None
    per_cell_generated: Counter[str] = Counter()
    per_cell_fixed: Counter[str] = Counter()
    for row in generated:
        per_cell_generated[str(row.get("generation", {}).get("prompt_cell", "unknown"))] += 1
    for row in fixed:
        per_cell_fixed[str(row.get("generation", {}).get("prompt_cell", "unknown"))] += 1
    generation_summary = load_json(generation_summary_path(config)) if generation_summary_path(config).exists() else {}
    generation_wall_time = float(generation_summary.get("wall_time_seconds") or 0.0)
    generated_rows_per_minute = round(len(generated) / generation_wall_time * 60.0, 3) if generation_wall_time > 0 else None
    retained_rows_per_minute = round(len(fixed) / generation_wall_time * 60.0, 3) if generation_wall_time > 0 else None
    family_counts = (validation or {}).get("template_family_counts", {}).get("synthetic_holdout", {})
    largest_size = int(family_counts.get("largest_discovered_family_size", 0)) if family_counts else 0
    payload = {
        "schema_version": PUBLIC_REPORT_SCHEMA_VERSION,
        "status": "DRAFT — awaiting whole-file human holdout decision",
        "corpus_id": config["corpus_id"],
        "generator_version": HOLDOUT_GENERATOR_VERSION,
        "model": {
            "repository": config["model"]["repository"],
            "revision": config["model"]["revision"],
            "license": config["model"]["license"],
        },
        "configuration_sha256": config_sha256(config),
        "prompt_revision": config["prompt_revision"],
        "prompt_cells": len(config["prompt_cells"]),
        "requested_generation_rows": int(config["requested_generation_rows"]),
        "fixed_holdout_rows": len(fixed),
        "generated_count": len(generated),
        "rejected_count": len(rejected),
        "rejection_counts_by_reason": rejection_counts(rejected),
        "per_cell_counts": {
            "generated": dict(sorted(per_cell_generated.items())),
            "fixed": dict(sorted(per_cell_fixed.items())),
        },
        "candidate_source": {
            "corpus_id": config["candidate_source"]["corpus_id"],
            "sha256": sha256_file(candidate_source_path(config)) if candidate_source_path(config).exists() else None,
            "rows": len(load_jsonl(candidate_source_path(config))) if candidate_source_path(config).exists() else None,
        },
        "holdout_file_hashes": {
            "generated_all_sha256": generated_hash,
            "fixed_holdout_sha256": fixed_hash,
        },
        "fingerprint_unique_counts": fingerprint_counts(fixed) if fixed else {},
        "family_summary": {
            "declared_family_count": family_counts.get("declared_family_count") if family_counts else None,
            "discovered_family_count": family_counts.get("discovered_family_count") if family_counts else None,
            "largest_family_size": largest_size,
            "largest_family_fraction": round(largest_size / max(1, len(fixed)), 6) if fixed else None,
        },
        "cross_partition_overlap_counts": (validation or {}).get("cross_partition_overlap_counts", {}),
        "protected_indexes": protected_indexes,
        "protected_overlap_counts": protected_counts,
        "validator": {
            "status": validation.get("final_text_status") if isinstance(validation, dict) else "NOT_RUN",
            "decision_reasons": validation.get("decision_reasons", []) if isinstance(validation, dict) else [],
            "checks": validation.get("checks", {}) if isinstance(validation, dict) else {},
            "fuzzy_review_pairs": validation.get("fuzzy_review_pair_counts", {}).get("pairs_requiring_review") if isinstance(validation, dict) else None,
        },
        "review_capsule": {
            "rows": len(fixed),
            "outcome_prefilled": False,
            "whole_file_decision_required": True,
        },
        "generation_performance": {
            "wall_time_seconds": round(generation_wall_time, 3) if generation_wall_time > 0 else None,
            "generated_rows_per_minute": generated_rows_per_minute,
            "retained_rows_per_minute": retained_rows_per_minute,
            "prompt_batch_size_used": generation_summary.get("prompt_batch_size_used"),
        },
        "gpu_monitor": summarize_gpu_monitor(gpu_monitor_path(config)) if gpu_monitor_path(config).exists() else {},
        "limitations": [
            "Whole-file human holdout review is outstanding.",
            "No TTS synthesis, ASR scoring, selected-training construction, data certificate, or training was performed.",
            "The holdout is a synthetic diagnostic partition only and is not real-generalization evidence.",
        ],
    }
    assert_public_payload_safe(payload)
    return payload


def assert_public_payload_safe(payload: Any) -> None:
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    forbidden = (
        re.compile(r"\bspoken_text\b|\btarget_text\b"),
        re.compile(r"\bgams9holdout-hcell\d{2}-a\d{2}-o\d{3}\b"),
        re.compile(r"\bgamsv2-cell\d{2}-a\d{2}-o\d{3}\b"),
        re.compile(r"/(?:home|mnt/data|tmp)/"),
    )
    for pattern in forbidden:
        if pattern.search(serialized):
            raise ValueError(f"public holdout report contains forbidden content matching {pattern.pattern}")
    assert_generation_summary_safe(payload)


def write_public_reports(config: dict[str, Any]) -> dict[str, Any]:
    payload = build_public_payload(config)
    atomic_write_json(public_json_report_path(config), payload)
    lines = [
        "# Corpus-v2 Independent Synthetic Holdout",
        "",
        "Status: DRAFT — awaiting whole-file human holdout decision.",
        "",
        "This privacy-safe report does not include raw generated sentences, candidate or holdout IDs, protected references, hypotheses, or local paths.",
        "",
        "## Identity",
        "",
        f"- Corpus ID: `{payload['corpus_id']}`",
        f"- Model: `{payload['model']['repository']}`",
        f"- Revision: `{payload['model']['revision']}`",
        f"- Configuration SHA256: `{payload['configuration_sha256']}`",
        f"- Fixed holdout SHA256: `{payload['holdout_file_hashes']['fixed_holdout_sha256']}`",
        "",
        "## Funnel",
        "",
        f"- Requested generation rows: {payload['requested_generation_rows']}",
        f"- Generated schema rows: {payload['generated_count']}",
        f"- Fixed holdout rows: {payload['fixed_holdout_rows']}",
        f"- Rejected rows: {payload['rejected_count']}",
        f"- Rejection counts: `{json.dumps(payload['rejection_counts_by_reason'], sort_keys=True)}`",
        "",
        "## Validation",
        "",
        f"- Validator status: `{payload['validator']['status']}`",
        f"- Decision reasons: `{', '.join(payload['validator']['decision_reasons']) if payload['validator']['decision_reasons'] else 'not run'}`",
        f"- Cross-partition overlap counts: `{json.dumps(payload['cross_partition_overlap_counts'], sort_keys=True)}`",
        f"- Fuzzy review pairs: {payload['validator']['fuzzy_review_pairs']}",
        "",
        "## Review Capsule",
        "",
        f"- Rows: {payload['review_capsule']['rows']}",
        "- Review outcome prefilled: no",
        "- Required human decision: ACCEPT or REJECT for the exact fixed-holdout hash.",
        "",
        "## A100 Measurement",
        "",
        f"- Generation wall time seconds: {payload['generation_performance'].get('wall_time_seconds', 'not recorded')}",
        f"- Generated rows per minute: {payload['generation_performance'].get('generated_rows_per_minute', 'not recorded')}",
        f"- Retained rows per minute: {payload['generation_performance'].get('retained_rows_per_minute', 'not recorded')}",
        f"- Prompt batch size used: {payload['generation_performance'].get('prompt_batch_size_used', 'not recorded')}",
        f"- Monitor samples: {payload['gpu_monitor'].get('sample_count', 0)}",
        f"- Mean utilization: {payload['gpu_monitor'].get('mean_utilization_percent', 'not recorded')}",
        f"- Median utilization: {payload['gpu_monitor'].get('median_utilization_percent', 'not recorded')}",
        f"- P95 utilization: {payload['gpu_monitor'].get('p95_utilization_percent', 'not recorded')}",
        f"- Fraction >=80%: {payload['gpu_monitor'].get('fraction_at_or_above_80_percent', 'not recorded')}",
        f"- Peak memory MiB: {payload['gpu_monitor'].get('peak_memory_mib', 'not recorded')}",
        "",
        "## Limitations",
        "",
        "- Human holdout review remains outstanding.",
        "- Acoustic suitability remains untested.",
        "- No ASR scoring, training selection, certificate, or model training occurred.",
        "",
    ]
    atomic_write_text(public_markdown_report_path(config), "\n".join(lines))
    return payload


def summarize_generation_metadata(config: dict[str, Any], *, wall_time_seconds: float, used_batch_size: int, fallback_notes: Sequence[str]) -> dict[str, Any]:
    generated = load_jsonl(generated_all_path(config)) if generated_all_path(config).exists() else []
    rejected = load_jsonl(rejected_path(config)) if rejected_path(config).exists() else []
    return {
        "stage": "generate",
        "wall_time_seconds": round(wall_time_seconds, 3),
        "requested_generation_rows": int(config["requested_generation_rows"]),
        "generated_schema_rows": len(generated),
        "rejections": len(rejected),
        "generated_by_cell": fixed_counts_by_cell(generated),
        "prompt_batch_size_used": used_batch_size,
        "batch_fallback_notes": list(fallback_notes),
    }


def parse_generated_outputs(
    *,
    config: dict[str, Any],
    outputs: Sequence[tuple[dict[str, Any], str]],
) -> tuple[list[dict[str, Any]], list[Rejection]]:
    all_records: list[dict[str, Any]] = []
    all_rejections: list[Rejection] = []
    for prompt_meta, raw in outputs:
        cell_id = str(prompt_meta["cell_id"])
        attempt_index = int(prompt_meta["attempt_index"])
        attempt_name = attempt_id(cell_id, attempt_index)
        raw_payload = {
            "cell_id": cell_id,
            "attempt_index": attempt_index,
            "seed": int(prompt_meta["seed"]),
            "requested_rows": int(prompt_meta["requested_rows"]),
            "prompt_sha256": output_text_hash(str(prompt_meta["prompt"])),
            "raw_output": raw,
        }
        atomic_write_json(raw_generation_dir(config) / f"{attempt_name}.json", raw_payload)
        lines, parser_rejections = extract_utterance_lines(raw, cell_id=cell_id, attempt_id=attempt_name)
        all_rejections.extend(parser_rejections)
        cell = prompt_cell_by_id(config)[cell_id]
        for line in lines:
            assert isinstance(line, ParsedLine)
            all_records.append(
                build_record(
                    config=config,
                    cell=cell,
                    attempt_index=attempt_index,
                    output_ordinal=line.output_ordinal,
                    text=line.text,
                    extraction_mode="line",
                )
            )
    return all_records, all_rejections
