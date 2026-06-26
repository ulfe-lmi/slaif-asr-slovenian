from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from slaif_asr.data_quality import atomic_write_json, load_json
from slaif_asr.scale200_corpus import stable_sha256


RECOVERABLE_REJECTION_REASONS = {
    "parser_failure",
    "parser_no_usable_lines",
    "parser_empty_text",
    "parser_json_or_markup",
    "parser_markdown_content",
    "parser_markdown_fence",
    "schema_invalid",
    "metadata_leak",
    "surface_duplicate",
    "number_masked_collision",
    "entity_masked_collision",
    "prohibited_carrier",
    "protected_surface_overlap",
    "protected_number_masked_overlap",
    "holdout_surface_overlap",
    "holdout_number_masked_overlap",
    "holdout_entity_masked_overlap",
    "inherited_surface_overlap",
    "inherited_number_masked_overlap",
    "inherited_entity_masked_overlap",
    "template_concentration",
    "token_shingle_concentration",
    "character_shingle_concentration",
    "per_cell_selection_shortfall",
}

NON_RETRYABLE_FAILURES = {
    "wrong_model_revision",
    "stale_protected_index",
    "wrong_inherited_corpus_hash",
    "wrong_holdout_hash",
    "missing_provenance",
    "cuda_policy_violation",
    "validator_implementation_failure",
    "missing_human_review",
    "human_reject",
}

FORBIDDEN_GUIDANCE_MARKERS = (
    "gamsv2-",
    "gamsv3-",
    "gamsv4-",
    "fleurs",
    "artur",
    "candidate_id",
    "spoken_text",
    "target_text",
)


@dataclass(frozen=True)
class RetryLimits:
    max_verification_rounds: int | None = 8
    max_attempts_per_shard: int | None = 12
    max_attempts_per_cell: int | None = 48
    max_total_attempts: int | None = 1440
    max_requested_rows: int | None = 86400
    requested_rows_per_attempt: int = 60
    max_refill_attempts_per_cell_per_round: int = 5


@dataclass(frozen=True)
class AttemptTask:
    cell_id: str
    shard_id: str
    attempt_index: int
    verification_round: int
    requested_rows: int
    seed: int
    reason: str
    diversity_guidance: tuple[str, ...] = ()

    @property
    def attempt_id(self) -> str:
        return f"{self.cell_id}-{self.shard_id}-attempt-{self.attempt_index:02d}"

    def to_json(self) -> dict[str, Any]:
        return {
            "attempt_id": self.attempt_id,
            "cell_id": self.cell_id,
            "shard_id": self.shard_id,
            "attempt_index": self.attempt_index,
            "verification_round": self.verification_round,
            "requested_rows": self.requested_rows,
            "seed": self.seed,
            "reason": self.reason,
            "diversity_guidance": list(self.diversity_guidance),
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "AttemptTask":
        return cls(
            cell_id=str(payload["cell_id"]),
            shard_id=str(payload["shard_id"]),
            attempt_index=int(payload["attempt_index"]),
            verification_round=int(payload["verification_round"]),
            requested_rows=int(payload["requested_rows"]),
            seed=int(payload["seed"]),
            reason=str(payload["reason"]),
            diversity_guidance=tuple(str(item) for item in payload.get("diversity_guidance", [])),
        )


@dataclass(frozen=True)
class AttemptRecord:
    task: AttemptTask
    status: str
    parsed_rows: int = 0
    retained_rows: int = 0
    rejection_counts: dict[str, int] | None = None
    error: str | None = None

    def to_json(self) -> dict[str, Any]:
        payload = {
            "task": self.task.to_json(),
            "status": self.status,
            "parsed_rows": self.parsed_rows,
            "retained_rows": self.retained_rows,
            "rejection_counts": dict(sorted((self.rejection_counts or {}).items())),
        }
        if self.error:
            payload["error"] = self.error
        return payload

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "AttemptRecord":
        return cls(
            task=AttemptTask.from_json(payload["task"]),
            status=str(payload["status"]),
            parsed_rows=int(payload.get("parsed_rows", 0)),
            retained_rows=int(payload.get("retained_rows", 0)),
            rejection_counts={str(k): int(v) for k, v in payload.get("rejection_counts", {}).items()},
            error=payload.get("error"),
        )


def deterministic_seed(*parts: str) -> int:
    return int(stable_sha256(":".join(parts))[:12], 16) % (2**31 - 1)


def shard_id(index: int) -> str:
    if index < 1:
        raise ValueError("shard indexes are 1-based")
    return f"shard{index:02d}"


def validate_diversity_guidance(items: Sequence[str]) -> tuple[str, ...]:
    cleaned: list[str] = []
    for item in items:
        text = " ".join(str(item).strip().split())
        lowered = text.casefold()
        if not text:
            continue
        if any(marker in lowered for marker in FORBIDDEN_GUIDANCE_MARKERS):
            raise ValueError("retry guidance must not include raw IDs, protected names, or row fields")
        cleaned.append(text)
    return tuple(cleaned)


def initial_tasks(
    cell_ids: Sequence[str],
    *,
    shards_per_cell: int = 9,
    limits: RetryLimits = RetryLimits(),
    seed_namespace: str = "scale2000-gams-v4",
) -> list[AttemptTask]:
    tasks: list[AttemptTask] = []
    for cell_id in sorted(cell_ids):
        for shard_index in range(1, shards_per_cell + 1):
            sid = shard_id(shard_index)
            tasks.append(
                AttemptTask(
                    cell_id=cell_id,
                    shard_id=sid,
                    attempt_index=0,
                    verification_round=0,
                    requested_rows=limits.requested_rows_per_attempt,
                    seed=deterministic_seed(seed_namespace, cell_id, sid, "attempt00"),
                    reason="initial",
                )
            )
    return tasks


def targeted_attempt_count(deficit: int) -> int:
    if deficit <= 0:
        return 0
    return min(5, max(1, math.ceil(deficit / 40)))


class RetryState:
    def __init__(self, records: Iterable[AttemptRecord] = ()) -> None:
        self.records: dict[str, AttemptRecord] = {}
        for record in records:
            self.records[record.task.attempt_id] = record

    @property
    def completed_attempt_ids(self) -> set[str]:
        return {attempt_id for attempt_id, record in self.records.items() if record.status == "completed"}

    def record(self, record: AttemptRecord) -> None:
        self.records[record.task.attempt_id] = record

    def attempts_by_cell(self) -> dict[str, int]:
        counts: Counter[str] = Counter()
        for record in self.records.values():
            counts[record.task.cell_id] += 1
        return dict(sorted(counts.items()))

    def attempts_by_shard(self) -> dict[str, int]:
        counts: Counter[str] = Counter()
        for record in self.records.values():
            counts[f"{record.task.cell_id}:{record.task.shard_id}"] += 1
        return dict(sorted(counts.items()))

    def next_attempt_index(self, cell_id: str, shard: str) -> int:
        current = [
            record.task.attempt_index
            for record in self.records.values()
            if record.task.cell_id == cell_id and record.task.shard_id == shard
        ]
        return max(current, default=-1) + 1

    def requested_rows_used(self) -> int:
        return sum(record.task.requested_rows for record in self.records.values())

    def total_attempts(self) -> int:
        return len(self.records)

    def retry_rounds_used(self) -> int:
        return max((record.task.verification_round for record in self.records.values()), default=0)

    def budget_summary(self, limits: RetryLimits = RetryLimits()) -> dict[str, Any]:
        return {
            "attempts_by_cell": self.attempts_by_cell(),
            "attempts_by_shard": self.attempts_by_shard(),
            "retry_rounds_used": self.retry_rounds_used(),
            "requested_rows_used": self.requested_rows_used(),
            "requested_rows_max": limits.max_requested_rows,
            "total_attempts_used": self.total_attempts(),
            "total_attempts_max": limits.max_total_attempts,
            "exhausted_budgets": exhausted_budgets(self, limits),
        }

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "records": [record.to_json() for record in sorted(self.records.values(), key=lambda item: item.task.attempt_id)],
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "RetryState":
        return cls(AttemptRecord.from_json(item) for item in payload.get("records", []))


def load_state(path: Path) -> RetryState:
    if not path.exists():
        return RetryState()
    return RetryState.from_json(load_json(path))


def save_state(path: Path, state: RetryState) -> None:
    atomic_write_json(path, state.to_json())


def exhausted_budgets(state: RetryState, limits: RetryLimits = RetryLimits()) -> list[str]:
    exhausted: list[str] = []
    if limits.max_total_attempts is not None and state.total_attempts() >= limits.max_total_attempts:
        exhausted.append("total_attempts")
    if limits.max_requested_rows is not None and state.requested_rows_used() >= limits.max_requested_rows:
        exhausted.append("requested_rows")
    if limits.max_attempts_per_cell is not None:
        for cell_id, count in state.attempts_by_cell().items():
            if count >= limits.max_attempts_per_cell:
                exhausted.append(f"cell:{cell_id}")
    if limits.max_attempts_per_shard is not None:
        for shard, count in state.attempts_by_shard().items():
            if count >= limits.max_attempts_per_shard:
                exhausted.append(f"shard:{shard}")
    return exhausted


def plan_refill_tasks(
    shortfalls: dict[str, int],
    *,
    verification_round: int,
    state: RetryState,
    limits: RetryLimits = RetryLimits(),
    diversity_guidance: Sequence[str] = (),
    seed_namespace: str = "scale2000-gams-v4",
) -> list[AttemptTask]:
    if verification_round < 1 or (
        limits.max_verification_rounds is not None and verification_round > limits.max_verification_rounds
    ):
        raise RuntimeError("verification round budget exhausted")
    guidance = validate_diversity_guidance(diversity_guidance)
    if exhausted_budgets(state, limits):
        raise RuntimeError(f"retry budget exhausted: {exhausted_budgets(state, limits)}")

    tasks: list[AttemptTask] = []
    attempts_by_cell = defaultdict(int, state.attempts_by_cell())
    attempts_by_shard = defaultdict(int, state.attempts_by_shard())
    for cell_id, deficit in sorted(shortfalls.items()):
        count = min(limits.max_refill_attempts_per_cell_per_round, targeted_attempt_count(deficit))
        if count <= 0:
            continue
        if limits.max_attempts_per_cell is not None:
            available_cell_budget = limits.max_attempts_per_cell - attempts_by_cell[cell_id]
            if available_cell_budget <= 0:
                raise RuntimeError(f"retry budget exhausted for {cell_id}")
            count = min(count, available_cell_budget)
        for offset in range(count):
            shard = shard_id(((attempts_by_cell[cell_id] + offset) % 9) + 1)
            shard_key = f"{cell_id}:{shard}"
            if limits.max_attempts_per_shard is not None and attempts_by_shard[shard_key] >= limits.max_attempts_per_shard:
                continue
            attempt_index = state.next_attempt_index(cell_id, shard)
            task = AttemptTask(
                cell_id=cell_id,
                shard_id=shard,
                attempt_index=attempt_index,
                verification_round=verification_round,
                requested_rows=limits.requested_rows_per_attempt,
                seed=deterministic_seed(seed_namespace, cell_id, shard, f"attempt{attempt_index:02d}"),
                reason="targeted_refill",
                diversity_guidance=guidance,
            )
            tasks.append(task)
            attempts_by_cell[cell_id] += 1
            attempts_by_shard[shard_key] += 1
            if limits.max_total_attempts is not None and state.total_attempts() + len(tasks) > limits.max_total_attempts:
                raise RuntimeError("total attempt budget would be exceeded")
            if (
                limits.max_requested_rows is not None
                and state.requested_rows_used() + len(tasks) * limits.requested_rows_per_attempt > limits.max_requested_rows
            ):
                raise RuntimeError("requested-row budget would be exceeded")
    return tasks


def rejection_counts_by_reason(records: Sequence[AttemptRecord]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        counts.update(record.rejection_counts or {})
    return dict(sorted(counts.items()))
