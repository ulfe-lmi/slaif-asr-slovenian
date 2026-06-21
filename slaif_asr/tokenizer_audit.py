from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


SLOVENIAN_AUDIT_TEXTS = [
    "abcčdefghijklmnoprsštuvzž",
    "ABCČDEFGHIJKLMNOPRSŠTUVZŽ",
    "Čez cesto švigne žaba.",
    "Ljubljana, 21. junij 2026.",
    "Zaženi Docker Compose in preveri GPU.",
    "Cena je 12,50 €, temperatura pa 23,7 °C.",
]


@dataclass(frozen=True)
class TokenizerAuditRecord:
    text: str
    ids: list[int]
    decoded_text: str
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TokenizerAuditReport:
    tokenizer_class: str
    records: list[TokenizerAuditRecord]

    @property
    def passed(self) -> bool:
        return all(record.passed for record in self.records)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tokenizer_class": self.tokenizer_class,
            "passed": self.passed,
            "records": [record.to_dict() for record in self.records],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def audit_tokenizer(tokenizer: Any, texts: list[str] | None = None) -> TokenizerAuditReport:
    samples = texts or SLOVENIAN_AUDIT_TEXTS
    records = []
    for text in samples:
        ids = encode_text(tokenizer, text)
        decoded = decode_ids(tokenizer, ids)
        records.append(
            TokenizerAuditRecord(
                text=text,
                ids=ids,
                decoded_text=decoded,
                passed=decoded == text,
            )
        )
    return TokenizerAuditReport(
        tokenizer_class=f"{tokenizer.__class__.__module__}.{tokenizer.__class__.__name__}",
        records=records,
    )


def write_audit_report(report: TokenizerAuditReport, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report.to_json(), encoding="utf-8")


def encode_text(tokenizer: Any, text: str) -> list[int]:
    for name in ("text_to_ids", "encode"):
        method = getattr(tokenizer, name, None)
        if method is not None:
            return [int(item) for item in method(text)]
    inner = getattr(tokenizer, "tokenizer", None)
    if inner is not None:
        return encode_text(inner, text)
    raise TypeError("Tokenizer does not expose text_to_ids or encode.")


def decode_ids(tokenizer: Any, ids: list[int]) -> str:
    for name in ("ids_to_text", "decode"):
        method = getattr(tokenizer, name, None)
        if method is not None:
            return str(method(ids))
    inner = getattr(tokenizer, "tokenizer", None)
    if inner is not None:
        return decode_ids(inner, ids)
    raise TypeError("Tokenizer does not expose ids_to_text or decode.")
