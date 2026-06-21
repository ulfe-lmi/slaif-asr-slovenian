from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


REQUIRED_SLOVENIAN_AUDIT_TEXTS = [
    ("slovenian_lowercase", "abcčdefghijklmnoprsštuvzž"),
    ("slovenian_uppercase", "ABCČDEFGHIJKLMNOPRSŠTUVZŽ"),
    ("diacritics_sentence", "Čez cesto švigne žaba."),
    ("punctuation_capitalization", "Ljubljana, 21. junij 2026."),
    ("ordinary_mixed_text", "Zaženi Docker Compose in preveri GPU."),
]
EXTENDED_SYMBOL_AUDIT_TEXTS = [
    ("extended_symbols", "Cena je 12,50 €, temperatura pa 23,7 °C."),
]
SLOVENIAN_AUDIT_TEXTS = [text for _, text in REQUIRED_SLOVENIAN_AUDIT_TEXTS + EXTENDED_SYMBOL_AUDIT_TEXTS]


@dataclass(frozen=True)
class TokenizerAuditRecord:
    category: str
    text: str
    ids: list[int]
    decoded_text: str
    passed: bool
    required: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TokenizerAuditReport:
    tokenizer_class: str
    records: list[TokenizerAuditRecord]

    @property
    def required_slovenian_passed(self) -> bool:
        return all(record.passed for record in self.records if record.required)

    @property
    def all_samples_passed(self) -> bool:
        return all(record.passed for record in self.records)

    @property
    def warnings(self) -> list[str]:
        return [
            f"{record.category}: decoded output differs from input"
            for record in self.records
            if not record.required and not record.passed
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "tokenizer_class": self.tokenizer_class,
            "required_slovenian_passed": self.required_slovenian_passed,
            "all_samples_passed": self.all_samples_passed,
            "warnings": self.warnings,
            "records": [record.to_dict() for record in self.records],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def audit_tokenizer(tokenizer: Any, texts: list[str] | None = None) -> TokenizerAuditReport:
    if texts is None:
        samples = [(category, text, True) for category, text in REQUIRED_SLOVENIAN_AUDIT_TEXTS]
        samples.extend((category, text, False) for category, text in EXTENDED_SYMBOL_AUDIT_TEXTS)
    else:
        samples = [(f"sample_{index}", text, True) for index, text in enumerate(texts)]
    records = []
    for category, text, required in samples:
        ids = encode_text(tokenizer, text)
        decoded = decode_ids(tokenizer, ids)
        records.append(
            TokenizerAuditRecord(
                category=category,
                text=text,
                ids=ids,
                decoded_text=decoded,
                passed=decoded == text,
                required=required,
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
