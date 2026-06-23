#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slaif_asr.corpus_v2_generation import (
    GpuMonitor,
    Rejection,
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_text,
    build_prompt,
    build_record,
    config_sha256,
    extract_utterance_lines,
    filter_records,
    generated_all_path,
    gpu_monitor_path,
    load_config,
    local_paths,
    output_text_hash,
    prompt_cell_by_id,
    raw_generation_dir,
    rejected_path,
    resolve_repo_path,
    run_dir,
    write_rejections,
)
from slaif_asr.gpu_policy import require_single_visible_cuda


def sanitize_generation_config(model: Any) -> None:
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None


def load_model(config: dict[str, Any]):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_cfg = config["model"]
    quant_cfg = config["quantization"]
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quant_cfg["quant_type"],
        bnb_4bit_use_double_quant=quant_cfg["double_quantization"],
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["repository"],
        revision=model_cfg["revision"],
        trust_remote_code=False,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["repository"],
        revision=model_cfg["revision"],
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        max_memory={0: f"{int(config['device_policy']['max_memory_gib'])}GiB"},
        trust_remote_code=False,
    )
    device_map = getattr(model, "hf_device_map", {})
    if any(str(device) in {"cpu", "disk"} for device in device_map.values()):
        raise RuntimeError(f"CPU or disk offload is forbidden: {device_map}")
    sanitize_generation_config(model)
    return tokenizer, model


def encode_prompts(tokenizer: Any, prompts: list[str], *, max_context_tokens: int = 2048) -> dict[str, Any]:
    rendered: list[str] = []
    for prompt in prompts:
        if getattr(tokenizer, "chat_template", None):
            rendered.append(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    add_generation_prompt=True,
                    tokenize=False,
                )
            )
        else:
            rendered.append(prompt)
    encoded = tokenizer(
        rendered,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_context_tokens,
        return_attention_mask=True,
    )
    if "attention_mask" not in encoded:
        raise RuntimeError("tokenizer did not return an attention_mask")
    return encoded


def generate_batch(
    *,
    tokenizer: Any,
    model: Any,
    prompts: list[str],
    seed: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> list[str]:
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    encoded = {
        key: value.to("cuda")
        for key, value in encode_prompts(tokenizer, prompts).items()
    }
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = model.generation_config.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    with torch.inference_mode():
        output = model.generate(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            remove_invalid_values=True,
            renormalize_logits=True,
            pad_token_id=pad_token_id,
            use_cache=True,
        )
    input_width = int(encoded["input_ids"].shape[1])
    texts = []
    for row in output:
        generated = row[input_width:]
        texts.append(tokenizer.decode(generated, skip_special_tokens=True))
    return texts


def generate_with_fallback(
    *,
    tokenizer: Any,
    model: Any,
    prompts: list[dict[str, Any]],
    config: dict[str, Any],
    requested_batch_size: int,
) -> tuple[list[tuple[dict[str, Any], str]], int, list[str]]:
    batch_size = requested_batch_size
    fallback_notes: list[str] = []
    results: list[tuple[dict[str, Any], str]] = []
    index = 0
    while index < len(prompts):
        batch = prompts[index : index + batch_size]
        seed = sum(int(item["seed"]) for item in batch) % (2**31 - 1)
        try:
            outputs = generate_batch(
                tokenizer=tokenizer,
                model=model,
                prompts=[str(item["prompt"]) for item in batch],
                seed=seed,
                max_new_tokens=int(config["generation"]["max_new_tokens"]),
                temperature=float(config["generation"]["temperature"]),
                top_p=float(config["generation"]["top_p"]),
            )
        except RuntimeError as exc:
            message = str(exc)
            if "out of memory" not in message.lower() and "cuda" not in message.lower():
                raise
            if batch_size <= 1:
                raise
            new_batch = 2 if batch_size > 2 else 1
            fallback_notes.append(f"batch {batch_size} failed at prompt {index}; retrying with {new_batch}: {exc.__class__.__name__}")
            batch_size = new_batch
            continue
        if len(outputs) != len(batch):
            raise RuntimeError(f"output-count mismatch: {len(outputs)} outputs for {len(batch)} prompts")
        results.extend(zip(batch, outputs, strict=True))
        index += len(batch)
    return results, batch_size, fallback_notes


def stage_verify(config: dict[str, Any]) -> int:
    info = require_single_visible_cuda(allowed_name_fragments=("A100",))
    tokenizer, model = load_model(config)
    metadata = {
        "stage": "verify",
        "gpu": info.to_dict(),
        "model": config["model"]["repository"],
        "revision": config["model"]["revision"],
        "quantization": config["quantization"]["policy"],
        "device_map": getattr(model, "hf_device_map", {}),
        "tokenizer_pad_token_id": tokenizer.pad_token_id,
        "config_sha256": config_sha256(config),
    }
    if any(str(device) in {"cpu", "disk"} for device in metadata["device_map"].values()):
        raise RuntimeError(f"CPU or disk offload detected: {metadata['device_map']}")
    print(json.dumps(metadata, ensure_ascii=False, sort_keys=True))
    return 0


def smoke_prompts(config: dict[str, Any]) -> list[dict[str, Any]]:
    prompts = []
    for cell in config["prompt_cells"][:4]:
        prompts.append(
            {
                "cell_id": cell["cell_id"],
                "attempt_index": 0,
                "seed": int(cell["seed_sequence"][0]),
                "prompt": build_prompt(cell, requested_rows=4),
            }
        )
    return prompts


def stage_smoke_batching(config: dict[str, Any]) -> int:
    require_single_visible_cuda(allowed_name_fragments=("A100",))
    tokenizer, model = load_model(config)
    prompts = smoke_prompts(config)
    raw_generation_dir(config).mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"stage": "smoke-batching", "batch_results": {}}
    for batch_size in (1, 4):
        outputs, used_batch, notes = generate_with_fallback(
            tokenizer=tokenizer,
            model=model,
            prompts=prompts,
            config={**config, "generation": {**config["generation"], "max_new_tokens": config["generation"]["smoke_max_new_tokens"]}},
            requested_batch_size=batch_size,
        )
        parsed_count = 0
        for prompt_meta, raw in outputs:
            path = raw_generation_dir(config) / f"smoke-b{batch_size}-{prompt_meta['cell_id']}.txt"
            atomic_write_text(path, raw)
            lines, _ = extract_utterance_lines(raw, cell_id=str(prompt_meta["cell_id"]), attempt_id="smoke")
            parsed_count += len(lines)
        if parsed_count <= 0:
            raise RuntimeError(f"batch size {batch_size} produced no parseable Slovenian output")
        summary["batch_results"][str(batch_size)] = {
            "used_batch_size": used_batch,
            "fallback_notes": notes,
            "parsed_lines": parsed_count,
        }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


def planned_prompts_for_attempt(config: dict[str, Any], accepted_by_cell: dict[str, int], attempt_index: int) -> list[dict[str, Any]]:
    prompts: list[dict[str, Any]] = []
    for cell in config["prompt_cells"]:
        cell_id = str(cell["cell_id"])
        requested = int(cell["requested_rows"])
        remaining = requested - int(accepted_by_cell.get(cell_id, 0))
        if remaining <= 0:
            continue
        if attempt_index > int(cell["maximum_retries"]):
            continue
        seed = int(cell["seed_sequence"][attempt_index])
        prompts.append(
            {
                "cell_id": cell_id,
                "attempt_index": attempt_index,
                "seed": seed,
                "prompt": build_prompt(cell, requested_rows=remaining),
                "requested_rows": remaining,
            }
        )
    return prompts


def stage_generate(config: dict[str, Any]) -> int:
    info = require_single_visible_cuda(allowed_name_fragments=("A100",))
    tokenizer, model = load_model(config)
    paths = local_paths(config)
    paths["run_dir"].mkdir(parents=True, exist_ok=True)
    paths["raw_generation"].mkdir(parents=True, exist_ok=True)
    all_records: list[dict[str, Any]] = []
    all_rejections: list[Rejection] = []
    accepted_by_cell: dict[str, int] = defaultdict(int)
    used_batch_size = int(config["generation"]["prompt_batch_size"])
    fallback_notes: list[str] = []
    start = time.perf_counter()
    with GpuMonitor(
        physical_selector=info.physical_selector,
        output_path=gpu_monitor_path(config),
        interval_seconds=float(config["generation"]["monitor_interval_seconds"]),
    ):
        max_attempt = max(int(cell["maximum_retries"]) for cell in config["prompt_cells"])
        for attempt_index in range(max_attempt + 1):
            prompts = planned_prompts_for_attempt(config, accepted_by_cell, attempt_index)
            if not prompts:
                continue
            outputs, used_batch_size, notes = generate_with_fallback(
                tokenizer=tokenizer,
                model=model,
                prompts=prompts,
                config=config,
                requested_batch_size=used_batch_size,
            )
            fallback_notes.extend(notes)
            for prompt_meta, raw in outputs:
                cell_id = str(prompt_meta["cell_id"])
                attempt_name = f"{cell_id}-attempt-{attempt_index:02d}"
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
                parsed_records = [
                    build_record(
                        config=config,
                        cell=cell,
                        attempt_index=attempt_index,
                        output_ordinal=line.output_ordinal,
                        text=line.text,
                        extraction_mode="line",
                    )
                    for line in lines
                ]
                retained, filter_rejections, _ = filter_records(
                    parsed_records,
                    config=config,
                    existing_rejections=(),
                    protected_indexes=(),
                )
                all_rejections.extend(filter_rejections)
                space = max(0, int(cell["requested_rows"]) - accepted_by_cell[cell_id])
                selected = retained[:space]
                for overflow in retained[space:]:
                    all_rejections.append(
                        Rejection(
                            "quota_overflow",
                            cell_id,
                            attempt_name,
                            candidate_id=str(overflow.get("candidate_id")),
                        )
                    )
                accepted_by_cell[cell_id] += len(selected)
                all_records.extend(selected)
    atomic_write_jsonl(generated_all_path(config), all_records)
    write_rejections(rejected_path(config), all_rejections)
    metadata = {
        "stage": "generate",
        "wall_time_seconds": round(time.perf_counter() - start, 3),
        "requested_initial_rows": int(config["target_generated_rows"]),
        "raw_schema_records_retained_before_protected_filter": len(all_records),
        "rejections": len(all_rejections),
        "accepted_by_cell": dict(sorted(accepted_by_cell.items())),
        "prompt_batch_size_used": used_batch_size,
        "batch_fallback_notes": fallback_notes,
    }
    atomic_write_json(run_dir(config) / "generation-summary.local.json", metadata)
    print(json.dumps(metadata, ensure_ascii=False, sort_keys=True))
    return 0 if len(all_records) >= int(config["minimum_structurally_admissible_rows"]) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a governed GaMS corpus-v2 candidate reservoir.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("verify", "smoke-batching", "generate"), required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.stage == "verify":
        return stage_verify(config)
    if args.stage == "smoke-batching":
        return stage_smoke_batching(config)
    if args.stage == "generate":
        return stage_generate(config)
    raise AssertionError(args.stage)


if __name__ == "__main__":
    raise SystemExit(main())
