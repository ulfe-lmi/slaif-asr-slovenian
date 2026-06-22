#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slaif_asr.gams import load_generation_config, parse_strict_json_candidates, validate_candidate_batch
from slaif_asr.prompt_experiment import atomic_write_text


def require_single_gpu() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be exactly 0")
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("GaMS generation requires exactly one visible CUDA device")
    if "2080 Ti" not in torch.cuda.get_device_name(0):
        raise RuntimeError(f"expected RTX 2080 Ti, saw {torch.cuda.get_device_name(0)}")


def build_prompt(*, round_id: str, count: int, brief_path: Path | None) -> str:
    brief_text = ""
    if brief_path is not None:
        brief = json.loads(brief_path.read_text(encoding="utf-8"))
        if brief.get("real_gate_reference_text_included"):
            raise ValueError("round brief must not include real-gate reference text")
        brief_text = json.dumps(brief, ensure_ascii=False, sort_keys=True)
    return (
        "Return strict JSON only, with no prose and no Markdown. "
        f"Generate exactly {count} Slovenian ASR synthetic candidates for round prefix {round_id}. "
        "The response must be a JSON array. Each object must use this exact schema: "
        "{\"candidate_id\":\""
        f"{round_id}-0001\",\"spoken_text\":\"Čez cesto gre moder avtomobil.\","
        "\"target_text\":\"Čez cesto gre moder avtomobil.\",\"language\":\"sl-SI\","
        "\"phenomena\":[\"ordinary\"],\"source_error_clusters\":[],\"generation_seed\":1234}. "
        f"candidate_id must be a string using {round_id}-0001, {round_id}-0002, and so on. "
        "spoken_text and target_text must be identical Slovenian sentences, not placeholders. "
        "language must be exactly sl-SI. phenomena must be a non-empty string array. "
        "generation_seed must be an integer. "
        f"Failure brief: {brief_text}"
    )


def load_model(config: dict[str, Any], *, use_fallback: bool, context_tokens: int):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_cfg = config["fallback_model" if use_fallback else "primary_model"]
    quant_cfg = config["quantization"]
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quant_cfg["quant_type"],
        bnb_4bit_use_double_quant=quant_cfg["double_quantization"],
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["repository"], revision=model_cfg["revision"], trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["repository"],
        revision=model_cfg["revision"],
        quantization_config=bnb_config,
        device_map={"": 0},
        max_memory={0: f"{config['device_policy']['max_memory_gib']}GiB"},
        trust_remote_code=False,
    )
    device_map = getattr(model, "hf_device_map", {})
    if any(str(device) in {"cpu", "disk"} for device in device_map.values()):
        raise RuntimeError(f"CPU or disk offload is forbidden: {device_map}")
    model.config.max_position_embeddings = min(getattr(model.config, "max_position_embeddings", context_tokens), context_tokens)
    return model_cfg, tokenizer, model


def generate_text(tokenizer: Any, model: Any, prompt: str, config: dict[str, Any]) -> str:
    import torch

    generation_cfg = config["generation"]
    input_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    ).to("cuda")
    if input_ids.shape[-1] > generation_cfg["initial_context_tokens"]:
        input_ids = input_ids[:, -generation_cfg["initial_context_tokens"] :]
    with torch.inference_mode():
        output = model.generate(
            input_ids=input_ids,
            max_new_tokens=generation_cfg["max_new_tokens"],
            do_sample=True,
            temperature=generation_cfg["temperature"],
            top_p=generation_cfg["top_p"],
            remove_invalid_values=True,
            renormalize_logits=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0][input_ids.shape[-1] :], skip_special_tokens=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate strict-json Slovenian candidates with pinned GaMS.")
    parser.add_argument("--config", type=Path, default=Path("configs/generation/gams_prompt_curriculum.json"))
    parser.add_argument("--round-id", required=True)
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--brief", type=Path)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--raw-output", type=Path)
    parser.add_argument("--use-fallback", action="store_true")
    args = parser.parse_args()

    require_single_gpu()
    config = load_generation_config(args.config)
    prompt = build_prompt(round_id=args.round_id, count=args.count, brief_path=args.brief)
    start = time.perf_counter()
    model_cfg, tokenizer, model = load_model(
        config,
        use_fallback=args.use_fallback,
        context_tokens=config["generation"]["fallback_context_tokens" if args.use_fallback else "initial_context_tokens"],
    )
    raw = generate_text(tokenizer, model, prompt, config)
    if args.raw_output is not None:
        atomic_write_text(args.raw_output, raw)
    try:
        rows = parse_strict_json_candidates(raw)
        valid, rejected = validate_candidate_batch(rows)
    except Exception as exc:
        import torch

        metadata = {
            "model": model_cfg["repository"],
            "revision": model_cfg["revision"],
            "fallback_used": args.use_fallback,
            "license": model_cfg["license"],
            "candidate_requested": args.count,
            "candidate_valid": 0,
            "candidate_rejected": args.count,
            "failure": str(exc),
            "wall_time_seconds": round(time.perf_counter() - start, 3),
            "peak_vram_mib": round(torch.cuda.max_memory_allocated(0) / 1024 / 1024, 1),
        }
        atomic_write_text(args.metadata, json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))
        return 1
    text = "".join(
        json.dumps(
            {
                "candidate_id": item.candidate_id,
                "spoken_text": item.spoken_text,
                "target_text": item.target_text,
                "language": item.language,
                "phenomena": list(item.phenomena),
                "source_error_clusters": list(item.source_error_clusters),
                "generation_seed": item.generation_seed,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n"
        for item in valid
    )
    atomic_write_text(args.output_jsonl, text)
    import torch

    metadata = {
        "model": model_cfg["repository"],
        "revision": model_cfg["revision"],
        "fallback_used": args.use_fallback,
        "license": model_cfg["license"],
        "candidate_requested": args.count,
        "candidate_valid": len(valid),
        "candidate_rejected": len(rejected),
        "rejections": rejected,
        "wall_time_seconds": round(time.perf_counter() - start, 3),
        "peak_vram_mib": round(torch.cuda.max_memory_allocated(0) / 1024 / 1024, 1),
    }
    atomic_write_text(args.metadata, json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
