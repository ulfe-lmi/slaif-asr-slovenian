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

from slaif_asr.gams import load_generation_config


def require_single_gpu() -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible or "," in visible:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must select exactly one physical GPU")

    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("GaMS CLI requires exactly one visible CUDA device")


def load_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file is not None:
        return args.prompt_file.read_text(encoding="utf-8")
    return args.prompt


def load_model(config: dict[str, Any], args: argparse.Namespace):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_cfg = config["fallback_model" if args.use_fallback else "primary_model"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["repository"],
        revision=model_cfg["revision"],
        trust_remote_code=False,
    )
    kwargs: dict[str, Any] = {
        "revision": model_cfg["revision"],
        "device_map": {"": 0},
        "max_memory": {0: f"{args.max_memory_gib}GiB"},
        "trust_remote_code": False,
    }
    if args.attn_implementation:
        kwargs["attn_implementation"] = args.attn_implementation
    if args.bf16:
        kwargs["torch_dtype"] = torch.bfloat16
    elif args.fp16:
        kwargs["torch_dtype"] = torch.float16
    else:
        quant_cfg = config["quantization"]
        compute_dtype = torch.bfloat16 if args.bnb_compute_dtype == "bfloat16" else torch.float16
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant_cfg["quant_type"],
            bnb_4bit_use_double_quant=quant_cfg["double_quantization"],
            bnb_4bit_compute_dtype=compute_dtype,
        )
        kwargs["torch_dtype"] = compute_dtype
    model = AutoModelForCausalLM.from_pretrained(model_cfg["repository"], **kwargs)
    device_map = getattr(model, "hf_device_map", {})
    if any(str(device) in {"cpu", "disk"} for device in device_map.values()):
        raise RuntimeError(f"CPU or disk offload is forbidden: {device_map}")
    return model_cfg, tokenizer, model


def encode_prompt(tokenizer: Any, prompt: str, *, use_chat_template: bool):
    import torch

    if use_chat_template and getattr(tokenizer, "chat_template", None):
        input_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        )
        return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}
    encoded = tokenizer(prompt, return_tensors="pt")
    return {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]}


def sanitize_generation_config(model: Any) -> None:
    # Some GaMS/Gemma generation configs carry sampling-only values that
    # Transformers warns about before applying explicit generate kwargs.
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the pinned GaMS model on one visible GPU and print raw decoded output to stdout.",
    )
    parser.add_argument("prompt", nargs="?", help="Prompt text. Omit when using --prompt-file.")
    parser.add_argument("--prompt-file", type=Path, help="UTF-8 file containing the prompt.")
    parser.add_argument("--config", type=Path, default=Path("configs/generation/gams_prompt_curriculum.json"))
    parser.add_argument("--use-fallback", action="store_true", help="Use the pinned fallback GaMS model.")
    parser.add_argument("--fp16", action="store_true", help="Load in FP16 instead of 4-bit NF4.")
    parser.add_argument("--bf16", action="store_true", help="Load in BF16 instead of 4-bit NF4.")
    parser.add_argument(
        "--bnb-compute-dtype",
        choices=("float16", "bfloat16"),
        default="bfloat16",
        help="Compute dtype for 4-bit NF4 mode.",
    )
    parser.add_argument("--no-chat-template", action="store_true", help="Tokenize the prompt directly.")
    parser.add_argument(
        "--attn-implementation",
        choices=("eager", "sdpa", "flash_attention_2"),
        help="Optional Transformers attention implementation override.",
    )
    parser.add_argument("--no-cache", action="store_true", help="Disable KV cache during generation.")
    parser.add_argument("--max-memory-gib", type=int, default=int(os.environ.get("SLAIF_GAMS_MAX_MEMORY_GIB", "76")))
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--greedy", action="store_true", help="Disable sampling.")
    parser.add_argument("--seed", type=int, help="Optional torch manual seed.")
    parser.add_argument("--metadata", type=Path, help="Optional JSON metadata path.")
    parser.add_argument("--show-special-tokens", action="store_true", help="Do not skip special tokens while decoding.")
    args = parser.parse_args()

    if not args.prompt and args.prompt_file is None:
        parser.error("provide a prompt argument or --prompt-file")
    if args.fp16 and args.bf16:
        parser.error("--fp16 and --bf16 are mutually exclusive")

    require_single_gpu()
    config = load_generation_config(args.config)
    prompt = load_prompt(args)

    import torch

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    start = time.perf_counter()
    model_cfg, tokenizer, model = load_model(config, args)
    sanitize_generation_config(model)
    encoded = {
        key: value.to("cuda")
        for key, value in encode_prompt(tokenizer, prompt, use_chat_template=not args.no_chat_template).items()
    }
    input_ids = encoded["input_ids"]

    stderr_metadata = {
        "model": model_cfg["repository"],
        "revision": model_cfg["revision"],
        "precision": "bf16" if args.bf16 else "fp16" if args.fp16 else f"4bit-nf4-{args.bnb_compute_dtype}",
        "cuda_device": torch.cuda.get_device_name(0),
        "visible_gpu_count": torch.cuda.device_count(),
        "device_map": getattr(model, "hf_device_map", {}),
    }
    print(json.dumps(stderr_metadata, ensure_ascii=False, sort_keys=True), file=sys.stderr)

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = model.generation_config.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    generation_kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": encoded["attention_mask"],
        "max_new_tokens": args.max_new_tokens,
        "do_sample": not args.greedy,
        "use_cache": not args.no_cache,
        "remove_invalid_values": True,
        "renormalize_logits": True,
        "pad_token_id": pad_token_id,
    }
    if not args.greedy:
        generation_kwargs["temperature"] = args.temperature
        generation_kwargs["top_p"] = args.top_p

    gen_start = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**generation_kwargs)
    generated_ids = output[0][input_ids.shape[-1] :]
    text = tokenizer.decode(generated_ids, skip_special_tokens=not args.show_special_tokens)

    metadata = {
        **stderr_metadata,
        "load_plus_generate_seconds": round(time.perf_counter() - start, 3),
        "generate_seconds": round(time.perf_counter() - gen_start, 3),
        "peak_vram_mib": round(torch.cuda.max_memory_allocated(0) / 1024 / 1024, 1),
        "prompt_tokens": int(input_ids.shape[-1]),
        "generated_tokens": int(output.shape[-1] - input_ids.shape[-1]),
        "generated_token_ids_head": [int(item) for item in generated_ids[:20].detach().cpu().tolist()],
    }
    if args.metadata is not None:
        args.metadata.parent.mkdir(parents=True, exist_ok=True)
        args.metadata.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, sort_keys=True), file=sys.stderr)
    print(text, end="" if text.endswith("\n") else "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
