#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
KLD (Kullback-Leibler Divergence) calculation script using vLLM's score mode

Compares a quantized model against reference logits from a full-precision
model to measure quantization quality via sliding window KLD. Reference
logits are saved per-window to a directory of safetensors files, keeping
only one window's logits in memory at a time; KL divergence is computed
on GPU when reference logits are provided in the prompt.

Usage:
    # Two-phase: generate reference logits then compute KLD
    python examples/score_mode_kld.py \\
        --model /path/to/quantized_model \\
        --reference-model /path/to/reference_model \\
        --dataset wikitext --dataset-config wikitext-2-raw-v1

    # Re-use pre-saved reference logits (directory of per-window files)
    python examples/score_mode_kld.py \\
        --model /path/to/quantized_model \\
        --reference-logits /path/to/ref_logits_dir/ \\
        --dataset wikitext --dataset-config wikitext-2-raw-v1
"""

import argparse
import gc
import json
import logging
import os
import time
from typing import Any

import torch
import torch.nn.functional as F
from datasets import load_dataset
from safetensors.torch import safe_open, save_file
from transformers import AutoTokenizer

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

logger = logging.getLogger(__name__)


def load_dataset_texts(
    dataset_name: str,
    dataset_config: str | None = None,
    split: str | None = None,
) -> list[str]:
    """Load and extract text from a HuggingFace dataset."""
    if split is None:
        for candidate_split in ["test", "train", "validation"]:
            try:
                if dataset_config:
                    dataset = load_dataset(
                        dataset_name, dataset_config, split=candidate_split
                    )
                else:
                    dataset = load_dataset(dataset_name, split=candidate_split)
                split = candidate_split
                break
            except Exception:
                continue

        if split is None:
            raise ValueError(
                f"Could not load dataset {dataset_name} with any split "
                "(test/train/validation)"
            )

    if dataset_config:
        dataset = load_dataset(dataset_name, dataset_config, split=split)
    else:
        dataset = load_dataset(dataset_name, split=split)

    texts = []
    for example in dataset:
        if "text" in example:
            text = example["text"]
            if text and text.strip():
                texts.append(text)
        elif "messages" in example:
            messages = example["messages"]
            if isinstance(messages, list):
                text = "\n".join(
                    msg.get("content", "") for msg in messages if isinstance(msg, dict)
                )
                if text and text.strip():
                    texts.append(text)
        else:
            for key, value in example.items():
                if isinstance(value, str) and value.strip():
                    texts.append(value)
                    break

    if not texts:
        raise ValueError(f"No valid text found in dataset {dataset_name}")

    return texts


def calculate_kld(
    model_path: str | None,
    texts: list[str],
    context_length: int,
    stride: int,
    reference_logits_path: str | None = None,
    reference_model_path: str | None = None,
    llm_kwargs: dict[str, Any] | None = None,
    num_samples: int | None = None,
    max_windows: int | None = None,
    reference_only: bool = False,
    trust_remote_code: bool = False,
) -> tuple[float, int]:
    """
    Calculate KLD using sliding window approach.

    Loads only one model at a time to avoid GPU OOM: Phase 1 (reference)
    runs first and is unloaded before Phase 2 (test model) starts.

    Args:
        model_path: Path to test model
        texts: List of text samples to evaluate
        context_length: Maximum context length for each window
        stride: Stride between windows
        reference_logits_path: Path to reference logits directory or file
        reference_model_path: Path to reference model (for Phase 1)
        llm_kwargs: Kwargs for initializing LLM (reference and test)
        num_samples: Maximum number of samples to process (None = all)
        max_windows: Maximum number of sliding windows to process (None = 100)
        reference_only: Generate reference logits and exit before loading test model
        trust_remote_code: Trust remote code when loading tokenizer

    Returns:
        Tuple of (mean_kld, total_positions)
    """
    kld_sum = 0.0
    kld_count = 0

    samples_to_process = texts[:num_samples] if num_samples else texts
    concatenated_text = "\n\n".join(samples_to_process)

    window_limit = max_windows if max_windows is not None else 100
    max_tokens_for_eval = context_length + max(0, window_limit - 1) * stride
    max_chars = max_tokens_for_eval * 5
    if len(concatenated_text) > max_chars:
        concatenated_text = concatenated_text[:max_chars]

    tokenizer_path = reference_model_path if reference_model_path else model_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, trust_remote_code=trust_remote_code
    )
    encoded = tokenizer(
        concatenated_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_tokens_for_eval,
    )
    tokens = encoded["input_ids"]
    if tokens and isinstance(tokens[0], list):
        tokens = tokens[0]

    if len(tokens) < 2:
        raise ValueError("Not enough tokens after concatenation")

    num_tokens = len(tokens)

    # Phase 1: Generate reference logits if reference_model_path provided
    if reference_model_path is not None:
        model_name = os.path.basename(reference_model_path.rstrip("/\\"))
        ref_logits_dir = reference_logits_path or os.path.join(
            os.getcwd(),
            f"ref_logits_{model_name}_ctx{context_length}_s{stride}",
        )
        reference_logits_path = ref_logits_dir
        if not os.path.exists(ref_logits_dir):
            os.makedirs(ref_logits_dir, exist_ok=True)
            print(f"Phase 1: Generating reference logits from {reference_model_path}")
            ref_llm = LLM(model=reference_model_path, **(llm_kwargs or {}))
            window_idx = 0
            for start_idx in range(0, num_tokens - context_length + stride, stride):
                if max_windows is not None and window_idx >= max_windows:
                    break
                end_idx = start_idx + context_length
                if end_idx > num_tokens:
                    break
                window_tokens = tokens[start_idx:end_idx]
                if len(window_tokens) < 2:
                    continue
                target_token_ids = window_tokens[1:]
                prompt: TokensPrompt = {
                    "prompt_token_ids": window_tokens,
                    "target_token_ids": target_token_ids,
                }
                sampling_params = SamplingParams(
                    prompt_logprobs=1,
                    max_tokens=1,
                    return_prompt_logits=True,
                )
                outputs = ref_llm.generate([prompt], sampling_params=sampling_params)
                out = outputs[0]
                if out.prompt_logits is not None:
                    window_file = os.path.join(
                        ref_logits_dir, f"logits_{window_idx}.safetensors"
                    )
                    save_file({"logits": out.prompt_logits}, window_file)
                    window_idx += 1
            del ref_llm
            gc.collect()
            torch.accelerator.empty_cache()
            print(f"Saved {window_idx} reference logits to {ref_logits_dir}/")

    if reference_only:
        print(f"Reference-only mode complete: {reference_logits_path}")
        return float("nan"), 0

    if reference_logits_path is None:
        raise ValueError(
            "Either --reference-logits or --reference-model must be provided"
        )
    if not os.path.exists(reference_logits_path):
        raise FileNotFoundError(
            f"Reference logits path not found: {reference_logits_path}"
        )
    ref_is_directory = os.path.isdir(reference_logits_path)

    # Phase 2: Compute KLD using test model with reference logits
    if model_path is None:
        raise ValueError("--model is required unless --reference-only is set")
    print("Phase 2: Computing KLD...")
    print(f"Loading test model: {model_path}")
    llm = LLM(model=model_path, **(llm_kwargs or {}))
    window_idx = 0
    for start_idx in range(0, num_tokens - context_length + stride, stride):
        if max_windows is not None and window_idx >= max_windows:
            break
        end_idx = start_idx + context_length
        if end_idx > num_tokens:
            break
        window_tokens = tokens[start_idx:end_idx]
        if len(window_tokens) < 2:
            continue

        target_token_ids = window_tokens[1:]
        if ref_is_directory:
            ref_file = os.path.join(
                reference_logits_path, f"logits_{window_idx}.safetensors"
            )
            ref_key = "logits"
        else:
            ref_file = reference_logits_path
            ref_key = f"logits_{window_idx}"

        prompt: TokensPrompt = {
            "prompt_token_ids": window_tokens,
            "target_token_ids": target_token_ids,
            "reference_logits_path": ref_file,
            "reference_logits_key": ref_key,
        }

        sampling_params = SamplingParams(
            prompt_logprobs=1,
            max_tokens=1,
            kld_mode=True,
        )

        outputs = llm.generate([prompt], sampling_params=sampling_params)
        out = outputs[0]

        if ref_is_directory and not os.path.exists(ref_file):
            logger.warning(
                "Reference logits file missing for window %d: %s — skipping "
                "(reference logits may be incomplete from a killed run)",
                window_idx,
                ref_file,
            )
            window_idx += 1
            continue

        if out.kld_result is not None:
            win_kld_sum, win_kld_count = out.kld_result
            kld_sum += win_kld_sum
            kld_count += win_kld_count
        else:
            sampling_params_fallback = SamplingParams(
                prompt_logprobs=1,
                max_tokens=1,
                return_prompt_logits=True,
            )
            prompt_fallback: TokensPrompt = {
                "prompt_token_ids": window_tokens,
                "target_token_ids": target_token_ids,
            }
            outputs = llm.generate(
                [prompt_fallback], sampling_params=sampling_params_fallback
            )
            out = outputs[0]
            if out.prompt_logits is not None:
                model_logits = out.prompt_logits
                with safe_open(
                    ref_file,
                    framework="pt",
                    device="cpu",
                ) as f:
                    ref_logits = f.get_tensor(ref_key)
                device = model_logits.device
                ref_logits = ref_logits.to(device)
                vs = min(model_logits.shape[-1], ref_logits.shape[-1])
                log_probs_model = F.log_softmax(model_logits[..., :vs].float(), dim=-1)
                log_probs_ref = F.log_softmax(ref_logits[..., :vs].float(), dim=-1)
                kld_per_pos = F.kl_div(
                    log_probs_model,
                    log_probs_ref,
                    reduction="none",
                    log_target=True,
                ).sum(dim=-1)
                kld_sum += kld_per_pos.sum().item()
                kld_count += kld_per_pos.numel()

        window_idx += 1
        logger.debug(
            "Window %d: kld_sum=%.6f, kld_count=%d",
            window_idx,
            kld_sum,
            kld_count,
        )

    if kld_count == 0:
        raise ValueError("No valid positions for KLD calculation")

    mean_kld = kld_sum / kld_count
    return mean_kld, kld_count


def main():
    parser = argparse.ArgumentParser(
        description="Calculate KLD using vLLM's score mode"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to test model (not required with --reference-only)",
    )
    parser.add_argument(
        "--reference-model",
        type=str,
        default=None,
        help="Path to reference model (generates ref logits if needed)",
    )
    parser.add_argument(
        "--reference-logits",
        type=str,
        default=None,
        help="Path to reference logits directory (per-window safetensors) "
        "or a single legacy safetensors file",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        help="Quantization method (e.g., 'awq', 'gptq')",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., 'wikitext')",
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default=None,
        help="Dataset configuration (e.g., 'wikitext-2-raw-v1')",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to process (default: all)",
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help="Maximum number of sliding windows to process (default: 100)",
    )
    parser.add_argument(
        "--reference-only",
        action="store_true",
        help="Only generate reference logits, then exit before loading test model",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=2048,
        help="Context length for each window (default: 2048)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=512,
        help="Stride between windows (default: 512)",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallelism size (default: 1)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.35,
        help="GPU memory utilization (default: 0.35). vLLM reserves this fraction "
        "of each GPU for model+KV cache. 0.7 on 95GB GPUs = 66GB/GPU, which "
            "is excessive for 8B models (~8GB). Use 0.35 or lower.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=None,
        help="Model dtype passed to vLLM LLM (default: vLLM auto)",
    )
    parser.add_argument(
        "--kv-cache-dtype",
        type=str,
        default=None,
        help="KV cache dtype passed to vLLM LLM (e.g. auto, fp8, fp8_e4m3)",
    )
    parser.add_argument(
        "--load-format",
        type=str,
        default=None,
        help="Model load format passed to vLLM LLM (default: auto)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Override vLLM max_model_len (default: context_length * 2)",
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=None,
        help="Override vLLM max_num_batched_tokens for profiling/scheduler sizing",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=None,
        help="Override vLLM max_num_seqs for profiling/scheduler sizing",
    )
    parser.add_argument(
        "--cpu-offload-gb",
        type=float,
        default=None,
        help="GiB of CPU offload per GPU passed to vLLM LLM",
    )
    parser.add_argument(
        "--attention-backend",
        type=str,
        default=None,
        help="Attention backend passed to vLLM LLM, e.g. FLASHINFER_MLA_SPARSE",
    )
    parser.add_argument(
        "--moe-backend",
        type=str,
        default=None,
        help="MoE backend passed to vLLM LLM, e.g. b12x or flashinfer_cutlass",
    )
    parser.add_argument(
        "--dense-equivalent-index-topk",
        type=int,
        default=None,
        help=(
            "For sparse-MLA GLM/DeepSeek models, set index_topk=0 and use dense "
            "MLA. This is valid only when context_length is <= the original "
            "index_topk value supplied here."
        ),
    )
    parser.add_argument(
        "--hf-overrides",
        type=str,
        default=None,
        help="JSON dict passed as vLLM hf_overrides",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Pass enforce_eager=True to vLLM LLM",
    )
    parser.add_argument(
        "--disable-custom-all-reduce",
        action="store_true",
        help="Pass disable_custom_all_reduce=True to vLLM LLM",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code when loading model",
    )
    parser.add_argument(
        "--language-model-only",
        action="store_true",
        help="Disable multimodal modules for text-only models (e.g., Qwen-3.5) "
        "to save GPU memory",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    if args.reference_model is None and args.reference_logits is None:
        parser.error("Either --reference-model or --reference-logits is required")
    if args.reference_only and args.reference_model is None:
        parser.error("--reference-only requires --reference-model")
    if not args.reference_only and args.model is None:
        parser.error("--model is required unless --reference-only is set")

    print(f"Loading dataset: {args.dataset}")
    texts = load_dataset_texts(args.dataset, args.dataset_config)
    print(f"Loaded {len(texts)} text samples")

    llm_kwargs: dict[str, Any] = {
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "trust_remote_code": args.trust_remote_code,
        "enable_prefix_caching": False,
        "max_model_len": args.max_model_len or args.context_length * 2,
    }
    if args.quantization:
        llm_kwargs["quantization"] = args.quantization
    if args.dtype:
        llm_kwargs["dtype"] = args.dtype
    if args.kv_cache_dtype:
        llm_kwargs["kv_cache_dtype"] = args.kv_cache_dtype
    if args.load_format:
        llm_kwargs["load_format"] = args.load_format
    if args.max_num_batched_tokens:
        llm_kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens
    if args.max_num_seqs:
        llm_kwargs["max_num_seqs"] = args.max_num_seqs
    if args.cpu_offload_gb is not None:
        llm_kwargs["cpu_offload_gb"] = args.cpu_offload_gb
    if args.attention_backend:
        llm_kwargs["attention_backend"] = args.attention_backend
    if args.moe_backend:
        llm_kwargs["moe_backend"] = args.moe_backend
    hf_overrides = json.loads(args.hf_overrides) if args.hf_overrides else {}
    if args.dense_equivalent_index_topk is not None:
        if args.context_length > args.dense_equivalent_index_topk:
            parser.error(
                "--dense-equivalent-index-topk is only valid when "
                "context_length <= the original index_topk"
            )
        hf_overrides["index_topk"] = 0
    if hf_overrides:
        llm_kwargs["hf_overrides"] = hf_overrides
    if args.enforce_eager:
        llm_kwargs["enforce_eager"] = True
    if args.disable_custom_all_reduce:
        llm_kwargs["disable_custom_all_reduce"] = True
    if args.language_model_only:
        llm_kwargs["language_model_only"] = True

    print("\nCalculating KLD...")
    print(f"  Context length: {args.context_length}")
    print(f"  Stride: {args.stride}")
    print(f"  Max windows: {args.max_windows or 100}")
    print(f"  Samples: {args.num_samples or len(texts)}")
    print(f"  Reference only: {args.reference_only}")

    start_time = time.time()
    mean_kld, total_positions = calculate_kld(
        args.model,
        texts,
        args.context_length,
        args.stride,
        reference_logits_path=args.reference_logits,
        reference_model_path=args.reference_model,
        llm_kwargs=llm_kwargs,
        num_samples=args.num_samples,
        max_windows=args.max_windows,
        reference_only=args.reference_only,
        trust_remote_code=args.trust_remote_code,
    )
    elapsed_time = time.time() - start_time

    if args.reference_only:
        print("\nResults: reference logits generated")
        print(f"  Time elapsed: {elapsed_time:.2f} seconds")
    else:
        print("\nResults:")
        print(f"  Mean KLD: {mean_kld:.6f}")
        print(f"  Total positions: {total_positions}")
        print(f"  Time elapsed: {elapsed_time:.2f} seconds")
        print(f"  Positions/second: {total_positions / elapsed_time:.2f}")


if __name__ == "__main__":
    main()
