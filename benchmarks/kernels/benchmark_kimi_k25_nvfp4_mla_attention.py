# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Microbenchmark for KimiK25Nvfp4MLAAttention decode forwards.

This benchmark instantiates the specialized Kimi-K2.5 NVFP4 MLA attention
module with the real `nvidia/Kimi-K2.5-NVFP4` text config, populates it with
random weights, and exercises a decode-shaped forward pass:

- one query token per request
- random sequence lengths / KV block tables
- a populated KV cache
- the real `torch.ops.vllm.monolithic_attn` path

The linear layers are initialized with random dense weights instead of loading
the checkpoint's NVFP4 tensors. This keeps the benchmark lightweight while
preserving the real module dimensions, cache layout, metadata builder, and
decode execution path.
"""

from __future__ import annotations

import argparse
import math
import os
import statistics
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import torch

from vllm.config import (
    CUDAGraphMode,
    CacheConfig,
    CompilationConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.distributed import (
    cleanup_dist_env_and_memory,
    graph_capture,
    init_distributed_environment,
    initialize_model_parallel,
    model_parallel_is_initialized,
)
from vllm.forward_context import (
    BatchDescriptor,
    get_forward_context,
    set_forward_context,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_and_maybe_dequant_weights,
)
from vllm.model_executor.specialized_models.kimi_k2_5_nvfp4.model import (
    ForkedKimiK25Nvfp4MLAAttention,
    KimiK25Nvfp4MLAAttention,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import (
    is_quantized_kv_cache,
    set_default_torch_dtype,
    set_random_seed,
)
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.worker.workspace import (
    init_workspace_manager,
    is_workspace_manager_initialized,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark KimiK25Nvfp4MLAAttention decode forwards."
    )
    parser.add_argument(
        "--model",
        default="nvidia/Kimi-K2.5-NVFP4",
        help="Model name or local path used to load the real text config.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Number of decode requests in the synthetic batch.",
    )
    parser.add_argument(
        "--min-seq-len",
        type=int,
        default=2048,
        help="Minimum per-request decoded sequence length.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=4096,
        help="Maximum per-request decoded sequence length.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=64,
        help="KV cache block size.",
    )
    parser.add_argument(
        "--kv-cache-dtype",
        choices=["auto", "float16", "bfloat16", "fp8", "fp8_e4m3"],
        default="fp8",
        help="KV cache dtype. fp8 best matches the recommended Kimi runtime.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup iterations.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=20,
        help="Number of timed iterations.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for synthetic inputs and random weights.",
    )
    parser.add_argument(
        "--cudagraph",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Capture the decode forward in a CUDA graph and replay it.",
    )
    parser.add_argument(
        "--torch-compile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Compile the attention module with torch.compile before benchmarking. "
            "Compilation time is excluded from the timed trials."
        ),
    )
    parser.add_argument(
        "--kernel-mode",
        choices=["original", "forked", "both"],
        default="original",
        help=(
            "Which attention implementation to benchmark: the original kernel, "
            "the forked kernel, or both."
        ),
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.min_seq_len <= 0:
        raise ValueError("--min-seq-len must be positive.")
    if args.max_seq_len < args.min_seq_len:
        raise ValueError("--max-seq-len must be >= --min-seq-len.")
    if args.block_size <= 0:
        raise ValueError("--block-size must be positive.")
    if args.warmup < 0 or args.trials <= 0:
        raise ValueError("--warmup must be >= 0 and --trials must be > 0.")
    if not current_platform.is_cuda():
        raise RuntimeError("This benchmark requires a CUDA device.")


def _build_vllm_config(args: argparse.Namespace) -> VllmConfig:
    model_config = ModelConfig(
        model=args.model,
        tokenizer=args.model,
        trust_remote_code=False,
        dtype="auto",
        seed=args.seed,
        max_model_len=args.max_seq_len,
        skip_tokenizer_init=True,
    )

    cache_config = CacheConfig(
        block_size=args.block_size,
        cache_dtype=args.kv_cache_dtype,
        enable_prefix_caching=False,
    )
    scheduler_config = SchedulerConfig(
        max_num_seqs=args.batch_size,
        max_num_batched_tokens=args.batch_size,
        max_model_len=args.max_seq_len,
        is_encoder_decoder=False,
        enable_chunked_prefill=True,
    )
    parallel_config = ParallelConfig(tensor_parallel_size=1)
    compilation_config = CompilationConfig()

    top_level_config = VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=parallel_config,
        scheduler_config=scheduler_config,
        compilation_config=compilation_config,
    )

    return top_level_config.with_hf_config(model_config.hf_text_config)


@contextmanager
def _maybe_init_single_rank_distributed(vllm_config: VllmConfig) -> Iterator[None]:
    created_distributed_env = False
    temp_path: str | None = None

    try:
        if not torch.distributed.is_initialized():
            fd, temp_path = tempfile.mkstemp(prefix="vllm_kimi_bench_")
            os.close(fd)
            with set_current_vllm_config(vllm_config):
                init_distributed_environment(
                    world_size=1,
                    rank=0,
                    distributed_init_method=f"file://{temp_path}",
                    local_rank=0,
                    backend="nccl",
                )
                initialize_model_parallel(
                    tensor_model_parallel_size=1,
                    pipeline_model_parallel_size=1,
                )
            created_distributed_env = True
        elif not model_parallel_is_initialized():
            with set_current_vllm_config(vllm_config):
                initialize_model_parallel(
                    tensor_model_parallel_size=1,
                    pipeline_model_parallel_size=1,
                )

        yield
    finally:
        if created_distributed_env:
            cleanup_dist_env_and_memory()
        if temp_path is not None:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass


def _initialize_random_weights(module: torch.nn.Module) -> None:
    for name, param in module.named_parameters():
        with torch.no_grad():
            if param.ndim >= 2:
                param.normal_(mean=0.0, std=0.02)
            elif "layernorm" in name.lower() or "norm" in name.lower():
                param.normal_(mean=1.0, std=0.1)
            else:
                param.zero_()


def _make_random_kv_cache(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    kv_cache_dtype: str,
) -> torch.Tensor:
    if is_quantized_kv_cache(kv_cache_dtype):
        fp8_values = torch.randn(shape, device=device, dtype=torch.float32)
        fp8_values = fp8_values.to(dtype=current_platform.fp8_dtype())
        if dtype == fp8_values.dtype:
            return fp8_values
        if dtype == torch.uint8:
            return fp8_values.view(torch.uint8)
        return fp8_values.to(dtype=dtype)

    kv_cache = torch.randn(shape, device=device, dtype=torch.float32)
    return kv_cache.to(dtype=dtype)


def _build_decode_batch(
    *,
    batch_size: int,
    min_seq_len: int,
    max_seq_len: int,
    block_size: int,
    hidden_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    CommonAttentionMetadata,
    int,
]:
    seq_lens_cpu = torch.randint(
        low=min_seq_len,
        high=max_seq_len + 1,
        size=(batch_size,),
        dtype=torch.int32,
    )
    seq_lens_cpu[-1] = max_seq_len
    seq_lens = seq_lens_cpu.to(device=device)

    positions = (seq_lens.to(torch.int64) - 1).contiguous()
    hidden_states = torch.randn(batch_size, hidden_size, device=device, dtype=dtype)

    query_start_loc_cpu = torch.arange(batch_size + 1, dtype=torch.int32)
    query_start_loc = query_start_loc_cpu.to(device=device)

    num_blocks_per_req = [
        math.ceil(int(seq_len) / block_size) for seq_len in seq_lens_cpu.tolist()
    ]
    max_num_blocks = max(num_blocks_per_req)
    block_table_tensor = torch.zeros(
        (batch_size, max_num_blocks),
        dtype=torch.int32,
        device=device,
    )
    slot_mapping = torch.empty(batch_size, dtype=torch.int64, device=device)

    current_block = 0
    for req_idx, seq_len in enumerate(seq_lens_cpu.tolist()):
        num_blocks = num_blocks_per_req[req_idx]
        block_ids = torch.arange(
            current_block,
            current_block + num_blocks,
            dtype=torch.int32,
            device=device,
        )
        block_table_tensor[req_idx, :num_blocks] = block_ids

        token_idx = seq_len - 1
        block_idx = token_idx // block_size
        offset_in_block = token_idx % block_size
        slot_mapping[req_idx] = (current_block + block_idx) * block_size + offset_in_block
        current_block += num_blocks

    common_attn_metadata = CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens=seq_lens,
        num_reqs=batch_size,
        num_actual_tokens=batch_size,
        max_query_len=1,
        max_seq_len=int(seq_lens_cpu.max().item()),
        block_table_tensor=block_table_tensor,
        slot_mapping=slot_mapping,
        _seq_lens_cpu=seq_lens_cpu,
        _num_computed_tokens_cpu=seq_lens_cpu - 1,
        causal=True,
    )

    return positions, hidden_states, common_attn_metadata, current_block


@dataclass(frozen=True)
class KernelVariantSpec:
    name: str
    layer_cls: type[KimiK25Nvfp4MLAAttention]
    prefix: str


@dataclass
class BenchmarkVariantResult:
    spec: KernelVariantSpec
    layer: KimiK25Nvfp4MLAAttention
    output: torch.Tensor
    times_ms: list[float]
    mla_forward_context_lines: list[str]


@dataclass(frozen=True)
class TensorFiniteSummary:
    shape: tuple[int, ...]
    dtype: torch.dtype
    numel: int
    finite_count: int
    nan_count: int
    posinf_count: int
    neginf_count: int
    min_finite: float | None
    mean_finite: float | None
    max_finite: float | None


def _tensor_preview(
    tensor: torch.Tensor | None,
    *,
    max_elems: int = 8,
    max_rows: int = 4,
    max_cols: int = 8,
) -> str:
    if tensor is None:
        return "None"

    cpu_tensor = tensor.detach().cpu()
    shape = tuple(cpu_tensor.shape)
    dtype = cpu_tensor.dtype
    if cpu_tensor.ndim == 0:
        return f"shape={shape}, dtype={dtype}, value={cpu_tensor.item()}"

    truncated = False
    if cpu_tensor.ndim == 1:
        preview = cpu_tensor[:max_elems].tolist()
        truncated = cpu_tensor.shape[0] > max_elems
    elif cpu_tensor.ndim == 2:
        preview = cpu_tensor[:max_rows, :max_cols].tolist()
        truncated = (
            cpu_tensor.shape[0] > max_rows or cpu_tensor.shape[1] > max_cols
        )
    else:
        flat = cpu_tensor.reshape(-1)[:max_elems].tolist()
        preview = flat
        truncated = cpu_tensor.numel() > max_elems

    suffix = ", truncated" if truncated else ""
    return f"shape={shape}, dtype={dtype}, preview={preview}{suffix}"


def _tensor_stats_line(name: str, tensor: torch.Tensor | None) -> str:
    if tensor is None:
        return f"  {name}: None"

    cpu_tensor = tensor.detach().cpu().to(torch.float32)
    if cpu_tensor.numel() == 0:
        return f"  {name}: empty"

    min_value = cpu_tensor.min().item()
    mean_value = cpu_tensor.mean().item()
    max_value = cpu_tensor.max().item()
    return (
        f"  {name}: min/mean/max="
        f"{min_value:.1f}/{mean_value:.1f}/{max_value:.1f}"
    )


def _summarize_tensor_finiteness(
    tensor: torch.Tensor,
    *,
    scale: float | None = None,
    chunk_size: int = 1_000_000,
) -> TensorFiniteSummary:
    flat_tensor = tensor.detach().reshape(-1)
    total_numel = flat_tensor.numel()

    if total_numel == 0:
        return TensorFiniteSummary(
            shape=tuple(tensor.shape),
            dtype=tensor.dtype,
            numel=0,
            finite_count=0,
            nan_count=0,
            posinf_count=0,
            neginf_count=0,
            min_finite=None,
            mean_finite=None,
            max_finite=None,
        )

    finite_count = 0
    nan_count = 0
    posinf_count = 0
    neginf_count = 0
    finite_sum = 0.0
    min_finite: float | None = None
    max_finite: float | None = None

    for start in range(0, total_numel, chunk_size):
        chunk = flat_tensor[start : start + chunk_size].to(torch.float32)
        if scale is not None:
            chunk = chunk * scale

        nan_count += int(torch.isnan(chunk).sum().item())
        posinf_count += int(torch.isposinf(chunk).sum().item())
        neginf_count += int(torch.isneginf(chunk).sum().item())

        finite_mask = torch.isfinite(chunk)
        chunk_finite_count = int(finite_mask.sum().item())
        finite_count += chunk_finite_count
        if chunk_finite_count == 0:
            continue

        finite_values = chunk[finite_mask]
        finite_sum += float(finite_values.sum().item())
        chunk_min = float(finite_values.min().item())
        chunk_max = float(finite_values.max().item())
        min_finite = chunk_min if min_finite is None else min(min_finite, chunk_min)
        max_finite = chunk_max if max_finite is None else max(max_finite, chunk_max)

    mean_finite = finite_sum / finite_count if finite_count > 0 else None
    return TensorFiniteSummary(
        shape=tuple(tensor.shape),
        dtype=tensor.dtype,
        numel=total_numel,
        finite_count=finite_count,
        nan_count=nan_count,
        posinf_count=posinf_count,
        neginf_count=neginf_count,
        min_finite=min_finite,
        mean_finite=mean_finite,
        max_finite=max_finite,
    )


def _format_tensor_finiteness(
    name: str,
    summary: TensorFiniteSummary,
    *,
    note: str | None = None,
) -> str:
    line = (
        f"  {name}: shape={summary.shape}, dtype={summary.dtype}, "
        f"finite={summary.finite_count}/{summary.numel}, "
        f"nan={summary.nan_count}, +inf={summary.posinf_count}, "
        f"-inf={summary.neginf_count}"
    )
    if summary.finite_count > 0:
        line += (
            ", finite_min/mean/max="
            f"{summary.min_finite:.6f}/{summary.mean_finite:.6f}/{summary.max_finite:.6f}"
        )
    else:
        line += ", finite_min/mean/max=n/a"
    if note is not None:
        line += f", note={note}"
    return line


def _summarize_input_tensor_sanity(
    layer: KimiK25Nvfp4MLAAttention,
    kv_cache_template: torch.Tensor,
) -> list[str]:
    lines = [
        "input_tensor_sanity:",
        (
            "  kv_b_proj_quant_method: "
            f"{type(layer.kv_b_proj.quant_method).__name__}"
        ),
    ]

    named_parameters = dict(layer.named_parameters())
    for name in [
        "fused_qkv_a_proj.weight",
        "q_b_proj.weight",
        "kv_b_proj.weight",
        "o_proj.weight",
    ]:
        param = named_parameters.get(name)
        if param is None:
            continue
        lines.append(
            _format_tensor_finiteness(
                name,
                _summarize_tensor_finiteness(param),
            )
        )

    kv_b_proj_dequant = get_and_maybe_dequant_weights(
        layer.kv_b_proj,
        out_dtype=torch.float32,
    )
    lines.append(
        _format_tensor_finiteness(
            "kv_b_proj.dequant_weight",
            _summarize_tensor_finiteness(kv_b_proj_dequant),
            note="identity cast in this benchmark because quant_config=None",
        )
    )
    lines.append(
        _format_tensor_finiteness(
            "mla.W_UK_T",
            _summarize_tensor_finiteness(layer.mla_attn.W_UK_T),
        )
    )
    lines.append(
        _format_tensor_finiteness(
            "mla.W_UV",
            _summarize_tensor_finiteness(layer.mla_attn.W_UV),
        )
    )

    kv_scale = float(layer.mla_attn._k_scale_float)
    kv_cache_is_quantized = is_quantized_kv_cache(layer.mla_attn.kv_cache_dtype)
    kv_cache_interpreted = (
        kv_cache_template.view(current_platform.fp8_dtype())
        if kv_cache_is_quantized and kv_cache_template.dtype == torch.uint8
        else kv_cache_template
    )
    lines.append(
        "  kv_cache_interpretation: "
        f"kv_cache_dtype={layer.mla_attn.kv_cache_dtype}, "
        f"is_quantized={kv_cache_is_quantized}, "
        f"_k_scale_float={kv_scale:.6f}"
    )
    lines.append(
        _format_tensor_finiteness(
            "kv_cache_template.storage_values",
            _summarize_tensor_finiteness(kv_cache_template),
        )
    )
    lines.append(
        _format_tensor_finiteness(
            "kv_cache_template.interpreted_values",
            _summarize_tensor_finiteness(kv_cache_interpreted),
            note=(
                "reinterpreted from uint8 storage as fp8 values"
                if kv_cache_is_quantized
                else "same as storage values for non-quantized cache"
            ),
        )
    )
    lines.append(
        _format_tensor_finiteness(
            "kv_cache_template.dequantized_values",
            _summarize_tensor_finiteness(kv_cache_interpreted, scale=kv_scale),
            note=(
                "interpreted fp8 values multiplied by _k_scale_float"
                if kv_cache_is_quantized
                else "same as interpreted values for non-quantized cache"
            ),
        )
    )
    return lines


def _summarize_mla_forward_context(layer_name: str) -> list[str]:
    forward_context = get_forward_context()
    assert isinstance(forward_context.attn_metadata, dict)
    assert isinstance(forward_context.slot_mapping, dict)

    attn_metadata = forward_context.attn_metadata.get(layer_name)
    slot_mapping = forward_context.slot_mapping.get(layer_name)

    lines = [
        "mla_forward_context:",
        f"  target_layer: {layer_name}",
        f"  cudagraph_runtime_mode: {forward_context.cudagraph_runtime_mode}",
        (
            "  batch_descriptor: "
            f"{forward_context.batch_descriptor!r}"
        ),
        (
            "  no_compile_layers: "
            f"count={len(forward_context.no_compile_layers)}, "
            f"contains_target={layer_name in forward_context.no_compile_layers}"
        ),
        (
            "  attn_metadata_type: "
            f"{type(attn_metadata).__name__ if attn_metadata is not None else 'None'}"
        ),
    ]

    if attn_metadata is None:
        return lines

    lines.extend(
        [
            (
                "  attention_tokens: "
                f"num_actual_tokens={attn_metadata.num_actual_tokens}, "
                f"num_decode_tokens={attn_metadata.num_decode_tokens}, "
                f"num_decodes={attn_metadata.num_decodes}, "
                f"num_prefills={attn_metadata.num_prefills}"
            ),
            (
                "  attention_shape: "
                f"num_reqs={attn_metadata.num_reqs}, "
                f"max_query_len={attn_metadata.max_query_len}, "
                f"max_seq_len={attn_metadata.max_seq_len}, "
                f"head_dim={attn_metadata.head_dim}"
            ),
            f"  query_start_loc: {_tensor_preview(attn_metadata.query_start_loc)}",
            f"  slot_mapping: {_tensor_preview(slot_mapping)}",
        ]
    )

    decode_metadata = attn_metadata.decode
    if decode_metadata is None:
        lines.append("  decode_metadata: None")
        return lines

    kv_seq_lens = decode_metadata.dcp_tot_seq_lens
    if kv_seq_lens is None:
        kv_seq_lens = decode_metadata.seq_lens
    total_kv_tokens = int(kv_seq_lens.sum().item()) if kv_seq_lens.numel() > 0 else 0

    lines.extend(
        [
            f"  kv_tokens_used_total: {total_kv_tokens}",
            _tensor_stats_line("kv_tokens_used_per_request", kv_seq_lens),
            f"  decode_seq_lens: {_tensor_preview(decode_metadata.seq_lens)}",
            (
                "  decode_dcp_tot_seq_lens: "
                f"{_tensor_preview(decode_metadata.dcp_tot_seq_lens)}"
            ),
            f"  decode_block_table: {_tensor_preview(decode_metadata.block_table)}",
        ]
    )
    return lines


def _get_kernel_variant_specs(kernel_mode: str) -> list[KernelVariantSpec]:
    all_specs = [
        KernelVariantSpec(
            name="original",
            layer_cls=KimiK25Nvfp4MLAAttention,
            prefix="benchmark.original.layer0.self_attn",
        ),
        KernelVariantSpec(
            name="forked",
            layer_cls=ForkedKimiK25Nvfp4MLAAttention,
            prefix="benchmark.forked.layer0.self_attn",
        ),
    ]
    if kernel_mode == "both":
        return all_specs
    return [spec for spec in all_specs if spec.name == kernel_mode]


def _make_attention_layer(
    *,
    spec: KernelVariantSpec,
    vllm_config: VllmConfig,
    hf_config,
    model_dtype: torch.dtype,
    device: torch.device,
    source_layer: KimiK25Nvfp4MLAAttention | None = None,
) -> KimiK25Nvfp4MLAAttention:
    with set_default_torch_dtype(model_dtype):
        layer = spec.layer_cls(
            vllm_config=vllm_config,
            config=hf_config,
            cache_config=vllm_config.cache_config,
            quant_config=None,
            prefix=spec.prefix,
        )

    layer = layer.to(device=device, dtype=model_dtype).eval()
    if source_layer is None:
        _initialize_random_weights(layer)
    else:
        layer.load_state_dict(source_layer.state_dict())

    # The monolithic decode path consumes the derived W_UK_T / W_UV tensors
    # that MLA materializes after loading weights.
    layer.mla_attn.process_weights_after_loading(model_dtype)
    return layer


def _benchmark_attention_variant(
    *,
    args: argparse.Namespace,
    spec: KernelVariantSpec,
    layer: KimiK25Nvfp4MLAAttention,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    vllm_config: VllmConfig,
    per_layer_attn_metadata: dict[str, object],
    per_layer_slot_mapping: dict[str, torch.Tensor],
) -> BenchmarkVariantResult:
    run_layer = layer
    if args.torch_compile:
        run_layer = torch.compile(
            layer,
            dynamic=False,
            backend=current_platform.simple_compile_backend,
        )

    start_event = torch.Event(enable_timing=True)
    end_event = torch.Event(enable_timing=True)
    times_ms: list[float] = []
    last_output: torch.Tensor | None = None
    warmup_iters = max(1, args.warmup) if args.torch_compile else args.warmup
    batch_descriptor = (
        BatchDescriptor(
            num_tokens=args.batch_size,
            num_reqs=args.batch_size,
            uniform=True,
        )
        if args.cudagraph
        else None
    )
    summary_cg_mode = CUDAGraphMode.FULL if args.cudagraph else CUDAGraphMode.NONE
    with set_forward_context(
        per_layer_attn_metadata,
        vllm_config,
        num_tokens=args.batch_size,
        slot_mapping=per_layer_slot_mapping,
        cudagraph_runtime_mode=summary_cg_mode,
        batch_descriptor=(
            batch_descriptor if summary_cg_mode == CUDAGraphMode.FULL else None
        ),
    ):
        mla_forward_context_lines = _summarize_mla_forward_context(layer.mla_attn.layer_name)

    def run_forward(cg_mode: CUDAGraphMode) -> torch.Tensor:
        with set_forward_context(
            per_layer_attn_metadata,
            vllm_config,
            num_tokens=args.batch_size,
            slot_mapping=per_layer_slot_mapping,
            cudagraph_runtime_mode=cg_mode,
            batch_descriptor=(
                batch_descriptor if cg_mode == CUDAGraphMode.FULL else None
            ),
        ):
            return run_layer(positions=positions, hidden_states=hidden_states)

    for _ in range(warmup_iters):
        last_output = run_forward(CUDAGraphMode.NONE)

    torch.cuda.synchronize()

    if args.cudagraph:
        graph = torch.cuda.CUDAGraph()
        with graph_capture(device=positions.device) as graph_capture_context:
            with torch.cuda.graph(graph, stream=graph_capture_context.stream):
                last_output = run_forward(CUDAGraphMode.FULL)

        torch.cuda.synchronize()
        for _ in range(max(1, warmup_iters)):
            graph.replay()
        torch.cuda.synchronize()

        def benchmark_fn() -> None:
            graph.replay()

    else:

        def benchmark_fn() -> None:
            nonlocal last_output
            last_output = run_forward(CUDAGraphMode.NONE)

    for _ in range(args.trials):
        start_event.record()
        benchmark_fn()
        end_event.record()
        torch.cuda.synchronize()
        times_ms.append(start_event.elapsed_time(end_event))

    assert last_output is not None
    return BenchmarkVariantResult(
        spec=spec,
        layer=layer,
        output=last_output.detach().clone(),
        times_ms=times_ms,
        mla_forward_context_lines=mla_forward_context_lines,
    )


def _compare_variant_outputs(
    *,
    original_layer: KimiK25Nvfp4MLAAttention,
    forked_layer: KimiK25Nvfp4MLAAttention,
    kv_cache_template: torch.Tensor,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    vllm_config: VllmConfig,
    per_layer_attn_metadata: dict[str, object],
    per_layer_slot_mapping: dict[str, torch.Tensor],
    num_tokens: int,
) -> dict[str, object]:
    original_layer.mla_attn.kv_cache = kv_cache_template.clone()
    forked_layer.mla_attn.kv_cache = kv_cache_template.clone()

    with set_forward_context(
        per_layer_attn_metadata,
        vllm_config,
        num_tokens=num_tokens,
        slot_mapping=per_layer_slot_mapping,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
        batch_descriptor=None,
    ):
        original_output = original_layer(positions=positions, hidden_states=hidden_states)
        forked_output = forked_layer(positions=positions, hidden_states=hidden_states)

    original_output_summary = _summarize_tensor_finiteness(original_output)
    forked_output_summary = _summarize_tensor_finiteness(forked_output)
    original_output_float = original_output.float()
    forked_output_float = forked_output.float()
    finite_mask = torch.isfinite(original_output_float) & torch.isfinite(
        forked_output_float
    )
    nan_mask = torch.isnan(original_output_float) & torch.isnan(forked_output_float)
    posinf_mask = torch.isposinf(original_output_float) & torch.isposinf(
        forked_output_float
    )
    neginf_mask = torch.isneginf(original_output_float) & torch.isneginf(
        forked_output_float
    )

    finite_positions = int(finite_mask.sum().item())
    if finite_positions > 0:
        diff = (original_output_float[finite_mask] - forked_output_float[finite_mask]).abs()
        total_abs_diff = diff.sum().item()
        mean_abs_diff = diff.mean().item()
        max_abs_diff = diff.max().item()
    else:
        total_abs_diff = float("nan")
        mean_abs_diff = float("nan")
        max_abs_diff = float("nan")

    return {
        "allclose": torch.allclose(
            original_output, forked_output, rtol=1e-3, atol=1e-3
        ),
        "original_output_summary": original_output_summary,
        "forked_output_summary": forked_output_summary,
        "finite_positions_compared": finite_positions,
        "matching_nan_positions": int(nan_mask.sum().item()),
        "matching_posinf_positions": int(posinf_mask.sum().item()),
        "matching_neginf_positions": int(neginf_mask.sum().item()),
        "total_positions": original_output.numel(),
        "total_abs_diff": total_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "max_abs_diff": max_abs_diff,
    }


def _print_variant_result(
    *,
    args: argparse.Namespace,
    result: BenchmarkVariantResult,
    num_blocks: int,
    seq_lens_cpu: torch.Tensor,
) -> None:
    mean_ms = statistics.mean(result.times_ms)
    stdev_ms = statistics.pstdev(result.times_ms) if len(result.times_ms) > 1 else 0.0
    tokens_per_second = args.batch_size / (mean_ms / 1000.0)
    kv_cache_mebibytes = (
        result.layer.mla_attn.kv_cache.numel()
        * result.layer.mla_attn.kv_cache.element_size()
    ) / (1024**2)

    print(f"kernel_variant: {result.spec.name}")
    print(f"backend: {result.layer.mla_attn.attn_backend.get_name()}")
    print(f"cudagraph: {'enabled' if args.cudagraph else 'disabled'}")
    print(
        "torch_compile: "
        f"{'enabled' if args.torch_compile else 'disabled'}"
        + (
            f", backend={current_platform.simple_compile_backend}"
            if args.torch_compile
            else ""
        )
    )
    print(
        "hidden_size: "
        f"{result.layer.hidden_size}, num_heads: {result.layer.num_heads}, "
        f"q_lora_rank: {result.layer.q_lora_rank}, "
        f"kv_lora_rank: {result.layer.kv_lora_rank}"
    )
    print(
        "decode batch: "
        f"batch_size={args.batch_size}, "
        f"seq_len[min/mean/max]={int(seq_lens_cpu.min()):d}/"
        f"{seq_lens_cpu.float().mean().item():.1f}/"
        f"{int(seq_lens_cpu.max()):d}, "
        f"block_size={args.block_size}, num_blocks={num_blocks}"
    )
    print(
        "kv cache: "
        f"dtype={args.kv_cache_dtype}, shape={tuple(result.layer.mla_attn.kv_cache.shape)}, "
        f"size_mib={kv_cache_mebibytes:.1f}"
    )
    for line in result.mla_forward_context_lines:
        print(line)
    print(
        "output: "
        f"shape={tuple(result.output.shape)}, dtype={result.output.dtype}"
    )
    print(
        _format_tensor_finiteness(
            "output_values",
            _summarize_tensor_finiteness(result.output),
        )
    )
    print(
        "timing_ms: "
        f"mean={mean_ms:.3f}, std={stdev_ms:.3f}, "
        f"min={min(result.times_ms):.3f}, max={max(result.times_ms):.3f}"
    )
    print(f"throughput_tokens_per_s: {tokens_per_second:.1f}")


@torch.inference_mode()
def main() -> None:
    args = _parse_args()
    _validate_args(args)
    set_random_seed(args.seed)

    device = torch.device("cuda")
    vllm_config = _build_vllm_config(args)
    model_dtype = vllm_config.model_config.dtype
    hf_config = vllm_config.model_config.hf_config
    variant_specs = _get_kernel_variant_specs(args.kernel_mode)
    input_tensor_sanity_lines: list[str] = []

    with _maybe_init_single_rank_distributed(vllm_config):
        if not is_workspace_manager_initialized():
            init_workspace_manager(device)

        with set_current_vllm_config(vllm_config):
            layers: dict[str, KimiK25Nvfp4MLAAttention] = {}
            source_layer: KimiK25Nvfp4MLAAttention | None = None
            for spec in variant_specs:
                layer = _make_attention_layer(
                    spec=spec,
                    vllm_config=vllm_config,
                    hf_config=hf_config,
                    model_dtype=model_dtype,
                    device=device,
                    source_layer=source_layer,
                )
                if source_layer is None:
                    source_layer = layer
                layers[spec.name] = layer

            positions, hidden_states, common_attn_metadata, num_blocks = (
                _build_decode_batch(
                    batch_size=args.batch_size,
                    min_seq_len=args.min_seq_len,
                    max_seq_len=args.max_seq_len,
                    block_size=args.block_size,
                    hidden_size=source_layer.hidden_size,
                    device=device,
                    dtype=model_dtype,
                )
            )

            vllm_config.cache_config.num_gpu_blocks = num_blocks
            vllm_config.cache_config.num_cpu_blocks = 0

            reference_layer = next(iter(layers.values()))
            kv_cache_spec = reference_layer.mla_attn.get_kv_cache_spec(vllm_config)
            kv_cache_shape = reference_layer.mla_attn.attn_backend.get_kv_cache_shape(
                num_blocks,
                kv_cache_spec.block_size,
                kv_cache_spec.num_kv_heads,
                kv_cache_spec.head_size,
                getattr(kv_cache_spec, "cache_dtype_str", "auto") or "auto",
            )
            kv_cache_template = _make_random_kv_cache(
                kv_cache_shape,
                kv_cache_spec.dtype,
                device,
                reference_layer.mla_attn.kv_cache_dtype,
            )
            for layer in layers.values():
                layer.mla_attn.kv_cache = kv_cache_template.clone()
            input_tensor_sanity_lines = _summarize_input_tensor_sanity(
                reference_layer,
                kv_cache_template,
            )

            layer_names = [layer.mla_attn.layer_name for layer in layers.values()]
            builder_cls = reference_layer.mla_attn.attn_backend.get_builder_cls()
            builder = builder_cls(
                kv_cache_spec=kv_cache_spec,
                layer_names=layer_names,
                vllm_config=vllm_config,
                device=device,
            )
            if args.cudagraph:
                attn_metadata = builder.build_for_cudagraph_capture(
                    common_attn_metadata
                )
            else:
                attn_metadata = builder.build(
                    common_prefix_len=0,
                    common_attn_metadata=common_attn_metadata,
                    fast_build=False,
                )

            per_layer_attn_metadata = {
                layer_name: attn_metadata for layer_name in layer_names
            }
            per_layer_slot_mapping = {
                layer_name: common_attn_metadata.slot_mapping for layer_name in layer_names
            }

            results: list[BenchmarkVariantResult] = []
            for spec in variant_specs:
                layer = layers[spec.name]
                layer.mla_attn.kv_cache = kv_cache_template.clone()
                results.append(
                    _benchmark_attention_variant(
                        args=args,
                        spec=spec,
                        layer=layer,
                        positions=positions,
                        hidden_states=hidden_states,
                        vllm_config=vllm_config,
                        per_layer_attn_metadata=per_layer_attn_metadata,
                        per_layer_slot_mapping=per_layer_slot_mapping,
                    )
                )

            comparison = None
            if args.kernel_mode == "both":
                comparison = _compare_variant_outputs(
                    original_layer=layers["original"],
                    forked_layer=layers["forked"],
                    kv_cache_template=kv_cache_template,
                    positions=positions,
                    hidden_states=hidden_states,
                    vllm_config=vllm_config,
                    per_layer_attn_metadata=per_layer_attn_metadata,
                    per_layer_slot_mapping=per_layer_slot_mapping,
                    num_tokens=args.batch_size,
                )

    seq_lens_cpu = common_attn_metadata._seq_lens_cpu
    print(f"model: {args.model}")
    print(f"kernel_mode: {args.kernel_mode}")
    for line in input_tensor_sanity_lines:
        print(line)
    print()
    for idx, result in enumerate(results):
        if idx > 0:
            print()
        _print_variant_result(
            args=args,
            result=result,
            num_blocks=num_blocks,
            seq_lens_cpu=seq_lens_cpu,
        )

    if comparison is not None:
        print()
        print("output_comparison:")
        print("  note: fresh forward after resetting both KV caches")
        print(f"  allclose_rtol_1e-3_atol_1e-3: {comparison['allclose']}")
        print(
            _format_tensor_finiteness(
                "original_output",
                comparison["original_output_summary"],
            )
        )
        print(
            _format_tensor_finiteness(
                "forked_output",
                comparison["forked_output_summary"],
            )
        )
        print(
            "  finite_positions_compared: "
            f"{comparison['finite_positions_compared']}/{comparison['total_positions']}"
        )
        print(
            "  matching_nonfinite_positions: "
            f"nan={comparison['matching_nan_positions']}, "
            f"+inf={comparison['matching_posinf_positions']}, "
            f"-inf={comparison['matching_neginf_positions']}"
        )
        print("  diff_stats_note: computed only on positions where both outputs are finite")
        print(f"  total_abs_diff: {comparison['total_abs_diff']:.6f}")
        print(f"  mean_abs_diff: {comparison['mean_abs_diff']:.6f}")
        print(f"  max_abs_diff: {comparison['max_abs_diff']:.6f}")


if __name__ == "__main__":
    main()
