# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Microbenchmark for the combined Kimi-K2.5 NVFP4 MLA + MoE layer.

This benchmark instantiates `KimiK25Nvfp4DecoderLayer` at the first MoE layer
index and times the upper-level decoder-layer API that combines:

    input RMSNorm -> specialized MLA attention -> post-attention RMSNorm
    -> specialized FlashInfer TRTLLM NVFP4 MoE

The synthetic batch is decode-shaped, uses the real Kimi text config, and
initializes random ModelOpt NVFP4-formatted weights before running the same
post-load conversion hooks used by model loading.

For multi-GPU tensor-parallel benchmarking, launch this script with `torchrun`
and set `--tensor-parallel-size` to match `WORLD_SIZE`.
"""

from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass

import torch
from benchmark_kimi_k25_nvfp4_mla_attention import (
    _barrier_if_distributed,
    _build_decode_batch,
    _get_distributed_runtime,
    _make_random_kv_cache,
    _maybe_init_distributed,
    _reduce_max_time_ms,
    _set_cuda_device,
    _summarize_mla_forward_context,
    _summarize_tensor_finiteness,
)
from benchmark_kimi_k25_nvfp4_moe import (
    _build_benchmark_vllm_config,
    _clone_state_dict,
    _compare_outputs,
    _format_tensor_finiteness,
    _initialize_random_quantized_weights,
    _print_output_comparison,
    _process_quant_methods,
)

from vllm.config import CUDAGraphMode, VllmConfig, set_current_vllm_config
from vllm.distributed import graph_capture
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.models.deepseek_v2 import DeepseekV2MoE
from vllm.model_executor.specialized_models.kimi_k2_5_nvfp4.model import (
    KimiK25Nvfp4DecoderLayer,
    KimiK25Nvfp4MLAAttention,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_default_torch_dtype, set_random_seed
from vllm.v1.worker.workspace import (
    init_workspace_manager,
    is_workspace_manager_initialized,
)


class OriginalKimiK25Nvfp4DecoderLayer(torch.nn.Module):
    """Benchmark-local pre-refactor layer: original MLA + generic MoE."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        config,
        layer_idx: int,
        prefix: str,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = KimiK25Nvfp4MLAAttention(
            vllm_config=vllm_config,
            config=config,
            cache_config=vllm_config.cache_config,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

        moe_layer_freq = getattr(config, "moe_layer_freq", 1)
        self.is_moe = (
            config.n_routed_experts is not None
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % moe_layer_freq == 0
        )
        if not self.is_moe:
            raise ValueError(
                f"layer_idx={layer_idx} is not a MoE layer for this Kimi config."
            )
        self.mlp = DeepseekV2MoE(
            config=config,
            parallel_config=vllm_config.parallel_config,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.mlp",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(positions, hidden_states)

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


DecoderLayerModule = KimiK25Nvfp4DecoderLayer | OriginalKimiK25Nvfp4DecoderLayer


@dataclass(frozen=True)
class DecoderVariantSpec:
    name: str
    prefix: str


@dataclass
class DecoderBenchmarkResult:
    spec: DecoderVariantSpec
    layer: DecoderLayerModule
    output: torch.Tensor
    times_ms: list[float]
    mla_context_lines: list[str]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark combined Kimi-K2.5 NVFP4 MLA attention + MoE."
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
        "--moe-backend",
        choices=["auto", "flashinfer_trtllm"],
        default="flashinfer_trtllm",
        help="MoE backend to request for the specialized Kimi MoE path.",
    )
    parser.add_argument(
        "--layer-mode",
        choices=["specialized", "original", "both"],
        default="specialized",
        help=(
            "Which decoder-layer implementation to benchmark. Use 'both' to "
            "run the original MLA + generic MoE path and compare outputs."
        ),
    )
    parser.add_argument(
        "--layer-idx",
        type=int,
        default=None,
        help=(
            "Decoder layer index to instantiate. Defaults to the first Kimi "
            "MoE layer."
        ),
    )
    parser.add_argument(
        "--residual-mode",
        choices=["none", "provided"],
        default="provided",
        help="Whether to pass an incoming residual tensor to the decoder layer.",
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
        "--compare-rtol",
        type=float,
        default=1e-3,
        help="Relative tolerance for original-vs-specialized output comparison.",
    )
    parser.add_argument(
        "--compare-atol",
        type=float,
        default=1e-3,
        help="Absolute tolerance for original-vs-specialized output comparison.",
    )
    parser.add_argument(
        "--cudagraph",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Capture the decoder layer forward in a CUDA graph and replay it.",
    )
    parser.add_argument(
        "--torch-compile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Compile the decoder layer with torch.compile before benchmarking. "
            "Compilation time is excluded from the timed trials."
        ),
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help=(
            "Tensor parallel world size for the benchmark. "
            "Use torchrun and match WORLD_SIZE when this is > 1."
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
    if args.tensor_parallel_size <= 0:
        raise ValueError("--tensor-parallel-size must be positive.")
    if args.warmup < 0 or args.trials <= 0:
        raise ValueError("--warmup must be >= 0 and --trials must be > 0.")
    if args.compare_rtol < 0 or args.compare_atol < 0:
        raise ValueError("--compare-rtol and --compare-atol must be non-negative.")
    if not current_platform.is_cuda():
        raise RuntimeError("This benchmark requires a CUDA device.")


def _get_decoder_variant_specs(
    layer_mode: str,
    layer_idx: int,
) -> list[DecoderVariantSpec]:
    all_specs = [
        DecoderVariantSpec(
            name="original",
            prefix=f"benchmark.original.layers.{layer_idx}",
        ),
        DecoderVariantSpec(
            name="specialized",
            prefix=f"benchmark.specialized.layers.{layer_idx}",
        ),
    ]
    if layer_mode == "both":
        return all_specs
    return [spec for spec in all_specs if spec.name == layer_mode]


def _get_moe_layer_idx(args: argparse.Namespace, hf_config) -> int:
    layer_idx = (
        args.layer_idx
        if args.layer_idx is not None
        else getattr(hf_config, "first_k_dense_replace", 0)
    )
    moe_layer_freq = getattr(hf_config, "moe_layer_freq", 1)
    if (
        hf_config.n_routed_experts is None
        or layer_idx < hf_config.first_k_dense_replace
        or layer_idx % moe_layer_freq != 0
    ):
        raise ValueError(
            f"layer_idx={layer_idx} is not a MoE layer for this Kimi config."
        )
    return layer_idx


def _make_raw_decoder_layer(
    *,
    args: argparse.Namespace,
    spec: DecoderVariantSpec,
    vllm_config: VllmConfig,
    device: torch.device,
) -> DecoderLayerModule:
    hf_config = vllm_config.model_config.hf_config
    layer_idx = _get_moe_layer_idx(args, hf_config)

    with set_default_torch_dtype(vllm_config.model_config.dtype):
        if spec.name == "original":
            layer = OriginalKimiK25Nvfp4DecoderLayer(
                vllm_config=vllm_config,
                config=hf_config,
                layer_idx=layer_idx,
                prefix=spec.prefix,
            )
        elif spec.name == "specialized":
            layer = KimiK25Nvfp4DecoderLayer(
                vllm_config=vllm_config,
                config=hf_config,
                layer_idx=layer_idx,
                prefix=spec.prefix,
            )
        else:
            raise ValueError(f"Unknown decoder-layer variant: {spec.name}")

    return layer.to(device=device).eval()


def _benchmark_decoder_variant(
    *,
    args: argparse.Namespace,
    device: torch.device,
    spec: DecoderVariantSpec,
    layer: DecoderLayerModule,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    vllm_config: VllmConfig,
    per_layer_attn_metadata: dict[str, object],
    per_layer_slot_mapping: dict[str, torch.Tensor],
) -> DecoderBenchmarkResult:
    run_layer = layer
    if args.torch_compile:
        run_layer = torch.compile(
            layer,
            dynamic=False,
            backend=current_platform.simple_compile_backend,
        )

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
        mla_context_lines = _summarize_mla_forward_context(
            layer.self_attn.mla_attn.layer_name
        )

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
            output, _ = run_layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )
            return output

    warmup_iters = max(1, args.warmup) if args.torch_compile else args.warmup
    last_output: torch.Tensor | None = None
    for _ in range(warmup_iters):
        last_output = run_forward(CUDAGraphMode.NONE)

    torch.cuda.synchronize()
    _barrier_if_distributed()

    if args.cudagraph:
        graph = torch.cuda.CUDAGraph()
        _barrier_if_distributed()
        with (
            graph_capture(device=device) as graph_capture_context,
            torch.cuda.graph(graph, stream=graph_capture_context.stream),
        ):
            last_output = run_forward(CUDAGraphMode.FULL)

        torch.cuda.synchronize()
        for _ in range(max(1, warmup_iters)):
            graph.replay()
        torch.cuda.synchronize()
        _barrier_if_distributed()

        def benchmark_fn() -> None:
            graph.replay()

    else:

        def benchmark_fn() -> None:
            nonlocal last_output
            last_output = run_forward(CUDAGraphMode.NONE)

    start_event = torch.Event(enable_timing=True)
    end_event = torch.Event(enable_timing=True)
    times_ms: list[float] = []
    _barrier_if_distributed()
    for _ in range(args.trials):
        start_event.record()
        benchmark_fn()
        end_event.record()
        torch.cuda.synchronize()
        times_ms.append(
            _reduce_max_time_ms(
                start_event.elapsed_time(end_event),
                device=device,
            )
        )

    assert last_output is not None
    _barrier_if_distributed()
    return DecoderBenchmarkResult(
        spec=spec,
        layer=layer,
        output=last_output.detach().clone(),
        times_ms=times_ms,
        mla_context_lines=mla_context_lines,
    )


def _print_result(
    *,
    args: argparse.Namespace,
    runtime_world_size: int,
    result: DecoderBenchmarkResult,
    num_blocks: int,
    seq_lens_cpu: torch.Tensor,
) -> None:
    layer = result.layer
    output = result.output
    mean_ms = statistics.mean(result.times_ms)
    stdev_ms = statistics.pstdev(result.times_ms) if len(result.times_ms) > 1 else 0.0
    tokens_per_second = args.batch_size / (mean_ms / 1000.0)
    moe = layer.mlp
    kv_cache = layer.self_attn.mla_attn.kv_cache
    kv_cache_mebibytes = (kv_cache.numel() * kv_cache.element_size()) / (1024**2)

    print(f"layer_variant: {result.spec.name}")
    print(f"moe_backend: {args.moe_backend}")
    print(
        "distributed: "
        f"tensor_parallel_size={args.tensor_parallel_size}, "
        f"world_size={runtime_world_size}"
    )
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
        "decode batch: "
        f"batch_size={args.batch_size}, "
        f"seq_len[min/mean/max]={int(seq_lens_cpu.min()):d}/"
        f"{seq_lens_cpu.float().mean().item():.1f}/"
        f"{int(seq_lens_cpu.max()):d}, "
        f"block_size={args.block_size}, num_blocks={num_blocks}"
    )
    print(
        "attention_shape: "
        f"hidden_size={layer.self_attn.hidden_size}, "
        f"num_local_heads={layer.self_attn.num_local_heads}, "
        f"q_lora_rank={layer.self_attn.q_lora_rank}, "
        f"kv_lora_rank={layer.self_attn.kv_lora_rank}"
    )
    print(
        "moe_shape: "
        f"num_experts={moe.n_routed_experts}, top_k={moe.experts.top_k}, "
        f"num_groups={moe.experts.num_expert_group}, "
        f"topk_group={moe.experts.topk_group}, "
        f"intermediate_per_partition="
        f"{moe.experts.moe_config.intermediate_size_per_partition}"
    )
    print(
        "kv cache: "
        f"dtype={args.kv_cache_dtype}, shape={tuple(kv_cache.shape)}, "
        f"size_mib={kv_cache_mebibytes:.1f}"
    )
    for line in result.mla_context_lines:
        print(line)
    print(f"output: shape={tuple(output.shape)}, dtype={output.dtype}")
    print(
        _format_tensor_finiteness(
            "output_values",
            _summarize_tensor_finiteness(output),
        )
    )
    if args.tensor_parallel_size > 1:
        print("timing_aggregation: max_across_tensor_parallel_ranks")
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
    runtime = _get_distributed_runtime(args)
    device = _set_cuda_device(runtime)
    set_random_seed(args.seed)

    vllm_config = _build_benchmark_vllm_config(args)
    model_dtype = vllm_config.model_config.dtype
    hf_config = vllm_config.model_config.hf_config
    layer_idx = _get_moe_layer_idx(args, hf_config)
    variant_specs = _get_decoder_variant_specs(args.layer_mode, layer_idx)

    with _maybe_init_distributed(
        vllm_config,
        runtime,
        args.tensor_parallel_size,
    ):
        if not is_workspace_manager_initialized():
            init_workspace_manager(device)

        with set_current_vllm_config(vllm_config):
            layers: dict[str, DecoderLayerModule] = {}
            source_state_dict: dict[str, torch.Tensor] | None = None
            for spec in variant_specs:
                layer = _make_raw_decoder_layer(
                    args=args,
                    spec=spec,
                    vllm_config=vllm_config,
                    device=device,
                )
                if source_state_dict is None:
                    _initialize_random_quantized_weights(layer)
                    source_state_dict = _clone_state_dict(layer)
                else:
                    layer.load_state_dict(source_state_dict)
                layers[spec.name] = layer

            for layer in layers.values():
                _process_quant_methods(layer)
                layer.self_attn.mla_attn.process_weights_after_loading(
                    vllm_config.model_config.dtype
                )

            reference_layer = next(iter(layers.values()))
            positions, hidden_states, common_attn_metadata, num_blocks = (
                _build_decode_batch(
                    batch_size=args.batch_size,
                    min_seq_len=args.min_seq_len,
                    max_seq_len=args.max_seq_len,
                    block_size=args.block_size,
                    hidden_size=reference_layer.self_attn.hidden_size,
                    device=device,
                    dtype=model_dtype,
                )
            )
            residual = (
                torch.randn_like(hidden_states)
                if args.residual_mode == "provided"
                else None
            )

            vllm_config.cache_config.num_gpu_blocks = num_blocks
            vllm_config.cache_config.num_cpu_blocks = 0
            mla = reference_layer.self_attn.mla_attn
            kv_cache_spec = mla.get_kv_cache_spec(vllm_config)
            kv_cache_shape = mla.attn_backend.get_kv_cache_shape(
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
                mla.kv_cache_dtype,
            )
            for layer in layers.values():
                layer.self_attn.mla_attn.kv_cache = kv_cache_template.clone()

            layer_names = [
                layer.self_attn.mla_attn.layer_name for layer in layers.values()
            ]
            builder_cls = mla.attn_backend.get_builder_cls()
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
                layer_name: common_attn_metadata.slot_mapping
                for layer_name in layer_names
            }

            results: list[DecoderBenchmarkResult] = []
            for spec in variant_specs:
                layer = layers[spec.name]
                layer.self_attn.mla_attn.kv_cache = kv_cache_template.clone()
                results.append(
                    _benchmark_decoder_variant(
                        args=args,
                        device=device,
                        spec=spec,
                        layer=layer,
                        positions=positions,
                        hidden_states=hidden_states,
                        residual=residual,
                        vllm_config=vllm_config,
                        per_layer_attn_metadata=per_layer_attn_metadata,
                        per_layer_slot_mapping=per_layer_slot_mapping,
                    )
                )

            comparison = None
            if args.layer_mode == "both":
                for layer in layers.values():
                    layer.self_attn.mla_attn.kv_cache = kv_cache_template.clone()
                _barrier_if_distributed()
                with set_forward_context(
                    per_layer_attn_metadata,
                    vllm_config,
                    num_tokens=args.batch_size,
                    slot_mapping=per_layer_slot_mapping,
                    cudagraph_runtime_mode=CUDAGraphMode.NONE,
                    batch_descriptor=None,
                ):
                    original_output, _ = layers["original"](
                        positions=positions,
                        hidden_states=hidden_states,
                        residual=residual,
                    )
                    specialized_output, _ = layers["specialized"](
                        positions=positions,
                        hidden_states=hidden_states,
                        residual=residual,
                    )
                torch.cuda.synchronize()
                _barrier_if_distributed()
                comparison = _compare_outputs(
                    original_name="original",
                    optimized_name="specialized",
                    original_output=original_output,
                    optimized_output=specialized_output,
                    rtol=args.compare_rtol,
                    atol=args.compare_atol,
                )

    if runtime.is_primary:
        seq_lens_cpu = common_attn_metadata._seq_lens_cpu
        print(f"model: {args.model}")
        print("benchmark: kimi_k25_nvfp4_mla_moe_decoder_layer")
        print(f"layer_mode: {args.layer_mode}")
        for idx, result in enumerate(results):
            if idx > 0:
                print()
            _print_result(
                args=args,
                runtime_world_size=runtime.world_size,
                result=result,
                num_blocks=num_blocks,
                seq_lens_cpu=seq_lens_cpu,
            )
        if comparison is not None:
            print()
            _print_output_comparison(comparison)


if __name__ == "__main__":
    main()
