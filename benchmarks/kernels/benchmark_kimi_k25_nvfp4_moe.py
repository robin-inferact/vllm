# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Microbenchmark for Kimi-K2.5 NVFP4 MoE decode-shaped forwards.

This benchmark instantiates the specialized Kimi-K2.5 NVFP4 MoE module with
the real text config, initializes random ModelOpt NVFP4-formatted weights, runs
the same post-load weight conversion hooks used by model loading, and exercises
the direct FlashInfer TRTLLM NVFP4 MoE path:

    router GEMM -> NVFP4 input quant -> trtllm_fp4_block_scale_moe
    -> shared expert add/scale -> tensor-parallel all-reduce

For multi-GPU tensor-parallel benchmarking, launch this script with `torchrun`
and set `--tensor-parallel-size` to match `WORLD_SIZE`.
"""

from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass

import torch
from benchmark_kimi_k25_nvfp4_mla_attention import (
    DistributedRuntime,
    TensorFiniteSummary,
    _barrier_if_distributed,
    _build_vllm_config,
    _get_distributed_runtime,
    _maybe_init_distributed,
    _reduce_max_time_ms,
    _set_cuda_device,
    _summarize_tensor_finiteness,
)

from vllm.config import CUDAGraphMode, VllmConfig, set_current_vllm_config
from vllm.distributed import graph_capture
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config
from vllm.model_executor.models.deepseek_v2 import DeepseekV2MoE
from vllm.model_executor.specialized_models.kimi_k2_5_nvfp4.model import (
    KimiK25Nvfp4MoE,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_default_torch_dtype, set_random_seed

MoEModule = DeepseekV2MoE | KimiK25Nvfp4MoE


@dataclass(frozen=True)
class MoEVariantSpec:
    name: str
    prefix: str


@dataclass
class MoEBenchmarkResult:
    spec: MoEVariantSpec
    moe: MoEModule
    output: torch.Tensor
    times_ms: list[float]


@dataclass(frozen=True)
class OutputComparison:
    original_name: str
    optimized_name: str
    allclose: bool
    rtol: float
    atol: float
    original_output_summary: TensorFiniteSummary
    optimized_output_summary: TensorFiniteSummary
    finite_positions_compared: int
    matching_nan_positions: int
    matching_posinf_positions: int
    matching_neginf_positions: int
    total_positions: int
    total_abs_diff: float
    mean_abs_diff: float
    max_abs_diff: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark KimiK25Nvfp4MoE decode-shaped forwards."
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
        help="Number of decode tokens in the synthetic batch.",
    )
    parser.add_argument(
        "--max-model-len",
        dest="max_seq_len",
        type=int,
        default=4096,
        help="Maximum model length used for config construction.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=64,
        help="KV block size used for config construction.",
    )
    parser.add_argument(
        "--kv-cache-dtype",
        choices=["auto", "float16", "bfloat16", "fp8", "fp8_e4m3"],
        default="fp8",
        help="KV cache dtype used for config construction.",
    )
    parser.add_argument(
        "--moe-backend",
        choices=["auto", "flashinfer_trtllm"],
        default="flashinfer_trtllm",
        help="MoE backend to request for the specialized Kimi MoE path.",
    )
    parser.add_argument(
        "--moe-mode",
        choices=["specialized", "original", "both"],
        default="specialized",
        help=(
            "Which MoE implementation to benchmark. Use 'both' to also run "
            "the original DeepseekV2MoE/FusedMoE path and compare outputs."
        ),
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
        help="Capture the MoE forward in a CUDA graph and replay it.",
    )
    parser.add_argument(
        "--torch-compile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Compile the MoE module with torch.compile before benchmarking. "
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
    if args.max_seq_len <= 0:
        raise ValueError("--max-model-len must be positive.")
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


def _modelopt_nvfp4_config() -> ModelOptNvFp4Config:
    return ModelOptNvFp4Config(
        is_checkpoint_nvfp4_serialized=True,
        kv_cache_quant_algo=None,
        exclude_modules=[],
    )


def _build_benchmark_vllm_config(args: argparse.Namespace) -> VllmConfig:
    vllm_config = _build_vllm_config(args)
    vllm_config.quant_config = _modelopt_nvfp4_config()
    vllm_config.kernel_config.moe_backend = args.moe_backend
    return vllm_config


def _initialize_random_quantized_weights(module: torch.nn.Module) -> None:
    for name, param in module.named_parameters():
        with torch.no_grad():
            if param.dtype == torch.uint8:
                param.random_(0, 256)
            elif param.dtype == torch.float8_e4m3fn:
                values = torch.rand(param.shape, device=param.device) + 0.5
                param.copy_(values.to(param.dtype))
            elif "scale" in name:
                param.fill_(1.0)
            elif param.ndim >= 2:
                param.normal_(mean=0.0, std=0.02)
            elif "layernorm" in name.lower() or "norm" in name.lower():
                param.normal_(mean=1.0, std=0.1)
            else:
                param.zero_()


def _process_quant_methods(module: torch.nn.Module) -> None:
    for _, submodule in module.named_modules():
        quant_method = getattr(submodule, "quant_method", None)
        process = getattr(quant_method, "process_weights_after_loading", None)
        if process is not None:
            process(submodule)


def _clone_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().clone()
        for name, tensor in module.state_dict().items()
    }


def _format_tensor_finiteness(name: str, summary: TensorFiniteSummary) -> str:
    line = (
        f"  {name}: shape={summary.shape}, dtype={summary.dtype}, "
        f"finite={summary.finite_count}/{summary.numel}, "
        f"nan={summary.nan_count}, +inf={summary.posinf_count}, "
        f"-inf={summary.neginf_count}"
    )
    if summary.finite_count > 0:
        line += (
            ", finite_min/mean/max="
            f"{summary.min_finite:.6f}/{summary.mean_finite:.6f}/"
            f"{summary.max_finite:.6f}"
        )
    return line


def _get_moe_variant_specs(moe_mode: str) -> list[MoEVariantSpec]:
    all_specs = [
        MoEVariantSpec(
            name="original",
            prefix="benchmark.original.layers.0.mlp",
        ),
        MoEVariantSpec(
            name="specialized",
            prefix="benchmark.specialized.layers.0.mlp",
        ),
    ]
    if moe_mode == "both":
        return all_specs
    return [spec for spec in all_specs if spec.name == moe_mode]


def _make_raw_moe_module(
    *,
    spec: MoEVariantSpec,
    vllm_config: VllmConfig,
    device: torch.device,
) -> MoEModule:
    hf_config = vllm_config.model_config.hf_config
    assert isinstance(vllm_config.quant_config, ModelOptNvFp4Config)
    with set_default_torch_dtype(vllm_config.model_config.dtype):
        if spec.name == "original":
            moe = DeepseekV2MoE(
                config=hf_config,
                parallel_config=vllm_config.parallel_config,
                quant_config=vllm_config.quant_config,
                prefix=spec.prefix,
            )
        elif spec.name == "specialized":
            moe = KimiK25Nvfp4MoE(
                vllm_config=vllm_config,
                config=hf_config,
                quant_config=vllm_config.quant_config,
                prefix=spec.prefix,
            )
        else:
            raise ValueError(f"Unknown MoE variant: {spec.name}")

    return moe.to(device=device).eval()


def _benchmark_moe_variant(
    *,
    args: argparse.Namespace,
    runtime: DistributedRuntime,
    device: torch.device,
    spec: MoEVariantSpec,
    moe: MoEModule,
    hidden_states: torch.Tensor,
    vllm_config: VllmConfig,
) -> MoEBenchmarkResult:
    run_module = moe
    if args.torch_compile:
        run_module = torch.compile(
            moe,
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

    def run_forward(cg_mode: CUDAGraphMode) -> torch.Tensor:
        with set_forward_context(
            None,
            vllm_config,
            num_tokens=args.batch_size,
            cudagraph_runtime_mode=cg_mode,
            batch_descriptor=(
                batch_descriptor if cg_mode == CUDAGraphMode.FULL else None
            ),
        ):
            return run_module(hidden_states)

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
    output = last_output if not runtime.is_primary else last_output.detach().clone()
    return MoEBenchmarkResult(
        spec=spec,
        moe=moe,
        output=output,
        times_ms=times_ms,
    )


def _compare_outputs(
    *,
    original_name: str,
    optimized_name: str,
    original_output: torch.Tensor,
    optimized_output: torch.Tensor,
    rtol: float,
    atol: float,
) -> OutputComparison:
    original_output_summary = _summarize_tensor_finiteness(original_output)
    optimized_output_summary = _summarize_tensor_finiteness(optimized_output)
    original_output_float = original_output.float()
    optimized_output_float = optimized_output.float()
    finite_mask = torch.isfinite(original_output_float) & torch.isfinite(
        optimized_output_float
    )
    nan_mask = torch.isnan(original_output_float) & torch.isnan(optimized_output_float)
    posinf_mask = torch.isposinf(original_output_float) & torch.isposinf(
        optimized_output_float
    )
    neginf_mask = torch.isneginf(original_output_float) & torch.isneginf(
        optimized_output_float
    )

    finite_positions = int(finite_mask.sum().item())
    if finite_positions > 0:
        diff = (
            original_output_float[finite_mask] - optimized_output_float[finite_mask]
        ).abs()
        total_abs_diff = float(diff.sum().item())
        mean_abs_diff = float(diff.mean().item())
        max_abs_diff = float(diff.max().item())
    else:
        total_abs_diff = float("nan")
        mean_abs_diff = float("nan")
        max_abs_diff = float("nan")

    return OutputComparison(
        original_name=original_name,
        optimized_name=optimized_name,
        allclose=torch.allclose(
            original_output,
            optimized_output,
            rtol=rtol,
            atol=atol,
        ),
        rtol=rtol,
        atol=atol,
        original_output_summary=original_output_summary,
        optimized_output_summary=optimized_output_summary,
        finite_positions_compared=finite_positions,
        matching_nan_positions=int(nan_mask.sum().item()),
        matching_posinf_positions=int(posinf_mask.sum().item()),
        matching_neginf_positions=int(neginf_mask.sum().item()),
        total_positions=original_output.numel(),
        total_abs_diff=total_abs_diff,
        mean_abs_diff=mean_abs_diff,
        max_abs_diff=max_abs_diff,
    )


def _print_output_comparison(comparison: OutputComparison) -> None:
    print("output_comparison:")
    print(f"  original: {comparison.original_name}")
    print(f"  optimized: {comparison.optimized_name}")
    print(
        "  allclose: "
        f"{comparison.allclose} "
        f"(rtol={comparison.rtol:g}, atol={comparison.atol:g})"
    )
    print(
        _format_tensor_finiteness(
            f"{comparison.original_name}_output",
            comparison.original_output_summary,
        )
    )
    print(
        _format_tensor_finiteness(
            f"{comparison.optimized_name}_output",
            comparison.optimized_output_summary,
        )
    )
    print(
        "  finite_positions_compared: "
        f"{comparison.finite_positions_compared}/{comparison.total_positions}"
    )
    print(
        "  matching_nonfinite_positions: "
        f"nan={comparison.matching_nan_positions}, "
        f"+inf={comparison.matching_posinf_positions}, "
        f"-inf={comparison.matching_neginf_positions}"
    )
    print("  diff_stats_note: computed only where both outputs are finite")
    print(f"  total_abs_diff: {comparison.total_abs_diff:.6f}")
    print(f"  mean_abs_diff: {comparison.mean_abs_diff:.6f}")
    print(f"  max_abs_diff: {comparison.max_abs_diff:.6f}")


def _print_variant_result(
    *,
    args: argparse.Namespace,
    runtime: DistributedRuntime,
    result: MoEBenchmarkResult,
) -> None:
    moe = result.moe
    mean_ms = statistics.mean(result.times_ms)
    stdev_ms = statistics.pstdev(result.times_ms) if len(result.times_ms) > 1 else 0.0
    tokens_per_second = args.batch_size / (mean_ms / 1000.0)

    print(f"moe_variant: {result.spec.name}")
    print(f"moe_backend: {args.moe_backend}")
    print(
        "distributed: "
        f"tensor_parallel_size={args.tensor_parallel_size}, "
        f"world_size={runtime.world_size}"
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
        "moe_shape: "
        f"batch_size={args.batch_size}, "
        f"hidden_size={moe.experts.moe_config.hidden_dim}, "
        f"num_experts={moe.n_routed_experts}, top_k={moe.experts.top_k}, "
        f"num_groups={moe.experts.num_expert_group}, "
        f"topk_group={moe.experts.topk_group}, "
        f"intermediate_per_partition="
        f"{moe.experts.moe_config.intermediate_size_per_partition}"
    )
    shared_expert_overlap = getattr(moe, "shared_expert_overlap", None)
    overlap_stream_enabled = (
        shared_expert_overlap is not None
        and shared_expert_overlap.stream is not None
    )
    print(
        "shared_experts: "
        f"{moe.n_shared_experts}, overlap_stream="
        f"{overlap_stream_enabled}"
    )
    print(f"output: shape={tuple(result.output.shape)}, dtype={result.output.dtype}")
    print(
        _format_tensor_finiteness(
            "output_values",
            _summarize_tensor_finiteness(result.output),
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
    variant_specs = _get_moe_variant_specs(args.moe_mode)

    with _maybe_init_distributed(
        vllm_config,
        runtime,
        args.tensor_parallel_size,
    ), set_current_vllm_config(vllm_config):
        moes: dict[str, MoEModule] = {}
        source_state_dict: dict[str, torch.Tensor] | None = None
        for spec in variant_specs:
            moe = _make_raw_moe_module(
                spec=spec,
                vllm_config=vllm_config,
                device=device,
            )
            if source_state_dict is None:
                _initialize_random_quantized_weights(moe)
                source_state_dict = _clone_state_dict(moe)
            else:
                moe.load_state_dict(source_state_dict)
            moes[spec.name] = moe

        for moe in moes.values():
            _process_quant_methods(moe)

        hidden_states = torch.randn(
            args.batch_size,
            vllm_config.model_config.hf_config.hidden_size,
            device=device,
            dtype=model_dtype,
        )

        results: list[MoEBenchmarkResult] = []
        for spec in variant_specs:
            results.append(
                _benchmark_moe_variant(
                    args=args,
                    runtime=runtime,
                    device=device,
                    spec=spec,
                    moe=moes[spec.name],
                    hidden_states=hidden_states,
                    vllm_config=vllm_config,
                )
            )

        comparison = None
        if args.moe_mode == "both":
            _barrier_if_distributed()
            with set_forward_context(
                None,
                vllm_config,
                num_tokens=args.batch_size,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
            ):
                original_output = moes["original"](hidden_states)
            with set_forward_context(
                None,
                vllm_config,
                num_tokens=args.batch_size,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
            ):
                specialized_output = moes["specialized"](hidden_states)
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
        print(f"model: {args.model}")
        print("benchmark: kimi_k25_nvfp4_moe")
        print(f"moe_mode: {args.moe_mode}")
        for idx, result in enumerate(results):
            if idx > 0:
                print()
            _print_variant_result(
                args=args,
                runtime=runtime,
                result=result,
            )
        if comparison is not None:
            print()
            _print_output_comparison(comparison)


if __name__ == "__main__":
    main()
