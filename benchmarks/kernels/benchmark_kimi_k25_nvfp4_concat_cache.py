# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Microbenchmark the Kimi-K2.5 MLA concat/cache kernel.

This benchmark isolates the KV-cache update used by the specialized Kimi-K2.5
NVFP4 path after KV RoPE has already run:

    kv_c:  [num_tokens, kv_lora_rank]
    k_pe:  [num_tokens, qk_rope_head_dim]
    cache: [num_blocks, block_size, kv_lora_rank + qk_rope_head_dim]

The CuTeDSL version writes the already-rotated `k_pe` into the FP8 MLA KV cache.

Example:

    .venv/bin/python benchmarks/kernels/benchmark_kimi_k25_nvfp4_concat_cache.py \
        --versions cutedsl cuda --num-tokens 7 64 512 4096 \
        --check-correctness

    .venv/bin/python benchmarks/kernels/benchmark_kimi_k25_nvfp4_concat_cache.py \
        --versions cutedsl --num-tokens 4096 --use-cudagraph
"""

import math
import statistics
from dataclasses import dataclass
from typing import Any

import torch

from vllm import _custom_ops as ops
from vllm.platforms import current_platform
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.torch_utils import set_random_seed

KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
BLOCK_SIZE = 16
DTYPE = torch.bfloat16
KV_CACHE_DTYPE = "fp8"
KV_CACHE_SCALE = 0.02


@dataclass(frozen=True)
class BenchmarkConfig:
    num_tokens: int
    kv_lora_rank: int = KV_LORA_RANK
    qk_rope_head_dim: int = QK_ROPE_HEAD_DIM
    block_size: int = BLOCK_SIZE
    dtype: torch.dtype = DTYPE

    @property
    def num_blocks(self) -> int:
        return math.ceil(self.num_tokens / self.block_size)

    @property
    def entry_size(self) -> int:
        return self.kv_lora_rank + self.qk_rope_head_dim

    @property
    def cutedsl_bytes_per_call(self) -> int:
        elem_size = torch.tensor([], dtype=self.dtype).element_size()
        long_size = torch.tensor([], dtype=torch.long).element_size()
        return self.num_tokens * (
            self.kv_lora_rank * elem_size
            + self.qk_rope_head_dim * elem_size
            + self.entry_size
            + long_size
        )

    @property
    def cuda_bytes_per_call(self) -> int:
        return self.cutedsl_bytes_per_call


@dataclass
class ConcatCacheInputs:
    kv_c: torch.Tensor
    k_pe: torch.Tensor
    kv_cache: torch.Tensor
    slot_mapping: torch.Tensor
    scale: torch.Tensor


@dataclass
class KernelVersion:
    name: str
    make_runner: Any
    run_once: Any
    bytes_per_call: Any


_KERNEL_VERSION_CACHE: dict[str, KernelVersion] = {}


def _make_inputs(
    cfg: BenchmarkConfig,
    *,
    slot_mapping: torch.Tensor | None = None,
) -> ConcatCacheInputs:
    if slot_mapping is None:
        slot_mapping = torch.arange(cfg.num_tokens, device="cuda", dtype=torch.long)

    kv_c = (
        torch.randn(
            cfg.num_tokens,
            cfg.kv_lora_rank,
            device="cuda",
            dtype=cfg.dtype,
        )
        * 0.3
    )
    k_pe = (
        torch.randn(
            cfg.num_tokens,
            cfg.qk_rope_head_dim,
            device="cuda",
            dtype=cfg.dtype,
        )
        * 0.3
    )
    kv_cache = torch.empty(
        cfg.num_blocks,
        cfg.block_size,
        cfg.entry_size,
        device="cuda",
        dtype=torch.uint8,
    )
    scale = torch.tensor(KV_CACHE_SCALE, device="cuda", dtype=torch.float32)
    return ConcatCacheInputs(
        kv_c=kv_c,
        k_pe=k_pe,
        kv_cache=kv_cache,
        slot_mapping=slot_mapping,
        scale=scale,
    )


def _run_cuda(inputs: ConcatCacheInputs, cfg: BenchmarkConfig) -> None:
    ops.concat_and_cache_mla(
        inputs.kv_c,
        inputs.k_pe,
        inputs.kv_cache,
        inputs.slot_mapping,
        kv_cache_dtype=KV_CACHE_DTYPE,
        scale=inputs.scale,
    )


def _make_cuda_runner(
    inputs_pool: list[ConcatCacheInputs],
    cfg: BenchmarkConfig,
):
    index = 0

    def run() -> None:
        nonlocal index
        _run_cuda(inputs_pool[index], cfg)
        index = (index + 1) % len(inputs_pool)

    return run


def _build_cuda_version() -> KernelVersion:
    return KernelVersion(
        name="cuda",
        make_runner=_make_cuda_runner,
        run_once=_run_cuda,
        bytes_per_call=lambda cfg: cfg.cuda_bytes_per_call,
    )


def _build_cutedsl_version() -> KernelVersion:
    from vllm.model_executor.specialized_models.kimi_k2_5_nvfp4.model import (
        _run_kimik25_concat_and_cache_mla,
    )

    def cutedsl_concat_and_cache(
        inputs: ConcatCacheInputs,
        cfg: BenchmarkConfig,
    ) -> None:
        if cfg.kv_lora_rank != KV_LORA_RANK:
            raise ValueError(
                f"cutedsl requires kv_lora_rank={KV_LORA_RANK}, "
                f"got {cfg.kv_lora_rank}."
            )
        if cfg.qk_rope_head_dim != QK_ROPE_HEAD_DIM:
            raise ValueError(
                f"cutedsl requires qk_rope_head_dim={QK_ROPE_HEAD_DIM}, "
                f"got {cfg.qk_rope_head_dim}."
            )
        _run_kimik25_concat_and_cache_mla(
            kv_c=inputs.kv_c,
            k_pe=inputs.k_pe,
            kv_cache=inputs.kv_cache,
            slot_mapping=inputs.slot_mapping,
            kv_cache_dtype=KV_CACHE_DTYPE,
            scale=inputs.scale,
        )

    def make_cutedsl_runner(
        inputs_pool: list[ConcatCacheInputs],
        cfg: BenchmarkConfig,
    ):
        index = 0

        def run() -> None:
            nonlocal index
            cutedsl_concat_and_cache(inputs_pool[index], cfg)
            index = (index + 1) % len(inputs_pool)

        return run

    return KernelVersion(
        name="cutedsl",
        make_runner=make_cutedsl_runner,
        run_once=cutedsl_concat_and_cache,
        bytes_per_call=lambda cfg: cfg.cutedsl_bytes_per_call,
    )


KERNEL_VERSION_BUILDERS = {
    "cutedsl": _build_cutedsl_version,
    "cuda": _build_cuda_version,
}


def _get_kernel_version(name: str) -> KernelVersion:
    if name not in _KERNEL_VERSION_CACHE:
        _KERNEL_VERSION_CACHE[name] = KERNEL_VERSION_BUILDERS[name]()
    return _KERNEL_VERSION_CACHE[name]


def _resolve_version_names(version_names: list[str]) -> list[str]:
    if "all" in version_names:
        return list(KERNEL_VERSION_BUILDERS)

    deduped: list[str] = []
    for name in version_names:
        if name not in deduped:
            deduped.append(name)
    return deduped


def _resolve_baseline_version(
    version_names: list[str],
    baseline_version: str | None,
) -> str | None:
    if baseline_version is not None:
        if baseline_version not in version_names:
            raise ValueError(
                f"Baseline version {baseline_version!r} must be included in "
                f"--versions {version_names!r}."
            )
        return baseline_version
    if "cutedsl" in version_names:
        return "cutedsl"
    return version_names[0] if version_names else None


def _benchmark_runner(
    run_once,
    *,
    num_warmup_iters: int,
    num_iters: int,
    num_repeats: int,
    use_cudagraph: bool,
) -> list[float]:
    if num_iters < 1:
        raise ValueError("--num-iters must be at least 1.")

    if use_cudagraph:
        for _ in range(max(num_warmup_iters, 1)):
            run_once()
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        capture_stream = torch.cuda.Stream()
        with torch.cuda.graph(graph, stream=capture_stream):
            for _ in range(num_iters):
                run_once()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        latencies_us: list[float] = []

        for _ in range(num_repeats):
            start.record()
            graph.replay()
            end.record()
            torch.cuda.synchronize()
            latencies_us.append(start.elapsed_time(end) * 1000.0 / num_iters)

        return latencies_us

    for _ in range(num_warmup_iters):
        run_once()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    latencies_us: list[float] = []

    for _ in range(num_repeats):
        start.record()
        for _ in range(num_iters):
            run_once()
        end.record()
        torch.cuda.synchronize()
        latencies_us.append(start.elapsed_time(end) * 1000.0 / num_iters)

    return latencies_us


def _format_diff(value: float | None) -> str:
    return "-" if value is None else f"{value:.3e}"


def _format_speedup(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}x"


def _format_bandwidth(bytes_per_call: int, latency_us: float) -> float:
    return bytes_per_call / latency_us / 1e3


def _dequantize_cache(cache: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(cache, dtype=torch.float16)
    ops.convert_fp8(output, cache.contiguous(), scale.item(), kv_dtype=KV_CACHE_DTYPE)
    return output


def _run_correctness(
    version: KernelVersion,
    cfg: BenchmarkConfig,
) -> float:
    reference_inputs = _make_inputs(cfg)
    actual_inputs = _make_inputs(
        cfg,
        slot_mapping=reference_inputs.slot_mapping,
    )
    actual_inputs.kv_c.copy_(reference_inputs.kv_c)
    actual_inputs.k_pe.copy_(reference_inputs.k_pe)

    _run_cuda(reference_inputs, cfg)
    version.run_once(actual_inputs, cfg)
    torch.cuda.synchronize()

    expected_cache = _dequantize_cache(
        reference_inputs.kv_cache,
        reference_inputs.scale,
    )
    actual_cache = _dequantize_cache(actual_inputs.kv_cache, actual_inputs.scale)
    cache_diff = (actual_cache - expected_cache).abs().max().item()
    k_pe_diff = (
        actual_inputs.k_pe.float() - reference_inputs.k_pe.float()
    ).abs().max().item()
    return max(cache_diff, k_pe_diff)


def main(args) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    if not current_platform.is_device_capability_family(100):
        print(
            "Warning: this benchmark is primarily intended for Blackwell GPUs "
            "(SM10x)."
        )

    set_random_seed(args.seed)
    version_names = _resolve_version_names(args.versions)
    baseline_version = _resolve_baseline_version(version_names, args.baseline_version)
    versions = {name: _get_kernel_version(name) for name in version_names}

    if baseline_version is not None:
        print(f"Baseline version: {baseline_version}")

    version_col_width = max(len("version"), max(len(name) for name in version_names))
    header = (
        f"{'version':<{version_col_width}} "
        f"{'tokens':>8} {'blocks':>8} {'block':>6} "
        f"{'median_us':>12} {'min_us':>12} {'max_us':>12} {'gbps':>10} "
        f"{'speedup':>10} {'max_abs_diff':>14}"
    )
    print(header)
    print("-" * len(header))

    for num_tokens in args.num_tokens:
        cfg = BenchmarkConfig(
            num_tokens=num_tokens,
            block_size=args.block_size,
        )
        sample_inputs = _make_inputs(cfg)

        diffs: dict[str, float | None] = {}
        if args.check_correctness:
            for version_name in version_names:
                diffs[version_name] = _run_correctness(
                    versions[version_name],
                    cfg,
                )
        else:
            diffs = {version_name: None for version_name in version_names}

        medians_us: dict[str, float] = {}
        mins_us: dict[str, float] = {}
        maxs_us: dict[str, float] = {}

        for version_name in version_names:
            data_pool = [
                _make_inputs(
                    cfg,
                    slot_mapping=sample_inputs.slot_mapping,
                )
                for _ in range(args.num_buffers)
            ]
            runner = versions[version_name].make_runner(data_pool, cfg)
            latencies_us = _benchmark_runner(
                runner,
                num_warmup_iters=args.num_warmup_iters,
                num_iters=args.num_iters,
                num_repeats=args.num_repeats,
                use_cudagraph=args.use_cudagraph,
            )
            medians_us[version_name] = statistics.median(latencies_us)
            mins_us[version_name] = min(latencies_us)
            maxs_us[version_name] = max(latencies_us)

        baseline_us = medians_us.get(baseline_version) if baseline_version else None
        for version_name in version_names:
            median_us = medians_us[version_name]
            speedup = None if baseline_us is None else baseline_us / median_us
            bandwidth = _format_bandwidth(
                versions[version_name].bytes_per_call(cfg),
                median_us,
            )

            print(
                f"{version_name:<{version_col_width}} "
                f"{num_tokens:>8d} "
                f"{cfg.num_blocks:>8d} "
                f"{cfg.block_size:>6d} "
                f"{median_us:>12.2f} "
                f"{mins_us[version_name]:>12.2f} "
                f"{maxs_us[version_name]:>12.2f} "
                f"{bandwidth:>10.2f} "
                f"{_format_speedup(speedup):>10} "
                f"{_format_diff(diffs[version_name]):>14}"
            )


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Benchmark the Kimi-K2.5 fused MLA concat/cache kernels."
    )
    parser.add_argument(
        "--versions",
        nargs="+",
        choices=["all", *KERNEL_VERSION_BUILDERS],
        default=["all"],
        help="Kernel versions to benchmark. Use `all` to run every registered version.",
    )
    parser.add_argument(
        "--baseline-version",
        choices=list(KERNEL_VERSION_BUILDERS),
        default=None,
        help=(
            "Version to use as the speedup baseline. Defaults to `cutedsl` "
            "if selected."
        ),
    )
    parser.add_argument(
        "--num-tokens",
        type=int,
        nargs="+",
        default=[7, 64, 512, 4096],
        help="Token counts to benchmark.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=BLOCK_SIZE,
        help="KV-cache block size.",
    )
    parser.add_argument(
        "--num-warmup-iters",
        type=int,
        default=20,
        help="Warmup iterations per provider and token count.",
    )
    parser.add_argument(
        "--num-iters",
        type=int,
        default=200,
        help=(
            "Timed iterations per repeat. When --use-cudagraph is set, this is "
            "the number of kernel launches captured into each graph replay."
        ),
    )
    parser.add_argument(
        "--num-repeats",
        type=int,
        default=5,
        help="Number of timed repeats.",
    )
    parser.add_argument(
        "--num-buffers",
        type=int,
        default=8,
        help="Number of preallocated input/cache buffers to rotate through.",
    )
    parser.add_argument(
        "--check-correctness",
        action="store_true",
        help="Compare each kernel against the CUDA cache reference.",
    )
    parser.add_argument(
        "--use-cudagraph",
        action="store_true",
        help=(
            "Capture a CUDA graph containing --num-iters kernel launches and "
            "time graph replay instead of eager per-iteration dispatch."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)

    main(parser.parse_args())
