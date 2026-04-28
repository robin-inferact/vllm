# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Microbenchmark the Kimi-K2.5 decode RoPE + FP8 quant kernel.

This benchmark isolates the decode-only query path used by the specialized
Kimi-K2.5 NVFP4 MLA attention implementation:

    ql_nope: [num_tokens, num_local_heads, q_lora_dim]
    q_pe:    [num_tokens, num_local_heads, qk_rope_head_dim]
    q_out:   [num_tokens, num_local_heads, q_lora_dim + qk_rope_head_dim]

The fused CuTeDSL kernel rotates `q_pe`, concatenates it with `ql_nope`, and
writes statically-scaled FP8 query bytes for the downstream MLA decode kernel.

Example:

    .venv/bin/python \
        benchmarks/kernels/benchmark_kimi_k25_nvfp4_decode_rope_concat_quant_fp8.py \
        --versions cutedsl unfused --num-tokens 7 64 512 4096 \
        --check-correctness

    .venv/bin/python \
        benchmarks/kernels/benchmark_kimi_k25_nvfp4_decode_rope_concat_quant_fp8.py \
        --versions cutedsl --num-tokens 4096 --use-cudagraph
"""

import statistics
from dataclasses import dataclass
from typing import Any

import torch

from vllm import _custom_ops as ops
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.platforms import current_platform
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.torch_utils import set_random_seed

Q_LORA_DIM = 512
QK_NOPE_HEAD_DIM = 128
QK_ROPE_HEAD_DIM = 64
NUM_LOCAL_HEADS = 64
MAX_POSITION = 262144
DTYPE = torch.bfloat16
DEFAULT_SCALE = 0.02
ROPE_PARAMETERS = {
    "rope_type": "deepseek_yarn",
    "rope_theta": 50000.0,
    "factor": 64.0,
    "beta_fast": 32.0,
    "beta_slow": 1.0,
    "mscale": 1.0,
    "mscale_all_dim": 1.0,
    "original_max_position_embeddings": 4096,
}


@dataclass(frozen=True)
class BenchmarkConfig:
    num_tokens: int
    num_local_heads: int = NUM_LOCAL_HEADS
    q_lora_dim: int = Q_LORA_DIM
    qk_nope_head_dim: int = QK_NOPE_HEAD_DIM
    qk_rope_head_dim: int = QK_ROPE_HEAD_DIM
    max_position_embeddings: int = MAX_POSITION
    scale: float = DEFAULT_SCALE
    dtype: torch.dtype = DTYPE

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def output_dim(self) -> int:
        return self.q_lora_dim + self.qk_rope_head_dim

    @property
    def half_rope_dim(self) -> int:
        return self.qk_rope_head_dim // 2

    @property
    def fused_bytes_per_call(self) -> int:
        elem_size = torch.tensor([], dtype=self.dtype).element_size()
        long_size = torch.tensor([], dtype=torch.long).element_size()
        tokens_heads = self.num_tokens * self.num_local_heads
        q_in_elems = tokens_heads * self.output_dim
        cache_elems = tokens_heads * self.qk_rope_head_dim
        return (
            q_in_elems * elem_size
            + cache_elems * elem_size
            + tokens_heads * self.output_dim
            + tokens_heads * (long_size + 4)
        )

    @property
    def unfused_bytes_per_call(self) -> int:
        elem_size = torch.tensor([], dtype=self.dtype).element_size()
        tokens_heads = self.num_tokens * self.num_local_heads
        # RoPE read/write, workspace concat write, quant read, and FP8 write.
        rope_elems = tokens_heads * self.qk_rope_head_dim
        concat_elems = tokens_heads * self.output_dim
        return (2 * rope_elems + 2 * concat_elems) * elem_size + concat_elems


@dataclass
class DecodeInputs:
    positions: torch.Tensor
    ql_nope: torch.Tensor
    q_pe: torch.Tensor
    q_out: torch.Tensor
    workspace: torch.Tensor
    cos_sin_cache: torch.Tensor
    scale: torch.Tensor


@dataclass
class KernelVersion:
    name: str
    make_runner: Any
    run_once: Any
    bytes_per_call: Any


_KERNEL_VERSION_CACHE: dict[str, KernelVersion] = {}


def _make_rope_module(cfg: BenchmarkConfig):
    with set_current_vllm_config(VllmConfig()):
        rope = get_rope(
            head_size=cfg.qk_rope_head_dim,
            max_position=cfg.max_position_embeddings,
            is_neox_style=False,
            rope_parameters=ROPE_PARAMETERS.copy(),
            dtype=cfg.dtype,
        )
    return rope.to(device="cuda", dtype=cfg.dtype)


def _make_mla_like_ql_nope_view(cfg: BenchmarkConfig) -> torch.Tensor:
    # This matches the BMM output after (N, B, L).transpose(0, 1).
    return torch.randn(
        cfg.num_local_heads,
        cfg.num_tokens,
        cfg.q_lora_dim,
        device="cuda",
        dtype=cfg.dtype,
    ).transpose(0, 1)


def _make_mla_like_q_pe_view(cfg: BenchmarkConfig) -> torch.Tensor:
    query_base = torch.randn(
        cfg.num_tokens,
        cfg.num_local_heads,
        cfg.qk_head_dim,
        device="cuda",
        dtype=cfg.dtype,
    )
    return query_base[..., cfg.qk_nope_head_dim :]


def _make_inputs(
    cfg: BenchmarkConfig,
    cos_sin_cache: torch.Tensor,
    *,
    positions: torch.Tensor | None = None,
    scale: torch.Tensor | None = None,
) -> DecodeInputs:
    if positions is None:
        positions = torch.randint(
            0,
            cfg.max_position_embeddings,
            (cfg.num_tokens,),
            device="cuda",
            dtype=torch.long,
        )
    if scale is None:
        scale = torch.tensor([cfg.scale], device="cuda", dtype=torch.float32)

    return DecodeInputs(
        positions=positions,
        ql_nope=_make_mla_like_ql_nope_view(cfg),
        q_pe=_make_mla_like_q_pe_view(cfg),
        q_out=torch.empty(
            cfg.num_tokens,
            cfg.num_local_heads,
            cfg.output_dim,
            device="cuda",
            dtype=torch.uint8,
        ),
        workspace=torch.empty(
            cfg.num_tokens,
            cfg.num_local_heads,
            cfg.output_dim,
            device="cuda",
            dtype=cfg.dtype,
        ),
        cos_sin_cache=cos_sin_cache,
        scale=scale,
    )


def _fp8_output(inputs: DecodeInputs) -> torch.Tensor:
    return inputs.q_out.view(current_platform.fp8_dtype())


def _run_unfused(inputs: DecodeInputs, cfg: BenchmarkConfig) -> None:
    from vllm.model_executor.specialized_models.kimi_k2_5_nvfp4.model import (
        _run_kimik25_rope,
    )

    _run_kimik25_rope(
        inputs.positions,
        inputs.q_pe,
        inputs.cos_sin_cache,
        cfg.num_local_heads,
        cfg.half_rope_dim,
    )
    inputs.workspace[..., : cfg.q_lora_dim].copy_(inputs.ql_nope)
    inputs.workspace[..., cfg.q_lora_dim :].copy_(inputs.q_pe)
    ops.scaled_fp8_quant(
        inputs.workspace.reshape(cfg.num_tokens, -1),
        inputs.scale,
        output=_fp8_output(inputs).reshape(cfg.num_tokens, -1),
    )


def _make_unfused_runner(
    inputs_pool: list[DecodeInputs],
    cfg: BenchmarkConfig,
):
    index = 0

    def run() -> None:
        nonlocal index
        _run_unfused(inputs_pool[index], cfg)
        index = (index + 1) % len(inputs_pool)

    return run


def _build_unfused_version() -> KernelVersion:
    return KernelVersion(
        name="unfused",
        make_runner=_make_unfused_runner,
        run_once=_run_unfused,
        bytes_per_call=lambda cfg: cfg.unfused_bytes_per_call,
    )


def _build_cutedsl_version() -> KernelVersion:
    import cutlass.torch as cutlass_torch
    from cutlass.cute.runtime import from_dlpack

    from vllm.model_executor.specialized_models.kimi_k2_5_nvfp4.model import (
        _cuda_device_cache_key,
        _cutedsl_arg_cache_key,
        _get_cutedsl_executor,
        _make_dynamic_cute_tensor,
        _make_fully_dynamic_cute_tensor,
        kimik25_decode_rope_concat_quant_fp8,
    )

    def _validate_cfg(cfg: BenchmarkConfig) -> None:
        if cfg.q_lora_dim != Q_LORA_DIM:
            raise ValueError(
                f"cutedsl requires q_lora_dim={Q_LORA_DIM}, got {cfg.q_lora_dim}."
            )
        if cfg.qk_rope_head_dim != QK_ROPE_HEAD_DIM:
            raise ValueError(
                "cutedsl requires "
                f"qk_rope_head_dim={QK_ROPE_HEAD_DIM}, got {cfg.qk_rope_head_dim}."
            )

    def _get_executor(inputs: DecodeInputs, cfg: BenchmarkConfig):
        positions_cute = _make_fully_dynamic_cute_tensor(inputs.positions)
        ql_nope_cute = _make_dynamic_cute_tensor(inputs.ql_nope)
        q_pe_cute = _make_fully_dynamic_cute_tensor(inputs.q_pe)
        q_out_cute = _make_dynamic_cute_tensor(inputs.q_out)
        cos_sin_cache_cute = from_dlpack(inputs.cos_sin_cache, assumed_align=16)
        scale_cute = from_dlpack(inputs.scale, assumed_align=4)
        cache_key = (
            "kimik25_decode_rope_concat_quant_fp8",
            _cuda_device_cache_key(),
            _cutedsl_arg_cache_key(positions_cute),
            _cutedsl_arg_cache_key(ql_nope_cute),
            _cutedsl_arg_cache_key(q_pe_cute),
            _cutedsl_arg_cache_key(q_out_cute),
            _cutedsl_arg_cache_key(cos_sin_cache_cute),
            _cutedsl_arg_cache_key(scale_cute),
            cfg.q_lora_dim,
            cfg.qk_rope_head_dim,
        )
        return _get_cutedsl_executor(
            cache_key,
            kimik25_decode_rope_concat_quant_fp8,
            positions=positions_cute,
            ql_nope=ql_nope_cute,
            q_pe=q_pe_cute,
            q_out=q_out_cute,
            cos_sin_cache=cos_sin_cache_cute,
            scale=scale_cute,
            q_lora_dim=cfg.q_lora_dim,
            pe_dim=cfg.qk_rope_head_dim,
            stream=cutlass_torch.current_stream(),
        )

    def cutedsl_decode_rope_concat_quant_fp8(
        inputs: DecodeInputs,
        cfg: BenchmarkConfig,
    ) -> None:
        _validate_cfg(cfg)
        executor = _get_executor(inputs, cfg)
        executor(
            positions=_make_fully_dynamic_cute_tensor(inputs.positions),
            ql_nope=_make_dynamic_cute_tensor(inputs.ql_nope),
            q_pe=_make_fully_dynamic_cute_tensor(inputs.q_pe),
            q_out=_make_dynamic_cute_tensor(inputs.q_out),
            cos_sin_cache=from_dlpack(inputs.cos_sin_cache),
            scale=from_dlpack(inputs.scale),
            stream=cutlass_torch.current_stream(),
        )

    def make_cutedsl_runner(
        inputs_pool: list[DecodeInputs],
        cfg: BenchmarkConfig,
    ):
        _validate_cfg(cfg)
        cute_positions = _make_fully_dynamic_cute_tensor(inputs_pool[0].positions)
        cute_ql_nope_pool = [
            _make_dynamic_cute_tensor(inputs.ql_nope) for inputs in inputs_pool
        ]
        cute_q_pe_pool = [
            _make_fully_dynamic_cute_tensor(inputs.q_pe) for inputs in inputs_pool
        ]
        cute_q_out_pool = [
            _make_dynamic_cute_tensor(inputs.q_out) for inputs in inputs_pool
        ]
        cute_cos_sin_cache = from_dlpack(
            inputs_pool[0].cos_sin_cache,
            assumed_align=16,
        )
        cute_scale = from_dlpack(inputs_pool[0].scale, assumed_align=4)
        executor = _get_executor(inputs_pool[0], cfg)
        index = 0

        def run() -> None:
            nonlocal index
            executor(
                positions=cute_positions,
                ql_nope=cute_ql_nope_pool[index],
                q_pe=cute_q_pe_pool[index],
                q_out=cute_q_out_pool[index],
                cos_sin_cache=cute_cos_sin_cache,
                scale=cute_scale,
                stream=cutlass_torch.current_stream(),
            )
            index = (index + 1) % len(cute_ql_nope_pool)

        return run

    return KernelVersion(
        name="cutedsl",
        make_runner=make_cutedsl_runner,
        run_once=cutedsl_decode_rope_concat_quant_fp8,
        bytes_per_call=lambda cfg: cfg.fused_bytes_per_call,
    )


KERNEL_VERSION_BUILDERS = {
    "cutedsl": _build_cutedsl_version,
    "unfused": _build_unfused_version,
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


def _clone_inputs_for_correctness(
    reference: DecodeInputs,
    cfg: BenchmarkConfig,
) -> DecodeInputs:
    actual = _make_inputs(
        cfg,
        reference.cos_sin_cache,
        positions=reference.positions,
        scale=reference.scale,
    )
    actual.ql_nope.copy_(reference.ql_nope)
    actual.q_pe.copy_(reference.q_pe)
    return actual


def _run_correctness(
    version: KernelVersion,
    cfg: BenchmarkConfig,
    rope,
) -> float:
    reference_inputs = _make_inputs(cfg, rope.cos_sin_cache)
    expected_inputs = _clone_inputs_for_correctness(reference_inputs, cfg)
    actual_inputs = _clone_inputs_for_correctness(reference_inputs, cfg)

    _run_unfused(expected_inputs, cfg)
    version.run_once(actual_inputs, cfg)
    torch.cuda.synchronize()

    return (
        _fp8_output(actual_inputs).float() - _fp8_output(expected_inputs).float()
    ).abs().max().item()


def main(args) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    if not current_platform.is_device_capability_family(100):
        print(
            "Warning: this benchmark is primarily intended for Blackwell GPUs "
            "(SM10x)."
        )

    if args.qk_rope_head_dim % 2 != 0:
        raise ValueError("--qk-rope-head-dim must be even.")
    if args.num_local_heads <= 0:
        raise ValueError("--num-local-heads must be positive.")
    if args.num_buffers <= 0:
        raise ValueError("--num-buffers must be positive.")

    set_random_seed(args.seed)
    if args.versions is not None:
        version_names = _resolve_version_names(args.versions)
    elif args.provider is not None:
        version_names = (
            ["cutedsl", "unfused"] if args.provider == "both" else [args.provider]
        )
    else:
        version_names = _resolve_version_names(["all"])
    baseline_version = _resolve_baseline_version(
        version_names,
        args.baseline_version,
    )
    versions = {name: _get_kernel_version(name) for name in version_names}

    if baseline_version is not None:
        print(f"Baseline version: {baseline_version}")

    version_col_width = max(len("version"), max(len(name) for name in version_names))
    header = (
        f"{'version':<{version_col_width}} "
        f"{'tokens':>8} {'heads':>8} {'ql_s0':>8} {'ql_s1':>8} "
        f"{'qpe_s0':>8} {'qpe_s1':>8} "
        f"{'median_us':>12} {'min_us':>12} {'max_us':>12} {'gbps':>10} "
        f"{'speedup':>10} {'max_abs_diff':>14}"
    )
    print(header)
    print("-" * len(header))

    for num_tokens in args.num_tokens:
        cfg = BenchmarkConfig(
            num_tokens=num_tokens,
            num_local_heads=args.num_local_heads,
            q_lora_dim=args.q_lora_dim,
            qk_nope_head_dim=args.qk_nope_head_dim,
            qk_rope_head_dim=args.qk_rope_head_dim,
            max_position_embeddings=args.max_position_embeddings,
            scale=args.scale,
        )
        rope = _make_rope_module(cfg)
        sample_inputs = _make_inputs(cfg, rope.cos_sin_cache)

        diffs: dict[str, float | None] = {}
        if args.check_correctness:
            for version_name in version_names:
                diffs[version_name] = _run_correctness(
                    versions[version_name],
                    cfg,
                    rope,
                )
        else:
            diffs = {version_name: None for version_name in version_names}

        medians_us: dict[str, float] = {}
        mins_us: dict[str, float] = {}
        maxs_us: dict[str, float] = {}

        positions = sample_inputs.positions
        scale = sample_inputs.scale
        for version_name in version_names:
            inputs_pool = [
                _make_inputs(
                    cfg,
                    rope.cos_sin_cache,
                    positions=positions,
                    scale=scale,
                )
                for _ in range(args.num_buffers)
            ]
            runner = versions[version_name].make_runner(inputs_pool, cfg)
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
        ql_stride0 = sample_inputs.ql_nope.stride(0)
        ql_stride1 = sample_inputs.ql_nope.stride(1)
        qpe_stride0 = sample_inputs.q_pe.stride(0)
        qpe_stride1 = sample_inputs.q_pe.stride(1)
        for version_name in version_names:
            median_us = medians_us[version_name]
            speedup = None if baseline_us is None else baseline_us / median_us
            bytes_per_call = versions[version_name].bytes_per_call(cfg)

            print(
                f"{version_name:<{version_col_width}} "
                f"{num_tokens:>8d} "
                f"{cfg.num_local_heads:>8d} "
                f"{ql_stride0:>8d} "
                f"{ql_stride1:>8d} "
                f"{qpe_stride0:>8d} "
                f"{qpe_stride1:>8d} "
                f"{median_us:>12.2f} "
                f"{mins_us[version_name]:>12.2f} "
                f"{maxs_us[version_name]:>12.2f} "
                f"{_format_bandwidth(bytes_per_call, median_us):>10.2f} "
                f"{_format_speedup(speedup):>10} "
                f"{_format_diff(diffs[version_name]):>14}"
            )


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Benchmark the Kimi-K2.5 decode RoPE + FP8 quant kernels."
    )
    parser.add_argument(
        "--versions",
        nargs="+",
        choices=["all", *KERNEL_VERSION_BUILDERS],
        default=None,
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
        "--provider",
        choices=["cutedsl", "unfused", "both"],
        default=None,
        help="Deprecated compatibility alias for the old two-version interface.",
    )
    parser.add_argument(
        "--num-tokens",
        type=int,
        nargs="+",
        default=[7, 64, 512, 4096],
        help="Token counts to benchmark.",
    )
    parser.add_argument(
        "--num-local-heads",
        type=int,
        default=NUM_LOCAL_HEADS,
        help="Number of local query heads in the decode query view.",
    )
    parser.add_argument(
        "--q-lora-dim",
        type=int,
        default=Q_LORA_DIM,
        help="Absorbed query LoRA dimension. The CuTeDSL kernel expects 512.",
    )
    parser.add_argument(
        "--qk-nope-head-dim",
        type=int,
        default=QK_NOPE_HEAD_DIM,
        help="Non-RoPE per-head dimension used to define the q_pe view stride.",
    )
    parser.add_argument(
        "--qk-rope-head-dim",
        type=int,
        default=QK_ROPE_HEAD_DIM,
        help="RoPE per-head dimension. The CuTeDSL kernel expects 64.",
    )
    parser.add_argument(
        "--max-position-embeddings",
        type=int,
        default=MAX_POSITION,
        help="Maximum position count used to build the cos/sin cache.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=DEFAULT_SCALE,
        help="Static FP8 query scale.",
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
        help="Number of preallocated buffers to rotate through during timing.",
    )
    parser.add_argument(
        "--check-correctness",
        action="store_true",
        help="Compare each kernel against the unfused RoPE + FP8 quant reference.",
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
