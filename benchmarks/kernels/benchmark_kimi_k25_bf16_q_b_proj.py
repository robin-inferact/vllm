# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Microbenchmark the Kimi-K2.5 BF16 q_b_proj GEMM.

This benchmark isolates the dense BF16 projection used by Kimi-K2.5's
`q_b_proj` call with the production local dimensions:

    input:   [num_tokens, 1536]
    weight:  [12288, 1536]
    output:  [num_tokens, 12288]

It compares three implementations:

1. The production FlashInfer TGV BF16 GEMM path
2. A simple Triton BF16 matmul kernel
3. A simple CuTeDSL BF16 matmul kernel

The Triton and CuTeDSL kernels are intentionally minimal scaffolding for
kernel development. They prioritize the right benchmark structure over
performance.

Example:

    .venv/bin/python benchmarks/kernels/benchmark_kimi_k25_bf16_q_b_proj.py \
        --versions production triton cutedsl --num-tokens 1 8 64 512 \
        --check-correctness

    .venv/bin/python benchmarks/kernels/benchmark_kimi_k25_bf16_q_b_proj.py \
        --versions production triton --num-tokens 64 --use-cudagraph

    .venv/bin/python benchmarks/kernels/benchmark_kimi_k25_bf16_q_b_proj.py \
        --versions triton cutedsl --num-tokens 64 \
        --dump-kernel-artifacts-dir /tmp/kimi_bf16_q_b_proj_artifacts
"""

import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.torch_utils import set_random_seed

Q_LORA_RANK = 1536
NUM_LOCAL_HEADS = 64
QK_NOPE_HEAD_DIM = 128
QK_ROPE_HEAD_DIM = 64
QK_HEAD_DIM = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM
Q_B_PROJ_OUT = NUM_LOCAL_HEADS * QK_HEAD_DIM
DTYPE = torch.bfloat16


@dataclass(frozen=True)
class BenchmarkConfig:
    num_tokens: int
    input_size: int = Q_LORA_RANK
    output_size: int = Q_B_PROJ_OUT
    dtype: torch.dtype = DTYPE
    production_backend: str = "tgv"

    @property
    def flops_per_call(self) -> int:
        return 2 * self.num_tokens * self.input_size * self.output_size


@dataclass
class SharedWeights:
    # Row-major production weight, matching torch.nn.Linear's storage.
    weight: torch.Tensor
    # Column-major [K, N] view expected by FlashInfer mm_bf16.
    weight_t: torch.Tensor


@dataclass
class QBProjInputs:
    x: torch.Tensor
    output: torch.Tensor


@dataclass
class KernelVersion:
    name: str
    make_runner: Any
    run_once: Any
    dump_artifacts: Any


_KERNEL_VERSION_CACHE: dict[str, KernelVersion] = {}


def _write_text_file(path: Path, contents: str | bytes) -> None:
    if isinstance(contents, bytes):
        contents = contents.decode("utf-8")
    path.write_text(contents, encoding="utf-8")


def _write_binary_file(path: Path, contents: bytes | bytearray | memoryview) -> None:
    path.write_bytes(bytes(contents))


def _read_text_artifact(artifact: Any) -> str:
    if isinstance(artifact, (bytes, bytearray, memoryview)):
        return bytes(artifact).decode("utf-8")
    artifact_path = Path(artifact)
    if artifact_path.exists():
        return artifact_path.read_text(encoding="utf-8")
    return str(artifact)


def _read_binary_artifact(artifact: Any) -> bytes:
    if isinstance(artifact, (bytes, bytearray, memoryview)):
        return bytes(artifact)
    artifact_path = Path(artifact)
    if artifact_path.exists():
        return artifact_path.read_bytes()
    if isinstance(artifact, str):
        return artifact.encode("utf-8")
    raise TypeError(f"Unsupported binary artifact type: {type(artifact)!r}")


def _find_single_artifact_path(output_dir: Path, suffix: str) -> Path | None:
    matches = sorted(output_dir.glob(f"*{suffix}"))
    if len(matches) == 1:
        return matches[0]
    return None


def _get_sass_from_cubin(cubin: bytes | bytearray | memoryview) -> str:
    from triton.tools.disasm import get_sass

    try:
        return get_sass(bytes(cubin))
    except Exception as exc:
        raise RuntimeError(
            "Failed to disassemble cubin into SASS. Make sure the local "
            "CUDA disassembler tools are available."
        ) from exc


def _sanitize_kernel_name(name: str) -> str:
    return name.lstrip("@").replace("/", "_")


def _make_dump_dir(
    dump_root: Path,
    version_name: str,
    cfg: BenchmarkConfig,
) -> Path:
    return (
        dump_root
        / version_name
        / (
            f"tokens_{cfg.num_tokens}_m_{cfg.num_tokens}"
            f"_n_{cfg.output_size}_k_{cfg.input_size}"
        )
    )


def _make_shared_weights(cfg: BenchmarkConfig) -> SharedWeights:
    weight = (
        torch.randn(
            cfg.output_size,
            cfg.input_size,
            device="cuda",
            dtype=cfg.dtype,
        )
        * 0.02
    ).contiguous()
    return SharedWeights(weight=weight, weight_t=weight.t())


def _make_inputs(cfg: BenchmarkConfig) -> QBProjInputs:
    x = (
        torch.randn(
            cfg.num_tokens,
            cfg.input_size,
            device="cuda",
            dtype=cfg.dtype,
        )
        * 0.3
    ).contiguous()
    output = torch.empty(
        cfg.num_tokens,
        cfg.output_size,
        device="cuda",
        dtype=cfg.dtype,
    )
    return QBProjInputs(x=x, output=output)


def _reference_q_b_proj(
    inputs: QBProjInputs,
    weights: SharedWeights,
    cfg: BenchmarkConfig,
) -> torch.Tensor:
    del cfg
    return torch.matmul(inputs.x, weights.weight_t)


def _production_q_b_proj(
    inputs: QBProjInputs,
    weights: SharedWeights,
    cfg: BenchmarkConfig,
) -> torch.Tensor:
    from flashinfer import mm_bf16

    assert inputs.x.dtype == torch.bfloat16
    assert weights.weight_t.dtype == torch.bfloat16
    assert inputs.output.dtype == torch.bfloat16
    assert inputs.x.shape == (cfg.num_tokens, cfg.input_size)
    assert weights.weight_t.shape == (cfg.input_size, cfg.output_size)
    assert inputs.output.shape == (cfg.num_tokens, cfg.output_size)

    return mm_bf16(
        inputs.x,
        weights.weight_t,
        out=inputs.output,
        out_dtype=cfg.dtype,
        backend=cfg.production_backend,
    )


@triton.jit
def _triton_q_b_proj_bf16_kernel(
    x_ptr,
    weight_ptr,
    output_ptr,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_om,
    stride_on,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in tl.range(0, K, BLOCK_K):
        k = k0 + offs_k
        a = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + k[None, :] * stride_xk,
            mask=(offs_m[:, None] < M) & (k[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            weight_ptr + offs_n[None, :] * stride_wn + k[:, None] * stride_wk,
            mask=(offs_n[None, :] < N) & (k[:, None] < K),
            other=0.0,
        )
        acc += tl.dot(a, b, out_dtype=tl.float32)

    tl.store(
        output_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        acc.to(output_ptr.dtype.element_ty),
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def _triton_launch_config(cfg: BenchmarkConfig) -> tuple[int, int, int, int]:
    del cfg
    return 16, 64, 64, 4


def triton_q_b_proj(
    inputs: QBProjInputs,
    weights: SharedWeights,
    cfg: BenchmarkConfig,
) -> torch.Tensor:
    block_m, block_n, block_k, num_warps = _triton_launch_config(cfg)
    grid = (triton.cdiv(cfg.num_tokens, block_m), triton.cdiv(cfg.output_size, block_n))
    _triton_q_b_proj_bf16_kernel[grid](
        inputs.x,
        weights.weight,
        inputs.output,
        inputs.x.stride(0),
        inputs.x.stride(1),
        weights.weight.stride(0),
        weights.weight.stride(1),
        inputs.output.stride(0),
        inputs.output.stride(1),
        cfg.num_tokens,
        cfg.output_size,
        cfg.input_size,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=3,
    )
    return inputs.output


def _get_triton_compiled_kernel(
    inputs: QBProjInputs,
    weights: SharedWeights,
    cfg: BenchmarkConfig,
):
    block_m, block_n, block_k, num_warps = _triton_launch_config(cfg)
    compiled = _triton_q_b_proj_bf16_kernel.warmup(
        inputs.x,
        weights.weight,
        inputs.output,
        inputs.x.stride(0),
        inputs.x.stride(1),
        weights.weight.stride(0),
        weights.weight.stride(1),
        inputs.output.stride(0),
        inputs.output.stride(1),
        cfg.num_tokens,
        cfg.output_size,
        cfg.input_size,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=3,
        grid=(
            triton.cdiv(cfg.num_tokens, block_m),
            triton.cdiv(cfg.output_size, block_n),
        ),
    )
    if hasattr(compiled, "result"):
        compiled = compiled.result()
    return compiled


def _dump_triton_artifacts(
    output_dir: Path,
    inputs: QBProjInputs,
    weights: SharedWeights,
    cfg: BenchmarkConfig,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    compiled = _get_triton_compiled_kernel(inputs, weights, cfg)
    kernel_name = _sanitize_kernel_name(compiled.name)
    for suffix in ("ttir", "ttgir", "llir", "ptx"):
        artifact = compiled.asm.get(suffix)
        if artifact is not None:
            _write_text_file(output_dir / f"{kernel_name}.{suffix}", artifact)
    _write_binary_file(output_dir / f"{kernel_name}.cubin", compiled.asm["cubin"])
    _write_text_file(
        output_dir / f"{kernel_name}.sass",
        _get_sass_from_cubin(compiled.asm["cubin"]),
    )


def _dump_no_artifacts(
    output_dir: Path,
    inputs: QBProjInputs,
    weights: SharedWeights,
    cfg: BenchmarkConfig,
) -> None:
    del output_dir, inputs, weights, cfg


def _make_production_runner(
    data_pool: list[QBProjInputs],
    weights: SharedWeights,
    cfg: BenchmarkConfig,
):
    index = 0

    def run() -> torch.Tensor:
        nonlocal index
        out = _production_q_b_proj(data_pool[index], weights, cfg)
        index = (index + 1) % len(data_pool)
        return out

    return run


def _make_triton_runner(
    data_pool: list[QBProjInputs],
    weights: SharedWeights,
    cfg: BenchmarkConfig,
):
    index = 0

    def run() -> torch.Tensor:
        nonlocal index
        out = triton_q_b_proj(data_pool[index], weights, cfg)
        index = (index + 1) % len(data_pool)
        return out

    return run


def _build_production_version() -> KernelVersion:
    return KernelVersion(
        name="production",
        make_runner=_make_production_runner,
        run_once=_production_q_b_proj,
        dump_artifacts=_dump_no_artifacts,
    )


def _build_triton_version() -> KernelVersion:
    return KernelVersion(
        name="triton",
        make_runner=_make_triton_runner,
        run_once=triton_q_b_proj,
        dump_artifacts=_dump_triton_artifacts,
    )


def _build_cutedsl_version() -> KernelVersion:
    try:
        import cutlass
        import cutlass.cute as cute
        import cutlass.torch as cutlass_torch
        from cuda.bindings.driver import CUstream
        from cutlass.base_dsl import compiler as cutlass_compiler
        from cutlass.cute.runtime import from_dlpack
    except ImportError as exc:  # pragma: no cover - benchmark-only path
        raise RuntimeError(
            "CuTeDSL benchmarking requires the `cutlass` Python package."
        ) from exc

    @cute.kernel
    def kimik25_bf16_q_b_proj_kernel(
        x: cute.Tensor,  # (M, K)
        weight: cute.Tensor,  # (N, K)
        output: cute.Tensor,  # (M, N)
        input_size: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        row, col_block, _ = cute.arch.block_idx()
        col = col_block * 32 + tidx

        if row < output.shape[0] and col < output.shape[1]:
            acc: cutlass.Float32 = 0.0
            for k in cutlass.range_constexpr(input_size):
                a = x[row, k].to(cutlass.Float32)
                b = weight[col, k].to(cutlass.Float32)
                acc += a * b
            output[row, col] = acc.to(cutlass.BFloat16)

    @cute.jit
    def kimik25_bf16_q_b_proj(
        x: cute.Tensor,
        weight: cute.Tensor,
        output: cute.Tensor,
        input_size: cutlass.Constexpr,
        stream: CUstream,
    ):
        grid = (x.shape[0], cute.ceil_div(output.shape[1], 32), 1)
        block = (32, 1, 1)
        kimik25_bf16_q_b_proj_kernel(
            x,
            weight,
            output,
            input_size,
        ).launch(grid=grid, block=block, stream=stream)

    compiled_function_cache: dict[tuple[Any, ...], Any] = {}
    compiled_executor_cache: dict[tuple[Any, ...], Any] = {}

    def _make_dynamic_cute_tensor(data: torch.Tensor):
        return from_dlpack(data, assumed_align=16).mark_layout_dynamic(
            leading_dim=cutlass_torch.get_leading_dim(data)
        )

    def _cutedsl_arg_cache_key(arg: Any) -> Any:
        cache_key = getattr(arg, "__cache_key__", None)
        return arg if cache_key is None else cache_key

    def _make_cute_args(
        inputs: QBProjInputs,
        weights: SharedWeights,
    ) -> tuple[Any, Any, Any]:
        return (
            _make_dynamic_cute_tensor(inputs.x),
            _make_dynamic_cute_tensor(weights.weight),
            _make_dynamic_cute_tensor(inputs.output),
        )

    def _get_cutedsl_cache_key(
        inputs: QBProjInputs,
        weights: SharedWeights,
        cfg: BenchmarkConfig,
        dump_dir: Path | None,
    ) -> tuple[Any, ...]:
        x_cute, weight_cute, output_cute = _make_cute_args(inputs, weights)
        return (
            torch.cuda.current_device(),
            _cutedsl_arg_cache_key(x_cute),
            _cutedsl_arg_cache_key(weight_cute),
            _cutedsl_arg_cache_key(output_cute),
            cfg.input_size,
            str(dump_dir) if dump_dir is not None else None,
        )

    def _get_cutedsl_compiled_function(
        inputs: QBProjInputs,
        weights: SharedWeights,
        cfg: BenchmarkConfig,
        *,
        dump_dir: Path | None = None,
    ):
        cache_key = _get_cutedsl_cache_key(inputs, weights, cfg, dump_dir)
        compiled = compiled_function_cache.get(cache_key)
        if compiled is None:
            x_cute, weight_cute, output_cute = _make_cute_args(inputs, weights)
            compile_callable = cute.compile
            if dump_dir is not None:
                dump_dir.mkdir(parents=True, exist_ok=True)
                compile_callable = cute.compile[(
                    cutlass_compiler.KeepPTX(True),
                    cutlass_compiler.KeepCUBIN(True),
                    cutlass_compiler.DumpDir(str(dump_dir)),
                )]
            compiled = compile_callable(
                kimik25_bf16_q_b_proj,
                x=x_cute,
                weight=weight_cute,
                output=output_cute,
                input_size=cfg.input_size,
                stream=cutlass_torch.current_stream(),
            )
            compiled_function_cache[cache_key] = compiled
        return compiled

    def _get_cutedsl_executor(
        inputs: QBProjInputs,
        weights: SharedWeights,
        cfg: BenchmarkConfig,
    ):
        cache_key = _get_cutedsl_cache_key(inputs, weights, cfg, dump_dir=None)
        executor = compiled_executor_cache.get(cache_key)
        if executor is None:
            executor = _get_cutedsl_compiled_function(
                inputs,
                weights,
                cfg,
                dump_dir=None,
            ).to(None)
            compiled_executor_cache[cache_key] = executor
        return executor

    def dump_cutedsl_artifacts(
        output_dir: Path,
        inputs: QBProjInputs,
        weights: SharedWeights,
        cfg: BenchmarkConfig,
    ) -> None:
        compiled = _get_cutedsl_compiled_function(
            inputs,
            weights,
            cfg,
            dump_dir=output_dir,
        )
        if compiled.__ptx__ is None or compiled.__cubin__ is None:
            raise RuntimeError(
                "CuTeDSL compilation did not produce PTX/CUBIN dump paths."
            )

        kernel_name = _sanitize_kernel_name(compiled.function_name)
        ptx_artifact = compiled.__ptx__
        cubin_artifact = compiled.__cubin__

        ptx_path = (
            Path(ptx_artifact)
            if isinstance(ptx_artifact, str) and Path(ptx_artifact).exists()
            else _find_single_artifact_path(output_dir, ".ptx")
            or output_dir / f"{kernel_name}.ptx"
        )
        cubin_path = (
            Path(cubin_artifact)
            if isinstance(cubin_artifact, str) and Path(cubin_artifact).exists()
            else _find_single_artifact_path(output_dir, ".cubin")
            or output_dir / f"{kernel_name}.cubin"
        )
        if not ptx_path.exists():
            _write_text_file(ptx_path, _read_text_artifact(ptx_artifact))
        if not cubin_path.exists():
            _write_binary_file(cubin_path, _read_binary_artifact(cubin_artifact))

        _write_text_file(
            cubin_path.with_suffix(".sass"),
            _get_sass_from_cubin(cubin_path.read_bytes()),
        )

    def cutedsl_q_b_proj(
        inputs: QBProjInputs,
        weights: SharedWeights,
        cfg: BenchmarkConfig,
    ) -> torch.Tensor:
        executor = _get_cutedsl_executor(inputs, weights, cfg)
        x_cute, weight_cute, output_cute = _make_cute_args(inputs, weights)
        executor(
            x=x_cute,
            weight=weight_cute,
            output=output_cute,
            stream=cutlass_torch.current_stream(),
        )
        return inputs.output

    def _make_cutedsl_runner(
        data_pool: list[QBProjInputs],
        weights: SharedWeights,
        cfg: BenchmarkConfig,
    ):
        index = 0

        def run() -> torch.Tensor:
            nonlocal index
            out = cutedsl_q_b_proj(data_pool[index], weights, cfg)
            index = (index + 1) % len(data_pool)
            return out

        return run

    return KernelVersion(
        name="cutedsl",
        make_runner=_make_cutedsl_runner,
        run_once=cutedsl_q_b_proj,
        dump_artifacts=dump_cutedsl_artifacts,
    )


KERNEL_VERSION_BUILDERS = {
    "production": _build_production_version,
    "triton": _build_triton_version,
    "cutedsl": _build_cutedsl_version,
}


def _get_kernel_version(name: str) -> KernelVersion:
    version = _KERNEL_VERSION_CACHE.get(name)
    if version is None:
        version = KERNEL_VERSION_BUILDERS[name]()
        _KERNEL_VERSION_CACHE[name] = version
    return version


def _benchmark_once(
    runner,
    *,
    use_cudagraph: bool,
    num_warmup_iters: int,
    num_iters: int,
) -> list[float]:
    if num_iters < 1:
        raise ValueError("--num-iters must be at least 1.")

    if use_cudagraph:
        for _ in range(max(num_warmup_iters, 1)):
            runner()
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(num_iters):
                runner()
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        return [start.elapsed_time(end) / num_iters / 1000.0]

    timings_s = []
    for _ in range(num_warmup_iters):
        runner()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(num_iters):
        start.record()
        runner()
        end.record()
        torch.cuda.synchronize()
        timings_s.append(start.elapsed_time(end) / 1000.0)
    return timings_s


def _check_correctness(
    version: KernelVersion,
    inputs: QBProjInputs,
    weights: SharedWeights,
    cfg: BenchmarkConfig,
) -> float:
    ref = _reference_q_b_proj(inputs, weights, cfg)
    actual = version.run_once(inputs, weights, cfg)
    torch.cuda.synchronize()
    return (actual.float() - ref.float()).abs().max().item()


def _dump_version_artifacts(
    version: KernelVersion,
    dump_root: Path,
    inputs: QBProjInputs,
    weights: SharedWeights,
    cfg: BenchmarkConfig,
) -> None:
    dump_dir = _make_dump_dir(dump_root, version.name, cfg)
    dump_dir.mkdir(parents=True, exist_ok=True)
    version.dump_artifacts(dump_dir, inputs, weights, cfg)


def _parse_args() -> Any:
    parser = FlexibleArgumentParser(
        description="Benchmark the Kimi-K2.5 BF16 q_b_proj GEMM kernel."
    )
    parser.add_argument(
        "--versions",
        nargs="+",
        default=["production", "triton", "cutedsl"],
        choices=list(KERNEL_VERSION_BUILDERS),
        help="Kernel versions to benchmark.",
    )
    parser.add_argument(
        "--num-tokens",
        nargs="+",
        type=int,
        default=[1, 8, 64, 512],
        help="Token counts to benchmark.",
    )
    parser.add_argument(
        "--production-backend",
        choices=["cutlass", "cudnn", "tgv", "auto"],
        default="tgv",
        help="FlashInfer mm_bf16 backend used by the production version.",
    )
    parser.add_argument(
        "--num-warmup-iters",
        type=int,
        default=10,
        help="Warmup iterations per repeat.",
    )
    parser.add_argument(
        "--num-iters",
        type=int,
        default=50,
        help="Timed iterations per repeat.",
    )
    parser.add_argument(
        "--num-repeats",
        type=int,
        default=5,
        help="How many benchmark repeats to run.",
    )
    parser.add_argument(
        "--num-buffers",
        type=int,
        default=2,
        help="How many input/output buffers to rotate through.",
    )
    parser.add_argument(
        "--use-cudagraph",
        action="store_true",
        help="Benchmark with Triton's CUDA-graph helper.",
    )
    parser.add_argument(
        "--check-correctness",
        action="store_true",
        help="Compare each kernel result against torch.matmul.",
    )
    parser.add_argument(
        "--dump-kernel-artifacts-dir",
        type=Path,
        default=None,
        help=(
            "If set, compile one sample for each version/configuration and dump "
            "PTX, cubin, and SASS artifacts under this directory."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed.",
    )
    return parser.parse_args()


def main() -> None:
    if not current_platform.is_cuda():
        raise RuntimeError("This benchmark requires CUDA.")
    args = _parse_args()
    if not current_platform.is_device_capability_family(100):
        print(
            "Warning: this benchmark is primarily intended for Blackwell GPUs "
            "(SM10x)."
        )

    set_random_seed(args.seed)
    versions = [_get_kernel_version(version_name) for version_name in args.versions]

    print(f"production_backend: flashinfer.mm_bf16/{args.production_backend}")
    print(
        f"{'tokens':>8}  {'version':<12}  {'median_us':>10}  {'tflops':>8}  "
        f"{'max_abs_diff':>12}"
    )

    for num_tokens in args.num_tokens:
        cfg = BenchmarkConfig(
            num_tokens=num_tokens,
            production_backend=args.production_backend,
        )
        weights = _make_shared_weights(cfg)
        data_pool = [_make_inputs(cfg) for _ in range(args.num_buffers)]

        for version in versions:
            if args.dump_kernel_artifacts_dir is not None:
                _dump_version_artifacts(
                    version,
                    args.dump_kernel_artifacts_dir,
                    data_pool[0],
                    weights,
                    cfg,
                )

            repeat_medians = []
            for _ in range(args.num_repeats):
                runner = version.make_runner(data_pool, weights, cfg)
                timings_s = _benchmark_once(
                    runner,
                    use_cudagraph=args.use_cudagraph,
                    num_warmup_iters=args.num_warmup_iters,
                    num_iters=args.num_iters,
                )
                repeat_medians.append(statistics.median(timings_s))

            median_s = statistics.median(repeat_medians)
            tflops = cfg.flops_per_call / median_s / 1e12
            max_abs_diff = float("nan")
            if args.check_correctness:
                max_abs_diff = _check_correctness(
                    version,
                    data_pool[0],
                    weights,
                    cfg,
                )

            diff_str = f"{max_abs_diff:.3e}" if args.check_correctness else "n/a"
            print(
                f"{num_tokens:8d}  {version.name:<12}  "
                f"{median_s * 1e6:10.2f}  {tflops:8.2f}  {diff_str:>12}"
            )


if __name__ == "__main__":
    main()
