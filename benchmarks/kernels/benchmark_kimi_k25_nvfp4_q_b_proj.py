# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Microbenchmark the Kimi-K2.5 NVFP4 q_b_proj GEMM kernel.

This benchmark isolates the quantized GEMM used by Kimi-K2.5's `q_b_proj`
projection with the production local dimensions:

    input:   [num_tokens, 1536]
    weight:  [12288, 1536]
    output:  [num_tokens, 12288]

The benchmark pre-quantizes activations and weights into the same NVFP4 packed
byte format used by the production kernel:

    fp4_data:         uint8 tensor with two e2m1 values packed per byte
    block_scales:     CUTLASS/FlashInfer swizzled scale layout
    alpha:            global scale correction term

It compares three implementations:

1. The production NVFP4 GEMM backend selected by vLLM
2. A new correctness-first Triton kernel
3. A new correctness-first CuTeDSL kernel

Example:

    .venv/bin/python benchmarks/kernels/benchmark_kimi_k25_nvfp4_q_b_proj.py \
        --versions production triton cutedsl --num-tokens 1 8 64 512 \
        --check-correctness

    .venv/bin/python benchmarks/kernels/benchmark_kimi_k25_nvfp4_q_b_proj.py \
        --versions production triton cutedsl --num-tokens 64 --use-cudagraph

    .venv/bin/python benchmarks/kernels/benchmark_kimi_k25_nvfp4_q_b_proj.py \
        --versions triton cutedsl --num-tokens 64 \
        --dump-kernel-artifacts-dir /tmp/kimi_q_b_proj_artifacts
"""

import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    dequantize_to_dtype,
    kE2M1ToFloat_handle,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
    NvFp4LinearBackend,
    select_nvfp4_linear_backend,
)
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types
from vllm.triton_utils import tl, triton
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.flashinfer import flashinfer_scaled_fp4_mm
from vllm.utils.torch_utils import set_random_seed

Q_LORA_RANK = 1536
NUM_LOCAL_HEADS = 64
QK_NOPE_HEAD_DIM = 128
QK_ROPE_HEAD_DIM = 64
QK_HEAD_DIM = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM
Q_B_PROJ_OUT = NUM_LOCAL_HEADS * QK_HEAD_DIM
GROUP_SIZE = 16
PACKED_GROUP_BYTES = GROUP_SIZE // 2
DTYPE = torch.bfloat16
SCALE_TILE_ROWS = 128
SCALE_TILE_COLS = 4
FP4_ABSMAX = scalar_types.float4_e2m1f.max()
FP8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max


@dataclass(frozen=True)
class BenchmarkConfig:
    num_tokens: int
    input_size: int = Q_LORA_RANK
    output_size: int = Q_B_PROJ_OUT
    group_size: int = GROUP_SIZE
    dtype: torch.dtype = DTYPE

    @property
    def packed_k(self) -> int:
        return self.input_size // 2

    @property
    def num_groups(self) -> int:
        return self.input_size // self.group_size

    @property
    def rounded_num_groups(self) -> int:
        return triton.cdiv(self.num_groups, SCALE_TILE_COLS) * SCALE_TILE_COLS

    @property
    def num_scale_k_tiles(self) -> int:
        return self.rounded_num_groups // SCALE_TILE_COLS

    @property
    def flops_per_call(self) -> int:
        return 2 * self.num_tokens * self.input_size * self.output_size


@dataclass
class SharedWeights:
    weight_fp4: torch.Tensor
    weight_scale: torch.Tensor
    weight_scale_f32: torch.Tensor
    weight_global_scale_inv: torch.Tensor


@dataclass
class QBProjInputs:
    input_fp4: torch.Tensor
    input_scale: torch.Tensor
    input_scale_f32: torch.Tensor
    input_global_scale_inv: torch.Tensor
    alpha: torch.Tensor


@dataclass
class KernelVersion:
    name: str
    make_runner: Any
    run_once: Any
    dump_artifacts: Any


_KERNEL_VERSION_CACHE: dict[str, KernelVersion] = {}


def _fp4_lut(device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ],
        device=device,
        dtype=torch.float32,
    )


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
        / f"tokens_{cfg.num_tokens}_m_{cfg.num_tokens}_n_{cfg.output_size}_k_{cfg.input_size}"
    )


def _selected_production_backend() -> NvFp4LinearBackend:
    backend = select_nvfp4_linear_backend()
    if backend not in (
        NvFp4LinearBackend.FLASHINFER_CUTLASS,
        NvFp4LinearBackend.FLASHINFER_CUDNN,
        NvFp4LinearBackend.VLLM_CUTLASS,
    ):
        raise RuntimeError(
            "This benchmark currently supports the CUTLASS-style NVFP4 linear "
            "backends used by Kimi q_b_proj. "
            f"Selected backend was {backend.value!r}."
        )
    return backend


def _global_scale_inv_from_tensor(x: torch.Tensor) -> torch.Tensor:
    amax = torch.clamp(torch.abs(x).max().to(torch.float32), min=1e-6)
    return (FP8_E4M3_MAX * FP4_ABSMAX / amax).reshape(1).to(device=x.device)


def _make_shared_weights(
    cfg: BenchmarkConfig,
    backend: NvFp4LinearBackend,
) -> SharedWeights:
    weight_dense = torch.randn(
        cfg.output_size,
        cfg.input_size,
        device="cuda",
        dtype=cfg.dtype,
    )
    weight_global_scale_inv = _global_scale_inv_from_tensor(weight_dense)
    weight_fp4, weight_scale = ops.scaled_fp4_quant(
        weight_dense,
        weight_global_scale_inv,
        is_sf_swizzled_layout=True,
        backend=backend.value,
    )
    return SharedWeights(
        weight_fp4=weight_fp4.contiguous(),
        weight_scale=weight_scale.contiguous(),
        weight_scale_f32=weight_scale.float().contiguous(),
        weight_global_scale_inv=weight_global_scale_inv.contiguous(),
    )


def _make_input_scale_inv(cfg: BenchmarkConfig) -> torch.Tensor:
    calibration = torch.randn(
        cfg.num_tokens,
        cfg.input_size,
        device="cuda",
        dtype=cfg.dtype,
    )
    return _global_scale_inv_from_tensor(calibration)


def _make_inputs(
    cfg: BenchmarkConfig,
    backend: NvFp4LinearBackend,
    input_global_scale_inv: torch.Tensor,
    weight_global_scale_inv: torch.Tensor,
) -> QBProjInputs:
    x_dense = torch.randn(
        cfg.num_tokens,
        cfg.input_size,
        device="cuda",
        dtype=cfg.dtype,
    )
    input_fp4, input_scale = ops.scaled_fp4_quant(
        x_dense,
        input_global_scale_inv,
        is_sf_swizzled_layout=True,
        backend=backend.value,
    )
    alpha = (1.0 / (input_global_scale_inv * weight_global_scale_inv)).to(
        device="cuda",
        dtype=torch.float32,
    )
    return QBProjInputs(
        input_fp4=input_fp4.contiguous(),
        input_scale=input_scale.contiguous(),
        input_scale_f32=input_scale.float().contiguous(),
        input_global_scale_inv=input_global_scale_inv.contiguous(),
        alpha=alpha.contiguous(),
    )


def _reference_q_b_proj(
    inputs: QBProjInputs,
    weights: SharedWeights,
    cfg: BenchmarkConfig,
) -> torch.Tensor:
    input_global_scale = 1.0 / inputs.input_global_scale_inv
    weight_global_scale = 1.0 / weights.weight_global_scale_inv
    x_dq = dequantize_to_dtype(
        inputs.input_fp4,
        inputs.input_scale,
        input_global_scale,
        torch.float32,
        cfg.group_size,
        swizzle=True,
    )
    w_dq = dequantize_to_dtype(
        weights.weight_fp4,
        weights.weight_scale,
        weight_global_scale,
        torch.float32,
        cfg.group_size,
        swizzle=True,
    )
    return torch.matmul(x_dq, w_dq.t()).to(cfg.dtype)


def _swizzled_scale_offset_expr(
    row,
    group,
    num_scale_k_tiles,
):
    row_tile = row // 128
    outer_row = row % 32
    inner_row = (row // 32) % 4
    k_tile = group // 4
    inner_k = group % 4
    return (
        (row_tile * num_scale_k_tiles + k_tile) * 512
        + outer_row * 16
        + inner_row * 4
        + inner_k
    )


@triton.jit
def _triton_q_b_proj_kernel(
    input_fp4_ptr,
    input_stride0,
    input_scale_ptr,
    weight_fp4_ptr,
    weight_stride0,
    weight_scale_ptr,
    alpha_ptr,
    lut_ptr,
    out_ptr,
    out_stride0,
    num_tokens,
    output_size,
    num_groups,
    num_scale_k_tiles,
    GROUP_BYTES: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < output_size
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    alpha = tl.load(alpha_ptr)

    if pid_m >= num_tokens:
        return

    for group in range(0, num_groups):
        input_scale_offset = (
            ((pid_m // 128) * num_scale_k_tiles + (group // 4)) * 512
            + (pid_m % 32) * 16
            + ((pid_m // 32) % 4) * 4
            + (group % 4)
        )
        a_scale = tl.load(input_scale_ptr + input_scale_offset)

        weight_scale_offset = (
            ((offs_n // 128) * num_scale_k_tiles + (group // 4)) * 512
            + (offs_n % 32) * 16
            + ((offs_n // 32) % 4) * 4
            + (group % 4)
        )
        b_scale = tl.load(weight_scale_ptr + weight_scale_offset, mask=n_mask, other=0.0)
        group_scale = a_scale * b_scale * alpha

        dot = tl.zeros((BLOCK_N,), dtype=tl.float32)
        base_byte = group * GROUP_BYTES
        for byte_idx in tl.static_range(0, GROUP_BYTES):
            a_byte = tl.load(
                input_fp4_ptr + pid_m * input_stride0 + base_byte + byte_idx
            ).to(tl.int32)
            b_byte = tl.load(
                weight_fp4_ptr
                + offs_n * weight_stride0
                + base_byte
                + byte_idx,
                mask=n_mask,
                other=0,
            ).to(tl.int32)

            a_lo = tl.load(lut_ptr + (a_byte & 0xF))
            a_hi = tl.load(lut_ptr + ((a_byte >> 4) & 0xF))
            b_lo = tl.load(lut_ptr + (b_byte & 0xF), mask=n_mask, other=0.0)
            b_hi = tl.load(
                lut_ptr + ((b_byte >> 4) & 0xF),
                mask=n_mask,
                other=0.0,
            )
            dot += b_lo * a_lo + b_hi * a_hi

        acc += dot * group_scale

    tl.store(
        out_ptr + pid_m * out_stride0 + offs_n,
        acc.to(out_ptr.dtype.element_ty),
        mask=n_mask,
    )


def triton_kimik25_q_b_proj(
    inputs: QBProjInputs,
    weights: SharedWeights,
    cfg: BenchmarkConfig,
) -> torch.Tensor:
    out = torch.empty(
        (cfg.num_tokens, cfg.output_size),
        device=inputs.input_fp4.device,
        dtype=cfg.dtype,
    )
    lut = _fp4_lut(inputs.input_fp4.device)
    block_n = 64
    grid = (cfg.num_tokens, triton.cdiv(cfg.output_size, block_n))
    _triton_q_b_proj_kernel[grid](
        inputs.input_fp4,
        inputs.input_fp4.stride(0),
        inputs.input_scale_f32.reshape(-1),
        weights.weight_fp4,
        weights.weight_fp4.stride(0),
        weights.weight_scale_f32.reshape(-1),
        inputs.alpha,
        lut,
        out,
        out.stride(0),
        cfg.num_tokens,
        cfg.output_size,
        cfg.num_groups,
        cfg.num_scale_k_tiles,
        GROUP_BYTES=PACKED_GROUP_BYTES,
        BLOCK_N=block_n,
        num_warps=4,
    )
    return out


def _get_triton_compiled_kernel(
    inputs: QBProjInputs,
    weights: SharedWeights,
    cfg: BenchmarkConfig,
):
    out = torch.empty(
        (cfg.num_tokens, cfg.output_size),
        device=inputs.input_fp4.device,
        dtype=cfg.dtype,
    )
    lut = _fp4_lut(inputs.input_fp4.device)
    block_n = 64
    compiled = _triton_q_b_proj_kernel.warmup(
        inputs.input_fp4,
        inputs.input_fp4.stride(0),
        inputs.input_scale_f32.reshape(-1),
        weights.weight_fp4,
        weights.weight_fp4.stride(0),
        weights.weight_scale_f32.reshape(-1),
        inputs.alpha,
        lut,
        out,
        out.stride(0),
        cfg.num_tokens,
        cfg.output_size,
        cfg.num_groups,
        cfg.num_scale_k_tiles,
        GROUP_BYTES=PACKED_GROUP_BYTES,
        BLOCK_N=block_n,
        num_warps=4,
        grid=(cfg.num_tokens, triton.cdiv(cfg.output_size, block_n)),
    )
    return compiled


def _dump_triton_artifacts(
    output_dir: Path,
    inputs: QBProjInputs,
    weights: SharedWeights,
    cfg: BenchmarkConfig,
) -> None:
    compiled = _get_triton_compiled_kernel(inputs, weights, cfg)
    kernel_name = _sanitize_kernel_name(compiled.name)
    _write_text_file(output_dir / f"{kernel_name}.ttir", compiled.asm["ttir"])
    _write_text_file(output_dir / f"{kernel_name}.ttgir", compiled.asm["ttgir"])
    _write_text_file(output_dir / f"{kernel_name}.llir", compiled.asm["llir"])
    _write_text_file(output_dir / f"{kernel_name}.ptx", compiled.asm["ptx"])
    _write_binary_file(output_dir / f"{kernel_name}.cubin", compiled.asm["cubin"])
    _write_text_file(
        output_dir / f"{kernel_name}.sass",
        _get_sass_from_cubin(compiled.asm["cubin"]),
    )


def _production_q_b_proj(
    inputs: QBProjInputs,
    weights: SharedWeights,
    cfg: BenchmarkConfig,
) -> torch.Tensor:
    del cfg
    backend = _selected_production_backend()
    if backend == NvFp4LinearBackend.VLLM_CUTLASS:
        return ops.cutlass_scaled_fp4_mm(
            inputs.input_fp4,
            weights.weight_fp4,
            inputs.input_scale,
            weights.weight_scale,
            inputs.alpha,
            DTYPE,
        )

    backend_name = backend.value[len("flashinfer-") :]
    return flashinfer_scaled_fp4_mm(
        inputs.input_fp4,
        weights.weight_fp4,
        inputs.input_scale,
        weights.weight_scale,
        inputs.alpha,
        DTYPE,
        backend=backend_name,
    )


def _dump_no_artifacts(
    output_dir: Path,
    inputs: QBProjInputs,
    weights: SharedWeights,
    cfg: BenchmarkConfig,
) -> None:
    del output_dir, inputs, weights, cfg


def _build_production_version() -> KernelVersion:
    return KernelVersion(
        name="production",
        make_runner=_make_production_runner,
        run_once=_production_q_b_proj,
        dump_artifacts=_dump_no_artifacts,
    )


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
        out = triton_kimik25_q_b_proj(data_pool[index], weights, cfg)
        index = (index + 1) % len(data_pool)
        return out

    return run


def _build_triton_version() -> KernelVersion:
    return KernelVersion(
        name="triton",
        make_runner=_make_triton_runner,
        run_once=triton_kimik25_q_b_proj,
        dump_artifacts=_dump_triton_artifacts,
    )


def _build_cutedsl_version() -> KernelVersion:
    try:
        import cutlass
        import cutlass.cute as cute
        import cutlass.torch as cutlass_torch
        from cutlass.base_dsl import compiler as cutlass_compiler
        from cuda.bindings.driver import CUstream
        from cutlass.cute.runtime import from_dlpack
    except ImportError as exc:  # pragma: no cover - benchmark-only path
        raise RuntimeError(
            "CuTeDSL benchmarking requires the `cutlass` Python package."
        ) from exc

    @cute.kernel
    def kimik25_q_b_proj_kernel(
        input_fp4: cute.Tensor,  # (M, packed_k)
        input_scale: cute.Tensor,  # (scale_numel,)
        weight_fp4: cute.Tensor,  # (N, packed_k)
        weight_scale: cute.Tensor,  # (scale_numel,)
        alpha: cute.Tensor,  # (1,)
        lut: cute.Tensor,  # (16,)
        output: cute.Tensor,  # (M, N)
        num_groups: cutlass.Constexpr,
        num_scale_k_tiles: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        row, col_block, _ = cute.arch.block_idx()
        col = col_block * 32 + tidx

        if row < output.shape[0] and col < output.shape[1]:
            alpha_val = alpha[0]
            acc: cutlass.Float32 = 0.0
            for group in cutlass.range_constexpr(num_groups):
                input_scale_offset = _swizzled_scale_offset_expr(
                    row,
                    group,
                    num_scale_k_tiles,
                )
                weight_scale_offset = _swizzled_scale_offset_expr(
                    col,
                    group,
                    num_scale_k_tiles,
                )
                group_scale = (
                    input_scale[input_scale_offset]
                    * weight_scale[weight_scale_offset]
                    * alpha_val
                )

                base_byte = group * PACKED_GROUP_BYTES
                group_dot: cutlass.Float32 = 0.0
                for byte_idx in cutlass.range_constexpr(PACKED_GROUP_BYTES):
                    a_byte = input_fp4[row, base_byte + byte_idx].to(cutlass.Int32)
                    b_byte = weight_fp4[col, base_byte + byte_idx].to(cutlass.Int32)

                    a_lo = lut[a_byte & 0xF]
                    a_hi = lut[(a_byte >> 4) & 0xF]
                    b_lo = lut[b_byte & 0xF]
                    b_hi = lut[(b_byte >> 4) & 0xF]
                    group_dot += a_lo * b_lo + a_hi * b_hi

                acc += group_dot * group_scale

            output[row, col] = acc.to(cutlass.BFloat16)

    @cute.jit
    def kimik25_q_b_proj(
        input_fp4: cute.Tensor,
        input_scale: cute.Tensor,
        weight_fp4: cute.Tensor,
        weight_scale: cute.Tensor,
        alpha: cute.Tensor,
        lut: cute.Tensor,
        output: cute.Tensor,
        num_groups: cutlass.Constexpr,
        num_scale_k_tiles: cutlass.Constexpr,
        stream: CUstream,
    ):
        grid = (input_fp4.shape[0], cute.ceil_div(output.shape[1], 32), 1)
        block = (32, 1, 1)
        kimik25_q_b_proj_kernel(
            input_fp4,
            input_scale,
            weight_fp4,
            weight_scale,
            alpha,
            lut,
            output,
            num_groups,
            num_scale_k_tiles,
        ).launch(grid=grid, block=block, stream=stream)

    compiled_function_cache: dict[tuple[Any, ...], Any] = {}
    compiled_executor_cache: dict[tuple[Any, ...], Any] = {}

    def _make_dynamic_cute_tensor(data: torch.Tensor):
        return from_dlpack(data, assumed_align=16).mark_layout_dynamic(
            leading_dim=cutlass_torch.get_leading_dim(data)
        )

    def _make_flat_cute_tensor(data: torch.Tensor):
        return from_dlpack(data.reshape(-1), assumed_align=16)

    def _get_cutedsl_cache_key(
        inputs: QBProjInputs,
        weights: SharedWeights,
        cfg: BenchmarkConfig,
        dump_dir: Path | None,
    ) -> tuple[Any, ...]:
        return (
            torch.cuda.current_device(),
            tuple(inputs.input_fp4.shape),
            tuple(inputs.input_fp4.stride()),
            tuple(inputs.input_scale_f32.shape),
            tuple(weights.weight_fp4.shape),
            tuple(weights.weight_fp4.stride()),
            tuple(weights.weight_scale_f32.shape),
            tuple(cfg.__dict__.items()),
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
            out = torch.empty(
                (cfg.num_tokens, cfg.output_size),
                device=inputs.input_fp4.device,
                dtype=cfg.dtype,
            )
            lut = _fp4_lut(inputs.input_fp4.device)
            compile_callable = cute.compile
            if dump_dir is not None:
                dump_dir.mkdir(parents=True, exist_ok=True)
                compile_callable = cute.compile[(
                    cutlass_compiler.KeepPTX(True),
                    cutlass_compiler.KeepCUBIN(True),
                    cutlass_compiler.DumpDir(str(dump_dir)),
                )]
            compiled = compile_callable(
                kimik25_q_b_proj,
                input_fp4=_make_dynamic_cute_tensor(inputs.input_fp4),
                input_scale=_make_flat_cute_tensor(inputs.input_scale_f32),
                weight_fp4=_make_dynamic_cute_tensor(weights.weight_fp4),
                weight_scale=_make_flat_cute_tensor(weights.weight_scale_f32),
                alpha=from_dlpack(inputs.alpha),
                lut=from_dlpack(lut),
                output=_make_dynamic_cute_tensor(out),
                num_groups=cfg.num_groups,
                num_scale_k_tiles=cfg.num_scale_k_tiles,
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
                inputs, weights, cfg, dump_dir=None
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
        out = torch.empty(
            (cfg.num_tokens, cfg.output_size),
            device=inputs.input_fp4.device,
            dtype=cfg.dtype,
        )
        lut = _fp4_lut(inputs.input_fp4.device)
        executor = _get_cutedsl_executor(inputs, weights, cfg)
        executor(
            input_fp4=_make_dynamic_cute_tensor(inputs.input_fp4),
            input_scale=_make_flat_cute_tensor(inputs.input_scale_f32),
            weight_fp4=_make_dynamic_cute_tensor(weights.weight_fp4),
            weight_scale=_make_flat_cute_tensor(weights.weight_scale_f32),
            alpha=from_dlpack(inputs.alpha),
            lut=from_dlpack(lut),
            output=_make_dynamic_cute_tensor(out),
            stream=cutlass_torch.current_stream(),
        )
        return out

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


def _build_version(name: str) -> KernelVersion:
    if name == "production":
        return _build_production_version()
    if name == "triton":
        return _build_triton_version()
    if name == "cutedsl":
        return _build_cutedsl_version()
    raise KeyError(f"Unknown kernel version {name!r}")


def _get_kernel_version(name: str) -> KernelVersion:
    version = _KERNEL_VERSION_CACHE.get(name)
    if version is None:
        version = _build_version(name)
        _KERNEL_VERSION_CACHE[name] = version
    return version


def _benchmark_once(
    runner,
    *,
    use_cudagraph: bool,
    num_warmup_iters: int,
    num_iters: int,
) -> list[float]:
    if use_cudagraph:
        ms = triton.testing.do_bench_cudagraph(
            runner,
            rep=num_iters,
            warmup=num_warmup_iters,
        )
        return [ms / 1000.0]

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
    diff = (actual.float() - ref.float()).abs().max().item()
    return diff


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
        description="Benchmark the Kimi-K2.5 NVFP4 q_b_proj GEMM kernel."
    )
    parser.add_argument(
        "--versions",
        nargs="+",
        default=["production", "triton", "cutedsl"],
        choices=["production", "triton", "cutedsl"],
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
        help="How many input buffers to rotate through.",
    )
    parser.add_argument(
        "--use-cudagraph",
        action="store_true",
        help="Benchmark with Triton's CUDA-graph helper.",
    )
    parser.add_argument(
        "--check-correctness",
        action="store_true",
        help="Compare each kernel result against a dequantized float32 reference.",
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
    backend = _selected_production_backend()
    kE2M1ToFloat_handle.val = kE2M1ToFloat_handle.val.to("cuda")
    set_random_seed(args.seed)

    print(f"production_backend: {backend.value}")
    print(
        f"{'tokens':>8}  {'version':<12}  {'median_us':>10}  {'tflops':>8}  "
        f"{'max_abs_diff':>12}"
    )

    for num_tokens in args.num_tokens:
        cfg = BenchmarkConfig(num_tokens=num_tokens)
        input_global_scale_inv = _make_input_scale_inv(cfg)
        weights = _make_shared_weights(cfg, backend)
        data_pool = [
            _make_inputs(
                cfg,
                backend,
                input_global_scale_inv,
                weights.weight_global_scale_inv,
            )
            for _ in range(args.num_buffers)
        ]

        for version_name in args.versions:
            version = _get_kernel_version(version_name)
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
                max_abs_diff = _check_correctness(version, data_pool[0], weights, cfg)

            diff_str = f"{max_abs_diff:.3e}" if args.check_correctness else "n/a"
            print(
                f"{num_tokens:8d}  {version.name:<12}  "
                f"{median_s * 1e6:10.2f}  {tflops:8.2f}  {diff_str:>12}"
            )


if __name__ == "__main__":
    main()
