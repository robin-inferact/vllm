# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Microbenchmark the Kimi-K2.5 fused Q/KV RMSNorm kernel.

This benchmark isolates just the fused RMSNorm step that operates on the
`qkv_c` MLA view with layout:

    qkv_a: [num_tokens, q_lora_rank + kv_lora_rank + qk_rope_head_dim]
    qkv_c: qkv_a[:, : q_lora_rank + kv_lora_rank]

The benchmark is registry-based so new kernel versions can be added without
changing the harness. It currently includes:

1. The original CuTeDSL kernel duplicated locally in this benchmark
2. A Triton implementation of the same fused RMSNorm math

Example:

    .venv/bin/python benchmarks/kernels/benchmark_kimi_k25_nvfp4_fused_rmsnorm.py \
        --versions cutedsl triton --num-tokens 7 64 512 4096 \
        --check-correctness

    .venv/bin/python benchmarks/kernels/benchmark_kimi_k25_nvfp4_fused_rmsnorm.py \
        --versions triton --num-tokens 4096 --use-cudagraph

    .venv/bin/python benchmarks/kernels/benchmark_kimi_k25_nvfp4_fused_rmsnorm.py \
        --versions cutedsl triton --num-tokens 4096 \
        --dump-kernel-artifacts-dir /tmp/kimi_rmsnorm_artifacts
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
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
EPS_Q = 1e-5
EPS_KV = 1e-5
DTYPE = torch.bfloat16


@dataclass(frozen=True)
class BenchmarkConfig:
    num_tokens: int
    q_lora_rank: int = Q_LORA_RANK
    kv_lora_rank: int = KV_LORA_RANK
    qk_rope_head_dim: int = QK_ROPE_HEAD_DIM
    eps_q: float = EPS_Q
    eps_kv: float = EPS_KV
    dtype: torch.dtype = DTYPE

    @property
    def total_lora_rank(self) -> int:
        return self.q_lora_rank + self.kv_lora_rank

    @property
    def bytes_per_call(self) -> int:
        # Approximate global-memory traffic: read input, read weights, write input.
        return self.num_tokens * self.total_lora_rank * 3 * torch.tensor(
            [], dtype=self.dtype
        ).element_size()


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


def _make_dump_dir(
    dump_root: Path,
    version_name: str,
    cfg: BenchmarkConfig,
    row_stride: int,
) -> Path:
    return dump_root / version_name / f"tokens_{cfg.num_tokens}_stride_{row_stride}"


def _sanitize_kernel_name(name: str) -> str:
    return name.lstrip("@").replace("/", "_")


@triton.jit
def _triton_fused_qkv_rmsnorm_kernel(
    data_ptr,
    data_row_stride,
    weights_q_ptr,
    weights_kv_ptr,
    q_lora_rank,
    kv_lora_rank,
    eps_q,
    eps_kv,
    BLOCK_SIZE: tl.constexpr,
):
    pid_kind = tl.program_id(0)
    pid_token = tl.program_id(1)

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < kv_lora_rank
    row_ptr = data_ptr + pid_token * data_row_stride

    if pid_kind == 0:
        x0 = tl.load(row_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        x1 = tl.load(row_ptr + offsets + kv_lora_rank, mask=mask, other=0.0).to(
            tl.float32
        )
        x2 = tl.load(
            row_ptr + offsets + 2 * kv_lora_rank,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        sum_sq = tl.sum(x0 * x0 + x1 * x1 + x2 * x2, axis=0)
        invnorm = tl.rsqrt(sum_sq / q_lora_rank + eps_q)

        # Match the CuTeDSL kernel's cast order.
        x0 = (x0 * invnorm).to(data_ptr.dtype.element_ty)
        x1 = (x1 * invnorm).to(data_ptr.dtype.element_ty)
        x2 = (x2 * invnorm).to(data_ptr.dtype.element_ty)

        w0 = tl.load(weights_q_ptr + offsets, mask=mask)
        w1 = tl.load(weights_q_ptr + offsets + kv_lora_rank, mask=mask)
        w2 = tl.load(weights_q_ptr + offsets + 2 * kv_lora_rank, mask=mask)

        tl.store(row_ptr + offsets, (x0 * w0).to(data_ptr.dtype.element_ty), mask=mask)
        tl.store(
            row_ptr + offsets + kv_lora_rank,
            (x1 * w1).to(data_ptr.dtype.element_ty),
            mask=mask,
        )
        tl.store(
            row_ptr + offsets + 2 * kv_lora_rank,
            (x2 * w2).to(data_ptr.dtype.element_ty),
            mask=mask,
        )
    else:
        x3 = tl.load(row_ptr + offsets + 3 * kv_lora_rank, mask=mask, other=0.0).to(
            tl.float32
        )
        sum_sq = tl.sum(x3 * x3, axis=0)
        invnorm = tl.rsqrt(sum_sq / kv_lora_rank + eps_kv)

        x3 = (x3 * invnorm).to(data_ptr.dtype.element_ty)
        w3 = tl.load(weights_kv_ptr + offsets, mask=mask)
        tl.store(
            row_ptr + offsets + 3 * kv_lora_rank,
            (x3 * w3).to(data_ptr.dtype.element_ty),
            mask=mask,
        )


def triton_fused_qkv_rmsnorm(
    data: torch.Tensor,
    weights_q: torch.Tensor,
    weights_kv: torch.Tensor,
    cfg: BenchmarkConfig,
) -> None:
    assert data.is_cuda
    assert data.dtype == cfg.dtype
    assert data.shape == (cfg.num_tokens, cfg.total_lora_rank)
    assert data.stride(1) == 1
    assert weights_q.shape == (cfg.q_lora_rank,)
    assert weights_kv.shape == (cfg.kv_lora_rank,)
    assert weights_q.dtype == cfg.dtype
    assert weights_kv.dtype == cfg.dtype
    assert weights_q.stride(0) == 1
    assert weights_kv.stride(0) == 1

    block_size = triton.next_power_of_2(cfg.kv_lora_rank)
    num_warps = min(max(block_size // 64, 1), 8)
    _triton_fused_qkv_rmsnorm_kernel[(2, data.shape[0])](
        data,
        data.stride(0),
        weights_q,
        weights_kv,
        cfg.q_lora_rank,
        cfg.kv_lora_rank,
        cfg.eps_q,
        cfg.eps_kv,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )


def _get_triton_launch_config(cfg: BenchmarkConfig) -> tuple[int, int]:
    block_size = triton.next_power_of_2(cfg.kv_lora_rank)
    num_warps = min(max(block_size // 64, 1), 8)
    return block_size, num_warps


def _get_triton_compiled_kernel(
    data: torch.Tensor,
    weights_q: torch.Tensor,
    weights_kv: torch.Tensor,
    cfg: BenchmarkConfig,
):
    block_size, num_warps = _get_triton_launch_config(cfg)
    compiled = _triton_fused_qkv_rmsnorm_kernel.warmup(
        data,
        data.stride(0),
        weights_q,
        weights_kv,
        cfg.q_lora_rank,
        cfg.kv_lora_rank,
        cfg.eps_q,
        cfg.eps_kv,
        grid=(2, data.shape[0]),
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    if hasattr(compiled, "result"):
        compiled = compiled.result()
    return compiled


def _dump_triton_artifacts(
    output_dir: Path,
    data: torch.Tensor,
    weights_q: torch.Tensor,
    weights_kv: torch.Tensor,
    cfg: BenchmarkConfig,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    compiled = _get_triton_compiled_kernel(data, weights_q, weights_kv, cfg)
    kernel_name = _sanitize_kernel_name(compiled.name)
    _write_text_file(output_dir / f"{kernel_name}.ptx", compiled.asm["ptx"])
    _write_binary_file(output_dir / f"{kernel_name}.cubin", compiled.asm["cubin"])
    _write_text_file(
        output_dir / f"{kernel_name}.sass",
        _get_sass_from_cubin(compiled.asm["cubin"]),
    )


def _make_mla_like_qkv_c_view(cfg: BenchmarkConfig) -> torch.Tensor:
    qkv_a = torch.randn(
        cfg.num_tokens,
        cfg.total_lora_rank + cfg.qk_rope_head_dim,
        device="cuda",
        dtype=cfg.dtype,
    )
    qkv_c, _ = qkv_a.split([cfg.total_lora_rank, cfg.qk_rope_head_dim], dim=-1)
    return qkv_c


def _make_weights(cfg: BenchmarkConfig) -> tuple[torch.Tensor, torch.Tensor]:
    q_weight = torch.randn(cfg.q_lora_rank, device="cuda", dtype=cfg.dtype)
    kv_weight = torch.randn(cfg.kv_lora_rank, device="cuda", dtype=cfg.dtype)
    return q_weight, kv_weight


def _pytorch_rmsnorm_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    x_fp32 = x.float()
    inv_rms = torch.rsqrt(x_fp32.square().mean(dim=-1, keepdim=True) + eps)
    return (x_fp32 * inv_rms).to(torch.bfloat16) * weight


def _pytorch_fused_reference(
    data: torch.Tensor,
    weights_q: torch.Tensor,
    weights_kv: torch.Tensor,
    cfg: BenchmarkConfig,
) -> torch.Tensor:
    q_x, kv_x = data.split([cfg.q_lora_rank, cfg.kv_lora_rank], dim=-1)
    q_expected = _pytorch_rmsnorm_reference(q_x, weights_q, cfg.eps_q)
    kv_expected = _pytorch_rmsnorm_reference(kv_x, weights_kv, cfg.eps_kv)
    return torch.cat([q_expected, kv_expected], dim=-1)


def _make_triton_runner(
    data_pool: list[torch.Tensor],
    weights_q: torch.Tensor,
    weights_kv: torch.Tensor,
    cfg: BenchmarkConfig,
):
    index = 0

    def run() -> None:
        nonlocal index
        triton_fused_qkv_rmsnorm(data_pool[index], weights_q, weights_kv, cfg)
        index = (index + 1) % len(data_pool)

    return run


def _build_triton_version() -> KernelVersion:
    return KernelVersion(
        name="triton",
        make_runner=_make_triton_runner,
        run_once=triton_fused_qkv_rmsnorm,
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

    nwarps = (Q_LORA_RANK + KV_LORA_RANK) // 256

    @cute.kernel
    def kimik25_rmsnorm_special_qkv_fused_kernel(
        data: cute.Tensor,  # (Sp, (2, lora_dim_kv // 2, 4))
        weights_q: cute.Tensor,  # (2, lora_dim_q // 2)
        weights_kv: cute.Tensor,  # (2, lora_dim_kv // 2)
        lora_dim_q: cutlass.Constexpr,  # must be lora_dim_kv * 3
        lora_dim_kv: cutlass.Constexpr,
        eps_q: cutlass.Constexpr,
        eps_kv: cutlass.Constexpr,
    ):
        allocator = cutlass.utils.SmemAllocator()
        sdata = allocator.allocate_tensor(
            cutlass.Float32,
            layout=cute.make_layout(nwarps),
            byte_alignment=16,
            swizzle=None,
        )

        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()

        Sp = data.shape[0]
        if bid < Sp:
            x0 = data[bid, (None, tid, 0)].load().to(cutlass.Float32)
            x1 = data[bid, (None, tid, 1)].load().to(cutlass.Float32)
            x2 = data[bid, (None, tid, 2)].load().to(cutlass.Float32)
            sum = x0 * x0 + x1 * x1 + x2 * x2
            sum = sum[0] + sum[1]

            sum = cute.arch.warp_reduction_sum(sum, threads_in_group=32)
            if tid % 32 == 0:
                sdata[tid // 32] = sum

            cute.arch.sync_threads()
            ssum: cutlass.Float32 = 0.0
            if tid < nwarps:
                ssum = sdata[tid]

            ssum = cute.arch.warp_reduction_sum(ssum, threads_in_group=nwarps)
            if tid == 0:
                sdata[0] = cute.math.rsqrt(ssum * (1.0 / lora_dim_q) + eps_q)

            cute.arch.sync_threads()
            invnorm = sdata[0]
            data[bid, (None, tid, 0)] = (
                x0 * invnorm
            ).to(cutlass.BFloat16) * weights_q[None, tid].load()
            data[bid, (None, tid, 1)] = (
                x1 * invnorm
            ).to(cutlass.BFloat16) * weights_q[None, tid + lora_dim_kv // 2].load()
            data[bid, (None, tid, 2)] = (
                x2 * invnorm
            ).to(cutlass.BFloat16) * weights_q[None, tid + lora_dim_kv].load()
        else:
            x3 = data[bid - Sp, (None, tid, 3)].load().to(cutlass.Float32)
            sum = x3 * x3
            sum = sum[0] + sum[1]

            sum = cute.arch.warp_reduction_sum(sum, threads_in_group=32)
            if tid % 32 == 0:
                sdata[tid // 32] = sum

            cute.arch.sync_threads()
            ssum: cutlass.Float32 = 0.0
            if tid < nwarps:
                ssum = sdata[tid]

            ssum = cute.arch.warp_reduction_sum(ssum, threads_in_group=nwarps)
            if tid == 0:
                sdata[0] = cute.math.rsqrt(ssum / lora_dim_kv + eps_kv)

            cute.arch.sync_threads()
            invnorm = sdata[0]
            data[bid - Sp, (None, tid, 3)] = (
                x3 * invnorm
            ).to(cutlass.BFloat16) * weights_kv[None, tid].load()

    @cute.jit
    def kimik25_rmsnorm_special_qkv_fused(
        data: cute.Tensor,
        weights_q: cute.Tensor,
        weights_kv: cute.Tensor,
        lora_dim_q: cutlass.Constexpr,
        lora_dim_kv: cutlass.Constexpr,
        eps_q: cutlass.Constexpr,
        eps_kv: cutlass.Constexpr,
        stream: CUstream,
    ):
        row_stride = cute.assume(data.stride[0], divby=2)
        data = cute.make_tensor(
            data.iterator,
            cute.make_layout(
                (data.shape[0], (2, lora_dim_kv // 2, 4)),
                stride=(row_stride, (1, 2, lora_dim_kv)),
            ),
        )
        weights_q = cute.make_tensor(weights_q.iterator, cute.make_layout((2, lora_dim_q // 2)))
        weights_kv = cute.make_tensor(weights_kv.iterator, cute.make_layout((2, lora_dim_kv // 2)))
        grid = (data.shape[0] * 2, 1, 1)
        block = (lora_dim_kv // 2, 1, 1)
        kimik25_rmsnorm_special_qkv_fused_kernel(
            data,
            weights_q,
            weights_kv,
            lora_dim_q,
            lora_dim_kv,
            eps_q,
            eps_kv,
        ).launch(grid=grid, block=block, stream=stream)

    compiled_function_cache: dict[tuple[Any, ...], Any] = {}
    compiled_executor_cache: dict[tuple[Any, ...], Any] = {}

    def _make_dynamic_cute_tensor(data: torch.Tensor):
        return from_dlpack(data, assumed_align=16).mark_layout_dynamic(
            leading_dim=cutlass_torch.get_leading_dim(data)
        )

    def _get_cutedsl_cache_key(
        data: torch.Tensor,
        weights_q: torch.Tensor,
        weights_kv: torch.Tensor,
        cfg: BenchmarkConfig,
        dump_dir: Path | None,
    ) -> tuple[Any, ...]:
        return (
            torch.cuda.current_device(),
            tuple(data.shape),
            tuple(data.stride()),
            data.dtype,
            tuple(weights_q.shape),
            tuple(weights_kv.shape),
            weights_q.dtype,
            weights_kv.dtype,
            cfg.q_lora_rank,
            cfg.kv_lora_rank,
            cfg.eps_q,
            cfg.eps_kv,
            str(dump_dir) if dump_dir is not None else None,
        )

    def _get_cutedsl_compiled_function(
        data: torch.Tensor,
        weights_q: torch.Tensor,
        weights_kv: torch.Tensor,
        cfg: BenchmarkConfig,
        *,
        dump_dir: Path | None = None,
    ):
        cache_key = _get_cutedsl_cache_key(data, weights_q, weights_kv, cfg, dump_dir)
        compiled = compiled_function_cache.get(cache_key)
        if compiled is None:
            compile_callable = cute.compile
            if dump_dir is not None:
                dump_dir.mkdir(parents=True, exist_ok=True)
                compile_callable = cute.compile[(
                    cutlass_compiler.KeepPTX(True),
                    cutlass_compiler.KeepCUBIN(True),
                    cutlass_compiler.DumpDir(str(dump_dir)),
                )]
            compiled = compile_callable(
                kimik25_rmsnorm_special_qkv_fused,
                data=_make_dynamic_cute_tensor(data),
                weights_q=from_dlpack(weights_q, assumed_align=16),
                weights_kv=from_dlpack(weights_kv, assumed_align=16),
                lora_dim_q=cfg.q_lora_rank,
                lora_dim_kv=cfg.kv_lora_rank,
                eps_q=cfg.eps_q,
                eps_kv=cfg.eps_kv,
                stream=cutlass_torch.current_stream(),
            )
            compiled_function_cache[cache_key] = compiled
        return compiled

    def _get_cutedsl_executor(
        data: torch.Tensor,
        weights_q: torch.Tensor,
        weights_kv: torch.Tensor,
        cfg: BenchmarkConfig,
    ):
        cache_key = _get_cutedsl_cache_key(
            data, weights_q, weights_kv, cfg, dump_dir=None
        )
        executor = compiled_executor_cache.get(cache_key)
        if executor is None:
            executor = _get_cutedsl_compiled_function(
                data, weights_q, weights_kv, cfg, dump_dir=None
            ).to(None)
            compiled_executor_cache[cache_key] = executor
        return executor

    def dump_cutedsl_artifacts(
        output_dir: Path,
        data: torch.Tensor,
        weights_q: torch.Tensor,
        weights_kv: torch.Tensor,
        cfg: BenchmarkConfig,
    ) -> None:
        compiled = _get_cutedsl_compiled_function(
            data,
            weights_q,
            weights_kv,
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

    def cutedsl_fused_qkv_rmsnorm(
        data: torch.Tensor,
        weights_q: torch.Tensor,
        weights_kv: torch.Tensor,
        cfg: BenchmarkConfig,
    ) -> None:
        assert cfg.q_lora_rank == Q_LORA_RANK
        assert cfg.kv_lora_rank == KV_LORA_RANK
        assert cfg.total_lora_rank == Q_LORA_RANK + KV_LORA_RANK
        executor = _get_cutedsl_executor(data, weights_q, weights_kv, cfg)
        executor(
            data=_make_dynamic_cute_tensor(data),
            weights_q=from_dlpack(weights_q),
            weights_kv=from_dlpack(weights_kv),
            stream=cutlass_torch.current_stream(),
        )

    def make_cutedsl_runner(
        data_pool: list[torch.Tensor],
        weights_q: torch.Tensor,
        weights_kv: torch.Tensor,
        cfg: BenchmarkConfig,
    ):
        assert cfg.q_lora_rank == Q_LORA_RANK
        assert cfg.kv_lora_rank == KV_LORA_RANK
        assert cfg.total_lora_rank == Q_LORA_RANK + KV_LORA_RANK
        cute_data_pool = [_make_dynamic_cute_tensor(data) for data in data_pool]
        cute_weights_q = from_dlpack(weights_q)
        cute_weights_kv = from_dlpack(weights_kv)
        executor = _get_cutedsl_executor(data_pool[0], weights_q, weights_kv, cfg)
        index = 0

        def run() -> None:
            nonlocal index
            executor(
                data=cute_data_pool[index],
                weights_q=cute_weights_q,
                weights_kv=cute_weights_kv,
                stream=cutlass_torch.current_stream(),
            )
            index = (index + 1) % len(cute_data_pool)

        return run

    return KernelVersion(
        name="cutedsl",
        make_runner=make_cutedsl_runner,
        run_once=cutedsl_fused_qkv_rmsnorm,
        dump_artifacts=dump_cutedsl_artifacts,
    )


KERNEL_VERSION_BUILDERS = {
    "cutedsl": _build_cutedsl_version,
    "triton": _build_triton_version,
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
    version_names: list[str], baseline_version: str | None
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
        # Warm up eagerly so lazy JIT compilation does not happen during capture.
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


def _run_correctness(
    version: KernelVersion,
    cfg: BenchmarkConfig,
    weights_q: torch.Tensor,
    weights_kv: torch.Tensor,
) -> float:
    test_input = _make_mla_like_qkv_c_view(cfg)
    expected = _pytorch_fused_reference(test_input, weights_q, weights_kv, cfg)
    actual = _make_mla_like_qkv_c_view(cfg)
    actual.copy_(test_input)
    version.run_once(actual, weights_q, weights_kv, cfg)
    torch.cuda.synchronize()
    return (actual.float() - expected.float()).abs().max().item()


def _dump_selected_kernel_artifacts(
    dump_root: Path,
    version_name: str,
    version: KernelVersion,
    cfg: BenchmarkConfig,
    weights_q: torch.Tensor,
    weights_kv: torch.Tensor,
    row_stride: int,
) -> None:
    dump_dir = _make_dump_dir(dump_root, version_name, cfg, row_stride)
    sample_input = _make_mla_like_qkv_c_view(cfg)
    version.dump_artifacts(dump_dir, sample_input, weights_q, weights_kv, cfg)


def main(args) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    if not current_platform.is_device_capability_family(100):
        print("Warning: this benchmark is primarily intended for Blackwell GPUs (SM10x).")

    set_random_seed(args.seed)
    if args.versions is not None:
        version_names = _resolve_version_names(args.versions)
    elif args.provider is not None:
        version_names = (
            ["cutedsl", "triton"] if args.provider == "both" else [args.provider]
        )
    else:
        version_names = _resolve_version_names(["all"])
    baseline_version = _resolve_baseline_version(
        version_names, args.baseline_version
    )
    versions = {name: _get_kernel_version(name) for name in version_names}

    if baseline_version is not None:
        print(f"Baseline version: {baseline_version}")
    if args.dump_kernel_artifacts_dir is not None:
        dump_root = Path(args.dump_kernel_artifacts_dir).expanduser().resolve()
        print(f"Kernel artifact dump root: {dump_root}")

    header = (
        f"{'version':<10} {'tokens':>8} {'row_stride':>10} {'median_us':>12} "
        f"{'min_us':>12} {'max_us':>12} {'gbps':>10} {'speedup':>10} {'max_abs_diff':>14}"
    )
    print(header)
    print("-" * len(header))

    for num_tokens in args.num_tokens:
        cfg = BenchmarkConfig(num_tokens=num_tokens)
        weights_q, weights_kv = _make_weights(cfg)
        stride0 = _make_mla_like_qkv_c_view(cfg).stride(0)

        if args.dump_kernel_artifacts_dir is not None:
            for version_name in version_names:
                _dump_selected_kernel_artifacts(
                    dump_root,
                    version_name,
                    versions[version_name],
                    cfg,
                    weights_q,
                    weights_kv,
                    stride0,
                )

        diffs: dict[str, float | None] = {}
        if args.check_correctness:
            for version_name in version_names:
                diffs[version_name] = _run_correctness(
                    versions[version_name], cfg, weights_q, weights_kv
                )
        else:
            diffs = {version_name: None for version_name in version_names}

        medians_us: dict[str, float] = {}
        mins_us: dict[str, float] = {}
        maxs_us: dict[str, float] = {}

        for version_name in version_names:
            data_pool = [_make_mla_like_qkv_c_view(cfg) for _ in range(args.num_buffers)]
            runner = versions[version_name].make_runner(
                data_pool, weights_q, weights_kv, cfg
            )
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
            speedup = None
            if baseline_us is not None:
                speedup = baseline_us / median_us

            print(
                f"{version_name:<10} "
                f"{num_tokens:>8d} "
                f"{stride0:>10d} "
                f"{median_us:>12.2f} "
                f"{mins_us[version_name]:>12.2f} "
                f"{maxs_us[version_name]:>12.2f} "
                f"{_format_bandwidth(cfg.bytes_per_call, median_us):>10.2f} "
                f"{_format_speedup(speedup):>10} "
                f"{_format_diff(diffs[version_name]):>14}"
            )


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Benchmark the Kimi-K2.5 fused Q/KV RMSNorm kernels."
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
        help="Version to use as the speedup baseline. Defaults to `cutedsl` if selected.",
    )
    parser.add_argument(
        "--provider",
        choices=["cutedsl", "triton", "both"],
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
        help="Compare each kernel against a PyTorch reference before timing.",
    )
    parser.add_argument(
        "--use-cudagraph",
        action="store_true",
        help=(
            "Capture a CUDA graph containing --num-iters kernel launches and "
            "time graph replay instead of eager per-iteration dispatch."
        ),
    )
    parser.add_argument(
        "--dump-kernel-artifacts-dir",
        type=str,
        default=None,
        help=(
            "If set, compile each selected kernel version for each token count "
            "and dump PTX, cubin, and SASS artifacts under this directory "
            "before benchmarking."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)

    main(parser.parse_args())
