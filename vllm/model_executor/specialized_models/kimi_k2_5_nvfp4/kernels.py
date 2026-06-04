# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CuTe DSL kernels for the Kimi-K2.5 NVFP4 specialized model.

These Blackwell (SM10x) kernels fuse the MLA-specific RoPE, RMSNorm,
FP8 quantization, and KV-cache-write steps used by the Kimi-K2.5 NVFP4
runtime profile. They are checked in independently of the specialized
model so the model implementation can import them once it lands.

Each public ``_run_*`` helper wraps a ``@cute.jit`` launcher and a
``@cute.kernel`` device function, compiling lazily through a per-process
executor cache keyed on tensor layouts and the active CUDA device.
"""

from typing import Any

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import torch
from cuda.bindings.driver import CUstream
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T, dsl_user_op

from vllm.platforms import current_platform

_CUTEDSL_EXECUTOR_CACHE: dict[tuple[Any, ...], Any] = {}


def _get_cutedsl_executor(
    cache_key: tuple[Any, ...],
    jit_fn: Any,
    **compile_kwargs: Any,
) -> Any:
    executor = _CUTEDSL_EXECUTOR_CACHE.get(cache_key)
    if executor is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "CuTeDSL executor cache miss during CUDA graph capture for "
                f"{cache_key[0]!r}. Run an eager warmup before capture."
            )
        executor = cute.compile(jit_fn, **compile_kwargs).to(None)
        _CUTEDSL_EXECUTOR_CACHE[cache_key] = executor
    return executor


def _cutedsl_arg_cache_key(arg: Any) -> Any:
    cache_key = getattr(arg, "__cache_key__", None)
    return arg if cache_key is None else cache_key


def _make_dynamic_cute_tensor(data: torch.Tensor):
    return from_dlpack(data, assumed_align=16).mark_layout_dynamic(
        leading_dim=cutlass_torch.get_leading_dim(data)
    )


def _make_fully_dynamic_cute_tensor(data: torch.Tensor):
    return from_dlpack(data, assumed_align=16).mark_layout_dynamic()


def _cuda_device_cache_key() -> int:
    return torch.cuda.current_device() if torch.cuda.is_available() else -1


@dsl_user_op
def _cvt_f32_to_e4m3(a: cutlass.Float32, *, loc=None, ip=None) -> cutlass.Uint32:
    return cutlass.Uint32(
        llvm.inline_asm(
            T.i32(),
            [cutlass.Float32(a).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .b16 fp8_pair;
                .reg .f32 zero;
                mov.f32 zero, 0f00000000;
                cvt.rn.satfinite.e4m3x2.f32 fp8_pair, zero, $1;
                cvt.u32.u16 $0, fp8_pair;
            }
            """,
            "=r,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.kernel
def kimik25_concat_and_cache_mla_kernel(
    kv_c: cute.Tensor,  # (Sp, kv_lora_rank)
    k_pe: cute.Tensor,  # (Sp, pe_dim)
    kv_cache: cute.Tensor,  # (num_blocks, block_size, kv_lora_rank + pe_dim)
    slot_mapping: cute.Tensor,  # (Sp,)
    scale: cute.Tensor,  # (1,)
    kv_lora_rank: cutlass.Constexpr,
    pe_dim: cutlass.Constexpr,
    kv_cache_block_factor: cutlass.Constexpr,
):
    tid, _, _ = cute.arch.thread_idx()
    token_idx, split_idx, _ = cute.arch.block_idx()
    kv_c_elems_per_split: cutlass.Constexpr = kv_lora_rank // kv_cache_block_factor

    slot_idx = slot_mapping[token_idx]
    if slot_idx >= 0:
        block_size = kv_cache.shape[1]
        cache_block_idx = slot_idx // block_size
        cache_block_offset = slot_idx % block_size
        scale_value = scale[0].to(cutlass.Float32)

        if split_idx > 0:
            kv_c_idx = (split_idx - 1) * kv_c_elems_per_split + tid
            kv_c_val = kv_c[token_idx, kv_c_idx].to(cutlass.Float32) / scale_value
            kv_cache[cache_block_idx, cache_block_offset, kv_c_idx] = cutlass.Uint8(
                _cvt_f32_to_e4m3(kv_c_val) & cutlass.Uint32(0xFF)
            )
        else:
            if tid < pe_dim:
                k_pe_val = k_pe[token_idx, tid].to(cutlass.Float32) / scale_value
                kv_cache[
                    cache_block_idx,
                    cache_block_offset,
                    kv_lora_rank + tid,
                ] = cutlass.Uint8(_cvt_f32_to_e4m3(k_pe_val) & cutlass.Uint32(0xFF))


@cute.jit
def kimik25_concat_and_cache_mla(
    kv_c: cute.Tensor,
    k_pe: cute.Tensor,
    kv_cache: cute.Tensor,
    slot_mapping: cute.Tensor,
    scale: cute.Tensor,
    kv_lora_rank: cutlass.Constexpr,
    pe_dim: cutlass.Constexpr,
    kv_cache_block_factor: cutlass.Constexpr,
    stream: CUstream,
):
    assert kv_lora_rank == 512
    assert pe_dim == 64
    assert pe_dim % 2 == 0
    assert kv_cache_block_factor > 0
    assert kv_lora_rank % kv_cache_block_factor == 0
    threads_per_block: cutlass.Constexpr = kv_lora_rank // kv_cache_block_factor
    assert threads_per_block >= pe_dim
    assert kv_c.stride[1] == 1
    assert k_pe.stride[1] == 1
    assert kv_cache.stride[2] == 1
    kv_c = cute.make_tensor(
        kv_c.iterator,
        cute.make_layout(
            (kv_c.shape[0], kv_lora_rank),
            stride=(kv_c.stride[0], 1),
        ),
    )
    k_pe = cute.make_tensor(
        k_pe.iterator,
        cute.make_layout(
            (k_pe.shape[0], pe_dim),
            stride=(k_pe.stride[0], 1),
        ),
    )
    kv_cache = cute.make_tensor(
        kv_cache.iterator,
        cute.make_layout(
            (kv_cache.shape[0], kv_cache.shape[1], kv_lora_rank + pe_dim),
            stride=(kv_cache.stride[0], kv_cache.stride[1], 1),
        ),
    )
    kimik25_concat_and_cache_mla_kernel(
        kv_c,
        k_pe,
        kv_cache,
        slot_mapping,
        scale,
        kv_lora_rank,
        pe_dim,
        kv_cache_block_factor,
    ).launch(
        grid=(slot_mapping.shape[0], kv_cache_block_factor + 1, 1),
        block=(threads_per_block, 1, 1),
        stream=stream,
    )


@cute.kernel
def kimik25_rmsnorm_special_qkv_fused_kernel(
    data: cute.Tensor,  # (Sp, (2, lora_dim_kv // 2, 4))
    positions: cute.Tensor,  # (Sp,)
    k_pe: cute.Tensor,  # (Sp, (2, pe_dim // 2))
    cos_sin_cache: cute.Tensor,  # (max_position_embeddings, pe_dim)
    weights_q: cute.Tensor,  # (2, lora_dim_q // 2)
    weights_kv: cute.Tensor,  # (2, lora_dim_kv // 2)
    lora_dim_q: cutlass.Constexpr,  # must be lora_dim_kv * 3
    lora_dim_kv: cutlass.Constexpr,
    pe_dim: cutlass.Constexpr,
    eps_q: cutlass.Constexpr,
    eps_kv: cutlass.Constexpr,
):
    nwarps = lora_dim_kv // 64
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
        w0 = weights_q[None, tid].load()
        w1 = weights_q[None, tid + lora_dim_kv // 2].load()
        w2 = weights_q[None, tid + lora_dim_kv].load()
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
            sdata[0] = cute.math.rsqrt(ssum / lora_dim_q + eps_q)

        cute.arch.sync_threads()
        invnorm = sdata[0]
        data[bid, (None, tid, 0)] = (x0 * invnorm).to(cutlass.BFloat16) * w0
        data[bid, (None, tid, 1)] = (x1 * invnorm).to(cutlass.BFloat16) * w1
        data[bid, (None, tid, 2)] = (x2 * invnorm).to(cutlass.BFloat16) * w2
    elif bid < Sp * 2:
        x3 = data[bid - Sp, (None, tid, 3)].load().to(cutlass.Float32)
        w3 = weights_kv[None, tid].load()
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
        data[bid - Sp, (None, tid, 3)] = (x3 * invnorm).to(cutlass.BFloat16) * w3
    else:
        token_idx = bid - Sp * 2
        half_pe_dim: cutlass.Constexpr = pe_dim // 2
        if tid < half_pe_dim:
            pos = positions[token_idx]
            cos = cos_sin_cache[pos, tid].to(cutlass.Float32)
            sin = cos_sin_cache[pos, tid + half_pe_dim].to(cutlass.Float32)
            in_scratch = cute.make_rmem_tensor(2, dtype=cutlass.BFloat16)
            cute.autovec_copy(k_pe[token_idx, (None, tid)], in_scratch)
            a = in_scratch[0].to(cutlass.Float32)
            b = in_scratch[1].to(cutlass.Float32)
            in_scratch[0] = (a * cos - b * sin).to(cutlass.BFloat16)
            in_scratch[1] = (a * sin + b * cos).to(cutlass.BFloat16)
            cute.autovec_copy(in_scratch, k_pe[token_idx, (None, tid)])


@cute.jit
def kimik25_rmsnorm_special_qkv_fused(
    data: cute.Tensor,
    positions: cute.Tensor,
    k_pe: cute.Tensor,
    cos_sin_cache: cute.Tensor,
    weights_q: cute.Tensor,
    weights_kv: cute.Tensor,
    lora_dim_q: cutlass.Constexpr,
    lora_dim_kv: cutlass.Constexpr,
    pe_dim: cutlass.Constexpr,
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
    weights_q = cute.make_tensor(
        weights_q.iterator, cute.make_layout((2, lora_dim_q // 2))
    )
    weights_kv = cute.make_tensor(
        weights_kv.iterator, cute.make_layout((2, lora_dim_kv // 2))
    )
    k_pe = cute.make_tensor(
        k_pe.iterator,
        cute.make_layout(
            (k_pe.shape[0], (2, pe_dim // 2)),
            stride=(cute.assume(k_pe.stride[0], divby=2), (1, 2)),
        ),
    )
    grid = (data.shape[0] * 3, 1, 1)
    block = (lora_dim_kv // 2, 1, 1)
    kimik25_rmsnorm_special_qkv_fused_kernel(
        data,
        positions,
        k_pe,
        cos_sin_cache,
        weights_q,
        weights_kv,
        lora_dim_q,
        lora_dim_kv,
        pe_dim,
        eps_q,
        eps_kv,
    ).launch(grid=grid, block=block, stream=stream)


@cute.kernel
def kimik25_rope_kernel(
    positions: cute.Tensor,  # (Sp,)
    query: cute.Tensor,  # (Sp, (K, N_local // K), (2, R // 2))
    cos_sin_cache: cute.Tensor,  # (max_position_embeddings, R)
    K: cutlass.Constexpr,
):
    scratch = cute.make_rmem_tensor((2, K), dtype=cutlass.BFloat16)

    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()

    pos = positions[bidx]
    cos, sin = cos_sin_cache[pos, tidx], cos_sin_cache[pos, tidx + 32]
    for i in cutlass.range_constexpr(K):
        cute.autovec_copy(query[bidx, (i, bidy), (None, tidx)], scratch[None, i])

    for i in cutlass.range_constexpr(K):
        a, b = scratch[0, i], scratch[1, i]
        scratch[0, i] = a * cos - b * sin
        scratch[1, i] = a * sin + b * cos
        cute.autovec_copy(scratch[None, i], query[bidx, (i, bidy), (None, tidx)])


@cute.jit
def kimik25_rope(
    positions: cute.Tensor,  # (Sp,)
    query: cute.Tensor,  # (Sp, N_local, R):(?, ?, 1)
    cos_sin_cache: cute.Tensor,  # (max_position_embeddings, R)
    N_local: cutlass.Constexpr,
    half_rope_dim: cutlass.Constexpr,
    stream: CUstream,
):
    K: cutlass.Constexpr = 8
    assert N_local % K == 0
    sp = positions.shape[0]
    query = cute.make_tensor(
        query.iterator,
        cute.make_layout(
            (sp, (K, N_local // K), (2, half_rope_dim)),
            stride=(
                cute.assume(query.stride[0], divby=2),
                (cute.assume(query.stride[1], divby=2), query.stride[1] * K),
                (1, 2),
            ),
        ),
    )
    kimik25_rope_kernel(positions, query, cos_sin_cache, K).launch(
        grid=(sp, N_local // K, 1),
        block=(half_rope_dim, 1, 1),
        stream=stream,
    )


@cute.kernel
def kimik25_decode_rope_concat_quant_fp8_kernel(
    positions: cute.Tensor,  # (B,)
    ql_nope: cute.Tensor,  # (B, N, q_lora_dim)
    q_pe: cute.Tensor,  # (B, N, (2, pe_dim // 2))
    q_out: cute.Tensor,  # uint8 bytes, (B, N, q_lora_dim + pe_dim)
    cos_sin_cache: cute.Tensor,  # (max_position_embeddings, pe_dim)
    scale: cute.Tensor,  # (1,)
    q_lora_dim: cutlass.Constexpr,
    pe_dim: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    token_idx, head_idx, block_kind = cute.arch.block_idx()
    scale_value = scale[0]
    q_lora_tiles: cutlass.Constexpr = q_lora_dim // 256
    ql_nope_paired = cute.logical_divide(ql_nope, (1, 1, 2))
    q_out_paired = cute.logical_divide(q_out, (1, 1, 2))
    half_pe_dim: cutlass.Constexpr = pe_dim // 2

    if block_kind < q_lora_tiles:
        q_pair_idx = block_kind * 128 + tidx
        in_scratch = cute.make_rmem_tensor(2, dtype=cutlass.BFloat16)
        out_scratch = cute.make_rmem_tensor(2, dtype=cutlass.Uint8)
        cute.autovec_copy(
            ql_nope_paired[token_idx, head_idx, (None, q_pair_idx)],
            in_scratch,
        )
        for i in cutlass.range_constexpr(2):
            q_val = in_scratch[i].to(cutlass.Float32)
            out_scratch[i] = cutlass.Uint8(
                _cvt_f32_to_e4m3(q_val / scale_value) & cutlass.Uint32(0xFF)
            )
        cute.autovec_copy(
            out_scratch,
            q_out_paired[token_idx, head_idx, (None, q_pair_idx)],
        )
    elif tidx < half_pe_dim:
        pos = positions[token_idx]
        cos = cos_sin_cache[pos, tidx]
        sin = cos_sin_cache[pos, tidx + half_pe_dim]
        in_scratch = cute.make_rmem_tensor(2, dtype=cutlass.BFloat16)
        out_scratch = cute.make_rmem_tensor(2, dtype=cutlass.Uint8)
        cute.autovec_copy(q_pe[token_idx, head_idx, (None, tidx)], in_scratch)
        a = in_scratch[0]
        b = in_scratch[1]
        qx = (a * cos - b * sin).to(cutlass.BFloat16)
        qy = (a * sin + b * cos).to(cutlass.BFloat16)
        out_scratch[0] = cutlass.Uint8(
            _cvt_f32_to_e4m3(qx.to(cutlass.Float32) / scale_value)
            & cutlass.Uint32(0xFF)
        )
        out_scratch[1] = cutlass.Uint8(
            _cvt_f32_to_e4m3(qy.to(cutlass.Float32) / scale_value)
            & cutlass.Uint32(0xFF)
        )
        cute.autovec_copy(
            out_scratch,
            q_out_paired[token_idx, head_idx, (None, q_lora_dim // 2 + tidx)],
        )


@cute.jit
def kimik25_decode_rope_concat_quant_fp8(
    positions: cute.Tensor,
    ql_nope: cute.Tensor,
    q_pe: cute.Tensor,
    q_out: cute.Tensor,
    cos_sin_cache: cute.Tensor,
    scale: cute.Tensor,
    q_lora_dim: cutlass.Constexpr,
    pe_dim: cutlass.Constexpr,
    stream: CUstream,
):
    sp = positions.shape[0]
    ql_nope = cute.make_tensor(
        ql_nope.iterator,
        cute.make_layout(
            (sp, ql_nope.shape[1], q_lora_dim),
            stride=(
                cute.assume(ql_nope.stride[0], divby=2),
                cute.assume(ql_nope.stride[1], divby=2),
                1,
            ),
        ),
    )
    q_pe = cute.make_tensor(
        q_pe.iterator,
        cute.make_layout(
            (sp, q_pe.shape[1], (2, pe_dim // 2)),
            stride=(
                cute.assume(q_pe.stride[0], divby=2),
                cute.assume(q_pe.stride[1], divby=2),
                (1, 2),
            ),
        ),
    )
    q_out = cute.make_tensor(
        q_out.iterator,
        cute.make_layout(
            (sp, q_out.shape[1], q_lora_dim + pe_dim),
            stride=(
                cute.assume(q_out.stride[0], divby=2),
                cute.assume(q_out.stride[1], divby=2),
                1,
            ),
        ),
    )
    q_lora_tiles: cutlass.Constexpr = q_lora_dim // 256
    kimik25_decode_rope_concat_quant_fp8_kernel(
        positions,
        ql_nope,
        q_pe,
        q_out,
        cos_sin_cache,
        scale,
        q_lora_dim,
        pe_dim,
    ).launch(
        grid=(sp, ql_nope.shape[1], q_lora_tiles + 1),
        block=(128, 1, 1),
        stream=stream,
    )


@cute.kernel
def kimik25_decode_rope_concat_quant_fp8_and_cache_mla_kernel(
    positions: cute.Tensor,  # (B,)
    ql_nope: cute.Tensor,  # (B, N, q_lora_dim)
    q_pe: cute.Tensor,  # (B, N, (2, pe_dim // 2))
    q_out: cute.Tensor,  # uint8 bytes, (B, N, q_lora_dim + pe_dim)
    cos_sin_cache: cute.Tensor,  # (max_position_embeddings, pe_dim)
    q_scale: cute.Tensor,  # (1,)
    kv_c: cute.Tensor,  # (Sp, kv_lora_rank)
    k_pe: cute.Tensor,  # (Sp, pe_dim)
    kv_cache: cute.Tensor,  # (num_blocks, block_size, kv_lora_rank + pe_dim)
    slot_mapping: cute.Tensor,  # (Sp,)
    kv_scale: cute.Tensor,  # (1,)
    q_lora_dim: cutlass.Constexpr,
    kv_lora_rank: cutlass.Constexpr,
    pe_dim: cutlass.Constexpr,
    kv_cache_block_factor: cutlass.Constexpr,
):
    tid, _, _ = cute.arch.thread_idx()
    linear_block, _, _ = cute.arch.block_idx()

    kv_cache_splits: cutlass.Constexpr = kv_cache_block_factor + 1
    num_cache_blocks = slot_mapping.shape[0] * kv_cache_splits

    if linear_block < num_cache_blocks:
        token_idx = linear_block // kv_cache_splits
        split_idx = linear_block % kv_cache_splits
        kv_c_elems_per_split: cutlass.Constexpr = kv_lora_rank // kv_cache_block_factor

        slot_idx = slot_mapping[token_idx]
        if slot_idx >= 0:
            block_size = kv_cache.shape[1]
            cache_block_idx = slot_idx // block_size
            cache_block_offset = slot_idx % block_size
            scale_value = kv_scale[0].to(cutlass.Float32)

            if split_idx > 0:
                kv_c_idx = (split_idx - 1) * kv_c_elems_per_split + tid
                kv_c_val = kv_c[token_idx, kv_c_idx].to(cutlass.Float32)
                kv_cache[cache_block_idx, cache_block_offset, kv_c_idx] = cutlass.Uint8(
                    _cvt_f32_to_e4m3(kv_c_val / scale_value) & cutlass.Uint32(0xFF)
                )
            else:
                if tid < pe_dim:
                    k_pe_val = k_pe[token_idx, tid].to(cutlass.Float32)
                    kv_cache[
                        cache_block_idx,
                        cache_block_offset,
                        kv_lora_rank + tid,
                    ] = cutlass.Uint8(
                        _cvt_f32_to_e4m3(k_pe_val / scale_value) & cutlass.Uint32(0xFF)
                    )
    else:
        decode_block = linear_block - num_cache_blocks
        q_lora_tiles: cutlass.Constexpr = q_lora_dim // 256
        decode_tile_count: cutlass.Constexpr = q_lora_tiles + 1
        block_kind = decode_block % decode_tile_count
        token_head_block = decode_block // decode_tile_count
        token_idx = token_head_block // ql_nope.shape[1]
        head_idx = token_head_block % ql_nope.shape[1]

        scale_value = q_scale[0]
        ql_nope_paired = cute.logical_divide(ql_nope, (1, 1, 2))
        q_out_paired = cute.logical_divide(q_out, (1, 1, 2))
        half_pe_dim: cutlass.Constexpr = pe_dim // 2

        if block_kind < q_lora_tiles:
            q_pair_idx = block_kind * 128 + tid
            in_scratch = cute.make_rmem_tensor(2, dtype=cutlass.BFloat16)
            out_scratch = cute.make_rmem_tensor(2, dtype=cutlass.Uint8)
            cute.autovec_copy(
                ql_nope_paired[token_idx, head_idx, (None, q_pair_idx)],
                in_scratch,
            )
            for i in cutlass.range_constexpr(2):
                q_val = in_scratch[i].to(cutlass.Float32)
                out_scratch[i] = cutlass.Uint8(
                    _cvt_f32_to_e4m3(q_val / scale_value) & cutlass.Uint32(0xFF)
                )
            cute.autovec_copy(
                out_scratch,
                q_out_paired[token_idx, head_idx, (None, q_pair_idx)],
            )
        elif tid < half_pe_dim:
            pos = positions[token_idx]
            cos = cos_sin_cache[pos, tid]
            sin = cos_sin_cache[pos, tid + half_pe_dim]
            in_scratch = cute.make_rmem_tensor(2, dtype=cutlass.BFloat16)
            out_scratch = cute.make_rmem_tensor(2, dtype=cutlass.Uint8)
            cute.autovec_copy(q_pe[token_idx, head_idx, (None, tid)], in_scratch)
            a = in_scratch[0]
            b = in_scratch[1]
            qx = (a * cos - b * sin).to(cutlass.BFloat16)
            qy = (a * sin + b * cos).to(cutlass.BFloat16)
            out_scratch[0] = cutlass.Uint8(
                _cvt_f32_to_e4m3(qx.to(cutlass.Float32) / scale_value)
                & cutlass.Uint32(0xFF)
            )
            out_scratch[1] = cutlass.Uint8(
                _cvt_f32_to_e4m3(qy.to(cutlass.Float32) / scale_value)
                & cutlass.Uint32(0xFF)
            )
            cute.autovec_copy(
                out_scratch,
                q_out_paired[token_idx, head_idx, (None, q_lora_dim // 2 + tid)],
            )


@cute.jit
def kimik25_decode_rope_concat_quant_fp8_and_cache_mla(
    positions: cute.Tensor,
    ql_nope: cute.Tensor,
    q_pe: cute.Tensor,
    q_out: cute.Tensor,
    cos_sin_cache: cute.Tensor,
    q_scale: cute.Tensor,
    kv_c: cute.Tensor,
    k_pe: cute.Tensor,
    kv_cache: cute.Tensor,
    slot_mapping: cute.Tensor,
    kv_scale: cute.Tensor,
    q_lora_dim: cutlass.Constexpr,
    kv_lora_rank: cutlass.Constexpr,
    pe_dim: cutlass.Constexpr,
    kv_cache_block_factor: cutlass.Constexpr,
    stream: CUstream,
):
    assert q_lora_dim == 512
    assert kv_lora_rank == 512
    assert pe_dim == 64
    assert pe_dim % 2 == 0
    assert kv_cache_block_factor > 0
    assert kv_lora_rank % kv_cache_block_factor == 0
    assert kv_lora_rank // kv_cache_block_factor == 128
    assert q_lora_dim % 256 == 0
    assert ql_nope.stride[2] == 1
    assert q_pe.stride[2] == 1
    assert kv_c.stride[1] == 1
    assert k_pe.stride[1] == 1
    assert kv_cache.stride[2] == 1

    sp = positions.shape[0]
    ql_nope = cute.make_tensor(
        ql_nope.iterator,
        cute.make_layout(
            (sp, ql_nope.shape[1], q_lora_dim),
            stride=(
                cute.assume(ql_nope.stride[0], divby=2),
                cute.assume(ql_nope.stride[1], divby=2),
                1,
            ),
        ),
    )
    q_pe = cute.make_tensor(
        q_pe.iterator,
        cute.make_layout(
            (sp, q_pe.shape[1], (2, pe_dim // 2)),
            stride=(
                cute.assume(q_pe.stride[0], divby=2),
                cute.assume(q_pe.stride[1], divby=2),
                (1, 2),
            ),
        ),
    )
    q_out = cute.make_tensor(
        q_out.iterator,
        cute.make_layout(
            (sp, q_out.shape[1], q_lora_dim + pe_dim),
            stride=(
                cute.assume(q_out.stride[0], divby=2),
                cute.assume(q_out.stride[1], divby=2),
                1,
            ),
        ),
    )
    kv_c = cute.make_tensor(
        kv_c.iterator,
        cute.make_layout(
            (kv_c.shape[0], kv_lora_rank),
            stride=(kv_c.stride[0], 1),
        ),
    )
    k_pe = cute.make_tensor(
        k_pe.iterator,
        cute.make_layout(
            (k_pe.shape[0], pe_dim),
            stride=(k_pe.stride[0], 1),
        ),
    )
    kv_cache = cute.make_tensor(
        kv_cache.iterator,
        cute.make_layout(
            (kv_cache.shape[0], kv_cache.shape[1], kv_lora_rank + pe_dim),
            stride=(kv_cache.stride[0], kv_cache.stride[1], 1),
        ),
    )

    q_lora_tiles: cutlass.Constexpr = q_lora_dim // 256
    decode_blocks = sp * ql_nope.shape[1] * (q_lora_tiles + 1)
    cache_blocks = slot_mapping.shape[0] * (kv_cache_block_factor + 1)
    kimik25_decode_rope_concat_quant_fp8_and_cache_mla_kernel(
        positions,
        ql_nope,
        q_pe,
        q_out,
        cos_sin_cache,
        q_scale,
        kv_c,
        k_pe,
        kv_cache,
        slot_mapping,
        kv_scale,
        q_lora_dim,
        kv_lora_rank,
        pe_dim,
        kv_cache_block_factor,
    ).launch(
        grid=(cache_blocks + decode_blocks, 1, 1),
        block=(128, 1, 1),
        stream=stream,
    )


@cute.kernel
def kimi_fused_rmsnorm_kernel(
    data: cute.Tensor,  # (Sp, (2, lora_dim_kv // 2, 4))
    weights_q: cute.Tensor,  # (2, lora_dim_q // 2)
    weights_kv: cute.Tensor,  # (2, lora_dim_kv // 2)
    lora_dim_q: cutlass.Constexpr,  # must be lora_dim_kv * 3
    lora_dim_kv: cutlass.Constexpr,
    eps_q: cutlass.Constexpr,
    eps_kv: cutlass.Constexpr,
):
    nwarps = lora_dim_kv // 64
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
        w0 = weights_q[None, tid].load()
        w1 = weights_q[None, tid + lora_dim_kv // 2].load()
        w2 = weights_q[None, tid + lora_dim_kv].load()
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
            sdata[0] = cute.math.rsqrt(ssum / lora_dim_q + eps_q)

        cute.arch.sync_threads()
        invnorm = sdata[0]
        data[bid, (None, tid, 0)] = (x0 * invnorm).to(cutlass.BFloat16) * w0
        data[bid, (None, tid, 1)] = (x1 * invnorm).to(cutlass.BFloat16) * w1
        data[bid, (None, tid, 2)] = (x2 * invnorm).to(cutlass.BFloat16) * w2
    else:
        x3 = data[bid - Sp, (None, tid, 3)].load().to(cutlass.Float32)
        w3 = weights_kv[None, tid].load()
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
        data[bid - Sp, (None, tid, 3)] = (x3 * invnorm).to(cutlass.BFloat16) * w3


@cute.jit
def kimi_fused_rmsnorm(
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
    weights_q = cute.make_tensor(
        weights_q.iterator, cute.make_layout((2, lora_dim_q // 2))
    )
    weights_kv = cute.make_tensor(
        weights_kv.iterator, cute.make_layout((2, lora_dim_kv // 2))
    )
    grid = (data.shape[0] * 2, 1, 1)
    block = (lora_dim_kv // 2, 1, 1)
    kimi_fused_rmsnorm_kernel(
        data,
        weights_q,
        weights_kv,
        lora_dim_q,
        lora_dim_kv,
        eps_q,
        eps_kv,
    ).launch(grid=grid, block=block, stream=stream)


def _run_kimik25_concat_and_cache_mla(
    *,
    kv_c: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    scale: torch.Tensor,
) -> None:
    assert kv_cache_dtype in {"fp8", "fp8_e4m3"}

    kv_lora_rank = kv_c.shape[1]
    pe_dim = k_pe.shape[1]
    assert kv_lora_rank == 512, "Kimi-K2.5 NVFP4 expects kv_lora_rank=512"
    assert pe_dim == 64, "Kimi-K2.5 NVFP4 expects qk_rope_head_dim=64"
    kv_cache_block_factor = 4
    scale = scale.view(1)

    kv_c_cute = _make_dynamic_cute_tensor(kv_c)
    k_pe_cute = _make_dynamic_cute_tensor(k_pe)
    kv_cache_cute = _make_dynamic_cute_tensor(kv_cache)
    slot_mapping_cute = _make_fully_dynamic_cute_tensor(slot_mapping)
    scale_cute = from_dlpack(scale, assumed_align=4)
    cache_key = (
        "kimik25_concat_and_cache_mla",
        _cuda_device_cache_key(),
        _cutedsl_arg_cache_key(kv_c_cute),
        _cutedsl_arg_cache_key(k_pe_cute),
        _cutedsl_arg_cache_key(kv_cache_cute),
        _cutedsl_arg_cache_key(slot_mapping_cute),
        _cutedsl_arg_cache_key(scale_cute),
        kv_lora_rank,
        pe_dim,
        kv_cache_block_factor,
    )
    executor = _get_cutedsl_executor(
        cache_key,
        kimik25_concat_and_cache_mla,
        kv_c=kv_c_cute,
        k_pe=k_pe_cute,
        kv_cache=kv_cache_cute,
        slot_mapping=slot_mapping_cute,
        scale=scale_cute,
        kv_lora_rank=kv_lora_rank,
        pe_dim=pe_dim,
        kv_cache_block_factor=kv_cache_block_factor,
        stream=cutlass_torch.current_stream(),
    )
    executor(
        kv_c=kv_c_cute,
        k_pe=k_pe_cute,
        kv_cache=kv_cache_cute,
        slot_mapping=slot_mapping_cute,
        scale=scale_cute,
        stream=cutlass_torch.current_stream(),
    )


def _run_kimik25_rmsnorm_special_qkv_fused(
    *,
    data: torch.Tensor,
    positions: torch.Tensor,
    k_pe: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    weights_q: torch.Tensor,
    weights_kv: torch.Tensor,
    lora_dim_q: int,
    lora_dim_kv: int,
    pe_dim: int,
    eps_q: float,
    eps_kv: float,
) -> None:
    data_cute = _make_dynamic_cute_tensor(data)
    positions_cute = _make_fully_dynamic_cute_tensor(positions)
    k_pe_cute = _make_dynamic_cute_tensor(k_pe)
    cos_sin_cache_cute = from_dlpack(cos_sin_cache, assumed_align=16)
    weights_q_cute = from_dlpack(weights_q, assumed_align=16)
    weights_kv_cute = from_dlpack(weights_kv, assumed_align=16)
    cache_key = (
        "kimik25_rmsnorm_special_qkv_fused",
        _cuda_device_cache_key(),
        _cutedsl_arg_cache_key(data_cute),
        _cutedsl_arg_cache_key(positions_cute),
        _cutedsl_arg_cache_key(k_pe_cute),
        _cutedsl_arg_cache_key(cos_sin_cache_cute),
        _cutedsl_arg_cache_key(weights_q_cute),
        _cutedsl_arg_cache_key(weights_kv_cute),
        lora_dim_q,
        lora_dim_kv,
        pe_dim,
        float(eps_q),
        float(eps_kv),
    )
    executor = _get_cutedsl_executor(
        cache_key,
        kimik25_rmsnorm_special_qkv_fused,
        data=data_cute,
        positions=positions_cute,
        k_pe=k_pe_cute,
        cos_sin_cache=cos_sin_cache_cute,
        weights_q=weights_q_cute,
        weights_kv=weights_kv_cute,
        lora_dim_q=lora_dim_q,
        lora_dim_kv=lora_dim_kv,
        pe_dim=pe_dim,
        eps_q=eps_q,
        eps_kv=eps_kv,
        stream=cutlass_torch.current_stream(),
    )
    executor(
        data=data_cute,
        positions=positions_cute,
        k_pe=k_pe_cute,
        cos_sin_cache=cos_sin_cache_cute,
        weights_q=weights_q_cute,
        weights_kv=weights_kv_cute,
        stream=cutlass_torch.current_stream(),
    )


def _run_kimik25_rope(
    positions: torch.Tensor,
    query: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    num_local_heads: int,
    half_rope_dim: int,
) -> None:
    positions_cute = _make_fully_dynamic_cute_tensor(positions)
    query_cute = _make_fully_dynamic_cute_tensor(query)
    cos_sin_cache_cute = from_dlpack(cos_sin_cache, assumed_align=16)
    cache_key = (
        "kimik25_rope",
        _cuda_device_cache_key(),
        _cutedsl_arg_cache_key(positions_cute),
        _cutedsl_arg_cache_key(query_cute),
        _cutedsl_arg_cache_key(cos_sin_cache_cute),
        num_local_heads,
        half_rope_dim,
    )
    executor = _get_cutedsl_executor(
        cache_key,
        kimik25_rope,
        positions=positions_cute,
        query=query_cute,
        cos_sin_cache=cos_sin_cache_cute,
        N_local=num_local_heads,
        half_rope_dim=half_rope_dim,
        stream=cutlass_torch.current_stream(),
    )
    executor(
        positions=positions_cute,
        query=query_cute,
        cos_sin_cache=cos_sin_cache_cute,
        stream=cutlass_torch.current_stream(),
    )


def _run_kimik25_decode_rope_concat_quant_fp8(
    *,
    positions: torch.Tensor,
    ql_nope: torch.Tensor,
    q_pe: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    q_lora_dim = ql_nope.shape[2]
    pe_dim = q_pe.shape[2]
    assert q_lora_dim == 512, "Kimi-K2.5 NVFP4 expects kv_lora_rank=512"
    assert pe_dim == 64, "Kimi-K2.5 NVFP4 expects qk_rope_head_dim=64"
    assert ql_nope.shape[:2] == q_pe.shape[:2]

    scale = scale.view(1)
    q_out = torch.empty(
        (ql_nope.shape[0], ql_nope.shape[1], q_lora_dim + pe_dim),
        device=ql_nope.device,
        dtype=torch.uint8,
    )

    positions_cute = _make_fully_dynamic_cute_tensor(positions)
    ql_nope_cute = _make_dynamic_cute_tensor(ql_nope)
    q_pe_cute = _make_fully_dynamic_cute_tensor(q_pe)
    q_out_cute = _make_dynamic_cute_tensor(q_out)
    cos_sin_cache_cute = from_dlpack(cos_sin_cache, assumed_align=16)
    scale_cute = from_dlpack(scale, assumed_align=4)
    cache_key = (
        "kimik25_decode_rope_concat_quant_fp8",
        _cuda_device_cache_key(),
        _cutedsl_arg_cache_key(positions_cute),
        _cutedsl_arg_cache_key(ql_nope_cute),
        _cutedsl_arg_cache_key(q_pe_cute),
        _cutedsl_arg_cache_key(q_out_cute),
        _cutedsl_arg_cache_key(cos_sin_cache_cute),
        _cutedsl_arg_cache_key(scale_cute),
        q_lora_dim,
        pe_dim,
    )
    executor = _get_cutedsl_executor(
        cache_key,
        kimik25_decode_rope_concat_quant_fp8,
        positions=positions_cute,
        ql_nope=ql_nope_cute,
        q_pe=q_pe_cute,
        q_out=q_out_cute,
        cos_sin_cache=cos_sin_cache_cute,
        scale=scale_cute,
        q_lora_dim=q_lora_dim,
        pe_dim=pe_dim,
        stream=cutlass_torch.current_stream(),
    )
    executor(
        positions=positions_cute,
        ql_nope=ql_nope_cute,
        q_pe=q_pe_cute,
        q_out=q_out_cute,
        cos_sin_cache=cos_sin_cache_cute,
        scale=scale_cute,
        stream=cutlass_torch.current_stream(),
    )
    return q_out.view(current_platform.fp8_dtype())


def _run_kimik25_decode_rope_concat_quant_fp8_and_cache_mla(
    *,
    positions: torch.Tensor,
    ql_nope: torch.Tensor,
    q_pe: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    q_scale: torch.Tensor,
    kv_c: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    kv_scale: torch.Tensor,
) -> torch.Tensor:
    assert kv_cache_dtype in {"fp8", "fp8_e4m3"}

    q_lora_dim = ql_nope.shape[2]
    kv_lora_rank = kv_c.shape[1]
    pe_dim = q_pe.shape[2]
    assert q_lora_dim == 512, "Kimi-K2.5 NVFP4 expects kv_lora_rank=512"
    assert kv_lora_rank == 512, "Kimi-K2.5 NVFP4 expects kv_lora_rank=512"
    assert pe_dim == 64, "Kimi-K2.5 NVFP4 expects qk_rope_head_dim=64"
    assert k_pe.shape[1] == pe_dim
    assert ql_nope.shape[:2] == q_pe.shape[:2]

    kv_cache_block_factor = 4
    q_scale = q_scale.view(1)
    kv_scale = kv_scale.view(1)
    q_out = torch.empty(
        (ql_nope.shape[0], ql_nope.shape[1], q_lora_dim + pe_dim),
        device=ql_nope.device,
        dtype=torch.uint8,
    )

    positions_cute = _make_fully_dynamic_cute_tensor(positions)
    ql_nope_cute = _make_dynamic_cute_tensor(ql_nope)
    q_pe_cute = _make_fully_dynamic_cute_tensor(q_pe)
    q_out_cute = _make_dynamic_cute_tensor(q_out)
    cos_sin_cache_cute = from_dlpack(cos_sin_cache, assumed_align=16)
    q_scale_cute = from_dlpack(q_scale, assumed_align=4)
    kv_c_cute = _make_dynamic_cute_tensor(kv_c)
    k_pe_cute = _make_dynamic_cute_tensor(k_pe)
    kv_cache_cute = _make_dynamic_cute_tensor(kv_cache)
    slot_mapping_cute = _make_fully_dynamic_cute_tensor(slot_mapping)
    kv_scale_cute = from_dlpack(kv_scale, assumed_align=4)
    cache_key = (
        "kimik25_decode_rope_concat_quant_fp8_and_cache_mla",
        _cuda_device_cache_key(),
        _cutedsl_arg_cache_key(positions_cute),
        _cutedsl_arg_cache_key(ql_nope_cute),
        _cutedsl_arg_cache_key(q_pe_cute),
        _cutedsl_arg_cache_key(q_out_cute),
        _cutedsl_arg_cache_key(cos_sin_cache_cute),
        _cutedsl_arg_cache_key(q_scale_cute),
        _cutedsl_arg_cache_key(kv_c_cute),
        _cutedsl_arg_cache_key(k_pe_cute),
        _cutedsl_arg_cache_key(kv_cache_cute),
        _cutedsl_arg_cache_key(slot_mapping_cute),
        _cutedsl_arg_cache_key(kv_scale_cute),
        q_lora_dim,
        kv_lora_rank,
        pe_dim,
        kv_cache_block_factor,
    )
    executor = _get_cutedsl_executor(
        cache_key,
        kimik25_decode_rope_concat_quant_fp8_and_cache_mla,
        positions=positions_cute,
        ql_nope=ql_nope_cute,
        q_pe=q_pe_cute,
        q_out=q_out_cute,
        cos_sin_cache=cos_sin_cache_cute,
        q_scale=q_scale_cute,
        kv_c=kv_c_cute,
        k_pe=k_pe_cute,
        kv_cache=kv_cache_cute,
        slot_mapping=slot_mapping_cute,
        kv_scale=kv_scale_cute,
        q_lora_dim=q_lora_dim,
        kv_lora_rank=kv_lora_rank,
        pe_dim=pe_dim,
        kv_cache_block_factor=kv_cache_block_factor,
        stream=cutlass_torch.current_stream(),
    )
    executor(
        positions=positions_cute,
        ql_nope=ql_nope_cute,
        q_pe=q_pe_cute,
        q_out=q_out_cute,
        cos_sin_cache=cos_sin_cache_cute,
        q_scale=q_scale_cute,
        kv_c=kv_c_cute,
        k_pe=k_pe_cute,
        kv_cache=kv_cache_cute,
        slot_mapping=slot_mapping_cute,
        kv_scale=kv_scale_cute,
        stream=cutlass_torch.current_stream(),
    )
    return q_out.view(current_platform.fp8_dtype())
