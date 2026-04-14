# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Microbenchmark the Kimi-K2.5 RoPE kernel.

This benchmark isolates just the in-place rotary embedding step that operates on
the same MLA-style views as the Kimi NVFP4 model:

    query_base: [num_tokens, num_local_heads, qk_nope_head_dim + qk_rope_head_dim]
    query:      query_base[..., qk_nope_head_dim:]

    key_base:   [num_tokens, kv_lora_rank + qk_rope_head_dim]
    key:        key_base[..., kv_lora_rank:].unsqueeze(1)

The benchmark is registry-based so new kernel versions can be added without
changing the harness. It currently includes:

1. The original CuTeDSL kernel duplicated locally in this benchmark
2. A Triton implementation of the same in-place RoPE math
3. The exact TorchInductor-generated fused Triton query kernel from Kimi-K2.5

Example:

    .venv/bin/python benchmarks/kernels/benchmark_kimi_k25_nvfp4_rope.py \
        --versions cutedsl triton --num-tokens 13 64 512 4096 \
        --check-correctness

    .venv/bin/python benchmarks/kernels/benchmark_kimi_k25_nvfp4_rope.py \
        --versions triton --num-tokens 4096 --use-cudagraph

    .venv/bin/python benchmarks/kernels/benchmark_kimi_k25_nvfp4_rope.py \
        --versions cutedsl triton --num-tokens 4096 \
        --dump-kernel-artifacts-dir /tmp/kimi_rope_artifacts
"""

import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.torch_utils import set_random_seed

QK_NOPE_HEAD_DIM = 128
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
NUM_LOCAL_HEADS = 64
MAX_POSITION = 262144
DTYPE = torch.bfloat16
EXACT_FUSED_QUERY_XBLOCK = 512
EXACT_FUSED_QUERY_HEAD_DIM = 192
EXACT_FUSED_QUERY_BLOCKS_PER_TOKEN = (
    NUM_LOCAL_HEADS * EXACT_FUSED_QUERY_HEAD_DIM // EXACT_FUSED_QUERY_XBLOCK
)
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
    qk_nope_head_dim: int = QK_NOPE_HEAD_DIM
    kv_lora_rank: int = KV_LORA_RANK
    qk_rope_head_dim: int = QK_ROPE_HEAD_DIM
    max_position_embeddings: int = MAX_POSITION
    dtype: torch.dtype = DTYPE

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def half_rope_dim(self) -> int:
        return self.qk_rope_head_dim // 2

    @property
    def bytes_per_call(self) -> int:
        elem_size = torch.tensor([], dtype=self.dtype).element_size()
        query_elems = self.num_tokens * self.num_local_heads * self.qk_rope_head_dim
        key_elems = self.num_tokens * self.qk_rope_head_dim
        cache_elems = self.num_tokens * (self.num_local_heads + 1) * self.qk_rope_head_dim
        position_bytes = self.num_tokens * torch.tensor([], dtype=torch.long).element_size()
        return (2 * (query_elems + key_elems) + cache_elems) * elem_size + position_bytes

    @property
    def fused_query_only_bytes_per_call(self) -> int:
        elem_size = torch.tensor([], dtype=self.dtype).element_size()
        query_base_elems = self.num_tokens * self.num_local_heads * self.qk_head_dim
        cache_elems = self.num_tokens * self.num_local_heads * self.qk_rope_head_dim
        position_bytes = self.num_tokens * torch.tensor([], dtype=torch.long).element_size()
        return (2 * query_base_elems + cache_elems) * elem_size + position_bytes


@dataclass
class RopeInputs:
    positions: torch.Tensor
    query_base: torch.Tensor
    query: torch.Tensor
    key: torch.Tensor
    cos_sin_cache: torch.Tensor
    query_out_base: torch.Tensor | None = None


@dataclass
class KernelVersion:
    name: str
    make_runner: Any
    run_once: Any
    dump_artifacts: Any
    get_outputs: Any
    needs_query_output_base: bool = False
    check_key_output: bool = True
    bytes_per_call: Any | None = None


_KERNEL_VERSION_CACHE: dict[str, KernelVersion] = {}


def _default_version_outputs(
    inputs: RopeInputs,
    cfg: BenchmarkConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    return inputs.query, inputs.key


def _default_bytes_per_call(cfg: BenchmarkConfig) -> int:
    return cfg.bytes_per_call


def _fused_query_only_bytes_per_call(cfg: BenchmarkConfig) -> int:
    return cfg.fused_query_only_bytes_per_call


def _fused_query_version_outputs(
    inputs: RopeInputs,
    cfg: BenchmarkConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    if inputs.query_out_base is None:
        raise RuntimeError("Fused-query benchmark inputs are missing query_out_base.")
    return inputs.query_out_base[..., cfg.qk_nope_head_dim :], inputs.key


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
    query_stride0: int,
    query_stride1: int,
) -> Path:
    return (
        dump_root
        / version_name
        / (
            f"tokens_{cfg.num_tokens}_heads_{cfg.num_local_heads}"
            f"_qstride_{query_stride0}_{query_stride1}"
        )
    )


def _sanitize_kernel_name(name: str) -> str:
    return name.lstrip("@").replace("/", "_")


@triton.jit
def _triton_kimik25_rope_kernel(
    positions_ptr,
    query_ptr,
    query_stride0,
    query_stride1,
    key_ptr,
    key_stride0,
    cos_sin_cache_ptr,
    cos_sin_cache_stride0,
    num_local_heads,
    HALF_ROPE_DIM: tl.constexpr,
):
    pid_token = tl.program_id(0)
    pid_head_or_key = tl.program_id(1)

    offsets = tl.arange(0, HALF_ROPE_DIM)
    mask = offsets < HALF_ROPE_DIM

    position = tl.load(positions_ptr + pid_token).to(tl.int32)
    cache_ptr = cos_sin_cache_ptr + position * cos_sin_cache_stride0
    cos = tl.load(cache_ptr + offsets, mask=mask, other=0.0)
    sin = tl.load(cache_ptr + offsets + HALF_ROPE_DIM, mask=mask, other=0.0)

    pair_offsets = offsets * 2
    if pid_head_or_key < num_local_heads:
        row_ptr = (
            query_ptr
            + pid_token * query_stride0
            + pid_head_or_key * query_stride1
        )
    else:
        row_ptr = key_ptr + pid_token * key_stride0

    a = tl.load(row_ptr + pair_offsets, mask=mask, other=0.0)
    b = tl.load(row_ptr + pair_offsets + 1, mask=mask, other=0.0)
    a_new = (a * cos - b * sin).to(query_ptr.dtype.element_ty)
    b_new = (a * sin + b * cos).to(query_ptr.dtype.element_ty)
    tl.store(row_ptr + pair_offsets, a_new, mask=mask)
    tl.store(row_ptr + pair_offsets + 1, b_new, mask=mask)


def triton_kimik25_rope(
    inputs: RopeInputs,
    cfg: BenchmarkConfig,
) -> None:
    assert inputs.query.is_cuda
    assert inputs.key.is_cuda
    assert inputs.cos_sin_cache.is_cuda
    assert inputs.positions.is_cuda
    assert inputs.query.dtype == cfg.dtype
    assert inputs.key.dtype == cfg.dtype
    assert inputs.cos_sin_cache.dtype == cfg.dtype
    assert inputs.query.shape == (
        cfg.num_tokens,
        cfg.num_local_heads,
        cfg.qk_rope_head_dim,
    )
    assert inputs.key.shape == (cfg.num_tokens, 1, cfg.qk_rope_head_dim)
    assert inputs.query.stride() == (
        cfg.num_local_heads * cfg.qk_head_dim,
        cfg.qk_head_dim,
        1,
    )
    assert inputs.key.stride() == (
        cfg.kv_lora_rank + cfg.qk_rope_head_dim,
        cfg.qk_rope_head_dim,
        1,
    )
    assert inputs.cos_sin_cache.shape == (
        cfg.max_position_embeddings,
        cfg.qk_rope_head_dim,
    )
    assert inputs.positions.shape == (cfg.num_tokens,)

    _triton_kimik25_rope_kernel[(cfg.num_tokens, cfg.num_local_heads + 1)](
        inputs.positions,
        inputs.query,
        inputs.query.stride(0),
        inputs.query.stride(1),
        inputs.key,
        inputs.key.stride(0),
        inputs.cos_sin_cache,
        inputs.cos_sin_cache.stride(0),
        cfg.num_local_heads,
        HALF_ROPE_DIM=cfg.half_rope_dim,
        num_warps=1,
    )


@triton.jit
def _triton_kimik25_key_only_rope_kernel(
    positions_ptr,
    key_ptr,
    key_stride0,
    cos_sin_cache_ptr,
    cos_sin_cache_stride0,
    max_position_embeddings,
    HALF_ROPE_DIM: tl.constexpr,
):
    pid_token = tl.program_id(0)

    offsets = tl.arange(0, HALF_ROPE_DIM)
    mask = offsets < HALF_ROPE_DIM

    position = tl.load(positions_ptr + pid_token).to(tl.int32)
    wrapped_position = tl.where(position < 0, position + max_position_embeddings,
                                position)
    tl.device_assert(((0 <= wrapped_position)
                      & (wrapped_position < max_position_embeddings)),
                     "index out of bounds: 0 <= position < max_position_embeddings")

    cache_ptr = cos_sin_cache_ptr + wrapped_position * cos_sin_cache_stride0
    cos = tl.load(cache_ptr + offsets, mask=mask, other=0.0)
    sin = tl.load(cache_ptr + offsets + HALF_ROPE_DIM, mask=mask, other=0.0)

    row_ptr = key_ptr + pid_token * key_stride0
    pair_offsets = offsets * 2
    a = tl.load(row_ptr + pair_offsets, mask=mask, other=0.0)
    b = tl.load(row_ptr + pair_offsets + 1, mask=mask, other=0.0)
    a_new = (a * cos - b * sin).to(key_ptr.dtype.element_ty)
    b_new = (a * sin + b * cos).to(key_ptr.dtype.element_ty)
    tl.store(row_ptr + pair_offsets, a_new, mask=mask)
    tl.store(row_ptr + pair_offsets + 1, b_new, mask=mask)


@triton.jit
def _triton_kimik25_fused_query_rope_kernel(
    in_ptr0,
    in_ptr1,
    in_ptr2,
    out_ptr0,
    xnumel,
    XBLOCK: tl.constexpr,
):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex % 192
    x3 = xindex
    x2 = xindex // 12288
    x4 = xindex // 192
    tmp33 = tl.load(in_ptr0 + x3, None).to(tl.float32)
    tmp0 = x0
    tmp1 = tl.full([1], 128, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + x3, tmp2, other=0.0).to(tl.float32)
    tmp4 = tl.load(in_ptr1 + x2, tmp2, eviction_policy="evict_last", other=0.0)
    tmp5 = tl.full([XBLOCK], 262144, tl.int32)
    tmp6 = tmp4 + tmp5
    tmp7 = tmp4 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp4)
    tl.device_assert(
        ((0 <= tl.broadcast_to(tmp8, [XBLOCK]))
         & (tl.broadcast_to(tmp8, [XBLOCK]) < 262144)) | ~(tmp2),
        "index out of bounds: 0 <= tl.broadcast_to(tmp8, [XBLOCK]) < 262144",
    )
    tmp10 = tl.load(
        in_ptr2 + (64 * tmp8 + (((x0 // 2) % 32))),
        tmp2,
        eviction_policy="evict_last",
        other=0.0,
    ).to(tl.float32)
    tmp11 = tmp3 * tmp10
    tmp12 = x3 % 2
    tmp13 = tl.full([1], 0, tl.int64)
    tmp14 = tmp12 >= tmp13
    tmp15 = tl.full([1], 1, tl.int64)
    tmp16 = tmp12 < tmp15
    tmp17 = tmp16 & tmp2
    tmp18 = tl.load(
        in_ptr0 + (129 + 2 * (((x0 // 2) % 32)) + 192 * x4),
        tmp17,
        eviction_policy="evict_last",
        other=0.0,
    ).to(tl.float32)
    tmp19 = -tmp18
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp17, tmp19, tmp20)
    tmp22 = tmp12 >= tmp15
    tmp23 = tl.full([1], 2, tl.int64)
    tmp24 = tmp12 < tmp23
    tmp25 = tmp22 & tmp2
    tmp26 = tl.load(
        in_ptr0 + (128 + 2 * (((x0 // 2) % 32)) + 192 * x4),
        tmp25,
        eviction_policy="evict_last",
        other=0.0,
    ).to(tl.float32)
    tmp27 = tl.where(tmp16, tmp21, tmp26)
    tmp28 = tl.load(
        in_ptr2 + (32 + 64 * tmp8 + (((x0 // 2) % 32))),
        tmp2,
        eviction_policy="evict_last",
        other=0.0,
    ).to(tl.float32)
    tmp29 = tmp27 * tmp28
    tmp30 = tmp11 + tmp29
    tmp31 = tl.full(tmp30.shape, 0.0, tmp30.dtype)
    tmp32 = tl.where(tmp2, tmp30, tmp31)
    tmp34 = tl.where(tmp2, tmp32, tmp33)
    tl.store(out_ptr0 + x3, tmp34, None)


def _validate_exact_fused_query_cfg(
    inputs: RopeInputs,
    cfg: BenchmarkConfig,
) -> None:
    if cfg.num_local_heads != NUM_LOCAL_HEADS:
        raise ValueError(
            "triton_fused_query requires "
            f"--num-local-heads={NUM_LOCAL_HEADS}, got {cfg.num_local_heads}."
        )
    if cfg.qk_nope_head_dim != QK_NOPE_HEAD_DIM:
        raise ValueError(
            "triton_fused_query requires "
            f"--qk-nope-head-dim={QK_NOPE_HEAD_DIM}, got {cfg.qk_nope_head_dim}."
        )
    if cfg.qk_rope_head_dim != QK_ROPE_HEAD_DIM:
        raise ValueError(
            "triton_fused_query requires "
            f"--qk-rope-head-dim={QK_ROPE_HEAD_DIM}, got {cfg.qk_rope_head_dim}."
        )
    if cfg.max_position_embeddings != MAX_POSITION:
        raise ValueError(
            "triton_fused_query requires "
            f"--max-position-embeddings={MAX_POSITION}, got "
            f"{cfg.max_position_embeddings}."
        )
    if inputs.query_base.stride() != (
        NUM_LOCAL_HEADS * EXACT_FUSED_QUERY_HEAD_DIM,
        EXACT_FUSED_QUERY_HEAD_DIM,
        1,
    ):
        raise ValueError(
            "triton_fused_query requires contiguous MLA query_base strides "
            f"(12288, 192, 1), got {inputs.query_base.stride()}."
        )
    if inputs.query_out_base is None:
        raise RuntimeError("Fused-query benchmark inputs are missing query_out_base.")
    if inputs.query_out_base.stride() != inputs.query_base.stride():
        raise ValueError(
            "triton_fused_query requires query_out_base to match query_base "
            f"strides, got {inputs.query_out_base.stride()} vs "
            f"{inputs.query_base.stride()}."
        )


def triton_kimik25_fused_query_rope(
    inputs: RopeInputs,
    cfg: BenchmarkConfig,
) -> None:
    _validate_exact_fused_query_cfg(inputs, cfg)
    assert inputs.query_base.is_cuda
    assert inputs.query.is_cuda
    assert inputs.key.is_cuda
    assert inputs.cos_sin_cache.is_cuda
    assert inputs.positions.is_cuda
    assert inputs.query_base.dtype == cfg.dtype
    assert inputs.query.dtype == cfg.dtype
    assert inputs.key.dtype == cfg.dtype
    assert inputs.cos_sin_cache.dtype == cfg.dtype
    assert inputs.query_base.shape == (
        cfg.num_tokens,
        cfg.num_local_heads,
        cfg.qk_head_dim,
    )
    assert inputs.query.shape == (
        cfg.num_tokens,
        cfg.num_local_heads,
        cfg.qk_rope_head_dim,
    )
    assert inputs.query.stride() == (
        cfg.num_local_heads * cfg.qk_head_dim,
        cfg.qk_head_dim,
        1,
    )
    assert inputs.query_out_base.shape == inputs.query_base.shape
    assert inputs.key.shape == (cfg.num_tokens, 1, cfg.qk_rope_head_dim)
    assert inputs.key.stride() == (
        cfg.kv_lora_rank + cfg.qk_rope_head_dim,
        cfg.qk_rope_head_dim,
        1,
    )
    assert inputs.cos_sin_cache.shape == (
        cfg.max_position_embeddings,
        cfg.qk_rope_head_dim,
    )
    assert inputs.positions.shape == (cfg.num_tokens,)
    _triton_kimik25_fused_query_rope_kernel[(cfg.num_tokens *
                                             EXACT_FUSED_QUERY_BLOCKS_PER_TOKEN,)](
        inputs.query_base,
        inputs.positions,
        inputs.cos_sin_cache,
        inputs.query_out_base,
        cfg.num_tokens * cfg.num_local_heads * cfg.qk_head_dim,
        XBLOCK=EXACT_FUSED_QUERY_XBLOCK,
        num_warps=8,
        num_stages=1,
    )


def _get_triton_compiled_kernel(
    inputs: RopeInputs,
    cfg: BenchmarkConfig,
):
    compiled = _triton_kimik25_rope_kernel.warmup(
        inputs.positions,
        inputs.query,
        inputs.query.stride(0),
        inputs.query.stride(1),
        inputs.key,
        inputs.key.stride(0),
        inputs.cos_sin_cache,
        inputs.cos_sin_cache.stride(0),
        cfg.num_local_heads,
        grid=(cfg.num_tokens, cfg.num_local_heads + 1),
        HALF_ROPE_DIM=cfg.half_rope_dim,
        num_warps=1,
    )
    if hasattr(compiled, "result"):
        compiled = compiled.result()
    return compiled


def _get_triton_key_only_compiled_kernel(
    inputs: RopeInputs,
    cfg: BenchmarkConfig,
):
    compiled = _triton_kimik25_key_only_rope_kernel.warmup(
        inputs.positions,
        inputs.key,
        inputs.key.stride(0),
        inputs.cos_sin_cache,
        inputs.cos_sin_cache.stride(0),
        cfg.max_position_embeddings,
        grid=(cfg.num_tokens,),
        HALF_ROPE_DIM=cfg.half_rope_dim,
        num_warps=1,
    )
    if hasattr(compiled, "result"):
        compiled = compiled.result()
    return compiled


def _get_triton_fused_query_compiled_kernel(
    inputs: RopeInputs,
    cfg: BenchmarkConfig,
):
    _validate_exact_fused_query_cfg(inputs, cfg)
    compiled = _triton_kimik25_fused_query_rope_kernel.warmup(
        inputs.query_base,
        inputs.positions,
        inputs.cos_sin_cache,
        inputs.query_out_base,
        cfg.num_tokens * cfg.num_local_heads * cfg.qk_head_dim,
        grid=(cfg.num_tokens * EXACT_FUSED_QUERY_BLOCKS_PER_TOKEN,),
        XBLOCK=EXACT_FUSED_QUERY_XBLOCK,
        num_warps=8,
        num_stages=1,
    )
    if hasattr(compiled, "result"):
        compiled = compiled.result()
    return compiled


def _dump_triton_artifacts(
    output_dir: Path,
    inputs: RopeInputs,
    cfg: BenchmarkConfig,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    compiled = _get_triton_compiled_kernel(inputs, cfg)
    kernel_name = _sanitize_kernel_name(compiled.name)
    _write_text_file(output_dir / f"{kernel_name}.ptx", compiled.asm["ptx"])
    _write_binary_file(output_dir / f"{kernel_name}.cubin", compiled.asm["cubin"])
    _write_text_file(
        output_dir / f"{kernel_name}.sass",
        _get_sass_from_cubin(compiled.asm["cubin"]),
    )


def _dump_triton_fused_query_artifacts(
    output_dir: Path,
    inputs: RopeInputs,
    cfg: BenchmarkConfig,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    fused_query_compiled = _get_triton_fused_query_compiled_kernel(inputs, cfg)
    fused_query_kernel_name = _sanitize_kernel_name(fused_query_compiled.name)
    _write_text_file(
        output_dir / f"{fused_query_kernel_name}.ptx",
        fused_query_compiled.asm["ptx"],
    )
    _write_binary_file(
        output_dir / f"{fused_query_kernel_name}.cubin",
        fused_query_compiled.asm["cubin"],
    )
    _write_text_file(
        output_dir / f"{fused_query_kernel_name}.sass",
        _get_sass_from_cubin(fused_query_compiled.asm["cubin"]),
    )


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


def _make_mla_like_qk_views(
    cfg: BenchmarkConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    query_base = torch.randn(
        cfg.num_tokens,
        cfg.num_local_heads,
        cfg.qk_head_dim,
        device="cuda",
        dtype=cfg.dtype,
    )
    key_base = torch.randn(
        cfg.num_tokens,
        cfg.kv_lora_rank + cfg.qk_rope_head_dim,
        device="cuda",
        dtype=cfg.dtype,
    )
    query = query_base[..., cfg.qk_nope_head_dim :]
    key = key_base[..., cfg.kv_lora_rank :].unsqueeze(1)
    return query_base, query, key


def _make_inputs(
    cfg: BenchmarkConfig,
    cos_sin_cache: torch.Tensor,
    *,
    positions: torch.Tensor | None = None,
    include_query_output_base: bool = False,
) -> RopeInputs:
    if positions is None:
        positions = torch.randint(
            0,
            cfg.max_position_embeddings,
            (cfg.num_tokens,),
            device="cuda",
            dtype=torch.long,
        )
    query_base, query, key = _make_mla_like_qk_views(cfg)
    return RopeInputs(
        positions=positions,
        query_base=query_base,
        query=query,
        key=key,
        cos_sin_cache=cos_sin_cache,
        query_out_base=(
            torch.empty_like(query_base) if include_query_output_base else None
        ),
    )


def _make_triton_runner(
    data_pool: list[RopeInputs],
    cfg: BenchmarkConfig,
):
    index = 0

    def run() -> None:
        nonlocal index
        triton_kimik25_rope(data_pool[index], cfg)
        index = (index + 1) % len(data_pool)

    return run


def _build_triton_version() -> KernelVersion:
    return KernelVersion(
        name="triton",
        make_runner=_make_triton_runner,
        run_once=triton_kimik25_rope,
        dump_artifacts=_dump_triton_artifacts,
        get_outputs=_default_version_outputs,
        bytes_per_call=_default_bytes_per_call,
    )


def _make_triton_fused_query_runner(
    data_pool: list[RopeInputs],
    cfg: BenchmarkConfig,
):
    index = 0

    def run() -> None:
        nonlocal index
        triton_kimik25_fused_query_rope(data_pool[index], cfg)
        index = (index + 1) % len(data_pool)

    return run


def _build_triton_fused_query_version() -> KernelVersion:
    return KernelVersion(
        name="triton_fused_query",
        make_runner=_make_triton_fused_query_runner,
        run_once=triton_kimik25_fused_query_rope,
        dump_artifacts=_dump_triton_fused_query_artifacts,
        get_outputs=_fused_query_version_outputs,
        needs_query_output_base=True,
        check_key_output=False,
        bytes_per_call=_fused_query_only_bytes_per_call,
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
    def kimik25_rope_kernel(
        positions: cute.Tensor,  # (Sp,)
        query: cute.Tensor,  # (Sp, N_local, (2, R // 2))
        key: cute.Tensor,  # (Sp, 1, (2, R // 2))
        cos_sin_cache: cute.Tensor,  # (max_position_embeddings, (R // 2, 2))
    ):
        tid, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()

        cache_row = cos_sin_cache[positions[bidx], (tid, None)]
        cos, sin = cache_row[0], cache_row[1]

        if bidy < query.shape[1]:
            query_slice = query[bidx, bidy, (None, tid)]
            a, b = query_slice[0], query_slice[1]
            a_new = a * cos - b * sin
            b_new = a * sin + b * cos
            query_slice[0] = a_new
            query_slice[1] = b_new
        else:
            key_slice = key[bidx, 0, (None, tid)]
            a, b = key_slice[0], key_slice[1]
            a_new = a * cos - b * sin
            b_new = a * sin + b * cos
            key_slice[0] = a_new
            key_slice[1] = b_new

    @cute.jit
    def kimik25_rope(
        positions: cute.Tensor,  # (Sp,)
        query: cute.Tensor,  # (Sp, N_local, R):(?, ?, 1)
        key: cute.Tensor,  # (Sp, 1, R)
        cos_sin_cache: cute.Tensor,  # (max_position_embeddings, R)
        N_local: cutlass.Constexpr,
        half_rope_dim: cutlass.Constexpr,
        stream: CUstream,
    ):
        sp = positions.shape[0]
        query = cute.make_tensor(
            query.iterator,
            cute.make_layout(
                (sp, N_local, (2, half_rope_dim)),
                stride=(query.stride[0], query.stride[1], (1, 2)),
            ),
        )
        key = cute.logical_divide(key, (1, 1, 2))[(0, None), (0, None), None]
        cos_sin_cache = cute.logical_divide(cos_sin_cache, (1, half_rope_dim))[
            (0, None), None
        ]
        kimik25_rope_kernel(positions, query, key, cos_sin_cache).launch(
            grid=(sp, N_local + 1, 1),
            block=(half_rope_dim, 1, 1),
            stream=stream,
        )

    compiled_function_cache: dict[tuple[Any, ...], Any] = {}
    compiled_executor_cache: dict[tuple[Any, ...], Any] = {}

    def _make_dynamic_cute_query(query: torch.Tensor):
        return from_dlpack(query, assumed_align=16).mark_layout_dynamic()

    def _get_cutedsl_cache_key(
        inputs: RopeInputs,
        cfg: BenchmarkConfig,
        dump_dir: Path | None,
    ) -> tuple[Any, ...]:
        return (
            torch.cuda.current_device(),
            tuple(inputs.positions.shape),
            inputs.positions.dtype,
            tuple(inputs.query.shape),
            tuple(inputs.query.stride()),
            inputs.query.dtype,
            tuple(inputs.key.shape),
            tuple(inputs.key.stride()),
            inputs.key.dtype,
            tuple(inputs.cos_sin_cache.shape),
            inputs.cos_sin_cache.dtype,
            cfg.num_local_heads,
            cfg.qk_rope_head_dim,
            str(dump_dir) if dump_dir is not None else None,
        )

    def _get_cutedsl_compiled_function(
        inputs: RopeInputs,
        cfg: BenchmarkConfig,
        *,
        dump_dir: Path | None = None,
    ):
        cache_key = _get_cutedsl_cache_key(inputs, cfg, dump_dir)
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
                kimik25_rope,
                positions=from_dlpack(inputs.positions),
                query=_make_dynamic_cute_query(inputs.query),
                key=from_dlpack(inputs.key, assumed_align=16),
                cos_sin_cache=from_dlpack(inputs.cos_sin_cache, assumed_align=16),
                N_local=cfg.num_local_heads,
                half_rope_dim=cfg.half_rope_dim,
                stream=cutlass_torch.current_stream(),
            )
            compiled_function_cache[cache_key] = compiled
        return compiled

    def _get_cutedsl_executor(
        inputs: RopeInputs,
        cfg: BenchmarkConfig,
    ):
        cache_key = _get_cutedsl_cache_key(inputs, cfg, dump_dir=None)
        executor = compiled_executor_cache.get(cache_key)
        if executor is None:
            executor = _get_cutedsl_compiled_function(
                inputs,
                cfg,
                dump_dir=None,
            ).to(None)
            compiled_executor_cache[cache_key] = executor
        return executor

    def dump_cutedsl_artifacts(
        output_dir: Path,
        inputs: RopeInputs,
        cfg: BenchmarkConfig,
    ) -> None:
        compiled = _get_cutedsl_compiled_function(
            inputs,
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

    def cutedsl_kimik25_rope(
        inputs: RopeInputs,
        cfg: BenchmarkConfig,
    ) -> None:
        executor = _get_cutedsl_executor(inputs, cfg)
        executor(
            positions=from_dlpack(inputs.positions),
            query=_make_dynamic_cute_query(inputs.query),
            key=from_dlpack(inputs.key),
            cos_sin_cache=from_dlpack(inputs.cos_sin_cache),
            stream=cutlass_torch.current_stream(),
        )

    def make_cutedsl_runner(
        data_pool: list[RopeInputs],
        cfg: BenchmarkConfig,
    ):
        cute_positions = from_dlpack(data_pool[0].positions)
        cute_query_pool = [_make_dynamic_cute_query(inputs.query) for inputs in data_pool]
        cute_key_pool = [from_dlpack(inputs.key) for inputs in data_pool]
        cute_cos_sin_cache = from_dlpack(data_pool[0].cos_sin_cache)
        executor = _get_cutedsl_executor(data_pool[0], cfg)
        index = 0

        def run() -> None:
            nonlocal index
            executor(
                positions=cute_positions,
                query=cute_query_pool[index],
                key=cute_key_pool[index],
                cos_sin_cache=cute_cos_sin_cache,
                stream=cutlass_torch.current_stream(),
            )
            index = (index + 1) % len(cute_query_pool)

        return run

    return KernelVersion(
        name="cutedsl",
        make_runner=make_cutedsl_runner,
        run_once=cutedsl_kimik25_rope,
        dump_artifacts=dump_cutedsl_artifacts,
        get_outputs=_default_version_outputs,
        bytes_per_call=_default_bytes_per_call,
    )


KERNEL_VERSION_BUILDERS = {
    "cutedsl": _build_cutedsl_version,
    "triton": _build_triton_version,
    "triton_fused_query": _build_triton_fused_query_version,
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


def _run_correctness(
    version: KernelVersion,
    cfg: BenchmarkConfig,
    rope,
) -> float:
    reference_inputs = _make_inputs(cfg, rope.cos_sin_cache)
    expected_query, expected_key = rope.forward_native(
        reference_inputs.positions,
        reference_inputs.query,
        reference_inputs.key,
    )
    if expected_key is None:
        raise RuntimeError("RoPE reference unexpectedly returned key=None.")

    actual_inputs = _make_inputs(
        cfg,
        rope.cos_sin_cache,
        positions=reference_inputs.positions,
        include_query_output_base=version.needs_query_output_base,
    )
    actual_inputs.query_base.copy_(reference_inputs.query_base)
    actual_inputs.key.copy_(reference_inputs.key)
    version.run_once(actual_inputs, cfg)
    torch.cuda.synchronize()

    actual_query, actual_key = version.get_outputs(actual_inputs, cfg)

    query_diff = (actual_query.float() - expected_query.float()).abs().max().item()
    key_diff = 0.0
    if version.check_key_output:
        key_diff = (actual_key.float() - expected_key.float()).abs().max().item()
    return max(query_diff, key_diff)


def _dump_selected_kernel_artifacts(
    dump_root: Path,
    version_name: str,
    version: KernelVersion,
    cfg: BenchmarkConfig,
    sample_inputs: RopeInputs,
) -> None:
    dump_dir = _make_dump_dir(
        dump_root,
        version_name,
        cfg,
        sample_inputs.query.stride(0),
        sample_inputs.query.stride(1),
    )
    version.dump_artifacts(dump_dir, sample_inputs, cfg)


def main(args) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    if not current_platform.is_device_capability_family(100):
        print("Warning: this benchmark is primarily intended for Blackwell GPUs (SM10x).")

    set_random_seed(args.seed)
    if args.qk_rope_head_dim % 2 != 0:
        raise ValueError("--qk-rope-head-dim must be even.")

    if args.versions is not None:
        version_names = _resolve_version_names(args.versions)
    elif args.provider is not None:
        version_names = (
            ["cutedsl", "triton"] if args.provider == "both" else [args.provider]
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
    if args.dump_kernel_artifacts_dir is not None:
        dump_root = Path(args.dump_kernel_artifacts_dir).expanduser().resolve()
        print(f"Kernel artifact dump root: {dump_root}")

    version_col_width = max(len("version"), max(len(name)
                                                 for name in version_names))
    header = (
        f"{'version':<{version_col_width}} "
        f"{'tokens':>8} {'heads':>8} {'q_s0':>8} {'q_s1':>8} "
        f"{'median_us':>12} {'min_us':>12} {'max_us':>12} {'gbps':>10} "
        f"{'speedup':>10} {'max_abs_diff':>14}"
    )
    print(header)
    print("-" * len(header))

    for num_tokens in args.num_tokens:
        cfg = BenchmarkConfig(
            num_tokens=num_tokens,
            num_local_heads=args.num_local_heads,
            qk_nope_head_dim=args.qk_nope_head_dim,
            kv_lora_rank=args.kv_lora_rank,
            qk_rope_head_dim=args.qk_rope_head_dim,
            max_position_embeddings=args.max_position_embeddings,
        )
        rope = _make_rope_module(cfg)
        sample_inputs = _make_inputs(cfg, rope.cos_sin_cache)

        if args.dump_kernel_artifacts_dir is not None:
            for version_name in version_names:
                version_sample_inputs = _make_inputs(
                    cfg,
                    rope.cos_sin_cache,
                    positions=sample_inputs.positions,
                    include_query_output_base=versions[
                        version_name
                    ].needs_query_output_base,
                )
                _dump_selected_kernel_artifacts(
                    dump_root,
                    version_name,
                    versions[version_name],
                    cfg,
                    version_sample_inputs,
                )

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
        for version_name in version_names:
            data_pool = [
                _make_inputs(
                    cfg,
                    rope.cos_sin_cache,
                    positions=positions,
                    include_query_output_base=versions[
                        version_name
                    ].needs_query_output_base,
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
        q_stride0 = sample_inputs.query.stride(0)
        q_stride1 = sample_inputs.query.stride(1)
        for version_name in version_names:
            median_us = medians_us[version_name]
            speedup = None
            if baseline_us is not None:
                speedup = baseline_us / median_us
            bytes_per_call = versions[version_name].bytes_per_call(cfg)

            print(
                f"{version_name:<{version_col_width}} "
                f"{num_tokens:>8d} "
                f"{cfg.num_local_heads:>8d} "
                f"{q_stride0:>8d} "
                f"{q_stride1:>8d} "
                f"{median_us:>12.2f} "
                f"{mins_us[version_name]:>12.2f} "
                f"{maxs_us[version_name]:>12.2f} "
                f"{_format_bandwidth(bytes_per_call, median_us):>10.2f} "
                f"{_format_speedup(speedup):>10} "
                f"{_format_diff(diffs[version_name]):>14}"
            )


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Benchmark the Kimi-K2.5 RoPE kernels."
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
        default=[13, 64, 512, 4096],
        help="Token counts to benchmark.",
    )
    parser.add_argument(
        "--num-local-heads",
        type=int,
        default=NUM_LOCAL_HEADS,
        help="Number of local query heads in the MLA RoPE view.",
    )
    parser.add_argument(
        "--qk-nope-head-dim",
        type=int,
        default=QK_NOPE_HEAD_DIM,
        help="Non-RoPE per-head dimension used to define the query view stride.",
    )
    parser.add_argument(
        "--kv-lora-rank",
        type=int,
        default=KV_LORA_RANK,
        help="KV LoRA rank used to define the key view stride.",
    )
    parser.add_argument(
        "--qk-rope-head-dim",
        type=int,
        default=QK_ROPE_HEAD_DIM,
        help="RoPE per-head dimension.",
    )
    parser.add_argument(
        "--max-position-embeddings",
        type=int,
        default=MAX_POSITION,
        help="Maximum position used to size the cosine/sine cache.",
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
        help="Compare each kernel against the vLLM PyTorch reference before timing.",
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
