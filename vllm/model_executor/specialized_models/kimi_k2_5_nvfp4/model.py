# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Text-only specialized Kimi-K2.5 NVFP4 implementation."""

import atexit
import threading
from collections.abc import Iterable
from typing import Any

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import torch
import torch.nn.functional as F
from cuda.bindings.driver import CUstream
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T, dsl_user_op
from torch import nn

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import (
    get_dp_group,
    get_node_count,
    get_pcp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_reduce,
)
from vllm.distributed.device_communicators.flashinfer_all_reduce import (
    _create_workspace as _create_flashinfer_allreduce_workspace,
)
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import MLAAttention
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.experts.trtllm_nvfp4_moe import (
    TrtLlmNvFp4ExpertsMonolithic,
)
from vllm.model_executor.layers.fused_moe.layer import (
    FusedMoE,
    determine_expert_map,
    determine_expert_placement_strategy,
    get_compressed_expert_map,
)
from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import NvFp4MoeBackend
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptNvFp4Config,
    ModelOptNvFp4FusedMoE,
    ModelOptNvFp4LinearMethod,
)
from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
    activation_to_flashinfer_int,
)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.models.deepseek_v2 import (
    DeepseekV2ForCausalLM,
    DeepSeekV2FusedQkvAProjLinear,
    DeepseekV2MLP,
    yarn_get_mscale,
)
from vllm.model_executor.models.interfaces import (
    SupportsEagle,
    SupportsEagle3,
    SupportsPP,
    SupportsQuant,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    make_empty_intermediate_tensors_factory,
)
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.utils.flashinfer import (
    flashinfer_scaled_fp4_mm,
    has_flashinfer_trtllm_fused_moe,
)
from vllm.utils.torch_utils import (
    aux_stream,
    current_stream,
    direct_register_custom_op,
    is_quantized_kv_cache,
)
from vllm.v1.attention.selector import get_attn_backend
from vllm.v1.worker.ubatching import dbo_current_ubatch_id

_TARGET_MODEL_NAMES = {"nvidia/Kimi-K2.5-NVFP4"}
logger = init_logger(__name__)

_KIMI_MOE_FINALIZE_AR_WORLD_SIZES = {2, 4, 8, 16}
_kimi_moe_finalize_ar_workspace: Any | None = None
_kimi_moe_finalize_ar_workspace_key: tuple[int, int, int, torch.dtype, int] | None = (
    None
)
_kimi_moe_finalize_ar_workspace_max_tokens = 0
_kimi_moe_finalize_ar_workspace_disabled = False
_kimi_moe_finalize_ar_workspace_lock = threading.Lock()


def _destroy_kimi_moe_finalize_ar_workspace() -> None:
    global _kimi_moe_finalize_ar_workspace
    global _kimi_moe_finalize_ar_workspace_key
    global _kimi_moe_finalize_ar_workspace_max_tokens

    workspace = _kimi_moe_finalize_ar_workspace
    _kimi_moe_finalize_ar_workspace = None
    _kimi_moe_finalize_ar_workspace_key = None
    _kimi_moe_finalize_ar_workspace_max_tokens = 0
    if workspace is not None:
        try:
            workspace.destroy()
        except Exception:
            logger.debug(
                "Failed to destroy Kimi-K2.5 MoE finalize all-reduce workspace.",
                exc_info=True,
            )


atexit.register(_destroy_kimi_moe_finalize_ar_workspace)


def _get_kimi_moe_finalize_ar_workspace(
    *,
    world_size: int,
    rank: int,
    max_token_num: int,
    hidden_dim: int,
    dtype: torch.dtype,
    group: Any,
) -> Any | None:
    global _kimi_moe_finalize_ar_workspace
    global _kimi_moe_finalize_ar_workspace_key
    global _kimi_moe_finalize_ar_workspace_max_tokens
    global _kimi_moe_finalize_ar_workspace_disabled

    if get_node_count() > 1:
        logger.warning_once(
            "FlashInfer TRTLLM MoE finalize all-reduce fusion is not supported "
            "for multi-node tensor-parallel groups."
        )
        _kimi_moe_finalize_ar_workspace_disabled = True
        return None
    if _kimi_moe_finalize_ar_workspace_disabled:
        return None

    key = (world_size, rank, hidden_dim, dtype, id(group))
    with _kimi_moe_finalize_ar_workspace_lock:
        workspace = _kimi_moe_finalize_ar_workspace
        if (
            workspace is not None
            and _kimi_moe_finalize_ar_workspace_key == key
            and _kimi_moe_finalize_ar_workspace_max_tokens >= max_token_num
            and workspace.is_buffer_size_sufficient(
                world_size,
                max_token_num,
                hidden_dim,
                dtype,
            )
        ):
            return workspace

        if workspace is not None and current_platform.is_cuda():
            torch.cuda.synchronize()
        _destroy_kimi_moe_finalize_ar_workspace()
        workspace = _create_flashinfer_allreduce_workspace(
            "trtllm",
            world_size,
            rank,
            max_token_num,
            hidden_dim,
            dtype,
            group,
        )
        if workspace is None:
            _kimi_moe_finalize_ar_workspace_disabled = True
            logger.warning_once(
                "Failed to initialize Kimi-K2.5 MoE finalize all-reduce "
                "workspace; falling back to separate MoE, all-reduce, and "
                "RMSNorm."
            )
            return None

        _kimi_moe_finalize_ar_workspace = workspace
        _kimi_moe_finalize_ar_workspace_key = key
        _kimi_moe_finalize_ar_workspace_max_tokens = max_token_num
        logger.info_once(
            "Initialized Kimi-K2.5 MoE finalize all-reduce workspace "
            "with backend=trtllm."
        )
        return workspace


def _is_kimi_nvfp4_checkpoint(vllm_config: VllmConfig) -> bool:
    model_name = vllm_config.model_config.model
    if model_name in _TARGET_MODEL_NAMES:
        return True

    hf_config = vllm_config.model_config.hf_config
    if getattr(hf_config, "model_type", None) != "kimi_k25":
        return False

    text_config = getattr(hf_config, "text_config", None)
    if text_config is None:
        return False

    quantization_config = getattr(hf_config, "quantization_config", None)
    if quantization_config is None:
        quantization_config = getattr(text_config, "quantization_config", None)

    return (
        getattr(text_config, "model_type", None) == "deepseek_v3"
        and getattr(quantization_config, "get", lambda *_: None)("quant_algo")
        == "NVFP4"
    )


def _get_kimi_nvfp4_specialization_rejection_reason(
    vllm_config: VllmConfig,
) -> str | None:
    if not _is_kimi_nvfp4_checkpoint(vllm_config):
        return "the checkpoint is not nvidia/Kimi-K2.5-NVFP4"

    quant_config = vllm_config.quant_config
    if not isinstance(quant_config, ModelOptNvFp4Config):
        return "the specialized Kimi-K2.5 MoE path requires ModelOpt NVFP4"
    if not quant_config.is_checkpoint_nvfp4_serialized:
        return "the specialized Kimi-K2.5 MoE path requires serialized NVFP4"

    cache_config = vllm_config.cache_config
    kv_cache_dtype = cache_config.cache_dtype if cache_config is not None else "auto"
    if not kv_cache_dtype.startswith("fp8"):
        return (
            "the specialized Kimi-K2.5 path requires an FP8 KV cache, "
            f"but got {kv_cache_dtype!r}"
        )

    config = vllm_config.model_config.hf_config
    text_config = getattr(config, "text_config", config)
    parallel_config = vllm_config.parallel_config

    if parallel_config.enable_eplb:
        return "the specialized Kimi-K2.5 MoE path does not support EPLB"
    if parallel_config.use_sequence_parallel_moe:
        return "the specialized Kimi-K2.5 MoE path does not support MoE SP"
    if parallel_config.data_parallel_size != 1:
        return "the specialized Kimi-K2.5 MoE path does not support DP"
    if (
        parallel_config.enable_expert_parallel
        and parallel_config.tensor_parallel_size > 1
        and parallel_config.expert_placement_strategy != "linear"
    ):
        return (
            "the specialized Kimi-K2.5 MoE path supports EP only with "
            "linear expert placement"
        )

    moe_backend = getattr(vllm_config.kernel_config, "moe_backend", "auto")
    if moe_backend not in ("auto", "flashinfer_trtllm"):
        return (
            "the specialized Kimi-K2.5 MoE path requires the auto or "
            f"flashinfer_trtllm MoE backend, but got {moe_backend!r}"
        )
    if (
        not current_platform.is_cuda()
        or not current_platform.is_device_capability_family(100)
    ):
        return "the specialized Kimi-K2.5 MoE path requires Blackwell CUDA"
    if not has_flashinfer_trtllm_fused_moe():
        return "FlashInfer TRTLLM fused NVFP4 MoE is unavailable"
    if envs.is_set("VLLM_USE_FLASHINFER_MOE_FP4") and not (
        envs.VLLM_USE_FLASHINFER_MOE_FP4
    ):
        return "VLLM_USE_FLASHINFER_MOE_FP4 disables FlashInfer NVFP4 MoE"
    if (
        envs.is_set("VLLM_FLASHINFER_MOE_BACKEND")
        and envs.VLLM_FLASHINFER_MOE_BACKEND != "latency"
    ):
        return "the specialized Kimi-K2.5 MoE path requires FlashInfer latency MoE"

    if getattr(text_config, "hidden_act", None) != "silu":
        return "the specialized Kimi-K2.5 MoE path only supports silu experts"
    if getattr(text_config, "hidden_size", None) != 7168:
        return "the specialized Kimi-K2.5 MoE path expects hidden_size=7168"
    if getattr(text_config, "n_routed_experts", None) != 384:
        return "the specialized Kimi-K2.5 MoE path expects 384 routed experts"
    if getattr(text_config, "topk_method", None) != "noaux_tc":
        return "the specialized Kimi-K2.5 MoE path requires noaux_tc routing"
    if getattr(text_config, "scoring_func", "softmax") != "sigmoid":
        return "the specialized Kimi-K2.5 MoE path requires sigmoid routing"
    if getattr(text_config, "n_group", 1) <= 0:
        return "the specialized Kimi-K2.5 MoE path requires grouped routing"

    num_local_heads = (
        text_config.num_attention_heads // get_tensor_model_parallel_world_size()
    )
    try:
        attn_backend = get_attn_backend(
            head_size=text_config.kv_lora_rank + text_config.qk_rope_head_dim,
            dtype=torch.get_default_dtype(),
            kv_cache_dtype=kv_cache_dtype,
            use_mla=True,
            use_sparse=False,
            num_heads=num_local_heads,
        )
    except Exception as exc:
        return f"no compatible non-sparse MLA backend was selected: {exc}"

    attn_backend_name = attn_backend.get_name()
    if attn_backend_name == "TRITON_MLA":
        return "TRITON_MLA does not support quantized query input for FP8 MLA"
    if attn_backend_name == "CUTLASS_MLA":
        return "CUTLASS_MLA requires q_pad_num_heads=128"

    return None


def _is_target_kimi_nvfp4(vllm_config: VllmConfig) -> bool:
    return _get_kimi_nvfp4_specialization_rejection_reason(vllm_config) is None


def _fallback_to_generic_kimi_k25(
    vllm_config: VllmConfig,
    prefix: str,
    reason: str,
) -> nn.Module:
    from vllm.model_executor.models.kimi_k25 import (
        KimiK25ForConditionalGeneration as GenericKimiK25ForConditionalGeneration,
    )

    logger.info_once(
        "Falling back to the generic Kimi-K2.5 implementation because %s.",
        reason,
    )
    return GenericKimiK25ForConditionalGeneration(
        vllm_config=vllm_config,
        prefix=prefix,
    )


# ---------------------------------------------------------------------------
# Monolithic MLA attention custom op
# ---------------------------------------------------------------------------
# Fuses KV-cache update + decode W_UK_T absorption + attention + W_UV
# up-projection into a single opaque op so that torch.compile treats the
# entire block as one node.
# ---------------------------------------------------------------------------


def _kimi_mla_attn(
    q: torch.Tensor,
    kv_c_normed: torch.Tensor,
    k_pe: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    """Monolithic MLA attention: KV cache update + attention forward."""
    mla: MLAAttention = get_forward_context().no_compile_layers[layer_name]

    fwd_ctx = get_forward_context()
    attn_metadata = fwd_ctx.attn_metadata
    if isinstance(attn_metadata, dict):
        attn_metadata = attn_metadata.get(layer_name)

    if attn_metadata is None:
        output.zero_()
        return output

    num_actual_toks = attn_metadata.num_actual_tokens
    if num_actual_toks == 0:
        output.zero_()
        return output

    # ---- Calculate FP8 KV cache scales if needed ----
    if mla.calculate_kv_scales:
        mla.calc_kv_scales(q, kv_c_normed, k_pe)

    # ---- KV cache update (inlined do_kv_cache_update) ----
    kv_cache = mla.kv_cache
    if kv_cache.numel() > 0:
        slot_mapping = fwd_ctx.slot_mapping
        if isinstance(slot_mapping, dict):
            slot_mapping = slot_mapping.get(layer_name)
        if slot_mapping is not None:
            ops.concat_and_cache_mla(
                kv_c_normed,
                k_pe.squeeze(1),
                kv_cache,
                slot_mapping.flatten(),
                kv_cache_dtype=mla.kv_cache_dtype,
                scale=mla._k_scale,
            )

    # ---- Attention forward (inlined forward_impl) ----
    # Lazy-init DCP world size (must happen before forward_mha/forward_mqa).
    if mla.impl.dcp_world_size == -1:
        from vllm.distributed.parallel_state import get_dcp_group

        mla.impl.dcp_world_size = get_dcp_group().world_size

    # Trim to actual tokens (inputs may be padded for CUDA graphs).
    output_padded = output
    output = output[:num_actual_toks]
    q = q[:num_actual_toks]
    kv_c_normed = kv_c_normed[:num_actual_toks]
    k_pe = k_pe[:num_actual_toks]

    fp8_attn = is_quantized_kv_cache(mla.kv_cache_dtype)
    if fp8_attn and mla.kv_cache_dtype != "fp8_ds_mla":
        kv_cache = kv_cache.view(current_platform.fp8_dtype())

    assert (
        attn_metadata.num_decodes is not None
        and attn_metadata.num_prefills is not None
        and attn_metadata.num_decode_tokens is not None
    )
    num_mqa_tokens = attn_metadata.num_decode_tokens
    num_mha_tokens = q.size(0) - num_mqa_tokens

    # -- Prefill path (MHA) --
    if num_mha_tokens > 0:
        mla.impl.forward_mha(
            q[num_mqa_tokens:],
            kv_c_normed[num_mqa_tokens:],
            k_pe[num_mqa_tokens:],
            kv_cache,
            attn_metadata,
            mla._k_scale,
            output=output[num_mqa_tokens:],
        )

    # -- Decode path (MQA) --
    if num_mqa_tokens > 0:
        mqa_q = q[:num_mqa_tokens]

        mqa_q_nope, mqa_q_pe = mqa_q.split(
            [mla.qk_nope_head_dim, mla.qk_rope_head_dim], dim=-1
        )

        # (B, N, P) -> (N, B, P) for batched matmul
        mqa_q_nope = mqa_q_nope.transpose(0, 1)
        N, B, P = mqa_q_nope.shape
        _, _, L = mla.W_UK_T.shape

        # Head padding if required by the backend kernel
        q_pad = mla.q_pad_num_heads
        if q_pad is not None:
            mqa_ql_nope = mqa_q_nope.new_empty((q_pad, B, L))
            mqa_ql_nope.resize_((N, B, L))
        else:
            mqa_ql_nope = mqa_q_nope.new_empty((N, B, L))

        # W_UK_T absorption: (N, B, P) x (N, P, L) -> (N, B, L)
        torch.bmm(mqa_q_nope, mla.W_UK_T, out=mqa_ql_nope)
        # (N, B, L) -> (B, N, L)
        mqa_ql_nope = mqa_ql_nope.transpose(0, 1)

        if q_pad is not None:
            B_pe, N_pe, L_pe = mqa_q_pe.shape
            mqa_pe_padded = mqa_q_pe.new_empty((B_pe, q_pad, L_pe))
            mqa_pe_padded.resize_((B_pe, N_pe, L_pe))
            mqa_pe_padded.copy_(mqa_q_pe)
            mqa_q_pe = mqa_pe_padded

        # FP8 concat+quantise or plain tuple
        if fp8_attn and mla.impl.supports_quant_query_input:
            mqa_q_final = mla._decode_concat_quant_fp8_op(
                mqa_ql_nope, mqa_q_pe, mla._q_scale
            )
        else:
            mqa_q_final = (mqa_ql_nope, mqa_q_pe)

        # Decode attention kernel
        attn_out, _ = mla.impl.forward_mqa(mqa_q_final, kv_cache, attn_metadata, mla)

        # W_UV up-projection: (N, B, L) x (N, L, V) -> (N, B, V)
        x = attn_out.view(-1, mla.num_heads, mla.kv_lora_rank).transpose(0, 1)
        out = output[:num_mqa_tokens].view(-1, mla.num_heads, mla.v_head_dim)
        out = out.transpose(0, 1)
        torch.bmm(x, mla.W_UV, out=out)

    return output_padded


def _kimi_mla_attn_fake(
    q: torch.Tensor,
    kv_c_normed: torch.Tensor,
    k_pe: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    del q, kv_c_normed, k_pe, layer_name
    return output


def _forked_kimi_mla_attn(
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    """Forked MLA attention block through W_UV for benchmarking."""
    layer = get_forward_context().no_compile_layers[layer_name]
    mla = layer.mla_attn

    qkv_a, _ = layer.fused_qkv_a_proj(hidden_states)
    qkv_c, k_pe = qkv_a.split(
        [layer.q_lora_rank + layer.kv_lora_rank, layer.qk_rope_head_dim],
        dim=-1,
    )

    _run_kimik25_rmsnorm_special_qkv_fused(
        data=qkv_c,
        positions=positions,
        k_pe=k_pe,
        cos_sin_cache=layer.rotary_emb.cos_sin_cache,
        weights_q=layer.q_a_layernorm.weight.detach(),
        weights_kv=layer.kv_a_layernorm.weight.detach(),
        lora_dim_q=layer.q_a_layernorm.hidden_size,
        lora_dim_kv=layer.kv_a_layernorm.hidden_size,
        pe_dim=layer.qk_rope_head_dim,
        eps_q=layer.q_a_layernorm.variance_epsilon,
        eps_kv=layer.kv_a_layernorm.variance_epsilon,
    )

    q_c, kv_c = qkv_c.split([layer.q_lora_rank, layer.kv_lora_rank], dim=-1)
    kv_c_cache = kv_c
    k_pe_cache = k_pe

    fwd_ctx = get_forward_context()
    if fwd_ctx.attn_metadata is None:
        output.zero_()
        return output
    attn_metadata = fwd_ctx.attn_metadata.get(mla.layer_name)

    num_actual_toks = attn_metadata.num_actual_tokens
    if num_actual_toks == 0:
        output.zero_()
        return output

    kv_cache = mla.kv_cache
    slot_mapping = None
    if kv_cache.numel() > 0:
        slot_mapping = fwd_ctx.slot_mapping.get(mla.layer_name)
        if slot_mapping is not None:
            slot_mapping = slot_mapping.flatten()

    q, _ = layer.q_b_proj(q_c)

    q = q.view(-1, layer.num_local_heads, layer.qk_head_dim)
    k_pe = k_pe.unsqueeze(1)

    if mla.impl.dcp_world_size == -1:
        from vllm.distributed.parallel_state import get_dcp_group

        mla.impl.dcp_world_size = get_dcp_group().world_size

    attn_output = output[:num_actual_toks]
    q = q[:num_actual_toks]
    kv_c = kv_c[:num_actual_toks]
    k_pe = k_pe[:num_actual_toks]

    assert mla.kv_cache_dtype.startswith("fp8"), "only FP8 KV cache is supported"
    kv_cache_storage = kv_cache
    if mla.kv_cache_dtype != "fp8_ds_mla":
        kv_cache = kv_cache.view(current_platform.fp8_dtype())

    assert (
        attn_metadata.num_decodes is not None
        and attn_metadata.num_prefills is not None
        and attn_metadata.num_decode_tokens is not None
    )
    num_mqa_tokens = attn_metadata.num_decode_tokens
    num_mha_tokens = q.size(0) - num_mqa_tokens

    mqa_q_final: torch.Tensor | None = None
    if num_mqa_tokens > 0:
        mqa_q = q[:num_mqa_tokens]
        mqa_q_nope, mqa_q_pe = mqa_q.split(
            [mla.qk_nope_head_dim, mla.qk_rope_head_dim], dim=-1
        )

        mqa_q_nope = mqa_q_nope.transpose(0, 1)
        N, B, P = mqa_q_nope.shape
        _, _, L = mla.W_UK_T.shape

        assert mla.q_pad_num_heads is None, "num_heads padding is unsupported"
        mqa_ql_nope = mqa_q_nope.new_empty((N, B, L))

        torch.bmm(mqa_q_nope, mla.W_UK_T, out=mqa_ql_nope)
        mqa_ql_nope = mqa_ql_nope.transpose(0, 1)

        if slot_mapping is not None:
            mqa_q_final = _run_kimik25_decode_rope_concat_quant_fp8_and_cache_mla(
                positions=positions[:num_mqa_tokens],
                ql_nope=mqa_ql_nope,
                q_pe=mqa_q_pe,
                cos_sin_cache=layer.rotary_emb.cos_sin_cache,
                q_scale=mla._q_scale,
                kv_c=kv_c_cache,
                k_pe=k_pe_cache,
                kv_cache=kv_cache_storage,
                slot_mapping=slot_mapping,
                kv_cache_dtype=mla.kv_cache_dtype,
                kv_scale=mla._k_scale,
            )
        else:
            mqa_q_final = _run_kimik25_decode_rope_concat_quant_fp8(
                positions=positions[:num_mqa_tokens],
                ql_nope=mqa_ql_nope,
                q_pe=mqa_q_pe,
                cos_sin_cache=layer.rotary_emb.cos_sin_cache,
                scale=mla._q_scale,
            )
    elif slot_mapping is not None:
        _run_kimik25_concat_and_cache_mla(
            kv_c=kv_c_cache,
            k_pe=k_pe_cache,
            kv_cache=kv_cache_storage,
            slot_mapping=slot_mapping,
            kv_cache_dtype=mla.kv_cache_dtype,
            scale=mla._k_scale,
        )

    if num_mha_tokens > 0:
        q_pe = q[..., layer.qk_nope_head_dim :]
        assert q_pe.dtype == layer.rotary_emb.cos_sin_cache.dtype
        _run_kimik25_rope(
            positions[num_mqa_tokens:num_actual_toks],
            q_pe[num_mqa_tokens:],
            layer.rotary_emb.cos_sin_cache,
            layer.num_local_heads,
            layer.qk_rope_head_dim // 2,
        )
        mla.impl.forward_mha(
            q[num_mqa_tokens:],
            kv_c[num_mqa_tokens:],
            k_pe[num_mqa_tokens:],
            kv_cache,
            attn_metadata,
            mla._k_scale,
            output=attn_output[num_mqa_tokens:],
        )

    if num_mqa_tokens > 0:
        assert mqa_q_final is not None

        decode_attn_out, _ = mla.impl.forward_mqa(
            mqa_q_final, kv_cache, attn_metadata, mla
        )

        x = decode_attn_out.view(-1, mla.num_heads, mla.kv_lora_rank).transpose(0, 1)
        out = attn_output[:num_mqa_tokens].view(-1, mla.num_heads, mla.v_head_dim)
        out = out.transpose(0, 1)
        torch.bmm(x, mla.W_UV, out=out)

    return output


def _forked_kimi_mla_attn_fake(
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    del positions, hidden_states, layer_name
    return output


direct_register_custom_op(
    op_name="monolithic_attn",
    op_func=_kimi_mla_attn,
    fake_impl=_kimi_mla_attn_fake,
    mutates_args=["output"],
    dispatch_key=current_platform.dispatch_key,
)

direct_register_custom_op(
    op_name="forked_monolithic_attn",
    op_func=_forked_kimi_mla_attn,
    fake_impl=_forked_kimi_mla_attn_fake,
    mutates_args=["output"],
    dispatch_key=current_platform.dispatch_key,
)


class KimiK25Nvfp4MLAAttention(nn.Module):
    """Inlined MLA path for the Kimi-K2.5 NVFP4 text checkpoint."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        config,
        cache_config,
        quant_config,
        prefix: str,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_local_heads = self.num_heads // get_tensor_model_parallel_world_size()
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.scaling = self.qk_head_dim**-0.5

        self.fused_qkv_a_proj = DeepSeekV2FusedQkvAProjLinear(
            self.hidden_size,
            [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
            quant_config=quant_config,
            prefix=f"{prefix}.fused_qkv_a_proj",
        )
        self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = ColumnParallelLinear(
            self.q_lora_rank,
            self.num_heads * self.qk_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.q_b_proj",
        )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj",
        )
        self.o_proj = RowParallelLinear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        if config.rope_parameters["rope_type"] != "default":
            config.rope_parameters["rope_type"] = (
                "deepseek_yarn"
                if config.rope_parameters.get("apply_yarn_scaling", True)
                else "deepseek_llama_scaling"
            )
        self.rotary_emb = get_rope(
            self.qk_rope_head_dim,
            max_position=getattr(config, "max_position_embeddings", 8192),
            rope_parameters=config.rope_parameters,
            is_neox_style=False,
        )
        if config.rope_parameters["rope_type"] == "deepseek_yarn":
            mscale_all_dim = config.rope_parameters.get("mscale_all_dim", False)
            scaling_factor = config.rope_parameters["factor"]
            mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
            self.scaling = self.scaling * mscale * mscale

        self.mla_attn = MLAAttention(
            num_heads=self.num_local_heads,
            scale=self.scaling,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            kv_b_proj=self.kv_b_proj,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            use_sparse=False,
            indexer=None,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv_a, _ = self.fused_qkv_a_proj(hidden_states)
        q_c, kv_lora = qkv_a.split(
            [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
            dim=-1,
        )

        q_c = self.q_a_layernorm(q_c)
        q, _ = self.q_b_proj(q_c)

        kv_c, k_pe = kv_lora.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_c_normed = self.kv_a_layernorm(kv_c)

        q = q.view(-1, self.num_local_heads, self.qk_head_dim)
        q_pe = q[..., self.qk_nope_head_dim :]
        k_pe = k_pe.unsqueeze(1)
        q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)
        q[..., self.qk_nope_head_dim :] = q_pe

        mla = self.mla_attn
        output = torch.empty(
            (hidden_states.shape[0], self.num_local_heads * self.v_head_dim),
            dtype=q.dtype,
            device=q.device,
        )
        attn_out = torch.ops.vllm.monolithic_attn(
            q,
            kv_c_normed,
            k_pe,
            output,
            mla.layer_name,
        )
        return self.o_proj(attn_out)[0]


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
                ] = cutlass.Uint8(
                    _cvt_f32_to_e4m3(k_pe_val) & cutlass.Uint32(0xFF)
                )


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
def kimik25_rmsnorm_kernel(
    data: cute.Tensor,  # (Sp, (lora_dim // k, k))
    weights: cute.Tensor,  # (lora_dim // k, k)
    lora_dim: cutlass.Constexpr,
    eps: cutlass.Constexpr,
    k: cutlass.Constexpr,
):
    assert lora_dim % k == 0
    assert lora_dim // k // 32 in [1, 2, 4, 8, 16, 32]
    allocator = cutlass.utils.SmemAllocator()
    sdata = allocator.allocate_tensor(
        cutlass.Float32,
        layout=cute.make_layout((lora_dim // k // 32,)),
        byte_alignment=16,
        swizzle=None,
    )

    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()

    sum: cutlass.Float32 = 0.0
    for i in cutlass.range_constexpr(k):
        x = data[bid, (tid, i)].to(cutlass.Float32)
        sum += x * x
    sum = cute.arch.warp_reduction_sum(sum, threads_in_group=32)
    if tid % 32 == 0:
        sdata[tid // 32] = sum
    cute.arch.sync_threads()
    if tid < 32:
        if tid < lora_dim // k // 32:  # noqa: SIM108
            sum = sdata[tid]
        else:
            sum = 0.0
        sum = cute.arch.warp_reduction_sum(sum, threads_in_group=lora_dim // k // 32)
        if tid == 0:
            sdata[0] = cute.math.rsqrt(sum / lora_dim + eps)
    cute.arch.sync_threads()
    invnorm = sdata[0]
    for i in cutlass.range_constexpr(k):
        x = data[bid, (tid, i)].to(cutlass.Float32)
        x = (x * invnorm).to(cutlass.BFloat16) * weights[tid, i]
        data[bid, (tid, i)] = x


@cute.jit
def kimik25_rmsnorm(
    data: cute.Tensor,
    weights: cute.Tensor,
    lora_dim: cutlass.Constexpr,
    eps: cutlass.Constexpr,
    k: cutlass.Constexpr,
    stream: CUstream,
):
    data = cute.make_tensor(
        data.iterator,
        cute.make_layout(
            (data.shape[0], (lora_dim // k, k)),
            stride=(data.stride[0], (1, lora_dim // k)),
        ),
    )
    weights = cute.make_tensor(
        weights.iterator,
        cute.make_layout((lora_dim // k, k), stride=(1, lora_dim // k)),
    )
    grid = (data.shape[0], 1, 1)
    block = (lora_dim // k, 1, 1)
    kimik25_rmsnorm_kernel(data, weights, lora_dim, eps, k).launch(
        grid=grid, block=block, stream=stream
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
        kv_c_elems_per_split: cutlass.Constexpr = (
            kv_lora_rank // kv_cache_block_factor
        )

        slot_idx = slot_mapping[token_idx]
        if slot_idx >= 0:
            block_size = kv_cache.shape[1]
            cache_block_idx = slot_idx // block_size
            cache_block_offset = slot_idx % block_size
            scale_value = kv_scale[0].to(cutlass.Float32)

            if split_idx > 0:
                kv_c_idx = (split_idx - 1) * kv_c_elems_per_split + tid
                kv_c_val = kv_c[token_idx, kv_c_idx].to(cutlass.Float32)
                kv_cache[cache_block_idx, cache_block_offset, kv_c_idx] = (
                    cutlass.Uint8(
                        _cvt_f32_to_e4m3(kv_c_val / scale_value)
                        & cutlass.Uint32(0xFF)
                    )
                )
            else:
                if tid < pe_dim:
                    k_pe_val = k_pe[token_idx, tid].to(cutlass.Float32)
                    kv_cache[
                        cache_block_idx,
                        cache_block_offset,
                        kv_lora_rank + tid,
                    ] = cutlass.Uint8(
                        _cvt_f32_to_e4m3(k_pe_val / scale_value)
                        & cutlass.Uint32(0xFF)
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
def kimik25_rmsnorm_special_qkv_split_kernel(
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
def kimik25_rmsnorm_special_qkv_split(
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
    kimik25_rmsnorm_special_qkv_split_kernel(
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


class ForkedKimiK25Nvfp4MLAAttention(KimiK25Nvfp4MLAAttention):
    """Forked MLA path for kernel benchmarking experiments."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        config,
        cache_config,
        quant_config,
        prefix: str,
    ) -> None:
        print("Initializing ForkedKimiK25Nvfp4MLAAttention")
        super().__init__(
            vllm_config=vllm_config,
            config=config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
        )
        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self
        self.layer_name = prefix

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        attn_output = torch.empty(
            (hidden_states.shape[0], self.num_local_heads * self.v_head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        attn_out = torch.ops.vllm.forked_monolithic_attn(
            positions,
            hidden_states,
            attn_output,
            self.layer_name,
        )
        return self.o_proj(attn_out)[0]


class _KimiK25SharedExpertOverlap:
    """Kimi-only shared expert overlap with the routed MoE path."""

    def __init__(
        self,
        layer: nn.Module,
        *,
        enable_dbo: bool,
    ) -> None:
        self.layer = layer
        self.enable_dbo = enable_dbo
        self.outputs: list[torch.Tensor | None] = [None, None]
        self.stream = (
            None
            if envs.VLLM_DISABLE_SHARED_EXPERTS_STREAM
            else aux_stream()
        )

    @property
    def output_idx(self) -> int:
        return dbo_current_ubatch_id() if self.enable_dbo else 0

    def should_overlap(self, hidden_states: torch.Tensor) -> bool:
        return (
            current_platform.is_cuda()
            and self.stream is not None
            and hidden_states.shape[0]
            <= envs.VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD
        )

    def start(self, hidden_states: torch.Tensor) -> bool:
        if not self.should_overlap(hidden_states):
            return False

        assert self.stream is not None
        idx = self.output_idx
        assert self.outputs[idx] is None
        hidden_states.record_stream(self.stream)
        self.stream.wait_stream(current_stream())
        with torch.cuda.stream(self.stream):
            self.outputs[idx] = self.layer(hidden_states)
        return True

    def finish(self, hidden_states: torch.Tensor, overlapped: bool) -> torch.Tensor:
        if not overlapped:
            return self.layer(hidden_states)

        assert self.stream is not None
        current_stream().wait_stream(self.stream)
        idx = self.output_idx
        output = self.outputs[idx]
        assert output is not None
        self.outputs[idx] = None
        return output


def _kimi_k25_nvfp4_moe(
    hidden_states: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    layer = get_forward_context().no_compile_layers[layer_name]
    return layer._forward_impl(hidden_states)


def _kimi_k25_nvfp4_moe_fake(
    hidden_states: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    del layer_name
    return torch.empty_like(hidden_states)


direct_register_custom_op(
    op_name="kimi_k25_nvfp4_moe",
    op_func=_kimi_k25_nvfp4_moe,
    fake_impl=_kimi_k25_nvfp4_moe_fake,
    dispatch_key=current_platform.dispatch_key,
)


def _kimi_k25_nvfp4_moe_finalize_allreduce_norm(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    layer_name: str,
    norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    layer = get_forward_context().no_compile_layers[layer_name]
    return layer._forward_finalize_allreduce_norm_impl(
        hidden_states,
        residual,
        norm_weight,
        norm_eps,
    )


def _kimi_k25_nvfp4_moe_finalize_allreduce_norm_fake(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    layer_name: str,
    norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    del norm_weight, layer_name, norm_eps
    return torch.empty_like(hidden_states), torch.empty_like(residual)


direct_register_custom_op(
    op_name="kimi_k25_nvfp4_moe_finalize_allreduce_norm",
    op_func=_kimi_k25_nvfp4_moe_finalize_allreduce_norm,
    fake_impl=_kimi_k25_nvfp4_moe_finalize_allreduce_norm_fake,
    dispatch_key=current_platform.dispatch_key,
)


class KimiK25Nvfp4RoutedExperts(nn.Module):
    """Kimi-K2.5 routed experts for the FlashInfer TRTLLM NVFP4 path."""

    weight_loader = FusedMoE.weight_loader
    _load_per_tensor_weight_scale = FusedMoE._load_per_tensor_weight_scale
    _load_combined_w13_weight_scale = FusedMoE._load_combined_w13_weight_scale
    _load_model_weight_or_group_weight_scale = (
        FusedMoE._load_model_weight_or_group_weight_scale
    )
    _load_per_channel_weight_scale = FusedMoE._load_per_channel_weight_scale
    _get_hidden_dim = staticmethod(FusedMoE._get_hidden_dim)
    _narrow_expert_data_for_padding = staticmethod(
        FusedMoE._narrow_expert_data_for_padding
    )
    _load_w13 = FusedMoE._load_w13
    _load_w2 = FusedMoE._load_w2
    _load_single_value = FusedMoE._load_single_value
    _load_g_idx = FusedMoE._load_g_idx

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        config,
        quant_config: ModelOptNvFp4Config,
        prefix: str,
        e_score_correction_bias: torch.Tensor | None,
    ) -> None:
        super().__init__()
        if not quant_config.is_checkpoint_nvfp4_serialized:
            raise ValueError("Kimi-K2.5 NVFP4 MoE requires serialized NVFP4 weights.")

        self.layer_name = prefix
        self.moe_parallel_config = FusedMoEParallelConfig.make(
            tp_size_=get_tensor_model_parallel_world_size(),
            pcp_size_=get_pcp_group().world_size,
            dp_size_=get_dp_group().world_size,
            sp_size_=1,
            vllm_parallel_config=vllm_config.parallel_config,
        )
        self.tp_size = self.moe_parallel_config.tp_size
        self.tp_rank = self.moe_parallel_config.tp_rank
        self.ep_size = self.moe_parallel_config.ep_size
        self.ep_rank = self.moe_parallel_config.ep_rank
        self.global_num_experts = config.n_routed_experts
        self.logical_num_experts = config.n_routed_experts
        self.expert_placement_strategy = (
            vllm_config.parallel_config.expert_placement_strategy
        )
        self.top_k = config.num_experts_per_tok
        self.num_expert_group = getattr(config, "n_group", 1)
        self.topk_group = getattr(config, "topk_group", 1)
        self.e_score_correction_bias = e_score_correction_bias
        self.apply_router_weight_on_input = False
        self.shared_experts = None
        self.activation = MoEActivation.from_str(config.hidden_act)
        if self.activation != MoEActivation.SILU:
            raise ValueError("Kimi-K2.5 NVFP4 MoE only supports SiLU experts.")
        self.routing_method_type = int(RoutingMethodType.DeepSeekV3)
        self.activation_type = activation_to_flashinfer_int(self.activation)

        if self.moe_parallel_config.enable_eplb:
            raise ValueError("Kimi-K2.5 NVFP4 specialized MoE does not support EPLB.")
        if self.moe_parallel_config.use_all2all_kernels:
            raise ValueError(
                "Kimi-K2.5 NVFP4 specialized MoE does not support DP/EP all2all."
            )
        self.expert_placement_strategy = determine_expert_placement_strategy(
            expert_placement_strategy=self.expert_placement_strategy,
            moe_parallel_config=self.moe_parallel_config,
            num_expert_group=self.num_expert_group,
            num_redundant_experts=0,
            enable_eplb=False,
        )
        if (
            self.moe_parallel_config.use_ep
            and self.expert_placement_strategy != "linear"
        ):
            raise ValueError(
                "Kimi-K2.5 NVFP4 specialized MoE supports EP only with "
                "linear expert placement."
            )

        if self.moe_parallel_config.use_ep:
            self.local_num_experts, expert_map, expert_mask = determine_expert_map(
                ep_size=self.ep_size,
                ep_rank=self.ep_rank,
                global_num_experts=self.global_num_experts,
                expert_placement_strategy=self.expert_placement_strategy,
            )
            assert expert_map is not None
            self.register_buffer("_expert_map", expert_map)
            self.register_buffer("expert_mask", expert_mask)
            local_experts = (expert_map >= 0).nonzero().flatten()
            if local_experts.numel() == 0:
                self.local_expert_offset = 0
            else:
                self.local_expert_offset = int(local_experts[0].item())
            logger.info_once(
                "[EP Rank %s/%s] Kimi-K2.5 NVFP4 routed experts use %s "
                "placement. Local/global experts: %s/%s. Local map: %s.",
                self.ep_rank,
                self.ep_size,
                self.expert_placement_strategy,
                self.local_num_experts,
                self.global_num_experts,
                get_compressed_expert_map(self._expert_map),
            )
        else:
            self.local_num_experts = self.global_num_experts
            self._expert_map = None
            self.expert_mask = None
            self.local_expert_offset = 0

        if config.moe_intermediate_size % self.tp_size != 0:
            raise ValueError(
                "Kimi-K2.5 NVFP4 MoE requires moe_intermediate_size to be "
                f"divisible by TP size, got {config.moe_intermediate_size} "
                f"and TP={self.tp_size}."
            )
        intermediate_size_per_partition = config.moe_intermediate_size // self.tp_size

        self.moe_config = FusedMoEConfig(
            num_experts=self.global_num_experts,
            experts_per_token=self.top_k,
            hidden_dim=config.hidden_size,
            hidden_dim_unpadded=config.hidden_size,
            intermediate_size_per_partition=intermediate_size_per_partition,
            intermediate_size_per_partition_unpadded=intermediate_size_per_partition,
            num_local_experts=self.local_num_experts,
            num_logical_experts=self.logical_num_experts,
            moe_parallel_config=self.moe_parallel_config,
            in_dtype=vllm_config.model_config.dtype,
            moe_backend=vllm_config.kernel_config.moe_backend,
            router_logits_dtype=torch.float32,
            max_num_tokens=vllm_config.scheduler_config.max_num_batched_tokens,
            has_bias=False,
            is_act_and_mul=True,
            is_lora_enabled=vllm_config.lora_config is not None,
            activation=self.activation,
            device=vllm_config.device_config.device,
            routing_method=RoutingMethodType.DeepSeekV3,
            disable_inplace=True,
        )

        self.quant_method = ModelOptNvFp4FusedMoE(quant_config, self.moe_config)
        if (
            self.quant_method.nvfp4_backend != NvFp4MoeBackend.FLASHINFER_TRTLLM
            or self.quant_method.experts_cls is not TrtLlmNvFp4ExpertsMonolithic
        ):
            raise ValueError(
                "Kimi-K2.5 NVFP4 specialized MoE requires the FlashInfer "
                "TRTLLM monolithic NVFP4 backend."
            )

        self.quant_method.create_weights(
            layer=self,
            num_experts=self.local_num_experts,
            hidden_size=config.hidden_size,
            intermediate_size_per_partition=intermediate_size_per_partition,
            params_dtype=torch.get_default_dtype(),
            weight_loader=self.weight_loader,
            global_num_experts=self.global_num_experts,
        )

    def _map_global_expert_id_to_local_expert_id(self, expert_id: int) -> int:
        if self._expert_map is None:
            return expert_id
        return self._expert_map[expert_id].item()

    def _maybe_init_expert_routing_tables(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        return None

    @property
    def expert_map(self) -> torch.Tensor | None:
        return self._expert_map

    def update_expert_map(self) -> None:
        if not self.moe_parallel_config.use_ep:
            return None
        self.local_num_experts, expert_map, expert_mask = determine_expert_map(
            ep_size=self.ep_size,
            ep_rank=self.ep_rank,
            global_num_experts=self.global_num_experts,
            expert_placement_strategy=self.expert_placement_strategy,
        )
        assert expert_map is not None
        self.register_buffer("_expert_map", expert_map)
        self.register_buffer("expert_mask", expert_mask)
        local_experts = (expert_map >= 0).nonzero().flatten()
        self.local_expert_offset = (
            int(local_experts[0].item()) if local_experts.numel() > 0 else 0
        )
        self.moe_config.num_local_experts = self.local_num_experts
        return None

    def _run_flashinfer_trtllm_moe(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        *,
        routed_scaling_factor: float,
        do_finalize: bool,
        output: torch.Tensor | None,
        expert_weights: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        quant_config = self.quant_method.moe_quant_config
        if quant_config is None:
            raise RuntimeError("Kimi-K2.5 NVFP4 MoE weights were not post-processed.")

        original_hidden_dim = hidden_states.shape[-1]
        if not do_finalize and self.moe_config.hidden_dim != original_hidden_dim:
            raise RuntimeError(
                "Kimi-K2.5 MoE finalize all-reduce fusion requires an "
                "unpadded routed-expert hidden dimension."
            )
        if self.moe_config.hidden_dim != original_hidden_dim:
            hidden_states = F.pad(
                hidden_states,
                (0, self.moe_config.hidden_dim - original_hidden_dim),
                mode="constant",
                value=0.0,
            )

        assert quant_config.a1_gscale is not None
        hidden_states, hidden_states_scale = ops.scaled_fp4_quant(
            hidden_states,
            quant_config.a1_gscale,
            is_sf_swizzled_layout=False,
        )
        assert hidden_states_scale is not None
        assert quant_config.w1_scale is not None
        assert quant_config.w2_scale is not None
        assert quant_config.g1_alphas is not None
        assert quant_config.g2_alphas is not None
        assert hasattr(self, "g1_scale_c")

        routing_bias = self.e_score_correction_bias
        if routing_bias is not None:
            routing_bias = routing_bias.to(torch.bfloat16)

        from flashinfer.fused_moe.core import get_trtllm_moe_sm100_module

        return get_trtllm_moe_sm100_module().trtllm_fp4_block_scale_moe(
            router_logits.to(torch.float32),
            None,
            expert_weights,
            routing_bias,
            hidden_states,
            hidden_states_scale.view(torch.float8_e4m3fn).reshape(
                *hidden_states.shape[:-1], -1
            ),
            self.w13_weight,
            quant_config.w1_scale.view(torch.float8_e4m3fn),
            None,
            None,
            None,
            None,
            self.w2_weight,
            quant_config.w2_scale.view(torch.float8_e4m3fn),
            None,
            self.g1_scale_c,
            quant_config.g1_alphas,
            quant_config.g2_alphas,
            self.global_num_experts,
            self.top_k,
            self.num_expert_group,
            self.topk_group,
            self.moe_config.intermediate_size_per_partition,
            self.local_expert_offset,
            self.local_num_experts,
            routed_scaling_factor,
            self.routing_method_type,
            do_finalize,
            None,
            self.activation_type,
            output,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:
        original_hidden_dim = hidden_states.shape[-1]
        output = hidden_states.new_empty(
            (*hidden_states.shape[:-1], original_hidden_dim)
        )
        result = self._run_flashinfer_trtllm_moe(
            hidden_states,
            router_logits,
            routed_scaling_factor=1.0,
            do_finalize=True,
            output=output,
        )[0]
        if result.data_ptr() != output.data_ptr():
            output.copy_(result)
        return output[..., :original_hidden_dim]

    def forward_unfinalized(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        *,
        routed_scaling_factor: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        expert_weights = hidden_states.new_empty(
            (hidden_states.shape[0], self.top_k),
        )
        result = self._run_flashinfer_trtllm_moe(
            hidden_states,
            router_logits,
            routed_scaling_factor=routed_scaling_factor,
            do_finalize=False,
            output=None,
            expert_weights=expert_weights,
        )
        if len(result) != 3:
            raise RuntimeError(
                "FlashInfer TRTLLM NVFP4 MoE returned an unexpected result "
                f"for do_finalize=False: {len(result)} tensors."
            )
        expanded_idx_to_permuted_idx = result[2]
        if expanded_idx_to_permuted_idx.dim() == 1:
            expanded_idx_to_permuted_idx = expanded_idx_to_permuted_idx.view(
                -1,
                self.top_k,
            )
        elif expanded_idx_to_permuted_idx.shape[-1] != self.top_k:
            raise RuntimeError(
                "FlashInfer TRTLLM NVFP4 MoE returned an unexpected "
                "expanded-index shape for do_finalize=False: "
                f"{tuple(expanded_idx_to_permuted_idx.shape)}."
            )
        if result[1].dtype != expert_weights.dtype:
            raise RuntimeError(
                "FlashInfer TRTLLM NVFP4 MoE returned an unexpected "
                "expert-weight dtype for do_finalize=False: "
                f"{result[1].dtype}, expected {expert_weights.dtype}."
            )
        return result[0], result[1], expanded_idx_to_permuted_idx


class KimiK25Nvfp4MoE(nn.Module):
    """Inlined Kimi-K2.5 NVFP4 MoE for the FlashInfer TRTLLM backend."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        config,
        quant_config: ModelOptNvFp4Config,
        prefix: str,
    ) -> None:
        super().__init__()
        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self
        self.layer_name = prefix

        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)
        self.n_routed_experts = config.n_routed_experts
        self.n_shared_experts = config.n_shared_experts
        self.n_redundant_experts = 0
        self.n_logical_experts = self.n_routed_experts
        self.n_physical_experts = self.n_logical_experts
        self.n_local_physical_experts = self.n_physical_experts

        self.gate = GateLinear(
            config.hidden_size,
            config.n_routed_experts,
            out_dtype=torch.float32,
            prefix=f"{prefix}.gate",
        )
        if getattr(config, "topk_method", None) == "noaux_tc":
            self.gate.e_score_correction_bias = nn.Parameter(
                torch.empty(config.n_routed_experts, dtype=torch.float32)
            )
        else:
            raise ValueError("Kimi-K2.5 NVFP4 MoE requires noaux_tc routing.")

        if config.n_shared_experts is None:
            self.shared_experts = None
            self.shared_expert_overlap = None
        else:
            self.shared_experts = DeepseekV2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_intermediate_size
                * config.n_shared_experts,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                prefix=f"{prefix}.shared_experts",
            )
            self.shared_expert_overlap = _KimiK25SharedExpertOverlap(
                self.shared_experts,
                enable_dbo=vllm_config.parallel_config.enable_dbo,
            )

        self.experts = KimiK25Nvfp4RoutedExperts(
            vllm_config=vllm_config,
            config=config,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            e_score_correction_bias=self.gate.e_score_correction_bias,
        )
        self.ep_size = self.experts.ep_size
        self.ep_rank = self.experts.ep_rank
        self.n_local_physical_experts = self.experts.local_num_experts
        self.physical_expert_start = self.experts.local_expert_offset
        self.physical_expert_end = (
            self.physical_expert_start + self.n_local_physical_experts
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.ops.vllm.kimi_k25_nvfp4_moe(
            hidden_states,
            self.layer_name,
        )

    def _can_use_trtllm_moe_finalize_allreduce_norm(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        norm_layer: RMSNorm,
    ) -> bool:
        if _kimi_moe_finalize_ar_workspace_disabled:
            return False
        if not current_platform.is_cuda() or not hidden_states.is_cuda:
            return False
        if hidden_states.dtype not in (torch.float16, torch.bfloat16):
            return False
        if hidden_states.dim() != 2 or residual.dim() != 2:
            return False
        if not hidden_states.is_contiguous() or not residual.is_contiguous():
            return False
        if norm_layer.variance_size_override is not None:
            return False
        if not norm_layer.weight.is_contiguous():
            return False

        tp_group = get_tp_group()
        if tp_group.world_size not in _KIMI_MOE_FINALIZE_AR_WORLD_SIZES:
            return False

        # The existing eager path has a special fp16 shared-expert scaling
        # ordering. Keep that exact split unless the dtype follows Kimi's bf16
        # path, where applying the routed scale before finalize is equivalent.
        return not (
            hidden_states.dtype == torch.float16
            and self.shared_expert_overlap is not None
            and self.routed_scaling_factor != 1.0
        )

    def forward_finalize_allreduce_norm(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        norm_layer: RMSNorm,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if not self._can_use_trtllm_moe_finalize_allreduce_norm(
            hidden_states,
            residual,
            norm_layer,
        ):
            return None

        return torch.ops.vllm.kimi_k25_nvfp4_moe_finalize_allreduce_norm(
            hidden_states,
            residual,
            norm_layer.weight,
            self.layer_name,
            float(norm_layer.variance_epsilon),
        )

    def _fallback_finalize_norm(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        norm_weight: torch.Tensor,
        norm_eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = self._forward_impl(hidden_states)
        result = RMSNorm.forward_static(
            output,
            norm_eps,
            int(norm_weight.shape[0]),
            output.dtype,
            weight=norm_weight,
            residual=residual,
        )
        assert isinstance(result, tuple)
        return result

    def _forward_finalize_allreduce_norm_impl(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        norm_weight: torch.Tensor,
        norm_eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            import flashinfer.comm as flashinfer_comm
        except ImportError:
            return self._fallback_finalize_norm(
                hidden_states,
                residual,
                norm_weight,
                norm_eps,
            )
        if not hasattr(flashinfer_comm, "trtllm_moe_finalize_allreduce_fusion"):
            return self._fallback_finalize_norm(
                hidden_states,
                residual,
                norm_weight,
                norm_eps,
            )

        _, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        residual = residual.view(-1, hidden_dim)

        shared_overlapped = False
        if self.shared_expert_overlap is not None:
            shared_overlapped = self.shared_expert_overlap.start(hidden_states)

        router_logits, _ = self.gate(hidden_states)
        (
            allreduce_in,
            expert_weights,
            expanded_idx_to_permuted_idx,
        ) = self.experts.forward_unfinalized(
            hidden_states,
            router_logits,
            routed_scaling_factor=float(self.routed_scaling_factor),
        )

        shared_output = None
        if self.shared_expert_overlap is not None:
            shared_output = self.shared_expert_overlap.finish(
                hidden_states,
                shared_overlapped,
            )

        tp_group = get_tp_group()
        workspace_token_num = max(1, int(allreduce_in.numel() // hidden_dim))
        workspace = _get_kimi_moe_finalize_ar_workspace(
            world_size=tp_group.world_size,
            rank=tp_group.rank_in_group,
            max_token_num=workspace_token_num,
            hidden_dim=hidden_dim,
            dtype=hidden_states.dtype,
            group=tp_group.device_group,
        )
        if workspace is None:
            return self._fallback_finalize_norm(
                hidden_states,
                residual,
                norm_weight,
                norm_eps,
            )

        norm_out = torch.empty_like(hidden_states)
        residual_out = torch.empty_like(residual)
        try:
            flashinfer_comm.trtllm_moe_finalize_allreduce_fusion(
                allreduce_in=allreduce_in,
                residual_in=residual,
                norm_weight=norm_weight,
                expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
                norm_out=norm_out,
                residual_out=residual_out,
                workspace_ptrs=workspace.workspace_tensor,
                launch_with_pdl=True,
                world_rank=tp_group.rank_in_group,
                world_size=tp_group.world_size,
                eps=norm_eps,
                shared_expert_output=shared_output,
                expert_scale_factor=expert_weights,
            )
        except ValueError as e:
            logger.warning_once(
                "FlashInfer TRTLLM MoE finalize all-reduce fusion rejected "
                "this problem size: %s. Falling back to the unfused tail.",
                e,
            )
            return self._fallback_finalize_norm(
                hidden_states,
                residual,
                norm_weight,
                norm_eps,
            )
        return norm_out, residual_out

    def _forward_impl(self, hidden_states: torch.Tensor) -> torch.Tensor:
        _, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        shared_overlapped = False
        if self.shared_expert_overlap is not None:
            shared_overlapped = self.shared_expert_overlap.start(hidden_states)

        router_logits, _ = self.gate(hidden_states)
        routed_output = self.experts(hidden_states, router_logits)

        shared_output = None
        if self.shared_expert_overlap is not None:
            shared_output = self.shared_expert_overlap.finish(
                hidden_states,
                shared_overlapped,
            )

        if self.routed_scaling_factor != 1.0:
            if routed_output.dtype != torch.float16 or shared_output is None:
                routed_output = routed_output * self.routed_scaling_factor
            else:
                shared_output = shared_output * (1.0 / self.routed_scaling_factor)
        if shared_output is not None:
            output = shared_output + routed_output
        else:
            output = routed_output

        if self.tp_size > 1 or self.experts.ep_size > 1:
            output = tensor_model_parallel_all_reduce(output)
        return output


class KimiK25Nvfp4DecoderLayer(nn.Module):
    """Single inlined decoder layer for Kimi-K2.5 NVFP4."""

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

        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        if not isinstance(quant_config, ModelOptNvFp4Config):
            raise ValueError(
                "Kimi-K2.5 NVFP4 specialized model requires ModelOpt NVFP4."
            )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = ForkedKimiK25Nvfp4MLAAttention(
            vllm_config=vllm_config,
            config=config,
            cache_config=cache_config,
            quant_config=quant_config,
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
        if self.is_moe:
            self.mlp = KimiK25Nvfp4MoE(
                vllm_config=vllm_config,
                config=config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = DeepseekV2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states, residual, _ = self.forward_with_optional_fused_moe_tail(
            positions=positions,
            hidden_states=hidden_states,
            residual=residual,
            skip_input_layernorm=False,
            next_input_layernorm=None,
        )
        return hidden_states, residual

    def forward_with_optional_fused_moe_tail(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        *,
        skip_input_layernorm: bool,
        next_input_layernorm: RMSNorm | None,
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        if skip_input_layernorm:
            assert residual is not None
        elif residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        assert residual is not None
        hidden_states = self.self_attn(positions, hidden_states)

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        if (
            next_input_layernorm is not None
            and self.is_moe
            and isinstance(self.mlp, KimiK25Nvfp4MoE)
        ):
            fused_result = self.mlp.forward_finalize_allreduce_norm(
                hidden_states,
                residual,
                next_input_layernorm,
            )
            if fused_result is not None:
                return fused_result[0], fused_result[1], True

        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual, False

    def fuse_shared_expert_act_quant(self) -> None:
        if not self.is_moe:
            return

        shared_experts = self.mlp.shared_experts
        if shared_experts is None:
            return
        if not isinstance(
            shared_experts.down_proj.quant_method, ModelOptNvFp4LinearMethod
        ):
            return

        down_proj = shared_experts.down_proj

        def _fused_forward(x: torch.Tensor) -> torch.Tensor:
            gate_up, _ = shared_experts.gate_up_proj(x)
            out_shape = gate_up.shape[:-1] + (gate_up.shape[-1] // 4,)
            bs_shape = gate_up.shape[:-1] + (gate_up.shape[-1] // 64,)
            x_fp4 = torch.empty(out_shape, dtype=torch.uint8, device=gate_up.device)
            x_bs = torch.empty(
                bs_shape,
                dtype=current_platform.fp8_dtype(),
                device=gate_up.device,
            )
            torch.ops._C.silu_and_mul_nvfp4_quant(
                x_fp4,
                x_bs,
                gate_up,
                down_proj.input_global_scale_inv,
            )
            return flashinfer_scaled_fp4_mm(
                x_fp4,
                down_proj.weight,
                x_bs,
                down_proj.weight_scale,
                down_proj.alpha,
                gate_up.dtype,
                backend="cutlass",
            )

        shared_experts.forward = _fused_forward  # type: ignore[method-assign]


@support_torch_compile
class KimiK25Nvfp4TextModel(nn.Module):
    """Text-only model body for Kimi-K2.5 NVFP4."""

    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.device = current_platform.device_type

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.embed_tokens",
        )
        self.layers = nn.ModuleList(
            [
                KimiK25Nvfp4DecoderLayer(
                    vllm_config=vllm_config,
                    config=config,
                    layer_idx=i,
                    prefix=f"{prefix}.layers.{i}",
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"],
            config.hidden_size,
        )
        self.aux_hidden_state_layers = tuple[int, ...]()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if intermediate_tensors is not None:
            raise ValueError(
                "Kimi-K2.5 NVFP4 specialized text model does not support PP."
            )
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            if input_ids is None:
                raise ValueError("Either input_ids or inputs_embeds must be provided.")
            hidden_states = self.embed_input_ids(input_ids)

        residual = None
        aux_hidden_states = []
        input_layernorm_done = False
        for idx, layer in enumerate(self.layers):
            if idx in self.aux_hidden_state_layers:
                if residual is None:
                    aux_hidden_states.append(hidden_states)
                else:
                    aux_hidden_states.append(hidden_states + residual)

            next_input_layernorm = None
            if not self.aux_hidden_state_layers:
                if idx + 1 < len(self.layers):
                    next_input_layernorm = self.layers[idx + 1].input_layernorm
                else:
                    next_input_layernorm = self.norm

            hidden_states, residual, input_layernorm_done = (
                layer.forward_with_optional_fused_moe_tail(
                    positions=positions,
                    hidden_states=hidden_states,
                    residual=residual,
                    skip_input_layernorm=input_layernorm_done,
                    next_input_layernorm=next_input_layernorm,
                )
            )

        if not input_layernorm_done:
            hidden_states, _ = self.norm(hidden_states, residual)
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states


class KimiK25Nvfp4TextForCausalLM(DeepseekV2ForCausalLM):
    model_cls = KimiK25Nvfp4TextModel

    def set_moe_parameters(self):
        self.expert_weights = []
        self.num_expert_groups = getattr(self.config, "n_group", 1)

        self.moe_layers = []
        self.moe_mlp_layers = []
        example_moe = None
        for layer in self.model.layers:
            if isinstance(layer, KimiK25Nvfp4DecoderLayer) and isinstance(
                layer.mlp, KimiK25Nvfp4MoE
            ):
                example_moe = layer.mlp
                self.moe_mlp_layers.append(layer.mlp)
                self.moe_layers.append(layer.mlp.experts)

        self.extract_moe_parameters(example_moe)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded = super().load_weights(weights)
        # TODO: re-enable after fixing inference correctness
        # for layer in self.model.layers:
        #     layer.fuse_shared_expert_act_quant()
        return loaded


class KimiK25ForConditionalGeneration(
    nn.Module,
    SupportsPP,
    SupportsQuant,
    SupportsEagle,
    SupportsEagle3,
):
    """Text-only Kimi-K2.5 wrapper for the NVFP4 checkpoint."""

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "language_model.layers.": "language_model.model.layers.",
        }
    )

    def __new__(
        cls,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> nn.Module:
        if cls is KimiK25ForConditionalGeneration:
            reason = _get_kimi_nvfp4_specialization_rejection_reason(vllm_config)
            if reason is not None:
                return _fallback_to_generic_kimi_k25(vllm_config, prefix, reason)
        return super().__new__(cls)

    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.quant_config = vllm_config.quant_config
        text_vllm_config = vllm_config.with_hf_config(config.text_config)

        if not _is_target_kimi_nvfp4(vllm_config):
            raise ValueError(
                "The Kimi-K2.5 specialized NVFP4 model was selected despite "
                "not meeting its runtime requirements."
            )

        self.language_model = KimiK25Nvfp4TextForCausalLM(
            vllm_config=text_vllm_config,
            prefix=f"{prefix}.language_model" if prefix else "language_model",
        )
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.language_model.set_aux_hidden_state_layers(layers)

    def get_eagle3_aux_hidden_state_layers(self) -> tuple[int, ...]:
        return self.language_model.get_eagle3_aux_hidden_state_layers()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.language_model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> IntermediateTensors:
        del kwargs
        return self.language_model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def compute_logits(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        del kwargs
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["vision_tower.", "mm_projector."],
            ignore_unexpected_prefixes=["vision_tower.", "mm_projector."],
        )
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
