# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Text-only specialized Kimi-K2.5 NVFP4 implementation."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

from vllm import _custom_ops as ops
from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention.mla_attention import MLAAttention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear
from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4LinearMethod
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.models.deepseek_v2 import (
    DeepseekV2ForCausalLM,
    DeepSeekV2FusedQkvAProjLinear,
    DeepseekV2MLP,
    DeepseekV2MoE,
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
from vllm.utils.flashinfer import flashinfer_scaled_fp4_mm
from vllm.utils.torch_utils import (
    direct_register_custom_op,
    is_quantized_kv_cache,
)

_TARGET_MODEL_NAMES = {"nvidia/Kimi-K2.5-NVFP4"}


def _is_target_kimi_nvfp4(vllm_config: VllmConfig) -> bool:
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


direct_register_custom_op(
    op_name="monolithic_attn",
    op_func=_kimi_mla_attn,
    fake_impl=_kimi_mla_attn_fake,
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
        parallel_config = vllm_config.parallel_config

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = KimiK25Nvfp4MLAAttention(
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
            self.mlp = DeepseekV2MoE(
                config=config,
                parallel_config=parallel_config,
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
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(positions, hidden_states)

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual

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
        for idx, layer in enumerate(self.layers):
            if idx in self.aux_hidden_state_layers:
                aux_hidden_states.append(hidden_states + residual)
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )

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
                layer.mlp, DeepseekV2MoE
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

    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.quant_config = vllm_config.quant_config
        text_vllm_config = vllm_config.with_hf_config(config.text_config)

        if not _is_target_kimi_nvfp4(vllm_config):
            raise ValueError(
                "The Kimi-K2.5 specialized NVFP4 model only supports "
                "`nvidia/Kimi-K2.5-NVFP4`."
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
