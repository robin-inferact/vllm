# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

if not torch.cuda.is_available() or not current_platform.is_device_capability_family(
    100
):
    pytest.skip(
        "This test only runs on Blackwell GPUs (SM10x).", allow_module_level=True
    )

from vllm.model_executor.specialized_models.kimi_k2_5_nvfp4 import model as kimi_model


@pytest.mark.parametrize("num_tokens", [1, 2, 4])
def test_router_logits_uses_triton_for_small_batches_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
    num_tokens: int,
) -> None:
    monkeypatch.setenv("VLLM_KIMI_ROUTER_GEMM", "triton_splitk_row")

    expected = torch.empty(0)
    calls: list[tuple[int, int]] = []

    def fake_triton(
        self: kimi_model.KimiK25Nvfp4MoE,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        calls.append(tuple(hidden_states.shape))
        return expected

    def fake_cublaslt(
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        raise AssertionError("small-batch router GEMM should use Triton")

    monkeypatch.setattr(
        kimi_model.KimiK25Nvfp4MoE,
        "_router_logits_triton_splitk_row",
        fake_triton,
    )
    monkeypatch.setattr(
        kimi_model.ops,
        "kimi_k25_router_gemm_bf16_fp32_cublaslt",
        fake_cublaslt,
    )

    moe = kimi_model.KimiK25Nvfp4MoE.__new__(kimi_model.KimiK25Nvfp4MoE)
    hidden_states = torch.empty(num_tokens, 7168)

    assert moe._router_logits(hidden_states) is expected
    assert calls == [(num_tokens, 7168)]


@pytest.mark.parametrize("num_tokens", [1, 4])
def test_router_logits_uses_torch_mm_for_small_batches_by_default(
    monkeypatch: pytest.MonkeyPatch,
    num_tokens: int,
) -> None:
    monkeypatch.delenv("VLLM_KIMI_ROUTER_GEMM", raising=False)

    expected = torch.empty(0)
    calls: list[tuple[tuple[int, int], tuple[int, int], torch.dtype | None]] = []

    def fake_triton(
        self: kimi_model.KimiK25Nvfp4MoE,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        raise AssertionError("default router GEMM should use torch.mm")

    def fake_cublaslt(
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        raise AssertionError("default router GEMM should use torch.mm")

    def fake_mm(
        left: torch.Tensor,
        right: torch.Tensor,
        *,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        calls.append((tuple(left.shape), tuple(right.shape), out_dtype))
        return expected

    monkeypatch.setattr(
        kimi_model.KimiK25Nvfp4MoE,
        "_router_logits_triton_splitk_row",
        fake_triton,
    )
    monkeypatch.setattr(
        kimi_model.ops,
        "kimi_k25_router_gemm_bf16_fp32_cublaslt",
        fake_cublaslt,
    )
    monkeypatch.setattr(kimi_model.torch, "mm", fake_mm)

    moe = kimi_model.KimiK25Nvfp4MoE.__new__(kimi_model.KimiK25Nvfp4MoE)
    object.__setattr__(moe, "gate", SimpleNamespace(weight=torch.empty(384, 7168)))
    hidden_states = torch.empty(num_tokens, 7168)

    assert moe._router_logits(hidden_states) is expected
    assert calls == [((num_tokens, 7168), (7168, 384), torch.float32)]


def test_router_logits_uses_torch_mm_for_larger_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_KIMI_ROUTER_GEMM", "triton_splitk_row")

    expected = torch.empty(0)
    calls: list[tuple[tuple[int, int], tuple[int, int], torch.dtype | None]] = []

    def fake_triton(
        self: kimi_model.KimiK25Nvfp4MoE,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        raise AssertionError("large-batch router GEMM should use torch.mm")

    def fake_cublaslt(
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        raise AssertionError("large-batch router GEMM should use torch.mm")

    def fake_mm(
        left: torch.Tensor,
        right: torch.Tensor,
        *,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        calls.append((tuple(left.shape), tuple(right.shape), out_dtype))
        return expected

    monkeypatch.setattr(
        kimi_model.KimiK25Nvfp4MoE,
        "_router_logits_triton_splitk_row",
        fake_triton,
    )
    monkeypatch.setattr(
        kimi_model.ops,
        "kimi_k25_router_gemm_bf16_fp32_cublaslt",
        fake_cublaslt,
    )
    monkeypatch.setattr(kimi_model.torch, "mm", fake_mm)

    moe = kimi_model.KimiK25Nvfp4MoE.__new__(kimi_model.KimiK25Nvfp4MoE)
    object.__setattr__(moe, "gate", SimpleNamespace(weight=torch.empty(384, 7168)))
    hidden_states = torch.empty(5, 7168)

    assert moe._router_logits(hidden_states) is expected
    assert calls == [((5, 7168), (7168, 384), torch.float32)]
