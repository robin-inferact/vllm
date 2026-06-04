# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Specialized Kimi-K2.5 NVFP4 model package.

Currently this package only ships the custom CuTe DSL kernels in
:mod:`vllm.model_executor.specialized_models.kimi_k2_5_nvfp4.kernels`.
The kernels are imported explicitly (rather than re-exported here) so that
importing this package does not require the optional ``cutlass`` dependency
or a Blackwell GPU.
"""
