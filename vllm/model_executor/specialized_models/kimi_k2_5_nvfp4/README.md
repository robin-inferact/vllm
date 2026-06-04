# Kimi-K2.5 NVFP4 specialized kernels

This package hosts the custom [CuTe DSL](https://docs.nvidia.com/cutlass/)
kernels used by the upcoming `nvidia/Kimi-K2.5-NVFP4` specialized model. The
kernels are checked in ahead of the model so they can be reviewed and tested in
isolation; the specialized model will import them from [`kernels.py`](./kernels.py)
once it lands.

All kernels target **Blackwell (SM10x)** GPUs and require the optional `cutlass`
(`nvidia-cutlass-dsl`) dependency. They specialize the MLA attention path for
this checkpoint, which fixes:

- `kv_lora_rank == q_lora_dim == 512`
- `qk_rope_head_dim` (`pe_dim`) `== 64`
- `bfloat16` activations, `e4m3` FP8 paged KV cache.

Inputs are `bfloat16`; FP8 quantization converts to `e4m3` stored as `uint8`
(returned as the platform FP8 dtype). RoPE is the interleaved (non-NeoX) MLA
variant: the two halves of each rotary pair are adjacent in memory.

## Compilation & caching convention

Each kernel is a `@cute.kernel` device function plus a `@cute.jit` launcher.
Compilation is cached following the convention in
[`vllm/v1/attention/ops/deepseek_v4_ops`](../../../v1/attention/ops/deepseek_v4_ops):

- A `functools.cache`-decorated `_compile_*` helper builds **fake tensors**
  (`cute.runtime.make_fake_tensor`) with symbolic shapes/strides
  (`cute.sym_int` / `cute.sym_int64`) and a fake stream, then calls
  `cute.compile(..., options="--enable-tvm-ffi")`. The cache key is the set of
  compile-time (constexpr) parameters only, so a kernel compiles once per
  configuration and is reused across token counts.
- The compiled executor is invoked **directly with torch tensors** and sources
  its launch stream from the TVM-FFI environment, so the public `_run_*`
  wrappers do not build CuTe tensors or pass a stream at call time.

## Kernels

Public entry points are the `_run_*` helpers (torch-tensor in / out).

### `_run_kimik25_concat_and_cache_mla`
FP8-quantizes the MLA latent KV (`kv_c`, 512 dims) and the rotary key part
(`k_pe`, 64 dims) for each token and writes them into the paged FP8 KV cache at
the `slot_mapping` slots. Values are divided by `scale` and converted to `e4m3`.
The grid is one block per `(token, split)`: `split 0` writes the 64 `k_pe`
elements, `splits 1..kv_cache_block_factor` (4) each write a 128-element chunk of
`kv_c`. Tokens with `slot_idx < 0` are skipped (padding).

### `_run_kimik25_rmsnorm_special_qkv_fused`
One launch that fuses the per-token work over the fused Q/KV LoRA projection
(`data`, width `lora_dim_q + lora_dim_kv`, written in place) and the key RoPE
(`k_pe`, in place). The grid has three block kinds over `Sp` tokens:
1. Q-LoRA RMSNorm with `weights_q`/`eps_q` (the first three of four column groups,
   since `lora_dim_q == 3 * lora_dim_kv`),
2. KV-LoRA RMSNorm with `weights_kv`/`eps_kv` (the fourth group),
3. interleaved RoPE on `k_pe` from `positions` + `cos_sin_cache`.

### `_run_kimik25_rope` (jit: `kimik25_rope`)
In-place interleaved RoPE on the decode query rotary part `query`
`(Sp, num_local_heads, 64)`. Each thread rotates `K = 8` heads using
`cos_sin_cache` indexed by `positions`.

### `_run_kimik25_decode_rope_concat_quant_fp8`
Decode query path: applies RoPE to `q_pe` `(Sp, num_heads, 64)`, concatenates it
after `ql_nope` `(Sp, num_heads, 512)`, and FP8-quantizes the full
576-wide query (divided by `scale`) into a fresh `uint8` buffer returned as FP8.
Fuses RoPE + concat + quantization in a single launch.

### `_run_kimik25_decode_rope_concat_quant_fp8_and_cache_mla`
The decode query path above fused with the KV-cache write: a single linearized
grid covers both the cache-write blocks (quantizing `kv_c`/`k_pe` by `kv_scale`
into the paged cache, as in `concat_and_cache_mla`) and the decode-query blocks
(quantizing the rotated/concatenated query by `q_scale`). Returns the quantized
query (FP8) and updates `kv_cache` in place.

### `kimi_fused_rmsnorm` (jit launcher)
A standalone split RMSNorm: the two RMSNorms of
`rmsnorm_special_qkv_fused` (Q-LoRA with `weights_q`, KV-LoRA with `weights_kv`)
without the RoPE step, over the fused `data` buffer in place.

## Testing

The kernels are validated against PyTorch / vLLM reference implementations in
[`tests/model_executor/specialized_models/kimi_k2_5_nvfp4`](../../../../tests/model_executor/specialized_models/kimi_k2_5_nvfp4).
The tests skip automatically unless they run on a Blackwell GPU with `cutlass`
installed:

```bash
pytest tests/model_executor/specialized_models/kimi_k2_5_nvfp4
```
