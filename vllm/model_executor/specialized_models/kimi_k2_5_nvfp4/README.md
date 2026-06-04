# Kimi-K2.5 NVFP4 specialized kernels

This package hosts the custom [CuTe DSL](https://docs.nvidia.com/cutlass/)
kernels used by the upcoming `nvidia/Kimi-K2.5-NVFP4` specialized model. The
kernels are checked in ahead of the model so they can be reviewed and tested in
isolation; the specialized model will import them from
[`kernels.py`](./kernels.py) once it lands.

All kernels target Blackwell (SM10x) GPUs and require the optional `cutlass`
(`nvidia-cutlass-dsl`) dependency. They are compiled lazily at first call and
cached per-process via an executor cache keyed on tensor layout and the active
CUDA device.

## Kernels

The MLA attention path for this checkpoint fixes `kv_lora_rank = q_lora_dim =
512` and `qk_rope_head_dim = 64`; the kernels bake in those shapes.

| Public helper | Fused work |
| --- | --- |
| `_run_kimik25_concat_and_cache_mla` | FP8-quantize `kv_c`/`k_pe` and write the paged MLA KV cache |
| `_run_kimik25_rmsnorm_special_qkv_fused` | Split RMSNorm over the fused Q/KV LoRA projection plus key RoPE |
| `_run_kimik25_rope` / `kimik25_rope` | In-place decode-query RoPE |
| `_run_kimik25_decode_rope_concat_quant_fp8` | Decode-query RoPE, concat, and FP8 quantization |
| `_run_kimik25_decode_rope_concat_quant_fp8_and_cache_mla` | The decode quant path fused with the KV-cache write |
| `kimi_fused_rmsnorm` | Standalone split (Q/KV) RMSNorm launcher |

## Testing

The kernels are validated against PyTorch / vLLM reference implementations in
[`tests/model_executor/specialized_models/kimi_k2_5_nvfp4`](../../../../tests/model_executor/specialized_models/kimi_k2_5_nvfp4).
The tests skip automatically unless they run on a Blackwell GPU with `cutlass`
installed:

```bash
pytest tests/model_executor/specialized_models/kimi_k2_5_nvfp4
```
