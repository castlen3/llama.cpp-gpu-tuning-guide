# KV Cache Sizing

KV cache grows with context length, layer count, KV heads, head dim, and dtype.

## Quick Method: Read the Log

The most reliable value is in the startup log:

```
llama_kv_cache:      GPU KV buffer size =  XXX MiB
llama_kv_cache:      CPU KV buffer size =  YYY MiB
```

Run with `--verbose` and your target `-c` to get exact numbers — no estimation needed.

## Estimation Formula

```
KV_bytes = 2 * n_kv_layers * n_kv_heads * head_dim * ctx * bytes_per_element
```

Where:

- `n_kv_layers`: layers that store KV cache (may be fewer than total layers for hybrid architectures)
- `n_kv_heads`: from GGUF metadata `attention.head_count_kv`
- `head_dim`: from `attention.key_length` or `attention.value_length`
- `bytes_per_element`: 1 for q8_0, 0.5 for q4_0
- Factor 2: one for K, one for V

## Scaling with Context

KV cache scales linearly with `-c`:

```text
ctx=4096   -> ~X MiB
ctx=8192   -> ~2X MiB
ctx=16384  -> ~4X MiB
ctx=24576  -> ~6X MiB
ctx=32768  -> ~8X MiB
```

Actual values depend on model architecture. Use the startup log as source of truth.

## Factors Affecting Actual Size

- **GQA ratio**: Higher ratio = fewer KV heads = smaller cache
- **Sliding window (SWA)**: Architectures like Gemma 4 use sliding window attention on some layers, which keep only a small window (e.g. 1024 tokens) instead of the full context. This can shrink KV cache by 50-70% vs the naive formula. **Always verify with the startup log** — estimation alone will overstate VRAM usage for SWA models.

  *Example:* Gemma 4 12B, 48 layers, 8 SWA layers (1024-token window), 40 full-attention layers. At 80K ctx q8_0, the naive formula predicts ~14 GB KV cache. Actual measured: ~765 MB — over 18x less. This is because SWA layers contribute negligible cache and the non-SWA layers use far fewer KV heads than the raw head_count_kv suggests.
- **Recurrent/SSM layers**: These don't store KV cache at all (e.g., hybrid models store KV only on full-attention layers)
- **Quantization implementation**: Actual bytes may include alignment padding
- **Flash attention**: May change buffer layout but not total size

## Mixed Dtype Warning

On CUDA, mixing KV cache types (e.g. `-ctk q8_0 -ctv q4_0`) may cause prefill stalls or hangs. Use symmetric settings:

- `-ctk q8_0 -ctv q8_0`
- `-ctk q4_0 -ctv q4_0`

Other backends (Vulkan, ROCm) may not have this issue. Verify on your specific build.

## VRAM Budgeting

When computing total VRAM budget:

```
total = model_weights_on_gpu + KV_cache_on_gpu + compute_buffer (~200 MB) + recurrent_state (~100 MB)
```

If VRAM is tight, the most impactful reduction is lowering `-c`, followed by switching KV to q4_0.

## DType Selection Guide

| Scenario | Recommended | Reason |
|----------|-------------|--------|
| VRAM >= 12GB, context <= 16K | `q8_0/q8_0` | Best precision, fits easily |
| VRAM >= 12GB, context > 16K | `q8_0/q8_0` or `q4_0/q4_0` | Monitor VRAM, switch if tight |
| VRAM < 12GB | `q4_0/q4_0` | Must conserve VRAM |
| CUDA + any | Symmetric only | Avoid prefill stalls |
