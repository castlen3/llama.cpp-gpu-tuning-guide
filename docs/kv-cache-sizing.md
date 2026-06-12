# KV Cache Sizing

## Quick Method: Read the Log

The most reliable value is in the startup log:

```
llama_kv_cache:      CUDA0 KV buffer size =  XXX MiB
llama_kv_cache:        CPU KV buffer size =  YYY MiB
```

Run with `--verbose` and your target `-c` to get exact numbers — no estimation needed.

## Estimation Formula

```
KV_bytes ≈ 2 × n_full_attention_layers × n_kv_heads × head_dim × ctx × bytes_per_element
```

Where:
- `n_full_attention_layers`: layers with full attention (not SSM/recurrent). For qwen35, this is every 4th layer (~16 out of 64).
- `n_kv_heads`: from GGUF metadata `attention.head_count_kv`
- `head_dim`: from `attention.key_length` or `attention.value_length`
- `bytes_per_element`: 1 for q8_0, 0.5 for q4_0
- Factor 2: one for K, one for V

## Example: Qwen3.6 27B (qwen35)

```
n_full_attention_layers = 16
n_kv_heads = 4
head_dim = 256
ctx = 24576
kv_type = q8_0 (1 byte/element)

KV = 2 × 16 × 4 × 256 × 24576 × 1
   = 805,306,368 bytes
   ≈ 768 MiB total
   ≈ 384 MiB on CUDA, 384 MiB on CPU (split evenly in this architecture)
```

Actual log value may differ due to alignment, padding, and architecture-specific buffers.

## Factors Affecting Actual Size

- **GQA ratio**: Higher ratio = fewer KV heads = smaller cache
- **Sliding window**: Some architectures use smaller window for local attention
- **Recurrent/SSM layers**: These don't store KV cache at all (e.g., qwen35 stores KV only on full-attention layers)
- **Quantization implementation**: Actual bytes may include alignment padding
- **Flash attention**: May change buffer layout but not total size

## Scaling with Context

KV cache scales **linearly** with `-c`:

```
ctx=4096  → ~128 MiB
ctx=8192  → ~256 MiB
ctx=16384 → ~512 MiB
ctx=24576 → ~768 MiB
ctx=32768 → ~1024 MiB
```

Example with Qwen3.6 27B, q8_0.

## Mixed Dtype Warning ⚠️

Some CUDA backends exhibit prefill stalls or hangs with asymmetric KV types (e.g. `-ctk q8_0 -ctv q4_0`). Use symmetric settings:
- `-ctk q8_0 -ctv q8_0`
- `-ctk q4_0 -ctv q4_0`

## VRAM Budgeting

When computing total VRAM budget:

```
total = model_weights_on_gpu + KV_cache_on_gpu + compute_buffer (~200 MB) + recurrent_state (~100 MB)
```

If VRAM is tight, the most impactful reduction is lowering `-c`, followed by switching KV to q4_0.
