# MoE Model Tuning

For mixture-of-experts models (`n_routed_experts` present in GGUF metadata).

Examples: Qwen3.6-35B-A3B, Mixtral, DeepSeek-V2.

## Key Difference from Dense

MoE has two independent GPU offload dimensions:

| Parameter | What it offloads | Impact |
|-----------|-----------------|--------|
| `-ngl` | Attention layers + shared expert | Prefill speed |
| `--n-cpu-moe` | Routed experts kept on CPU | **Decode speed** (larger impact) |

`--n-cpu-moe` often has greater effect on decode tok/s than `-ngl`.

## Prerequisites

- [CUDA backend verified](cuda-backend-check.md)
- GGUF metadata: `block_count`, `n_routed_experts`, `num_experts_per_tok`
- All llama processes killed before each bench run

## Phase 1: n_cpu_moe Sweep (Most Impactful)

### Goal
Find the `--n-cpu-moe` value that maximizes decode speed.

### Strategy

```
1. Fix ngl at: total_layers × 0.5
2. Sweep: --n-cpu-moe 0, 4, 8, 12, 16, 20, 24, 28, 32
3. Step size: 4~8
4. Look for highest tg128 (decode tok/s)
```

### Example

```bat
llama-bench -ngl 40 --n-cpu-moe 0  -c 4096 ...
llama-bench -ngl 40 --n-cpu-moe 8  -c 4096 ...
llama-bench -ngl 40 --n-cpu-moe 16 -c 4096 ...
llama-bench -ngl 40 --n-cpu-moe 24 -c 4096 ...
llama-bench -ngl 40 --n-cpu-moe 28 -c 4096 ...
llama-bench -ngl 40 --n-cpu-moe 32 -c 4096 ...
```

### Judgment

- `--n-cpu-moe = 0`: all experts on GPU → fastest if VRAM fits
- `--n-cpu-moe > 0`: offloads some expert compute to CPU → saves VRAM, decode may still be fast if CPU is strong
- Typical sweet spot: `total_experts × 0.6` to `total_experts × 0.75`

## Phase 2: ngl Sweep

### Goal
Fine-tune ngl with n_cpu_moe fixed at best value.

```bat
llama-bench --n-cpu-moe <best> -ngl 32 ...
llama-bench --n-cpu-moe <best> -ngl 36 ...
llama-bench --n-cpu-moe <best> -ngl 40 ...
llama-bench --n-cpu-moe <best> -ngl 44 ...
```

## Phase 3: Context Sweep

Same as dense: double ctx until OOM.
MoE models often support larger ctx because expert weights dominate VRAM, not KV cache.

## Phase 4: Thread Sweep

Same as dense. Usually negligible at high GPU residency.

## MTP (Multi-Token Prediction) Variant Notes

MTP models have draft heads that consume extra VRAM. On tight VRAM budgets:
- General (non-MTP) variant often outperforms MTP
- If using MTP: reduce `--n-cpu-moe` to compensate for draft head VRAM

## Final Configuration

```bat
set NGL=<best>
set N_CPU_MOE=<best>
set CTX=<best>
set THREADS=<best>

llama-server.exe ^
  -m "model.gguf" ^
  -ngl %NGL% ^
  --n-cpu-moe %N_CPU_MOE% ^
  -c %CTX% ^
  -ctk q8_0 -ctv q8_0 ^
  -fa on ^
  -b 2048 -ub 512 ^
  -t %THREADS% ^
  -np 1 ^
  -fit off
```
