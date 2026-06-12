# MoE Model Tuning

For models with mixture-of-experts (has `n_routed_experts` in GGUF metadata).

## Key Difference from Dense

MoE models have two critical parameters:

1. **`-ngl`**: Controls how many transformer layers are on GPU (same as Dense)
2. **`--n-cpu-moe`**: Controls how many routed experts are kept on CPU

For MoE, raw ngl is not the whole story. Expert placement can dominate both VRAM and speed.

## Prerequisites

- Backend verified
- GGUF metadata known: `block_count`, `expert_count`, `expert_used_count`
- All llama processes killed before each bench run

## Phase 1: Expert Placement Sweep

### Goal

Find the optimal balance between GPU and CPU expert placement.

### Strategy

```text
1. Try full GPU offload if VRAM allows (ngl=max, n_cpu_moe=0).
2. If OOM, try: ngl=50%, n_cpu_moe = total_experts * 0.6.
3. Sweep n_cpu_moe: total_experts * 0.4, 0.5, 0.6, 0.7, 0.8.
4. For each n_cpu_moe, note VRAM usage and decode speed.
```

### What to Record

```text
- Active experts per token (from model config)
- CPU expert placement (n_cpu_moe value)
- VRAM usage
- Decode speed (tg128)
```

### Judgment

- Higher n_cpu_moe = less VRAM, slower decode
- Impact on decode speed often exceeds ngl for MoE models
- Find the sweet spot where VRAM fits and decode is acceptable

## Phase 2: ngl Sweep

After finding optimal n_cpu_moe:

```sh
llama-bench -ngl <n_cpu_moe=best> -ngl 30 ...
llama-bench -ngl <n_cpu_moe=best> -ngl 40 ...
llama-bench -ngl <n_cpu_moe=best> -ngl 50 ...
...
```

Same strategy as Dense: start at 50%, increase by +4~8, fine-tune near ceiling.

## Phase 3: Context Sweep

Same as Dense. See [docs/dense-tuning.md](docs/dense-tuning.md) Phase 2.

## Phase 4: Thread & Batch Sweep

Same as Dense. See [docs/dense-tuning.md](docs/dense-tuning.md) Phases 3-4.

## Common MoE Pitfalls

### Missing `--n-cpu-moe`

If MoE decode is unexpectedly slow, check whether `--n-cpu-moe` is available in your build. Without it, all experts may be forced onto GPU (OOM) or CPU (slow).

### Expert Offload Combinations

Some builds support fine-grained expert offloading. Check `--help` for:

- `--n-cpu-moe` / `-ncmoe`
- `--n-gpu-layers-moe` (rare)
- Expert sharding options

### VRAM vs Speed Tradeoff

MoE models have a steeper VRAM/speed tradeoff than Dense models. A small change in expert placement can have a large impact on decode speed.
