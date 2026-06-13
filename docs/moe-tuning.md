# MoE Model Tuning

For models with mixture-of-experts (has `n_routed_experts` in GGUF metadata).

## Key Difference from Dense

MoE models have two critical parameters:

1. **`-ngl`**: Controls how many transformer layers are on GPU (same as Dense)
2. **`--n-cpu-moe`**: Controls how many routed experts are kept on CPU

For MoE, raw ngl is not the whole story. Expert placement can dominate both VRAM and speed.

Important: CPU expert offload is a VRAM-saving tool, not a performance tool. If
VRAM allows it, test `-ncmoe 0` first. On older CPUs or memory-bound platforms,
moving experts to CPU can dominate decode latency.

## Prerequisites

- Backend verified
- GGUF metadata known: `block_count`, `expert_count`, `expert_used_count`
- All llama processes killed before each bench run

## Phase 1: Expert Placement Sweep

### Goal

Find the optimal balance between GPU and CPU expert placement.

### Strategy

```text
1. Try full GPU residency if VRAM allows:
   -ngl 99
   -ncmoe 0
2. Verify with llama-server at target context and KV dtype.
3. If OOM, reduce context or KV dtype before pushing experts to CPU.
4. Only then sweep n_cpu_moe: 0, 5, 10, 20, 28, 40, ...
5. For each n_cpu_moe, note VRAM usage, prefill speed, and decode speed.
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
- If `n_cpu_moe=0` fits, it is often the best decode setting
- Do not copy a low-VRAM GPU's `n_cpu_moe` value to a high-VRAM GPU

## Phase 2: ngl Sweep

After finding optimal n_cpu_moe:

```sh
llama-bench -ngl <n_cpu_moe=best> -ngl 30 ...
llama-bench -ngl <n_cpu_moe=best> -ngl 40 ...
llama-bench -ngl <n_cpu_moe=best> -ngl 50 ...
...
```

Same strategy as Dense: start at 50%, increase by +4~8, fine-tune near ceiling.

If VRAM is abundant, reverse the sweep: start at `-ngl 99`, then step down only
if OOM or unstable. Some models have a large decode cliff when one final layer
remains off GPU.

## Phase 3: Context Sweep

Same as Dense. See [docs/dense-tuning.md](docs/dense-tuning.md) Phase 2.

## Phase 4: Thread & Batch Sweep

Same as Dense. See [docs/dense-tuning.md](docs/dense-tuning.md) Phases 3-4.

## Common MoE Pitfalls

### Missing `--n-cpu-moe`

If MoE decode is unexpectedly slow, check whether `--n-cpu-moe` is available in your build. Without it, all experts may be forced onto GPU (OOM) or CPU (slow).

### Copying Low-VRAM Settings to High-VRAM GPUs

Do not copy `n_cpu_moe=20/28/40` from a 12GB GPU guide to a 22GB GPU. The
12GB setting may be a VRAM compromise. On the larger GPU, `n_cpu_moe=0` may fit
and be much faster.

### Server vs Bench Confusion

Use `llama-bench` only for quick scouting. A MoE model at 64K context must be
verified with `llama-server`, because server context, KV cache, prompt cache,
and MTP/speculative settings can change the result.

### Expert Offload Combinations

Some builds support fine-grained expert offloading. Check `--help` for:

- `--n-cpu-moe` / `-ncmoe`
- `--n-gpu-layers-moe` (rare)
- Expert sharding options

### VRAM vs Speed Tradeoff

MoE models have a steeper VRAM/speed tradeoff than Dense models. A small change in expert placement can have a large impact on decode speed.
