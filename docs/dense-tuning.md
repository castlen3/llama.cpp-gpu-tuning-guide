# Dense Model Tuning

For models without mixture-of-experts (no `n_routed_experts` in GGUF metadata).

## Prerequisites

- Backend verified (see [cuda.md](cuda.md), [vulkan.md](vulkan.md), [rocm.md](rocm.md))
- Model GGUF metadata known: `block_count` (total layers), `head_count_kv`, `embedding_length`
- All llama processes killed before each bench run

## Phase 1: ngl Sweep (Find GPU Layer Ceiling)

### Goal

Find highest stable ngl without OOM or performance regression.

> **Note on VRAM-abundant GPUs:** On GPUs with ample VRAM (e.g. 22GB+ for a 7GB model), starting at `total_layers * 0.5` is overly conservative. You can jump directly to `ngl=99` and verify. The "don't maximize ngl" rule applies when VRAM is the bottleneck — not when it's clearly sufficient.

> **Note on SWA (Sliding Window Attention) models:** Architectures like Gemma 4 use sliding window attention on some layers, dramatically reducing KV cache size. VRAM estimates from the formula in `kv-cache-sizing.md` will overstate actual usage. Always verify with the startup log's `KV buffer size` line.

> **Note on high-VRAM dense models:** If the model nearly fits, test `ngl=99`
> at the target context before spending time on mid-range sweeps. A single layer
> left on CPU can create a large decode cliff on some builds.

### Strategy

```text
1. Read total layer count from startup log or GGUF metadata.
2. Start at: ngl = total_layers * 0.5 (conservative).
3. Step size: +4~+8 layers per run (wide sweep).
4. Near ceiling (<4 layers from OOM): +-1~2 layers (fine sweep).
5. Stop at: OOM or VRAM margin < 500 MB.
6. If only a few layers remain on CPU, check whether CPU/RAM becomes the critical path.
```

### Example Sweep (64-layer model)

```sh
llama-bench -ngl 32 ...
llama-bench -ngl 40 ...
llama-bench -ngl 48 ...
llama-bench -ngl 56 ...
llama-bench -ngl 60 ...
llama-bench -ngl 62 ...  # OOM -> ceiling = 61

# Fine-tune near ceiling
llama-bench -ngl 61 ...
```

### Judgment

- If tok/s increases linearly with ngl -> keep going
- If tok/s plateaus or drops near ceiling -> stop, CPU fallback is kicking in
- If OOM -> back off 1~2 layers

### Fixed Parameters During Sweep

llama-bench does NOT support `-c` (context size). Use `-p` (prompt tokens) and `-n` (gen tokens) instead. For context sweep (Phase 2), use llama-server.

```sh
# llama-bench — does NOT have -c flag. Use -p + -n to set test length:
-p 512 -n 128           # typical benchmark
-p 4096 -n 128          # longer prefill test

-ctk q8_0 -ctv q8_0     # or q4_0 if VRAM < 12GB
-fa on                   # or whatever --help shows for your build
-t = physical_cores / 2
-b 2048 -ub 512
-fit off                 # for reproducible benchmarking
```

For `llama-server` benchmarks, also add `--cache-ram 0` unless you are
specifically testing prompt cache behavior.

`llama-bench` is only a scout. For final dense-model decisions, use
`llama-server` at the target context and record both `prompt_per_second` and
`predicted_per_second`.

## Phase 2: Context Sweep

### Goal

Find maximum stable context length.

### Strategy

```text
1. Start at: -c 4096
2. Step size: double each time (8192 -> 16384 -> 24576 -> 32768)
3. Test with llama-server --verbose, monitor VRAM
```

### Check from Verbose Log

```
llama_kv_cache:      GPU KV buffer size =  XXX MiB
```

### OOM Mitigation

If desired ctx causes OOM:

1. Reduce ctx
2. Switch KV to q4_0
3. Accept smaller ctx

## Phase 3: Thread Sweep

### Goal

Find the thread sweet spot for your platform.

### Strategy

```sh
llama-bench -ngl <best> -t <physical_cores/2> ...
llama-bench -ngl <best> -t <physical_cores>   ...
llama-bench -ngl <best> -t <physical_cores+2> ...
llama-bench -ngl <best> -t <logical_threads>   ...
```

Typically all within +-2%. On memory-bound platforms (DDR3), fewer threads may be faster. On modern DDR4/DDR5, physical cores is usually the sweet spot.

## Phase 4: Batch Sweep

### Goal

Find prefill batch sweet spot.

### Strategy

```sh
llama-bench -ngl <best> -t <best> -b 1024 ...
llama-bench -ngl <best> -t <best> -b 2048 ...
llama-bench -ngl <best> -t <best> -b 4096 ...
```

For pp512, all values >=512 perform identically. Only matters for long prompts.

For long-prompt server tests, use a 4K+ prompt. Do not judge prefill from
`pp512` alone.

Larger batch is not automatically faster. Test:

```text
-b 2048 -ub 512
-b 4096 -ub 512
-b 4096 -ub 1024
-b 8192 -ub 1024
```

Keep the setting that improves long-prompt prefill without increasing VRAM too
much. If all larger values are slower, keep `-b 2048 -ub 512`.

## Phase 5: MTP / Speculative Decoding

For MTP-capable dense models, first establish a no-MTP baseline. Then follow
[speculative-mtp.md](speculative-mtp.md).

Remember:

- MTP improves decode, not prefill
- Measure short-prompt decode and long-prompt prefill separately
- Higher draft count is not automatically faster

## Final Configuration

Compile best values into a startup script. See `start-server.bat` for a template.
