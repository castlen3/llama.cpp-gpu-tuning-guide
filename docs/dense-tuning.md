# Dense Model Tuning

For models without mixture-of-experts (no `n_routed_experts` in GGUF metadata).

## Prerequisites

- [CUDA backend verified](cuda-backend-check.md)
- Model GGUF metadata known: `block_count` (total layers), `head_count_kv`, `embedding_length`
- All llama processes killed before each bench run

## Phase 1: ngl Sweep (Find GPU Layer Ceiling)

### Goal
Find highest stable ngl without OOM or performance regression.

### Strategy

```
1. Start at: ngl = total_layers × 0.5 (conservative)
2. Step size: +8 layers per run (wide sweep)
3. Near ceiling (< 4 layers from OOM): ±1~2 layers (fine sweep)
4. Stop at: OOM or VRAM margin < 500 MB
```

### Example: 64-layer model

```bat
:: Sweep
llama-bench -ngl 32 ...
llama-bench -ngl 40 ...
llama-bench -ngl 48 ...
llama-bench -ngl 56 ...
llama-bench -ngl 60 ...
llama-bench -ngl 62 ...  :: OOM → ceiling = 61

:: Fine-tune near ceiling
llama-bench -ngl 61 ...
```

### Judgment

- If tok/s increases linearly with ngl → keep going
- If tok/s plateaus or drops near ceiling → stop, CPU fallback is kicking in
- If OOM → back off 1~2 layers

### Fixed Parameters During Sweep

```
-c 4096
-ctk q8_0 -ctv q8_0   (or q4_0 if VRAM < 12GB)
-fa on (or whatever --help shows)
-t = min(physical_cores, 8)
-b 2048 -ub 512
-fit off
```

## Phase 2: Context Sweep

### Goal
Find maximum stable context length.

### Strategy

```
1. Start at: -c 4096
2. Step size: double each time (8192 → 16384 → 24576 → 32768)
3. Test with llama-server --verbose, monitor VRAM
```

### Check from verbose log

```
llama_kv_cache:      CUDA0 KV buffer size =  XXX MiB
```

### OOM mitigation

If desired ctx causes OOM:
1. Reduce ctx
2. Switch KV to q4_0
3. Accept smaller ctx

## Phase 3: Thread Sweep

### Goal
Confirm threads don't matter at high ngl (they usually don't).

### Strategy

```bat
llama-bench -ngl <best> -c <best> -t <cores-2> ...
llama-bench -ngl <best> -c <best> -t <cores>   ...
llama-bench -ngl <best> -c <best> -t <cores+2> ...
```

Typically all within ±2%. Pick the lowest that matches the best score.

## Phase 4: Batch Sweep

### Goal
Find prefill batch sweet spot.

### Strategy

```bat
llama-bench -ngl <best> -c <best> -t <best> -b 1024 ...
llama-bench -ngl <best> -c <best> -t <best> -b 2048 ...
llama-bench -ngl <best> -c <best> -t <best> -b 4096 ...
```

For pp512, all values ≥512 perform identically. Only matters for long prompts.

## Final Configuration

Compile best values into a startup script. Template:

```bat
set NGL=<best>
set CTX=<best>
set THREADS=<best>

llama-server.exe ^
  -m "model.gguf" ^
  -ngl %NGL% ^
  -c %CTX% ^
  -ctk q8_0 -ctv q8_0 ^
  -fa on ^
  -b 2048 -ub 512 ^
  -t %THREADS% ^
  -np 1 ^
  -fit off
```
