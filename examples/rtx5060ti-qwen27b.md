# Case Study: RTX 5060 Ti 16GB + Qwen3.6 27B Q4_K_M

## Hardware

- GPU: NVIDIA GeForce RTX 5060 Ti 16GB
- CPU: Intel Xeon E5-2666 v3 (10c/20t, DDR3)
- RAM: DDR3

## Backend

- CUDA (ggml-cuda.dll)
- llama.cpp build 9596 (18ef86ece)

## Model

- Qwen3.6-27B-Q4_K_M.gguf
- 64 layers, Dense architecture
- File size: ~15.4 GiB

## Goal

Maximize inference speed within 16GB VRAM at 24K context with q8_0 KV cache.

## Failed Configs

| Config | Result | Reason |
|--------|--------|--------|
| ngl=62, ctx=24576, q8_0 KV | OOM | Weights + KV + compute buffer exceed 16GB |
| ngl=61, ctx=32768, q8_0 KV | OOM | KV cache too large at 32K |
| -fa (bare flag) | Parse error | Build b9596 requires `-fa on` |

## Working Configs

| Config | pp512 (t/s) | tg128 (t/s) | Notes |
|--------|-------------|-------------|-------|
| ngl=58, ctx=24576, q8_0 | 563 | 10.52 | Conservative |
| ngl=60, ctx=24576, q8_0 | 633 | 12.53 | Good |
| ngl=61, ctx=24576, q8_0 | 676 | 13.88 | Best |

## Best Config

```bat
llama-server.exe ^
  --host 0.0.0.0 ^
  -m "Qwen3.6-27B-Q4_K_M.gguf" ^
  -ngl 61 ^
  -c 24576 ^
  -ctk q8_0 ^
  -ctv q8_0 ^
  -fa on ^
  -b 2048 ^
  -ub 512 ^
  -t 8 ^
  -tb 16 ^
  -np 1
```

## Observed Bottleneck

- Decode speed (13.4 t/s) is the bottleneck, not prefill (676 t/s)
- ngl=61 is the ceiling for 16GB VRAM at 24K context
- Thread count (6-8) has negligible impact at ngl=61 (GPU-dominated)
- Batch size (1024-4096) has negligible impact on pp512

## Lessons Learned

1. `-fa` syntax changed between builds — always check `--help`
2. `-fit off` helps reproducible benchmarks; test `--no-mmap` only as a separate diagnostic after a clean mmap baseline
3. `^` continuation in .bat files must have zero trailing spaces
4. Kill all llama processes before each run to avoid VRAM conflicts
5. ngl sweep from 50% with +8 steps is efficient for finding the ceiling
6. Thread tuning is low-impact at high ngl — don't waste time on it
7. DDR3 memory bandwidth limits CPU-side work — fewer threads can be faster
