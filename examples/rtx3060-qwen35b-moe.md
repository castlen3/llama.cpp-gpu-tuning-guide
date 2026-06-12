# Case Study: RTX 3060 12GB + Qwen3.6 35B A3B Q4_K_M

## Hardware

- GPU: NVIDIA GeForce RTX 3060 12GB
- CPU: (fill in)
- RAM: (fill in)

## Backend

- CUDA (ggml-cuda.dll)
- llama.cpp build: (fill in)

## Model

- Qwen3.6-35B-A3B-Q4_K_M.gguf
- MoE architecture (35B total, 3B active per token)

## Goal

Optimize MoE expert placement for maximum decode speed within 12GB VRAM.

## Failed Configs

| Config | Result | Reason |
|--------|--------|--------|
| ngl=max, n_cpu_moe=0 | OOM | All experts on GPU exceeds 12GB |
| ngl=40, n_cpu_moe=0 | Slow decode | Too many experts on CPU via layer offload |

## Working Configs

| Config | n_cpu_moe | tg128 (t/s) | Notes |
|--------|-----------|-------------|-------|
| ngl=40, n_cpu_moe=20 | 20 | ~22 | Conservative |
| ngl=40, n_cpu_moe=28 | 28 | 26.3 | Best balance |

## Best Config

```bat
llama-server.exe ^
  --host 0.0.0.0 ^
  -m "Qwen3.6-35B-A3B-Q4_K_M.gguf" ^
  -ngl 40 ^
  -c 32768 ^
  --n-cpu-moe 28 ^
  -ctk q8_0 ^
  -ctv q8_0 ^
  -fa on ^
  -b 2048 ^
  -ub 512 ^
  -t 8 ^
  -np 1
```

## Observed Bottleneck

- Expert placement dominates decode speed for MoE models
- n_cpu_moe has a larger impact on tg128 than ngl
- 32K context is feasible at 12GB with q8_0 KV

## Lessons Learned

1. MoE tuning requires a separate sweep for expert placement
2. `--n-cpu-moe` is the most impactful parameter for MoE decode speed
3. Don't assume Dense tuning rules apply to MoE models
4. Record active experts and CPU expert placement alongside VRAM and speed
