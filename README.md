# llama.cpp GPU Parameter Tuning Guide

> **When to use**: User asks to configure, benchmark, or troubleshoot llama.cpp on NVIDIA GPU. Covers both Dense and MoE models. Applies to any VRAM size.

---

## Quick Reference: Parameter Defaults by Hardware

| Hardware Spec | Parameter | Initial Value | Tune Range |
|--------------|-----------|---------------|------------|
| CPU physical cores N | `-t` | `min(N, 8)` | ±2 |
| GPU total layers L | `-ngl` | `L × 0.5` | +4~+8 steps until OOM |
| VRAM < 8GB | `-ctk -ctv` | `q4_0` | skip q8_0 |
| VRAM ≥ 12GB | `-ctk -ctv` | `q8_0` | fallback to q4_0 if OOM |
| MoE model | `--n-cpu-moe` | `total_experts × 0.6` | ±4 steps |
| Any | `-fa` | `on` | never off |
| Any | `-fit` | `off` | never on |
| Windows | `--no-mmap` | always | required |
| Any | `-b` | `2048` | 1024/2048/4096 |
| Any | `-ub` | `512` | = b/4 |

---

## Decision Tree

```
1. Check binary: --list-devices → CUDA0: ... ?
   ├─ NO  → switch to cudart-llama-bin-win-cuda build
   └─ YES → continue

2. Check model type (GGUF metadata: general.architecture):
   ├─ has n_routed_experts → MoE route
   └─ no experts → Dense route

3. Dense route:
   a. ngl sweep (ctx=4096): 32→40→48→56→60→61→62→OOM
   b. ctx sweep (ngl=best): 4096→8192→16384→24576→32768→OOM
   c. t sweep (ngl=best, ctx=best): N-2, N, N+2
   d. b sweep: 1024, 2048, 4096

4. MoE route:
   a. n_cpu_moe sweep (ngl=L×0.5, ctx=4096): 0→8→16→24→28→32
   b. ngl sweep (n_cpu_moe=best, ctx=4096): L×0.5→+6→+6...
   c. ctx sweep (ngl=best, n_cpu_moe=best)
   d. t sweep, b sweep
```

---

## Mandatory Pre-Bench Discipline

```powershell
# Run BEFORE every single llama-bench or llama-server launch
Get-Process -Name llama-cli,llama-server,llama-bench -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep 3

# Verify clean
Get-Process -Name llama-cli,llama-server,llama-bench
# MUST return nothing
```

Rules:
- Kill after every run, before every next run
- Never run 2 llama-bench in parallel
- One variable changed per run
- OOM = ceiling reached, stop going higher

---

## Verifying CUDA Backend

Three checks, all must pass:

```
1. File exists: ggml-cuda.dll in binary directory (≈150MB)
2. --list-devices shows: CUDA0: NVIDIA GeForce ... (VRAM MiB)
3. Startup log (--verbose) contains:
   - ggml_cuda_init: found N CUDA devices
   - llama_prepare_model_devices: using device CUDA0
   - CUDA0 KV buffer size = ...
```

If nvidia-smi VRAM doesn't rise after model load → offload failed → add `-fit off --no-mmap`.

---

## Parameter Reference

### `-ngl / --n-gpu-layers`
Layers to offload to GPU VRAM.
```
-ngl 0   = all CPU (slowest)
-ngl L   = all GPU (may OOM)
Per-layer VRAM ≈ model_file_size / total_layers
```

### `-c / --ctx-size`
Context length. KV cache grows linearly with ctx.
```
-c 4096   = safe start
-c 24576  = 24K
-c 32768  = 32K
```

### `-ctk / -ctv`
KV cache quantization type.
```
q8_0 = best quality, higher VRAM
q4_0 = half VRAM, minor quality loss
```
Do NOT mix (e.g. q8_0 + q4_0) — may cause prefill stall on some CUDA versions.

### `-fa / --flash-attn`
```
-fa on    ← REQUIRED (build ≥b9596 needs explicit "on")
```
Without FA, long prompts cause linear VRAM growth → OOM.

### `-t / --threads`
CPU threads for non-GPU work.
```
-t = min(physical_cores, 8)
```
Near-max ngl → threads have negligible impact.

### `--n-cpu-moe` (MoE only)
Number of routed experts to keep on CPU.
```
Higher = more experts on CPU = less VRAM but slower decode
Lower = more experts on GPU = faster but more VRAM
```
Impact on decode speed > ngl for MoE models.

### `-fit off`
Always off. Auto-fit can stall or make wrong decisions.

### `--no-mmap`
Always on for Windows. Prevents mmap-related load failures.

### `-b / --batch-size`
Prefill batch. Matters for long prompts (>512 tokens). For pp512, any value ≥512 works identically.

### `-ub / --ubatch-size`
Micro-batch for GPU compute. Typically b/4.

### `-np / --parallel`
Concurrent request slots. Each slot eats proportional VRAM. Use 1 for single user.

---

## VRAM Budget Formula

```
VRAM_used ≈ model_size × (ngl / total_layers)
          + 2 × n_full_attn_layers × n_kv_heads × head_dim × ctx × kv_bytes
          + 200 MB (compute + recurrent state)

Available VRAM = nvidia-smi total - idle usage - 500 MB margin
```

### KV Cache Quick Estimate (q8_0)

```
qwen35 (GQA 6:1, head_dim=256): ctx × 0.044 MB per 1K tokens
llama  (GQA 8:1, head_dim=128): ctx × 0.025 MB per 1K tokens
```

### OOM Mitigation Order
1. Reduce `-c`
2. Switch KV to `q4_0`
3. Reduce `-ngl`
4. Reduce `-b`

---

## llama-bench Quick Start

```bat
# Dense
llama-bench.exe -m model.gguf -ngl N -ctk q8_0 -ctv q8_0 -fa on -t T

# MoE
llama-bench.exe -m model.gguf -ngl N --n-cpu-moe M -ctk q8_0 -ctv q8_0 -fa on -t T

# Custom prompt length
llama-bench.exe ... -p 2048 -n 256
```

Output columns: `pp512` = prefill t/s, `tg128` = decode t/s.

---

## Final Startup Template

```bat
@echo off
echo %MODEL% | ngl=%NGL% | ctx=%CTX%
"%LLAMA_DIR%\llama-server.exe" ^
  -m "%MODEL_PATH%" ^
  -ngl %NGL% ^
  -c %CTX% ^
  -ctk q8_0 ^
  -ctv q8_0 ^
  -fa on ^
  -b 2048 ^
  -ub 512 ^
  -t %THREADS% ^
  -np 1 ^
  -fit off ^
  --no-mmap
pause
```

Expected output: `server is listening on http://127.0.0.1:8080`

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `--list-devices` empty | Fake CUDA binary | Get `cudart-llama-bin-win-cuda` build |
| VRAM unchanged in nvidia-smi | Offload failed | Add `-fit off --no-mmap` |
| `-fa` parse error | Build ≥b9596 needs `-fa on` | Change to `-fa on` |
| .bat double-click closes instantly | Stale process on port 8080 | Kill all llama processes first |
| llama-bench OOM | ngl too high OR stale process | Kill processes, reduce ngl |
| llama-bench fails to start | Two running in parallel | Run sequentially only |
| Slow decode on MoE | Missing `--n-cpu-moe` | Check model type, add parameter |
| Model load takes >3 min | Missing `--no-mmap` on Windows | Add `--no-mmap -fit off` |

---

## Pre-Flight Checklist

```
□ --list-devices shows CUDA0
□ ggml-cuda.dll exists
□ -fit off
□ --no-mmap added
□ -fa on (not bare -fa)
□ -t = min(physical_cores, 8)
□ --n-cpu-moe set for MoE models
□ All llama processes killed before each bench
□ No parallel bench runs
```

---

## Appendix: Real-World Data Points

### RTX 5060 Ti 16GB + Qwen3.6 27B Q4_K_M (Dense, 64 layers)
| Param | Value | Note |
|-------|-------|------|
| ngl max | 61/64 | 62 OOM |
| ctx max | 24576 | 32K OOM |
| t final | 6 | |
| pp512 | 676 t/s | |
| tg128 | 13.4 t/s | |

### RTX 3060 12GB + Qwen3.6 35B A3B Q4_K_M (MoE)
| Param | Value | Note |
|-------|-------|------|
| ngl | 40 | |
| n_cpu_moe | 28 | |
| ctx | 32768 | 64K also tested |
| t final | 8 | |
| tg128 | 26.3 t/s | |
