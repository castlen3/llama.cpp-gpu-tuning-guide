# llama.cpp GPU Parameter Tuning Guide

> Dense & MoE model optimization SOP for NVIDIA GPUs.
>
> **TL;DR**: Maximize stable GPU residency, not just ngl:
> highest stable ngl + safe KV cache + normal graph splits + no CPU fallback.
>
> 30 seconds to start, 5 minutes to diagnose, 30 minutes to a meaningful benchmark.

---

## Quick Reference

| What | Parameter | Conservative Start | Notes |
|------|-----------|-------------------|-------|
| CPU threads | `-t` | `min(physical_cores, 8)` | Impact shrinks as ngl rises |
| Batch threads | `-tb` | same as `-t` | Check `--help`, not all builds have it |
| GPU layers | `-ngl` | `total_layers × 0.5` | +4~+8 steps toward ceiling |
| MoE CPU experts | `--n-cpu-moe` | `total_experts × 0.6` | Often bigger impact than ngl on decode |
| Context | `-c` | `4096` | Double each step until OOM |
| KV cache type | `-ctk -ctv` | `q8_0` if VRAM ≥ 12GB, else `q4_0` | Never mix q8_0 + q4_0 |
| Flash attention | `-fa` | check `--help` first | Some builds: `-fa`, others: `-fa on` or `--flash-attn on` |
| Prefill batch | `-b` | `2048` | Matters only for long prompts |
| Micro batch | `-ub` | `512` | Typically `b / 4` |
| Auto-fit | `-fit` | `off` for benchmarking | Can use `on` for casual daily use |
| Memory map | `--mmap` | `on` by default | Use `--no-mmap` if model load fails on Windows |
| Parallel slots | `-np` | `1` | Each slot eats proportional VRAM |

---

## Step 0: Know Your Binary (Do This First)

Every build is different. Before tuning any parameter:

```bat
llama-server.exe --version
llama-server.exe --help > llama-help.txt 2>&1
llama-server.exe --list-devices
```

Verify from the output:

| Check | What to look for |
|-------|-----------------|
| CUDA backend exists | `ggml-cuda.dll` in binary directory (~150 MB) |
| GPU detected | `--list-devices` shows `CUDA0: NVIDIA ...` |
| ngl flag | `-ngl`, `--n-gpu-layers`, or `--gpu-layers` |
| KV cache flags | `-ctk`/`--cache-type-k`, `-ctv`/`--cache-type-v` |
| Flash attention flag | `-fa`, `-fa on`, or `--flash-attn` — use **exactly** what `--help` shows |
| Thread flags | `-t`/`--threads`, `-tb`/`--threads-batch` (not all builds have `-tb`) |
| MoE flag | `--n-cpu-moe` or `-ncmoe` |

**If `--list-devices` does NOT show CUDA0, stop here.** Your binary does not have CUDA support. Get a `cudart-llama-bin-win-cuda` build.

---

## Decision Tree

```
1. --list-devices → CUDA0: ... ?
   ├─ NO  → Get CUDA build, restart
   └─ YES → continue

2. Check GGUF metadata: general.architecture
   ├─ has n_routed_experts → MoE route
   └─ no experts → Dense route

3. Dense route: docs/dense-tuning.md
   a. ngl sweep (ctx=4096)
   b. ctx sweep (ngl=best)
   c. t sweep
   d. b sweep

4. MoE route: docs/moe-tuning.md
   a. --n-cpu-moe sweep (ngl=50%, ctx=4096)
   b. ngl sweep (n_cpu_moe=best)
   c. ctx sweep
   d. t, b sweep
```

---

## Pre-Bench Discipline

```powershell
# Run BEFORE every single bench or server launch
Get-Process -Name llama-cli,llama-server,llama-bench -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep 3

# Verify clean
Get-Process -Name llama-cli,llama-server,llama-bench
# MUST return nothing
```

- Kill after every run, before every next run
- Never parallelize llama-bench runs
- One variable changed per test
- OOM = ceiling reached, stop

---

## Parameter Reference

### `-ngl / --n-gpu-layers`

Offloads transformer layers to GPU VRAM.

```
-ngl 0          = all CPU (slowest)
-ngl max        = all GPU (may OOM)
per-layer VRAM ≈ model_file_size / total_layers
```

**Tuning strategy (Dense):**
1. Start at `total_layers × 0.5`
2. Increase by +6~+8 layers per step
3. When approaching ceiling (<4 layers from OOM), switch to ±1~2 step
4. Pick the highest ngl that is stable and does not regress tok/s

See: [docs/dense-tuning.md](docs/dense-tuning.md)

### `-c / --ctx-size`

Context window size. KV cache grows proportionally.

```
-c 4096   = safe baseline
-c 24576  = 24K
-c 32768  = 32K
```

### `-ctk / -ctv`

KV cache quantization.

| Setting | VRAM | Precision | When |
|---------|------|-----------|------|
| `q8_0 q8_0` | higher | more conservative | VRAM has headroom |
| `q4_0 q4_0` | half | good enough for most use | VRAM tight |

**⚠️ Do NOT mix types** (e.g. `-ctk q8_0 -ctv q4_0`). Some CUDA backends exhibit prefill stalls or hangs with asymmetric KV cache dtypes.

### `-fa / --flash-attn`

Flash attention reduces VRAM growth for long prompts.

**Check your build's `--help` for exact syntax.** Common variants:
- `-fa` (older builds)
- `-fa on` (builds ≥ b9596)
- `--flash-attn on`

### `-t / --threads`

CPU threads for non-GPU compute.

```
-t = min(physical_cores, 8)
```

Near-max ngl → threads have minimal impact (±2 = negligible).

### `-tb / --threads-batch`

Batch processing threads. Not all builds expose this flag. If available, set equal to `-t`.

### `--n-cpu-moe` (MoE only)

Number of routed experts kept on CPU. Higher = less VRAM, slower decode.

Impact on decode speed often exceeds `-ngl` for MoE models.

See: [docs/moe-tuning.md](docs/moe-tuning.md)

### `-fit`

Auto-fit mode. Off for reproducible benchmarking; can use on for casual daily driving.

### `--mmap / --no-mmap`

Memory-mapped model loading. Leave on by default. If model load hangs or fails on Windows, try `--no-mmap` as a workaround.

### `-b / --batch-size`

Prefill batch. Only matters when prompt > batch size. For pp512, any value ≥ 512 behaves identically.

### `-ub / --ubatch-size`

Micro-batch. Typically `b / 4`.

### `-np / --parallel`

Concurrent request slots. Each slot adds proportional VRAM cost.

---

## Verifying CUDA Is Actually Working

Three checks:

```
1. ggml-cuda.dll exists (≈150 MB)
2. --list-devices shows CUDA0
3. Startup log (--verbose) contains:
   ggml_cuda_init: found N CUDA devices
   llama_prepare_model_devices: using device CUDA0
   CUDA0 KV buffer size = ...
```

**If nvidia-smi shows VRAM not rising after model load**, CUDA offload is not working. Try `-fit off --no-mmap`, then check `--verbose` log for `offloaded 0/N layers`.

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `--list-devices` empty | No CUDA binary | Get `cudart-llama-bin-win-cuda` build |
| VRAM unchanged in nvidia-smi | 0 layers offloaded | Add `-fit off --no-mmap`, check verbose log |
| `-fa` parse error | Build syntax changed | Run `--help`, use exact flag shown |
| .bat flashes and closes | Port 8080 in use | Kill all llama processes first |
| Bench OOM | ngl too high / stale process | Kill processes, reduce ngl |
| Bench fails to start | Parallel runs | Sequential only |
| **VRAM used but GPU utilization ~0%** | See below | See below |
| **Prompt eval never finishes** | Mixed KV dtype bug | Use symmetric `-ctk -ctv` |
| **`/health` OK but `/slots` timeout** | CUDA graph hang | Check verbose log for graph errors |
| Slow decode on MoE | Missing `--n-cpu-moe` | Check model type, add parameter |

### GPU "Not Working" Despite VRAM Usage

If VRAM is occupied but GPU compute utilization stays near 0%:

1. **Monitor correctly**: Use `nvidia-smi -l 1`, not Windows Task Manager (which shows the wrong engine for CUDA workloads).
2. **Check offload count**: In `--verbose` log, confirm `offloaded N/M layers` where N > 0.
3. **Mixed KV dtype**: If using asymmetric `-ctk -ctv`, switch to symmetric (both q8_0 or both q4_0).
4. **CUDA graphs**: Some builds hit graph capture bugs. Try `--no-graph` if available.
5. **CUDA backend loaded?**: Confirm `load_backend: loaded CUDA backend from ... ggml-cuda.dll` in log.

---

## VRAM Budget

```
VRAM_used ≈ model_size × (ngl / total_layers)
          + KV_cache (see docs/kv-cache-sizing.md)
          + ~200 MB (compute + recurrent state buffers)

Available = nvidia-smi total − idle usage − 500 MB safety margin
```

When OOM, reduce in this order:
1. `-c` (most effective)
2. KV to `q4_0`
3. `-ngl`
4. `-b`

---

## Benchmark SOP

### llama-bench Quick Start

```bat
# Dense
llama-bench.exe -m model.gguf -ngl N -ctk q8_0 -ctv q8_0 -fa on -t T

# MoE
llama-bench.exe -m model.gguf -ngl N --n-cpu-moe M -ctk q8_0 -ctv q8_0 -fa on -t T

# Custom prompt length
llama-bench.exe ... -p 2048 -n 256
```

Output: `pp512` = prefill t/s, `tg128` = decode t/s.

### Sweep Strategy

1. One variable per run
2. Wide steps first (ngl +8, n_cpu_moe +8, ctx ×2)
3. Narrow near ceiling (±1~2 layers)
4. Kill processes between every run

---

## Benchmark Result Template

```text
| Field | Value |
|-------|-------|
| GPU | NVIDIA GeForce RTX ____ |
| Driver | ____ |
| llama.cpp commit/build | ____ |
| Model | ____ |
| Quant | ____ |
| ctx | ____ |
| ngl | ____ |
| --n-cpu-moe | ____ (MoE only) |
| ctk / ctv | ____ / ____ |
| fa | ____ |
| b / ub | ____ / ____ |
| t / tb | ____ / ____ |
| pp512 | ____ t/s |
| tg128 | ____ t/s |
| VRAM peak | ____ MiB |
| graph splits | ____ |
| notes | ____ |
```

---

## Repo Structure

```
README.md                     ← You are here
start-server.bat              ← Optimized launch template
docs/
  cuda-backend-check.md       ← Verifying CUDA is actually loaded
  dense-tuning.md             ← Dense model tuning workflow
  moe-tuning.md               ← MoE model tuning workflow
  kv-cache-sizing.md          ← KV cache size estimation
  troubleshooting.md          ← Detailed symptom→fix map
examples/
  rtx5060ti-qwen27b-24k-q8.bat
  rtx5060ti-qwen27b-32k-q4.bat
  rtx3060-qwen35b-moe.bat
```

---

## Appendix: Real-World Data

### RTX 5060 Ti 16GB + Qwen3.6 27B Q4_K_M (Dense, 64 layers)

| Param | Value |
|-------|-------|
| ngl max stable | 61 (62 OOM) |
| ctx max stable | 24576 (32K OOM) |
| t final | 6 |
| pp512 | 676 t/s |
| tg128 | 13.4 t/s |

### RTX 3060 12GB + Qwen3.6 35B A3B Q4_K_M (MoE)

| Param | Value |
|-------|-------|
| ngl | 40 |
| n_cpu_moe | 28 |
| ctx | 32768 |
| t final | 8 |
| tg128 | 26.3 t/s |
