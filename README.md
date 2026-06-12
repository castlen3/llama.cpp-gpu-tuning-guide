# llama.cpp GPU Tuning Playbook

A general-purpose tuning playbook for llama.cpp GPU inference.

This guide is designed for agents and humans who need to optimize unknown hardware, model, backend, and context-length combinations. It is not tied to a specific GPU. Hardware-specific results are kept as case studies.

---

## TL;DR

Do not maximize `-ngl` blindly. Maximize stable GPU residency:

- highest stable `-ngl`
- safe KV cache type
- normal graph splits
- no CPU fallback
- no backend-specific prefill stalls

30 seconds to start, 5 minutes to diagnose, 30 minutes to a meaningful benchmark.

---

## Agent Decision Framework

When you receive a llama.cpp tuning task, follow this order:

### 1. Identify Backend

```text
What GPU / accelerator is available?
  - NVIDIA → CUDA (docs/cuda.md)
  - AMD desktop → Vulkan (docs/vulkan.md) or ROCm (docs/rocm.md)
  - Apple Silicon → Metal (docs/metal.md)
  - No GPU → CPU-only fallback
```

Run `--list-devices` to confirm. If the expected backend is absent, stop and fix the binary first.

### 2. Identify Model Type

```text
Check GGUF metadata: general.architecture
  - has n_routed_experts → MoE route
  - no experts → Dense route
```

MoE and Dense models have fundamentally different tuning strategies.

### 3. Identify Pressure Source

VRAM pressure can come from multiple sources. Identify which one is the bottleneck:

| Source | How to detect |
|--------|---------------|
| Model weights | `offloaded N/M layers` in verbose log |
| KV cache | `CUDA0 KV buffer size = XXX MiB` in verbose log |
| Compute buffer | Startup log, typically 100-300 MB |
| CPU offloaded layers | High CPU usage during inference |
| Backend fallback | GPU VRAM occupied but GPU utilization near 0% |

### 4. Choose Tuning Route

```text
Dense model    → docs/dense-tuning.md
MoE model      → docs/moe-tuning.md
Long context   → docs/kv-cache-sizing.md
Backend issue  → docs/cuda.md | docs/vulkan.md | docs/metal.md | docs/rocm.md
```

### 5. Run a Minimal Reproducible Benchmark

Start with conservative parameters, establish a baseline, then sweep one variable at a time.

---

## Quick Reference

| What | Parameter | Conservative Start | Notes |
|------|-----------|-------------------|-------|
| CPU threads | `-t` | `physical_cores / 2` | Sweep around physical core count |
| Batch threads | `-tb` | same as `-t` | Check `--help`, not all builds have it |
| GPU layers | `-ngl` | `total_layers * 0.5` | +4~+8 steps toward ceiling |
| MoE CPU experts | `--n-cpu-moe` | `total_experts * 0.6` | Often bigger impact than ngl on decode |
| Context | `-c` | `4096` | Double each step until OOM |
| KV cache type | `-ctk -ctv` | `q8_0/q8_0` if VRAM allows, else `q4_0/q4_0` | Avoid mixing types on CUDA |
| Flash attention | `-fa` | check `--help` first | Syntax varies by build |
| Prefill batch | `-b` | `2048` | Matters only for long prompts |
| Micro batch | `-ub` | `512` | Typically `b / 4` |
| Auto-fit | `-fit` | `off` for benchmarking | Can use `on` for casual daily use |
| Memory map | `--mmap` | `on` by default | `--no-mmap` if model load fails |
| Parallel slots | `-np` | `1` | Each slot eats proportional VRAM |

---

## Step 0: Know Your Binary

Every build is different. Before tuning any parameter:

```sh
llama-server --version
llama-server --help > llama-help.txt 2>&1
llama-server --list-devices
```

Verify from the output:

| Check | What to look for |
|-------|-----------------|
| Backend exists | Backend-specific DLL/dylib/so present |
| GPU detected | `--list-devices` shows your device |
| ngl flag | `-ngl`, `--n-gpu-layers`, or `--gpu-layers` |
| KV cache flags | `-ctk`/`--cache-type-k`, `-ctv`/`--cache-type-v` |
| Flash attention flag | Use exactly what `--help` shows |
| Thread flags | `-t`/`--threads`, `-tb`/`--threads-batch` |
| MoE flag | `--n-cpu-moe` or `-ncmoe` |

---

## Decision Tree

```
1. --list-devices → shows GPU?
   +- NO  → Fix binary / driver, restart
   +- YES → continue

2. Check GGUF metadata: general.architecture
   +- has n_routed_experts → MoE route
   +- no experts → Dense route

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
per-layer VRAM ~ model_file_size / total_layers
```

**Tuning strategy (Dense):**

1. Start at `total_layers * 0.5` (conservative)
2. Increase by +4~+8 layers per step
3. When approaching ceiling (<4 layers from OOM), switch to +-1~2 step
4. Pick the highest ngl that is stable and does not regress tok/s
5. If only a few layers remain on CPU, check whether CPU/RAM bandwidth has become the critical path before assuming the GPU is the bottleneck

See: [docs/dense-tuning.md](docs/dense-tuning.md)

### `-c / --ctx-size`

Context window size. KV cache grows proportionally.

```
-c 4096   = safe baseline
-c 8192   = 8K
-c 32768  = 32K
```

### `-ctk / -ctv`

KV cache quantization. See [docs/kv-cache-sizing.md](docs/kv-cache-sizing.md).

| Setting | VRAM | When |
|---------|------|------|
| `q8_0 q8_0` | higher | VRAM has headroom |
| `q4_0 q4_0` | half | VRAM tight |

On CUDA, avoid asymmetric KV cache dtypes (e.g. `-ctk q8_0 -ctv q4_0`) unless verified safe on your build, because it may cause prefill stall/hang. Other backends may not have this issue.

Use the startup log as the source of truth for KV buffer size:

```
llama_kv_cache:      GPU KV buffer size =  XXX MiB
```

Run with `--verbose` and your target `-c` to get exact numbers.

### `-fa / --flash-attn`

Reduces VRAM growth for long prompts. Recommended for long-context GPU runs where supported.

Check your build's `--help` for exact syntax. Common variants:

- `-fa` (older builds)
- `-fa on` (newer builds)
- `--flash-attn on`

### `-t / --threads`

CPU threads for non-GPU compute.

**General strategy:**

Start near physical cores rather than logical threads. On older DDR3 or memory-bound systems, fewer threads may outperform full logical-thread usage.

```
Start: physical_cores / 2
Then:  physical_cores
Then:  physical_cores + 2
Then:  logical_threads
```

Use `-t` primarily to tune decode/generation throughput.
Use `-tb` primarily to tune prompt processing / prefill throughput when the build exposes this flag.

For older DDR3 / memory-bound platforms, fewer threads may be faster. For modern DDR4/DDR5, physical cores is usually the sweet spot. Near-max ngl, threads have minimal impact on GPU-dominated inference.

### `-tb / --threads-batch`

Batch processing threads. Not all builds expose this flag. If available, set equal to `-t` or higher.

### `--n-cpu-moe` (MoE only)

Number of routed experts kept on CPU. Higher = less VRAM, slower decode.

Impact on decode speed often exceeds `-ngl` for MoE models. See [docs/moe-tuning.md](docs/moe-tuning.md).

### `-fit`

Auto-fit mode. Off for reproducible benchmarking and manual tuning; can use on for casual daily driving.

### `--mmap / --no-mmap`

Memory-mapped model loading. Leave on by default. `--no-mmap` is a useful workaround when loading is slow, fails, or mmap behavior is suspicious on your platform.

### `-b / --batch-size`

Prefill batch. Only matters when prompt > batch size. For pp512, any value >=512 behaves identically.

### `-ub / --ubatch-size`

Micro-batch. Typically `b / 4`.

### `-np / --parallel`

Concurrent request slots. Each slot adds proportional VRAM cost.

---

## Verifying GPU Backend Is Actually Working

Three checks:

```
1. Backend library exists (e.g. ggml-cuda.dll, ggml-vulkan.dll)
2. --list-devices shows your GPU
3. Startup log (--verbose) contains:
   [backend]_init: found N devices
   llama_prepare_model_devices: using device GPU0
   GPU0 KV buffer size = ...
```

If GPU VRAM is not rising after model load, offload is not working. Try `-fit off --no-mmap`, then check `--verbose` log for `offloaded 0/N layers`.

Backend-specific verification: [docs/cuda.md](docs/cuda.md) | [docs/vulkan.md](docs/vulkan.md) | [docs/metal.md](docs/metal.md) | [docs/rocm.md](docs/rocm.md)

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `--list-devices` empty | No GPU backend in binary | Get a build with your backend |
| VRAM unchanged after load | 0 layers offloaded | Add `-fit off --no-mmap`, check verbose log |
| `-fa` parse error | Build syntax changed | Run `--help`, use exact flag shown |
| .bat flashes and closes | Port in use | Kill all llama processes first |
| .bat says `'ngl' is not recognized` | `^` has trailing space | Use single-line command |
| Bench OOM | ngl too high / stale process | Kill processes, reduce ngl |
| Bench fails to start | Parallel runs | Sequential only |
| VRAM used but GPU utilization ~0% | Backend fallback | See backend-specific docs |
| Prompt eval never finishes | Mixed KV dtype bug | Use symmetric `-ctk -ctv` |
| `/health` OK but `/slots` timeout | Graph hang | Check verbose log for graph errors |
| Slow decode on MoE | Missing `--n-cpu-moe` | Check model type, add parameter |

---

## VRAM Budget

```
VRAM_used ~ model_size * (ngl / total_layers)
          + KV_cache (see docs/kv-cache-sizing.md)
          + ~200 MB (compute + recurrent state buffers)

Available = GPU total VRAM - idle usage - 500 MB safety margin
```

When OOM, reduce in this order:

1. `-c` (most effective)
2. KV to `q4_0`
3. `-ngl`
4. `-b`

---

## Benchmark SOP

### llama-bench Quick Start

```sh
# Dense
llama-bench -m model.gguf -ngl N -ctk q8_0 -ctv q8_0 -fa on -t T

# MoE
llama-bench -m model.gguf -ngl N --n-cpu-moe M -ctk q8_0 -ctv q8_0 -fa on -t T

# Custom prompt length
llama-bench ... -p 2048 -n 256
```

Output: `pp512` = prefill t/s, `tg128` = decode t/s.

### Sweep Strategy

1. One variable per run
2. Wide steps first (ngl +8, n_cpu_moe +8, ctx *2)
3. Narrow near ceiling (+-1~2 layers)
4. Kill processes between every run

### Benchmark Result Template

```text
| Field | Value |
|-------|-------|
| GPU | ____ |
| Driver | ____ |
| llama.cpp commit/build | ____ |
| Backend | CUDA / Vulkan / Metal / ROCm |
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

## Case Studies

Real-world tuning results for specific hardware/model combinations.

| Hardware | Backend | Model | File |
|----------|---------|-------|------|
| RTX 5060 Ti 16GB | CUDA | Qwen3.6 27B Q4_K_M | [examples/rtx5060ti-qwen27b.md](examples/rtx5060ti-qwen27b.md) |
| RTX 3060 12GB | CUDA | Qwen3.6 35B A3B Q4_K_M | [examples/rtx3060-qwen35b-moe.md](examples/rtx3060-qwen35b-moe.md) |
| RX 6600 8GB | Vulkan | Qwen3.5 MoE | [examples/rx6600-vulkan-qwen35b-moe.md](examples/rx6600-vulkan-qwen35b-moe.md) |
| RX 580 2048SP | Vulkan | Gemma 12B | [examples/rx580-vulkan-gemma12b.md](examples/rx580-vulkan-gemma12b.md) |
| GTX 1070 8GB | CUDA | Gemma 12B | [examples/gtx1070-cuda-gemma12b.md](examples/gtx1070-cuda-gemma12b.md) |

---

## Repo Structure

```
README.md                     ← You are here (generic SOP)
start-server.bat              ← Generic launch template (edit paths before use)
docs/
  cuda.md                     ← CUDA backend verification & gotchas
  vulkan.md                   ← Vulkan backend verification & gotchas
  metal.md                    ← Metal backend verification & gotchas
  rocm.md                     ← ROCm backend verification & gotchas
  dense-tuning.md             ← Dense model tuning workflow
  moe-tuning.md               ← MoE model tuning workflow
  kv-cache-sizing.md          ← KV cache sizing & dtype selection
  troubleshooting.md          ← Detailed symptom→fix map
examples/
  rtx5060ti-qwen27b.md        ← Case study: RTX 5060 Ti + Qwen 27B
  rtx3060-qwen35b-moe.md      ← Case study: RTX 3060 + Qwen 35B MoE
  rx6600-vulkan-qwen35b-moe.md
  rx580-vulkan-gemma12b.md
  gtx1070-cuda-gemma12b.md
```
