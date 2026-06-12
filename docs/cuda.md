# CUDA Backend

NVIDIA GPU backend for llama.cpp. Uses `ggml-cuda.dll` (Windows) or `libggml-cuda.so` (Linux).

## Verification

### 1. File Check

```sh
# Windows
dir /b | findstr ggml-cuda

# Linux
ls -la libggml-cuda.so
```

The file should be ~150 MB. If absent, the binary has no CUDA support.

### 2. Device Detection

```sh
llama-server --list-devices
```

Must show:

```
Available devices:
  CUDA0: NVIDIA GeForce ... (XXXX MiB, XXXX MiB free)
```

If CUDA0 is absent, get a `cudart-llama-bin-win-cuda` build (Windows) or rebuild with CUDA support (Linux).

### 3. Runtime Verification

Start with `--verbose` and confirm in log:

```
ggml_cuda_init: found N CUDA devices
load_backend: loaded CUDA backend from .../ggml-cuda.dll
llama_prepare_model_devices: using device CUDA0
load_tensors: offloaded XX/NN layers to GPU
```

### 4. VRAM Monitor

```sh
nvidia-smi -l 1
```

After model load, VRAM should rise significantly. If not, offload is not active.

## CUDA-Specific Gotchas

### KV Cache: Avoid Asymmetric Dtypes

On CUDA, mixing KV cache types (e.g. `-ctk q8_0 -ctv q4_0`) may cause prefill stalls or hangs. Use symmetric settings:

- `-ctk q8_0 -ctv q8_0` (recommended when VRAM allows)
- `-ctk q4_0 -ctv q4_0` (when VRAM is tight)

This is a CUDA backend issue. Other backends (Vulkan, Metal) may not have this problem.

### Flash Attention Syntax

Syntax varies by build. Check `--help`:

- Older builds: `-fa`
- Newer builds (>= b9596): `-fa on`
- Some builds: `--flash-attn on`

Use exactly what `--help` shows.

### nvidia-smi vs Windows Task Manager

Windows Task Manager shows the wrong engine for CUDA workloads. Use `nvidia-smi -l 1` to monitor GPU utilization.

### VRAM Occupied but GPU Utilization ~0%

Possible causes:

1. Offload failed (check `offloaded N/M layers` in verbose log)
2. Mixed KV dtype causing prefill stall
3. CUDA graph capture bug (try `--no-graph` if available)
4. Backend not loaded (check `load_backend: loaded CUDA backend` in log)

### mmap Issues on Windows

If model load hangs or fails on Windows, try `--no-mmap` as a workaround.

## Common False Positives

- Binary named "cuda" but no `ggml-cuda.dll` → fake build
- `--list-devices` shows CPU only → no CUDA runtime linked
- VRAM rises but `offloaded 0/NN layers` → offload failed, weights on CPU
- GPU utilization low but VRAM occupied → check above gotchas
