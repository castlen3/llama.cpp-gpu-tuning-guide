# CUDA Backend Verification

Before any tuning, confirm the binary actually uses CUDA.

## Quick Check

```bat
llama-server.exe --list-devices
```

Must show:
```
Available devices:
  CUDA0: NVIDIA GeForce ... (XXXX MiB, XXXX MiB free)
  CPU   : ...
```

If CUDA0 is absent → binary has no CUDA backend. Get a `cudart-llama-bin-win-cuda` build.

## Full Verification

### 1. File Check

```bat
dir /b | findstr ggml-cuda
```

Must return `ggml-cuda.dll` (~150 MB).

### 2. Binary Info

```bat
llama-server.exe --version
```

Look for CUDA arch info in output (e.g. `CUDA : ARCHS = ...`).

### 3. Help Dump

```bat
llama-server.exe --help > llama-help.txt 2>&1
```

Search for GPU-related flags:
```
-ngl, --n-gpu-layers, --gpu-layers
-ctk, --cache-type-k
-ctv, --cache-type-v
-fa, --flash-attn
-t, --threads
-tb, --threads-batch
--n-cpu-moe, -ncmoe
```

**Use exactly the flag syntax shown in `--help`.** Different builds use different forms.

### 4. Runtime Check

Start server with `--verbose` and confirm in log:
```
ggml_cuda_init: found N CUDA devices
load_backend: loaded CUDA backend from ...\ggml-cuda.dll
llama_prepare_model_devices: using device CUDA0
```

After model loads, confirm offloaded layers > 0:
```
load_tensors: offloaded XX/NN layers to GPU
```

If `offloaded 0/NN layers` → offload failure. Try `-fit off --no-mmap`.

### 5. VRAM Monitor

```bat
nvidia-smi -l 1
```

After model load, VRAM should rise significantly (several GB). If not, CUDA offload is not active.

## Common False Positives

- Binary named "cuda" but no `ggml-cuda.dll` → fake
- `--list-devices` shows CPU only → no CUDA runtime linked
- VRAM rises but `offloaded 0/NN layers` → offload failed, weights on CPU
- GPU utilization low but VRAM occupied → check [troubleshooting.md](troubleshooting.md)
