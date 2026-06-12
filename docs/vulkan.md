# Vulkan Backend

Cross-platform GPU backend for llama.cpp. Uses `ggml-vulkan.dll` (Windows) or `libggml-vulkan.so` (Linux).

Works on NVIDIA, AMD, and Intel GPUs. Primary use case: AMD GPUs on Windows where ROCm is unavailable.

## Verification

### 1. File Check

```sh
# Windows
dir /b | findstr ggml-vulkan

# Linux
ls -la libggml-vulkan.so
```

### 2. Device Detection

```sh
llama-server --list-devices
```

Must show your GPU. If absent, the binary has no Vulkan support.

### 3. Runtime Verification

Start with `--verbose` and confirm in log:

```
ggml_vulkan_init: found N Vulkan devices
```

## Vulkan-Specific Gotchas

### Queue Family Issues

Vulkan has multiple queue families (compute, graphics, transfer). Some AMD GPUs require the graphics queue for compute operations.

**Symptom:** Model loads but inference hangs or crashes.

**Fix:** Set environment variable:

```sh
# Windows
set GGML_VK_ALLOW_GRAPHICS_QUEUE=1

# Linux
export GGML_VK_ALLOW_GRAPHICS_QUEUE=1
```

This is particularly relevant for older AMD GPUs (Polaris/RX 500 series).

### AMD VRAM vs GTT

On AMD Windows drivers, `nvidia-smi` equivalent tools may not show accurate VRAM usage. AMD reports:

- **VRAM**: Actual GPU memory
- **GTT**: Shared memory (similar to system RAM, much slower)

llama.cpp may silently fall back to GTT when VRAM is exhausted, causing severe performance degradation without an OOM error. Monitor with:

```sh
# Windows: use AMD Radeon Software or GPU-Z
# Linux: check /sys/class/drm/card0/device/mem_info_vram_used
```

### Shader Compilation

First inference on Vulkan is slow due to shader compilation. Subsequent runs use cached shaders. Don't benchmark the first run.

### Performance vs CUDA

Vulkan is generally slower than CUDA on NVIDIA GPUs. Use CUDA when available. Vulkan's primary value is on AMD Windows where ROCm is not an option.

### Memory Allocation

Vulkan may report different VRAM totals than expected. Some drivers reserve VRAM for display output. Check actual available VRAM with `--list-devices` output.
