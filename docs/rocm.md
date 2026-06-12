# ROCm Backend

AMD GPU backend for llama.cpp (Linux). Uses `libggml-hip.so`.

ROCm is AMD's compute platform, analogous to CUDA for NVIDIA. Availability is limited compared to CUDA.

## Verification

### 1. ROCm Installation

```sh
rocminfo
```

Must show your GPU. If not installed, see [AMD ROCm docs](https://rocm.docs.amd.com/).

### 2. File Check

```sh
ls -la libggml-hip.so
```

### 3. Device Detection

```sh
llama-server --list-devices
```

### 4. Runtime Verification

Start with `--verbose` and confirm in log:

```
ggml_hip_init: found N HIP devices
```

## ROCm-Specific Considerations

### Supported GPUs

ROCm officially supports:

- MI-series (datacenter)
- RX 7900 series (consumer, partial)
- RX 6000 series (consumer, limited)

Older AMD GPUs (RX 500, Vega) have community support but may be unstable.

### Windows Limitation

ROCm is not available on Windows for consumer GPUs. Use Vulkan instead. See [docs/vulkan.md](docs/vulkan.md).

### VRAM Monitoring

```sh
# Check VRAM usage
rocm-smi
```

### Performance

ROCm performance is generally competitive with CUDA on supported hardware. The main barrier is hardware compatibility, not performance.

### Alternative: Vulkan

If ROCm is not available for your GPU, Vulkan is the fallback. See [docs/vulkan.md](docs/vulkan.md).
