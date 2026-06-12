# Metal Backend

Apple Silicon GPU backend for llama.cpp. Uses `ggml-metal.dylib` (macOS only).

## Verification

### 1. File Check

```sh
ls -la libggml-metal.dylib
```

### 2. Device Detection

```sh
llama-server --list-devices
```

On macOS, Metal is typically the default backend.

### 3. Runtime Verification

Start with `--verbose` and confirm in log:

```
ggml_metal_init: found Metal device
```

## Metal-Specific Considerations

### Unified Memory

Apple Silicon uses unified memory — CPU and GPU share the same physical RAM. This means:

- No separate "VRAM" — GPU uses system memory
- `-ngl` still controls layer offloading, but there's no VRAM ceiling in the traditional sense
- Performance benefit comes from GPU compute, not memory bandwidth separation

### ngl Behavior

Because there's no separate VRAM, `-ngl max` (all layers on GPU) is usually safe and recommended. The limiting factor is total system memory, not a GPU VRAM pool.

### Context Size

With unified memory, larger context windows are more feasible than on discrete GPUs. The main constraint is total system RAM.

### Metal Performance Shaders

llama.cpp uses Metal Performance Shaders (MPS) for acceleration. Performance scales well with:

- Memory bandwidth (M1 Pro/Max/Ultra have higher bandwidth)
- GPU core count

### Quantization

Metal supports all common quantization types. `q4_0` and `q8_0` are well-optimized. Exotic quants may fall back to slower paths.

### No Mixed KV Dtype Issue

Unlike CUDA, Metal does not have the asymmetric KV dtype prefill stall issue. You can safely use different K and V cache types if needed.
