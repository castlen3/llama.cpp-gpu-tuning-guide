# Case Study: RX 6600 8GB + Qwen3.5 MoE (Vulkan)

## Hardware

- GPU: AMD Radeon RX 6600 8GB
- CPU: (fill in)
- RAM: (fill in)

## Backend

- Vulkan (ggml-vulkan.dll)
- llama.cpp build: (fill in)

## Model

- (fill in model filename)
- MoE architecture

## Goal

Run MoE model on AMD Windows via Vulkan with acceptable decode speed.

## Status

Pending testing. See [docs/vulkan.md](../docs/vulkan.md) for Vulkan-specific considerations.

## Key Questions to Answer

1. Does `GGML_VK_ALLOW_GRAPHICS_QUEUE=1` help?
2. What is the VRAM/GTT boundary?
3. How does shader compilation overhead affect first-run benchmarks?
4. What n_cpu_moe values are supported?

## Template

```bat
set GGML_VK_ALLOW_GRAPHICS_QUEUE=1
llama-server.exe --list-devices
llama-server.exe -m "model.gguf" -ngl N --n-cpu-moe M -c C -fa on -t T
```
