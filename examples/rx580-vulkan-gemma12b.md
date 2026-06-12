# Case Study: RX 580 2048SP + Gemma 12B (Vulkan)

## Hardware

- GPU: AMD Radeon RX 580 2048SP (Polaris)
- CPU: (fill in)
- RAM: (fill in)

## Backend

- Vulkan (ggml-vulkan.dll)
- llama.cpp build: (fill in)

## Model

- (fill in model filename)
- Dense architecture, ~12B parameters

## Goal

Run Gemma 12B on older AMD GPU via Vulkan.

## Status

Pending testing. See [docs/vulkan.md](../docs/vulkan.md) for Vulkan-specific considerations.

## Key Questions to Answer

1. Does `GGML_VK_ALLOW_GRAPHICS_QUEUE=1` help (likely yes for Polaris)?
2. What is the VRAM/GTT boundary for 2GB VRAM?
3. Is 12B feasible on 8GB (2GB VRAM + 6GB GTT)?
4. What quantization level is needed to fit?

## Template

```bat
set GGML_VK_ALLOW_GRAPHICS_QUEUE=1
llama-server.exe --list-devices
llama-server.exe -m "model.gguf" -ngl N -c C -fa on -t T
```
