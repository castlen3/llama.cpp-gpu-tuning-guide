# Case Study: GTX 1070 8GB + Gemma 12B (CUDA)

## Hardware

- GPU: NVIDIA GeForce GTX 1070 8GB
- CPU: (fill in)
- RAM: (fill in)

## Backend

- CUDA (ggml-cuda.dll)
- llama.cpp build: (fill in)

## Model

- (fill in model filename)
- Dense architecture, ~12B parameters

## Goal

Run Gemma 12B on older NVIDIA GPU via CUDA.

## Status

Pending testing.

## Key Questions to Answer

1. What is the max stable ngl for 8GB VRAM?
2. Is flash attention supported on Pascal?
3. What context length is feasible?
4. What quantization level is needed to fit?

## Template

```bat
llama-server.exe --list-devices
llama-server.exe -m "model.gguf" -ngl N -c C -fa on -t T
```
