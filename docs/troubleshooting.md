# Troubleshooting

## Quick Symptom Map

| Symptom | Section |
|---------|---------|
| `--list-devices` empty | [No CUDA backend](#no-cuda-backend) |
| VRAM unchanged after model load | [Offload failure](#offload-failure) |
| nvidia-smi shows VRAM used but GPU utilization ~0% | [GPU idle despite VRAM](#gpu-idle-despite-vram) |
| Prompt eval never finishes | [Prefill hang](#prefill-hang) |
| `-fa` parse error | [Flag syntax mismatch](#flag-syntax-mismatch) |
| .bat flashes and closes | [Port conflict](#port-conflict) |
| .bat says `'ngl' is not recognized ...` | [Batch continuation trap](#batch-continuation-trap) |
| llama-bench OOM at previously-working ngl | [Stale process](#stale-process) |
| `/health` OK but `/slots` timeout | [CUDA graph hang](#cuda-graph-hang) |

---

## No CUDA Backend

**Symptoms**: `--list-devices` shows no CUDA0, or not found.

**Causes**:
- Binary compiled without CUDA (`GGML_CUDA=OFF`)
- Downloaded `llama-bin-win-cuda` instead of `cudart-llama-bin-win-cuda`
- Missing `ggml-cuda.dll`

**Fix**: Get a `cudart-llama-bin-win-cuda` build. Verify `ggml-cuda.dll` exists and `--list-devices` shows CUDA0.

---

## Offload Failure

**Symptoms**: nvidia-smi VRAM doesn't rise after model load. `offloaded 0/N layers` in verbose log.

**Causes**:
- `-fit on` (auto-fit) making wrong decisions
- mmap issues on Windows

**Fix**:
```bat
-fit off --no-mmap
```
Restart and check verbose log for `offloaded XX/NN layers` where XX > 0.

---

## GPU Idle Despite VRAM

**Symptoms**:
- nvidia-smi shows VRAM occupied (e.g. 14 GB)
- GPU utilization stays near 0%
- Inference works but feels slow

**Causes and checks**:

### 1. Wrong monitoring tool
Windows Task Manager often shows the wrong GPU engine for CUDA workloads. Use:
```bat
nvidia-smi -l 1
```
Look at the `Volatile GPU-Util` column.

### 2. Low batch size
At `-np 1` with small prompts, GPU finishes compute in microseconds and idles waiting for CPU to prepare the next token. This is **normal** — low utilization ≠ problem.

### 3. CUDA graphs idle between tokens
CUDA graph mode completes each decode step in one launch then waits. Utilization spikes are brief.

### 4. Offload actually working?
Confirm from verbose log:
```
llama_prepare_model_devices: using device CUDA0
CUDA0 KV buffer size = ...
```

### 5. CPU bottleneck
If CPU is pegged at 100% during inference, the GPU is waiting on CPU. Reduce `-t` or check for background processes.

---

## Prefill Hang

**Symptoms**: Prompt eval starts but never finishes. No error, just stalls.

**Causes**:
- **Mixed KV dtype** (`-ctk q8_0 -ctv q4_0`): known to cause prefill stalls on some CUDA backends
- CUDA graph capture issue with certain batch/ctx combinations
- Driver timeout on very large prompts

**Fix**:
1. Use symmetric KV types: both `q8_0` or both `q4_0`
2. Try `--no-graph` if available in your build
3. Reduce `-b` or `-ub` temporarily

---

## Flag Syntax Mismatch

**Symptoms**: Error parsing `-fa` or other flags.

**Causes**: Builds use different flag syntax.

**Fix**: Always check `llama-server.exe --help` first. Common variations:

| Old builds | Newer builds (≥b9596) |
|-----------|----------------------|
| `-fa` | `-fa on` or `--flash-attn on` |
| `--n-gpu-layers` | `-ngl` or `--gpu-layers` |

Use the exact form shown in your build's `--help`, not what you see in online guides.

---

## Port Conflict

**Symptoms**: .bat flashes open and immediately closes.

**Causes**: Another llama-server is already running on port 8080.

**Fix**:
```powershell
Get-Process -Name llama-server | Stop-Process -Force
```
Or use a different port: `--port 8081`

---

## Batch Continuation Trap

**Symptoms**: Double-click `.bat`, window opens then immediately shows:
```
'ngl' is not recognized as an internal or external command
```
(or any parameter name as a command).

**Cause**: The `^` line-continuation character in `.bat` files **must** be the very last character before the newline — no trailing spaces. If there's any space after `^`, cmd.exe treats the next line as a new command instead of continuing the previous one.

```
:: BROKEN — space after ^
llama-server.exe ^
  -ngl 61
:: cmd sees: llama-server.exe [broken] then tries to run "-ngl" as a command

:: WORKING — ^ is last char
llama-server.exe ^
-ngl 61
```

**Fix**: Two options:
1. **Single-line command** (recommended, most reliable):
   ```bat
   "%LLAMA_DIR%\llama-server.exe" --host 0.0.0.0 -m "%MODEL%" -ngl 61 -c 24576 ...
   ```
2. **`^` continuation** — ensure ZERO trailing characters after `^` before the newline. Use a text editor that shows whitespace, or write in a tool that doesn't add trailing spaces.

**Why this keeps happening**: Many editors and AI-generated content silently add trailing spaces. The `.bat` looks correct visually but fails at runtime. The single-line approach eliminates this entirely.

## Stale Process

**Symptoms**: llama-bench OOM at ngl that previously worked.

**Causes**: Previous llama-bench or llama-server process still holding VRAM.

**Fix**:
```powershell
Get-Process -Name llama-cli,llama-server,llama-bench -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep 3
```
Verify with `nvidia-smi` that VRAM returned to idle levels.

---

## CUDA Graph Hang

**Symptoms**: Server starts, `/health` returns OK, but `/slots` or chat requests timeout.

**Causes**:
- CUDA graph capture failed silently
- Specific ctx/batch combination breaks graph building

**Fix**:
1. Check `--verbose` log for `CUDA graph warmup complete` or errors
2. Try different `-c` or `-b` values
3. Try `--no-graph` if build supports it

---

## Driver/OS Issues

### Windows-specific

- **DLL not found**: Ensure all CUDA DLLs (`cublas64_*.dll`, `cudart64_*.dll`) are in the same directory as `llama-server.exe`
- **Antivirus**: Some AV software blocks `ggml-cuda.dll` — add exception
- **Power mode**: Set to "High Performance" to prevent GPU throttling

### General

- **Driver version**: Use latest Game Ready or Studio driver from NVIDIA
- **WSL**: CUDA in WSL requires `nvidia-smi` to work inside WSL first
- **Multiple GPUs**: Use `--main-gpu` to select which GPU
