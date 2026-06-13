# Troubleshooting

## Quick Symptom Map

| Symptom | Section |
|---------|---------|
| `--list-devices` empty | [No GPU backend](#no-gpu-backend) |
| VRAM unchanged after model load | [Offload failure](#offload-failure) |
| GPU VRAM occupied but utilization ~0% | [Backend fallback](#backend-fallback) |
| Prompt eval never finishes | [Prefill hang](#prefill-hang) |
| `-fa` parse error | [Flag syntax mismatch](#flag-syntax-mismatch) |
| .bat flashes and closes | [Port conflict](#port-conflict) |
| .bat says `'ngl' is not recognized` | [Batch continuation trap](#batch-continuation-trap) |
| llama-bench OOM at previously-working ngl | [Stale process](#stale-process) |
| `/health` OK but `/slots` timeout | [Graph hang](#graph-hang) |
| Server loads model then stops logging, never ready | [Graph hang](#graph-hang) |
| Slow decode on MoE | [Missing MoE parameter](#missing-moe-parameter) |
| Random OOM or slowdown on Windows | [RAM pressure from no-mmap or prompt cache](#ram-pressure-from-no-mmap-or-prompt-cache) |
| Fast MTP decode but slow long prompts | [MTP prefill misconception](#mtp-prefill-misconception) |
| Big slowdown from one lower `-ngl` | [Layer offload cliff](#layer-offload-cliff) |

---

## No GPU Backend

**Symptoms**: `--list-devices` shows only CPU, or is empty.

**Cause**: Binary has no GPU backend support.

**Fix**:
- CUDA: Get `cudart-llama-bin-win-cuda` build, verify `ggml-cuda.dll` exists (~150 MB)
- Vulkan: Verify `ggml-vulkan.dll` exists
- ROCm: Verify `libggml-hip.so` exists (Linux only)

---

## Offload Failure

**Symptoms**: VRAM unchanged after model load, `offloaded 0/NN layers` in verbose log.

**Cause**: Layer offload failed despite GPU being detected.

**Fix**:
1. Try `-fit off`
2. Check `--verbose` log for error messages
3. Verify backend library loaded: `load_backend: loaded [backend] from ...`
4. Use `--no-mmap` only as a separate diagnostic after a clean mmap baseline
5. See backend-specific docs: [cuda.md](cuda.md), [vulkan.md](vulkan.md), [rocm.md](rocm.md)

---

## Backend Fallback

**Symptoms**: GPU VRAM occupied but GPU utilization near 0%.

**Cause**: Weights loaded to VRAM but compute falling back to CPU.

**Possible Causes**:
1. Offload failed (check `offloaded N/M layers` in verbose log)
2. Mixed KV dtype causing prefill stall (CUDA)
3. CUDA graph capture bug (try `--no-graph` if available)
4. Backend not fully loaded (check `load_backend` in log)

**Fix**:
- Switch to symmetric KV dtype (both q8_0 or both q4_0)
- Try `--no-graph` if available
- Monitor with backend-specific tool (nvidia-smi, rocm-smi, GPU-Z)

---

## Prefill Hang

**Symptoms**: Prompt evaluation never completes.

**Cause**: Usually backend-specific. On CUDA, commonly caused by mixed KV dtype.

**Fix**:
- Switch to symmetric KV: `-ctk q8_0 -ctv q8_0` or `-ctk q4_0 -ctv q4_0`
- Reduce context size
- Try `--no-graph`

---

## Flag Syntax Mismatch

**Symptoms**: `-fa` parse error at startup.

**Cause**: Different builds use different flag syntax.

**Fix**: Run `--help` and use exactly what it shows:
- `-fa` (older builds)
- `-fa on` (newer builds)
- `--flash-attn on`

---

## Port Conflict

**Symptoms**: .bat opens then closes immediately.

**Cause**: Port already in use by another llama process.

**Fix**:
```bat
taskkill /F /IM llama-server.exe /IM llama-cli.exe /IM llama-bench.exe
```
Or use a different port: `--port 8081`

---

## Batch Continuation Trap

**Symptoms**: `.bat` shows `'ngl' is not recognized as an internal or external command`.

**Cause**: The `^` line-continuation character must be the very last character before the newline — no trailing spaces. If there's any space after `^`, cmd.exe treats the next line as a new command.

```bat
:: BROKEN — space after ^
llama-server.exe ^
  -ngl 61

:: WORKING — ^ is last char
llama-server.exe ^
-ngl 61
```

**Fix**: Use single-line command, or ensure zero trailing characters after `^`.

---

## Stale Process

**Symptoms**: llama-bench OOM at ngl that previously worked.

**Cause**: Previous llama process still holding VRAM.

**Fix**:
```bat
taskkill /f /im llama-server.exe /im llama-cli.exe /im llama-bench.exe
```
> `taskkill /f` is more reliable than `Stop-Process -Force` on Windows for clearing VRAM.

Kill before every run, not just when problems appear.

---

## Graph Hang

**Symptoms**: Server loads model, allocates KV cache, then stops producing output. No "HTTP server listening" message. `/health` may return OK but `/slots` times out. On large contexts (>= 32K), the process may appear running but produce zero log output for minutes.

**Cause**: CUDA graph compilation can hang silently, especially on large contexts and in background/daemon mode. Foreground execution often works fine with the same parameters.

**Fix**:
1. Run in **foreground** first to confirm server starts correctly
2. Try `--no-graph` if available in your build
3. Reduce context size as temporary workaround
4. Once verified in foreground, switch to background

---

## Missing MoE Parameter

**Symptoms**: MoE model decode is unexpectedly slow.

**Cause**: `--n-cpu-moe` not set, experts placed suboptimally.

**Fix**: Check if `--n-cpu-moe` is available in your build. See [docs/moe-tuning.md](moe-tuning.md).

---

## RAM Pressure From no-mmap or Prompt Cache

**Symptoms**:
- Windows becomes unstable during tests
- a previously working config OOMs
- VRAM looks fine but system RAM is low
- repeated bench/server runs get slower or crash

**Cause**:
Large GGUF files plus `--no-mmap` can put heavy pressure on system RAM. Newer
`llama-server` builds may also enable prompt cache by default, often with an
8192 MiB limit.

**Fix**:
1. Leave mmap enabled by default
2. Add `--cache-ram 0` for benchmarks
3. Kill all llama processes before each run
4. Record free RAM and VRAM before testing
5. Test `--no-mmap` only as an isolated experiment

---

## MTP Prefill Misconception

**Symptoms**:
- MTP decode is fast
- long prompt ingestion still feels slow
- total request time is disappointing for document-heavy prompts

**Cause**:
MTP/speculative decoding accelerates decode, not prompt prefill. Dense models
can remain prefill-bound even when MTP is working perfectly.

**Fix**:
Measure separately:

```text
short prompt + 128/256 generated tokens -> decode
4K prompt + 16 generated tokens         -> prefill
```

Compare MTP on/off. If prefill is similar in both cases, MTP is not the cause.
See [speculative-mtp.md](speculative-mtp.md).

---

## Layer Offload Cliff

**Symptoms**:
- `-ngl N` is much slower than `-ngl N+1`
- CPU usage rises during decode
- a high-VRAM GPU appears slower than a smaller newer GPU

**Cause**:
One or more final layers may remain on CPU/host memory. On some models this
creates a large decode cliff.

**Fix**:
1. If VRAM allows, test `-ngl 99`
2. Verify offload count from the startup log
3. Run target-context `llama-server`, not only `llama-bench`
4. Keep mmap on and use `--cache-ram 0` during verification
