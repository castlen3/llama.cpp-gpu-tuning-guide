@echo off
setlocal

:: RTX 3060 12GB + Qwen3.6 35B A3B Q4_K_M (MoE) + 32K + q8_0 KV
:: Benchmark: tg128=26.3 t/s
:: Key: --n-cpu-moe=28 is the sweet spot for 12GB

set LLAMA_DIR=<your-llama-cpp-directory>
set MODEL=<path-to>\Qwen3.6-35B-A3B-Q4_K_M.gguf
set NGL=40
set N_CPU_MOE=28
set CTX=32768
set THREADS=8

echo ============================================
echo   Qwen3.6 35B A3B Q4_K_M (MoE) | ngl=%NGL% | n_cpu_moe=%N_CPU_MOE% | 32K
echo ============================================

"%LLAMA_DIR%\llama-server.exe" ^
  --host 0.0.0.0 ^
  -m "%MODEL%" ^
  -ngl %NGL% ^
  --n-cpu-moe %N_CPU_MOE% ^
  -c %CTX% ^
  -ctk q8_0 ^
  -ctv q8_0 ^
  -fa on ^
  -b 2048 ^
  -ub 512 ^
  -t %THREADS% ^
  -tb %THREADS% ^
  -np 1 ^
  -fit off ^
  --no-mmap

pause
