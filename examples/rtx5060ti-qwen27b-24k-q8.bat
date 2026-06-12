@echo off
setlocal

:: RTX 5060 Ti 16GB + Qwen3.6 27B Q4_K_M + 24K + q8_0 KV
:: Max stable ngl for 16GB card
:: Benchmark: pp512=676 t/s, tg128=13.4 t/s

set LLAMA_DIR=C:\Users\castlen3\llama-cuda-5060ti-release
set MODEL=C:\Users\castlen3\.lmstudio\models\lmstudio-community\Qwen3.6-27B-GGUF\Qwen3.6-27B-Q4_K_M.gguf
set NGL=61
set CTX=24576
set THREADS=6

echo ============================================
echo   Qwen3.6 27B Q4_K_M | ngl=%NGL% | 24K | q8_0 KV
echo ============================================

"%LLAMA_DIR%\llama-server.exe" ^
  --host 0.0.0.0 ^
  -m "%MODEL%" ^
  -ngl %NGL% ^
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
