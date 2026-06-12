@echo off
setlocal

:: RTX 5060 Ti 16GB + Qwen3.6 27B Q4_K_M + 32K + q4_0 KV
:: Sacrifice KV precision for longer context
:: Expect ~10% lower decode quality, minimal speed difference

set LLAMA_DIR=C:\Users\castlen3\llama-cuda-5060ti-release
set MODEL=C:\Users\castlen3\.lmstudio\models\lmstudio-community\Qwen3.6-27B-GGUF\Qwen3.6-27B-Q4_K_M.gguf
set NGL=61
set CTX=32768
set THREADS=6

echo ============================================
echo   Qwen3.6 27B Q4_K_M | ngl=%NGL% | 32K | q4_0 KV
echo ============================================

"%LLAMA_DIR%\llama-server.exe" ^
  --host 0.0.0.0 ^
  -m "%MODEL%" ^
  -ngl %NGL% ^
  -c %CTX% ^
  -ctk q4_0 ^
  -ctv q4_0 ^
  -fa on ^
  -b 2048 ^
  -ub 512 ^
  -t %THREADS% ^
  -tb %THREADS% ^
  -np 1 ^
  -fit off ^
  --no-mmap

pause
