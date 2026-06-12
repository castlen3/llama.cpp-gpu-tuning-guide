@echo off
setlocal enabledelayedexpansion

:: RTX 5060 Ti 16GB + Qwen3.6 27B Q4_K_M + 32K + q4_0 KV
:: Sacrifice KV precision for longer context

set LLAMA_DIR=C:\Users\castlen3\llama-cuda-5060ti-release
set MODEL=C:\Users\castlen3\.lmstudio\models\lmstudio-community\Qwen3.6-27B-GGUF\Qwen3.6-27B-Q4_K_M.gguf
set NGL=61
set CTX=32768
set THREADS=6
set PORT=8080

echo ============================================
echo   Qwen3.6 27B Q4_K_M | ngl=%NGL% | 32K | q4_0 KV
echo ============================================

echo [1/3] Cleaning up old processes...
taskkill /F /IM llama-server.exe /IM llama-cli.exe /IM llama-bench.exe >nul 2>&1
timeout /t 3 /nobreak >nul

echo [2/3] Checking port %PORT%...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":%PORT% " ^| findstr "LISTENING"') do (
    echo   Killing PID %%a on port %PORT%
    taskkill /F /PID %%a >nul 2>&1
)
timeout /t 1 /nobreak >nul

echo [3/3] Starting...
echo.

"%LLAMA_DIR%\llama-server.exe" ^
  --host 0.0.0.0 ^
  --port %PORT% ^
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
