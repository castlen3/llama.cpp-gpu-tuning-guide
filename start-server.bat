@echo off
setlocal

:: ============================================================
:: Generic llama-server launch template
:: Edit the paths and parameters below for your hardware/model
:: ============================================================

echo ============================================
echo   llama-server launch template
echo   Edit this file before use
echo ============================================
echo.

echo [1/3] Cleaning up old processes...
taskkill /F /IM llama-server.exe /IM llama-cli.exe /IM llama-bench.exe >nul 2>&1
timeout /t 3 /nobreak >nul

echo [2/3] Freeing port 8080...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8080 " ^| findstr "LISTENING"') do (taskkill /F /PID %%a >nul 2>&1)
timeout /t 1 /nobreak >nul

echo [3/3] Starting llama-server...
echo.

:: ============================================================
:: EDIT THESE PATHS
:: ============================================================
set LLAMA_DIR=C:\path\to\llama-cpp\bin
set MODEL=C:\path\to\model.gguf

:: ============================================================
:: EDIT THESE PARAMETERS (see README.md for tuning guide)
:: ============================================================
:: -ngl N          GPU layers (start at 50%% of total, sweep up)
:: -c N            Context size (start at 4096, double until OOM)
:: -ctk/-ctv       KV cache type (q8_0 if VRAM allows, q4_0 if tight)
:: -fa on          Flash attention (check --help for exact syntax)
:: -b/-ub          Batch sizes (2048/512 is safe default)
:: -t/-tb          CPU threads (start at physical_cores/2)
:: -np 1           Parallel slots (1 for single-user)
:: ============================================================

"%LLAMA_DIR%\llama-server.exe" ^
  --host 0.0.0.0 ^
  -m "%MODEL%" ^
  -ngl 32 ^
  -c 8192 ^
  -ctk q8_0 ^
  -ctv q8_0 ^
  -fa on ^
  -b 2048 ^
  -ub 512 ^
  -t 4 ^
  -np 1

pause
