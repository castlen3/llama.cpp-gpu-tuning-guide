@echo off
setlocal

echo ============================================
echo   Qwen3.6 27B Q4_K_M ^| ngl=61 ^| ctx=32768 ^| q8_0 KV
echo   http://0.0.0.0:8080
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

set LLAMA_DIR=C:\Users\castlen3\llama-cuda-5060ti-release
set MODEL=C:\Users\castlen3\.lmstudio\models\lmstudio-community\Qwen3.6-27B-GGUF\Qwen3.6-27B-Q4_K_M.gguf

"%LLAMA_DIR%\llama-server.exe" ^
  -m "%MODEL%" ^
  -c 32768 ^
  -ngl 61 ^
  -ctk q8_0 ^
  -ctv q8_0 ^
  -fa on ^
  -b 2048 ^
  -ub 512 ^
  -t 8 ^
  -tb 16 ^
  -np 1

pause
