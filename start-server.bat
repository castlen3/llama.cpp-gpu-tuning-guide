@echo off
echo Qwen3.6 27B Q4_K_M ngl=61 24Kctx q8_0
echo Starting...
"C:\Users\castlen3\llama-cuda-5060ti-release\llama-server.exe" -m "C:\Users\castlen3\.lmstudio\models\lmstudio-community\Qwen3.6-27B-GGUF\Qwen3.6-27B-Q4_K_M.gguf" -ngl 61 -c 24576 -ctk q8_0 -ctv q8_0 -fa on -b 2048 -ub 512 -t 6 -np 1 -fit off --no-mmap
pause
