# Case Study: RTX 2080 Ti 22GB + Qwen3.6

This case study documents tuning traps found on a 22GB modded RTX 2080 Ti.

## Hardware

- GPU: RTX 2080 Ti 22GB
- CPU: i7-6900K
- RAM: 32GB DDR4
- Backend: CUDA
- llama.cpp build: `1a7718b4c` / b9622

## Lesson 1: One Missing GPU Layer Can Be a Decode Cliff

For Qwen3.6-35B-A3B Q4_K_M at 64K context with q8 KV:

| Config | Decode |
|---|---:|
| `-ngl 40` | 20.66 t/s |
| `-ngl 99` | about 26 t/s |

The model could run with all intended layers on GPU, but `-ngl 40` left enough
work off GPU to create a large decode penalty.

Do not assume a lower `-ngl` is harmless. If VRAM allows it, test `-ngl 99`
directly and verify from the startup log.

## Lesson 2: 64K Can Be Stable but Very Close to the Edge

Qwen3.6-35B-A3B Q4_K_M:

| Context | Config | VRAM Used | Notes |
|---|---|---:|---|
| 32K | `-ngl 99`, q8 KV | about 21214 MiB | fast decode, more headroom |
| 64K | `-ngl 99`, q8 KV | about 21654 MiB | works, but only about 874 MiB free |

For tight 64K setups, avoid background VRAM users and do not use multiple
parallel slots.

## Lesson 3: `--no-mmap` Is Not a Default Tuning Flag

On this 32GB RAM system, testing mmap on/off with 20GB-class models caused
instability. Leave mmap enabled by default. Use `--no-mmap` only as an isolated
diagnostic after a clean mmap baseline exists.

## Lesson 4: Disable Prompt Cache for Benchmarks

The server build enabled prompt cache by default with an 8192 MiB limit. On a
32GB RAM machine, this can create RAM pressure unrelated to GPU tuning.

Use:

```text
--cache-ram 0
```

for reproducible benchmarks.

## Lesson 5: MoE and Dense Have Different Prefill Behavior

Qwen3.6-35B-A3B had much faster prefill than dense 27B models on the same GPU.
That is expected: A3B/MoE activates far fewer parameters per token.

Do not call the dense model broken only because its prefill is lower.

## Lesson 6: MTP Helps Decode, Not Prefill

Qwen3.6-27B-MTP Q4_K_S at 64K q8 KV:

| Config | Decode | Draft Acceptance |
|---|---:|---:|
| no MTP | 24.88 t/s | n/a |
| MTP `n_max=1` | 37.64 t/s | 88.89% |
| MTP `n_max=2` | 40.96 t/s | 80.10% |
| MTP `n_max=4` | 33.75 t/s | 55.03% |

The sweet spot was:

```text
--spec-type draft-mtp --spec-draft-n-max 2
```

Long-prompt prefill stayed around 650-680 t/s with or without MTP. That means
MTP was not the cause of slow prefill.

## Known Good Launch Patterns

### 35B-A3B 64K q8 KV

```bat
llama-server.exe ^
  -m "Qwen3.6-35B-A3B-Q4_K_M.gguf" ^
  -ngl 99 ^
  -c 65536 ^
  -ctk q8_0 -ctv q8_0 ^
  -fa on ^
  -t 8 ^
  -ncmoe 0 ^
  -fit off ^
  --cache-ram 0 ^
  -b 2048 -ub 512 ^
  --host 127.0.0.1 ^
  --port 8080 ^
  -np 1
```

### 27B-MTP 64K q8 KV

```bat
llama-server.exe ^
  -m "Qwen3.6-27B-Q4_K_S.gguf" ^
  -ngl 99 ^
  -c 65536 ^
  -ctk q8_0 -ctv q8_0 ^
  -fa on ^
  -t 8 ^
  -fit off ^
  --cache-ram 0 ^
  -b 2048 -ub 512 ^
  --spec-type draft-mtp ^
  --spec-draft-n-max 2 ^
  --host 127.0.0.1 ^
  --port 8080 ^
  -np 1
```
