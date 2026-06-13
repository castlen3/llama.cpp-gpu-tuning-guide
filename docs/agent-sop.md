# Agent SOP

This page is written for agents. Follow it literally before making claims about
GPU performance.

## Non-Negotiable Rules

1. Do not compare `llama-bench` results to `llama-server` results as if they are
   the same benchmark.
2. Do not use `--no-mmap` by default on Windows. It can put heavy pressure on
   system RAM and make a previously stable configuration slow or unstable.
3. Do not leave llama processes running between tests.
4. Do not conclude that a GPU is slow until you have verified:
   - clean VRAM and RAM before the run
   - exact model file and quant
   - exact llama.cpp build
   - context size
   - KV cache dtype
   - `-ngl`
   - MoE expert placement
   - speculative/MTP settings
   - server timing fields for prefill and decode
5. Always separate prefill speed from decode speed. A setting can improve decode
   and do nothing for prefill.

## Clean-State Check

Before every real test on Windows:

```bat
taskkill /F /IM llama-server.exe /IM llama-cli.exe /IM llama-bench.exe
timeout /t 3 /nobreak >nul
nvidia-smi
```

Expected:

- no llama process remains
- VRAM is close to desktop idle usage
- system RAM has enough free space for the model loader and server overhead

If VRAM is unexpectedly high, stop. Find the owner before testing.

## Benchmark Order

Use this order:

1. `llama-server --help`
2. `llama-server --list-devices`
3. one safe server startup with target context and KV dtype
4. short decode request
5. long prompt request
6. only then run sweeps

`llama-bench` is useful for quick layer and decode scouting. It is not a
replacement for a target-context server run.

## Server Timing Fields

When testing `llama-server`, record these separately:

```text
prompt_per_second      = prefill speed
predicted_per_second   = decode speed
draft_n                = speculative draft tokens, if enabled
draft_n_accepted       = accepted draft tokens, if enabled
```

For a long-context setup, a useful minimum is:

- short prompt, `n_predict=128` or `256`, `ignore_eos=true`
- 4K-token prompt, `n_predict=16`

The short prompt isolates decode. The 4K prompt exposes prefill.

## mmap Policy

Use mmap on by default.

Only test `--no-mmap` after a clean mmap baseline exists, and record it as a
separate experiment. On Windows with 20GB-class GGUF files and 32GB RAM,
`--no-mmap` can cause paging pressure, unstable load behavior, or OOM-like
failures.

## Prompt Cache Policy

Newer `llama-server` builds may enable prompt cache by default, often with an
8192 MiB limit.

For reproducible benchmarking, add:

```text
--cache-ram 0
```

Do this especially on 32GB RAM machines. Otherwise RAM pressure can dominate the
test and make unrelated parameters look bad.

## `-ngl` Policy

If VRAM is clearly enough, start with:

```text
-ngl 99
```

Then confirm from the startup log that all intended layers are offloaded. Do not
assume that a lower `-ngl` is safer or faster. Some models show a large decode
cliff when only one layer remains off GPU.

If VRAM is tight, sweep `-ngl` down from the highest successful value. Keep the
highest stable value that does not regress speed.

## MoE Policy

For MoE models, do not blindly push experts to CPU.

If VRAM allows it, first test:

```text
-ngl 99
-ncmoe 0
```

CPU expert offload is a VRAM-saving tool, not a performance tool. On older CPUs
or memory-bound platforms, CPU MoE can dominate decode latency.

## Dense vs MoE Prefill

Do not infer prefill speed from parameter count alone.

A 35B MoE/A3B model may prefill faster than a dense 27B model because fewer
parameters are active per token. Dense models can have slower prefill even when
their decode is acceptable.

## Speculative / MTP Policy

MTP/speculative decoding improves decode only. It does not make prompt prefill
fast.

For MTP, sweep draft count:

```text
--spec-type draft-mtp --spec-draft-n-max 1
--spec-type draft-mtp --spec-draft-n-max 2
--spec-type draft-mtp --spec-draft-n-max 3
--spec-type draft-mtp --spec-draft-n-max 4
```

Record acceptance rate. If acceptance falls sharply, more draft tokens can be
slower.

Do not judge MTP using long-prompt total time alone. Measure:

- prefill with a long prompt
- decode with a short prompt
- acceptance rate

## Minimum Result Table

Every tuning result should include:

```text
| model | quant | llama.cpp build | backend | ctx | ngl | ctk/ctv | mmap | cache-ram | MoE/MTP flags | prefill t/s | decode t/s | VRAM MiB | RAM free | notes |
```

If any field is missing, the result is not comparable.
