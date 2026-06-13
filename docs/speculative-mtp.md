# Speculative Decoding and MTP

Speculative decoding is a decode optimization. It is not a prefill
optimization.

## When It Helps

Use MTP/speculative decoding when:

- prompts are short or moderate
- generated outputs are long
- draft acceptance is high
- VRAM has room for the draft context

It is less useful when:

- workload is dominated by very long prompt ingestion
- draft acceptance is low
- VRAM is already at the edge

## Required Measurements

Always measure both paths:

```text
short prompt + 128/256 generated tokens  -> decode speed
4K+ prompt + 16 generated tokens         -> prefill speed
```

Record from server timing:

```text
prompt_per_second
predicted_per_second
draft_n
draft_n_accepted
```

Acceptance:

```text
acceptance = draft_n_accepted / draft_n
```

## MTP Sweep

Start with:

```text
--spec-type draft-mtp --spec-draft-n-max 1
--spec-type draft-mtp --spec-draft-n-max 2
--spec-type draft-mtp --spec-draft-n-max 3
--spec-type draft-mtp --spec-draft-n-max 4
```

Stop increasing when decode speed drops or acceptance falls sharply.

Higher `n_max` is not automatically better. Extra rejected draft tokens are
wasted work.

## Draft KV

The draft context has its own KV cache options:

```text
-ctkd q8_0 -ctvd q8_0
```

Do not force draft KV quantization unless VRAM requires it. Test first. On some
CUDA builds, default draft KV can be faster.

## Example Interpretation

If results look like this:

```text
baseline decode: 25 t/s
MTP n=1:         38 t/s, acceptance 89%
MTP n=2:         41 t/s, acceptance 80%
MTP n=4:         34 t/s, acceptance 55%
```

Choose `n=2`. The larger draft window is slower because acceptance drops.

If prefill remains slow with MTP on, compare with MTP off. If both are similar,
MTP is not the cause; the model/backend path is prefill-bound.
