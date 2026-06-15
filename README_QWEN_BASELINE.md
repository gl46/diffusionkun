# Qwen baseline note

Use Qwen as a ruler, not as the main DiffusionKun line.

Recommended first baseline:

- `Qwen/Qwen2.5-0.5B` Base for simplest AR-SFT baseline.
- `Qwen/Qwen3-0.6B-Base` as the stronger small-base baseline.

Avoid starting with a non-standard 0.8B hybrid/multimodal-style model unless you explicitly want to debug architecture adaptation.

Baseline goal:

```text
Same data -> AR-SFT small model -> quality/latency reference
Same eval -> DiffusionKun diffusion -> quality/latency trade-off
```

Do not train the main diffusion translator on chat/instruct formatted outputs.
