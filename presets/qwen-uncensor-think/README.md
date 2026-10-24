# Qwen3.8-Flash-Next: thinking-aware uncensor preset (opt-in)

The default preset in the model repo was fitted on non-thinking chat. This one also uses thinking-mode activations, so it holds up better with thinking on.

Use it by pointing the runtime at the spec (the direction file must sit next to it):

```
EXL3_ABLIT_RUNTIME=/path/to/presets/qwen-uncensor-think/uncensor_spec.json
```

Measured on Qwen3.8-Flash-Next-EXL3-Yamz, batch 1:

| | default preset | this preset |
|---|---|---|
| Refusals, harmful set, thinking on (n=100) | 15 | 9 |
| Refusals, harmful set, thinking off, greedy (n=100) | 14 | 5 |
| XSTest over-refusal, thinking on (n=50) | - | 0 |
| KL vs base, thinking off / top-1 | 0.013 / 0.988 | 0.017 / 0.985 |
| KL vs base, thinking on / top-1 | 0.032 / 0.946 | 0.078 / 0.906 |
| Greedy loops, thinking on (20 prompts) | - | 0 |

It moves the model more than the default in thinking mode, so it stays opt-in for now.
