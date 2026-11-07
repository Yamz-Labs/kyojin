# Prompt-cache usage

Kyojin's OpenAI-compatible responses report cache reads and measured writes in
the standard input-token breakdown:

```json
{
  "usage": {
    "prompt_tokens": 100,
    "completion_tokens": 10,
    "total_tokens": 110,
    "prompt_tokens_details": {
      "cached_tokens": 40,
      "cache_write_tokens": 60
    }
  }
}
```

Both cache buckets are already included in `prompt_tokens`. Adding them again
would inflate context usage. The field names follow the
[OpenAI Chat Completions usage schema](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create).

The shared generator measures retained input positions for the Qwen, GLM, MiMo
and Flint servers, independently of the loaded model architecture. Native
DeepSeek measures its separate retained prefix store for both plain decoding
and PortSpec. Existing OpenAI chat endpoints report the fields in JSON replies
and final SSE usage chunks; servers with `/v1/completions` report them there too.

A write counts an original input-token position successfully saved for prompt
reuse during this logical request. Overlapping checkpoints, replays and
requeues do not count a position twice. Generated tokens, rejected speculative
rows, restored/copied input and refused retention are excluded. Recurrent
models need usable recurrent state as well as valid K/V rows. Pending
asynchronous copies and disk-write queues alone are not completed writes.
Only input coverage confirmed reusable when observed is credited. Later
eviction does not subtract coverage already credited.

These counters measure logical token coverage, not bytes copied, cache capacity
or current occupancy. A prompt's last partial page or recurrent tail may not be
retained, so a cold request need not report every prompt token as a write.
Measured no-write is zero; third-party engines or hooks without measurement
omit the field rather than infer it from prompt size.

The counters describe existing backend behavior. `cache_prompt: false` disables
the native DeepSeek store and yields zero writes there. Some paged servers
clear earlier cache before running that request but can still retain its new
pages; those real writes are reported. This reporting feature does not change
any cache policy, capacity, sampler or model computation.
