# Request log and stream keep-alive

The three servers write the same request log and answer a silent stream the same way: both are one shared
implementation, `tools/strix_halo/serve_reqlog.py` (the `Lifecycle` and `StreamStats` classes) and
`tools/strix_halo/serve_ping.py` (`ServePing`), reached through `serve_reqlog.py` / `serve_ping.py`
symlinks in `tools/glm`, `tools/qwen` and `tools/mimo`. `tools/test_serve_logging.py` runs the same
requests against all three and fails if one of them writes an event the others do not, or a field that
differs, so this document describes one format, not three.

Everything here is **off by default**: a deployment that sets neither switch writes no file and not one
extra byte on the wire.

## Switches

| variable | default | meaning |
|---|---|---|
| `KYOJIN_REQLOG_DIR` | **unset = off** | the directory to write in, or `1`/`on`/`yes` for `~/.cache/kyojin`; a path ending in a file name means that file's directory; `0`/`off`/`none`/`no`/`false` also disable it |
| `KYOJIN_REQLOG_KEEP` | `20` | how many runs keep their file; the oldest are removed at start-up |
| `KYOJIN_REQLOG_HEARTBEAT_S` | `10` | seconds between `alive` lines while a job is queued, prefilling or decoding; `0` disables the heartbeat (the log itself stays on) |
| `KYOJIN_SSE_PING_S` | `0` = off | seconds a stream may go without a data event before the server writes one SSE comment line |

**One file per run.** The switch names a directory, and a run writes `requests-<run>.log` in it — the
default directory is `~/.cache/kyojin`, so the file is `~/.cache/kyojin/requests-<run>.log`, and every line
carries `run=<id>`. A path that ends in a file name is read as that file's directory
(`/var/log/kyojin/requests.log` means `/var/log/kyojin`), which is what pointing the switch at a log means;
that file itself is never written to, the run's own `requests-<run>.log` is. One run's lines can
therefore be read on their own, instead of every run of every server piling into one file that has to be
filtered by hand. At start-up the run files of the oldest runs are removed, keeping the newest
`KYOJIN_REQLOG_KEEP` of them (20 by default) — the directory is checked at start-up only, and nothing
while a run is going on. A file this module did not write, such as an archive or the plain file an earlier
layout appended to, is never deleted.

**A long-running server keeps writing.** There is no size limit: while the log is on, a run writes its
events plus one `alive` line per heartbeat interval per request that is queued, prefilling or decoding, so
a server left running for weeks grows its file for weeks. Measurements from this deployment: about 14-21
lines per request, 144 bytes a line, a few MB an hour of normal agent traffic — and much more for long
image prefills, where the heartbeat is what fills the file. Keep the switch on for the runs that need
diagnosing and turn it off for a server that is meant to stay up; if it must stay on, lower
`KYOJIN_REQLOG_KEEP` and read the run you need from its own file.

**Every file this writes is owner-only.** The log (and the picture cache the GLM server keeps on disk)
holds material derived from what clients sent, so the files are created `0600` regardless of the process
umask — a host with the common `0022` would otherwise get `0644` logs. A file that already exists with a
wider mode is tightened on the next start, because `open(..., "a")` does not change the mode of a file it
opens.

The **directory** is made `0700` when this server creates it, and a directory that already exists is left
exactly as it is. One directory is deliberately not given that mode: the shared folder where the servers
keep state for each other — `~/.cache/kyojin`, or wherever `KYOJIN_HEALTH_DIR` points
([env_vars.md](env_vars.md)) — is created with the process umask like any other shared directory on the
host, and the folders above it are created the same way when they are missing. It is not this server's to
restrict: more than one process writes there, and which of them makes it first must not change what it
looks like. Everywhere else the modes above hold, a directory of the server's own *inside* that folder
(`image-embeddings`, for instance) included, and the files inside are `0600` either way, so their contents
are not readable by others on the box.

Each server prints one line per switch at start-up, so the state is discoverable without looking for a
file:

```
serve: request log off (set KYOJIN_REQLOG_DIR=<directory> to enable, numbers and event names only)
serve: request log /home/you/.cache/kyojin/requests-20261016T153012.log (run 20261016T153012, from KYOJIN_REQLOG_DIR, heartbeat 10 s, keeping 20 runs, numbers and event names only, no prompt or reply text)
serve: SSE keep-alive: a comment line after 10 s without output (KYOJIN_SSE_PING_S)
serve: SSE keep-alive off (set KYOJIN_SSE_PING_S=<seconds> to hold a silent stream open)
```

## Format

Every line is `<local ISO timestamp with milliseconds> <event> srv=<glm|qwen|mimo> key=value ...`.
Every line of one request carries the same opaque `req=` id, so `grep <id>` follows one request from
arrival to reply across the queue and the worker. A field whose value is not known is left out rather
than written empty (an unknown value is `None` at the call site and the writer skips it), which is why
the field lists below are what a request *can* report, not what every line of it carries.

## Shared events (all three servers)

In the order a request that goes all the way through produces them:

| event | written when | useful fields |
|---|---|---|
| `req_start` | request arrives | `path`, `method`, `bytes` |
| `req_refused` | 400/404/… answer | `status`, `reason` (the server's own text; a JSON parse error is reduced to line/column) |
| `req_queued` | handed to the worker queue | `queued_ahead`, `prompt_tokens`, `messages`, `images`, `stream` |
| `job_start` | worker picks it up | `wait_s` (queue time), `prompt_tokens`, `images`, `sessions` (GLM/Qwen only), `processing`/`running` (whichever the server keeps) |
| `alive` | every `KYOJIN_REQLOG_HEARTBEAT_S` while tracked | `phase` (`parse`/`queued`/`prefill`/`decode`/`stream`), `elapsed_s`, the in-flight counters, and for a streaming reply **`out_events` / `out_chars` / `last_out_s`** |
| `job_first_token` | first generated token | `elapsed_s` — when the model first produced anything |
| `sse_headers` | streaming: the SSE response is open | `path` |
| `sse_first_content` | streaming: first frame with text | `first_s` since arrival — the gap the client experiences as silence |
| `sse_ping` | a keep-alive comment went to the client | `n` (which comment of this request), `waited_s` — how long the client had been waiting |
| `sse_done` | stream closed | `out_tokens` |
| `sse_summary` | one line when a streaming reply ends (or the client leaves) | `out_events`, `out_chars`, `last_out_s` (since the last *data event*), `comments`, `comment_chars`, `last_comment_s`, `errors`/`error`, `path` |
| `job_done` | job finished | `prompt_tokens`, `cache_n`, `cache_pct`, `new_tokens`, `prefill_s`, `decode_s`, `elapsed_s` |
| `req_done` | response returned | `status`, `elapsed_s`, `path` |
| `client_left` | client hung up first | `path`, `stage` (`handler`/`waiting`/`stream`) |
| `job_skipped` | job cancelled while still queued | `queued_s` |
| `job_error` | the job failed in the worker | `error`, `elapsed_s` |
| `stream_error` | a streaming reply failed mid-stream | `error` |
| `req_error` | the route raised something else | `error`, `elapsed_s`, `path` |

## GLM-only events

GLM is the server with an image cache and a disk tier, so it is the one with lines that cannot exist
elsewhere. They go through the log's raw `log()` call, and no other server needs to match them:

| event | written when | useful fields |
|---|---|---|
| `serve_start` | once, when the server is ready | `model`, `model_id`, `pid`, `ctx`, `cache_tokens`, `sessions`, `num_draft`, `row_budget`, `max_history`, `vision`, `conv_gemm`, `dense_tune`, `moe_fused`, `image_cache_mb`, `image_cache_items`, `image_cache_format`, `image_cache_policy`, `image_cache_dir`, `image_cache_dir_cap_mb`, `image_cache_dir_lock_s`, `heartbeat_s`, `sse_ping_s` |
| `img_cache` | an image request, after `job_done` | `imgs`, `hits`, `misses`, `first_missing`, `entries`, `mb`, `budget_mb`, `item_cap`, `evicted` |
| (inside `img_cache`) | the same request, as the `disk=` report, when the disk tier is on | `env`, `files`, `mb`, `cap_mb`, `locked`, `loaded`, `stored`, `dropped`, `skipped`, `written_bytes`, `written_mb`, `skipped_mb`, `errors`, `stale_tmp` |

`img_cache` is the line for image sessions: `hits` and `misses` are the embeddings the cache already held
and the ones it re-encoded, and **`first_missing` is the index of the first picture that had to be
re-encoded — the paged KV prefix can only match up to it.** A full hit has no `first_missing` field at
all; `evicted` counts entries this call pushed out. See
[tools/glm/SERVE.md](../tools/glm/SERVE.md) for what the cache does and
[env_vars.md](env_vars.md) for its switches.

## Reading it: output accounting

**`out_events` is the number that answers "was the server actually writing?"** A client that reports
silence for ten minutes while `out_events` climbs means the bytes were on the wire and the client did not
show them; a flat `out_events` with a rising `last_out_s` means the server had nothing to send, and the
`phase` then says what it was doing instead (`prefill` = still processing the prompt, `decode` =
generating into something not sent yet, typically a tool call held until its XML closes). A keep-alive
comment does not count as output, so a stream that is only being kept alive still reads as flat.

Measured example of the decode case (a `write_file` call with a 6.8 KB argument, one generation of
98.7 s): nine comments at 17 s … 98 s, while the heartbeat reads the same numbers throughout —
`phase=decode out_events=135 out_chars=29397 last_out_s=2.5 → 12.5 → … → 82.5` — because the 2 144 tokens
of the tool call were held back and the client was sent nothing for 95 s. The closing summary is
`out_events=138 out_chars=40067 last_out_s=0 comments=9`: the reply arrived in three frames at the end.
A client that gave up at 90 s of silence would have been aborted there, with the whole generation lost.

Cost when enabled: one `write`+`flush` per event on the event loop, tens of bytes per line; with the log
off every call is a no-op. For a slow request,
`grep -E 'alive|job_first_token|job_done' ~/.cache/kyojin/requests.log` shows whether it was waiting in
the queue, prefilling a prompt the cache could not serve, or decoding.

## The stream keep-alive (`KYOJIN_SSE_PING_S`)

The request log makes a quiet server visible **to you**; this switch makes it visible **to the client**. A
streaming request is answered with its headers and one empty role chunk, and then nothing at all until the
prompt is prefilled — hundreds of seconds for a large image conversation — so a client that aborts a
stream it has not heard from in two minutes gives up on a server that is working, and (the page cache
being empty for that request) a retry starts the whole prefill again.

With `KYOJIN_SSE_PING_S=<seconds>` set, a stream that has written no data event for that long gets one
SSE **comment** line, which every event-stream client skips:

```
: waiting 105 s

```

It counts from the request's arrival, appears once per interval, and covers **every** silence, not only
the prefill:

* the prefill of a long prompt (the `role` chunk a client gets immediately is not output);
* the decode of a reply that writes a file — the whole `<tool_call>` is held back until its XML closes,
  so a 7 KB `content` argument is ~100 s of generation the client never sees. Measured on this box with a
  `write_file` call: `sse_first_content first_s=1.3`, then nine comments at 16 s … 96 s, while `job_done`
  reported `new_tokens=2195 prefill_s=1.1 decode_s=99.5`. Before the comments covered decode, agent
  clients aborted those turns at ~300 s of silence and every retry paid the full prefill again (the page
  cache of a cancelled request is released).

A stream that is producing text never gets a comment. Default `0` = off: nothing is ever written and a
stream is byte-for-byte what it was. Each comment is also one `sse_ping` line in the request log, so the
two timelines can still be lined up. It does not replace a client-side timeout — it keeps a client that
tolerates comments from giving up, and tells you the server was alive if one does anyway.

## Privacy

The files are owner-only (see above) and nothing derived from the request is written except numbers: token counts, message and image counts,
durations, byte counts, HTTP statuses, an opaque request id, a server tag and event names. Prompt text,
completion text, tool arguments and image paths from the body are never logged. Of the server's own error
text, only the exception type is logged plus a truncated message; JSON parse errors are reduced to
line/column because their message can quote the request body.

## Implementation, in one paragraph

`Lifecycle` (in `tools/strix_halo/serve_reqlog.py`) owns the request table, the context variable that
names the request, the `alive` heartbeat, the route wrapper that stamps `req=` on every line, the summary
line and one method per event above; `StreamStats`/`StreamWriter` count what a streaming reply actually
wrote; `ReqLog` opens the file, formats the lines and skips unknown fields. `ServePing` (in
`serve_ping.py`) owns the comment timing and asks `StreamStats` whether the request has been silent. A
server's own part is **where each phase begins** — its worker picks a job up, its route opens a stream —
and the counters it keeps; an event or a field a server does not have is simply not passed, so nothing
anywhere branches on which server is running.
