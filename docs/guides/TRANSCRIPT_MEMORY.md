# Transcript-backed memory

An API usually gives you messages, not its internal cache. This optional path lets
your local GLM read an explicitly supplied conversation, exports GLM's latent
rows, transfers those rows over MCDMA and lets local Qwen answer from translated
memory. It does not reconstruct the API model's KV cache, hidden reasoning or
unobserved context. No distillation or backbone training happens.

The import, schema, export parser, transfer validation and capture-to-reader
composition have local fixture tests. A native smoke check completed GLM capture,
hash-verified MCDMA receive and Qwen recall, with full-text, no-memory and
different-memory controls on three questions from one synthetic conversation.
This is not a general recall qualification or a measured advantage over text.
An execution receipt marked `PASSED` is not a recall or attribution score.

## Boundaries

The source receives text on an explicitly declared ingestion path. Only the
stripped `DRIFTS01` latent frame goes through the activation transport. The raw
GLM export includes prompt token IDs and must stay on the source host.
The receiver command reads memory plus its own question, never the transcript
or the source provenance receipt. There is no automatic text fallback.

Each capture is a fresh, complete snapshot. Repeated captures recompute the
whole supplied transcript and create distinct files; this is not incremental
cache replay, a continuously coupled session, or a replacement for the native
OMP coordinator. Context overflow fails rather than silently trimming messages.

Recorded provider and model labels are caller-supplied provenance, not proof
of authorship. They are encoded as part of the quoted conversation for GLM to
read, but correct downstream source attribution still needs measurement.
The input can contain prompt injection; JSON quoting is not a security boundary.
Neither command executes tool calls from the transcript.

## Import messages

Use the final text-only `messages` array maintained by your API client, including
the assistant's completed output and any tool results. The importer understands
`role`, `content`, `tool_calls` with `type: "function"`, and `tool_call_id`.
It rejects unknown fields, multimodal parts, duplicate IDs and mismatched tool
results rather than guessing their meaning. Streaming chunks, hidden reasoning,
provider-specific blocks and tool-definition schemas need explicit conversion
and are not supported by this importer.

Create a private directory outside repositories and model-tool sandboxes, then
use absolute paths throughout:

```bash
drift transcript import \
  --input /private/memory/chat-messages.json \
  --provider example --model api-model --conversation-id conversation-1 \
  --output /private/memory/transcript.json

drift transcript validate --input /private/memory/transcript.json
```

Outputs require an existing owner-only directory, refuse overwrites and have
mode `0400`. The input limit is 1 MiB and 512 messages. Validation loads no model.
For mixed API sources, use `drift.transcript.v1` directly:

```json
{
  "schema": "drift.transcript.v1",
  "conversation_id": "conversation-1",
  "sources": {"api-a": {"provider": "example", "model": "model-a"}},
  "messages": [
    {"id": "m1", "source": "api-a", "role": "user", "content": "Delivery arrives at 10:45."},
    {"id": "m2", "source": "api-a", "role": "assistant", "content": "Recorded."}
  ]
}
```

The source map can contain multiple providers/models and every message names
its source. Tool records use `tool_call_id`; assistant records may contain
`tool_calls` with `id`, `name` and a JSON-object string in `arguments`.
SDK users can call `from_chat_messages` from `drift.transcript.importer` to
create the same validated snapshot without intercepting traffic or API keys.

## Capture on the GLM host

Prerequisites are the existing qualified GLM connector, `fp8_ds_mla`, the exact
local tokenizer, and an owned export root already configured for that server.
The root must be canonical and not writable by group or others; a shared `0755`
parent is allowed, while each capture request directory is created as `0700`.
The server alias and tokenizer must match the deployed checkpoint; the command
checks the response alias, token count and exported prompt IDs, but does not
independently attest the server's checkpoint weights. Install no new runtime
over a working serving environment; use its approved `tokenizers` package.

```bash
drift transcript capture \
  --input /private/memory/transcript.json \
  --tokenizer /models/glm/tokenizer.json \
  --tokenizer-sha256 "$TOKENIZER_SHA256" --model "$GLM_MODEL" \
  --export-root /memory/exports \
  --output /memory/exports/private/transcript.memory \
  --max-tokens 4096 --timeout 60
```

Only the owned loopback `/v1/completions` endpoint is allowed; redirects and
environment HTTP proxies are refused. `DRIFT_GLM_KEY` is read only from the
environment and is not included in receipts. Supply `--arena /memory/arena.bin`
when the connector writes arena exports; inode and byte-range checks apply.
The command requests one generated token solely to finish export, counts it,
and exports only the verified prefill rows. No remote deployment or restart
is attempted, and a missing connector fails at the bounded export wait.
Create the output's `private` parent beforehand with mode `0700`; relaxing the
shared export parent's permissions does not relax output privacy.

The source receipt is `transcript.memory.receipt.json`; it binds transcript and
rendering hashes, message provenance, tokenizer hash, token counts and the
latent payload hash. Keep this receipt private, as identifiers and hashes can
reveal information even without plaintext messages. Raw exports are retained
for operator inspection, including on failure; arrange a private retention
policy on **all** tensor-parallel ranks before repeated use.

## Receive on the Qwen host

Use the source receipt's activation digest and exact byte count through your
trusted operator control path. The source file must be inside the existing
`handoffd` export allowlist, and the daemon must have exclusive access to its
destination shared-memory slot during this operation.

```bash
drift transcript receive \
  --peer "$MCDMA_PEER" --remote /memory/exports/private/transcript.memory \
  --sha256 "$MEMORY_SHA256" --size "$MEMORY_BYTES" \
  --socket-path /tmp/handoffd.sock \
  --output /private/memory/received.memory --timeout 60
```

This calls the existing MCDMA daemon, opens no verbs objects, copies the received
view into owned bytes and checks the digest and complete tensor layout before
publishing the local artifact. It neither deletes the source file nor starts,
stops or reconfigures a daemon. The daemon's own registration and transfer-size
bounds remain prerequisites because its PULL interface has no expected-size
argument. Source receipts and API text must not be substituted for this file.
An unavailable control connection or socket timeout returns a redacted
`FAILED / TRANSCRIPT_TRANSPORT_UNAVAILABLE` report without publishing a received
artifact or loading Qwen; it never falls back to SSH or text transport.

### Keep the control session alive

On the tested macOS host, launching the daemon with `nohup` and immediately
closing SSH left its TCP control connection unusable. A minimal TCP-only probe
also failed with `EHOSTUNREACH` after detachment, while the same executable
succeeded in an active SSH session. Spark-side registration worked independently;
changing the translator, shared-memory size or registration flags was not needed.

Keep the SSH control session open while using a session-launched daemon:

```bash
nohup /path/to/handoffd studio /tmp/handoffd.sock \
  --shm-bytes 1073741824 "$PEER_SPEC" \
  > /private/handoff/daemon.log 2>&1 < /dev/null &
wait
```

This is a launch example for an operator-approved, unused socket and peer, not
a command to run alongside an existing owner: each Spark accepts one control
session at a time. Stop the daemon only with its `SHUTDOWN` socket command;
never kill a Studio verbs owner. `nohup` protects against session hangup but
does not guarantee continued network access after the session disappears.
For unattended service deployment, qualify the service's network permissions
and lifecycle separately, including a transfer after the launching session exits;
see [Apple's local-network privacy guidance](https://developer.apple.com/documentation/technotes/tn3179-understanding-local-network-privacy).
Do not disable privacy protections or assume that successful ping proves the
daemon itself can connect.

## Answer on the Qwen host

Run inside the approved oMLX Python environment with the installed Drift CLI,
the supported Qwen checkpoint, a hash-pinned v4 forward recipe and its matching
selector-key translator. Recipe fields follow the existing forward recipe
manifest: `forward_gain_power`, `artifacts` and `sha256` entries for
`forward_base`, `forward_fanout` and `correction`.

```bash
drift transcript answer \
  --memory /private/memory/received.memory --sha256 "$MEMORY_SHA256" \
  --question /private/memory/question.txt --output /private/memory/answer.json \
  --checkpoint /models/qwen --manifest /translators/forward.json \
  --manifest-sha256 "$RECIPE_SHA256" \
  --index /translators/index3.npz --index-sha256 "$INDEX_SHA256" \
  --need-gb 160 --max-rows 8192 --max-new 64 --timeout 120
```

The Qwen loader uses the existing Studio lock and memory guard. It validates
translation and capacity before appending memory to a fresh cache, then decodes
only from that memory and the question. The answer stays in the private output
file; stdout returns accounting without the answer text. `--need-gb` must reflect
your measured runtime requirement, not a smaller value chosen to bypass the guard.

Capture and answer CLI operations run in bounded child processes; their timeout
includes startup and model loading. Expiry reaps only that child, never a shared
daemon. A timed-out GLM HTTP operation does **not** prove the server released its
request, and the failure receipt labels server cleanup `UNVERIFIED`. Direct
low-level Python functions have cooperative checks; use the CLI for the outer
process deadline. Imported memories can encode sensitive content even though
their files contain no text fields; do not upload them with model translators.

## Native smoke evidence and remaining qualification

The native check used two fresh GLM snapshots of synthetic conversations with
different facts, frozen translators and a fresh Qwen cache for every answer.
Both 130-row payloads crossed MCDMA with matching SHA-256 digests; no transcript
text or token IDs crossed that activation path. Only the explicitly separate
full-text control supplied the transcript to Qwen.

| Input to Qwen | Target facts recalled |
| --- | --- |
| Translated memory of the target conversation | 3/3 |
| Complete target transcript as text | 3/3 |
| No memory | 0/3 |
| Memory of the other conversation | 0/3; returned all three alternative facts |

One of the three questions had already been used for the CLI pilot, so this is
a smoke check, not a preregistered held-out evaluation. Native CLI `answer`
also completed independently; execution receipts intentionally keep
`native_recall: BLOCKED` because an unscored invocation cannot certify recall.

Measured capture times were 0.746 and 0.718 seconds, each including one generated
export-completion token. Verified receive took 6.33 and 5.52 milliseconds;
the RDMA read loops were about 0.62 milliseconds for each 2.93 MB payload.
The CLI reader's model-loading-plus-answer operation took 9.42 seconds, including
31.4 milliseconds for translation and append; the separate twelve-answer control
process took 14.59 seconds including loading. These are individual smoke-run
observations, not steady-state latency distributions or text-versus-memory speedups.
Failed transport attempts, setup and operational diagnosis are not included in
those component timings; money and energy were not measured.

Before enabling this on a workflow, compare shadow-memory answers against the
same complete transcript given directly to Qwen, plus no-memory and wrong-memory
controls on held-out questions. Measure source attribution separately from fact
recall and include preprocessing, GLM prefill, the extra export token, transport,
translation, Qwen prompt/decode, failed attempts and repeated snapshots in costs.
Keep answers and scorers outside both model workers' access. No quality or latency
advantage is established by the local fixture tests.
