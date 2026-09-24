# OMP Drift

This plugin controls the local reference worker service inside OMP through typed commands and a status board. `/drift start --reference` explicitly selects that demonstration; a separately configured KV-only subagent entry point is described below, with native qualification still BLOCKED.

Duo remains the separate text-based comparison workflow. The model runtime owns private caches and translated memory; the separate native exchange path uses MCDMA, but this command entry point does not launch it. The remaining integration work is recorded in `docs/reference/omp/OMP_LIVE_INTEGRATION.md` in the Drift repository.

## Promptless subagents

`/drift subagent enable`, `status` and `disable` are separate from `/duo` and its text-based room. They require owner-pinned workers, boundary and exchange profiles, a running coordinator and fresh mailbox consumers; selecting a model is not enough.

The first prerequisite is implemented in the experimental worker provider: an owner-pinned `communication_mode: "kv_only"` with `memory_mode: "linked"`, `experimental_multi_turn: true` and `session_binding: "fixed"`. Both workers must select that mode. The provider requires pinned exchange and parking configurations, and the parking configuration must use version 2. Existing `text_and_artifacts` runs remain a separate, explicitly text-enabled mode.

In this restricted mode, each worker receives one own setup prompt and an immutable system prompt. Its only tool is `drift_sync {}`, which returns the fixed control receipt `ready` after the exchange gate releases it. The provider rejects peer developer messages, additional user prompts, arbitrary tool results, tool arguments and other tools before they reach the worker or tool dispatcher. The parent waits for the child even at the first sync; every sync requires a fresh exchange, and both workers must settle before completion. Failure poisons the route instead of falling back to text.

The restricted wrapper admits one empty-argument `task`, repeated `drift_sync` calls and an empty-argument child `yield`; it delegates only fixed owner controls and discards returned text. Paired receipt checking now releases epochs automatically, and completed routes block OMP's asynchronous text follow-up before it reaches the provider. The installed-OMP synthetic command check passes, not native GPUs/RDMA qualification. Follow the [owner setup guide](../../docs/guides/DRIFT_SUBAGENTS.md); stock Hub traffic, arbitrary file tools and nested subagents remain unsupported.

"Promptless" describes communication between models. It does not mean that local setup, tool receipts or the user's original task contain no text, and a `kv_only` flag alone does not attest to the coordinator's transport or operating-system isolation.

The service is a separate Python process, started from the Drift project (`python -m drift.runtime.service`, documented there). It speaks the protocol in `docs/reference/service/SERVICE_PROTOCOL.md`: newline-delimited JSON over TCP on `127.0.0.1`, every line carrying an `auth` field that is the hex HMAC-SHA256 of the message's canonical JSON (keys sorted at every level, no whitespace, ASCII-only strings) without `auth`. Unauthenticated or malformed lines close the connection in both directions.

## Environment

The plugin reads two variables when you run `/drift start`:

~~~text
DRIFT_SERVICE_PORT     TCP port the service listens on (host is always 127.0.0.1)
DRIFT_SERVICE_SECRET   per-session HMAC secret shared with the service out of band
~~~

Neither value is written to the session file, echoed in chat, shown on the board or included in an error message. If either is missing, `/drift start` reports which one and does not connect.

## Commands

~~~text
/drift assign <member-index> <text>            stage an assignment (before start only)
/drift start --reference <manifest-path> [--task <text>] [--checkpoint-every <n>]
/drift tick [n]                                run n epochs (default 1)
/drift pause                                   stop ticking; state kept
/drift status                                  refresh the board from the service
/drift checkpoint                              write a checkpoint; its sha256 is announced
/drift mail <member-index> [slots]             the member writes mail from its own tokens
/drift complete                                end the run; outputs go to the audit store
/drift abort                                   poison the run
/drift stop                                    close the socket; ABORT first if still STRICT
~~~

`start --reference` sends SETUP (manifest path, task text, staged assignments) and then START (`checkpoint_every`, default 10), which moves the run into STRICT at epoch 0. Assignments are keyed by member enum index, the same index `mail` uses as `sender`.

Phases go `SETUP → STRICT → CLOSED`. Free text is accepted only before STRICT: once the run is STRICT, the plugin refuses `assign` and `start` itself, and every other command accepts integers only (`tick three` is refused before anything is sent). `tick`, `pause`, `checkpoint`, `mail` and `complete` need a connected STRICT run. `abort` works in any phase. `stop` closes the connection and clears the board; if the run is still STRICT it sends ABORT first.

One request is in flight at a time, with a 60 second deadline. A timeout, a malformed line or a response that fails authentication closes the socket, shows an error notification, and leaves the last known picture on the board marked disconnected. The plugin never reconnects on its own.

## What you see

- A board widget above the editor with the phase, epoch and manifest digest, then for each member its epoch, sequence, source position, writer, local and foreign token counts, mail sent and mail-foreign tokens, gate state per layer and mean injected mass per layer; the mailbox counts; and the detector flags (non-finite, norm drift, mass oscillation).
- A status line `drift: <phase> e<epoch>`.
- `[Drift] …` chat lines (custom type `drift-room`) on start, phase changes, checkpoints, poison, completion, abort, disconnect and stop.
- At every model turn, one system-prompt line identifying reference mode, blocked live integration, phase and epoch. Nothing else from the run reaches the model.

## What it will never display

Only fields of the typed status object are rendered, and every value passes through a formatter that accepts numbers, booleans, enum names (`STRICT`, `open`, `pending`, …) and hex digests. Anything else renders as `?`, and a status object that is off schema is treated as a malformed line and drops the connection. In particular the plugin never shows or forwards:

- the task text, assignments or manifest path after SETUP (the manifest is identified by its sha256 only);
- the audit path, final outputs, probe estimates, replay results or scores;
- token ids, mail text or any decoded text;
- the service port or the HMAC secret.

## Persistence

On start, complete and abort the plugin appends a `drift-run-state` entry to the session file with the manifest sha256, phase and epoch. A resumed session shows that last known phase on the board and status line (marked resumed) and in a chat line, but does not reconnect; run `/drift stop` to clear it or `/drift start` to begin a new run. Shutting down OMP closes the socket without aborting; the service keeps the run's state.

## Install

Link the plugin, then restart OMP:

~~~bash
omp plugin link /absolute/path/to/omp-drift
~~~

Run `bun test` for the regression suite, which includes an in-test fake service that validates HMAC and answers every op, and `bun run typecheck` for strict TypeScript (`bun install` once for the TypeScript and Node type dev dependencies; the host import is typed by the shim in `types/`).

## Canonical JSON note

The plugin's canonical form matches Python's `json.dumps(obj, sort_keys=True, separators=(",", ":"))` (default `ensure_ascii=True`) for integers, booleans, null, strings and nested containers. Floats differ in edge cases (`1.0` vs `1`, `1e-07` vs `1e-7`), so the service should emit floats in their shortest round-trip form without a trailing `.0`, or both sides will disagree on a status object's digest.
