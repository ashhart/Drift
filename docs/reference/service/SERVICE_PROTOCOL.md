# Drift worker service protocol v1

The service is a local process that owns the models, private states, translators,
banks and the hive scheduler (spec D15, M2.1, M5.1). The OMP plugin `omp-drift`
owns room lifecycle and presentation only. During strict trials the plugin can send
and receive **typed lifecycle/status events only**: no task text, no token ids, no
probe output, no scorer output.

## Transport

Newline-delimited JSON over TCP on `127.0.0.1` (loopback only). Every request and
response is one JSON object per line. Every message carries `"auth"`: hex HMAC-SHA256
over the canonical JSON of the message without the `auth` field, keyed with a
per-session secret the owner gives to both sides out of band. Unauthenticated or
malformed lines close the connection, and so does a line longer than 1 MiB or one with no
final newline, before any authentication; a line whose signature cannot be computed, such as
one holding an infinite number, counts as unauthenticated and leaves the run untouched. A
response value that is NaN or infinite is sent as `null`, so the status's `nonfinite` flag
reports a diverged run. Numbers are signed in JavaScript's shortest form, so a whole number
of 1e16 or more signs as its shortest digits followed by zeros, as JavaScript prints it.

## Phases

`SETUP` → `STRICT` → `CLOSED`. Free text (the task, the split, the run manifest path)
is accepted only in `SETUP`. Entering `STRICT` pins the manifest hash; from then on
only the typed ops below are accepted, and the service refuses any string field
longer than 0 characters except fixed enum names and hex digests.

## Requests (`op` is a fixed integer enum; `id` is a client-chosen request id)

| op | name | phase | fields | effect |
|---:|---|---|---|---|
| 1 | SETUP | SETUP | `manifest` (path), `task` (text), `assignments` (member → text) | load manifest, build members, prefill each with its assignment; returns `manifest_sha256`, `members` |
| 2 | START | SETUP | `checkpoint_every` (int epochs) | enter STRICT at epoch 0 |
| 3 | TICK | STRICT | `epochs` (int ≥ 1) | run that many epochs; each member decodes `decode_tokens_per_epoch` local tokens per epoch |
| 4 | STATUS | any | — | typed status (below) |
| 5 | PAUSE | STRICT | — | stop ticking; state kept |
| 6 | CHECKPOINT | STRICT | — | write a checkpoint; returns its sha256 |
| 7 | ABORT | any | — | poison the run; no completion |
| 8 | COMPLETE | STRICT | — | end the run; final outputs are written to the **audit path** only, never returned |
| 9 | MAIL | STRICT | `sender` (member enum index), `slots` (int) | the sender writes mail from its **own most recent local tokens** (bounded by `slots`); the plugin never supplies mail text |

## Status object (all typed; no strings except enum names and hex)

```json
{"phase": "STRICT", "epoch": 12, "poisoned": false,
 "members": [{"index": 0, "writer": 1, "epoch": 11, "sequence": 12, "source_position": 40,
              "local_tokens": 40, "mail_sent": 1, "foreign_tokens": 18, "mail_foreign_tokens": 3,
              "gate": {"3": "open", "7": "open"}, "mass_mean": {"3": 0.12}}],
 "mailbox": {"counts": {"pending": 0, "visible": 1, "attended_not_proven": 0, "causally_incorporated": 0,
             "expired_without_incorporation": 0, "rejected": 0, "cancelled": 0}},
 "detectors": {"nonfinite": false, "norm_drift": {"3": 0.01}, "mass_oscillation": {"3": 0.2}},
 "manifest_sha256": "…", "last_checkpoint_sha256": "…"}
```

## Responses

`{"id": <request id>, "ok": true, "result": {...}}` or `{"id": …, "ok": false, "error": "<enum name>"}`.
Error names: `AUTH`, `PHASE`, `MALFORMED`, `POISONED`, `BUDGET`, `UNKNOWN_OP`, `INTERNAL`.

## What the plugin may show

Only fields from the status object. Audit artifacts (final outputs, probe estimates,
replay results, scores) live under the audit path, which is outside every worker's
readable tree and is never sent over this protocol.
