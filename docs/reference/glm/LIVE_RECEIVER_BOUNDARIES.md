# Live receiver boundaries

Incoming memory is admitted as one complete layer set before the receiver mutates any cache layer. Archive dimensions, row counts, byte limits and finite values use the same loader as the coordinator. Studio checks cache layouts and offsets, commits cursor metadata only after append and lazy-device evaluation complete, and refuses further commands after failure until a fresh start. GLM stages all packed layers and destination indices before writes, synchronizes GPU writes before acknowledging its filled cursor, and permanently poisons the session identity after any live-step failure. A partial device write cannot be rolled back; the session is invalid and must not be reused.

This is a local mechanical gate, exercised with synthetic NumPy and CPU Torch caches. It does not establish real-model parity, tensor-parallel rank agreement, distributed transaction atomicity, remote request cancellation, source ownership, or quality. The existing GLM one-shot injection path is unchanged. Terminating or draining an invalid serving request remains the controller's responsibility; receiver poisoning prevents later receiver writes and taps, not the server's ordinary decoder from running.

The `uint8` GLM page requirement comes from the existing `_drift_write` adapter and its existing `fp8_ds_mla` fixture contract: 512 packed latent bytes plus 16 scale bytes precede the untouched RoPE slot. Deployment needs the new `live_receiver_glm.py` and `live_publication.py` beside the flat connector, alongside `glm53_handoff.py`, or the installed package equivalents; no deployment or serving restart was performed for this change. Studio imports the package helper `live_receiver_studio.py`.

## Studio command reference

Persistent reader/writer on the Studio (oMLX runtime): the model stays loaded and a session's cache stays alive
while foreign memory is APPENDED to it and its own new entries are tapped. JSON lines on stdin/stdout.

ops: start {reserve}            fresh session; own tokens will live at positions reserve.., foreign memory below that
     extend {ids | text | chat} run own tokens through the model (prefill). If the system prompt contains @@DRIFT@@ and
                                `span` is given, that many positions are reserved THERE for the partner's memory, so the
                                reader's own words frame it (provenance); later appends fill the span upward
     append {memory}            append foreign canonical entries (npz k<l>/v<l>) at the next reserved positions
     generate {tokens}          greedy decode, returns text
     tap {first, out}           write canonical entries of own slots [first, now) to an npz
     quit

## Existing GLM injection semantics

Write foreign memory into GLM-5.3's cache inside the owner's vLLM stack (Qwen -> GLM direction).

The connector entrypoint is deployed next to the owner's `glm53_handoff_connector.py` and selected with
  --kv-transfer-config {"kv_connector": "DriftGlm53Connector", "kv_connector_module_path": "drift_glm53_connector", ...same extra config...}
It subclasses the owner's connector, so every handoff/export behaviour is inherited unchanged. It adds one thing:

  kv_transfer_params {"drift_inject": "<name>", "drift_tokens": N, "drift_own_start": S}   (S >= N, default N)

The request's prompt starts with S placeholder tokens; the model's own prompt begins at position S. After the first
engine step that has computed positions 0..N-1 and none of the own tokens (step end in [N, S]), the MLA latents of
the first N positions (11 sparse-attention layers, fp8_ds_mla pages, replicated on every TP rank) are
overwritten with the entries in <handoff_path>/tp-inject/<name>.npz (`l<layer>` float [M <= N, 512], tiled to N;
canonical latents; GLM-5.3 is NoPE so there is nothing to rotate). The rest of the prompt then attends to them.

Why placeholders are computed rather than skipped: this hybrid stack also keeps recurrent KDA state, pooled
selector keys and a drafter window per request. Letting the model compute placeholder positions leaves all of
those self-consistent; only the latent pages are replaced. On this build the first prefill step of a prompt of
L tokens ends at floor(L/64)*64 - 64 (64-token blocks, one held back for the drafter), so the caller sizes S so that
this boundary falls inside [N, S]; placeholders in [N, S) keep their own latents. Fail closed: if no step ends
inside [N, S], nothing is written and <name>.rank<r>.error says why (the caller must not trust that answer).
