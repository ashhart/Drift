# GLM recurrent state at the reserve

Status: deployed on both Sparks and qualified on 22 September 2026 by the [own-cache gate](../../evaluation/GLM_OWN_STATE_GATE.md), 24 of 24 answers with 13 identical to text.

## Why

GLM-5.3-Flash has 45 layers. Eleven are sparse MLA attention, whose per-token latents Drift has always written into the reserve. The other 34 are KDA linear-attention layers. Each keeps one recurrent state per request, 64 heads of 128 by 128 plus a short-convolution window, and never reads per-token rows. Until now those layers saw only the placeholder tokens, so memory reached GLM through a quarter of its layers.

Two further facts made this worse in the live loop:

- The server ran without the Drift scheduler, so GLM computed its whole prompt in one step. Memory landed after that step, and only generated tokens could read it. The question tokens never saw the memory in any layer.
- The earlier scheduler could not have helped. It deferred to vLLM's align-mode split, which only stops at block boundaries. This server's blocks are 3,584 tokens, because a KDA state page is 2.35 MB and the attention page must match it.

Evidence that the state matters on GLM's own side, from four questions only: in reverse run `rev1`, GLM given its own exact latents tiled into 128 positions, under two copies of 75 rows, answered 2 of 4 where text answered 4 of 4. The misses said it had no information. At 4 and 12 copies it answered all 4, so copies masked the gap. The companion run `rev_mech1` is void: its own-latent write failed because the engine step ran past the placeholder boundary.

## What changed

- `glm_prefill_boundary.split` ends a linked request's first prefill chunk exactly at `drift_prefill_boundary` when that position lies inside the chunk's current block. vLLM already allows sub-block chunks that keep a private running state. Before a block boundary, vLLM aligns the chunk capped at the boundary, so every block's state is still materialized. The boundary may now equal the reserve's end and needs no 64-token alignment.
- `glm_prefill_scheduler.DriftScheduler` applies that split. It refuses to start unless the cache is in align mode and async scheduling is on, which dflash speculative decoding keeps enabled.
- A live session may name `drift_state_blob`, the handoff id of an export. The scheduler admits it only when the boundary is exactly the reserve's end.
- At the step that applies the first publication, each rank reads its own export, from `rank{r}.bin` or from the handoff daemon's arena as its ready file says, takes the prompt-end state page of every KDA layer, and copies the bytes into the block holding the request's running state, slot `(position - 1) // block_tokens`. It does this before the publication's receipt, then writes `tp-live-out/<session>/state.rank{r}.json`.

## A state built from per-layer inputs

`drift_state_rows: true`, with `drift_state_blob` naming an export of the prompt's head alone, asks each rank to write that head state and then advance it by the memory. The session folder carries `state.npz` with `h{layer}` rows for all 34 KDA layers, each the 4096-wide input a memory token would give that layer. `glm_state_compute` runs the layer's own input projection, short convolution and `chunk_kda_with_fused_gate` over those rows, on the rank's own heads, and writes the resulting recurrent state and convolution window into the running block through the layer's typed cache views. Any bad input refuses every layer before one changes. The layer modules come from the forward context's registry, so the path needs a real vLLM worker. With GLM's own captured inputs it rebuilt a state that answered all 20 ASCII gate questions, as GLM's own exported state and text did; 10 of 20 outputs were identical to the exported state's, the rest diverging after a few dozen characters, which fits float16 rounding of the captured inputs and different chunking. Results SHA-256, private local copy: `4dcc67b61cde9251725bab688a2f1e08ebe93796bfd73a8651161294baabe3ea`. The first call took 41 s to compile kernels, later calls 0.1 s.

## Capturing layer inputs

With `drift_hidden_capture` in the connector's extra config, a capture request (`drift_capture` with `drift_capture_rows`) also records each KDA layer's input at those rows. Rank 0 writes `drift-capture/<name>.hidden.npz`, since every rank sees the same input. The wrapper sits inside compiled code, so it records only on a server started with `--enforce-eager`; a compiled server fails the capture instead of writing an empty file.

## Limits

- The copy is byte for byte, so each page's registered shape and dtype must match the live cache. Any mismatch refuses the whole write before a byte changes.
- An arena export is read only inside its 120-second lease, from the arena this worker mapped, and its header must name the requested handoff id.
- Any failed live step still stops the engine by design, so a driver must check both ranks' exports before it sends a state.
- The export must be the same model on the same TP split. This is the own-cache path; a translated state from Qwen is separate work.
- The sparse attention's selector keys for reserve rows still come from placeholders. Under 2,048 selected positions every row is read, so short prompts are unaffected. Longer contexts need translated selector keys.
- The dflash drafter's own cache still holds placeholder context for the reserve. Verification keeps outputs exact; only draft acceptance can drop.

## Qualification

The server runs with `--scheduler-cls glm_prefill_scheduler.DriftScheduler` from the Drift env, and both ranks carry the bundle pinned at `5a11cc1`. `scripts/live/glm_state_gate.py` is the gate; the result is in the [own-cache gate](../../evaluation/GLM_OWN_STATE_GATE.md).
