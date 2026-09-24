# Drift connectors on the Sparks

Each model the Sparks serve can start with a Drift KV connector loaded. A connector lets a request receive memory into a reserved span of its own prompt, and read a span back out. Stock requests pass through untouched.

| Model | Connector | Status |
| --- | --- | --- |
| GLM-5.3-Flash | `DriftGlm53Connector` in `drift/serving/vllm_glm53_inject.py` | Live. The unit starts it through a drop-in that sources the Drift env. |
| Qwen3.8-Flash-Next | `DriftQwen38Connector` in `drift/serving/vllm_qwen38_drift.py` | Unit-tested. The live gate is in `scripts/live/qwen38_drift_qualify.py`. |
| DeepSeek V4 | `DriftRawRowConnector` in `drift/serving/vllm_raw_rows.py` | Live. The gate in `scripts/live/dsv4_drift_qualify.py` passed WRITE and RECALL on 23 September 2026; the unit starts it through a drop-in that points `COMPOSE_FILE` at the derived Drift compose file. |

Unit tests show the logic against stand-ins for vLLM. Only a passing live gate shows a connector works on the real server.

## Requests

A request opts in through `kv_transfer_params`:

- `drift_span_start` and `drift_tokens` name the span [start, start + tokens) in prompt positions.
- `drift_inject: name` writes memory from `<path>/drift-inject/<name>.npz` over the span's placeholders, after the engine step that computes them.
- `drift_own_start` is where the request's own tokens begin. The step that finishes the span must not reach past it, so own tokens never attend to placeholder rows. Put it at the end of the engine's first prefill step: 8192 tokens for both recipes when the server is otherwise idle.
- `drift_tap: name` reads the span back once it is computed. The raw-row connector supports it. Qwen reads spans through the owner's `qwen38_handoff` export instead.

An injecting request needs a `cache_salt` the server has not seen. Neither vLLM build has a flag to keep a request's blocks out of the prefix cache, so a fresh salt is what stops injected memory serving another request's prefix.

Spans align to the model's largest cache grouping: 4 tokens for Qwen's selector groups and 128 for DeepSeek's compressed caches.

## What the client checks

Every rank writes a marker file per request:

- `<name>.rank<r>.done` or `.error` for Qwen, under `drift-inject/`.
- `<name>.<kind>.rank<r>.done` or `.error` for the raw-row connector, under `drift-marks/`.
- `<name>.preempted` means the request lost its blocks after the write. Discard that answer.

A request that fails validation still runs as a normal request, and the error file says why no memory was written. A client must check the markers of every rank before trusting an answer.

GLM's live receiver is the exception: two limits stop its engine instead of the request. A reserve longer than 4,096 rows fails the scheduler's boundary check (`glm_prefill_boundary.py`). A publication larger than 128 MiB (`live_publication.MAX_BYTES`) is refused inside the worker's save step, which poisons the session. Rows are published as float32, 11 layers of 512 values, so the reserve limit binds first. A per-token translated state, 34 layers of 4,096 values, fits up to about 480 tokens as float16. `glm_state_gate.py` skips an arm that would break either limit and records why. Any other client must check both before it sends a request.

## What each connector carries

- Qwen: full-attention K and V for each rank's head, plus the compressed selector keys when the client sends them. The Gated DeltaNet and PLE layers keep the state the placeholders gave them.
- DeepSeek V4: the compressed MLA entries and compressed indexer keys, as raw bytes in the cache's own format. Sliding-window and compressor-state caches are never touched, so they see placeholders.
- Raw bytes make an identity round trip exact. Memory from another model must be packed into DeepSeek's page format first: `drift/translate/dsv4_member.py` does it for the shared space's DeepSeek decoder, over a tapped placeholder span, which keeps its positional values ([six directions](../evaluation/SIX_DIRECTIONS.md)).

## Starting a model with its connector

- GLM: `systemctl start glm53-vllm`. The drop-in sources `/root/drift-live/glm53-drift-env.sh`.
- Qwen: run `scripts/deploy_qwen38/start_drift.py <recipe>` to derive `start-drift.sh`, then source `qwen38-drift-env.sh` and run `start-drift.sh --launch` from the recipe directory. The recipe's TP=2 launch ignores `EXTRA_DOCKER_ARGS`, so the derived script overlays the connector through the recipe's own `add_overlay`.
- DeepSeek V4: run `scripts/deploy_dsv4/compose_drift.py <recipe>` to derive `docker-compose.dspark-drift.yml`, and start the recipe with `COMPOSE_FILE` pointing at it. The launcher copies that file to the worker. Each Spark needs the connector modules in `/root/drift-live/dsv4-connector`.

Both derivations refuse when the recipe's anchors move, and neither edits the recipe checkout. Regenerate them after a recipe update. Unit drop-ins for Qwen and DeepSeek follow only after each passes its live gate.
