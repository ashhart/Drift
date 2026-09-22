# Running the live exchange (GLM-5.3 on the Sparks <-> Qwen3.8 on the Studio)

> Historical record, preserved from the pre-cleanup documentation.
> Not current operating instructions; see [current status](../STATUS.md).

Everything is driven from the orchestrating machine over ssh. Host aliases come from `DRIFT_SPARK` (head),
`DRIFT_SPARK_PEERS` (other tensor-parallel hosts) and `DRIFT_STUDIO`; no secret passes through these
scripts (the Spark-side scripts read the server's API key inside the host, `scripts/live/spark_run.sh`).

| What | Command |
|---|---|
| GLM reads, Qwen answers | `PYTHONPATH=. .venv/bin/python scripts/live/drift_live.py --cases scripts/live/cases.json --out local/live/demo --controls` |
| Qwen reads, GLM answers | `PYTHONPATH=. .venv/bin/python scripts/live/drift_live_reverse.py --cases scripts/live/cases.json --out local/live/rev --gain-power 1.0 --copies 12 --controls` |
| Preregistered QA, forward / reverse | `scripts/live/qa_eval.py`, `scripts/live/qa_eval_reverse.py` (add `--packs local/live/packs/w2048` to go through pool.v2) |
| Late-arriving memory on a running cache | `scripts/live/drift_probe_qwen.py` (Studio worker: `scripts/live/studio_drift_worker.py`) |
| Refit | taps: `spark_tap_glm.py`, `studio_tap_qwen.py`; direct translators: `fit_stacked_stream.py [--direction qwen2glm]`; packs: `fit_pool2.py` |

Requirements on the hosts: the GLM server must run with `DriftGlm53Connector` for the reverse direction
(`/root/drift-live/restart_drift.sh`; the owner's original settings: `source /root/glm53-split-env.sh &&
./start.sh restart`). The forward direction only needs the owner's stock handoff connector. While taps run,
`scripts/live/spark_janitor.sh` must be running on every non-head rank (livelib starts it), and nothing else should be
generating on the GLM server (exports stall behind its 4 request slots).

Limits of this build: the reader's own prompt must be shorter than 128 tokens for GLM injection (scheduler step
boundary); memories are ~150-token passages; Qwen's model load (~40 s) dominates forward latency unless the drift
worker is used; artifacts under `local/` are tied to these exact checkpoints, quantizations and cache dtypes.
