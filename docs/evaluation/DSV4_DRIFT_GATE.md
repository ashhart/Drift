# DeepSeek V4 Drift gate

Status: engineering gate, PASSED on 23 September 2026. The criteria were fixed in `scripts/live/dsv4_drift_qualify.py` before the first run.

## Question

Does the raw-row connector write DeepSeek V4's own compressed cache rows exactly, and does DeepSeek answer from them as it does from the text?

## Setup

- DeepSeek-V4-Flash on both Sparks, started by the `dspark-vllm` unit with `COMPOSE_FILE` pointing at the file `scripts/deploy_dsv4/compose_drift.py` derives. That file adds the connector's modules and one `--kv-transfer-config` argument, and turns off expandable CUDA segments. vLLM refuses a KV connector while they are on.
- 12 validation passages and 24 questions, the same as the GLM and Qwen own-cache gates.
- Layout per passage: an opening padded to 128 tokens, the passage padded to a multiple of 128, newline filler up to position 8,192, then the question. The question is far outside the 128-token sliding window, so only the compressed caches can carry the passage.
- Arms: native, the passage in the span; inject, placeholders in the span, overwritten with rank 0's tapped rows; no_memory, placeholders only.

## Result

| Arm | Exact match |
| --- | --- |
| native | 0.958 |
| inject | 1.000 |
| no_memory | 0.042 |

- WRITE PASSED: after injection, both ranks tapped rows identical to the native rows in all 62 tensors, the compressed MLA caches and the compressed indexer caches at ratios 4 and 128.
- RECALL PASSED: inject at least 0.8 of native, and no_memory at most 0.2.

DeepSeek reading its own compressed rows answered every question, one more than native text.

## What it cannot show

This is DeepSeek's own cache. No translator between DeepSeek and Qwen or GLM exists yet. The compressed rows are one latent per 4 or 128 tokens, so those translators will work at group level.

Record SHA-256, private local copy: `d5f6084a739941a80f4c1456c7f914bacf984020655283d27c5fe7051c257c5f`. Run time 7 min 14 s.
