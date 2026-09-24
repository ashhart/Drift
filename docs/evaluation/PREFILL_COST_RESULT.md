# Receiver prefill cost: result

Run on 22 September 2026 against [the preregistration](../../configs/preregistration.prefill-cost.json) fixed at commit `8972655`, SHA-256 `90eb7dbf3dce28c4851643ff3fb64a99a425db20d6ba3a0da0f0a6ae9b391cb2`. Outcome: **SUPPORTED**.

## What was compared

Qwen on the Studio answered the 240 questions of Test A, over the same 120 passages with the same frozen artifacts. GLM on the Sparks read each passage, and its exported cache reached the Studio by RDMA through the handoff daemons. Time to first token was measured inside the Studio worker around the single prefill step:

- `drift_v4`: Qwen prefills only the question, over a cache already holding GLM's translated memory.
- `text_in_prompt`: Qwen prefills the passage and the question together from an empty cache.

## Result

| Measure | drift_v4 | text_in_prompt |
| --- | --- | --- |
| Median time to first token | 0.0720 s | 0.3304 s |
| Mean time to first token | 0.0720 s | 0.3356 s |
| Exact match | 0.9708, 233 of 240 | 0.9833, 236 of 240 |

- P1, the receiver's prefill is cheaper: memory was faster on all 240 questions, clustered sign-flip p = 7.5e-37 over 120 passages. Holds.
- P2, the saving is material: text takes 4.59 times as long as memory, against a threshold of 2.0. Holds.

The preregistration voids a latency win that comes with an accuracy change. Text in prompt reproduced the earlier MCDMA run exactly, 0.9833. Memory scored 0.9708, one question below that run's 0.975 and two below the original run's 0.9792. That matches the one-question spread between the two earlier runs, so the comparison stands.

## What the sender paid, separately

Median 13.2 ms to translate both memory versions on the Studio, and 18.6 ms for the RDMA transfer. GLM's own export and its reading of the passage are not in these numbers.

## How to read it

The receiver skipped prefilling the shared passage because the passage arrived already computed. MCDMA carried the memory; it did not make prefill faster, and this result is not a transport speed claim. The saving is real only when the sender would have read the context anyway, or when more than one receiver uses the same memory.

The passages are short and synthetic, about 200 tokens each. The saving should grow with context length, and this run does not measure that curve. It also says nothing about the reverse direction, where a vLLM receiver does prefill a reserved placeholder span.

## Evidence

Private local copies, SHA-256:

- rows `73b12bb2f81c23ead21c7cc0aed67709d09caef3ced3e57228a71e604fcc2e9b`
- report `9448d9d19802d3102d6e6f746c099abae760ba7a49c5f9f14e00e95fb7db4c71`
- score `74c31f637c2fca3330fd2c1aaff23bfe8d77757b8ef7bb01d3fe781878d5dbbd`

The run took 3,846 seconds from 20:14:41Z, at commit `9fc878f`, on GLM restarted through its Drift unit and with a Studio handoff daemon kept attached to its SSH session.
