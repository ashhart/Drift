# Why Qwen misses GLM's recommendation in the loop

Status: exploratory diagnostic, 22 September 2026. Not preregistered, and no claim rests on it.

## The failure

In the appointment loop, Qwen's follow-up names the shop GLM recommended in only 6 of 24 linked scenarios of the reverse confirmation, and 1 or 2 of 6 in the selection ablation. The forward taps are not the problem: in every linked ablation run they covered GLM's whole output, the recommendation line included, and all of them reached Qwen's cache.

## The test

For the 22 ablation runs where GLM named a shop, GLM's listing and output became a passage, and its recommendation became the answer. The Test A worker asked each passage two ways: the loop's exact follow-up wording, and a plain "Which coffee shop was recommended?". It answered in every arm the worker has, over MCDMA and RDMA with the same frozen artifacts.

| Arm | Loop wording | Plain wording |
| --- | --- | --- |
| Text in the prompt | 22 of 22 | 22 of 22 |
| Qwen's own cache of the text | 10 | 16 |
| GLM's memory, v4 translator | 8 | 12 |
| GLM's memory, v3 translator | 8 | 5 |
| No memory | 0 | 0 |
| Another passage's memory | 0 | 0 |

## What it shows

- The question is answerable from the content: text scores 22 of 22.
- Translation is not the main loss. Qwen with its own untranslated cache of the same text still misses 6 to 12 of 22.
- The loop's decode-time path is not the main loss either, since memory prefilled the Test A way fails too.
- The loop's wording, which offers "unknown", lowers every memory arm.

Memory enters only the full-attention layers. The recurrent layers, which carry much of a sentence's structure, start fresh. A single lookup survives that: which clinic, which code, which variable. A relation among several similar items does not: which of four described shops was the recommended one. The reverse confirmation's misses look like the same limit on GLM's side. GLM recalled the right clinic in 22 of 24 but bound the wrong street in most misses.

## What follows

The fix has to stay inside direct whole-context reading: the translated rows must carry the structure that the receiver's own cache would, so translator training should target how the receiver's attention reads memory rather than how closely the rows match.

Rows SHA-256, private local copy: `36b7c3475acdc5645ece2d9f4ecb5d8a1e3e346887637bc8425a823c6331ab0c`.
