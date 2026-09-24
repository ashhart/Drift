# Payoff at long contexts: one reader, five joiners

Status: measured 24 September 2026, one run per context. Engineering evidence of cost, not a quality claim: the needles
check that an attached cache is read at all.

## Question

When five fresh models join work over a long shared context, what does attaching a cache save against each reading the
context as text, at 32k, 64k and 128k tokens? For a joiner of the same model, and for Qwen joining from GLM's cache.

## Setup

- Contexts (`scripts/live/payoff_contexts.py`): whole Python standard-library modules, in path order under their
  headers, cut at character targets chosen so Qwen reads 32,919, 64,426 and 131,290 tokens. Each carries five needle
  questions, which function fails with a given message, from modules at evenly spaced depths.
- Qwen3.8-Flash on the Studio, every way with the full-attention budget raised to 262,144, so all of them attend the
  whole context densely. Prefill runs in chunks of 2,048 tokens, as a server does (`scripts/live/studio_payoff.py`).
- Ways, each joiner answering one needle:
  - none: head and question only;
  - text: head, context and question;
  - same: a resident Qwen reads the context once and exports its full-attention rows and recurrent state; each joiner
    prefills its head, attaches the rows, takes the state and prefills its question;
  - glm: GLM-5.3 reads the context on the Sparks (`spark_dropin_export.py`), the export is pulled over MCDMA, translated
    once by the final contextual reader and state translator, and each joiner attaches it as above.
- Time to first token is measured from an empty cache for every joiner, so the attached ways include the joiner's head,
  attaching the rows and the state. One-time costs are listed apart.
- GLM's handoff connector computes a full read in one engine step. At 121,548 GLM tokens that step never finished in
  two tries, stopped after 30 and 20 minutes, while 60k took 38 s. The 128k context was read in two requests: a full
  export of the first 60,928 tokens, then the whole prompt from the prefix cache with a delta export, and the two were
  stitched (`--split-at`, `drift/serving/glm53_delta.py`). The delta request reused exactly the 60,928 cached tokens.

## Results

Per joiner, the median of five:

| Context (Qwen tokens) | Text | Same model | GLM to Qwen | Text prefilled | Attached ways prefilled |
| --- | --- | --- | --- | --- | --- |
| 32,919 | 43.2 s | 0.40 s | 0.65 s | 33,011 tokens | 92 tokens |
| 64,426 | 114.0 s | 0.57 s | 1.08 s | 64,517 tokens | 91 tokens |
| 131,290 | 380.8 s | 1.14 s | 2.90 s | 131,382 tokens | 92 tokens |

A joiner from GLM's cache has its first token 66, 105 and 131 times sooner than one reading the text; a joiner from the
same model's cache 110, 200 and 333 times sooner. Attaching grows with the context (0.23 to 0.69 s for the same model's
rows, 0.49 to 1.80 s for GLM's translated rows) but stays small beside a prefill.

One-time costs:

| Context | Resident Qwen reads | Its export | GLM reads | Pull over MCDMA | Parse | Translate rows and state |
| --- | --- | --- | --- | --- | --- | --- |
| 32k | 42.8 s | 0.2 s | 19.3 s | 0.35 s, 408 MB at 34 Gbit/s | 0.44 s | 7.0 s |
| 64k | 113.9 s | 0.7 s | 37.7 s | 0.61 s, 626 MB at 43 Gbit/s | 0.85 s | 12.5 s |
| 128k | 381.4 s | 1.3 s | 77.3 s, in two reads | 1.14 s, 1,279 MB at 46 Gbit/s | 0.88 s | 24.4 s |

Five joiners in all, one after another on the one Studio:

| Context | Five read the text | Same model: read once, five attach | GLM to Qwen: read once, five attach |
| --- | --- | --- | --- |
| 32k | 217 s | 45 s | 31 s |
| 64k | 571 s | 119 s | 59 s |
| 128k | 1,904 s | 390 s | 119 s |

At 128k, five joiners from GLM's cache are all answering in 119 s, where reading the text costs 381 s each. GLM's side
is cheaper than the same-model resident here because GLM reads on the two Sparks, faster than Qwen prefills on the
Studio.

Needles found, of five per context:

| Context | None | Text | Same model | GLM to Qwen |
| --- | --- | --- | --- | --- |
| 32k | 1 | 5 | 5 | 5 |
| 64k | 2 | 5 | 5 | 5 |
| 128k | 3 | 5 | 5 | 5 |

## What it cannot show

- The contexts are standard-library code, which the model knows: with no memory it named 1, 2 and 3 of the needles.
  The needles show the attached caches are read, not that they carry what text carries; answer quality is measured on
  the drop-in, held-out and Stockledger sets.
- Each context ran once. The joiners ran one after another in one process, which is a lower bound on what five
  separate joiners would pay, since none of them pays a model load.
- The translation cost (7 to 24 s) runs the contextual reader and advances the recurrent state on the Studio's GPU once
  per context; it grows about linearly with the context.

## Records

SHA-256, private local copies: contexts `c374f4361f11c6cf53357de9c46c84b5fdecb39b089fb36259a36a85453a97e7`, the jobs
file with GLM's reads and pulls `a7bdbb737c883dbb94aab2e8660855c09094ad4150e10f7b25fc08d32c5e8294` and its source lines
`fb467644c4265ffb5e516ed6fc4add9f3ee9710256e7eac80a1b710c47e74622`, results
`3d9c4438143d3ad9ab706786ebd92c6be555cb67801aa6b95ab5292535ab52db`; on the Studio, GLM's exports: 32k
`72e633f413957aa6f445e9f13ef7e272f44ad70b56eaf07974a56e58afb24cf3`, 64k
`201fc49a1681d2af1472df6862ade3c318230070d3204d95218d1535f801e651`, 128k first part
`12ac3cc0b9d2f36242b4b794152f7d57c44695673cfd5a41419658a7fd32838c` and delta
`46101763abc0e91a8cc6181a8fa5c9374167be51c9ee6d30ce04c13655228b40`. The run took 58 minutes on the Studio.
