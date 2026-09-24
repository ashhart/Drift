# Six directions between GLM, Qwen and DeepSeek

Status: every direction has a translator and a gate result, 24 September 2026. GLM and Qwen reach text on short code
contexts both ways; the four directions that touch DeepSeek V4 carry a weak signal and are not usable.

| Sender to receiver | Translator | Best gate result | Record |
| --- | --- | --- | --- |
| GLM to Qwen | contextual reader with answer-level fine-tuning | 32 of 32 short code questions, text 32; over MCDMA, 69 of 70 on a long context, text 70; a held-out project, 33 to 35 of 40, text 37 | [DROPIN_REAL_PROJECT.md](DROPIN_REAL_PROJECT.md), [HELD_OUT_PROJECT.md](HELD_OUT_PROJECT.md) |
| Qwen to GLM | reverse contextual reader for the rows; per-token map for the state | 32 of 32 short code questions from rows alone, text 32; on held-out code 1 of 5, text 5; 24 of 24 prose questions with rows and state, text 23 | below, [TRANSLATED_STATE_DEVELOPMENT.md](TRANSLATED_STATE_DEVELOPMENT.md) |
| DeepSeek to Qwen | per-token maps from DeepSeek's grouped entries, then answer-level training | 2 of 20, then 22 of 60 after training, text 58 of 60 | [DSV4_TRANSLATOR_DEVELOPMENT.md](DSV4_TRANSLATOR_DEVELOPMENT.md) |
| DeepSeek to GLM | per-token maps from DeepSeek's grouped entries | 3 of 20 | [DSV4_TRANSLATOR_DEVELOPMENT.md](DSV4_TRANSLATOR_DEVELOPMENT.md) |
| GLM to DeepSeek | DeepSeek's decoder in the shared space | 1 of 20, and that one a lucky guess; DeepSeek's own rows 20 of 20 | below |
| Qwen to DeepSeek | Qwen's shared-space encoder and DeepSeek's decoder | 0 of 20; DeepSeek's own rows 20 of 20 | below |

## Into DeepSeek, through the shared space

DeepSeek V4 keeps no per-token cache for its compressed layers: one entry per 4 tokens in 21 layers, each with an
indexer key, and one per 128 tokens in 20 layers. `drift/translate/dsv4_member.py` gives it a decoder in the shared
space (`drift/translate/hub.py`). For each entry, the hub vectors of the sender's tokens that end inside the entry's
characters are averaged, and ridge maps take the average to the entry's 448 content values in every layer of that
ratio, and at ratio 4 to the indexer keys. The 64 rope values carry position, so the writer keeps DeepSeek's own
values for placeholder text at the same positions. GLM enters the hub through the anchor projection and Qwen through its
encoder, so one decoder serves both senders: a new receiver needs one map, not one per sender.

`scripts/live/hub_fit_dsv4.py` fitted it on 380 passages with DeepSeek's page captures and GLM's latents, 9,055 ratio-4
entries and 138 ratio-128 entries. Averaging the tokens beat placing each token's vector in its own slot (R-squared 0.17
against 0.06 at ratio 4). On 41 held-out passages:

| Output | From GLM | From Qwen |
| --- | --- | --- |
| ratio-4 content | 0.17 | 0.18 |
| ratio-4 indexer keys | 0.43 | 0.38 |
| ratio-128 content | 0.08 | -0.27 |

For comparison, the maps out of DeepSeek reached 0.33 for Qwen's rows, and GLM to Qwen reaches 0.58.

The gate is DeepSeek's own-cache gate (`dsv4_drift_qualify.py`) with two more arms: the placeholder span is tapped once,
the translated content replaces its entries' content, and the result is written over the placeholders on both ranks.
Spans now start on a page boundary (`--align 256`), since translated entries are written as whole pages. The 10 ASCII
passages of the gate, 20 questions:

| Arm | Correct |
| --- | --- |
| native text | 20 |
| DeepSeek's own rows injected | 20 |
| no memory | 0 |
| GLM's cache through the shared space | 1 |
| Qwen's cache through the shared space | 0 |

Every write landed on both ranks. The translated arms answer with near misses and fragments (4308 for 7508, 120 for
126, 2061 for 6551), and GLM's one hit is "otter" inside a list of animals, which counts by containment but is a guess.
The DeepSeek side works: its own rows answer every question. The maps into it do not carry enough.

## Qwen to GLM on code

GLM's own-cache gate (`glm_state_gate.py`) on the drop-in's 32 short-context code questions, with GLM memory built on
the Studio from Qwen's rows over the four contexts (`translate_passages.py`), one row per Qwen token:

| Arm | Correct |
| --- | --- |
| text | 32 |
| GLM's own rows | 32 |
| GLM's own rows and state | 32 |
| no memory | 3 |
| Qwen's rows, prose-fitted stacked translator | 24 |
| Qwen's rows, shared-space pair | 4 |
| Qwen's rows, reverse contextual reader | 32 |

The reverse reader is the forward reader turned around (`studio_train_context_reader.py --direction qwen-to-glm`): a
four-layer bidirectional transformer over Qwen's rows for the whole context, correcting the shared-space pair, trained
on the forward reader's 1,919 windows for 6,000 steps (414 s), with GLM's spread restored by a fitted gain (median
1.25). Held-out R-squared rose from 0.528 to 0.584, and the share of translated latents nearest their own token's true
latent from 0.81 to 0.86. It was not fine-tuned on answers. On the 32 questions it answered as text did; GLM's own rows
answered 31 in the same run. The misses of the other two translators were mostly refusals, GLM saying it had no memory
of the file the question names; the shared-space pair leaves GLM's side at the anchor's projection with no spread
restored, which flattens GLM's attention as it did the forward reader's before its gain.

Qwen's translated state could not be sent. The first attempt poisoned GLM's session and stopped its server: the state
for a 1,518-token context is 34 layers of per-token inputs, 367 MB as float16, and the live receiver refuses any
publication over 128 MiB (`live_publication.MAX_BYTES`). The connector turns that refusal into an exception in the
worker, which stops the engine, where refusing the one request would do. The prose gate's passages of about 150 tokens
always fitted. A second limit stopped the engine on the long context, before any translated memory was sent: GLM's
own rows for 4,641 tokens failed the scheduler's causal-boundary check, which allows reserves of at most 4,096 rows.
The gate now skips any arm that would break either limit and records why; the server was restarted through its unit
both times. So Qwen to GLM through the live receiver is limited to contexts under 4,096 tokens: of the held-out
project's five contexts, only the first fits.

On that context, `packaging/direct_url.py`, code none of the translators trained on, the reverse reader did far worse
than on this repository's code:

| Arm | Code questions, 5 | Def lines, 4 |
| --- | --- | --- |
| text | 5 | 4 |
| GLM's own rows | 5 | 4 |
| Qwen's rows, reverse reader | 1 | 3 |

Every code-question miss is a refusal, GLM saying it has no memory of the message, and the one hit is a refusal that
happens to name the function. The def lines come through. This is how the forward reader behaved before it was
fine-tuned on answers, which is the step this direction cannot take while GLM runs only inside vLLM.

## What each weak direction needs

- GLM to Qwen: a closer fit on long methods and table rows; see the held-out record.
- Qwen to GLM: a way to send a translated state for contexts past about 480 tokens (the receiver's publication bound, or a state sent in parts), and tests past the short contexts: the long context, a held-out project and a coding task. Answer-level fine-tuning is blocked in this direction: it needs gradients through the receiver, and GLM runs only inside vLLM on the Sparks.
- The four DeepSeek directions: a reader that sees the sequence of DeepSeek's entries rather than one entry at a time, more than 380 captured passages, and for DeepSeek as receiver its 128-token sliding window, which the raw-row connector does not write. Answer-level training helped DeepSeek to Qwen (2 to 22 of 60) and would apply to DeepSeek as a sender; with DeepSeek as the receiver it has the same blocker as GLM.

## Records

SHA-256: DeepSeek's decoder `695a4171d38166dafb9a5b6b8d3fbf7e0ae2ac5e1f8fb19177089dcb9781dc97` and its fit report
`3a5b61daff297f203e7a736264edb13bfff5aef020fa57fc22c51d0e6994359e`, on the Studio; the gate's jobs
`dbd7bc5b9f0b7046b8ff67d03d5042af1a8cab30abeea2b3c623e432fa050aaf` and script
`85b9ea2518a194d67a1ea70a4b6f17e9194183911c3db8ba7ac3d445e28e5184`, on the head Spark; the gate report, private local
copy, `99cc580626c1aa646aa5f77c85cca92dcb3b42145b28606e39e38bbefb117a28`; Qwen to GLM on code, private local copies,
prose stack `75a9384a125acefe0e49c6e744a5e0ef1d04b7161a25169c8b42df40e9dc58f7`, shared-space pair
`4b32ecde86dd6395e9d138020a992fa1395a4776a0206d7606c39b00097b7422` reverse reader
`25326ceea9527727d42755f99af7708f1175326ae5b3eb5a5680d3c13e6d85cf`, and on the held-out context, code questions
`5b36c9d5684c6969564f2fe5f7e3940707271682c370228dbea8951621e126c2` and def lines
`dc56041f3637f45ca398d1659b17db2679edd2581a9dff70942db6ab53c11dc9`; on the Studio, the reverse reader
`baf4f16d7c852030d030c02aa08a975523cce6c6d03ee8b9201d102a631b000f`, its gain
`57c14c0134e4b7e411e68d80e251fad21ccdb0a36b12706506f3dbcee2f0f74b`, the pair it corrects
`57bf4c3fb2f6ba74c2a3559546a482fd78a5cc6dcc43fef9bd53f6d8a4c6607e` and the state translator
`b994804831d8950573a3c3d758d4d7d0fb440780f2c1686093dd132497695d84`. The DeepSeek gate took 611 s. Switching the Sparks
from GLM to DeepSeek took 5 minutes; switching back took 15, because the first start ran while DeepSeek was still
stopping on the worker and failed GLM's memory check.
