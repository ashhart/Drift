# Drop-in on a real project, development results

Status: exploratory, 23 and 24 September 2026. No claim rests on these runs.

## The question

A model joins work already in progress. Normally it must read the whole shared context first: its initial prompt processing grows with the context. With Drift, a resident model that has read the context shares its cache, and the new model attaches to that instead. Two things decide whether that beats text: the new model must answer as well as it would from text, and attaching must cost far less than reading.

## Setup

- Resident: GLM-5.3 on the Sparks reads each context in its reading frame; its handoff connector exports the cache.
- Joining model: Qwen3.8-Flash on the Studio. Its prompt is its system prompt, a line "Project files, from our shared memory:", the shared context, then the question.
- Arms: text (Qwen reads the context in its prompt), drift (Qwen prefills its own head, attaches GLM's translated rows and advances its recurrent state by GLM's translated layer inputs, then reads the question), no memory, and Qwen's own cache as the mechanism's ceiling.
- Translators: the loop's row stack and the reading-context state translator, unchanged.
- Time to first token is measured from an empty cache, so the drift arm includes the head prefill, translation on the GPU, attaching the rows and advancing the state.

## Scoring

Answers are scored by `drift/eval/answer_match.py`: the value an answer states for a constant or a default is compared as a number, a function name must appear as a whole identifier outside the question's own message and path, and a def line must parse as callable as the source defines it. Every question table here was rescored with it on 25 September from the recorded result files, whose hashes are below. The first scoring compared keys by substring after normalisation: it credited a key found inside a longer name, number or the question's own message (`40` in `400`, `pages` in `_load_pages`), counted a correct quote of a constant written in hex or with underscores as wrong (`64_000` against the key 64000), and never found a def line longer than eight lines. Most first-scoring numbers moved by one or two; the no-memory arms lost the hits they had from repeating the question, and the second dev project moved most, since five of its questions ask for constants written with underscores.

## Code questions from this project

`project_questions.py` reads modules of this repository with `ast` and writes questions whose answers are literally in the source: which function fails with a given error message, a parameter's default value. Answers are checked by exact containment after normalising case and punctuation. The contexts are the modules themselves, each under a header with its path.

Four contexts of 1,013 to 1,406 GLM tokens, 32 questions, within Qwen's 2,048-token full-attention budget:

| Arm | Correct | Time to first token, median | Tokens prefilled, median |
| --- | --- | --- | --- |
| text | 32 | 1.882 s | 1,413 |
| no memory | 0 | | |
| Qwen's own cache | 32 | 0.404 s | 96 |
| GLM's translated cache | 26 | 0.470 s | 96 |
| GLM's translated rows alone | 25 | | |
| GLM's translated state alone | 1 | | |

The joining model answered 26 of 32 from GLM's cache while prefilling none of the project: its first token came 4.0 times sooner, after 96 prefilled tokens instead of 1,413. The six misses are real confusions between neighbouring functions with similar messages, such as `final_frontier` and `processed_frontier` in one module. On code the translated rows carry the answers; the translated state adds nothing here.

An earlier run with a 48-token answer limit scored 26; five of its six misses were answers cut off while restating the long error message, so the limit was raised to 120.

## Past Qwen's sparse-attention budget

Qwen3.8-Flash attends the complete prefix only while the context fits its `indexer_budget` of 2,048 tokens; past that, a selector picks key blocks by selector keys. Appended memory rows carry no real selector keys: Qwen's own rows get zeros, GLM's translated rows get the row stack's approximate ones. On the first 6 questions over one context of 4,603 GLM tokens (three modules), with the budget as shipped:

| Arm | Correct |
| --- | --- |
| text | 6 |
| Qwen's own cache | 0 |
| GLM's translated cache | 2 |

Even Qwen's own cache went unread: the selector never picked its rows. `--budget` raises every full-attention layer's budget for the joining model, so it attends the whole memory densely; text and Drift run with the same budget.

With the budget raised to 16,384 for both arms, the first 30 questions over the same context:

| Arm | Correct | Time to first token, median |
| --- | --- | --- |
| text | 30 | 5.05 s |
| Qwen's own cache | 30 | 0.94 s |
| GLM's translated cache | 12 | 1.30 s |
| GLM's translated rows alone | 7 | 0.42 s |
| GLM's translated state alone | 0 | 0.97 s |

Attaching a cache instead of reading works at this length: Qwen's own cache answers exactly as text does, 5.4 times sooner. What separates GLM's translated cache from text is translation, not the mechanism. These 30 questions lean on one function, `decode` in `mcdma_forward.py`, which holds 19 of the 70 questions' answers; the full set follows.

## Across MCDMA

`spark_dropin_export.py` has GLM read each context on the Sparks and leaves the handoff export in place. `studio_dropin.py` stays resident on the Studio: `handoffd` RDMA-reads the export into shared memory, the context's latents are translated on the GPU, and for each question a fresh Qwen attaches them after its own head. The same question is answered from text in the same process. Budget 16,384.

| Context | Export pulled | Transfer | Parse | Translate |
| --- | --- | --- | --- | --- |
| 1,013 to 1,406 tokens (4 contexts) | 150 to 166 MB | 55 to 71 ms, 48.7 to 49.1 Gbit/s | 12 to 15 ms | 68 to 85 ms |
| 4,603 tokens | 218 MB | 81 ms, 48.9 Gbit/s | 75 ms | 273 ms |

| Questions | Drift correct | Text correct | First token, Drift | First token, text |
| --- | --- | --- | --- | --- |
| short contexts, 32 | 26 | 32 | 0.41 s | 1.33 s |
| long context, 70 | 42 | 69 | 0.95 s | 4.63 s |

The first-token times are per question after the context's memory was pulled and translated once; counting that one-time cost as well, the joining model's first answer on the long context still starts about 3.4 times sooner. The export carries far more than the latents Qwen uses (KDA state pages, selector caches); the transfer is not the bottleneck at 49 Gbit/s.

On the long context 18 of Drift's 28 misses are questions whose answer is `decode`, a function of about 50 lines holding many error messages: Drift answers with a name built from the module's docstring or a neighbouring method, never placing a message that deep inside the function. On the other 52 questions Drift answered 42.

## Two more measures on the same contexts

A balanced long set: 25 questions over the 4,603-token context, no answer used more than twice (14 different answers). And a coding task: for 25 functions with two or more parameters across all five contexts, write the function's def line exactly as the file has it, scored by parsing (`drift/eval/signatures.py`): callable means the names, kinds, order and defaults match, exact adds annotations. Budget 16,384, the loop's stack and prose-fitted state translator.

| Arm | Balanced long set | def lines callable | def lines exact |
| --- | --- | --- | --- |
| text | 24 of 25 | 25 of 25 | 25 of 25 |
| Qwen's own cache | 25 of 25 | 25 of 25 | 25 of 25 |
| GLM's translated rows and state | 18 of 25 | 16 of 25 | 13 of 25 |
| GLM's translated rows alone | 16 of 25 | 13 of 25 | 10 of 25 |
| no memory | 0 of 25 | 0 of 25 | 0 of 25 |

## Translators fitted on code

The forward translators were fitted on prose. Refitting them on code (`fit_state_translator.py` on 125 of this repository's own modules, none of the test modules, 133,957 aligned token pairs, GLM latents in the reading frame against Qwen's own rows and recurrent-layer inputs in the drop-in's layout) reached validation R-squared 0.63 for the rows and 0.36 for the state on the four held-out short contexts. The spread gain these maps carry was fitted on the rows of those same four contexts, which their 32 questions then score, so the short-context numbers of every translator built on these maps are not fully held out; it is one scale per dimension, so the effect should be small. On their 32 questions:

| Rows and state from | Correct |
| --- | --- |
| the loop's prose-fitted row stack and state translator | 26 |
| ridge maps fitted on code | 26 |

A better-fitted linear map does not help: both land where the per-token translators always land. What a token means in its context is what gets lost, and a map that sees one token at a time cannot recover it.

## A contextual reader

`drift/translate/context_reader.py` reads GLM's latents over the whole context before writing any row. It is a four-layer bidirectional transformer, width 1,024, eight heads with rotary positions, over GLM's latents reduced to 2,048 principal directions; its output is a correction to the code-fitted rows map, in units of each target dimension's spread, and its output layer starts at zero, so the untrained reader is that map. The recurrent state still comes from the code-fitted state map.

`studio_train_context_reader.py` trains it against Qwen's own rows. The corpus is this repository's Python files in 921 overlapping windows of about 5,200 characters (1,400 to 1,600 tokens), leaving out the nine test modules and every file that names one. GLM read each window in its reading frame (`export_passages.py`, with token offsets), and Qwen read the same window in the drop-in's layout (`studio_tap_passages.py`). Where a GLM token and a Qwen token end on the same character (`drift/translate/alignment.py`), Qwen's own rows are the target at that GLM position; the loss is the mean squared error there. Validation uses the four short test contexts.

In 3,000 steps (209 s on the Studio, 1,179,064 aligned rows) the reader raised validation R-squared from 0.578, the code-fitted map alone, to 0.645, and every one of Qwen's twelve full-attention layers improved by 0.05 to 0.09.

### What squared error did to the rows

On the 32 short questions the trained reader did far worse than the map it corrects: its rows alone answered 7 where the map's answer 27, and with the state 18 where the map's answer 26. The answers had the gist and lost the details. Qwen read the file header correctly, then decided the code came from some other fp8 module and could not find the exact error messages.

Three suspects were ruled out. Unaligned positions: on these contexts 99.8% of GLM's tokens end on the same character as a Qwen token, so almost every row the reader writes was trained. A different export: the gate's GLM export of these contexts differs from the validation export at depth (layer 3 identical, layer 43 different by 17 to 21% relative), but both translators fit either export equally well. Lost token identity: matched against every true row in its context, each of the reader's K rows found its own token as often as the map's did, or more.

The spread was the cause. The map carries a per-dimension gain that gives its rows the spread of Qwen's own (ratio 1.00), at some cost in squared error. Trained on squared error, the reader pulled every row toward the average: its rows had 0.78 to 0.86 of Qwen's spread in K and 0.71 to 0.85 in V. That lowers the error and flattens Qwen's attention over the memory, which is what the answers showed.

`drift/translate/spread.py` puts the spread back: each output dimension of the reader's rows is rescaled about its mean to the targets' spread. Fitted on 160 training windows (206,439 rows), the gain has median 1.24 and restores the held-out spread from 0.80 to 1.00; held-out R-squared goes from 0.645 to 0.606, still above the map's 0.578. The trainer now fits and saves the gain (`gain.npz`), `ContextRows` applies it, and the answer-level trainer keeps it fixed while fine-tuning.

### With Qwen's spread restored

Budget 16,384, the code-fitted state map, GLM's export as the gates read it. The stack is the loop's per-token row stack.

| Measure | text | Qwen's own cache | stack, rows and state | stack, rows | reader, rows and state | reader, rows |
| --- | --- | --- | --- | --- | --- | --- |
| short contexts, 32 questions | 32 | 32 | 26 | 25 | 28 | 27 |
| balanced long set, 25 | 24 | 25 | 18 | 16 | 24 | 21 |
| def lines callable, 25 | 25 | 25 | 16 | 13 | 18 | 18 |
| def lines exact, 25 | 25 | 25 | 13 | 10 | 18 | 18 |

On the balanced long set, 25 questions over 4,603 tokens, GLM's translated cache now answers as many as text. On the short contexts the reader resolves three of the neighbouring-function confusions the per-token maps never did (`finish_stream`, `final_frontier`, `_exchange`) and loses two `encode_entries` answers the stack had. Whenever its def line has the right parameters, it also has the exact annotations.

In these gates each question translates the context again, so the reader's time to first token (median 1.11 s on the short contexts, against text's 1.88 s) includes running it; the drop-in over MCDMA translates each context once.

### More data, GLM's full latents, and reading in windows

Two more readers, each with the spread gain fitted the same way. The second adds 1,000 windows of Python's standard library (1,919 windows, 2,497,244 aligned rows, 9,000 steps). The third trains on the same windows but reads GLM's full latents, 5,632 values per token, standardised, instead of the rows map's 2,048 principal directions: those directions keep 92% of GLM's variance overall but only 74 to 85% in its first three MLA layers, where token identity sits. The readers trained on windows of about 1,500 tokens, and on the 4,603-token context each token attends three times as far as it ever did in training, so `ContextRows` can now read a long context in overlapping windows of 1,536 tokens (`--reader-window`), each token taking the window whose centre is nearest it.

| Reader | Held-out R-squared | Short contexts, rows / rows and state | Balanced long set, rows and state, whole / in windows | def lines callable, rows / rows and state |
| --- | --- | --- | --- | --- |
| first, this repository only | 0.645 | 28 / 29 | 24 / 25 | 18 / 18 |
| second, with the standard library | 0.670 | 31 / 30 | 22 / 24 | 18 / 16 |
| third, reading full latents | 0.674 | 26 / 29 | 25 / 25 | 17 / 19 |
| text | | 32 | 24 | 25 |

Reading in windows lifts the balanced long set, and two readers now answer 25 of its 25 questions, one more than text. The second reader answers 31 of the 32 short questions from rows alone; its misses name a neighbouring method (`pull` for `view`) or the neighbouring frontier function. No reader closes the def-line gap.

### Fine-tuning on exact copying

Every remaining miss had one cause: the translation carried what a token means and not always the exact token. Qwen named a neighbouring function, wrote `world_size` for a parameter called `world` or `timeout_s=40.0` for `40`, or refused because a file's path did not quite match. The third reader was fine-tuned on answers (`studio_train_memory_answer.py`): the teacher is Qwen reading the text, the student is Qwen given GLM's cache through the reader, and the loss is the KL divergence on the teacher's answer tokens, with gradients through Qwen's frozen layers into the reader and a rank-32 correction of the state map. The spread gain stays fixed.

The questions ask for exact copying and come from text GLM had already read, none of it from the test modules (`drift/eval/window_questions.py`, `code_triples.py`): write out a function, write its def line, name the function that fails with a message, quote the line that follows a given line. There are 7,716 of them: 1,305 on 126 whole modules of this repository, 5,520 on the 1,919 windows of this repository and the standard library, and 899 line quotations from 298 windows of public Markdown package documentation, for text written like a specification. They use the same question forms as the short-context and def-line tests, on other code. 125 updates of 4 answers, learning rate 5e-5, 102 minutes on the Studio. The trainer's own check on the 32 short questions rose from 28 to 31 during training.

| Measure | before fine-tuning, rows / rows and state | after, rows / rows and state | text |
| --- | --- | --- | --- |
| short contexts, 32 | 26 / 29 | 31 / 30 | 32 |
| def lines callable, 25 | 17 / 19 | 23 / 25 | 25 |
| def lines exact, 25 | 17 / 19 | 23 / 25 | 25 |
| balanced long set in windows, 25 | 22 / 25 | 24 / 25 | 24 |

GLM's translated cache now writes every def line as text does, annotations and defaults included, and answers the balanced long set better than text. The short contexts' last misses are the neighbouring pairs in one module, `final_frontier` answered as `processed_frontier`, and once `finish_stream` as `finish_taps`.

On Stockledger the field names now come through. Greedy and three samples at 0.7, graded by the 33 hidden checks:

| Seed | text | rows | rows and state |
| --- | --- | --- | --- |
| greedy | 33 | 30 | 18 |
| 1 | 33 | 30 | 29 |
| 2 | 23 | 25 | 28 |
| 3 | 33 | 29 | 24 |

Six of the eight modules written from GLM's translated cache accept valid events and pass 28 to 30 checks; those that fail miss finer rules of the specification: invalid UTF-8, the byte order mark, a dot in a SKU, duplicate keys, impossible dates and leap seconds. Text's own sampled module failed the same way once (seed 2, 23 checks). First tokens came 0.43 to 0.68 s after 168 prefilled tokens, against 2.25 to 2.27 s after 1,852.

### Across MCDMA with the reader

The same drop-in as above on fresh exports: GLM read the five contexts again on the Sparks, `handoffd` pulled each export at 47.4 to 49.1 Gbit/s, and `studio_dropin.py --context-reader` translated each context once on the GPU (62 to 83 ms for the short contexts, 259 ms for the long one, reader and state together). Budget 16,384, the code-fitted state translator.

| Questions | Drift correct | Text correct | First token, Drift | First token, text |
| --- | --- | --- | --- | --- |
| short contexts, 32 | 27 | 32 | 0.41 s | 1.31 s |
| long context, 70 | 66 | 69 | 0.97 s | 4.62 s |

On the long context GLM's translated cache now answers 66 of 70 against text's 69, with the joining model's first token 4.8 times sooner after 95 prefilled tokens instead of 4,992. Counting the one-time pull, parse and translation (0.40 s), its first answer starts 3.4 times sooner. It answers all 19 questions whose answer is `decode`, where the stack answered 1; its four misses all ask for `remember`, a method of `ForeignPositionBank`, and name its neighbours `replay` or `rephase` instead. The first run used the prose-fitted state translator, so the rows and the state both changed between the two runs. A control gate separates them: the stack's rows with the code-fitted state answer 44 of the 70 long-context questions and 26 of the 32 short ones, against 42 and 26 with the prose-fitted state, so the state accounts for two answers on the long context and none on the short ones, and the reader for the rest.

### Stockledger with the reader

The Stockledger task below, with GLM's cache read by the reader: greedy decoding, then three samples at temperature 0.7 with text sampled beside it, each module graded by the 33 hidden checks. Greedy text is deterministic, so its row is the earlier greedy run's.

| Seed | text | reader, rows | reader, rows and state |
| --- | --- | --- | --- |
| greedy | 33 | 0 | 0 |
| 1 | 30 | 24 | 23 |
| 2 | 33 | 0 | 24 |
| 3 | 31 | 23 | 24 |

When the reader's module compiles it passes 23 or 24 checks, as the stack's did; text passes 30 to 33. The three zeros are generation failures, not missing rules: one answer turned into tool calls to explore the project, one repeated a comment until it ran out of tokens, and one left `# ... implementation ...` as a function body.

Every compiled module written from GLM's translated cache, by either translator and at every seed, fails the same nine checks, and all nine feed it a valid event. The cause is two field names. The modules require `quantity` and `timestamp` where the specification names `delta` and `occurred_at`, so every valid event fails the key check, and the error then names the wrong line. The format rules came through: the event_id, warehouse and timestamp patterns match those in text's modules, and the SKU pattern lacks only its hyphen. The 23 or 24 checks these modules pass are mostly the ones an over-strict validator passes by rejecting bad input. The translation carried what the two fields mean and lost their exact names. First tokens came 3.4 to 5.5 times sooner than text's: 0.40 to 0.65 s after 168 prefilled tokens, against 2.18 to 2.26 s after 1,852.

### A second phase, the average, and a third phase

Two more fine-tuning phases of 125 updates each, same loss and learning rate. The second continued from the first on 9,923 questions that add 2,498 requests to quote an 8-line block (shuffle seed 1). The average is the elementwise mean of the first two phases' reader and state corrections. The third continued from the average on 7,785 questions that keep 292 block quotes and add 368 questions about Markdown tables, to target Stockledger's field table (shuffle seed 2, 105 minutes). The dev gates, rows / rows and state:

| Translator | Short contexts, 32 | Def lines, 25 | Balanced long set in windows, 25 |
| --- | --- | --- | --- |
| first phase | 31 / 30 | 23 / 25 | 24 / 25 |
| second phase | 30 / 32 | 23 / 24 | 24 / 25 |
| average of the two | 32 / 32 | 24 / 25 | 24 / 25 |
| third phase | 30 / 30 | 24 / 25 | 24 / 25 |
| average of the average and the third phase | 32 / 32 | 23 / 25 | 24 / 25 |
| text | 32 | 25 | 24 |

Stockledger, rows / rows and state, greedy then three samples at 0.7:

| Translator | greedy | 1 | 2 | 3 | Mean, rows |
| --- | --- | --- | --- | --- | --- |
| first phase | 30 / 18 | 30 / 29 | 25 / 28 | 29 / 24 | 28.5 |
| second phase | 25 / 19 | 0 / 29 | 0 / 31 | 18 / 0 | 10.75 |
| average of the two | 29 / 0 | 25 / 31 | 30 / 30 | 32 / 0 | 29.0 |
| third phase | 32 / 33 | 0 / 31 | 29 / 12 | 29 / 23 | 22.5 |
| average of the average and the third phase | 30 / 23 | 0 / 23 | 25 / 0 | 29 / 29 | 21.0 |
| text | 33 | 33 | 23 | 33 | 30.5 |

A second dev project tests code none of these translators trained on: 40 code questions and 25 def lines over files of
fsspec, urllib3, yaml, networkx, jinja2 and pytest. None of their code is in training or in the held-out set, though the
Markdown training windows include the urllib3 and fsspec READMEs. Rows / rows and state:

| Translator | Code questions, 40 | Def lines, 25 |
| --- | --- | --- |
| average of the first two phases | 36 / 39 | 18 / 21 |
| the same, reading in windows of 3,072 tokens | 38 / 38 | 18 / 21 |
| third phase | 37 / 37 | 21 / 23 |
| average of the average and the third phase | 38 / 38 | 22 / 23 |
| text | 40 | 24 |

With rows and state, the last average is level with text or ahead on the short contexts, def lines and the balanced
long set, and two code questions and a def line short of it on the second dev project; it also wrote two modules that
pass nothing on Stockledger.
Every translator after the first phase wrote at least one; no text arm did. Modules written from the translated cache carry more
comments than text's (60 to 111 comment lines against 24 to 61), and the failures are a module cut off at the answer
limit, a `try` with no `except`, and validators that reject valid events.

At this point the average of the first two phases stayed the final translator. It was the only one at text level on the short contexts in both arms, level with the best on def lines and the long set, and its rows arm had the best Stockledger mean with no module at 0. After a fourth phase, a rule fixed beforehand chose again; see [the final choice](#a-fourth-phase-and-the-final-choice). The table questions did not fix the table: the third phase fell back on the short contexts, and its rows arm wrote no working module on one seed. Every zero is a generation failure, not a missing rule. One module had `f'...")`, a quote closed with the wrong mark, and the others filled the 3,500-token limit with planning comments before the code was done. The rows' commonest miss is still the SKU pattern: the specification's table gives event_id dots and SKU none, and the translated cache gives SKU the dots of the row above it.

With the average, the drop-in across MCDMA on fresh exports answered 32 of the 32 short questions, as text did, and 69 of the 70 long-context ones against text's 70. First tokens came after 0.41 s against 1.31 s on the short contexts and 0.97 s against 4.56 s on the long one, after 96 and 95 prefilled tokens instead of 1,413 and 4,992. The pulls ran at 48.3 to 48.9 Gbit/s, and translating a context took 60 to 75 ms, or 276 ms for the long one.

### A fourth phase and the final choice

The fourth phase continued from the average of the average and the third phase on 4,143 questions. Half of them, 2,072,
come from 599 windows of other installed packages' source that GLM read for this phase, among them huggingface_hub,
pygments, setuptools, sympy and torch; the rest are drawn from the earlier sets. 125 updates of 4 answers at 5e-5,
shuffle seed 3, 108 minutes on the Studio. The trainer's own check on the 32 short questions fell from 31 to 29.
Rows / rows and state:

| Translator | Short contexts, 32 | Def lines, 25 | Balanced long set in windows, 25 | Second dev project, 40 and 25 |
| --- | --- | --- | --- | --- |
| average of the average and the third phase | 32 / 32 | 23 / 25 | 24 / 25 | 38 / 38 and 22 / 23 |
| fourth phase | 28 / 30 | 23 / 24 | 24 / 25 | 37 / 37 and 22 / 22 |
| text | 32 | 25 | 24 | 40 and 24 |

Stockledger, rows / rows and state, greedy then three samples at 0.7:

| Translator | greedy | 1 | 2 | 3 | Mean, rows and state |
| --- | --- | --- | --- | --- | --- |
| average of the average and the third phase | 30 / 23 | 0 / 23 | 25 / 0 | 29 / 29 | 18.75 |
| fourth phase | 29 / 32 | 6 / 32 | 25 / 0 | 31 / 32 | 24.0 |
| text | 33 | 33 | 23 | 33 | 30.5 |

The fourth phase's modules from rows and state passed 32 of the 33 checks on three seeds, one short of text. Its zero
is a new kind of failure. The answer is the context's own stub of `validation.py`, copied whole: a docstring and
`raise NotImplementedError`. On the questions it gave up five hits: two short contexts, a def line, and a code
question and a def line of the second dev project.

A rule written before the fourth phase's results picked the final translator from three candidates: the average of the
first two phases, the average of the average and the third phase, and the fourth phase. The rule, `choose_final.py`,
is a private local file. It takes the most rows-and-state hits over the short contexts, def lines, the balanced long set and both
halves of the second dev project, among candidates whose rows-and-state Stockledger modules never pass 0 checks. If
none qualifies, it takes the most hits outright. Held-out results play no part. Every candidate had a module at 0, the
average of the first two phases on two seeds, so the rule fell back on hits: 139 for the average of the average and the
third phase, 137 for the average of the first two phases and 133 for the fourth phase. Rescored with the corrected
scorer they are 143, 142 and 138, in the same order.

The average of the average and the third phase is the final translator. Its reader and state translator are the ones
recorded below for the last average, and they are the GLM to Qwen files in the `translators-glm-qwen-experimental-v2`
release. It is below text on Stockledger: its modules from rows and state average 18.75 checks against text's 30.5,
with one at 0.

## A coding task on the Stockledger benchmark

Stockledger is the owner-selected benchmark project: a warehouse-movement ledger CLI whose SPEC.md fixes every rule, with the implementation left as stubs. It is used read-only; nothing here changes its repository. GLM reads SPEC.md, `stockledger/validation.py` and `stockledger/cli.py` (1,574 GLM tokens). The joining Qwen is asked, with the interface only, to write `stockledger/validation.py` with `read_events(stream)`; every field rule has to come from the specification, which it gets from GLM's cache, from text, or not at all. `stockledger_task.py` grades the returned module with 33 hidden checks derived from the specification's event-schema section (valid events, blank lines, bounds, each field's format, duplicate keys, null, byte order mark, invalid UTF-8, and errors that name the physical line), each in an isolated Python process with a bare environment and a timeout.

The first run allowed 1,500 answer tokens, and four of the seven answers ran past it mid-module, leaving no complete code block to grade. With 3,500:

| Arm | Hidden checks passed | Time to first token | Tokens prefilled |
| --- | --- | --- | --- |
| text | 33 of 33 | 2.28 s | 1,852 |
| Qwen's own cache | 33 of 33 | 0.58 s | 168 |
| GLM's translated rows | 24 of 33 | 0.41 s | 168 |
| GLM's translated rows and state | 0 of 33 | 0.66 s | 168 |
| GLM's translated state alone | 12 of 33 | 0.59 s | 168 |
| no memory | 11 of 33 | 0.31 s | 168 |

A joining model that attaches a cache writes a validator that passes every hidden check, 3.9 times sooner than reading the specification. A module that rejects every event passes 24 of the 33 checks, since most of them feed a bad event, so 24 shows little: the module from translated rows passes 24 because it rejects every valid event (see below), and guessing passes 11. The rows-and-state module failed every check on a single character, `{line_number]` in an f-string, so it never compiled; it had also dropped dots from the event_id rule. Each arm is one greedy sample, so these are single observations.

Results SHA-256, private local copies: short contexts `62b05fcde75b864685d2fc72297a065ff5f43caacabc7e5406301c4070b99e11`; long context with the budget raised, first 30 questions, `8f70dc00ea4c06922ba349a164f2805a95a04af0a2f8192f9dc7398e2dfd8c22`; across MCDMA `db1ab4e69e5a4d4e5228ab4bc16086ad538e9c4b4df56412c00067c05796ec8c`; Stockledger answers `8b6fa4f06a2ae1cab5b86e3de945364135d49f16aeea21b7fb7afc322f74c027` and grades `2fdd983f5d17df61165927716b3546eaffc27e77044c8b24372fcdb9fe2c3291`; balanced long set `df96587fac52169502fae7a529bbba3637ea791b859c25f71234910620a83f89`; def lines `c02d8cc95159b865d323f37d9c20a31f9eb1f5ee8a48d78cb326a64a9ae05fd4`; prose stack and code ridge on the short contexts `ff9504c2793681a1d1c735efbe3b5b3a972935407d69e0d076e1873c430d784b` and `65379f7cf8256f4aeb490837a311b598ebb344693d6c67c956b8ca73474159dd`; long context with the budget as shipped, first 6 questions, `1b55fc8a5baf82ac2a967846ac30e9f57c591b9a5cb8a2288143cc5f7932754e`.

Contextual reader results SHA-256, private local copies: the reader before its spread was restored, short contexts `b59655333de31ab7e4231899797aa2971977218fe5e10a11c8a63a49a8e56b18`; with the spread restored, short contexts `39f1b7b6680e94a4f51143de473ed5b1b6d4c87f31a8bea981dc63aeb63ae4ce`; balanced long set `b4848cebfad31ede69aef6b0e21f90eac74bbc7a5e299b341153402ed9a56fef`; def lines `dedab4444a88cb2cebb0d13e05f5bf20b682d0240498f0f879f4b01af85d02e4`; the gain's fit report `7d6aefc8f81f2a893136693d5ec58b5a51ba26ea07ec2f56b0652611993cd5f7`; the reader's training history `fcc0b5fff62f5e7061feb88819bc1a958429aa737d15c14628b2d4d3d392fe92`; the MCDMA drop-in with the reader `253e44dd602e7ae4a8c7e2bff39d73cdbf58b8d25ea9b145cc25ea39d74b381a`; on the Studio, the reader's weights `abb366a6878ba5f1c756beca4dcff9a5b14dc8d20bfc15ae598ec0b6b387711f` and its gain `90d7ac45db4029ffd0841c22f5f43e81e0fbd13540fe74b25f0c94fdd8370725`.

Stockledger and control results SHA-256, private local copies: Stockledger with the reader, greedy `49115fe1413c26dfc7045635fad1c53d5640f63b55b9b706dd79d4cad3d4ecf4`; seed 1 `ffa516a595664def2d4308422b31a314cffe7c4e46b3c50e09c81df93db8cbf8`; seed 2 `3d0ea92c985eadbba9f150789ab6787ee6114f8bc11c70abb537f4ae69572242`; seed 3 `561567d69728f157f107b8cf52afaf8fdcc7f348ced3c043fa085b03c40f2987`; their grades `840f5af182a1db03fcfa043397dca8c1690fd07113f9c212d41c5a6ceea03929`; control gates, the stack's rows with the code-fitted state, long context `d4d1d3baa50f27ae43a5dfbf152364888781c317da81ba1b921deeffaa5bd3d3`; short contexts `16adf20e9a5ad03e49b11075a1396690ad6182ec26112329b7624eb95ced0a1c`.

Second and third reader results SHA-256, private local copies: second reader, short contexts `02773c87e8e3e33a5a1c1309c696e9ca516bd6f454facabab0eeeb73ec30806c`; balanced long set `300c250929f6c108e846580fdd0360ee676b083e9173b92419bc42f4b0f0d3eb`; def lines `aa71bb816c30bd1f720ac7e262369ccb6afec67738252d98067c6e0dc8d360bd`; in windows `3a65312f53aa1ec2c50ce6d09691fc94f17f6bb22d9407b415d3ab61980f4f35`; third reader, short contexts `575adacddc8ba9ccd7e1647d34c253e42b58f1561b1e1afd1a15a3e1136a5a3a`; its balanced long set `dc941bf1f735d5c48ba402043b669d19e10d8a6b2677f12db2884f68dc417af4`; its def lines `c8d05638d9bda85a27c00c1ed76a40a1e7d94653d581e0dcc6730b2feb6de9f9`; its balanced long set in windows `f6c807cc32dcde1f4ee97e22114811d3fd0150c8560d0e1367599ec9cddc8986`; first reader's balanced long set in windows `890c39162dff85d68141c42ab2a5bb9b5680de9ad28b69e65c0204cab557e8e0`.

Fine-tuning results SHA-256, private local copies: fine-tuned reader, short contexts `8114574b77ddb43986708d39c58a2f7d8eff1538d57c8daab9f78b5f366f0893`; def lines `aef3ad135f4fae86a1a38ab710a7a1886441dcc6d865afad46367f3ee119c4f6`; balanced long set in windows `c3608c1dcdd84f8eb20e9011e24dfef933f5544c5caf0c7b2f7378aa65a3fcd0`; training history `aec54194599d29ec088c6aadf21713c714ec95905b167a55e9a5867585a14fcf`; Stockledger greedy `a5d41561fe80c3a1cef55740d9fa4950083da17747cf6bc4368e643dbae608be`; seed 1 `906a982272765d3b12297b3ffd8eda9233347767a31b06a343636ee157b8f1aa`; seed 2 `0d081ca95883badd85ba2c0e4d66ce3f7361d1bb79d2acd02da37686d0bf420d`; seed 3 `2bbedea228f604f240a76587c14ab1e2d63adbefe54c33ad6400b9108287011d`; their grades `57604d871ba91aea7b21b1d8aafe00940ca0b62726be19a5df791fcfce5f901d`; on the Studio, the fine-tuned reader's weights `9438aecbe569dbdc3e146bda745287a79ae58f7c5fa679b1a7e57d8c95c5c15c` and state translator `cf39aff5c9ef5ca3732c8b588eeec12696c25356a3bd56746420bb94e9eea8d1`.

Later phases results SHA-256, private local copies: second phase, short contexts `9f46137c649b3946b980c58650db452495c74ddf73cc675a77ed65dd6ea46a2c`, def lines `6230becdf4ec4ba21c2f89fb904b3a4b72e6d03ca571793139958e62dd56a604`, balanced long set `362b539ff7c24006480d488a62294c8a60f9c829bd08f0dacdaba0499515b973`, Stockledger grades `fc6acf758a43cfb96a2cd39d6303e52624278db49a598c9b9b872183f7ae042e`; the average, short contexts `419e36b77b24cb152a3b73ff83c504c074736127201912f1cce882576a74ca9b`, def lines `bc9c9627ff2406d8b21e7dcc7b4a46034cafb30688ef49cfe75378fbb7779f7c`, balanced long set `0c8cb82a80762d830f6cece60beeeaa3c7802e23d5302d7403f83bc2f9f50fa9`, Stockledger grades `db7003629f3fcd8da4099ef32c7cb9b0c25b9ab993bcd024e64f1a234e15d372`, the MCDMA drop-in `91f1504173b6884f61cb680004f65fd4a1f4f34588e88b8cb0d9fcbcdf74706d`; third phase, short contexts `a3c0c24ed44ef35d758f5659897e9ed4032541da5edb164a6540a1ffb06c35d3`, def lines `e228e145ac9bfe7da97d3f3f05b1841f5650b9ba9231dae4fe50fb0e56f58386`, balanced long set `549d42f1373793b0ba869fc8c6e405886add35007e52fd9fadfcbeed2a29990f`, Stockledger grades `6adccfa8ffd7796d7075e2b93fb7b86f13be076af9c6699f2c904c6ed4cb64ea`; on the Studio, the average's reader `ad8666de3b335537c3fe3a02588a0916ca20ff0985125360a620322db153fa20` and state translator `bee4d090081563a22f0f379262321d811e75a20e7276f1530734fd18570dd549`, the third phase's reader `a54e0e13b7fe8a7a461b6e9c5b20fe798778d3d2379dd64caf3fa4b399013e2f` and state translator `2284747c8bed4774c890169be14b6d92e29157a19b2445c45948ad22d69344d6`.

Second dev project and the last average, SHA-256, private local copies: items, code questions `f2eaf1a6113b9b3b6eb2824f26757245ded068bb45e5a2c393156741c07c076e` and def lines `a217aec827a6936ee0f79eedea0f06eb8bae3446750090df973245bc5d17e8d8`; average of the first two phases, code questions `769489b47705858b180106a72c01cfd85fc6984e4726ef77f7ec4e9aa29cdb7d`, def lines `9d14f89f503b6fc37884355d861e739635bfe2060e1856180014e169c65ac4fa`, and reading in windows of 3,072 tokens `290618c472de092d77f76f8315708357b08fc14653f1f4233af1b5ec45aae2d1` and `e4a826f0c845e7a7837ffb00afc2c3cc61a0fd6e7f944ff08d1e03ca36138455`; third phase `b5bcf6a73b1009c62c3f9132ec163ac81284a79c30da985c7283a1f41177f6d4` and `2dc6206895148bcf6100afee44fbf1046a8d93eb5e40b8e7b424598a75bef819`; the last average, short contexts `ae5787bbf5389968595172867ac18d3df2bb2b85fd51a6296f553f23a6ebda7b`, def lines `092c9a6166c2484f7253c266bc67b9b71dc8b277d5134a93246aa0300aba3843`, balanced long set `b36e04d8fd216fbd095fe09d4c19630dd9abb7bc2a080fa0e8c3d518d5ed0f8b`, second dev project `30b009f075edbeacfdfc9e042906c82d7d729713e6873ac4b3dacf85865c1c06` and `15712f66a8f630ce06ebb8151577bb7f0e008a899dc5049a39d604fef9501830`, Stockledger grades `253247b783a2295984bdfd548a705879b6c67cef26f4f23303540060499f5d49`; on the Studio, its reader `29aef68de60cc44d3daae0e6ff05f3b4fddf5bcd4711013404f99bcaedd9d157` and state translator `69bd379af92dae6b0201e1eeaaa74b674844914ce0f82034e4a5109097c2e535`.

Fourth phase and the final choice, SHA-256, private local copies: training questions `ab82f00f1c3bbfb36a2cd2edafce5c19fa4a0ad825e0cbad0a3e68d5b4bd969b`; short contexts `148f68127c8a69dc60b279866ec0b405676e7de14722f3af0a115c087438d3d9`, def lines `968ddf7fb0a3e5c0e58ddf0bb14a8427937334eb0da9aa70d2bbe6216eaf1b98`, balanced long set `00aa6028c00c6d21d2fd7c3357eccb9062ecf0c4255581c154ff16c3eb69b428`, second dev project `8c678c8b0bc4b3e11d30d45d541ed69a80839a8c12ff548aa646f9ed7f94b311` and `4ef2e38135277ade5992acb30611acdb052a0261f73aaa41e17ad07e8a5c1084`; Stockledger greedy `9ef7c4827bf097c93e727609119bd82957eeaaebc82ce1a8e763ff9197f5277d`, seed 1 `c6bdef1db7f9d731d935c490e52ce471bceebaae7175e75aef6f8b9736d2a71a`, seed 2 `4607d406c260f340ea641a090c0559b329880145c775661fb4233f2a1107e4a7`, seed 3 `ca5367eb89fe5b68a39d9f8bbbcc7ba00a8682b5a41dae502749c78909c3db56`, their grades `e002740d31020778d0c79e7d56457813e8fcd27f5d74a8d7f2d1786e9cdac0f1`; training history `2a025bc764f3542d94d68c8769dea7548123687ea74abd64780d9c0d459c8c8b`; the choice rule `91e988690f8526140c46632ad663e7cda77c3d6cb6a3bc5a2097fbbc1feb735f` and its report `e00de6e6d3366f0dc963d70958b75120f707d49bd3f6ab03fba7a07938fa3366`; on the Studio, the fourth phase's reader `f0431298314a0d3a1f6e6a11e618932377a944219549f668539dd42bbe93be9b` and state translator `b6729773217d66eacd6fbebc23ab56840844ed31520cc0fad6cd5106ce98d536`.
