# Translated recurrent state, development results

Status: exploratory, 23 September 2026. No claim rests on these runs; a preregistered confirmation must follow on passages the translators never saw.

## What was built

GLM's 34 KDA layers keep a recurrent state that attention rows alone cannot fill. The [own-cache gates](GLM_OWN_STATE_GATE.md) showed both models answer as they do from text when given their own state. Between the two models the state has to be built from what arrives:

- `glm_capture_hidden.py` recorded GLM's KDA-layer inputs and MLA latents over 380 training and 41 validation passages, on an eager capture server. `studio_tap_passages.py` recorded Qwen's full-attention K/V features and linear-attention inputs over the same passages.
- `fit_state_translator.py` paired tokens that end at the same character, reduced the sender's features to 2,048 principal directions, and fitted one ridge map per receiver layer. Qwen to GLM reached a mean validation R-squared of 0.31 over 34 layers. GLM to Qwen reached 0.35 over 36 layers. Both used 37,275 training pairs and 5,908 validation pairs.
- On the Sparks, the connector writes a head-only exported state and advances it by the translated layer inputs with GLM's own projections, short convolution and chunked delta-rule kernel.

## Qwen to GLM on the gate questions

The 24 questions of the [GLM own-cache gate](GLM_OWN_STATE_GATE.md), memory placed in a reserve of exactly its length, one copy, the prefill stopping at the reserve's end.

| Arm | Answer present | Output identical to text |
| --- | --- | --- |
| text | 23 of 24 | |
| none | 0 of 24 | |
| GLM's own rows | 22 of 24 | 2 |
| GLM's own rows and own state | 24 of 24 | 17 |
| Qwen's translated rows | 20 of 24 | 3 |
| Qwen's translated rows and translated state | 24 of 24 | 7 |

The translated state fixed all four misses of translated rows alone: two refusals that said no such document was in memory, one wrong person, and one vague answer. Those are the failures the missing state was diagnosed to cause.

The gate questions come from the first 12 validation passages. Validation passages set the translators' gains, so this is not a held-out measurement.

## Check of the compute path

GLM's state rebuilt from its own captured inputs answered all 20 ASCII gate questions, as its exported state did, with 10 of 20 outputs identical to the exported state's. See [GLM recurrent state](../reference/glm/GLM_RECURRENT_STATE.md).

Results SHA-256, private local copies: translated gate `34390a837f41d3813725803bcd3462c2eea05075648b0ece5882b1983df76352`; compute-path check `4dcc67b61cde9251725bab688a2f1e08ebe93796bfd73a8651161294baabe3ea`.

## GLM to Qwen, the forward direction

Qwen's follow-up in the loop, naming the shop GLM recommended, stayed at 2 of 24 in the causal confirmation. Offline tests on the 22 cases of the [forward diagnostic](FORWARD_FOLLOWUP_DIAGNOSTIC.md), on the Studio, separate the causes. GLM's latents came from GLM reading each case, and the GLM-to-Qwen state translator used ridge 0.1, mean validation R-squared 0.40.

GLM's message placed as a contiguous span inside Qwen's framing system prompt, before the question:

| Arm | Loop wording | Plain wording |
| --- | --- | --- |
| text | 22 of 22 | 22 of 22 |
| Qwen's own rows | 22 of 22 | 22 of 22 |
| GLM's translated rows | 2 of 22 | 11 of 22 |
| GLM's translated rows and translated state | 19 of 22 | 22 of 22 |

The same cases in the loop's own layout, after Qwen's conversation and its roughly 200-token answer, with the loop's wording:

| Arm | Memory right after Qwen's answer | After a line "My partner's message, from our shared memory:" |
| --- | --- | --- |
| text | 9 of 22 | 18 of 22 |
| Qwen's own rows and own state | 8 of 22 | 18 of 22 |
| GLM's translated rows | 5 of 22 | 0 of 22 |
| GLM's translated rows and translated state | 17 of 22 | 22 of 22 |

Two findings. The loop's layout capped even text at 9 of 22, so its follow-up measure never tested the channel fairly. With a framing line that carries no content, translated rows plus translated state named the recommendation in 22 of 22 against text's 18. The state, not the rows, carries it.

Live loop development runs, the 12 scenarios of the [causal runs](CAUSAL_LOOP_DEVELOPMENT.md), Qwen's follow-up:

| Run | Qwen names GLM's shop |
| --- | --- |
| d3, causal, no forward state | 2 of 12 |
| e3, state advanced as each tap arrives | 0 of 12 |
| f3, state held until the follow-up | 1 of 12 |
| g3, as f3 with GLM's rows in their own block of positions | 2 of 12 |
| h3, as g3 after the framing line | 5 of 12, plus one partial "The Tin" |
| i3, as h3 with a translator fitted on GLM writing each passage as its own reply | 3 of 12, plus one partial "The Tin" |

Every shop Qwen named in h3 was right; the misses said "unknown". The offline framed test reached 22 of 22, so a gap remains between offline and live.

## Why the live forward runs trail the offline test

The 12-case comparisons in this section and the next are superseded by the [48-case set](#a-48-case-forward-set) further down: at 12 cases a difference of two or three answers is within noise. Only the large gaps below, such as reading against live latents, survive it.

The offline test gave Qwen GLM's latents from GLM reading each case as a document. Live taps differ in two ways. They come from GLM writing its reply inside its own chat. And the connector sends every row after the memory reserve, so about 144 rows of GLM's own prompt (the end of its system prompt, the user's instructions with the coffee-shop list, the assistant opener) arrive before the reply. The instructions end with "If you truly recall no appointment, finish with RECOMMENDATION: unknown."

The 12 live messages of i3 were rebuilt offline in Qwen's live layout (its conversation, its answer, the framing line, GLM's block, the follow-up), with GLM's latents exported from each context in turn. Scored against the shop GLM recommended (GLM wrote "unknown" in one case):

| GLM's latents | Translator | Text | Translated rows and state |
| --- | --- | --- | --- |
| reply read as a document | reading, ridge 0.1 | 12 | 10 |
| reply as GLM's own turn after a generic request | generation | 12 | 8 |
| reply in GLM's live prompt, reply rows only | reading | 12 | 3 |
| reply in GLM's live prompt, reply rows only | generation | 12 | 4 |
| GLM's live prompt tail and reply, as the connector sends | reading | 12 | 8 |
| GLM's live prompt tail and reply, as the connector sends | generation | 12 | 8 |

Reading latents came from `export_passages.py`, generation latents from `glm_export_generation.py`, and live-prompt latents from `glm_export_live_block.py`. Padding the reserve with placeholder rows, as live does, changed neither live-prompt result.

Qwen reading the same text never missed. What changes is GLM's latents. For the same reply tokens, latents from GLM's live prompt and from reading differ by about three quarters of their norm per token. Early layers barely move (cosine 0.98 at layer 3, 0.91 at layer 11); the middle layers move most (0.55 to 0.62 at layers 19 to 27). A constant offset explains little of it: centring each message raised the cosine only from 0.70 to 0.71. The generation export is as far from the live prompt as reading is (0.57 to 0.65 in the middle layers).

So the translators were fitted on latents from a different context than the one the connector taps, and the reply rows alone lose most of the answer. With GLM's prompt tail in front, both translators recover 8 of 12. The prompt rows are not simply noise, and dropping them from the stream would lose more than it gains with these translators.

### A translator fitted on the live layout

`glm_export_live_block.py` exports GLM's latents in the live session's prompt layout: the loop's system prompt, a 1,536-row reserve holding another passage's first sentences and then placeholder rows, a sampled user instruction, and each training passage as GLM's reply. 380 training and 41 validation passages, the same ones as before. The state translator fitted on these latents (to Qwen reading each passage after its live conversation and the framing line) reached a mean validation R-squared of 0.44, against 0.40 for the reading translator on reading latents.

On the same 12 live cases, latents from the live layout with the padded reserve, rows from the loop's reading-context row translator:

| Block | Translated rows and state | Translated state alone | Translated rows alone |
| --- | --- | --- | --- |
| GLM's prompt tail and reply | 8 | 1 | 3 |
| reply only | 9 | 1 | 1 |

The live-layout translator lifts the reply-only block from 3 to 9. State alone names almost nothing: the translated rows and the translated state work together. A ridge row translator fitted on the same latents (validation R-squared 0.68 against Qwen's K/V) did worse downstream than the loop's row stack, 7 and 6 of 12, with near-miss names such as "The Tin Cup" and "Linen Cup".

### Live run j3 and its replay

j3 repeated i3 with the live-layout state translator, and the Studio kept every forward tap the loop applied (`--save-taps`). Qwen named GLM's pick in 5 of 12 (4 of 12 true shops; GLM wrote no pick in one case and Qwen said "unknown").

`replay_forward_items.py` rebuilt each follow-up offline from the saved taps, in the loop's own layout. The offline gate then answered from the exact latents the loop received:

| Source of GLM's latents | Matches GLM's pick |
| --- | --- |
| live, the loop's own answers | 5 of 12 |
| live taps, replayed offline | 5 of 12, 11 of 12 answers the same as live |
| the same messages exported in the live layout | 8 of 12 |

The loop's mechanics reproduce offline: positions, rephasing, the framing step and the state advance cost nothing. The loss sits in the latents. Live taps and the export agree closely (median cosine 1.00 at layer 3, 0.89 to 0.94 in the middle layers, the same for the prompt tail and the reply), and that small difference costs 3 of 12. The two differ in the reserve: live GLM reads Qwen's translated rows there and its KDA layers ran over placeholders, while the export reads GLM's own tokens.

### A translator fitted on the connector's own taps

`glm_export_live_taps.py` collects training rows through the live connector: one causal session per passage, the loop's system prompt and reserve sizing, three copies of another passage's translated Qwen rows staged as the first publication, a sampled instruction, and the passage as GLM's reply. 380 training and 41 validation passages; the state translator fitted on them reached a validation R-squared of 0.43.

| GLM's latents for j3's messages | Live-layout translator | Tap translator |
| --- | --- | --- |
| exported in the live layout | 8 of 12 | 8 of 12 |
| the live taps, replayed | 5 of 12 | 4 of 12 |
| the live taps, reply rows only | | 3 of 12 |

The tap translator is no better on live latents. Whatever makes the live latents harder, a translator fitted on unrelated memory in the reserve does not capture it.

### Reproducing live latents offline

`studio_replay_memory.py` rebuilds the memory Qwen published in a run: its prompt read in the loop's chat layout, its K/V tapped and translated by the loop's reverse stack. `glm_export_live_taps.py --items` stages that memory in three copies, as the bridge does, and forces GLM's live reply through a causal session. Its taps match the live taps to a median cosine of 0.99 to 1.00 in every layer, for the prompt tail and the reply, where the export with GLM's own tokens in the reserve reached 0.89 to 0.94. Decoding against prefill makes no difference; what GLM reads in its reserve does.

Two changes to that reserve, measured on the replica with the live-layout translator:

| GLM's reserve | Cosine to the own-token export, layer 23 | Matches GLM's pick |
| --- | --- | --- |
| three copies of translated rows (live) | 0.918 | 7 of 12 |
| the same, plus the translated state, as the reverse gate sends it | 0.919 | 7 of 12 |
| one copy of translated rows | 0.936 | 7 of 12 |

The replica itself scored 7 where the live replay scored 5, with 9 of 12 answers the same. At 12 cases a swing of two or three is noise, so these runs cannot rank small changes.

### A 48-case forward set

`glm_export_live_taps.py --generate` runs the GLM half of the live loop offline: the memory staged as the loop stages it, the causal session, and GLM writing its own reply while the connector taps. On j3's twelve scenarios it wrote different words but the same recommendation in 11 cases; in the twelfth, live GLM wrote no recommendation and the offline run did. `appointment_items.py` then gave 48 fresh-vocabulary scenarios (seeds 1001 to 1004, balanced), and GLM recommended the true shop in 44 and "unknown" in 4. Qwen answered in the short layout (its system prompt, the framing line, GLM's block, the question), which gave the same score as the full live layout on j3.

| Arm | Named GLM's pick, of 48 |
| --- | --- |
| text | 47 |
| Qwen's own rows and state | 47 |
| translated rows and state, reading translator | 31 |
| translated rows and state, live-layout translator | 29 |
| translated rows and state, tap translator | 28 |
| translated rows alone | 4 |
| translated state alone | 4 |

The three translators do not differ at this size. The forward channel carries GLM's pick about 60% of the time against text's 98%. The differences between the 12-case runs above were within noise; what limits the forward direction is translation fidelity for one chosen name among several similar ones.

An answer-level correction of the state translator, trained on general passage questions from the tap latents (`studio_train_memory_answer.py`, then named `studio_train_state_answer.py`), left the KL flat around 0.06 and the j3 replay at 5 of 12; those questions were already answered 59 of 60 from memory, so they carry little signal for this task.

Results SHA-256, private local copies: offline gates `0d09f8be4473c4b37db46925c49bde784c86f932fd09c448cb7ce40e1269e939` and `286dc8575454cbc21a47695531db61bd1bee06addb86a599d5b6ec00eab88d79`; runs e3 `8fc61cfd1992a9a5d79d27ae210f98f9ec5c40de1001c46b40384892e8ea03fe`, f3 `e7b55ce225ce04692dbb9f965454da165917752f8c072acfe1596546bc6607cc`, g3 `1edd348f70967b580a03d3a33f7dc6bd2a0d10635609aeb01331946c0d9068b5`, h3 `3a67ced96bf9768951f54f8cf5ee37061d5fab6ccc5476dc060a2cacde76ad11`, i3 `743578e209d4e049a8c00b491516a06f9cc0819aeb09bfc91cd1a722364a05ee`, j3 `62a172f86bbbdd208726bc0f710696e375e833bb24d73c0e9615f22c5f4bcd4b`; live-message rebuilds, reading `19b3daeca44d4597d353a5302901097d4e481b016a57e878fc7ef1971c850509`, live prompt block `c5bf2094d8431fc7c17cf1c0118354ca733320cb584521b0192d32d93704c92f`, reply only `200f57bb580212240e36ae86b73471c6360c424f8f323f0a4b167e72ec6f97f9`, live-layout translator block `b5aa999ddff6da5accec037c2c4979ab947a8b82989c40ba3576c835429840e6` and reply `bc8e191668afb5a0a46e89222929add7233fe7044fcb190def7e17e0ba9b4e87`; j3 replay `751eec3fe7fd0e110e4e6bd590d4b361f58e93bad78608528fee94542d5515b8`, j3 export `4a48ea19a8c9d0a78c8f3ac48ef2d28233d4a0af0f6670100c1fef500bce813a`; tap translator on the replay `fc0d0685e640523d8d4371586eda05cedf411bea60ca5ce27a31bb54eafeaad2`; replicas, three copies `eccaa0f62302ece285437e036a5e1ade52cb142d984a036f9d14247113ea07d9`, with state `cc0838006355910a2a1f72d1dfd42a7549973efaa512c16566a78f43ca38f21c`, one copy `f4662a6265bc1737bc10ee8a4cea3871556d9a63cde0a348765d0796ec43a13d`; 48-case set, reading `1ee82e325b59124dcfdf9dd21ba4d82a748f12ec288b06920a17e5aaeac22b1b`, live layout `c596694b14fbcb7bedd656827dde0ba42826757216b68b7d3cfac51ae0be9cc9`, tap `7dad52d45829314f9ac056ddc142e3a931f01aa8f38a47e9b3715976bedde35b`.
