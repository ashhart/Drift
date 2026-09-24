# DeepSeek V4 translators, development results

Status: exploratory, 23 September 2026. First translators from DeepSeek V4's cache to Qwen and to GLM. No claim rests on these runs.

## Reading DeepSeek's cache

DeepSeek V4 Flash keeps two kinds of compressed attention cache. Its ratio-4 layers (21 of them) keep one entry per group of four tokens, with an indexer key per entry. Its ratio-128 layers (20) keep one entry per 128 tokens. Every layer also keeps a sliding window of the last 128 tokens, which the raw-row connector does not capture.

`drift/translate/dsv4_pages.py` decodes the pages. Per vLLM's DeepSeek V4 cache code, a compressed page holds every entry's 576 data bytes first (448 float8_e4m3fn values, then 64 bfloat16 rope values), then every entry's 8 scale bytes (7 power-of-two exponents, one per 64 values, and a zero pad byte). An indexer page holds 128 float8 values per entry, then one float32 scale per entry. The raw-row connector returns fixed-size chunks of these pages, which is why the captures cover whole 256-token pages.

All 421 captures (380 training, 41 validation passages) decode: 62 tensors each, every pad byte zero, every value finite. A round trip through `encode_entries` and `entries` is exact for the rope values and within float8 rounding for the rest.

## Features and translators

`dsv4_token_features.py` gives each passage token the 448 dequantized values of its group's entry from all 21 ratio-4 layers, 9,408 values per token, and the four tokens of a group share them. `fit_state_translator.py` then fits ridge maps, paired by shared character ends as for GLM and Qwen, 35,935 training and 5,773 validation pairs.

| Map | Validation R-squared |
| --- | --- |
| DeepSeek to Qwen, linear-attention inputs (state) | 0.14 |
| DeepSeek to Qwen, full-attention K/V (rows) | 0.33 |
| DeepSeek to Qwen, state, a separate map per position in the group | 0.11 |
| DeepSeek to Qwen, rows, a separate map per position in the group | 0.34 |
| DeepSeek to GLM, KDA inputs (state) | 0.11 |
| DeepSeek to GLM, MLA latents (rows) | 0.22 |
| for comparison, GLM to Qwen, state | 0.40 |

DeepSeek's state fit is weakest on Qwen's early layers (0.05 to 0.11), which follow individual tokens most closely. GLM's reaches 0.50 there. One entry stands for four tokens, and a linear map cannot pull them apart: per-position maps did not help.

## Qwen answering from DeepSeek's cache

The Qwen own-cache gate's 24 questions; 20 have DeepSeek features (the other 4 passages are not ASCII). On those 20, text answered 19 and Qwen's own rows and state 20.

| Arm | Answer present |
| --- | --- |
| DeepSeek's translated rows | 1 of 20 |
| DeepSeek's translated state | 0 of 20 |
| DeepSeek's translated rows and state | 2 of 20 |

Most answers say the text does not mention the thing asked about. Some carry fragments: "Stefan" for Stefan Sato, "120 kilograms" for 126, "651" for 6551. A weak signal gets through, not enough to answer.

## GLM answering from DeepSeek's cache

The GLM own-cache gate's 24 questions, the memory built by `translate_passages.py` from DeepSeek's features with the two DeepSeek-to-GLM maps, placed in a reserve of exactly its length. On the 20 items with DeepSeek features, text and GLM's own rows and state each answered 20.

| Arm | Answer present |
| --- | --- |
| DeepSeek's translated rows | 2 of 20 |
| DeepSeek's translated rows and state | 3 of 20 |

GLM gets closer than Qwen, with near misses such as "231 kilograms" for 271, "23:07" for 23:23, "purple" for violet and "Dubois" for Astrid Dubois, but it is not usable.

## Training the translators on answers

`studio_train_memory_answer.py` trains on top of the two fixed DeepSeek-to-Qwen maps: a low-rank correction of the translated rows (rank 32), a low-rank correction of the state inputs (rank 32) and a per-layer scale. Teacher: Qwen reading each training passage as text after the framing line. Student: Qwen given DeepSeek's features, translated. Loss: KL between them on the teacher's answer tokens, with gradients through Qwen's own layers. 2,084 training questions over 380 passages; 60 validation questions over the validation passages, scored after the framing line, where text answered 58.

The first attempt diverged (KL rising from 0.5 to 1.6, validation falling to 2 of 60). The reduced features were not standardised, so small parameter steps made large output changes. With each reduced component scaled by its training spread and a learning rate of 3e-4:

| Update | KL | Answered from DeepSeek's memory, of 60 |
| --- | --- | --- |
| 0 | | 10 |
| 50 | 0.47 | 15 |
| 100 | 0.36 | 18 |
| 150 | 0.28 | 22 |
| 200 | 0.25 | 19 |

Training on answers roughly doubles what Qwen recalls from DeepSeek's cache, still far below text. It ran 200 updates of 4 questions in 50 minutes on the Studio, peak 134 GB. The validation passages also set the fixed maps' gains, so this is a development measurement.

## Next

Linear maps from grouped entries fall short, and answer-level training closes part of the gap. More training passages from DeepSeek (380 so far, from a corpus of 780) and longer training are the next steps. Beyond that: a reader that sees the sequence of entries rather than one group at a time, and DeepSeek's sliding-window rows, which are per token but cover only the last 128 tokens.

Results SHA-256, private local copies: Qwen gate with DeepSeek arms `8b334e41152614d984e6cb8bcee06f6d01718b1a3a3c84623dcec2b80f9a8042`; GLM gate with DeepSeek memory `219b1b3e36c53a118ee9ecafe6ea1172e08f54c78ca9f83eae255ccfff418ac7`; answer-level training history `a4f96a77b96590b74a59ac1b92cee185a5c902a169547b989b778f8d4e53265d`.
