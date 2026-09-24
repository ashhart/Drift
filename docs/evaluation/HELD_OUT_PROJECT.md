# Held-out project

Status: code questions FAILED, def lines FAILED for rows and state (rows alone met the bar), coding task INVALID. Run 23 September 2026 with the averaged translator `reader_answer3fg`.

## Question

Does Qwen answer from GLM's translated cache of a project it has never seen, within one question of Qwen reading the same files as text? The bar: within one question on 30 or more code questions, on 25 def lines and on a coding task scored by hidden checks, on each of 3 seeds.

## The project

Files from four installed packages, none of them in any training set or checkpoint choice:

| Context | Files | Characters |
| --- | --- | --- |
| h1 | `packaging/direct_url.py` | 11,687 |
| h2 | `idna/codec.py`, `filelock/_util.py`, `packaging/_elffile.py` | 18,123 |
| h3 | `packaging/_manylinux.py`, `packaging/_parser.py` | 22,721 |
| h4 | `packaging/pylock.py` | 35,270 |
| h5 | `idna/core.py` | 32,601 |

`project_questions.py` wrote the questions from the source with `ast`, and `gate_items.py` framed them as the drop-in does: 40 code questions (which function raises a given message, parameter defaults, constants) and 25 def lines drawn with seed 0. The coding task asks for iniconfig's `_parse.py` from its specification and the package's other files (`iniparse_task.py`), graded by 33 hidden checks recorded from iniconfig 2.3.0.

The translator was fixed before the run. The training data were this repository's Python files, 1,000 standard-library windows and 300 Markdown windows from other packages' documentation. A search of every training set's file headers found none of the four packages; one Markdown window has a "Packaging" section heading in another project's changelog. The translator was chosen by the dev gates and Stockledger alone.

GLM read each context in its reading frame on the Sparks (`export_passages.py`). Qwen ran on the Studio with the drop-in's framing and a 16,384-token dense budget. Arms: text, no memory, GLM's translated rows, and GLM's translated rows and recurrent state. Seed 0 is greedy; seeds 1 and 2 sample at 0.7.

## Results

Code questions, of 40:

| Seed | Text | Rows and state | Rows | No memory |
| --- | --- | --- | --- | --- |
| 0 | 37 | 34 | 32 | 9 |
| 1 | 37 | 35 | 29 | 8 |
| 2 | 37 | 33 | 35 | 7 |

Def lines, of 25:

| Seed | Text | Rows and state | Rows | No memory |
| --- | --- | --- | --- | --- |
| 0 | 24 | 23 | 23 | 0 |
| 1 | 24 | 22 | 24 | 1 |
| 2 | 24 | 23 | 24 | 0 |

The code questions miss the bar by 2 to 4 questions with rows and state, and by 2 to 8 with rows alone. On def lines, rows alone stay within one question on every seed; rows and state fall two short on seed 1.

The coding task is INVALID. Every arm scored 0 or 2 of 33 on every seed, text included. Text's answer reached the 3,500-token limit partway through a module full of planning comments, so no arm's module was complete. Two fixes did not make it usable, each tried on the text arm alone:

- More room. At 12,000 answer tokens the gate first crashed: Qwen's selector cache concatenates its raw index keys once per token, nothing reads them under a dense budget, and the lazy graph grew until Metal ran out of shared events. With the cache evaluated every 256 tokens it ran through, and text wrote 49,167 characters, mostly comments restating the specification's rules, and never closed its module.
- An instruction to write the code directly, with no planning in comments. Greedy and sampled, text wrote 31,470 and 33,622 characters within 8,000 tokens and again scored 0.

Only the no-memory arm, which never sees the specification, finishes a module. With thinking turned off, as the drop-in framing does, Qwen works through this specification's rules in comments and does not get to the end. The run says nothing about the translator until the task is one text can complete, for example with thinking allowed.

## Where the translated cache misses

Across the three seeds, the misses sit on five questions. All ask which function raises a message, and the answer is a long method: `_from_dict` in `direct_url.py` and `pylock.py`, and `_parse_marker_var` in `_parser.py`. The translated cache quotes the message correctly and names a neighbouring function, usually `__init__` or `validate`. In `direct_url.py`, the three messages it misses sit 142 to 239 GLM tokens below the `def _from_dict` line, and the two it answers in the same method sit 67 and 117 below; the misses in `pylock.py` sit 301 and 343 below. The `_parser.py` miss is only 107 tokens down, so distance is not the whole story. In every case the `def` line is inside the reader window that writes the message's rows, so the window is not the cause. The translated rows carry which function a line belongs to less well than Qwen's own rows do, and worse the further the line is from the function's first line.

## What it cannot show

The agent that built the translator also chose these files and wrote the questions, before the run and without tuning on them. The owner-run protocol in [HELD_OUT_PROTOCOL.md](HELD_OUT_PROTOCOL.md) would keep the test set from the agent entirely.

## Records

SHA-256 of private local copies:

- items: code questions `72b12b3d67bba0831b00740d2b3ecbbed677716dee7582316d84a26dcf243124`, def lines `1b25793af3b82b6798fd3a899ae1e4f5a32d1f4673e6e2a968645f4321a5a825`, coding task `7a49c82132ab4d0eed582cfea9e3a240c93effa3421677175303284a11e59f80`;
- code questions, seeds 0 to 2: `6ecc8fadeb37fb4e9c8ae610f8c473f3016ee4dbc63c55346c2182bcf7dd8e6e`, `e0567c1b368dfb32c502cbd450339430833f3213728518da9099072ab58c9cc6`, `673dbd84da2f99905bebe4254967e1d61c110675e446b0636791a8fe75b588cd`;
- def lines, seeds 0 to 2: `53a03fd980852c963cad5cdc8e2611e54339ff99440e3328e4eb3f85ded686d4`, `8e56dac49fd54084a1dbadc057a02abd8ae548976da42013a6e8b3c067f8cf24`, `3c802fa85cdc78528fbd777f4658350dee0baae8e8be99775aca5fa6835563b3`;
- coding task, seeds 0 to 2: `8118bd1c5d65b4d11904c6f49754af19ebcbb932bf223b2e6f202c7ccf217131`, `2ef18127816f6e274362636a8dcb11c35998b206cac83c61854dc40c6df568b9`, `98602588597433e31de7fb0c60ef32257a96717a8a5d8189f249e41ce2354f1b`, and its grades `be6062372bada463b700828985215b4418b96ffeab8dcc0a702fb90e6f390e24`;
- on the Studio, the translator: reader `ad8666de3b335537c3fe3a02588a0916ca20ff0985125360a620322db153fa20`, rows `0108973bda7e5e254131b5580fb2432d72e871c8270d635a4508007c0325b259`, scales `5767846aa886dbc9ef9ce53d3958fa996b9f58d6ef0b5c35b1881617c8b46c2b`, gain `6b8c0774107cab48e0423af0bd573e4b4bc72ccb9b59621a40f7d3296222c593`, state `bee4d090081563a22f0f379262321d811e75a20e7276f1530734fd18570dd549`;
- on the Studio, GLM's exports: h1 `7a079542c14080f06bf8aa69d059b5faecff2906a6eac67f51ef7a71eb218b76`, h2 `0106a52c091b10c3711b5fc9d892db16258ddafcb311e9bd0ff9b61ee628307c`, h3 `d6720c05246b3fe9d49092267f80ee8ec61c29b27369075261a6b7d234759f32`, h4 `7b58befa2d5743926ad8414c25d028f65da3d99235cdc1fec4ea4704e3c14a60`, h5 `0ff84b98a6367ee6f331ccd5a91658600d20333b38ddfd2ecd5a86b7fe542800`, coding task `1664a1df097ccd7ed20508edd8cadd607c53fa61bd2de80e40f59c69f4802233`.

Each seed's code questions took about 20 minutes on the Studio.
