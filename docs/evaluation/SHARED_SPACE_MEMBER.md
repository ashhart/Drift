# Shared space: Qwen joins with one command

Status: engineering gate, PASSED on 23 September 2026. The command fitted Qwen's maps into and out of the shared space and ran the drop-in gate on GLM-to-Qwen through the shared space, which answered as well as the direct pairwise translator it replaces.

## Question

Can one command add a model to the shared space and give a translator that works, without a pairwise fit? `scripts/live/hub_add_model.py` and `drift/translate/hub.py` are described in [the shared-space guide](../guides/SHARED_SPACE.md).

## Setup

- The hub is the reduction inside `rows_g2q_code.npz`: GLM-5.3's latents from 11 layers, 5,632 values per token, reduced to 2,048 principal directions. GLM is the anchor.
- Qwen's training pairs were the 1,919 windows the contextual reader trained on: this repository's Python files and 1,000 standard-library windows, GLM's latents against Qwen's own full-attention rows (12,288 values per token), paired where tokens end on the same character. 201,495 pairs were sampled evenly over the windows. The 4 short test contexts (4,950 pairs) were held out.
- The gate was the drop-in's 32 short-context questions with Qwen's code-fitted state translator, the same one the direct translator was gated with.

The command, run on the Studio:

```bash
python3 scripts/live/hub_add_model.py --name qwen --anchor out/hub/glm --features out/mix_taps \
    --anchor-features out/mix_latents --val-features out/code_qwen_val --val-anchor out/code_glm_val --out out/hub/qwen \
    --test-from out/hub/glm --test "python3 scripts/live/studio_qwen_state_gate.py --items out/project_A_items.json \
    --out out/hub_gate_A.json --max-new 160 --budget 16384 --glm-latents out/project_latents \
    --glm-rows {translator} --forward-state out/state_g2q_code.npz --arms text,none,t_rows,t_rows_state"
```

It fitted both maps in 208 seconds on the Studio's CPU, then ran the gate in 598 seconds.

## Fit on the held-out windows

| Map | R-squared |
| --- | --- |
| Qwen's rows into the hub (encoder) | 0.576 |
| hub to Qwen's rows (decoder), which is GLM to Qwen | 0.549 |
| Qwen's rows through the hub to GLM's latents | 0.528 |

The same windows scored the same way:

| GLM to Qwen | With the spread restored | Without it |
| --- | --- | --- |
| direct pairwise translator (`rows_g2q_code.npz`) | 0.578 | 0.624 |
| through the shared space | 0.549 | 0.611 |

Restoring the spread lowers R-squared on purpose: it trades squared error for the rows' natural scale, which Qwen answers from better.

## The drop-in gate

The 32 short-context questions:

| Arm | Correct |
| --- | --- |
| text | 32 |
| no memory | 1 |
| GLM's rows through the shared space | 27 |
| GLM's rows and state through the shared space | 27 |
| GLM's rows by the direct pairwise translator | 27 |
| GLM's rows and state by the direct pairwise translator | 26 |

The shared-space translator did as well as the pairwise one it would replace: level from rows alone and one question better with the state. These counts were rescored on 25 September with `drift/eval/answer_match.py`; the first scoring gave the shared space 28 and 29 and no memory 4, by crediting answers that only repeated the question's message or wrote 400 for a default of 40. Both are linear maps. The contextual reader with its fine-tuned state answered all 32 ([DROPIN_REAL_PROJECT.md](DROPIN_REAL_PROJECT.md)), and a hub pair can be the linear start such a reader trains from.

## What it cannot show

This is one member added to a hub anchored on the model it pairs with, so the GLM side of the pair is the anchor's exact projection. Two non-anchor members composed through the hub, such as Qwen to DeepSeek, would carry both members' errors. The recurrent state still comes from a per-receiver state map.

## Records

SHA-256, on the Studio: Qwen's member `4c2cfdca1f0564b1d3a490b7c7e643f0f9ada5be0349ff7d271dd2a9909300fe`, GLM's anchor `c222c99c1cf51730cc5b469430df8912d0b0fe5cb9c1367f5f2d94836c0cfc9a`, the paired translator `abe8204837f61c643af7a85bc2cc1895ed6849f4c436a2a6c6d0c1be33441d96`, the state translator `9739a16f2ae35ec86227749a84f2c70f7bfada6ae34f8bc461e06e96c9d0bb07`, the direct translator `0108973bda7e5e254131b5580fb2432d72e871c8270d635a4508007c0325b259`, the items `797056637146440947bc102acd7fcf49923df3930f83c3bbf7b77397012294e6`. Private local copies: the gate through the shared space `0fd73b43e26001de3aa71ea603a88307a1fedb7203a7112159467134c349a241`, the direct translator's gate `65379f7cf8256f4aeb490837a311b598ebb344693d6c67c956b8ca73474159dd`, the member report `a6fa2d953369e60005e926326240aa607b89d4d30fd1997cff3887f8685cc40a`.
