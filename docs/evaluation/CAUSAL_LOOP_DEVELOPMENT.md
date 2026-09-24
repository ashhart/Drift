# Causal loop, development runs

Status: exploratory, 22 September 2026. No claim rests on these runs; a preregistered confirmation must follow.

## What changed

The MCDMA loop released GLM and Qwen together, and GLM's session ran without a prefill boundary. GLM therefore computed its whole prompt, question included, before Qwen's first publication landed, and only its generated tokens could read the memory. With `--causal`, GLM's session waits for the first publication, then the Drift scheduler stops its prefill at the reserve's end, so the question is computed with the memory in place. Nothing else changes: the same translator `stacked3_rev`, the same copies, reserve and scenarios.

## Appointment scenarios

12 fresh balanced scenarios from seed 4242, selection `all`, linked arm only.

| Run | Commit | GLM names the true street | Qwen's follow-up names GLM's shop |
| --- | --- | --- | --- |
| a3-1536, concurrent | `c3d1e58` | 9 of 12 | 1 of 12 |
| b3-384, concurrent | `8f615ae` | 9 of 12 | 0 of 12 |
| d3-1536, causal | `e82ac23` | 12 of 12 | 2 of 12 |

The causal run fixed every scenario the concurrent runs missed, including the two where GLM had invented a street.

Qwen's follow-up is unchanged. Its question already comes after GLM's taps, so ordering is not its problem; Qwen's recurrent layers never see GLM's memory, which is the gap the [forward diagnostic](FORWARD_FOLLOWUP_DIAGNOSTIC.md) found.

Report SHA-256, private local copy: `8e5f596baaa88107d71b6a80d124206470102caecd8c49f99018d7ebdfad3ddc`. Run time 4 min 50 s.
