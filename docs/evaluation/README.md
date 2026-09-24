# Evaluation and evidence

[Current status](../STATUS.md) is the reader-facing summary. This section keeps
scientific results, diagnosis and planned comparisons separate from engineering
checks and transport timings.

- [Recorded recall misses](MISS_DIAGNOSIS.md): prior outcomes, controls and what their evidence does not establish.
- [Evaluation admission](EVALUATION_READINESS.md): plan completeness is not permission or a passing experiment.
- [Duo comparison design](DUO_DRIFT_COMPARISON.md): frozen-baseline and declared-channel requirements.
- [Project fixture](DUO_DRIFT_PROJECT.md): the proposed disposable comparison task, not a qualified result.
- [Receiver prefill cost](PREFILL_COST_RESULT.md): SUPPORTED, memory prefill 4.59 times cheaper with accuracy held.
- [Reverse recall confirmation](REVERSE_RECALL_CONFIRMATION.md): PARTLY, 16 of 24 against an 18 bar.
- [Joint code task](JOINT_CODE_TASK.md): NOT SUPPORTED, identifiers did not cross whole.
- [Appointment selection ablation](APPOINTMENT_SELECTION_ABLATION.md) and [forward follow-up diagnostic](FORWARD_FOLLOWUP_DIAGNOSTIC.md): exploratory.
- [Held-out protocol](HELD_OUT_PROTOCOL.md): owner-run hidden names and seed.
- [GLM own-cache gate](GLM_OWN_STATE_GATE.md): PASSED, own rows plus own recurrent state answer as text does.
- [DeepSeek V4 Drift gate](DSV4_DRIFT_GATE.md): PASSED, its own compressed rows answer 24 of 24 against native text's 23.
- [Qwen own-cache gate](QWEN_OWN_STATE_GATE.md): PASSED, 21 of 24 outputs identical to text with its own state, 12 without.
- [Causal loop development runs](CAUSAL_LOOP_DEVELOPMENT.md): exploratory, GLM 12 of 12 once it reads its question after the memory.
- [Translated recurrent state development results](TRANSLATED_STATE_DEVELOPMENT.md): exploratory, Qwen's translated state lifts GLM from 20 to 24 of 24; the forward live gap sits in GLM's live latents, not in the loop.
- [DeepSeek V4 translators](DSV4_TRANSLATOR_DEVELOPMENT.md): exploratory, DeepSeek's cache pages decode; linear translators from its grouped entries answer 2 to 3 of 20.
- [Causal recall confirmation](CAUSAL_RECALL_CONFIRMATION.md) and [joint code causal confirmation](JOINT_CODE_CAUSAL_CONFIRMATION.md): preregistered, with text arms.

Historical scores and failed improvement criteria remain unchanged. Do not
publish raw private activations, answer keys or scorer output, or let evaluated
models access them. Preserve denominators, failed runs, uncertainty and costs.

## Repository-text corpus reproducibility

`scripts/live/build_corpus.py` can use repository prose as corpus input. Moving
or editing documentation changes both selected paragraphs and source labels;
an unchanged random seed does not reproduce a previous corpus from a new tree.
Reproduce a historical corpus from its exact source checkout and pinned
tokenizers, then verify the existing artifact digest. Do not silently regenerate
training or held-out artifacts after a documentation change.

This cleanup changes no existing corpus, token IDs, checkpoint, translator,
preregistration or scorer output. The pre-cleanup public source remains available
at commit `e334afc589bee2386b9ad41732af447dce985605`, and the corresponding
development source was privately archived before edits.

The standalone handoff was subsequently removed from corpus inputs; its last
public source snapshot is `daa6c90645cfed7fe5aed1fd8189b2dd066a3654`.
Existing calibration and held-out artifacts remain unchanged; use their original
source revision and hashes rather than rebuilding them from current documentation.
