# Evaluation and evidence

[Current status](../STATUS.md) is the reader-facing summary. This section keeps
scientific results, diagnosis and planned comparisons separate from engineering
checks and transport timings.

- [Recorded recall misses](MISS_DIAGNOSIS.md): prior outcomes, controls and what their evidence does not establish.
- [Evaluation admission](EVALUATION_READINESS.md): plan completeness is not permission or a passing experiment.
- [Duo comparison design](DUO_DRIFT_COMPARISON.md): frozen-baseline and declared-channel requirements.
- [Project fixture](DUO_DRIFT_PROJECT.md): the proposed disposable comparison task, not a qualified result.

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
