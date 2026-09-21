# Appointment evaluation readiness

The live appointment demonstration remains exploratory: six scenarios were used during development, and none may be relabelled as held out.
The current formal-trial status is **BLOCKED**, independently of Python test results or complete manifest fields.
This work changes no running model, target, connector or live publication path.
An authorized exploratory tester can exercise public development examples without calling them held-out results; this preregistration check is not an obstacle to such a bounded smoke test.
The tester must still use the approved hosts and budget, observe process cleanup, and stop on corrupted activations, stale sessions or unexplained application failures.

## Offline plan check

`scripts/evaluation_readiness.template.json` extends the existing preregistration approach with appointment-specific requirements.
It deliberately leaves approvals, measured evidence and experimental choices unresolved.
Copy it to an experimenter-controlled location and inspect public metadata with:

```bash
PYTHONPATH=. python scripts/evaluation_readiness.py /path/to/public-preregistration.json
```

An incomplete or malformed plan returns `BLOCKED` and exit 2.
Complete metadata returns `PASSED`, exit 0 and `scope=plan_completeness_only`.
Every result sets `launch_authorized=false` and `publication_authorized=false`.
The command never launches a trial, opens a hidden dataset or evidence file, checks approval authenticity, or qualifies a scientific stage.
The SHA-256 in its output identifies the exact input bytes, including whitespace.
This is an offline prerequisite, not a substitute for host enforcement or an independently verified admission decision.

The generic `scripts/validate_manifest.py` remains the broad run-manifest completeness check.
The appointment plan names the existing `drift.eval.metrics.paired_bootstrap` calculation: first average all repeats within each scenario and arm, then bootstrap paired scenario means.
Repeated decodes are not independent scenarios.
The scenario count, effect threshold and precision or power justification must be frozen before evaluation.
Two scenarios is only the estimator's mechanical minimum, never adequate evidence by itself.
Five repeats is the SERIES pilot floor, never a guarantee of sufficient precision.
The proposed appointment check does not classify this experiment as formal E2 or apply E2's separate 100-secret advancement rule.

## What the fields mean

Each digest points to an immutable artifact or experimenter-held evidence record, not a claim that the referenced record has been inspected.
The source commit does not replace the runner digest, which must cover the candidate code and any changes not yet committed.
Model and toolchain records must pin exact revisions and actual files, not mutable model names.
The injection record includes copy counts, layer choices, publication selection and all other settings tuned on development scenarios.
The hidden-set commitment and development-exclusion record reveal neither the scenarios nor their answer keys.
Do not put private evidence, answer keys, raw scorer outputs or private paths in the public plan.

Budget caps cover all arms, repeats, retries and startup, including local token-conditioned mail.
An owner-approval digest is not permission by itself: the controller must verify that approval covers these exact limits and enforce them.
Timeouts, crashes, rejected work and partial completion remain in denominators as unsuccessful attempts.
Leakage, tampering, corrupted activations, stale sessions and schedule violations invalidate the experiment and stop scoring.
Failed deliveries and missing application receipts remain visible rather than disappearing from latency summaries.

Isolation records must come from denied-access tests under the actual model-worker and coding-agent identities, including tool, filesystem, shared Git-store and network paths.
Keeping a directory outside the repository or asking a model not to read it does not establish isolation.
The independent experimenter must check those records without returning hidden material or scorer stdout to either model or coding agent.
The plan checker intentionally cannot grant permission based on syntactically valid digests.

## Schedule and remaining gates

`exploratory_appointment` permits the existing asynchronous schedule only when labelled `exploratory_async`.
It does not establish M2's prior-epoch schedule.
`formal_m2` requires prior-epoch evidence, all-layer pinned snapshots before either worker publishes, upstream M-1/M0/M1 evidence and the solo-A, solo-B and text-pair arms required by the engineering specification.
Even complete declarations do not prove that the live worker implements this schedule.

Before a fresh formal trial, the live path still needs independently checked session, sequence and payload-length validation before forward cache application, acknowledgement only after successful application, and run invalidation rather than continued generation after reverse-publisher failure.
Bounded process cleanup, independent scorer isolation and budget enforcement must be verified on the deployed execution path.
These are technical admission gaps, not fields that can be filled in to make the experiment valid.
A new hidden run must not begin merely because this offline checker passes.

The SERIES vocabulary remains separate from engineering status: demonstrated improvement, failed challenge, inconclusive and invalid experiment.
No current appointment result establishes general cross-family performance or an M0-M6 advancement.

## Choices still required

The owner and independent experimenter must choose the approved compute and storage caps, fresh scenario count, repeats, primary baseline, worthwhile effect and precision plan.
They must also decide whether the next trial is a narrowly labelled asynchronous appointment study or a formal M2 study after the prior-epoch implementation and upstream gates are qualified.
No hidden scenarios have been generated or read and no GPU/model jobs have been started by this change.
