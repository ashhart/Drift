# Review of the adoption roadmap

Reviewed on 2026-09-20 against source `032f912fbcecc030ab925eed0722fb745663ccec`,
the current engineering records and the primary sources linked below.
This is a research and product assessment, not a preregistration, new result or
authorization to run models. Claude retains the deployment handoff.

The economic emphasis is right. A useful target is lower total cost per correctly
completed task, at a predeclared quality margin, for a workload that actually needs
different model families. An open standard can follow a convincing implementation;
it is not a prerequisite to every pilot adoption. Scientific novelty, adoption and
shipping readiness are separate claims.

## Economic evidence

Keep normal stock Duo as the primary project baseline already requested by the owner.
For a separate specialist-handover benchmark, compare full rereading, a competent
summary, selective text retrieval, and translated memory under matched task conditions.
Give text baselines their supported prefix caching and record cold and warm states.
vLLM already avoids repeated prefill for matching prefixes, so a cold-reread-only
comparison would overstate the opportunity. This does not mean an A cache is usable
directly by a different family B. Measure B's first arrival separately from later
arrivals that can reuse B's own cached prefix. [vLLM prefix caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/).

Report both incremental handover cost, conditional on A already doing useful work,
and complete workflow cost including A's ingestion. Count capture, projection,
serialization, transport, selector work, receiving-cache construction, receiver
prefill and decode, tools, retries, failures, cleanup and retained cache memory.
Report p50/p95 arrival-to-first-useful-output and time-to-correct-completion,
device-seconds by hardware type, measured memory/bytes and energy if instrumented.
Use explicit rates if converting device time to money; tokens alone are not cost.
Total cost divided by successful tasks must include the costs of failed attempts.

Freeze an acceptable quality-loss margin and assess its paired interval separately
from the economic improvement. A non-significant difference is not equivalence.
The 235/240 versus 238/240 lookup aggregates neither establish equal quality nor
measure economics, project completion or task-generalization. Preserve the formal
Test A FAILED verdict. Keep wrong-memory and no-link controls for attribution:
the proposed benefit needs to depend on the sender's relevant information.
[Causal audit of relayed caches](https://arxiv.org/abs/2608.04893).

## The 100k / 200 ms example is a hypothesis

Neither 200 ms nor 40 seconds has been measured here. With the configured forward
layout in scripts/live/livelib.py, one emitted Qwen row carries
12 layers × 2 K/V × 2 heads × 256 dimensions × 2 float16 bytes = 24,576 bytes.
Hypothetically transferring 100,000 such rows uncompressed means 2,457,600,000 bytes,
or 2.4576 GB, before metadata, selector keys or temporary buffers. Moving only that
payload in 200 ms needs 12.288 GB/s, or 98.304 Gbit/s, before computation and overhead.
This arithmetic is not a measurement, a hardware guarantee or a supported live run;
100,000 source tokens need not produce 100,000 reader rows, especially with fan-out.
The current publication and reserve limits also prohibit treating this as one large
existing transfer. A compact pool, selective transfer or pre-positioned cache may
change the economics and must be measured with its preparation costs disclosed.

## Long context is a prerequisite to that particular benchmark

The missing selector treatment is real: omlx_cache.append_entries writes zero
index keys, while the GLM injection registration describes selector state inherited
from placeholder tokens. These are different approximations. The top-2,048 budget
describes sparse attention selection; it does not by itself establish that all
other cache rows have been evicted. Translate or otherwise train compatible
selector state, then verify actual selection and behavior.

Test around the configured selection threshold before scaling through 8k, 32k
and 100k contexts. Vary fact location, distractors, evidence density, number
tokenization, multiple-hop questions and conflicting sources, with native-cache
and full-text controls at each length. Measure selector recall where supported,
end-task quality, memory and cost. Do not extrapolate short-note performance to
long contexts or change held-out Test A data to develop this capability.

## Provenance and safety should run alongside economics

Separate three requirements: authenticated producer identity, correctly answered
source-attribution questions, and authority to perform an action. An authenticated
malicious writer remains malicious. A learned marker and attention gate are useful
candidate mechanisms, but neither establishes all three requirements.
Bind writer, session, sequence, model/pack and trust scope in authenticated metadata;
enforce permissions in the runtime/tool boundary, not in the model's recollection
of a source label. Test a true foreign-memory off path and bounded gates, conflicting
facts, spoofed ownership, revoked writers and adversarial tool instructions.

Saying a cache bypasses every text filter is too broad: it bypasses checks attached
only to transmitted text, while source/tool admission and output/action controls can
still apply. Inversion and cache-based injection are documented risks in other
systems, but their success rates do not transfer automatically to Drift's projected
pool. Define attacker access and test token recovery, exact sensitive-field recovery,
semantic inference and membership leakage against no-cache baselines, using consented
or synthetic data. An unsuccessful attack is not a general privacy guarantee.
[Shadow in the Cache](https://arxiv.org/abs/2508.09442).

The project already has reference HMAC framing, hard-off attention behavior and
fingerprint checks. They are useful components, not a qualified production security
boundary for the current live file loop. Logs must record enough metadata for audit
without publishing the activations themselves.

## Packs and a standard

Pool/model-pack code and tests already exist, so "none of the three" overstates
the absence of foundations. ModelPack currently checks shapes, finite parameters
and matching pool fingerprints; a metadata description does not itself verify the
runtime checkpoint, quantization or tokenizer. Turn those into mandatory admission
checks with documented versions, positional/selector conventions, source metadata,
precision, lifetime, revocation, failure behavior and negative conformance cases.

A common byte format is not a common learned meaning. To show linear pack enrollment,
freeze the pool and existing packs, enroll a third then fourth family using only each
new pack, and measure both new-writer-to-old-reader and old-writer-to-new-reader
directions on held-out tasks, with old-to-old non-regression checks. Pairwise
qualification still scales with combinations even if trainable artifacts scale
linearly. A DeepSeek store-and-swap read alone would establish only that tested
direction, not universal interoperability. A fourth family is stronger evidence
than another checkpoint from an already represented family.

Propose a small provisional contract now; stabilize it through independent
implementations before presenting it as a standard. Existing vLLM connector APIs
offer an integration route, but upstream acceptance and SGLang compatibility are
not established by this repository. [vLLM connector documentation](https://docs.vllm.ai/en/latest/features/disagg_prefill/).

## Native training and replication

Remove "provenance comes for free" and treat "the translator shrinks to almost
nothing" as a hypothesis. Shared-space objectives can help alignment but do not
automatically encode trust, identity or permissions, and different architectures,
tokenizations and runtime layouts still require adapters. Shared cache alignment
with frozen backbones is already studied, so novelty must be assessed against that
work rather than inferred from the broad concept. [K-V cache alignment](https://arxiv.org/abs/2601.06123).

Replicate the exact frozen 240-question setup separately from a new independently
generated evaluation after weights freeze. The former checks reproducibility;
the latter also tests generalization. Publish permitted code, packs, manifests,
preregistrations and aggregate failures after privacy/licensing review, with a
private path for any necessary sensitive evidence. Publication does not require
releasing raw private activations. Retain version-bump, quantization, malicious-
writer and long-session failures, including unsuccessful attempts and costs.

## Recommended execution order

Finish the existing deployment and bounded reliability qualification. Instrument a
short-context economic pilot immediately while building long-context selector
support and mechanism-level provenance in parallel. Use those gates to earn the
100k-context experiment, and continue the existing normal-Duo project comparison
only after its OMP, source-ownership, isolation and resource prerequisites are met.
Keep a provisional pack contract during development; freeze a claimed interoperable
standard after new-family enrollment and outside replication. Native model training
is a later collaboration proposal, not evidence already obtained.
