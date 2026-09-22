# Engineering contracts

These requirements govern implementation and evaluation; they do not establish
that a model pair or runtime is qualified. Read [AGENTS.md](../../AGENTS.md),
[current status](../STATUS.md), the [stage gates](../ROADMAP.md#research-sequence)
and the relevant [interface reference](README.md) before changing a component.
The original standalone handoff and duplicated source appendix are archived
outside the working repository and remain recoverable from Git history.

## Model and cache boundaries

Freeze backbone weights and pin checkpoint, tokenizer, configuration, runtime,
adapter, dtype, quantization, layer map, chat template and rotary conventions.
Unknown model layouts fail closed; matching tensor shapes do not qualify an
adapter. Qualify stock-forward parity and same-model prefix import before fitting
translators, including prefill, incremental decode, hard-off identity and cache
content. Required environment skips are BLOCKED, not passes; declare precision
tolerances before observing errors and never widen them to pass a failed gate.

Canonical reference K/V have shape `[tokens, kv_heads, head_dim]`, batch one.
Capture K after native key normalization and before RoPE, and V unrotated.
Translate K/V separately, then apply the receiver's complete rotary policy once;
do not renormalize imported K unless a separate declared sidecar requires it.
Receiver GQA expansion is distinct from cross-family head mapping. Specialized
latent formats must declare their own qualified capture and reconstruction rules.

Identity replacement imports the complete native prefix with exact positions
and no learned projector; foreign-memory append is a separate learned
intervention at selected layers. Append success does not prove native-prefix
equivalence. Position policies must preserve ordering and be trained/evaluated
consistently; token indices from different tokenizers are not interchangeable.

Reference foreign attention places its gate prior inside the shared softmax:
`softmax([native_scores + mask, foreign_scores + mask + log(g)])`.
An exact hard-off gate bypasses foreign attention; zero-valued KV does not remove
its softmax contribution. A learned gate controls interference, not security.

## Ownership, scheduling and transport

Workers own caches, local tokens, GPU work and registered buffers; OMP owns
lifecycle/control. Generic completion APIs do not expose native cache taps.
Publish an immutable complete interval across every selected layer, validate
before applying, project each delta once and reuse bounded receiver-local memory.
Pin one view for the entire model step; reject gaps, duplicates, wrong sessions,
wrong directions, missing layers, nonfinite values and poisoned state.

The reference causal schedule is `A(k) reads B(k-1)` and `B(k) reads A(k-1)`.
Do not mix epochs between layers or wait for a peer inside an attention layer.
Keep source positions, publication sequences and coupling epochs distinct.
After partial failure, restore a consistent checkpoint or start a new session;
never attach a stale packet stream to new model state. Checkpoints include
caches, banks, translator identities, cursors, RNG and pending work/mail.

Registered memory is not automatically coherent remote GPU memory. Require
producer completion, transport completion, receiver GPU visibility, validation
and consumer release before acknowledgement/reuse; keep buffers and registrations
alive throughout. Count copies and refuse unsupported memory types or stale
handles. A byte-exact transfer alone is not inference parity or zero-copy proof.
Use the actual pinned MCDMA APIs, not a proposed shim as evidence of availability.

Wire schemas permit tensors and bounded declared metadata, not task text, token
IDs, answer labels or arbitrary fields. Authenticate and bound frames before
tensor construction; HMAC is not encryption. Use approved protected endpoints
and never unrestricted deserialization. TCP reference transport and MCDMA are
different recorded conditions, never a silent fallback for a native trial.
Test limits, corruption, cancellation, delayed peers, restart and buffer reuse.
See [native operation](../guides/NATIVE_OWNER_EXCHANGE.md) for host-specific rules.

## Training and mailbox evidence

Fit source-layer selection and sidecars on training/development splits only;
freeze them before held-out evaluation. Align supervised rows at shared causal
text endpoints without future context. Behavior distillation compares teacher
and student in the receiver vocabulary, preserving gradients into declared
sidecars while keeping donor captures detached and backbone parameters frozen.
Report regression, behavior and mailbox training budgets separately.

ThoughtWriter is token-conditioned: it uses an isolated private branch, counts
its local tokens and forward passes, and must not mutate the parent's cache.
Ground-truth answers must never be presented as the model's own thoughts.
Mailbox streams need separate banks/cursors, bounded controller-assigned IDs,
expiry, deduplication, cancellation and restart rules; metadata is not a text
channel. Attention and probe labels are diagnostic, not successful incorporation.

Establish causal use by forking complete model/scheduler/RNG state before first
exposure, then comparing active, clean ablated and authorized wrong-content
replays with matched external inputs. Removing memory after it already changed
the native cache is not a clean control. Keep probes and scorer results outside
model access; report expired, unnoticed and rejected mail in denominators.

## Evaluation and isolation

E1/E2 prohibit peer artifact reads. E3 compares both solo models, the coupled
pair and the text pair on matched tasks with declared total-work and wall-time
budgets. E4 begins from the same permitted commit and treats subsequent peer
code, diffs and artifacts as a separately controlled, logged communication
channel. Private worktrees sharing Git objects are not isolation by themselves.
Use enforced private checkouts and an independent merge/test/scoring service.

Freeze manifests, splits, baselines, controls, budgets, tolerances and stopping
rules before formal evaluation; read the actual SERIES.md rules. Preserve failed
and invalid runs, confidence intervals, grouped sampling units and uncertainty.
The original proposed E2 gate used at least 100 independent secret units and a
paired 95% improvement interval above the no-context floor, plus clean isolation
and causal controls; this proposal is not a substitute for a frozen trial plan.

Count local/generated/mailbox tokens, training, projections, tools, activation
and artifact bytes, copies, failures and cleanup. Distinguish startup from steady
state; report repeated latency distributions and errors, not only successful
samples. Use monotonic round trips rather than unsynchronized host-clock
subtraction, and record unmeasured energy as unknown rather than zero.

Limit claims to the audited channel and tested scope: no explicit task text or
token-ID fields on the activation channel does not imply no other communication.
Stop on unexplained native parity failures, forbidden access, corrupt activations
or poisoned sessions. Hardware/security changes and new compute require the
applicable owner authorization, not an attempted workaround to pass a gate.
