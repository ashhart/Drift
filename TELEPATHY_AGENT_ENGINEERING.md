# TELEPATHY — Agent engineering handoff

**Version:** 0.2 · **Issued:** 18 September 2026  
**Deliverable:** staged build specification, executable reference source, tests, commands, and integration contracts.  
**Build status:** CPU reference tested. Real-checkpoint transfer, Studio/MLX, Spark/CUDA, MCDMA, OMP, and E1–E4 research results are not validated by this handoff.

> Build the reviewed architecture, not the earlier illustrative rendering. Keep the original project and component names. Start at M−1, prove native correctness at M0, and advance only when the specified evidence exists. A runnable toy is not evidence of cross-family language transfer.

## 0. Instructions to the implementation agent

You are implementing **Telepathy**, a collaboration system between two frozen language models. The internal bridge is the **callosum**. The intended deployment is model A on the Studio and model B on a Spark. Preserve the original `telepathy-core`, `telepathy-mailbox`, `telepathy-train`, `telepathy-eval`, transport, and `omp-telepathy` separation. [P1 §§1–6]

Read this specification and `AGENTS.md`. Materialize the source appendix or use the accompanying extracted source bundle. Run the reference suite before editing. Inspect the local repositories, checkpoint directories, host capabilities, and `SERIES.md`; fill the run manifest from evidence. Implement the first incomplete gate, test it, record the result, and only then continue. Do not mark optional/skipped tests as passes. Do not substitute a text exchange, shared file, provider API, or fabricated native adapter for the requested activation channel.

At each ticket completion, update `docs/agent-progress.md` with the stage, commit, changed files, exact commands, pass/fail/skip counts, artifact hashes, measured costs, blockers, and next ticket. Preserve failing scientific results. Distinguish **FAILED** experiments from **INVALID** evidence and **BLOCKED** infrastructure. These four engineering labels are proposed here; the actual series verdict vocabulary must be read from `SERIES.md`, which was not supplied. [P1 §8; D1]

### 0.1 What is actually included

| Area | Included implementation | Status and limit |
|---|---|---|
| Native decoder | Frozen dense decoder; canonical KV capture; private native state; self-prefix import; greedy diagnostic generation | Tested with the supplied random toy. Optional Qwen3/Llama adapter and parity tests are supplied but unexecuted here. |
| Attention | Receiver-native GQA, causal masks, shared-softmax foreign prior, exact hard-off gate | Tested on CPU. Not a fused production kernel. |
| Projection and positions | Separate K/V ridge and learned head-mixing maps; receiver rotary application; recency policy | Numerical and gradient tests passed. No claim of semantic alignment. |
| Memory and scheduling | Atomic all-layer publications; bounded sink/recent bank; pinned views; one-epoch-lag scheduler | Single-process correctness reference; remote worker service still to build. |
| Transport | Typed binary float32 frames, HMAC, bounded in-process queue, framed socket/TCP transport | CPU copy baseline; loopback TCP tested. No native GPU transport implementation. |
| Mailbox | Private token-conditioned writer, trainable marker, external ledger, diagnostic label probe | Tested mechanics; no trained addressing protocol or causal replay experiment supplied. |
| Training/evaluation tools | Causal endpoint alignment; extraction, fitting and diagnostic E1 scripts; paired bootstrap; external scorer | Core utilities tested. Scripts requiring actual HF checkpoints are not validated here. |
| Hardware and OMP | Native shim contract and typed control schema | Integration specifications, not working vendor bindings. MCDMA construction deliberately raises an actionable error. |

**Reference validation:** 42 tests passed; one optional HF test module skipped because `transformers` was unavailable. The toy self-handoff maximum absolute logit error was `0.0`; full-prefill versus incremental decoding error was approximately `2.98e-7`. Synthetic affine KV reconstruction error was approximately `4.77e-7`. These are plumbing checks, not E1/E2 scores. See the packaged `validation/` records.

### 0.2 Source and decision precedence

`[P1]` is the supplied **Telepathy: engineering plan**, archived verbatim at `docs/history/original_plan.md`. `[R1]` is the reviewed build schematic shown in the conversation, including receiver-local memory, canonical pre-RoPE export, causal scheduling, isolated audit, and the workspace side-channel boundary. `[D#]` denotes an explicit implementation decision in this handoff. `[S#]` denotes outside primary-source verification, listed in §14.

Preserve P1 where it is not amended. R1 and the explicit changes in §1 define what to build. The original archive is historical input, not permission to revive a superseded assumption. The reference implementation does not override an integration gate or justify a scientific claim.

## 1. Issues resolved before implementation

| ID | Original plan issue | Required implementation decision |
|---|---|---|
| D1 | A transport window was treated as an automatically coherent remote GPU address space. | Use registered, owned buffers and explicit completion/visibility. Project received deltas once into a receiver-local bank. Native memory is not presumed remotely dereferenceable. |
| D2 | Appending foreign entries was conflated with replacing a complete native prefix. | Implement **identity/replacement** and **foreign-memory append** as distinct interventions. Eight bridged layers cannot substitute for every layer of a native prefix. |
| D3 | “Gate near zero” and zero-valued KV were treated as sufficiently closed. | Put the foreign prior inside softmax; provide an exact native-only bypass. Do not ablate by merely zeroing V. |
| D4 | K capture could occur after RoPE or before model-native key normalization. | Export K after any native key normalization, before RoPE; export V unrotated. Rephase projected K using the receiver's full rotary implementation exactly once. |
| D5 | The model examples were treated as one generic 32-layer layout. | Discover every layer/head/rotary/tokenizer/checkpoint contract. The published Qwen3-8B config has 36 layers, eight KV heads and head dimension 128, not 32 layers. [S3] |
| D6 | “No tokens, ever” contradicted token-conditioned ThoughtWriter and a shared repository. | Make the measurable claim about the designated live collaboration channel. Local tokens remain; E4 artifacts are a separately declared communication channel. |
| D7 | The projector was called the only trainable component despite learned gates, markers and writers. | Enumerate all trainable sidecars. Freeze backbones, not the entire receiver computation graph. Separate probe training and never feed probes back. |
| D8 | Token positions from different tokenizers were implicitly equivalent. | Use explicit causal alignment for offline supervision. Source-token recency in live foreign memory is a proposed positional policy, not semantic equivalence. |
| D9 | Attention mass/probe output was treated as successful mail delivery. | Attention and probe outputs are diagnostic. Demonstrate correct incorporation with counterfactual replay starting before first exposure. |
| D10 | Live cross-reads lacked snapshot ownership and a causal order. | Both workers pin the prior complete epoch before either publishes the next one. Fail closed on incomplete layers, stale sessions or corrupted state. |
| D11 | Wire inspection alone was treated as proof of the whole channel claim. | Combine serializer/schema audit, process isolation, tool/file audit, endpoint traces and causal controls. Metadata, timing and artifacts must also be accounted for. |
| D12 | Prior work was characterized as absent at scale or without demonstrations. | Do not repeat blanket novelty claims. The cited KV-transfer paper includes a Qwen3 14B→32B setting; the public MCDMA README also describes a cross-runtime Qwen3-4B handoff. Neither establishes this proposal's cross-family mailbox result. [S1, S4] |

The public MCDMA documentation distinguishes registered host-memory measurements from separate shared Metal/CUDA buffer correctness checks. It says inference integration must manage compatible allocations, registration lifetime, and GPU synchronization; arbitrary existing `cudaMalloc` and Metal-private buffers are not a generally established path. Treat the local checkout and host tests as authoritative for the installation you actually use. [S4]

A learned gate is an interference-control mechanism, **not a security firewall**. The resource limits, isolation, explicit ownership, authentication, and failure policy are the engineering boundaries. [D3, D11]

## 2. Build schematic and execution boundary

```text
                        OMP Duo + omp-telepathy
                  setup + immutable run manifest + typed status
                          |                        |
                          v                        v
  STUDIO / WORKER A                                SPARK / WORKER B
  Frozen model A                                  Frozen model B
  Native cache A, private                         Native cache B, private
      |                                                 |
  AttentionTap: canonical K,V                      AttentionTap: canonical K,V
      |                                                 |
  owned outbound snapshot                         owned outbound snapshot
      |                                                 |
      +------ A->B callosum stream --------------------->+
      +<----- B->A callosum stream ----------------------+
      |        inproc | TCP | qualified MCDMA            |
      v                                                 v
  B->A projector, once                            A->B projector, once
  ForeignKVBank A                                 ForeignKVBank B
  receiver PositionAligner                        receiver PositionAligner
  Gate + ForeignKVInjector                        Gate + ForeignKVInjector
      |                                                 |
      +--> native attention output                    <--+

  Private ThoughtWriter branches produce separately scoped mailbox KV streams.
  SyncController: A(k) reads B(k-1); B(k) reads A(k-1).
  Read-only audit/scoring is outside both model processes and tool sandboxes.
  E4 workspace exchange is a declared, independently logged artifact channel.
```

**[D13] Initial software target:** one-process PyTorch reference, float32, batch one, unquantized dense full attention. Use the supplied toy for the first smoke test. Then qualify the dedicated adapter against pinned local Qwen3 and Llama checkpoints. Port to MLX only after those results exist. This is a build-order choice, not a change to the eventual Studio/Metal + Spark/CUDA deployment.

**[D14] Initial cross-family choice:** Qwen3-8B and the Llama-3.1-8B option from P1. Pin the exact base/instruct variants and licensed local files; the specification does not assert they are installed. GLM-4-9B remains an alternative requiring its own adapter. Keep Qwen3-4B→Qwen3-8B as the intended same-family development pair, after discovering its actual contracts. [P1 §2]

**[D15] Runtime ownership:** the worker owns model state, caches, tensor buffers, local tokens, and GPU work. The OMP plugin owns room lifecycle and presentation, not activations. A generic provider completion API cannot perform these interventions. The new local worker service is part of the implementation, not an assumed OMP feature. [P1 §1; R1]

## 3. Repository and quick start

The Python package retains the original component divisions, with `runtime/` added for stateful model workers.

```text
telepathy/
  core/         types, attention, projector, position, memory, sync
  runtime/      dedicated decoder, toy model, native state, manifests
  transport/    binary codec, inproc/socket backends, native shim contract
  mailbox/      ThoughtWriter, marker, ledger, diagnostic ProbeHead
  train/        causal alignment, ridge/MLP fitting, receiver-vocabulary KL
  eval/         paired statistics, verdict helper, cost ledger
  plugin/       typed control schema
scripts/        inventory, extraction, fitting, self-handoff, E1 diagnostics,
                external scoring, manifest validation, Markdown extraction
configs/        unresolved real-run template and honest stage status
tests/          CPU correctness tests and optional HF architecture parity gates
docs/           original plan and agent progress log
validation/     actual reference-run evidence, not research benchmark claims
```

### 3.1 Materialize this Markdown alone

Every source file in §15 has a relative path, SHA-256, and four-backtick code block. Save the `scripts/materialize.py` block as a local Python file, then run:

```bash
python3 /path/to/materialize.py TELEPATHY_AGENT_ENGINEERING.md ./telepathy
cd telepathy
```

The extractor refuses a nonempty destination, duplicate paths, traversal and bad hashes. It creates source files; it does not execute their contents. The optional source ZIP already contains the same extracted files.

### 3.2 Run the supplied reference

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest -q
mkdir -p local/reference
PYTHONPATH=. python scripts/doctor.py --out local/reference/inventory.json
PYTHONPATH=. python scripts/self_handoff.py --out local/reference/self_handoff.json
PYTHONPATH=. python scripts/reference_demo.py --out local/reference/transport_demo.json
```

These commands use a random small model, not a downloaded checkpoint. Installing dependencies requires an approved package source. The reference pins are Python 3.11–3.13, PyTorch 2.10.0, NumPy 2.3.5, safetensors 0.7.0, and pytest 9.0.2. They describe this baseline; do not assume that the same wheel or CUDA build is available on every target architecture. Create per-host lock files, record any approved build variation, and rerun the gates. Never upgrade a production environment in place.

The optional HF adapter is deliberately pinned to `transformers==4.56.2`, whose Qwen3/Llama implementation boundaries were inspected. Its tests were skipped in this environment because the dependency was unavailable; an attempted installation could not reach the package index. No real model was loaded here. [S5, S6]

```bash
python -m pip install -e '.[test,hf]'
python -m pytest tests/test_hf_optional.py -q
```

Any skip here blocks the real-adapter gate. A package version change is an adapter requalification, not a routine dependency bump.

## 4. Core engineering contracts

### 4.1 ModelAdapter / AttentionTap / native state

`FrozenDecoder.forward()` is the reference implementation of the ModelAdapter contract. It returns logits, a new private `NativeState`, canonical KV for the newly consumed local input, and diagnostic foreign-attention mass. It uses the original checkpoint modules and freezes their parameters. It does not patch a provider, rely on every internal tensor being visible through a generic hook, or inject into the native cache.

The canonical transport layout is `K,V: [T, Hkv, D]`, batch one. K is captured after native K normalization and before rotary application. V is captured at its native unrotated boundary. The dedicated decoder uses each checkpoint's rotary module for its cosine/sine policy. Qwen3's inspected implementations put key normalization before RoPE and do not rotate V. [S5, S8]

Native state is different: its stored K is already rotated. Do not serialize it as canonical K. Do not normalize a projected K again merely because the receiver normally normalizes newly generated native K. A separately learned normalization sidecar is permissible only when declared, trained and ablated as such. [D4]

The reference rejects unknown model types, quantization, tensor-parallel variants, sliding/hybrid attention and dynamic/unsupported rotary policies. It currently admits fixed-frequency default and `llama3` rotary configurations, subject to HF parity tests. A shape-compatible model is not automatically a compatible adapter. [D13]

For a real worker, make model revision, tokenizer files, configuration, rotary convention, layer count, query/KV head counts, head dimension, dtype, quantization, device, framework version, adapter commit and chat template part of an immutable manifest. Compute source-shard hashes outside the timed inference section. Assert that every backbone parameter is frozen and checkpoint hashes remain unchanged after training.

### 4.2 Distinguish identity replacement from foreign-memory append

**M0 identity replacement:** the sender and receiver are the same checkpoint and runtime. Capture every prefix layer, import exactly that prefix with the same positions, consume the same continuation, and compare against ordinary native decoding. This test uses no learned projector, no sink truncation and no eight-layer approximation. `import_self_prefix()` exists only for this correctness gate.

**Foreign-memory append:** the receiver has its own native cache and separately attends to projected foreign entries at selected layers. This is an intervention that must be learned and evaluated. It does not magically reconstruct every missing native layer or prove prefill equivalence. `ForeignKVBank` and `attend()` implement this second path. Never report append-mode success as a faithful cache-replacement result. [D2]

### 4.3 BridgeProjector and head mapping

Each direction and layer pair has separate K and V maps. The ridge reference fits centered affine regression in float64 and saves float32 weights. The learned reference combines a trainable head-mixing matrix, a direct channel map and a normalized residual MLP. There is no forced sharing between K and V.

Receiver-native GQA expansion is still required after projection: a receiver KV head may serve multiple receiver query heads. That replication is not the cross-family head map. Report identity/shape-only mapping where applicable, naive head repeat/truncation, ridge and learned maps as distinct baselines. Never use arbitrary reshaping as evidence of representation compatibility. [P1 §3; D8]

For illustration, the supplied learned map at `8×128 → 8×128`, rank 64, has **66,816 parameters per layer including K and V**, or **1,069,056** for eight layers in both directions, excluding gates/markers/probes. This is a calculation from the supplied module, not a training-time estimate. Compute actual counts for the chosen pair.

The single-source flattened ridge in this kit is a useful diagnostic, not a faithful reproduction of all details in the cited paper. That paper describes per-head fitting and selection of multiple predictive source layers. Implement its actual matching conditions and fitting procedure before calling M0b a replication. [S1]

### 4.4 PositionAligner

Use canonical K → projector → receiver rotary → attention. Values are never rotated. Preserve source slot ordering and sink identities. Keep the receiver's rotary scaling, rotation layout, any partial rotary dimension, position-ID convention and cache-origin policy explicit. A RoPE base number alone is not a complete adapter.

**[D16] Reference live-memory policy:** at the beginning of one forward/decode step,

```text
foreign_virtual_position[j] = receiver_first_query_position - 1
                              - (latest_source_position - source_position[j])
```

This makes the latest foreign entry one slot before the step's first query and preserves source-token age differences. Negative virtual positions are intentional. All queries in a multi-token prefill use one consistent pinned view/anchor for that call. Validate this policy against the actual receiver rotary implementation and train/evaluate under the same policy. It is not the same as aligning different tokenizers or reconstructing original document positions.

Keep exact prefix positions for identity import. The original learned phase-remap alternative remains a later ablation, not an implemented feature in this kit. Do not implement it before the fixed, explicitly tested policy produces a meaningful baseline. [P1 §3; D16]

### 4.5 Gate and attention equation

For one receiver layer, let `S_native = Q K_native^T / sqrt(D)` and `S_foreign = Q K_foreign^T / sqrt(D)`. Use the checkpoint's actual attention scaling when it differs. Then:

```text
g = sigmoid(gate_logit)
P = softmax(concat(S_native + native_mask,
                   S_foreign + foreign_mask + log(g)))
output = P_native V_native + P_foreign V_foreign
```

**[D17] Initial learned logit:** `-6`. This is a proposed starting point, not a guarantee of stability. The foreign attention mass still depends on the content and number of foreign keys.

`override=0` must take the exact native-only path. `override=1` removes gate suppression for an explicit ungated control. A wholly masked foreign bank must also reduce to native-only behavior. Zeroing K or V is not a clean removal: foreign entries may remain in the softmax denominator. Preserve the receiver's native causal mask and disallow unavailable foreign entries. [D3]

### 4.6 ForeignKVBank and publications

A delta contains one contiguous source-slot interval for **all selected source layers** in a directional stream. The reference `Delta` carries only a session UUID, direction, epoch, sequence, start slot and tensors. Layer identity/shape comes from the pinned session contract. It carries no prompt strings, token IDs, arbitrary metadata or answer labels.

A bank verifies the entire delta, projects each new entry once, assembles the next bounded view, then publishes it. A reader pins that view for the entire model step. Later publication cannot mutate the pinned old view. The reference is single-threaded; implement a lock or equivalent publish/read ownership scheme before multi-threaded serving.

**[D18] Reference window:** four initial source sinks plus 128 recent source slots per selected layer. Preserve sink order and source distances. Do not accumulate every transferred entry forever, and do not re-fetch or re-project the whole remote window for each receiver query. Training-time differentiable injections use explicit `ForeignView` tensors; the inference bank's commit path is intentionally under `no_grad()`.

Duplicate, reordered, gapped, wrong-session, wrong-direction, missing-layer or nonfinite publications fail closed. The reference does not implement transparent retransmission, durable restart, chunked mode or sliding asynchronous scheduling. Add those only with their own state-machine and failure tests. [D10]

## 5. Stage M−1 — Discover, pin and establish boundaries

**Goal:** know what exists before modifying either host. This added stage is the prerequisite to the original M0–M6 sequence. [R1]

**Ticket M−1.1: inventory.** Locate the existing MCDMA checkout, OMP/Duo plugin source, local checkpoints and `SERIES.md` using read-only discovery. Record their paths and commits, OS/build, CPU/GPU/RAM, CUDA/Metal versions, installed framework builds, driver/provider versions, configured interfaces and approved storage paths. Do not assume a repository described as “built” exposes the functions needed here. Do not install a driver or alter host security to make discovery pass.

```bash
# Set these to discovered, existing local directories; do not leave placeholders.
export MODEL_A_DIR=/absolute/path/to/pinned/model_a
export MODEL_B_DIR=/absolute/path/to/pinned/model_b
mkdir -p local/mminus1
PYTHONPATH=. python scripts/doctor.py --model "$MODEL_A_DIR" --out local/mminus1/model_a.json
PYTHONPATH=. python scripts/doctor.py --model "$MODEL_B_DIR" --out local/mminus1/model_b.json
cp configs/run.template.json local/mminus1/run.json
PYTHONPATH=. python scripts/validate_manifest.py local/mminus1/run.json
```

The final command should initially return `BLOCKED`, exit 2, listing missing values. Populate them from evidence; do not invent hash strings. The validator checks completeness only. It does not prove that a manifest, hash, permission or experiment is valid. A later-stage-only item may be explicitly marked not applicable for an earlier stage; it must be resolved before the stage that needs it.

**Ticket M−1.2: sandbox and authority.** Define separate model-worker identities, private cache/artifact directories, a controller identity and an experimenter-only audit/scoring identity. Keep secrets, hidden tests, reference answers and audit outputs outside both workers' readable paths. Verify attempted access fails. Document authorized network endpoints and local tools. Set owner-approved training/inference/memory/storage budgets before loading large models or booking GPU work.

**Ticket M−1.3: manifests and layer map.** Pin exact checkpoint variants and every input artifact. Discover rather than hard-code layer counts and head shapes. Choose explicit directional source→target layer maps using training/development data only. Distinct tokenizer files are a project stress condition; tokenizer difference alone neither defines nor proves distinct model families. Document family provenance separately. [D5, D8, D14]

**Exit evidence:** two host inventories; verified source/config/tokenizer/weight manifests; actual `SERIES.md` hash; explicit layer-pair files; sandbox access tests; approved resource caps; unresolved-integration register. No scientific experiment is credited at this stage.

## 6. Stage M0 — Native correctness, then same-family replication

### M0a — Adapter and self-handoff gate

**Ticket M0.1:** run the reference suite and optional tiny HF parity tests. Qualify each real checkpoint adapter against its unmodified stock forward before enabling foreign attention. Test full prefill versus incremental decode, exact hard-off gating, canonical tap normalization/rotation boundaries, every layer, GQA shapes, and unchanged backbone weights.

```bash
mkdir -p local/m0
python -m pytest -q
python -m pytest tests/test_hf_optional.py -q
PYTHONPATH=. python scripts/self_handoff.py --model "$MODEL_A_DIR" \
  --device cpu --out local/m0/model_a_identity.json
PYTHONPATH=. python scripts/self_handoff.py --model "$MODEL_B_DIR" \
  --device cpu --out local/m0/model_b_identity.json
```

Choose a qualified device that fits the unquantized checkpoint; the example does not imply that CPU memory or runtime is suitable on every machine. The supplied CLI checks a short deterministic prefix. Expand qualification to lengths `1, 2, 7, 16, 127, 256`, multiple seeds, single-token and multi-token continuations, repeated prefills and the intended deployment context sizes. Unsupported padding/batching must fail explicitly rather than silently take an untested path.

**[D19] Proposed float32 tolerance:** absolute and relative error `2e-5` for stock-forward and identity comparisons, accompanied by the error distribution, not only a pass flag. The reference CLI uses a conservative maximum-absolute check. Lower precision needs a separately preregistered tolerance and corresponding logit/task analysis; do not widen tolerance after seeing a failure simply to proceed.

**Ticket M0.2:** prove identity replacement at every layer with unchanged checkpoint and exact prefix positions. Prove that hard-off append mode leaves native outputs and cache content unchanged. Reject any attempt to infer success from a visually similar answer or one matching token.

**Exit:** real-checkpoint parity and self-import reports, complete test matrix, parameter/shard hashes before and after, no skipped required tests. Any unexplained mismatch blocks training.

### M0b — Same-family ridge baseline and faithful replication

**Ticket M0.3:** use the discovered Qwen3-4B→Qwen3-8B pair, or document an approved matched-family replacement. Implement the cited paper's precise replication arm separately from the kit's simple single-source ridge diagnostic. Source-layer selection and hyperparameters use training/development data only. Keep calibration corpora and held-out test units separate. [P1 M0; S1]

For the kit's first diagnostic layer, the following commands are already implemented. Training JSONL has exactly `id`, `text`, `split`, with `split="train"`. Use a new output directory for each run.

```bash
export SOURCE_DIR=/absolute/path/to/pinned/same_family_source
export TARGET_DIR=/absolute/path/to/pinned/same_family_receiver
export SOURCE_LAYER=0
export TARGET_LAYER=0
mkdir -p local/m0/alignment
PYTHONPATH=. python scripts/extract_alignment.py \
  --source-model "$SOURCE_DIR" --target-model "$TARGET_DIR" \
  --source-layer "$SOURCE_LAYER" --target-layer "$TARGET_LAYER" \
  --input /absolute/path/to/training_only.jsonl \
  --out local/m0/alignment/pair_0.npz --device cpu --max-tokens 512 --max-rows 8192
PYTHONPATH=. python scripts/fit_projector.py \
  --data local/m0/alignment/pair_0.npz --kind ridge --out local/m0/ridge_pair_0
```

The extractor uses validated original-text character offsets, converts endpoints to UTF-8 byte coordinates, and pairs only shared causal endpoints. It rejects normalization/round-trip differences instead of assuming they align. It excludes zero-length special-token offsets. It does not use future source rows to predict earlier receiver states. Its raw-text/no-chat-template mode is explicit; a production benchmark needs audited, consistent template handling.

**Ticket M0.4:** build formal E1 around the diagnostic with no-context floor, receiver reread ceiling, English-summary baseline, identity/naive/ridge controls and independent held-out questions. A score above the floor is a minimum development gate, not sufficient evidence of faithful paper reproduction.

**Exit:** quantitative E1 result with interval and controls; faithful-replication result separately labeled; causal-alignment audit; frozen backbone proof; row/token/compute budget ledger. If the floor is not beaten, debug alignment, positions, normalization, layer mapping and evaluator before adding novelty. [P1 M0]

## 7. Stage M1 — Cross-family one-way transfer

**Ticket M1.1: change one dimension of the experiment.** Move to the pinned Qwen3-8B→Llama-3.1-8B direction after M0. Initially use one source→target layer pair and in-process transport. Add the planned eight-layer subset only after one-layer diagnostics are understood. Preserve the reverse direction as a separately trained, separately measured map.

Train both ridge and learned head-mixing maps using the same split and budget accounting. The following supplied command fits the regression component of a learned map; it does not perform behavior distillation:

```bash
PYTHONPATH=. python scripts/fit_projector.py \
  --data local/m1/alignment/pair_0.npz --kind mlp --steps 200 \
  --out local/m1/mlp_pair_0 --seed 11
```

Create that alignment file with the M0 extraction command using the cross-family directories. Do not mistake similar KV reconstruction loss for useful receiver behavior.

**Ticket M1.2: behavior training.** Implement the receiver-task training loop around the supplied differentiable decoder and `receiver_kl()`:

```text
Donor: frozen, source-private training context -> detached canonical KV.
Map: trainable K/V sidecar -> projected canonical receiver-space KV.
Teacher: frozen RECEIVER with its legitimate training context -> receiver logits.
Student: frozen RECEIVER with foreign KV + matched query/continuation -> receiver logits.
Loss: lambda_K * K regression + lambda_V * V regression
      + lambda_KL * KL(teacher_receiver || student_receiver)
      + declared task loss / regularization when applicable.
```

Teacher and student distributions use the **receiver vocabulary** and the same scored output positions. Never compare Qwen and Llama vocabularies directly. Teacher execution may use `no_grad`; student execution must preserve gradients into maps/gates even though backbone parameters do not require gradients. Use detached donor captures and explicitly bounded backpropagation horizons. The regression, behavior and mailbox-training budgets are separate ledger entries. Loss weights, stopping rules and validation selection are preregistered before held-out evaluation. [P1 §3; D7, D8]

**Ticket M1.3: E1 and E2.** The supplied one-layer E1 diagnostic runner accepts only `id`, `context`, `question`. Keep `id`, `answers` in a separate scorer file. It runs floor, reread ceiling, foreign-memory, hard-closed and wrong-context controls:

```bash
PYTHONPATH=. python scripts/run_e1_diagnostic.py \
  --source-model "$MODEL_A_DIR" --target-model "$MODEL_B_DIR" \
  --source-layer 0 --target-layer 0 --projector local/m1/ridge_pair_0 \
  --input /absolute/path/to/e1_inputs_without_answers.jsonl \
  --out local/m1/e1_diagnostic.jsonl --device cpu --new-tokens 32
# Run as the EXPERIMENTER identity, not a model-accessible tool:
PYTHONPATH=. python scripts/score_e1.py \
  --predictions local/m1/e1_diagnostic.jsonl \
  --answers /private/scorer/e1_answers.jsonl --out /private/scorer/e1_summary.json
```

This runner is expressly **not** formal E1/E2: it lacks the summary arm, complete run/cost audit and strict process/channel controls. It also commits directly to an in-process bank. Route publications through the common transport interface before transport-equivalence experiments. Do not rename its output to a formal result.

For E2, assign randomly generated held-out secrets to only the donor; the receiver gets only its question and authorized foreign KV. Include no-context, hard-off, wrong-secret, time-shuffled, random-map and source-text oracle conditions. Randomize assignment before the trial and make leakage tests part of the run, not an afterthought. [P1 E2; D11]

**[D20] Proposed E2 advancement gate:** at least 100 independent secret units; a preregistered paired 95% interval for the improvement over no-context whose lower bound is above zero; clean channel/isolation audit; and causal ablations consistent with the claimed source. Multiple questions or seeds for one secret are grouped at secret level. Power analysis may require more units. Success on this gate does not establish E3/E4 value. The supplied bootstrap and gate helper are utilities, not automatic validation of these conditions.

**Exit:** both projection baselines reported; formal one-way E1/E2 evidence; no illicit control/artifact path; exact training and inference costs; complete failure analysis when the gate is not met.

## 8. Stage M2 — Live coupling and E3

**Ticket M2.1: worker service.** Build a stateful local process around `FrozenDecoder`, native state, a source cursor, outbound ownership, inbound banks, sidecar checkpoint and run manifest. Commands after setup are typed lifecycle/status events. The worker, not a generic provider response, supplies the KV tap. Add authentication, timeouts, cancellation and resource limits to the actual service interface after inspecting the host integration.

A worker consumes its own authorized local input, runs one model step against a pinned incoming view, retains its new private native state, and publishes only its new canonical rows. Local generated tokens remain inside that worker; any mailbox writing or tool invocation is counted. Keep native source position, transport sequence and coupling epoch distinct: a prefill can consume many source positions in one epoch.

**Ticket M2.2: causal schedule.** Implement the supplied `SyncController` semantics remotely:

```text
At epoch k:
  pin B's committed snapshot k-1 for A; pin A's committed snapshot k-1 for B;
  run A(k) and B(k) using those pinned views for every bridged layer;
  finish GPU producers; publish complete outbound blocks;
  transfer -> validate -> project once -> publish incoming views;
  advance only after the required bounded acknowledgments/completions.
```

The reference may compute workers serially for reproducibility; a qualified implementation may compute them concurrently because neither consumes same-epoch output. Never wait for the peer inside an attention layer. Never let early receiver layers see one peer epoch and later layers see another.

**Ticket M2.3: recovery and degeneration.** Implement checkpoints containing native state, unphased foreign banks, projector/gate/marker identities, both cursors, transport sequence/epoch, sampling RNG states, scheduled work and pending mailbox state. On a partial publication or process failure, poison the run; restore a recorded consistent checkpoint or start a new run/session. Do not attach a stale packet stream to new model state. Define startup and missing-peer policy before benchmarks; a safety hard-off abort is not an ordinary successful coupled sample.

Add bounded native/outbound/foreign memory, expiry, queue-backpressure, finite-value checks, model-output repetition metrics, KV norm/covariance drift, gate saturation, mailbox storm counts and echo diagnostics. A high attention oscillation alone is not proof of semantic echo; report it with task outputs and failure cases. Thresholds are tuned on development data and frozen for evaluation.

**Ticket M2.4: E3.** Compare A alone, B alone, coupled pair, and the Duo/text pair on the same reasoning tasks. Use declared budgets for both total inference work and wall-clock; include local decode, mailbox tokens, projector/attention work, training cost and tool calls. Report the quality/cost trade-off when budgets cannot be exactly matched, rather than presenting one token-count match as compute equality. [P1 E3]

**Exit:** deterministic causal-schedule tests; stale/duplicate/timeout/crash tests; stable bounded resource use; measured E3 comparisons. A coupled score below a solo or text baseline is a valid result, not permission to change the benchmark.

## 9. Stage M3 — Mailbox, addressing and causal incorporation

**Ticket M3.1: preserve the truthful writer description.** The supplied `ThoughtWriter` is token-conditioned: locally generated or locally authorized token IDs plus a learned marker are forwarded through a private decoder branch. Only canonical KV may be exported. The parent's native cache must not change. Ground-truth answer text must never be supplied as the model's “own thought.” Count the local tokens and forward passes. A direct latent writer is a different later experiment, not what this code does. [P1 ThoughtWriter; D6]

**Ticket M3.2: define mailbox streams.** The reference wire v1 has no message-address envelope and the supplied ledger does not transmit one. Implement mailbox addressing deliberately. Use separately predeclared stream/session identities and separate banks for native-context and mailbox KV; do not reuse native cursor/sequence counters for private branches. A native prefix offset is not automatically a mailbox stream slot.

**[D21] Starting mailbox policy:** one private branch per sender; at most 32 local mail tokens plus a marker; a bounded fixed-slot mailbox; deterministic host-assigned message IDs and fixed enum statuses; predeclared TTL and per-epoch creation cap. Pad/cadence policies and their compute costs are part of the manifest. IDs, slots and acknowledgments are controller-assigned, not free-form model text. Reserve channel metadata before the task where possible. Add explicit expiry, cancellation, retry/dedup and restart rules without placing question/answer strings in metadata.

Train marker salience and addressing on training-only question/answer tasks. Train probes separately on labeled development data and report their held-out accuracy/calibration. The supplied linear `ProbeHead` predicts diagnostic labels; it does not reconstruct arbitrary thoughts with guaranteed fidelity. Probe outputs are experimenter-only, including UI renderings and derived files. [P1 §3; D9]

**Ticket M3.3: build counterfactual replay.** The supplied ledger records externally established causal results; it does **not** perform replay. Implement the replay runner as follows:

```text
Checkpoint BEFORE the receiver first sees the candidate question/answer entries.
Fork all relevant native caches, foreign banks, scheduler state and RNG state.
Active replay: use the original foreign entries.
Ablated replay: remove/mask the candidate entries before first exposure.
Wrong-content replay: use a matched incorrect answer/control where authorized.
Hold other exogenous inputs and schedules fixed; let downstream model states diverge.
Score correctness/use outside both workers; never return scorer output to them.
```

Removing an answer after it has already changed the receiver's native cache is not a clean no-exposure control. Preserve common random numbers where meaningful, and report when trajectories make a comparison ill-defined rather than forcing apparent agreement.

**Ticket M3.4: metrics.** Separate created, visible, attended, correctly incorporated, expired/unnoticed, and rejected messages. Report attention without incorporation. For the first preregistered implementation, count causal use when the active replay is correct and the clean ablated replay is not; also report graded distributional/task effects rather than treating every nonzero effect as success. Log delivery/incorporation latency in decode epochs and wall-clock. Include expired/unnoticed messages in denominators and latency censoring; do not report latency only for delivered mail without its failure rate.

**Exit:** private-branch and marker-gradient tests; complete stream/addressing state machine; no audit feedback; nonzero **causally supported correct** incorporation on held-out tasks; latency, starvation and storms honestly reported. Mere attention or a probe guess does not pass. [P1 M3; D9]

## 10. Stage M4 — MLX port, TCP parity, then MCDMA

### 10.1 MLX / Metal model adapter

Port the dedicated adapter's verified operator boundaries, not just its Python class names. Preserve each native norm, Q/K/V layout, rotary policy, causal mask, cache offset, output projection and MLP. Keep foreign memory separate. Explicitly force lazy producers to complete at export and preserve the source allocation while a transfer can read it. Use the exact locally pinned MLX/MLX-LM implementation. The public Qwen3 MLX implementation was inspected only as an interface reference; no MLX code was run in this environment. [S8]

Write a runtime-specific fixture exchange format for captured canonical tensors and logits. First compare within each runtime against its stock model; then compare cross-runtime fixtures with declared precision/quantization tolerances. Do not use an unqualified MPS/PyTorch run as proof that the MLX/Metal adapter is correct.

### 10.2 Transport v1 and correctness fallback

`FrameCodec` is a deliberately copied CPU float32 format:

```text
Header, network byte order, 56 bytes:
  magic[8], version:u8, direction:u8, session_uuid[16],
  epoch:u64, sequence:u64, source_start:u64, token_count:u32, layer_count:u16
For each selected source layer:
  layer_index:u16, kv_heads:u16, head_dim:u16
  K: contiguous little-endian float32 [T,H,D]
  V: contiguous little-endian float32 [T,H,D]
Trailer: HMAC-SHA256, 32 bytes
TCP framing: additional big-endian u32 frame length
```

The codec caps frames at 64 MiB and bounds token/layer/head dimensions before tensor construction. It rejects authentication failure, truncation, trailing untyped bytes, nonfinite payloads and invalid shapes. Session/model contract and sequence validation happen in addition to decoding. HMAC is not encryption; use loopback or an approved protected tunnel, then mutual TLS for a remotely exposed service. Never use pickle or unrestricted deserialization on activation traffic.

**Ticket M4.1:** make in-process, TCP and native candidate backends implement the same logical publication API and replay fixtures. Use the same dtype, projector, snapshot schedule, masks and seed. Test cancellation, frame limits, malicious lengths, byte corruption, duplicate/stale sessions, slow peers and receiver restarts. The current in-process codec and actual loopback TCP tests passed; this is not a cross-host throughput result.

### 10.3 Native bridge contract and lifecycle

`native_contract.h` is a **new bridge-owned interface proposal**, not a claim that these symbols exist in MCDMA. Implement a native shim only after reading the actual pinned allocator/registration/completion APIs. The reference `MCDMATransport` guard remains until that implementation passes the following contract. [D1]

```text
Allocate compatible region -> register -> GPU writes -> producer completion fence
-> publish immutable descriptor -> submit transfer -> transport completion
-> establish receiver GPU visibility -> validate -> project/publish local bank
-> all consumers release -> acknowledge/recycle -> deregister/free when idle.
```

Verify byte ranges, alignment, capability/region permissions, rkey/handle lifetime, outstanding work, partial failure and process death. Keep compatible registrations and buffers alive through both transfer completion and consuming GPU work. Copy payloads into an approved buffer when required and count that copy; do not call the result end-to-end zero-copy merely because the NIC transfer uses DMA.

**Ticket M4.2: targeted hardware tests.** Start with known byte patterns and checksums with GPUs idle, then GPU-produced/GPU-consumed data, then one canonical KV block, then a long wraparound ring, then concurrent directions and failure recovery. Prove all-layer publication atomicity at the Telepathy layer; DMA completion alone does not provide it. Deny out-of-bounds descriptors, premature reuse/free, stale registration handles and unsupported memory types. No fake successful fallback is allowed: `backend=tcp` and `backend=mcdma` are different recorded conditions.

Do not copy installation/security commands from a web page into an autonomous setup script. Experimental driver installation, security changes, management-interface changes or rebooting require separate owner authorization and a documented rollback route. This handoff intentionally performs none of them.

**Ticket M4.3: benchmark.** Record producer-ready, submit, complete, receiver-visible, projection and decode timings; payload/wire bytes; copies; queue depth; p50/p95/p99; warm-up; transfer sizes; repetitions; clocks; host load; dtype; raw samples and errors. Use a sender-side monotonic clock for round-trip measurements; unsynchronized host-clock subtraction is not valid one-way latency. Include any keepalive workloads in energy/compute accounting. `joules=null` means unmeasured, not free.

**Exit:** native memory-lifetime/concurrency tests; qualified Metal/CUDA fixture exchange; E1/E2 replay equivalence with matching statistical tolerances across backends; actual throughput/latency/copy/energy report. A green link, successful driver load, or matching wire checksum alone does not pass inference integration.

## 11. Stage M5 — E4 duel and evidence package

**Ticket M5.1: OMP integration.** Inspect Duo's actual room lifecycle, provider boundary, command registration, event routing, tool grants, persistence and audit UI. Bind `omp-telepathy` to the new worker service through a documented interface. Keep `/telepathy start <pair>` and `/telepathy duel <task>` as the intended user surface from P1, but do not invent host API calls to implement them. The supplied numeric `ControlEvent` schema is the control-policy reference, not an OMP SDK.

Before trial start, allow authorized setup/task allocation and pin the manifest. During strict trials, models may read only fixed typed lifecycle/status events, not task-derived hints, answers, token IDs, probe reconstructions or scorer results. A UI may display audit data to a human only when it cannot be read back by model tools, files, screenshots or provider context.

**Ticket M5.2: workspace boundary.** E1/E2 prohibit peer artifact reads. E4 begins from the same permitted starting commit, but subsequent peer code/diffs/artifacts are another communication channel. Use private isolated checkouts, or worktrees with enforcement that also prevents access through shared Git objects/refs. Separate clones are often simpler to isolate. A shared `.git` store is not private merely because working directories differ.

Use a controlled merge/test service and a fixed peer-artifact visibility policy. Log each authorized peer artifact read and its bytes. Neither model may read hidden tests, scorer outputs or audit caches. A telepathy arm that can read its peer's new code cannot support the global statement that collaboration occurred only through KV. It may support the narrower declared-channel statement. [D6, D11]

**Ticket M5.3: fair duel.** Run the existing Duo text arm and the coupled/mailbox arm on the same initial commit, backlog, tools, scheduling rules, model checkpoints and approved budgets. Counterbalance arm order, reset state between trials, and use independent held-out tasks. Account for training amortization separately from per-task runtime. Record task completion/hidden-test results externally, wall time, local/generated/mailbox tokens, explicit text-channel tokens, all activation and artifact bytes, failures, incorporation rates and causal deltas. [P1 E4]

**Exit package:** frozen manifests; split/preregistration hashes; raw predictions; hidden scores under access control; channel/cost audit; baseline tables with intervals; failed runs; degeneration cases; replay artifacts; backend qualification; exact reproduction commands. Publish only approved/redacted artifacts. Do not publish caches or training data simply because they were convenient audit files.

Use the claim: **“No explicit task-derived text or token-ID fields traversed the designated live activation channel under the audited protocol.”** State the setup, local-token and E4-artifact qualifications alongside it. A tensor may encode linguistic information; a binary schema does not prove nonlinguistic cognition or the absence of all covert communication. [D6, D11]

## 12. Stage M6 — Competition harness

Preserve P1's competition structure: competitor agents propose projector, gate, sync and mailbox policies between frozen models with declared training budgets; development pairs are public and evaluation settings are frozen/hidden. Implement the policy ABI only after M5 has a stable reference. [P1 M6]

Freeze permitted hooks, architecture descriptors, allowed calibration data, transfer resource limits, training tokens/device-hours, runtime limits and scoring before submissions. Competitors must not change backbones, adapter correctness, answer stores or channel boundaries. Hidden-pair use needs a specific generalization contract: a fixed-shape projector cannot silently acquire new shapes by training on the hidden evaluation task. Either provide approved architecture descriptors to a shape-general method, or define a separately budgeted calibration split that contains no evaluation answers.

Run untrusted submissions in isolated sandboxes with no hidden-score feedback. Add conformity tests for all frozen invariants, not just a top-line task score. Report quality and costs plus INVALID/BLOCKED results rather than ranking an invalid side channel as an algorithmic improvement.

**Exit:** immutable policy API; reference competitor; reproducible isolated runner; budget enforcement; hidden-data policy; signed submission manifests; complete evidence and adjudication rules consistent with the actual `SERIES.md`.

## 13. Capacity, accounting and acceptance matrix

### 13.1 Calculate from the actual model, not the label “8B”

For one direction, one source token and the selected source layers:

```text
KV bytes/token = 2 * sum_over_layers(Hkv_layer * D_layer * bytes_per_element)
new-delta bytes/second = KV bytes/token * new_source_tokens_per_second
raw resident foreign bytes = 2 * sum_over_receiver_layers(Hkv * D * dtype_bytes * retained_slots)
```

Add headers, framing, authentication, ACKs, padding, outbound rings, temporary projection buffers and any alternate representation kept resident. Native model weights/cache and training graphs are separate memory terms. FP32 wire payloads in this reference are twice the raw FP16 payload of the same shape. Do not compare these as if the dtype were identical.

For the **hypothetical** 32-layer, eight-KV-head, head-dimension-128, FP16 layout in P1: one layer is 4 KiB/token, all layers are 128 KiB/token, a 4,096-token prefix is 512 MiB per direction, and 10 new tokens/s in both directions is 2.5 MiB/s of raw deltas. Bridging eight layers gives 0.625 MiB/s total at the same rates. These are binary-unit calculations, not measurements. [P1 §4; D5]

For the published 36-layer Qwen3-8B shape, the corresponding all-layer prefix is 576 MiB at 4,096 tokens and FP16. A 36-layer source plus a hypothetical matching-head 32-layer source at 10 tokens/s each produces 2.65625 MiB/s combined raw deltas. Discover the actual second model before treating that pair calculation as applicable. [S3; calculation]

An eight-layer, 4,096-token FP16 window reread at 10 steps/s in both directions would instead imply **2.5 GiB/s** of remote payload reads. This is why the design transfers each delta once and reuses receiver-local projected memory. The difference between a delta stream and repeated full-window reads is three orders of magnitude in this example. No transport or attention bottleneck is asserted before measurement. [D1; calculation]

### 13.2 Evidence gates

| Gate | Required evidence | Advancement condition |
|---|---|---|
| M−1 | Inventories, immutable contracts, budgets, source access, isolation | Required stage inputs verified; unresolved later dependencies explicitly tracked |
| M0a | Stock parity, self-import, hard-off identity, frozen hashes | All intended real checkpoints/runtimes pass; no required skip |
| M0b | Same-family E1 and separately faithful replication | Above-floor development result and understood replication behavior |
| M1 | Cross-family E1/E2, baselines, causal/channel audit | Preregistered effect over floor; no illicit path; full cost report |
| M2 | Prior-epoch schedule, boundedness/failure tests, E3 comparison | Stable measured coupling; no mixed snapshots or hidden feedback |
| M3 | Addressing, private branch, clean replay, held-out incorporation | Correct causal mail use demonstrated; latency/starvation retained |
| M4 | Runtime parity, buffer lifetime, native transport, backend replay | Qualified inference path and honest pipe measurements |
| M5 | Isolated fair E4 duel and evidence package | Reproducible results and a claim no broader than the audited channels |
| M6 | Frozen policy ABI, hidden-data contract, enforcement | Conforming reproducible competition harness |

### 13.3 Failure triage and stopping rules

**Identity differs:** inspect capture boundary, native K norm, rotary convention, cache offset, mask shape, GQA expansion, dtype and stock attention scaling. Stop before fitting a projector.

**KV regression improves but E1 does not:** test causal endpoint alignment, source-layer selection, native-versus-foreign intervention mismatch, receiver positions, value scaling, source context truncation and missing behavior training. Do not “fix” this by supplying receiver text that the experimental arm is meant not to see.

**E2 succeeds with hard-off gates or wrong secrets:** treat as a leakage/evaluation alarm. Check context assembly, workspace/tool routes, control metadata, dataset memorization and scorer access. Do not report a KV-channel success.

**Mail is attended but unused:** report attended-not-incorporated, inspect marker/addressing and train the behavior objective. Do not relabel attention as delivery.

**Transport checksums pass but inference changes:** inspect dtype/packing, stale buffers, premature reuse, GPU visibility, atomic layer sets and schedule parity. A checksum over the wrong snapshot can still match.

**Unknown hardware/API or exhausted budget:** record BLOCKED with the exact observed missing input. Continue only independent reference tasks that do not rely on it. Do not emulate a native success, invent measurements, weaken host security, or keep spending against an unspecified cap.

## 14. Sources and unresolved inputs

### 14.1 Supplied basis

**[P1]** `Pasted markdown(9).md`, “Telepathy: engineering plan,” provided in this conversation; archived at `docs/history/original_plan.md`. Component terminology, deployment intent, experiment sequence and M0–M6 originate there. References in this file identify its original section labels rather than pretending it proves implementation success.

**[R1]** Reviewed build schematic displayed in the conversation. Receiver-local foreign banks, explicit completion, private mailbox branches, one-epoch-lag scheduling, external audit and declared E4 artifact channels are retained. The small contact-sheet images were not treated as a substitute for an unavailable editable engineering source or local repository.

### 14.2 Outside primary-source checks, accessed 18 September 2026

**[S1]** Heo et al., *Cross-Model KV Cache Transfer in LLM Families: A Closed-Form Linear Mapping for Prefill Reuse*, arXiv:2608.03893. The abstract documents matched-KV conditions, source-layer selection, canonicalized keys and ridge/MLP comparisons. Read the complete method and any released code before claiming replication. Source: `https://arxiv.org/abs/2608.03893`

**[S2]** Flamant, Ghai and Shimizu, *The Bicameral Model: Bidirectional Hidden-State Coupling Between Parallel Language Models*, arXiv:2605.11167. Its coupling is on intermediate hidden states with a learned interface/gate, not an interchangeable implementation of this proposal's KV mailbox. Do not assume every finding transfers. Source: `https://arxiv.org/abs/2605.11167`

**[S3]** Qwen's published Qwen3-8B configuration: used only to check the example's layer/head dimensions and the need for exact configuration discovery. Pin the local revision rather than `main`. Source: `https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json`

**[S4]** MCDMA maintainer README: used to distinguish host-memory timing, shared-buffer correctness, engine integration and documented memory limitations. Its contents are not a warranty for an installation; inspect the upstream README from the operator's pinned MCDMA checkout.

**[S5]** Transformers 4.56.2 Qwen3 source: operator boundary and adapter-qualification reference. Source: `https://raw.githubusercontent.com/huggingface/transformers/v4.56.2/src/transformers/models/qwen3/modeling_qwen3.py`

**[S6]** Transformers 4.56.2 Llama source: dense decoder, native cache and rotary integration reference. Source: `https://raw.githubusercontent.com/huggingface/transformers/v4.56.2/src/transformers/models/llama/modeling_llama.py`

**[S7]** Transformers 4.56.2 cache implementation: reference for native cache ownership/position behavior; the supplied dedicated decoder keeps its own explicitly typed state. Source: `https://raw.githubusercontent.com/huggingface/transformers/v4.56.2/src/transformers/cache_utils.py`

**[S8]** MLX-LM Qwen3 implementation: inspected for Q/K normalization, rotary and cache boundaries, not executed here. Pin the user's actual revision before porting. Source: `https://raw.githubusercontent.com/ml-explore/mlx-lm/main/mlx_lm/models/qwen3.py`

No exhaustive novelty review was performed. Current public source observations above are separated from the supplied plan and from this file's proposed engineering policies.

### 14.3 Inputs the agent must resolve locally

The actual MCDMA/OMP repositories and installed commits; licensed exact model variants; host memory/driver/runtime contracts; MLX checkpoint conversion and quantization choices; `SERIES.md`; authorized datasets and split ownership; approved resource budgets; per-host audit and sandbox policy; and formal baseline/scoring/preregistration files were not supplied. The code must not manufacture them. Their absence does not prevent running the supplied CPU reference, but blocks the relevant integration or research gate.

## 15. Complete source appendix

These are the full reference files, not abbreviated pseudocode. Native MCDMA is deliberately a failing guard plus an unimplemented shim contract; OMP and MLX integration remain the explicit staged work above. Optional real-HF code is included but must pass its currently unexecuted gates before use.

The SHA-256 on each block covers its UTF-8 contents with one final newline. The materializer reads only blocks with this exact marker and four-backtick syntax. It does not scrape arbitrary illustrative code from this specification.

### Source 01 — `.gitignore`

<!-- file: .gitignore sha256: 49ca2cc4dc3a3e62a45a304612b45aa44851af00c42a00590851d8b53fc42239 -->
````text
.venv/
__pycache__/
.pytest_cache/
*.egg-info/
local/
results/
*.safetensors
*.npz
````

### Source 02 — `AGENTS.md`

<!-- file: AGENTS.md sha256: 2af43d3fefb37d72936d6918c5804d7dc852b024a15755019c4eb2744b264072 -->
````markdown
# Agent execution contract

Read TELEPATHY_AGENT_ENGINEERING.md before modifying the project. Treat this kit
as a correctness reference, not evidence that cross-family telepathy already works.

Start at M-1. Preserve the user's component names and M0-M6 research sequence.
Read existing MCDMA/OMP source and SERIES.md before writing integrations. Locate
files with read-only commands; ask for paths only after discovery cannot resolve them.
Do not invent provider APIs, GPU memory guarantees, model revisions, benchmark
scores, energy values, paper novelty, or driver compatibility.

Run the included tests before changing code. A skipped HF test is a BLOCKED real
adapter gate, not a pass. Keep all backbone parameters frozen. Never change model
weights to get identity, transfer, or mailbox tests to pass. Keep task text/token
IDs off the designated live activation transport. Local token generation and
private token-conditioned mail are allowed and must be counted.

Implement the first failing/unimplemented gate only. After each meaningful change,
run focused tests and then the full suite. Do not advance a scientific stage without
its evidence. Use PASSED, FAILED, BLOCKED, INVALID for engineering records; reconcile
these with the actual SERIES.md rules before formal trials.

Put model-inaccessible evidence outside agent-readable repos/worktrees. Do not expose
answer keys, probe output, peer private files, hidden tests or scorer stdout to models.
Use real process/filesystem isolation, not prompt instructions as an access boundary.
Do not auto-install kernel drivers, disable security controls, run privileged changes,
publish private activations, or spend unapproved compute budgets.

Maintain docs/agent-progress.md with stage, exact commit, changes, commands, test
results, evidence hashes, measured costs, unresolved blockers and the next ticket.
Distinguish a measured failure from invalid evidence. Stop on unexplained native
parity errors, forbidden channel access, corrupted activations or poisoned sessions.
````

### Source 03 — `README.md`

<!-- file: README.md sha256: 86259f27a9ef923c621068622f118b1b0ff91fbca66267bbe060dc267c2a33e1 -->
````markdown
# Telepathy reference kit

The complete handoff is TELEPATHY_AGENT_ENGINEERING.md. The source files here are
the same hash-verified files embedded in its source appendix. AGENTS.md defines the
execution contract. This is not a trained checkpoint or a completed hardware system.

## Reference quick start

Use an isolated Python 3.11-3.13 environment. These pins identify the reference
baseline, not a claim that every target host has a matching CUDA/Metal wheel.
Install the host-approved PyTorch build when required, record the variation, and
rerun every affected gate. Do not upgrade production libraries in place.

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest -q
mkdir -p local/reference
PYTHONPATH=. python scripts/doctor.py --out local/reference/inventory.json
PYTHONPATH=. python scripts/self_handoff.py --out local/reference/self_handoff.json
```

For optional local HF adapters, add `python -m pip install -e '.[test,hf]'`, then
rerun tests. Real-checkpoint examples and staged integration tasks are in the main
document. All model loading is local-only, unquantized and opt-in. The default
random toy has no language skill and cannot demonstrate semantic transfer.
````

### Source 04 — `configs/run.template.json`

<!-- file: configs/run.template.json sha256: b60f869eaebf5cf68bb803497c394121dd758d45d2ca9ba2171c9063b0dd5215 -->
````json
{
  "schema": "telepathy.run.v1",
  "run_uuid": null,
  "model_a": {
    "role": "Studio",
    "family_choice": "Qwen3-8B",
    "local_path": null,
    "revision": null,
    "weights_manifest_sha256": null,
    "tokenizer_manifest_sha256": null,
    "config_sha256": null,
    "adapter_commit": null,
    "runtime_lock_sha256": null,
    "quantization": "none_for_reference",
    "dtype": "float32_for_reference"
  },
  "model_b": {
    "role": "Spark",
    "family_choice": "Llama-3.1-8B_variant_to_be_pinned",
    "local_path": null,
    "revision": null,
    "weights_manifest_sha256": null,
    "tokenizer_manifest_sha256": null,
    "config_sha256": null,
    "adapter_commit": null,
    "runtime_lock_sha256": null,
    "quantization": "none_for_reference",
    "dtype": "float32_for_reference"
  },
  "layer_pairs": [],
  "foreign_window": {"sinks": 4, "recent": 128},
  "schedule": {"mode": "one_epoch_lag", "decode_tokens_per_epoch": 1},
  "transport": {"backend": "inproc", "queue_capacity": 2, "max_frame_bytes": 67108864},
  "budgets": {
    "approved_training_device_hours": null,
    "approved_training_token_presentations": null,
    "approved_inference_wall_seconds_per_trial": null,
    "approved_peak_memory_bytes_per_host": null
  },
  "channel_policy": {
    "model_readable_live_text": "forbidden",
    "task_derived_token_id_fields": "forbidden",
    "timing_and_length_policy": "fixed_schedule_declared_metadata",
    "e1_e2_peer_artifact_reads": "forbidden",
    "e4_peer_artifact_reads": "separately_metered_declared_channel",
    "audit_visibility": "experimenter_only"
  },
  "evidence_root": null,
  "series_rules_hash": null,
  "preregistration_hash": null,
  "host_integration": {
    "mcdma_repository_path": null,
    "mcdma_commit": null,
    "omp_repository_path": null,
    "omp_commit": null
  }
}
````

### Source 05 — `configs/stage-status.json`

<!-- file: configs/stage-status.json sha256: 4d8b34a534843911ef19b7a783e4a40ca6d3e9324861a8417f1e53ef585e3f5b -->
````json
{
  "reference_kit": "TESTED_CPU_SEE_VALIDATION",
  "M-1_contracts": "NOT_RUN_ON_USER_HOSTS",
  "M0_identity_real_checkpoints": "NOT_RUN",
  "M0_same_family_replication": "NOT_RUN",
  "M1_cross_family": "NOT_RUN",
  "M2_live_coupling": "NOT_RUN",
  "M3_mailbox_science": "NOT_RUN",
  "M4_mlx_mcdma": "BLOCKED_PENDING_LOCAL_INTEGRATION",
  "M5_duel": "NOT_RUN",
  "M6_competition": "NOT_RUN"
}
````

### Source 06 — `docs/agent-progress.md`

<!-- file: docs/agent-progress.md sha256: edb5ad277a83af24996a01d01b08e5409920f3f98ff4006fd358b81fdfa33548 -->
````markdown
# Agent progress

## Starting state

The supplied reference suite ran on CPU; see validation/. No user-host, real-model,
MLX, MCDMA, OMP or semantic-transfer gate is marked complete.

## Required update after each ticket

Stage / ticket:
Commit:
Files changed:
Commands run:
Tests passed / failed / skipped:
Evidence paths and SHA-256:
Training / inference / communication costs:
Engineering verdict:
Blocker and observed error:
Next ticket:
````

### Source 07 — `docs/original_plan.md`

<!-- file: docs/original_plan.md sha256: 5967d99624b96bb0748e2e99bc9ad00cab9b49a6366d33b04aab347b4f98ec3a -->
````markdown
# Telepathy: engineering plan

Two models, one project, one shared mind. Model A runs on the Studio, model B runs on a Spark, both work the same repo the way the OMP Duo room already does, except they never speak in tokens. Each reads the other's KV cache directly across the MCDMA link, each can leave a question inside its own cache for the other to find. Pure telepathy, on hardware you already built the pipe for.

The bridge layer inside keeps the anatomical name, the callosum, after the nerve that links the brain's hemispheres. The project is called what it is.

Related work that must be cited and beaten, not rediscovered: Cross-Model KV Cache Transfer in LLM Families (arXiv 2608.03893, Aug 2026, same-family prefill reuse, closed-form linear maps) and the Bicameral Model (Flamant, May 2026, two frozen same-model 0.5B streams coupled by a 6.2M-parameter interface). Neither is cross-family. Neither is at real scale. Neither has a public demo, a mailbox, or a head-to-head. That is the delta.

## 1. What exists and what is new

| Piece | Status | Role here |
|---|---|---|
| MCDMA | Built | Zero-copy memory windows between Studio (Metal) and Sparks (CUDA). The data plane. |
| OMP Duo plugin | Built | Two models, one repo, negotiated split, ASK/ANSWER over hub messages. The control plane and the text-only baseline arm. |
| telepathy-core | New | Activation taps, foreign-KV injection, bridge projectors, gates. The data plane's brain. |
| telepathy-mailbox | New | Questions and answers as pure thought, parked in KV. The novel protocol. |
| telepathy-eval | New | Telepathy, organism, mailbox-duel experiments plus degeneration detectors. The scorer. |
| omp-telepathy | New plugin | The room UI: audit view, mailbox latency board, run control. |

Sharp boundary to keep the claim honest: Duo cannot touch activations. It coordinates models through OMP's provider APIs. Everything below the coordination layer is new code, and the text-channel Duo room remains the baseline the telepathy arm must beat or match.

## 2. Architecture

```
[ Studio, Metal/MLX ]                      [ Spark, CUDA ]
  model A (frozen)                           model B (frozen)
   attention hooks ──┐                        ┌── attention hooks
                     │                        │
   ForeignKVInjector │   MCDMA zero-copy      │ ForeignKVInjector
   BridgeProjector <─┼──── memory windows ────┼→ BridgeProjector
   ThoughtWriter     │   (fallback: TCP)      │ ThoughtWriter
   Gate/Sync         ┘                        ┘ Gate/Sync

        control plane: OMP Duo room (hub messages, todo board, audit log)
        data plane:   KV tensors over MCDMA, no tokens, ever
```

Two planes, cleanly separated. The control plane stays OMP: room lifecycle, task split negotiation at session start (that can be text, it is setup), the audit log, the scoreboard. The data plane is every thought after the handshake: KV entries flow across the wire, nothing else does.

Hardware mapping. Model A and its projector kernels on the Studio GPU reading Spark-resident KV through the MCDMA window; model B mirrored on the Spark reading Studio KV the same way. If mutual reads through the window measure slower than expected, the transport layer swaps to async copy without changing any science code. The in-process backend runs both models on the Studio alone for development.

Model choices, cross-family by design: Qwen3-8B and GLM-4-9B or Llama-3.1-8B, whichever pair you stock locally, plus one same-family pair (Qwen3-4B to Qwen3-8B) reserved purely for the M0 replication. Different tokenizers are mandatory. Identical tokenizers invalidate the cross-family claim.

## 3. Component specs

### telepathy-core

`AttentionTap`: forward hooks on every attention block's KV path. Captures per-layer K and V after RoPE is applied (and again before, see PositionAligner). No model weights change, ever.

`ForeignKVInjector`: extends the attention context with projected foreign entries before the softmax. Three knobs, all logged: which layers are bridged (start with 8 of 32), the foreign window (sliding recent window plus sink entries, StreamingLLM-style), and whether injection is read-only (A observes B) or bidirectional.

`BridgeProjector`: per-layer learned map from A's (n_kv_heads_A, head_dim_A) to B's (n_kv_heads_B, head_dim_B). MLP with layer-norm, a few million parameters total, same spirit as the Bicameral interface but cross-family. GQA head-count mismatch handled by a learned head-mixing matrix, not naive repeat-interleave, and the naive version is kept as a reported baseline because the same-family papers claim closed-form linear maps suffice. If a ridge-regression map gets 90 percent of the learned projector's accuracy, that is a finding, publish it as one.

`PositionAligner`: the nastiest correctness detail. Foreign entries carry A's rotary phase. Two strategies, both implemented: re-encode foreign entries at B's RoPE base with A's relative distances preserved, or learn a phase remap. Attention sinks (the first few tokens models lean on) are never dropped and never reordered.

`Gate`: per-layer learned scalar gate on foreign attention weight, initialized near zero and trained open. This is the firewall against thought contagion. The ARC trials showed naked models imitate whatever frame pattern dominates their input; the gate is the mechanical answer, and its per-layer trajectory during training is itself a chart worth posting.

`SyncController`: lockstep, chunked, or sliding modes. Bandwidth budget below. Full-cache sync only at handoff events; steady-state sync is per-step deltas.

### telepathy-mailbox (the novel part)

A question is a thought aimed at someone. Mechanics:

1. `ThoughtWriter` forces a short token sequence ("Q: does the parser handle nested unions?") through model A's input embedding path while suppressing emission to any user-facing stream. Those tokens now have live KV entries in A's cache. A has not spoken. A has thought at B.
2. A learned marker entry (one soft token, trained for salience) prefixes the thought so B's attention treats it as addressed mail rather than background reasoning.
3. The bridge projects the marker plus the thought into B's cache. B attends, reads, and answers the same way: B thinks the answer at A. The conversation never becomes text.
4. There is no delivery step, because nothing is ever sent. Reception is attention: B thinks the answer, the answer lives in B's cache, and A, already attending to that cache, simply knows, the way you notice a friend nod without being mailed a nod. So the measurable event is noticing, and it has a causal test: track A's attention mass on the foreign marker and answer entries, then ablate those entries and verify A's behavior reverts to the pre-answer distribution. If masking the answer makes A act uninformed, the thought demonstrably crossed. Metrics: noticing rate (fraction of answers A demonstrably used), incorporation latency (decode steps between the answer entering B's cache and A's behavior reflecting it), and ablation delta (how much A's output depends on the foreign entries).
5. Starvation watchdog, reframed: if A's attention never lands on a marker, or lands but incorporation never follows, the mailbox logs an unnoticed question. Unnoticed mail is a first-class metric, not an anomaly. Attention without incorporation is its own interesting failure: the model glanced at the letter and did not read it.

Delivery verification stays outside the minds: `ProbeHead`, a small linear probe trained per model, reads each cache on the experimenter's side and reconstructs what thoughts are present, purely for the audit log. The probes see the caches; neither model sees the probes' output. This keeps scoring external per the series rules while the collaboration channel itself stays token-free.

Failure modes to instrument from day one: echo lock (both models re-attending each other's marker in a loop, counted by marker-attention oscillation), question storms (mailbox fill rate exceeding drain rate), and silent starvation.

### telepathy-train

Alignment corpus: real repo-shaped data, since the deployment scenario is two models building one project. Code, docs, issues, and genuine Q&A pairs over both models' tokenizers in parallel. Losses: projected-KV regression against the receiver's own activations on the same text, plus a behavior-matching term, KL between receiver-with-injected-KV and receiver-with-own-KV on next-token distributions. Backbones frozen throughout. Training budget capped and reported per the series accounting rules. The projector is the only trainable thing, so this fits comfortably on the Sparks in an afternoon per run.

### telepathy-eval

Four experiments in escalation order, each with a pre-registered pass bar:

E1 Handoff. A reads a document, B inherits only the bridged cache, B answers. Floor: no context. Ceiling: B re-reads the text. Baseline: equal-length English summary. The transplant demo.

E2 Telepathy. B holds a secret A never saw as text. Coupled live, can A answer questions about it? Pass bar: statistically above the no-context floor over 100 held-out secrets. This is the clip that travels.

E3 Organism. Reasoning suite: A solo, B solo, coupled pair, and the pair talking in plain English with the same compute. If coupled beats max(solo A, solo B) it is a two-brain organism. If English-chat wins, that is the upset result and it still airs.

E4 Mailbox duel. The Duo scenario end to end: same repo, same task backlog, two arms. Text arm is the existing Duo room with ASK/ANSWER. Telepathy arm is the coupled room with mailbox only. Score: task completion, wall clock, tokens on the collaboration channel (must be zero, verified by wire capture), noticing rate, incorporation latency, and the ablation delta proving the thoughts were used. The headline if it holds: the pair built the feature without exchanging a single word.

Degeneration detectors run through everything: echo index, cache drift, gate saturation, storm counters. The contagion reel is content by design.

## 4. Transport and the bandwidth math

Per-token KV footprint for an 8B-class model, 32 layers, GQA 8 heads, head_dim 128, fp16: 4 KB per layer, 128 KB per token all layers. Steady-state delta sync at 10 tok/s each way is roughly 2.5 MB/s. MCDMA shrugs at that. Full 4k-context sync at handoff is 512 MB per direction, a one-time cost, and bridging only 8 of 32 layers cuts everything by 4. The transport is not the bottleneck; foreign attention compute is. That is why the foreign window slides and why bridge-layer count is a competitor-tunable knob.

Transport backends behind one interface: inproc (dev, both models in one Studio process), tcp (correctness fallback if the window reads misbehave), mcdma (target, zero-copy, instrumented). The MCDMA swap-in milestone exists to produce the pipe numbers: round-trip KV read latency, projected entries per second, joules per synced thought. Those numbers are an episode on their own for your audience.

## 5. OMP integration

New plugin, `omp-telepathy`, modeled on Duo's host API usage. It does not replace Duo; it runs beside it as the other arm. Room surface: `/telepathy start <pair>`, the audit view rendering ProbeHead reconstructions of what each mind currently holds, the mailbox board (delivered, pending, starved, latency histogram), gate trajectories, and `/telepathy duel <task>` launching the E4 two-arm run with Duo on the text side. Session persistence and restart semantics follow Duo's patterns. The repo, todos, and negotiation-at-start all stay in OMP where they already work.

## 6. Milestones

M0, replication gate (week 1). Same-family pair, in-process, ridge-regression projector, E1 only. Accept: beats the no-context floor, reproducing the published linear-structure claim on your hardware. If this fails, stop and debug the harness before touching anything novel.

M1, cross-family handoff (weeks 2 to 3). Learned projector, mismatched tokenizers and head shapes, E1 plus E2. Accept: E2 above floor. First postable chart either way.

M2, live coupling (weeks 3 to 4). Bidirectional injection, gates, sync controller, E3 with the English-chat control. Accept: gates train stable, echo index bounded, organism comparison measured.

M3, mailbox (week 5). ThoughtWriter, markers, probes, delivery metrics, E4 in-process. Accept: nonzero delivery rate with latency and starvation honestly reported. Even 60 percent delivery is a result; the paper says zero.

M4, MCDMA swap-in (week 6). Transport backend switch, zero-copy windows, pipe instrumentation. Accept: identical E1/E2 scores across backends within noise, plus the latency/joule table.

M5, episode package (week 7). Full duel across the wire, degeneration reel, matrix if a third model joins. Pre-registered bars, receipts post.

M6, competition harness (week 8+). The Invention Trials format: 4 to 5 competitor agents each design projector, gate, sync, and mailbox policies between two frozen models with a fixed training budget. Public development pairs, hidden eval pair. Scoreboard: E1 to E4 plus delivery rate and cost. This becomes the series brief.

## 7. Risks

| Risk | Mitigation |
|---|---|
| RoPE phase mismatch corrupts reads | PositionAligner with both strategies; sink tokens preserved; E1 catches it early |
| GQA head mismatch loses information | Learned head-mixing; report against repeat-interleave baseline |
| Echo/imitation attractors collapse both minds | Gates initialized closed; storm watchdog; the failure itself is logged content, ARC precedent documented |
| Foreign attention compute blows the step budget | Sliding foreign window, bridge-layer subset, chunked sync |
| vLLM-style paged caches make injection miserable | Do not use them. Transformers/MLX with hooks for all science; speed comes from MCDMA reads, not serving-stack tricks |
| Mailbox never delivers | Starvation as a metric; marker salience training; if thought-mail fails, the honest fallback finding is "coupling works, addressing does not", still novel |
| MCDMA window reads add latency | tcp fallback preserves the science; backend equivalence test in M4 |

## 8. Validity and accounting, per SERIES.md

Scorer, probes, and reference answers outside both models' writable reach. Wire capture proves the zero-token claim rather than trusting it. Frozen checkpoints, quantization, seeds, and training budgets declared per run. Repeated trials with variance, pre-registered pass bars, and the four-verdict taxonomy. Development pairs public, evaluation pair hidden and frozen before M5.

## 9. Repo layout

```
telepathy/
  core/        tap, injector, projector, aligner, gate, sync
  mailbox/     thought writer, markers, probes, watchdog
  train/       alignment data build, projector training, budget ledger
  eval/        E1-E4, detectors, matrix runner, manifests
  transport/   inproc | tcp | mcdma backends, instrumentation
  plugin/      omp-telepathy
  results/     runs, charts, degeneration reels, episode packages
```

Build order within M1: core with the inproc backend and E1, everything else hangs off those two. The first honest number you want is the ridge-projector same-family E1 score. Everything else is that number, made stranger.
````

### Source 08 — `pyproject.toml`

<!-- file: pyproject.toml sha256: fe6acf92ede0f18aab2b16e6184559d387af89646ad6081c9f3aa593cc3ad298 -->
````toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "telepathy-reference"
version = "0.2.0"
description = "Frozen-model KV collaboration: correctness reference and staged agent handoff"
requires-python = ">=3.11,<3.14"
dependencies = ["numpy==2.3.5", "torch==2.10.0", "safetensors==0.7.0"]

[project.optional-dependencies]
test = ["pytest==9.0.2"]
hf = ["transformers==4.56.2"]

[tool.setuptools.packages.find]
include = ["telepathy*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra"
````

### Source 09 — `scripts/doctor.py`

<!-- file: scripts/doctor.py sha256: b9dcc5c54fd24ed17956292fd380b3818f02352740a64546c3722edff1f38e12 -->
````python
"""Read-only environment/config inventory. Does not load drivers or download weights."""
import argparse
import json
from pathlib import Path
from telepathy.runtime.manifest import inventory

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=Path)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
args.out.parent.mkdir(parents=True, exist_ok=True)
if args.out.exists():
    parser.error("output exists; choose a new audit file")
args.out.write_text(json.dumps(inventory(args.model), indent=2) + "\n")
print(args.out)
````

### Source 10 — `scripts/extract_alignment.py`

<!-- file: scripts/extract_alignment.py sha256: c7c6faac1ece2ecb1601947998cb781a3d1224772bad438bfad981f204c270a3 -->
````python
"""Offline, training-only extraction for ONE explicit directional layer pair."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from telepathy.runtime.decoder import FrozenDecoder
from telepathy.runtime.manifest import inventory, sha256_file
from telepathy.train.alignment import shared_causal_endpoints

parser = argparse.ArgumentParser()
parser.add_argument("--source-model", type=Path, required=True)
parser.add_argument("--target-model", type=Path, required=True)
parser.add_argument("--source-layer", type=int, required=True)
parser.add_argument("--target-layer", type=int, required=True)
parser.add_argument("--input", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--device", default="cpu")
parser.add_argument("--max-tokens", type=int, default=512)
parser.add_argument("--max-rows", type=int, default=8192)
args = parser.parse_args()
if args.max_tokens < 2 or args.max_rows < 2:
    parser.error("positive nontrivial token/row budgets required")
if args.out.exists() or args.out.with_suffix(".json").exists():
    parser.error("output exists")
from transformers import AutoTokenizer
source = FrozenDecoder.from_local_hf(args.source_model, args.device)
target = FrozenDecoder.from_local_hf(args.target_model, args.device)
if not 0 <= args.source_layer < len(source.layers) or not 0 <= args.target_layer < len(target.layers):
    parser.error("layer outside checkpoint")
tokenizers = [AutoTokenizer.from_pretrained(str(path), local_files_only=True,
                                            trust_remote_code=False, use_fast=True)
              for path in (args.source_model, args.target_model)]
if not all(t.is_fast for t in tokenizers):
    parser.error("validated original-text offsets require a fast tokenizer")
collected = {name: [] for name in ("source_k", "source_v", "target_k", "target_v")}
seen, accepted, count, skipped = 0, 0, 0, []
with args.input.open() as handle, torch.no_grad():
    for line in handle:
        record = json.loads(line)
        if set(record) != {"id", "text", "split"} or record["split"] != "train":
            parser.error("training JSONL requires exactly id/text/split with split=train")
        seen += 1
        text = record["text"]
        batches = [t(text, add_special_tokens=False, return_offsets_mapping=True) for t in tokenizers]
        lengths = [len(x["input_ids"]) for x in batches]
        if min(lengths) == 0 or max(lengths) > args.max_tokens:
            skipped.append({"id": record["id"], "reason": "token_budget_or_empty"}); continue
        for tokenizer, batch in zip(tokenizers, batches):
            decoded = tokenizer.decode(batch["input_ids"], skip_special_tokens=False,
                                       clean_up_tokenization_spaces=False)
            if decoded != text:
                parser.error("tokenizer normalization/round-trip differs: implement and audit a text-coordinate adapter")
        aligned = shared_causal_endpoints(text, batches[0]["offset_mapping"], batches[1]["offset_mapping"])
        aligned = aligned[:args.max_rows - count]
        if not aligned:
            skipped.append({"id": record["id"], "reason": "no_shared_endpoint"}); continue
        outs = [worker.forward(torch.tensor(batch["input_ids"], dtype=torch.long))
                for worker, batch in zip((source, target), batches)]
        for side, layer, column, output in (("source", args.source_layer, 0, outs[0]),
                                            ("target", args.target_layer, 1, outs[1])):
            rows = torch.tensor([pair[column] for pair in aligned], device=output.logits.device)
            kv = output.canonical_delta[layer]
            for field in ("k", "v"):
                collected[f"{side}_{field}"].append(getattr(kv, field)[rows].detach().cpu().numpy())
        count += len(aligned); accepted += 1
        if count >= args.max_rows:
            break
if count < 2:
    parser.error("insufficient causally aligned rows")
args.out.parent.mkdir(parents=True, exist_ok=True)
with args.out.open("wb") as handle:
    np.savez(handle, **{name: np.concatenate(values) for name, values in collected.items()})
manifest = {"mode": "offline_kv_regression", "text_mode": "raw_no_chat_template",
            "split": "train", "input_sha256": sha256_file(args.input),
            "array_sha256": sha256_file(args.out), "source_layer": args.source_layer,
            "target_layer": args.target_layer, "rows": count, "records_seen": seen,
            "records_accepted": accepted, "skipped": skipped,
            "source": inventory(args.source_model), "target": inventory(args.target_model)}
args.out.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
print(args.out)
````

### Source 11 — `scripts/fit_projector.py`

<!-- file: scripts/fit_projector.py sha256: fa40bc02fa13b495e0587849b866df9f27c5fa176874863b3cb959ce4657e615 -->
````python
"""Fit ONE directional layer pair from already aligned training-only arrays."""
import argparse
from pathlib import Path
import numpy as np
import torch
from telepathy.core.types import KV
from telepathy.core.projector import BridgeProjector
from telepathy.train.fit import fit_mlp, save_projector
from telepathy.runtime.manifest import sha256_file

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--kind", choices=["ridge", "mlp"], default="ridge")
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--seed", type=int, default=11)
args = parser.parse_args()
with np.load(args.data, allow_pickle=False) as file:
    if set(file.files) != {"source_k", "source_v", "target_k", "target_v"}:
        parser.error("expected exactly source_k/source_v/target_k/target_v")
    tensors = {name: torch.from_numpy(file[name].astype(np.float32)) for name in file.files}
source, target = KV(tensors["source_k"], tensors["source_v"]), KV(tensors["target_k"], tensors["target_v"])
source.check(); target.check()
torch.manual_seed(args.seed)
projector = BridgeProjector(tuple(source.k.shape[1:]), tuple(target.k.shape[1:]), args.kind)
if args.kind == "ridge":
    projector.fit(source, target)
    report = {"kind": "ridge_kv_regression", "training_rows": source.tokens}
else:
    report = fit_mlp(projector, source, target, steps=args.steps, seed=args.seed)
report.update({"data_sha256": sha256_file(args.data), "seed": args.seed,
               "behavior_distillation": "NOT_RUN", "semantic_transfer": "NOT_EVALUATED"})
save_projector(projector, args.out, report)
print(args.out)
````

### Source 12 — `scripts/materialize.py`

<!-- file: scripts/materialize.py sha256: 4c2a5d6261d11ff7f7137fd9b23c8c8761207bdaa323dde8528f6e7b56c98997 -->
````python
"""Extract hashed source blocks from the standalone engineering Markdown file.

Usage: python materialize.py TELEPATHY_AGENT_ENGINEERING.md ./telepathy
Refuses nonempty destinations, path traversal, duplicate files and bad hashes.
"""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path, PurePosixPath
import re


def extract(document: Path, destination: Path) -> int:
    text = document.read_text(encoding="utf-8")
    pattern = re.compile(
        r'<!-- file: ([^\n]+) sha256: ([0-9a-f]{64}) -->\n'
        r'````[^\n]*\n(.*?)\n````(?=\n)', re.DOTALL)
    files = {}
    for match in pattern.finditer(text):
        name, expected, body = match.groups()
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name or not name:
            raise ValueError(f"unsafe source path: {name}")
        if name in files:
            raise ValueError(f"duplicate source: {name}")
        payload = (body + "\n").encode("utf-8")
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError(f"source hash mismatch: {name}")
        files[name] = payload
    if not files:
        raise ValueError("no extractable source blocks")
    if destination.is_symlink() or (destination.exists() and any(destination.iterdir())):
        raise FileExistsError("destination must be a new or empty nonsymlink directory")
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    for name, payload in files.items():
        target = root.joinpath(*PurePosixPath(name).parts)
        if not target.resolve().is_relative_to(root):
            raise ValueError("source escapes destination")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    return len(files)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("document", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(f"Extracted {extract(args.document, args.destination)} verified files")
````

### Source 13 — `scripts/reference_demo.py`

<!-- file: scripts/reference_demo.py sha256: 355d3be8862c7eb2d4ac07551d2de06248f4e5d6b4219a37b7dcd11b25317551 -->
````python
"""Synthetic affine-map + framed-transfer smoke test, not a language experiment."""
import argparse
import json
from pathlib import Path
from uuid import UUID
import torch
from telepathy.core.types import Delta, KV
from telepathy.core.projector import BridgeProjector
from telepathy.core.memory import ForeignKVBank
from telepathy.transport.wire import FrameCodec
from telepathy.transport.backends import InProcTransport

parser = argparse.ArgumentParser()
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
if args.out.exists(): parser.error("output exists")
torch.set_num_threads(1); torch.manual_seed(91)
src = KV(torch.randn(256, 2, 4), torch.randn(256, 2, 4))
wk, wv = torch.randn(8, 6), torch.randn(8, 6)
dst = KV((src.k.flatten(1) @ wk + 0.5).reshape(256, 1, 6),
         (src.v.flatten(1) @ wv - 0.2).reshape(256, 1, 6))
projector = BridgeProjector((2, 4), (1, 6)); projector.fit(src, dst, ridge=1e-6)
heldout = KV(torch.randn(16, 2, 4), torch.randn(16, 2, 4))
expected_k = (heldout.k.flatten(1) @ wk + 0.5).reshape(16, 1, 6)
session = UUID(int=91); frame = Delta(session, 0, 0, 0, 0, {0: heldout})
codec = FrameCodec(b"reference-demo-only-key-not-for-real-runs")
transport = InProcTransport(codec); transport.send(frame)
bank = ForeignKVBank(session, 0, [(0, 0, projector)], sinks=2, recent=32)
bank.commit(transport.receive())
error = float((bank.pin().layers[0].k - expected_k).abs().max())
report = {"fixture": "synthetic_affine_mapping", "max_abs_key_error": error,
          "status": "PASSED" if error < 1e-4 else "FAILED",
          "frame_bytes_including_hmac": len(codec.encode(frame)),
          "tensor_payload_bytes_float32": sum(x.k.numel() * 8 for x in frame.layers.values()),
          "receiver_resident_slots": bank.pin().positions.numel(),
          "cross_family_language_transfer": "NOT_EVALUATED",
          "mcdma": "NOT_USED", "energy_joules": None}
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
raise SystemExit(0 if report["status"] == "PASSED" else 1)
````

### Source 14 — `scripts/run_e1_diagnostic.py`

<!-- file: scripts/run_e1_diagnostic.py sha256: 3d9f3e445d37d5ecbeaf99ca3264a61d7b164e1364b3c2b8e249d90c4b2aab89 -->
````python
"""Small offline E1 DIAGNOSTIC, one mapped layer; NOT a preregistered science run.

Input fields: id, context, question. No answer key. Outputs floor, reread ceiling,
foreign-memory, hard-closed gate, and wrong-context control. Raw text prompts are
explicitly a diagnostic formatting choice. OMP, mailbox, and MCDMA are not used.
"""
import argparse
import json
from pathlib import Path
from uuid import uuid4
import torch
from telepathy.core.types import Delta
from telepathy.core.memory import ForeignKVBank
from telepathy.runtime.decoder import FrozenDecoder
from telepathy.train.fit import load_projector
from telepathy.runtime.manifest import sha256_file

parser = argparse.ArgumentParser()
parser.add_argument("--source-model", type=Path, required=True)
parser.add_argument("--target-model", type=Path, required=True)
parser.add_argument("--source-layer", type=int, required=True)
parser.add_argument("--target-layer", type=int, required=True)
parser.add_argument("--projector", type=Path, required=True)
parser.add_argument("--input", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--device", default="cpu")
parser.add_argument("--new-tokens", type=int, default=32)
parser.add_argument("--foreign-window", type=int, default=128)
parser.add_argument("--source-max-tokens", type=int, default=512)
args = parser.parse_args()
if args.out.exists():
    parser.error("output exists")
from transformers import AutoTokenizer
source = FrozenDecoder.from_local_hf(args.source_model, args.device)
target = FrozenDecoder.from_local_hf(args.target_model, args.device)
projector = load_projector(args.projector, args.device)
if not 0 <= args.source_layer < len(source.layers) or not 0 <= args.target_layer < len(target.layers):
    parser.error("invalid layer pair")
if projector.source != (source.kvheads, source.dim) or projector.target != (target.kvheads, target.dim):
    parser.error("projector/checkpoint shape mismatch")
st, tt = [AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
          for path in (args.source_model, args.target_model)]
records = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
if len(records) < 2 or any(set(row) != {"id", "context", "question"} for row in records):
    parser.error("at least two rows, exactly id/context/question; keep answer keys outside this runner")
if len({row["id"] for row in records}) != len(records):
    parser.error("duplicate example IDs")

def tokens(tokenizer, text):
    return torch.tensor(tokenizer.encode(text, add_special_tokens=False), dtype=torch.long)

@torch.no_grad()
def bank_for(text):
    ids = tokens(st, text)
    if not 1 <= ids.numel() <= args.source_max_tokens:
        raise ValueError("source context exceeds declared budget; no silent truncation")
    out = source.forward(ids)
    session = uuid4()
    bank = ForeignKVBank(session, 0, [(args.source_layer, args.target_layer, projector)],
                         sinks=4, recent=args.foreign_window)
    bank.commit(Delta(session, 0, 0, 0, 0, {args.source_layer: out.canonical_delta[args.source_layer]}))
    return bank.pin()

args.out.parent.mkdir(parents=True, exist_ok=True)
with args.out.open("x") as output:
    for index, row in enumerate(records):
        question = "Question: " + row["question"] + "\nAnswer:"
        prompt = tokens(tt, question)
        bank, wrong = bank_for(row["context"]), bank_for(records[(index + 1) % len(records)]["context"])
        arms = {"floor": (prompt, None, 0.0),
                "reread_ceiling": (tokens(tt, row["context"] + "\n" + question), None, 0.0),
                "foreign": (prompt, bank, 1.0), "gate_closed": (prompt, bank, 0.0),
                "wrong_context": (prompt, wrong, 1.0)}
        result = {}
        for name, (ids, view, gate) in arms.items():
            generated = target.generate(ids, args.new_tokens, view, gate, tt.eos_token_id)
            result[name] = {"text": tt.decode(generated, skip_special_tokens=True),
                            "receiver_prompt_tokens": ids.numel(), "generated_tokens": len(generated)}
        if result["floor"]["text"] != result["gate_closed"]["text"]:
            raise AssertionError("hard-closed gate changed output")
        output.write(json.dumps({"id": row["id"], "mode": "E1_DIAGNOSTIC_NOT_PREREGISTERED",
                                 "arms": result}) + "\n")
meta = {"scientific_status": "NOT_ESTABLISHED", "backend": "inproc_direct_bank",
        "input_sha256": sha256_file(args.input), "output_sha256": sha256_file(args.out),
        "source_layer": args.source_layer, "target_layer": args.target_layer,
        "missing_for_formal_E1": ["complete model/run pin audit", "text-summary arm",
                                  "full cost ledger", "process isolation audit",
                                  "preregistered independent-unit scoring"]}
args.out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
print(args.out)
````

### Source 15 — `scripts/score_e1.py`

<!-- file: scripts/score_e1.py sha256: 4edc11a9d60e5a6aa08f00f881aa582b38866c2674969094d87229f8a39d6a05 -->
````python
"""External exact-match scorer. Never route this output back to either model."""
import argparse
import json
from pathlib import Path
from telepathy.eval.metrics import paired_bootstrap

parser = argparse.ArgumentParser()
parser.add_argument("--predictions", type=Path, required=True)
parser.add_argument("--answers", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
if args.out.exists():
    parser.error("output exists")
answer_rows = [json.loads(line) for line in args.answers.read_text().splitlines() if line.strip()]
answers = {r["id"]: r["answers"] for r in answer_rows}
if len(answers) != len(answer_rows) or any(set(r) != {"id", "answers"} for r in answer_rows):
    parser.error("answer JSONL needs unique id/answers rows")
rows = [json.loads(line) for line in args.predictions.read_text().splitlines() if line.strip()]
if len(rows) != len(answers) or {r["id"] for r in rows} != set(answers):
    parser.error("missing, duplicate or extra trials; reconcile failures explicitly")
if len({r["id"] for r in rows}) != len(rows):
    parser.error("duplicate predictions")
score = lambda text, accepted: float(text.strip().casefold() in {s.strip().casefold() for s in accepted})
active = [score(r["arms"]["foreign"]["text"], answers[r["id"]]) for r in rows]
floor = [score(r["arms"]["floor"]["text"], answers[r["id"]]) for r in rows]
summary = paired_bootstrap(active, floor)
summary["status"] = "DIAGNOSTIC_ONLY_NOT_FORMAL_E1_OR_E2"
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(summary, indent=2) + "\n")
print(args.out)
````

### Source 16 — `scripts/self_handoff.py`

<!-- file: scripts/self_handoff.py sha256: 750a312ac66ea1ecd2311281f3f2f0aa9bd798a95bb4b33e72e6f3905fdc0fdb -->
````python
"""M0 identity correctness gate. Does not establish same-family/cross-family gains."""
import argparse
import json
from pathlib import Path
import torch
from telepathy.runtime.decoder import FrozenDecoder
from telepathy.runtime.toy import ToyModel
from telepathy.runtime.manifest import parameter_digest

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=Path)
parser.add_argument("--device", default="cpu")
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
if args.out.exists():
    parser.error("output exists; do not overwrite evidence")
torch.set_num_threads(1)
worker = (FrozenDecoder(ToyModel()) if args.model is None
          else FrozenDecoder.from_local_hf(args.model, args.device))
ids = torch.tensor([1, 7, 3, 11, 2, 5, 13, 17], dtype=torch.long)
before = parameter_digest(worker.model)
with torch.no_grad():
    prefix = worker.forward(ids[:5])
    ordinary = worker.forward(ids[5:], prefix.state)
    imported = worker.import_self_prefix(prefix.canonical_delta, torch.arange(5))
    transplant = worker.forward(ids[5:], imported)
    full = worker.forward(ids)
    errors = {"self_handoff_max_abs": float((ordinary.logits - transplant.logits).abs().max()),
              "prefill_vs_incremental_max_abs": float((full.logits[5:] - ordinary.logits).abs().max())}
    if args.model is not None:
        stock = worker.model(input_ids=ids[None].to(worker.device), use_cache=False).logits[0]
        errors["stock_hf_forward_max_abs"] = float((stock - full.logits).abs().max())
unchanged = before == parameter_digest(worker.model)
passed = max(errors.values()) < 2e-5 and unchanged
report = {"fixture": "random_toy" if args.model is None else "local_checkpoint",
          "status": "PASSED" if passed else "FAILED", "errors": errors,
          "weights_unchanged": unchanged, "semantic_transfer": "NOT_EVALUATED"}
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
raise SystemExit(0 if passed else 1)
````

### Source 17 — `scripts/validate_manifest.py`

<!-- file: scripts/validate_manifest.py sha256: 81c1473461e446a2765411e0ef1804e1f7ce4ba23b8652f8f12788e7d71e36e1 -->
````python
"""Completeness guard, not a replacement for scientific/permission review."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("manifest", type=Path)
args = parser.parse_args()
manifest = json.loads(args.manifest.read_text())
missing = []

def walk(value, path="run"):
    if value is None or value == "UNRESOLVED":
        missing.append(path)
    elif isinstance(value, dict):
        for key, child in value.items(): walk(child, path + "." + key)
    elif isinstance(value, list):
        for index, child in enumerate(value): walk(child, f"{path}[{index}]")

walk(manifest)
required = {"schema", "run_uuid", "model_a", "model_b", "layer_pairs", "budgets",
            "channel_policy", "evidence_root", "series_rules_hash", "preregistration_hash"}
missing += ["run." + key for key in sorted(required - manifest.keys())]
if not manifest.get("layer_pairs"):
    missing.append("run.layer_pairs(nonempty)")
print(json.dumps({"status": "BLOCKED" if missing else "COMPLETE_NOT_YET_VALIDATED",
                  "missing": missing}, indent=2))
raise SystemExit(2 if missing else 0)
````

### Source 18 — `telepathy/__init__.py`

<!-- file: telepathy/__init__.py sha256: 90cc787c8c832fc41f564f76a60d997bc4bbda2ea7c24afd8e0a43eea0ec3ebd -->
````python
"""Telepathy reference implementation; see TELEPATHY_AGENT_ENGINEERING.md."""
````

### Source 19 — `telepathy/core/__init__.py`

<!-- file: telepathy/core/__init__.py sha256: 90cc787c8c832fc41f564f76a60d997bc4bbda2ea7c24afd8e0a43eea0ec3ebd -->
````python
"""Telepathy reference implementation; see TELEPATHY_AGENT_ENGINEERING.md."""
````

### Source 20 — `telepathy/core/attention.py`

<!-- file: telepathy/core/attention.py sha256: 392db600bd21c0f120c7daf7dad332cad86e74965d0f21f94a05db64ef7cbca7 -->
````python
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
from .types import KV


class Gate(nn.Module):
    def __init__(self, initial_logit: float = -6.0):
        super().__init__()
        self.logit = nn.Parameter(torch.tensor(float(initial_logit)))


def expand_gqa(x: torch.Tensor, query_heads: int) -> torch.Tensor:
    if x.ndim != 3 or query_heads % x.shape[1]:
        raise ValueError("receiver query heads must be a multiple of receiver KV heads")
    # This is receiver-native GQA expansion, NOT a cross-family projector.
    return x.repeat_interleave(query_heads // x.shape[1], dim=1)


def attend(q: torch.Tensor, native_rotated: KV, allowed_native: torch.Tensor,
           foreign_rotated: KV | None = None, gate: Gate | None = None,
           override: float | None = None,
           allowed_foreign: torch.Tensor | None = None,
           scale: float | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Eager, batch-one attention with foreign PRIOR inside the shared softmax.

    q:[Q,Hq,D], KV:[T,Hkv,D], boolean masks:[Q,T], True=allowed.
    override=0 is an EXACT native-only path; override=1 removes gate suppression.
    A zero value tensor is not a valid ablation: its logits would still compete.
    Returned foreign mass:[Q,Hq], diagnostic only, never proof of incorporation.
    """
    if q.ndim != 3 or not torch.isfinite(q).all():
        raise ValueError("invalid query")
    native_rotated.check()
    nq, hq, d = q.shape
    if allowed_native.dtype != torch.bool or allowed_native.shape != (nq, native_rotated.tokens):
        raise ValueError("native mask must be boolean [Q,T]")
    if not bool(allowed_native.any(dim=-1).all()):
        raise ValueError("each query needs at least one permitted native key")
    if native_rotated.k.shape[-1] != d:
        raise ValueError("native head dimension mismatch")
    kn = expand_gqa(native_rotated.k, hq)
    vn = expand_gqa(native_rotated.v, hq)
    multiplier = (1 / math.sqrt(d)) if scale is None else scale
    native_scores = torch.einsum("qhd,thd->hqt", q, kn).float() * multiplier
    native_scores = native_scores.masked_fill(~allowed_native[None], -torch.inf)
    use_foreign = foreign_rotated is not None and override != 0.0
    if override is not None and not 0.0 <= override <= 1.0:
        raise ValueError("gate override must be in [0,1]")
    if use_foreign:
        foreign_rotated.check()
        if foreign_rotated.k.shape[1:] != native_rotated.k.shape[1:]:
            raise ValueError("project foreign KV into RECEIVER KV shape first")
        nf = foreign_rotated.tokens
        if allowed_foreign is None:
            allowed_foreign = torch.ones((nq, nf), dtype=torch.bool, device=q.device)
        if allowed_foreign.dtype != torch.bool or allowed_foreign.shape != (nq, nf):
            raise ValueError("foreign mask must be boolean [Q,Tf]")
        if not bool(allowed_foreign.any()):
            use_foreign = False
    if not use_foreign:
        weights = native_scores.softmax(dim=-1).to(q.dtype)
        output = torch.einsum("hqt,thd->qhd", weights, vn)
        return output, torch.zeros((nq, hq), device=q.device, dtype=q.dtype)
    if override is None and gate is None:
        raise ValueError("foreign attention requires a gate or explicit override")
    log_prior = (math.log(override) if override is not None
                 else F.logsigmoid(gate.logit).float())
    kf = expand_gqa(foreign_rotated.k, hq)
    vf = expand_gqa(foreign_rotated.v, hq)
    foreign_scores = torch.einsum("qhd,thd->hqt", q, kf).float() * multiplier + log_prior
    foreign_scores = foreign_scores.masked_fill(~allowed_foreign[None], -torch.inf)
    weights = torch.cat((native_scores, foreign_scores), dim=-1).softmax(-1).to(q.dtype)
    nnative = native_rotated.tokens
    output = (torch.einsum("hqt,thd->qhd", weights[..., :nnative], vn)
              + torch.einsum("hqt,thd->qhd", weights[..., nnative:], vf))
    return output, weights[..., nnative:].sum(-1).transpose(0, 1)
````

### Source 21 — `telepathy/core/memory.py`

<!-- file: telepathy/core/memory.py sha256: 3cc18969ddb477be6d835a0c82922a97760eaaac2c8e223a738b78472dc8056f -->
````python
from __future__ import annotations
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping
from uuid import UUID
import torch
from .types import Delta, KV
from .projector import BridgeProjector


@dataclass(frozen=True)
class ForeignView:
    epoch: int
    positions: torch.Tensor
    layers: Mapping[int, KV]         # RECEIVER layer numbers


class ForeignKVBank:
    """Inference-only, receiver-local, copy-on-publication foreign memory.

    Each delta is projected once. Commit is atomic across selected layers. Readers
    pin the resulting view for an entire forward/decode step and NEVER mutate it.
    This reference is single-threaded; a production worker needs a lock/RCU handoff.
    """
    def __init__(self, session: UUID, direction: int,
                 pairs: list[tuple[int, int, BridgeProjector]],
                 sinks: int = 4, recent: int = 128):
        if not pairs or sinks < 0 or recent <= 0 or direction not in (0, 1):
            raise ValueError("invalid bank configuration")
        if len({t for _, t, _ in pairs}) != len(pairs):
            raise ValueError("each receiver layer needs exactly one selected source")
        self.session, self.direction = session, direction
        self.pairs, self.sinks, self.recent = tuple(pairs), sinks, recent
        self.next_sequence, self.next_start, self.last_epoch = 0, 0, -1
        self._view: ForeignView | None = None

    @torch.no_grad()
    def commit(self, delta: Delta) -> None:
        delta.check()
        if delta.session != self.session or delta.direction != self.direction:
            raise ValueError("wrong run/session or direction")
        if delta.sequence != self.next_sequence or delta.start != self.next_start:
            raise ValueError("duplicate, gap, reorder, or missing full-reset handshake")
        if delta.epoch <= self.last_epoch:
            raise ValueError("nonmonotonic publication epoch")
        expected = {s for s, _, _ in self.pairs}
        if set(delta.layers) != expected:
            raise ValueError("publication is missing or adding a bridged layer")
        converted: dict[int, KV] = {}
        for source, target, projector in self.pairs:
            kv = delta.layers[source]
            parameter = next(projector.parameters())
            kv = KV(kv.k.to(parameter), kv.v.to(parameter))
            projected = projector(kv)
            projected.check()
            converted[target] = projected.clone()
        device = next(iter(converted.values())).k.device
        positions = torch.arange(delta.start, delta.start + delta.tokens, device=device)
        if self._view is not None:
            positions = torch.cat((self._view.positions, positions))
            converted = {t: KV(torch.cat((self._view.layers[t].k, kv.k)),
                               torch.cat((self._view.layers[t].v, kv.v)))
                         for t, kv in converted.items()}
        # First source positions are sinks; keep order and preserve real distances.
        keep = (positions < self.sinks) | (positions >= positions[-1] - self.recent + 1)
        trimmed = {t: KV(kv.k[keep], kv.v[keep]) for t, kv in converted.items()}
        new_view = ForeignView(delta.epoch, positions[keep], MappingProxyType(trimmed))
        # All validation/projection/allocation happened before these state updates.
        self._view = new_view
        self.next_sequence += 1
        self.next_start += delta.tokens
        self.last_epoch = delta.epoch

    def pin(self, *, now_epoch: int | None = None,
            max_age: int | None = None) -> ForeignView | None:
        if max_age is not None:
            if now_epoch is None or max_age < 0:
                raise ValueError("TTL requires a current epoch and nonnegative age")
            if self._view is not None and now_epoch - self._view.epoch > max_age:
                return None
        return self._view
````

### Source 22 — `telepathy/core/position.py`

<!-- file: telepathy/core/position.py sha256: f26e6393ff0128cfb91932f1a7c1ce80c451694da1fbfc6fbb420ccabf33e74e -->
````python
from __future__ import annotations
import torch


def apply_cos_sin(x: torch.Tensor, cos: torch.Tensor,
                  sin: torch.Tensor) -> torch.Tensor:
    """Split-half RoPE. x:[T,H,D], cos/sin:[T,D]. No value rotation.

    This function is NOT the receiver's frequency policy. Real adapters obtain
    cos/sin from the checkpoint's own rotary module, including its scaling.
    """
    if x.ndim != 3 or x.shape[-1] % 2 or cos.shape != (x.shape[0], x.shape[-1]):
        raise ValueError("bad RoPE shape or odd rotary dimension")
    if sin.shape != cos.shape:
        raise ValueError("cos/sin mismatch")
    half = x.shape[-1] // 2
    quarter_turn = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return x * cos[:, None, :] + quarter_turn * sin[:, None, :]


def basic_rope(x: torch.Tensor, positions: torch.Tensor,
               theta: float = 10000.0) -> torch.Tensor:
    """Toy/default RoPE only; do not substitute this for a Llama-3.1 adapter."""
    if positions.shape != (x.shape[0],) or theta <= 1:
        raise ValueError("invalid position vector or base")
    freq = theta ** (-torch.arange(0, x.shape[-1], 2,
                                  device=x.device, dtype=torch.float32) / x.shape[-1])
    phase = positions.to(x.device).float()[:, None] * freq[None, :]
    phase = torch.cat((phase, phase), dim=-1)
    return apply_cos_sin(x, phase.cos().to(x.dtype), phase.sin().to(x.dtype))


def recency_positions(source_positions: torch.Tensor,
                      first_query_position: int) -> torch.Tensor:
    """Proposed foreign-memory policy: newest entry is one slot before query.

    Preserve SOURCE token distances, not semantic/tokenizer alignment. Negative
    virtual positions are intentional. A runtime must validate this policy.
    Source positions stay ordered; sinks retain their original distances.
    """
    if source_positions.ndim != 1 or source_positions.numel() == 0:
        raise ValueError("positions must be a nonempty vector")
    if not bool(torch.all(source_positions[1:] > source_positions[:-1])):
        raise ValueError("source positions must be strictly increasing")
    return first_query_position - 1 - (source_positions[-1] - source_positions)
````

### Source 23 — `telepathy/core/projector.py`

<!-- file: telepathy/core/projector.py sha256: 3b0b7e497b30560131981196269dfa0b71e2145efd0c92f35a2ec6255ae4e707 -->
````python
from __future__ import annotations
import torch
from torch import nn
from .types import KV


class LinearMap(nn.Module):
    def __init__(self, source: tuple[int, int], target: tuple[int, int]):
        super().__init__()
        self.source, self.target = source, target
        self.map = nn.Linear(source[0] * source[1], target[0] * target[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if tuple(x.shape[1:]) != self.source:
            raise ValueError("projector source shape mismatch")
        return self.map(x.flatten(1)).reshape(x.shape[0], *self.target)

    @torch.no_grad()
    def fit(self, x: torch.Tensor, y: torch.Tensor, ridge: float = 1e-3) -> None:
        """Offline FP64 centered ridge; separate intercept, not regularized.

        This simple primal solver is appropriate for a bounded reference dataset.
        Stream X'X/X'Y or use a stable factorization for large-scale fitting.
        """
        if ridge <= 0 or x.shape[0] != y.shape[0] or x.shape[0] < 2:
            raise ValueError("need paired rows and positive ridge")
        if tuple(x.shape[1:]) != self.source or tuple(y.shape[1:]) != self.target:
            raise ValueError("ridge shape mismatch")
        a, b = x.detach().cpu().double().flatten(1), y.detach().cpu().double().flatten(1)
        if not (torch.isfinite(a).all() and torch.isfinite(b).all()):
            raise ValueError("nonfinite training data")
        am, bm = a.mean(0), b.mean(0)
        ac, bc = a - am, b - bm
        lhs = ac.T @ ac + ridge * torch.eye(ac.shape[1], dtype=torch.float64)
        weight = torch.linalg.solve(lhs, ac.T @ bc)
        self.map.weight.copy_(weight.T.to(self.map.weight))
        self.map.bias.copy_((bm - am @ weight).to(self.map.bias))


class HeadMap(nn.Module):
    """Learned head mixing + channel map + normalized residual MLP.

    The residual preserves a direct amplitude-carrying path; K and V do not share
    weights. A baseline map is not evidence that this architecture will transfer.
    """
    def __init__(self, source: tuple[int, int], target: tuple[int, int], rank: int = 64):
        super().__init__()
        self.source, self.target = source, target
        hs, ds = source
        ht, dt = target
        self.mix = nn.Parameter(torch.randn(ht, hs) / hs**0.5)
        self.channel = nn.Linear(ds, dt)
        self.residual = nn.Sequential(nn.LayerNorm(ds), nn.Linear(ds, rank),
                                      nn.SiLU(), nn.Linear(rank, dt))
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if tuple(x.shape[1:]) != self.source:
            raise ValueError("head-map shape mismatch")
        mixed = torch.einsum("oh,thd->tod", self.mix, x)
        return self.channel(mixed) + self.residual(mixed)


class BridgeProjector(nn.Module):
    def __init__(self, source: tuple[int, int], target: tuple[int, int],
                 kind: str = "ridge", rank: int = 64):
        super().__init__()
        if kind not in {"ridge", "mlp"}:
            raise ValueError("kind must be ridge or mlp")
        self.source, self.target, self.kind, self.rank = source, target, kind, rank
        factory = ((lambda: LinearMap(source, target)) if kind == "ridge"
                   else (lambda: HeadMap(source, target, rank)))
        self.k_map, self.v_map = factory(), factory()

    def forward(self, kv: KV) -> KV:
        kv.check()
        return KV(self.k_map(kv.k), self.v_map(kv.v))

    def fit(self, source: KV, target: KV, ridge: float = 1e-3) -> None:
        if self.kind != "ridge":
            raise ValueError("use the optimizer training stage for MLP maps")
        self.k_map.fit(source.k, target.k, ridge)
        self.v_map.fit(source.v, target.v, ridge)
````

### Source 24 — `telepathy/core/sync.py`

<!-- file: telepathy/core/sync.py sha256: 3c22595c8d9e28d954139bc264c4465585aa38a7c56c81da6fb0a2d49b80922e -->
````python
from __future__ import annotations
from collections.abc import Callable
from .memory import ForeignKVBank, ForeignView
from .types import Delta


class SyncController:
    """Single-threaded, one-epoch-lag reference scheduler.

    A(k) and B(k) both pin their incoming state BEFORE either computes/publishes.
    Any error poisons this controller; resume requires a recorded checkpoint/replay.
    A bank's update is atomic across layers. A distributed pair is NOT a global
    transaction; failure after one publication aborts the run, never continues it.
    """
    def __init__(self, incoming_a: ForeignKVBank, incoming_b: ForeignKVBank):
        if incoming_a.direction != 1 or incoming_b.direction != 0:
            raise ValueError("incorrect incoming-bank direction wiring")
        self.a, self.b, self.epoch, self.failed = incoming_a, incoming_b, 0, False

    def tick(self, compute_a: Callable[[ForeignView | None, int], Delta],
             compute_b: Callable[[ForeignView | None, int], Delta]) -> None:
        if self.failed:
            raise RuntimeError("controller is poisoned; restore a clean run")
        try:
            a_view, b_view = self.a.pin(), self.b.pin()
            for view in (a_view, b_view):
                if view is not None and view.epoch != self.epoch - 1:
                    raise ValueError("lockstep requires precisely the prior epoch")
            a_delta = compute_a(a_view, self.epoch)
            b_delta = compute_b(b_view, self.epoch)
            if a_delta.epoch != self.epoch or b_delta.epoch != self.epoch:
                raise ValueError("worker returned a different epoch")
            self.b.commit(a_delta)
            self.a.commit(b_delta)
            self.epoch += 1
        except Exception:
            self.failed = True
            raise
````

### Source 25 — `telepathy/core/types.py`

<!-- file: telepathy/core/types.py sha256: 6791490b173d0861b23fd31c09f85f0d600aae44471e1bd6a5e11dd5362d9ed5 -->
````python
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
from uuid import UUID
import torch


@dataclass(frozen=True)
class KV:
    """Canonical K is pre-RoPE, AFTER native K normalization; V is unrotated.

    Layout is [sequence, kv_heads, head_dim]. Batch size is deliberately one.
    Frozen dataclasses do not make tensors immutable: ownership boundaries clone.
    """
    k: torch.Tensor
    v: torch.Tensor

    def check(self) -> None:
        if self.k.ndim != 3 or self.k.shape != self.v.shape:
            raise ValueError("K/V must share [sequence, kv_heads, head_dim]")
        if min(self.k.shape) <= 0:
            raise ValueError("empty K/V blocks are not allowed")
        if self.k.dtype != self.v.dtype or self.k.device != self.v.device:
            raise ValueError("K/V dtype and device must match")
        if not self.k.is_floating_point() or not self.v.is_floating_point():
            raise TypeError("K/V must be floating-point activations")
        if not (torch.isfinite(self.k).all() and torch.isfinite(self.v).all()):
            raise ValueError("nonfinite activation")

    def clone(self, *, detach: bool = True) -> KV:
        f = (lambda x: x.detach().clone()) if detach else (lambda x: x.clone())
        return KV(f(self.k), f(self.v))

    @property
    def tokens(self) -> int:
        return self.k.shape[0]


@dataclass(frozen=True)
class Delta:
    """One complete, contiguous source block for ALL agreed bridged layers.

    Session UUID, direction and layer shapes bind to a separately pinned manifest.
    No text, token IDs, answer labels or free-form metadata are represented here.
    """
    session: UUID
    direction: int                 # 0: A -> B; 1: B -> A
    epoch: int
    sequence: int
    start: int                     # source slot / native source position
    layers: Mapping[int, KV]

    def check(self) -> None:
        if self.direction not in (0, 1):
            raise ValueError("bad direction")
        if min(self.epoch, self.sequence, self.start) < 0 or not self.layers:
            raise ValueError("negative counter or empty layer set")
        if any(type(i) is not int or not 0 <= i <= 65535 for i in self.layers):
            raise ValueError("bad layer index")
        for kv in self.layers.values():
            kv.check()
        if len({x.tokens for x in self.layers.values()}) != 1:
            raise ValueError("a publication must be complete across layers")

    @property
    def tokens(self) -> int:
        return next(iter(self.layers.values())).tokens

    def clone(self) -> Delta:
        return Delta(self.session, self.direction, self.epoch, self.sequence,
                     self.start, {i: x.clone() for i, x in self.layers.items()})
````

### Source 26 — `telepathy/eval/__init__.py`

<!-- file: telepathy/eval/__init__.py sha256: 90cc787c8c832fc41f564f76a60d997bc4bbda2ea7c24afd8e0a43eea0ec3ebd -->
````python
"""Telepathy reference implementation; see TELEPATHY_AGENT_ENGINEERING.md."""
````

### Source 27 — `telepathy/eval/metrics.py`

<!-- file: telepathy/eval/metrics.py sha256: ec2e80aae5a03a3990ab8b01467afe5aeb3a3218f97904dc3af01368156be34c -->
````python
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


def paired_bootstrap(active: list[float], baseline: list[float],
                     repetitions: int = 10000, seed: int = 73) -> dict:
    """Paired experimental-unit bootstrap. Pass one score PER independent secret/task.

    Multiple questions/seeds for a secret must first be grouped at secret level.
    This is a descriptive percentile interval, not a universal power guarantee.
    """
    a, b = np.asarray(active, dtype=float), np.asarray(baseline, dtype=float)
    if a.ndim != 1 or a.shape != b.shape or a.size < 2 or repetitions < 100:
        raise ValueError("paired nontrivial samples and >=100 resamples required")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("include failed trials explicitly; do not drop them as NaN")
    difference = a - b
    generator = np.random.default_rng(seed)
    values = []
    for start in range(0, repetitions, 256):
        rows = generator.integers(0, a.size, size=(min(256, repetitions - start), a.size))
        values.append(difference[rows].mean(axis=1))
    lower, upper = np.quantile(np.concatenate(values), [0.025, 0.975])
    return {"units": int(a.size), "active_mean": float(a.mean()), "baseline_mean": float(b.mean()),
            "paired_delta": float(difference.mean()), "ci95": [float(lower), float(upper)],
            "bootstrap_seed": seed, "repetitions": repetitions}


def e2_gate(summary: dict, *, valid_channel_audit: bool,
            preregistered: bool, minimum_units: int = 100) -> str:
    if not valid_channel_audit:
        return "INVALID"
    if not preregistered or summary["units"] < minimum_units:
        return "BLOCKED"
    return "PASSED" if summary["ci95"][0] > 0 else "FAILED"


@dataclass
class CostLedger:
    # All costs, including local token-conditioned mailbox writing, are declared.
    source_prefill_tokens: int = 0
    receiver_prefill_tokens: int = 0
    local_decode_tokens: int = 0
    mailbox_local_tokens: int = 0
    projector_calls: int = 0
    payload_bytes: int = 0
    wire_bytes: int = 0
    wall_seconds: float = 0.0
    joules: float | None = None       # None means UNMEASURED, never zero.

    def validate(self) -> None:
        integers = [self.source_prefill_tokens, self.receiver_prefill_tokens,
                    self.local_decode_tokens, self.mailbox_local_tokens,
                    self.projector_calls, self.payload_bytes, self.wire_bytes]
        if any(type(x) is not int or x < 0 for x in integers) or self.wall_seconds < 0:
            raise ValueError("negative/invalid accounting")
        if self.joules is not None and self.joules < 0:
            raise ValueError("invalid energy")
````

### Source 28 — `telepathy/mailbox/__init__.py`

<!-- file: telepathy/mailbox/__init__.py sha256: 90cc787c8c832fc41f564f76a60d997bc4bbda2ea7c24afd8e0a43eea0ec3ebd -->
````python
"""Telepathy reference implementation; see TELEPATHY_AGENT_ENGINEERING.md."""
````

### Source 29 — `telepathy/mailbox/protocol.py`

<!-- file: telepathy/mailbox/protocol.py sha256: c5aef9cea70c6ffaaefa29927a49bfa8fad1b736469da44d49a00c2e62b6f22d -->
````python
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import torch
from torch import nn
from telepathy.core.types import KV
from telepathy.runtime.decoder import FrozenDecoder, NativeState


class ThoughtWriter(nn.Module):
    """Token-conditioned v0: local tokens become activations on a PRIVATE branch.

    Not a direct latent writer and not tokenless cognition. The controller must
    ensure the text/IDs originated locally, not from an answer key or a peer tool.
    This function does not serialize anything; only returned canonical KV may
    enter the declared transport, after normal session/window validation.
    """
    def __init__(self, hidden_size: int, max_tokens: int = 32):
        super().__init__()
        self.marker = nn.Parameter(torch.zeros(1, hidden_size))
        self.max_tokens = max_tokens

    def write(self, decoder: FrozenDecoder, private_parent: NativeState,
              local_ids: torch.Tensor) -> dict[int, KV]:
        if local_ids.ndim != 1 or not 1 <= local_ids.numel() <= self.max_tokens:
            raise ValueError("mail token budget exceeded")
        embeddings = decoder.model.model.embed_tokens(local_ids.to(decoder.device))
        embeddings = torch.cat((self.marker.to(embeddings), embeddings))
        output = decoder.forward(None, private_parent.clone(), embeddings=embeddings)
        # No native caller cache is updated; training may backpropagate into marker.
        return dict(output.canonical_delta)


class MailState(str, Enum):
    PENDING = "pending"
    ATTENDED = "attended_not_proven"
    INCORPORATED = "causally_incorporated"
    STARVED = "expired_without_incorporation"


@dataclass
class MailRecord:
    created: int
    expires: int
    state: MailState = MailState.PENDING
    first_attention: int | None = None
    incorporation: int | None = None


class MailboxLedger:
    """Experimenter-side ledger. Models cannot read it or receive its verdicts."""
    def __init__(self, capacity: int = 32):
        if capacity <= 0:
            raise ValueError("positive capacity required")
        self.capacity, self.records = capacity, {}

    def create(self, message_id: int, epoch: int, ttl: int) -> None:
        if min(message_id, epoch) < 0 or ttl <= 0 or message_id in self.records:
            raise ValueError("invalid or duplicate mail ID")
        active = sum(r.state in {MailState.PENDING, MailState.ATTENDED} for r in self.records.values())
        if active >= self.capacity:
            raise OverflowError("mailbox storm/backpressure")
        self.records[message_id] = MailRecord(epoch, epoch + ttl)

    def observe_attention(self, message_id: int, epoch: int, mass: float,
                          threshold: float = 0.01) -> None:
        record = self.records[message_id]
        if not 0 <= mass <= 1 or not record.created <= epoch < record.expires:
            raise ValueError("bad observation time or attention mass")
        if mass >= threshold and record.state == MailState.PENDING:
            record.first_attention, record.state = epoch, MailState.ATTENDED

    def record_causal_use(self, message_id: int, epoch: int, *,
                          active_correct: bool, ablated_correct: bool,
                          replay_started_before_exposure: bool) -> None:
        record = self.records[message_id]
        if not record.created <= epoch < record.expires:
            raise ValueError("outside delivery window")
        if not replay_started_before_exposure:
            raise ValueError("late masking cannot establish a clean causal counterfactual")
        if active_correct and not ablated_correct:
            record.state, record.incorporation = MailState.INCORPORATED, epoch

    def expire(self, epoch: int) -> None:
        for record in self.records.values():
            if epoch >= record.expires and record.state in {MailState.PENDING, MailState.ATTENDED}:
                record.state = MailState.STARVED


class ProbeHead(nn.Module):
    """Diagnostic label probe, NOT a guaranteed natural-language thought decoder.

    Labels, training data and outputs belong to the external audit process.
    """
    def __init__(self, kv_heads: int, head_dim: int, labels: int):
        super().__init__()
        self.head = nn.Linear(2 * kv_heads * head_dim, labels)

    def forward(self, kv: KV) -> torch.Tensor:
        features = torch.cat((kv.k.detach().mean(0).flatten(), kv.v.detach().mean(0).flatten()))
        return self.head(features)
````

### Source 30 — `telepathy/plugin/__init__.py`

<!-- file: telepathy/plugin/__init__.py sha256: 90cc787c8c832fc41f564f76a60d997bc4bbda2ea7c24afd8e0a43eea0ec3ebd -->
````python
"""Telepathy reference implementation; see TELEPATHY_AGENT_ENGINEERING.md."""
````

### Source 31 — `telepathy/plugin/control.py`

<!-- file: telepathy/plugin/control.py sha256: 480df38d64a18f15955c3063a4d264c637f87b8c220e3b4fdb3fcfeb6c04a1d9 -->
````python
"""Typed reference control schema, NOT a claim about OMP's actual plugin API."""
from dataclasses import dataclass
from enum import IntEnum
from uuid import UUID


class Op(IntEnum):
    START = 1
    TICK = 2
    PAUSE = 3
    ABORT = 4
    COMPLETE = 5


@dataclass(frozen=True)
class ControlEvent:
    run: UUID
    op: Op
    epoch: int

    @classmethod
    def parse(cls, event: dict) -> "ControlEvent":
        if set(event) != {"run", "op", "epoch"}:
            raise ValueError("control plane rejects free text, hints, token IDs and extra fields")
        if type(event["epoch"]) is not int or event["epoch"] < 0:
            raise ValueError("invalid epoch")
        if type(event["op"]) is not int:
            raise ValueError("op must be a fixed numeric enum")
        return cls(UUID(event["run"]), Op(event["op"]), event["epoch"])
````

### Source 32 — `telepathy/runtime/__init__.py`

<!-- file: telepathy/runtime/__init__.py sha256: 90cc787c8c832fc41f564f76a60d997bc4bbda2ea7c24afd8e0a43eea0ec3ebd -->
````python
"""Telepathy reference implementation; see TELEPATHY_AGENT_ENGINEERING.md."""
````

### Source 33 — `telepathy/runtime/decoder.py`

<!-- file: telepathy/runtime/decoder.py sha256: 5cdd3dde81b9c1889a67d3171b73f4884a9ad31049b11b745f42e7e01bcfd03d -->
````python
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
import torch
from torch import nn
from telepathy.core.types import KV
from telepathy.core.position import apply_cos_sin, recency_positions
from telepathy.core.attention import Gate, attend
from telepathy.core.memory import ForeignView


@dataclass(frozen=True)
class NativeState:
    length: int
    layers: tuple[KV, ...]           # native K is ALREADY rotated here

    def clone(self, *, detach: bool = True) -> NativeState:
        return NativeState(self.length, tuple(kv.clone(detach=detach) for kv in self.layers))


@dataclass(frozen=True)
class StepOutput:
    logits: torch.Tensor             # [T, receiver_vocab]
    state: NativeState
    canonical_delta: Mapping[int, KV]
    foreign_mass: Mapping[int, torch.Tensor]


class FrozenDecoder:
    """Dedicated reference forward, not a generic provider/completion API.

    Supports the dense full-attention Qwen3/Llama layout and the supplied toy.
    No monkeypatches or native cache pollution. Batch one, unquantized, one device,
    eager attention, fixed-frequency split-half rotary only. HF parity is a gate.
    All modules/parameters are the original frozen checkpoint modules.
    """
    def __init__(self, model: nn.Module):
        cfg = model.config
        if cfg.model_type not in {"qwen3", "llama", "telepathy_toy"}:
            raise ValueError("unsupported model architecture; implement and verify an adapter")
        if getattr(model, "is_quantized", False):
            raise ValueError("quantized models require a separately validated adapter")
        if getattr(cfg, "pretraining_tp", 1) != 1:
            raise ValueError("tensor-parallel checkpoint forward not supported")
        if getattr(cfg, "use_sliding_window", False) or any(
                x != "full_attention" for x in getattr(cfg, "layer_types", [])):
            raise ValueError("sliding/hybrid layers require a separate mask/cache adapter")
        rope = getattr(cfg, "rope_scaling", None) or {}
        if rope.get("rope_type", rope.get("type", "default")) not in {"default", "llama3"}:
            raise ValueError("dynamic/other RoPE needs explicit lifetime and parity tests")
        model.eval().requires_grad_(False)
        self.model = model
        self.layers = model.model.layers
        self.qheads, self.kvheads = cfg.num_attention_heads, cfg.num_key_value_heads
        self.dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // self.qheads
        if self.dim % 2 or self.qheads % self.kvheads:
            raise ValueError("unsupported rotary/GQA shape")
        devices = {p.device for p in model.parameters()}
        if len(devices) != 1 or next(iter(devices)).type == "meta":
            raise ValueError("materialize the whole model on one device")
        self.device = next(iter(devices))
        self.dtype = next(model.parameters()).dtype

    @classmethod
    def from_local_hf(cls, path: str | Path, device: str = "cpu",
                      dtype: torch.dtype = torch.float32) -> FrozenDecoder:
        import transformers
        if transformers.__version__ != "4.56.2":
            raise RuntimeError("reference HF adapter requires transformers==4.56.2; requalify upgrades")
        if not Path(path).is_dir():
            raise FileNotFoundError("provide an already authorized local checkpoint directory")
        model = transformers.AutoModelForCausalLM.from_pretrained(
            str(path), local_files_only=True, trust_remote_code=False,
            torch_dtype=dtype, attn_implementation="eager")
        return cls(model.to(device))

    def rotate(self, k: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        cos, sin = self.model.model.rotary_emb(k.unsqueeze(0), positions[None])
        return apply_cos_sin(k, cos[0], sin[0])

    def empty(self) -> NativeState:
        return NativeState(0, ())

    def import_self_prefix(self, canonical: Mapping[int, KV],
                           positions: torch.Tensor) -> NativeState:
        """M0 identity replacement only. NOT ordinary cross-family memory append."""
        if set(canonical) != set(range(len(self.layers))):
            raise ValueError("self-handoff requires every layer, not an eight-layer subset")
        n = positions.numel()
        if not torch.equal(positions.cpu(), torch.arange(n)) or n == 0:
            raise ValueError("self-handoff requires a complete prefix starting at position zero")
        result = []
        for i in range(len(self.layers)):
            kv = canonical[i]
            kv.check()
            if kv.k.shape != (n, self.kvheads, self.dim):
                raise ValueError("self-prefix shape mismatch")
            k, v = kv.k.to(device=self.device, dtype=self.dtype), kv.v.to(device=self.device, dtype=self.dtype)
            result.append(KV(self.rotate(k, positions.to(self.device)), v.clone()))
        return NativeState(n, tuple(result))

    def forward(self, ids: torch.Tensor | None, state: NativeState | None = None,
                foreign: ForeignView | None = None,
                gates: Mapping[int, Gate] | None = None,
                override: float | None = None,
                embeddings: torch.Tensor | None = None) -> StepOutput:
        # Do NOT wrap this method in no_grad: frozen backbones must pass gradients
        # from the task loss into injected activations and trainable sidecars.
        if (ids is None) == (embeddings is None):
            raise ValueError("provide exactly one of local token IDs or embeddings")
        state = self.empty() if state is None else state
        if ids is not None:
            if ids.ndim != 1 or ids.numel() == 0 or ids.dtype != torch.long:
                raise ValueError("local token IDs must be nonempty int64 [T]")
            hidden = self.model.model.embed_tokens(ids.to(self.device))
        else:
            if embeddings.ndim != 2 or embeddings.shape[0] == 0:
                raise ValueError("local embeddings must be [T,hidden]")
            hidden = embeddings.to(device=self.device, dtype=self.dtype)
        n = hidden.shape[0]
        if state.length < 0 or (state.length == 0 and state.layers) or (
                state.length > 0 and len(state.layers) != len(self.layers)):
            raise ValueError("invalid native state")
        if state.length + n > self.model.config.max_position_embeddings:
            raise ValueError("reference context budget exceeded")
        positions = torch.arange(state.length, state.length + n, device=self.device)
        all_positions = torch.arange(state.length + n, device=self.device)
        causal = all_positions[None] <= positions[:, None]
        native, captured, masses = [], {}, {}
        for index, layer in enumerate(self.layers):
            attention = layer.self_attn
            normed = layer.input_layernorm(hidden)
            query = attention.q_proj(normed).reshape(n, self.qheads, self.dim)
            key = attention.k_proj(normed).reshape(n, self.kvheads, self.dim)
            value = attention.v_proj(normed).reshape(n, self.kvheads, self.dim)
            if hasattr(attention, "q_norm"):
                query = attention.q_norm(query)
            if hasattr(attention, "k_norm"):
                key = attention.k_norm(key)
            captured[index] = KV(key, value)
            qr, kr = self.rotate(query, positions), self.rotate(key, positions)
            if state.length:
                past = state.layers[index]
                if past.k.shape != (state.length, self.kvheads, self.dim):
                    raise ValueError("native cache length/shape mismatch")
                kr = torch.cat((past.k, kr))
                value = torch.cat((past.v, value))
            local = KV(kr, value)
            native.append(local)
            imported = None
            if foreign is not None and index in foreign.layers and override != 0.0:
                memory = foreign.layers[index]
                fk = memory.k.to(device=self.device, dtype=self.dtype)
                fv = memory.v.to(device=self.device, dtype=self.dtype)
                virtual = recency_positions(foreign.positions.to(self.device), state.length)
                imported = KV(self.rotate(fk, virtual), fv)
            selected_gate = None if gates is None else gates.get(index)
            output, mass = attend(qr, local, causal, imported, selected_gate, override,
                                  scale=getattr(attention, "scaling", self.dim**-0.5))
            masses[index] = mass
            hidden = hidden + attention.o_proj(output.reshape(n, self.qheads * self.dim))
            hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
        logits = self.model.lm_head(self.model.model.norm(hidden))
        return StepOutput(logits, NativeState(state.length + n, tuple(native)), captured, masses)

    @torch.no_grad()
    def generate(self, prompt_ids: torch.Tensor, max_new_tokens: int,
                 foreign: ForeignView | None = None, override: float = 1.0,
                 eos_id: int | None = None) -> list[int]:
        """Greedy diagnostic generation, local IDs only. Count all local tokens."""
        if max_new_tokens <= 0:
            raise ValueError("positive generation budget required")
        output = self.forward(prompt_ids, foreign=foreign, override=override)
        generated = []
        for step in range(max_new_tokens):
            token = int(output.logits[-1].argmax())
            generated.append(token)
            if token == eos_id or step == max_new_tokens - 1:
                break
            output = self.forward(torch.tensor([token], dtype=torch.long, device=self.device),
                                  output.state, foreign=foreign, override=override)
        return generated
````

### Source 34 — `telepathy/runtime/manifest.py`

<!-- file: telepathy/runtime/manifest.py sha256: e9fbbb44cca5ca576e3bfbc00224cd6cf7276f702f1d2d38285dee16292559b9 -->
````python
from __future__ import annotations
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parameter_digest(model: torch.nn.Module) -> str:
    """Correctness helper. For 8B checkpoints prefer streaming on-disk shard hashes."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        data = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((tuple(data.shape), data.dtype)).encode())
        digest.update(data.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def inventory(model_directory: Path | None = None) -> dict:
    packages = {}
    for package in ("torch", "numpy", "transformers", "mlx", "mlx-lm", "safetensors", "pytest"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    result = {"python": platform.python_version(), "platform": platform.platform(),
              "packages": packages, "cuda_available": torch.cuda.is_available(),
              "mps_available": torch.backends.mps.is_available(),
              "model": None, "mcdma": "NOT_INSPECTED", "omp": "NOT_INSPECTED"}
    if model_directory is not None:
        config_path = model_directory / "config.json"
        config = json.loads(config_path.read_text())
        dim = config.get("head_dim") or config["hidden_size"] // config["num_attention_heads"]
        result["model"] = {"config_sha256": sha256_file(config_path),
                           "model_type": config["model_type"],
                           "layers": config["num_hidden_layers"],
                           "kv_heads": config["num_key_value_heads"], "head_dim": dim,
                           "rope_theta": config.get("rope_theta"),
                           "rope_scaling": config.get("rope_scaling"),
                           "tokenizer_files": {p.name: sha256_file(p) for p in model_directory.glob("tokenizer*") if p.is_file()},
                           "weight_shards": {p.name: sha256_file(p) for p in model_directory.glob("*.safetensors")},
                           "bytes_per_token_fp16_all_layers": 2 * 2 * config["num_hidden_layers"] * config["num_key_value_heads"] * dim}
    return result
````

### Source 35 — `telepathy/runtime/toy.py`

<!-- file: telepathy/runtime/toy.py sha256: 17d23f68a5917aab8a5980985283c516f329b2a8c2dcfe1e489506863999a435 -->
````python
"""Random small decoder for plumbing tests. It is NOT a trained language model."""
from types import SimpleNamespace
import torch
from torch import nn


class ToyRotary(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.dimension = dimension

    def forward(self, x, position_ids):
        freq = 10000.0 ** (-torch.arange(0, self.dimension, 2,
                                       device=x.device).float() / self.dimension)
        phase = position_ids.float()[..., None] * freq
        phase = torch.cat((phase, phase), dim=-1)
        return phase.cos().to(x.dtype), phase.sin().to(x.dtype)


class ToyAttention(nn.Module):
    def __init__(self, width, qheads, kvheads, dim):
        super().__init__()
        self.q_proj = nn.Linear(width, qheads * dim, bias=False)
        self.k_proj = nn.Linear(width, kvheads * dim, bias=False)
        self.v_proj = nn.Linear(width, kvheads * dim, bias=False)
        self.o_proj = nn.Linear(qheads * dim, width, bias=False)
        self.q_norm = nn.LayerNorm(dim)
        self.k_norm = nn.LayerNorm(dim)
        self.scaling = dim**-0.5


class ToyLayer(nn.Module):
    def __init__(self, width, qheads, kvheads, dim):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(width)
        self.self_attn = ToyAttention(width, qheads, kvheads, dim)
        self.post_attention_layernorm = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, width * 2), nn.SiLU(),
                                 nn.Linear(width * 2, width))


class ToyModel(nn.Module):
    def __init__(self, seed: int = 7, width: int = 24, qheads: int = 4,
                 kvheads: int = 2, dim: int = 6, layers: int = 3, vocab: int = 41):
        super().__init__()
        if dim % 2 or qheads % kvheads:
            raise ValueError("invalid toy dimensions")
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.model = nn.Module()
            self.model.embed_tokens = nn.Embedding(vocab, width)
            self.model.layers = nn.ModuleList(
                ToyLayer(width, qheads, kvheads, dim) for _ in range(layers))
            self.model.norm = nn.LayerNorm(width)
            self.model.rotary_emb = ToyRotary(dim)
            self.lm_head = nn.Linear(width, vocab, bias=False)
        self.config = SimpleNamespace(model_type="telepathy_toy", hidden_size=width,
                                      num_attention_heads=qheads,
                                      num_key_value_heads=kvheads, head_dim=dim,
                                      num_hidden_layers=layers, vocab_size=vocab,
                                      max_position_embeddings=4096,
                                      rope_scaling=None, pretraining_tp=1)
````

### Source 36 — `telepathy/train/__init__.py`

<!-- file: telepathy/train/__init__.py sha256: 90cc787c8c832fc41f564f76a60d997bc4bbda2ea7c24afd8e0a43eea0ec3ebd -->
````python
"""Telepathy reference implementation; see TELEPATHY_AGENT_ENGINEERING.md."""
````

### Source 37 — `telepathy/train/alignment.py`

<!-- file: telepathy/train/alignment.py sha256: d129fbef0a3a0ff0a0bfe840a701e79e28144a53bca7336291ded2d2318aa5e2 -->
````python
from __future__ import annotations


def shared_causal_endpoints(text: str, source_offsets: list[tuple[int, int]],
                            target_offsets: list[tuple[int, int]]) -> list[tuple[int, int, int]]:
    """Align rows ending at the SAME original-text byte boundary.

    Input offsets must already be validated as original-string CHARACTER offsets
    (e.g. a checked fast tokenizer). Zero-length special tokens are excluded.
    If several tokens end at one boundary, select the final token there. No
    nearest-neighbor or future-looking overlap averaging is done. This only
    supplies offline supervision; inference receives no shared text/offset map.
    """
    byte_ends = [0]
    for char in text:
        byte_ends.append(byte_ends[-1] + len(char.encode("utf-8")))

    def index(offsets):
        result, previous_end = {}, 0
        for token, (start, end) in enumerate(offsets):
            if not 0 <= start <= end <= len(text):
                raise ValueError("offset is not in the original character coordinate system")
            if start == end:
                continue
            if end < previous_end:
                raise ValueError("nonmonotonic tokenizer offsets")
            previous_end = end
            result[byte_ends[end]] = token
        return result

    source, target = index(source_offsets), index(target_offsets)
    return [(source[end], target[end], end) for end in sorted(source.keys() & target.keys())]
````

### Source 38 — `telepathy/train/fit.py`

<!-- file: telepathy/train/fit.py sha256: 55637c6983dd77e1639c56deb07610e136ffe65a4f5b2ca0823543e21ff111ad -->
````python
from __future__ import annotations
import hashlib
import json
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from safetensors.torch import save_file, load_file
from telepathy.core.projector import BridgeProjector
from telepathy.core.types import KV


def receiver_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                temperature: float = 1.0) -> torch.Tensor:
    """Both logits MUST be from the RECEIVER vocabulary at aligned task positions."""
    if student_logits.shape != teacher_logits.shape or temperature <= 0:
        raise ValueError("receiver logits must align; never KL across different vocabularies")
    if not student_logits.requires_grad:
        raise ValueError("student graph is detached; frozen backbones still need autograd")
    logp = F.log_softmax(student_logits.float() / temperature, dim=-1)
    q = F.softmax(teacher_logits.detach().float() / temperature, dim=-1)
    return F.kl_div(logp, q, reduction="batchmean") * temperature**2


def fit_mlp(projector: BridgeProjector, source: KV, target: KV,
            steps: int = 200, batch_size: int = 64, learning_rate: float = 1e-3,
            seed: int = 11) -> dict:
    """KV regression stage only; task/behavior training is a separate build gate."""
    source.check()
    target.check()
    if projector.kind != "mlp" or source.tokens != target.tokens:
        raise ValueError("MLP and aligned training rows required")
    if min(steps, batch_size) <= 0 or learning_rate <= 0:
        raise ValueError("invalid budget")
    device = next(projector.parameters()).device
    if source.k.device != device or target.k.device != device:
        raise ValueError("put training tensors on the projector device explicitly")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    optimizer = torch.optim.AdamW(projector.parameters(), lr=learning_rate, weight_decay=1e-4)
    started, losses = time.perf_counter(), []
    projector.train()
    for _ in range(steps):
        rows = torch.randint(source.tokens, (batch_size,), generator=generator).to(device)
        predicted = projector(KV(source.k[rows].detach(), source.v[rows].detach()))
        loss = (F.mse_loss(predicted.k.float(), target.k[rows].detach().float())
                + F.mse_loss(predicted.v.float(), target.v[rows].detach().float()))
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(projector.parameters(), max_norm=1.0, error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach()))
    projector.eval()
    return {"kind": "kv_regression_only", "steps": steps, "row_presentations": steps * batch_size,
            "unique_input_rows": source.tokens, "seed": seed,
            "wall_seconds": time.perf_counter() - started,
            "first_loss": losses[0], "last_loss": losses[-1],
            "trainable_parameters": sum(p.numel() for p in projector.parameters())}


def save_projector(projector: BridgeProjector, directory: Path, provenance: dict) -> None:
    directory.mkdir(parents=True, exist_ok=False)
    tensors = {name: value.detach().cpu().contiguous() for name, value in projector.state_dict().items()}
    weights = directory / "projector.safetensors"
    save_file(tensors, str(weights))
    metadata = {"format": 1, "source": list(projector.source), "target": list(projector.target),
                "kind": projector.kind, "rank": projector.rank,
                "sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
                "provenance": provenance}
    (directory / "projector.json").write_text(json.dumps(metadata, indent=2) + "\n")


def load_projector(directory: Path, device: str = "cpu") -> BridgeProjector:
    metadata = json.loads((directory / "projector.json").read_text())
    if set(metadata) != {"format", "source", "target", "kind", "rank", "sha256", "provenance"}:
        raise ValueError("unknown projector manifest fields")
    if metadata["format"] != 1:
        raise ValueError("unknown projector artifact version")
    weights = directory / "projector.safetensors"
    if hashlib.sha256(weights.read_bytes()).hexdigest() != metadata["sha256"]:
        raise ValueError("projector checkpoint hash mismatch")
    for shape in (metadata["source"], metadata["target"]):
        if (len(shape) != 2 or any(type(n) is not int for n in shape)
                or not 1 <= shape[0] <= 128 or not 1 <= shape[1] <= 1024
                or shape[0] * shape[1] > 4096):
            raise ValueError("unsafe projector shape")
    if type(metadata["rank"]) is not int or not 1 <= metadata["rank"] <= 1024:
        raise ValueError("unsafe rank")
    hs, ds = metadata["source"]; ht, dt = metadata["target"]; r = metadata["rank"]
    estimated = (2 * (hs * ds * ht * dt + ht * dt) if metadata["kind"] == "ridge"
                 else 2 * (ht * hs + ds * dt + dt + 2 * ds + ds * r + r + r * dt + dt))
    if estimated > 20_000_000:
        raise ValueError("projector exceeds the reference allocation budget")
    projector = BridgeProjector(tuple(metadata["source"]), tuple(metadata["target"]),
                                metadata["kind"], metadata["rank"])
    projector.load_state_dict(load_file(str(weights), device="cpu"), strict=True)
    return projector.to(device).eval()
````

### Source 39 — `telepathy/transport/__init__.py`

<!-- file: telepathy/transport/__init__.py sha256: 90cc787c8c832fc41f564f76a60d997bc4bbda2ea7c24afd8e0a43eea0ec3ebd -->
````python
"""Telepathy reference implementation; see TELEPATHY_AGENT_ENGINEERING.md."""
````

### Source 40 — `telepathy/transport/backends.py`

<!-- file: telepathy/transport/backends.py sha256: 8de8243afd8b9b2b705ec57305a851e1648d3988adf7fc8dc84fdea15d1580d0 -->
````python
from __future__ import annotations
import queue
import socket
import struct
from typing import Protocol
from telepathy.core.types import Delta
from .wire import FrameCodec


class Transport(Protocol):
    def send(self, delta: Delta) -> None: ...
    def receive(self) -> Delta: ...
    def close(self) -> None: ...


class InProcTransport:
    """Bounded COPY baseline using exactly the network wire codec."""
    def __init__(self, codec: FrameCodec, capacity: int = 2, timeout: float = 2.0):
        if capacity <= 0 or timeout <= 0:
            raise ValueError("positive capacity and timeout required")
        self.codec, self.timeout = codec, timeout
        self.queue: queue.Queue[bytes] = queue.Queue(capacity)
        self.closed = False

    def send(self, delta: Delta) -> None:
        if self.closed:
            raise RuntimeError("transport closed")
        self.queue.put(self.codec.encode(delta), timeout=self.timeout)

    def receive(self) -> Delta:
        if self.closed:
            raise RuntimeError("transport closed")
        return self.codec.decode(self.queue.get(timeout=self.timeout))

    def close(self) -> None:
        self.closed = True
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break


class SocketTransport:
    """Length-framed TCP/socket reference; closes on partial-frame failures.

    Only use raw sockets on loopback or within an approved protected tunnel.
    For nonlocal production traffic wrap a connected socket in mutual TLS, then
    pass it here. HMAC alone is not confidentiality.
    """
    def __init__(self, sock: socket.socket, codec: FrameCodec, timeout: float = 2.0):
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.sock, self.codec = sock, codec
        self.sock.settimeout(timeout)
        self.closed = False

    def _read(self, length: int) -> bytes:
        chunks, remaining = [], length
        while remaining:
            chunk = self.sock.recv(min(remaining, 65536))
            if not chunk:
                raise EOFError("peer disconnected before complete frame")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def send(self, delta: Delta) -> None:
        if self.closed:
            raise RuntimeError("transport closed")
        frame = self.codec.encode(delta)
        try:
            self.sock.sendall(struct.pack("!I", len(frame)) + frame)
        except (OSError, TimeoutError):
            self.close()
            raise

    def receive(self) -> Delta:
        if self.closed:
            raise RuntimeError("transport closed")
        try:
            size, = struct.unpack("!I", self._read(4))
            if not 0 < size <= self.codec.max_frame:
                raise ValueError("announced frame length exceeds limit")
            return self.codec.decode(self._read(size))
        except (OSError, EOFError, ValueError):
            self.close()
            raise

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self.sock.close()


class MCDMATransport:
    """Deliberately unavailable, not a silent TCP fallback or invented vendor API.

    M4 replaces this guard after inspecting the user's pinned MCDMA source,
    native buffer registration API, completion semantics and GPU visibility.
    """
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "M4 BLOCKED: native MCDMA integration is not supplied or hardware-validated. "
            "Implement the registration/completion contract in the engineering file. "
            "Select inproc or an explicitly declared TCP run instead.")
````

### Source 41 — `telepathy/transport/native_contract.h`

<!-- file: telepathy/transport/native_contract.h sha256: 4656481eda31f3cbe7e7de3d541e63fb347314ed89c9e1fffe941e791e943094 -->
````c
#ifndef TELEPATHY_NATIVE_CONTRACT_H
#define TELEPATHY_NATIVE_CONTRACT_H
/* NEW bridge-owned interface proposal, NOT MCDMA's actual ABI.
 * Declarations only: M4 implements and tests a shim against pinned native APIs.
 * Region handles are process-local and never serialized as dereferenceable pointers.
 */
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef uint64_t tp_region;
typedef uint64_t tp_transfer;
typedef enum {
    TP_HOST_REGISTERED = 1,
    TP_METAL_SHARED = 2,
    TP_CUDA_MAPPED_HOST = 3
} tp_memory_kind;
typedef enum {
    TP_OK = 0, TP_PENDING = 1, TP_UNSUPPORTED = 2,
    TP_BAD_ARGUMENT = 3, TP_FAILED = 4, TP_IN_USE = 5
} tp_status;
typedef struct {
    uint32_t contract_version;
    tp_memory_kind kind;
    size_t bytes;
    size_t required_alignment;
} tp_allocation_request;
typedef struct {
    tp_region region;
    void *local_cpu_mapping;
    size_t bytes;
} tp_allocation;
typedef struct {
    tp_transfer transfer;
    uint64_t completed_bytes;
    uint32_t producer_visible;
    uint32_t receiver_visible;
    int32_t native_error;
} tp_completion;

/* Allocate memory that the actual GPU runtime and RDMA stack can both use. */
tp_status tp_allocate(const tp_allocation_request *, tp_allocation *);
/* Runtime-specific producer-completion fence. Must not report ready early. */
tp_status tp_producer_ready(tp_region, uint64_t epoch);
/* Peer region identifiers are validated capabilities from the run handshake. */
tp_status tp_submit(tp_region source, size_t source_offset,
                    uint64_t peer_region_capability, size_t peer_offset,
                    size_t bytes, tp_transfer *);
tp_status tp_poll(tp_transfer, tp_completion *);
/* Establish receiver GPU visibility before its consuming kernel may run. */
tp_status tp_receiver_acquire(tp_region, tp_transfer);
/* Refuse release while DMA work or GPU readers still own the region. */
tp_status tp_release(tp_region);
#ifdef __cplusplus
}
#endif
#endif
````

### Source 42 — `telepathy/transport/wire.py`

<!-- file: telepathy/transport/wire.py sha256: 3122d442a54f4b36c12f9245bc0c180cf367fe2cb4d5a3b1393ebe74da575dcc -->
````python
from __future__ import annotations
import hashlib
import hmac
import struct
from uuid import UUID
import numpy as np
import torch
from telepathy.core.types import Delta, KV

# Big-endian structural integers; little-endian float32 tensor payload.
HEADER = struct.Struct("!8sBB16sQQQIH")
LAYER = struct.Struct("!HHH")
MAGIC = b"TELEKV01"
MAX_FRAME = 64 * 1024 * 1024
MAX_TOKENS, MAX_LAYERS, MAX_HEADS, MAX_DIM = 8192, 128, 128, 1024


class FrameCodec:
    """CPU/copied reference format. No pickle, JSON payload, strings, or token IDs.

    HMAC authenticates bytes; it does NOT encrypt or prove absence of covert
    channels. Session policy and sequence/layer validation are additional checks.
    Production GPU formats need new versioned dtype/packing contracts.
    """
    def __init__(self, key: bytes, max_frame: int = MAX_FRAME):
        if len(key) < 32 or not 1024 <= max_frame <= MAX_FRAME:
            raise ValueError("use >=32-byte secret and bounded frame capacity")
        self.key, self.max_frame = key, max_frame

    def encode(self, delta: Delta) -> bytes:
        delta.check()
        if not 1 <= delta.tokens <= MAX_TOKENS or len(delta.layers) > MAX_LAYERS:
            raise ValueError("frame dimension limit exceeded")
        size = HEADER.size + 32 + sum(LAYER.size + kv.k.numel() * 8 for kv in delta.layers.values())
        if size > self.max_frame:
            raise ValueError("frame too large; chunk the publication")
        body = bytearray(HEADER.pack(MAGIC, 1, delta.direction, delta.session.bytes,
                                     delta.epoch, delta.sequence, delta.start,
                                     delta.tokens, len(delta.layers)))
        for index, kv in sorted(delta.layers.items()):
            _, heads, dim = kv.k.shape
            if heads > MAX_HEADS or dim > MAX_DIM:
                raise ValueError("head shape exceeds wire limit")
            body += LAYER.pack(index, heads, dim)
            for tensor in (kv.k, kv.v):
                array = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
                if not np.isfinite(array).all():
                    raise ValueError("activation cannot be represented in wire float32")
                body += array.astype("<f4", copy=False).tobytes(order="C")
        return bytes(body) + hmac.digest(self.key, body, "sha256")

    def decode(self, frame: bytes) -> Delta:
        if not HEADER.size + 32 <= len(frame) <= self.max_frame:
            raise ValueError("invalid frame size")
        body, signature = frame[:-32], frame[-32:]
        if not hmac.compare_digest(signature, hmac.digest(self.key, body, "sha256")):
            raise ValueError("frame authentication failed")
        magic, version, direction, session, epoch, seq, start, tokens, count = HEADER.unpack_from(body)
        if magic != MAGIC or version != 1 or direction not in (0, 1):
            raise ValueError("unknown protocol/version/direction")
        if not 1 <= tokens <= MAX_TOKENS or not 1 <= count <= MAX_LAYERS:
            raise ValueError("unsafe frame dimensions")
        offset, layers = HEADER.size, {}
        for _ in range(count):
            if offset + LAYER.size > len(body):
                raise ValueError("truncated descriptor")
            index, heads, dim = LAYER.unpack_from(body, offset)
            offset += LAYER.size
            if index in layers or not 1 <= heads <= MAX_HEADS or not 1 <= dim <= MAX_DIM:
                raise ValueError("duplicate layer or unsafe shape")
            nbytes = tokens * heads * dim * 4
            if offset + 2 * nbytes > len(body):
                raise ValueError("truncated tensor")
            arrays = []
            for _ in range(2):
                # copy: the decoded tensor must not alias a reusable socket buffer.
                array = np.frombuffer(body, dtype="<f4", count=tokens * heads * dim,
                                      offset=offset).copy().astype(np.float32, copy=False)
                arrays.append(torch.from_numpy(array.reshape(tokens, heads, dim)))
                offset += nbytes
            layers[index] = KV(*arrays)
        if offset != len(body):
            raise ValueError("trailing untyped data is forbidden")
        delta = Delta(UUID(bytes=session), direction, epoch, seq, start, layers)
        delta.check()
        return delta
````

### Source 43 — `tests/conftest.py`

<!-- file: tests/conftest.py sha256: def820fe5d7b9441ad9d73b6f17cf5ff75158c7cbf70075d20537f701c1ef7ce -->
````python
import torch
import pytest


def pytest_sessionstart(session):
    torch.set_num_threads(1)


@pytest.fixture(autouse=True)
def fixed_seed():
    torch.manual_seed(123)
````

### Source 44 — `tests/test_core.py`

<!-- file: tests/test_core.py sha256: 6527d9620440c60745093ca34117cc2581efe086062f8979ff0967016b62b41d -->
````python
from dataclasses import replace
from uuid import UUID
import pytest
import torch
from telepathy.core.types import Delta, KV
from telepathy.core.position import basic_rope, recency_positions
from telepathy.core.attention import Gate, attend
from telepathy.core.projector import BridgeProjector
from telepathy.core.memory import ForeignKVBank

SESSION = UUID(int=41)


def identity(heads=2, dim=4):
    projector = BridgeProjector((heads, dim), (heads, dim))
    with torch.no_grad():
        for sub in (projector.k_map, projector.v_map):
            sub.map.weight.copy_(torch.eye(heads * dim)); sub.map.bias.zero_()
    return projector


def sample(tokens=3, heads=2, dim=4):
    return KV(torch.randn(tokens, heads, dim), torch.randn(tokens, heads, dim))


def test_canonical_type_and_nan_rejection():
    sample().check()
    kv = sample(); kv.k[0, 0, 0] = float("nan")
    with pytest.raises(ValueError): kv.check()
    with pytest.raises(ValueError): KV(torch.ones(2, 3), torch.ones(2, 3)).check()


def test_rotary_inverse_and_relative_positions():
    kv = sample()
    positions = torch.tensor([2, 5, 9])
    recovered = basic_rope(basic_rope(kv.k, positions), -positions)
    torch.testing.assert_close(recovered, kv.k)
    assert recency_positions(positions, 1).tolist() == [-7, -4, 0]
    with pytest.raises(ValueError): recency_positions(torch.tensor([0, 0]), 1)


def test_hard_closed_gate_is_exact_native_path():
    q, native, foreign = torch.randn(2, 4, 4), sample(5), sample(3)
    mask = torch.ones(2, 5, dtype=torch.bool)
    baseline, _ = attend(q, native, mask)
    foreign.k.fill_(1000)
    closed, mass = attend(q, native, mask, foreign, override=0.0)
    assert torch.equal(baseline, closed) and torch.count_nonzero(mass) == 0


def test_foreign_prior_is_inside_softmax():
    q = torch.zeros(1, 2, 4)
    native = KV(torch.zeros(1, 1, 4), torch.ones(1, 1, 4))
    foreign = KV(torch.zeros(1, 1, 4), torch.full((1, 1, 4), 3.0))
    output, mass = attend(q, native, torch.ones(1, 1, dtype=torch.bool), foreign, override=0.5)
    torch.testing.assert_close(mass, torch.full((1, 2), 1/3))
    torch.testing.assert_close(output, torch.full((1, 2, 4), 5/3))


def test_gate_and_projector_gradients_exist():
    q, native, foreign, gate = torch.randn(1, 4, 4), sample(3), sample(2), Gate()
    foreign.k.requires_grad_(True); foreign.v.requires_grad_(True)
    output, _ = attend(q, native, torch.ones(1, 3, dtype=torch.bool), foreign, gate=gate)
    output.square().sum().backward()
    assert gate.logit.grad is not None and gate.logit.grad.abs() > 0
    assert foreign.k.grad is not None and foreign.v.grad is not None


def test_masked_foreign_is_native_only():
    q, native, foreign = torch.randn(1, 4, 4), sample(3), sample(2)
    mask = torch.ones(1, 3, dtype=torch.bool)
    expected, _ = attend(q, native, mask)
    actual, _ = attend(q, native, mask, foreign, override=1,
                       allowed_foreign=torch.zeros(1, 2, dtype=torch.bool))
    assert torch.equal(expected, actual)


def test_causal_future_key_has_no_effect():
    q, native = torch.randn(2, 4, 4), sample(2)
    mask = torch.tensor([[True, False], [True, True]])
    original, _ = attend(q, native, mask)
    changed = native.clone(); changed.k[1].fill_(999); changed.v[1].fill_(999)
    after, _ = attend(q, changed, mask)
    torch.testing.assert_close(original[0], after[0])


@pytest.mark.parametrize("override", [-1.0, 1.1])
def test_bad_gate_rejected(override):
    with pytest.raises(ValueError):
        attend(torch.randn(1, 4, 4), sample(), torch.ones(1, 3, dtype=torch.bool), override=override)


def test_ridge_recovers_heldout_affine_maps():
    source = sample(200, 2, 4)
    oracle_k, oracle_v = torch.randn(8, 6), torch.randn(8, 6)
    target = KV((source.k.flatten(1) @ oracle_k + 0.4).reshape(200, 1, 6),
                (source.v.flatten(1) @ oracle_v - 0.3).reshape(200, 1, 6))
    projector = BridgeProjector((2, 4), (1, 6))
    projector.fit(source, target, ridge=1e-6)
    heldout = sample(17, 2, 4); predicted = projector(heldout)
    torch.testing.assert_close(predicted.k.flatten(1), heldout.k.flatten(1) @ oracle_k + 0.4)
    torch.testing.assert_close(predicted.v.flatten(1), heldout.v.flatten(1) @ oracle_v - 0.3)


def test_bank_projects_once_preserves_sinks_and_pinned_old_view():
    projector = identity()
    bank = ForeignKVBank(SESSION, 0, [(0, 1, projector)], sinks=2, recent=3)
    first = Delta(SESSION, 0, 0, 0, 0, {0: sample(5)})
    bank.commit(first)
    old = bank.pin(); old_values = old.layers[1].k.clone()
    first.layers[0].k.fill_(777)  # sender mutation cannot change received view.
    assert torch.equal(old.layers[1].k, old_values)
    bank.commit(Delta(SESSION, 0, 1, 1, 5, {0: sample(4)}))
    assert bank.pin().positions.tolist() == [0, 1, 6, 7, 8]
    assert old.positions.tolist() == [0, 1, 2, 3, 4]
    assert torch.equal(old.layers[1].k, old_values)
    assert bank.pin(now_epoch=20, max_age=3) is None


@pytest.mark.parametrize("mutation", ["session", "direction", "sequence", "start", "epoch", "layers"])
def test_bad_publication_does_not_change_bank(mutation):
    bank = ForeignKVBank(SESSION, 0, [(0, 1, identity())])
    bank.commit(Delta(SESSION, 0, 0, 0, 0, {0: sample(3)}))
    good = Delta(SESSION, 0, 1, 1, 3, {0: sample(2)})
    values = {"session": UUID(int=99), "direction": 1, "sequence": 0,
              "start": 4, "epoch": 0, "layers": {1: sample(2)}}
    old = bank.pin()
    with pytest.raises(ValueError): bank.commit(replace(good, **{mutation: values[mutation]}))
    assert bank.pin() is old and bank.next_sequence == 1


def test_atomic_layer_set_validation():
    bank = ForeignKVBank(SESSION, 0, [(0, 0, identity()), (2, 2, identity())])
    with pytest.raises(ValueError): bank.commit(Delta(SESSION, 0, 0, 0, 0, {0: sample()}))
    assert bank.pin() is None
````

### Source 45 — `tests/test_hf_optional.py`

<!-- file: tests/test_hf_optional.py sha256: 4c4515508565ac5ec92fb085108e43a009dc17c108950ef6f9ccd8e85ee6b091 -->
````python
"""Actual HF parity gates: automatically skipped when the optional package is absent.

Random tiny configurations exercise architecture code, not pretrained semantics.
These gates must ALSO pass against each pinned real local checkpoint in M0.
"""
import pytest
import torch
from telepathy.runtime.decoder import FrozenDecoder

hf = pytest.importorskip("transformers", reason="optional HF adapter dependency not installed")


@pytest.mark.parametrize("family", ["qwen3", "llama", "llama3_rope"])
def test_stock_hf_forward_parity(family):
    if hf.__version__ != "4.56.2":
        pytest.fail("requalify this adapter before using a different Transformers version")
    common = dict(vocab_size=97, hidden_size=32, intermediate_size=64,
                  num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                  max_position_embeddings=256, attention_dropout=0.0)
    if family == "qwen3":
        config = hf.Qwen3Config(head_dim=8, **common)
        model = hf.Qwen3ForCausalLM(config)
    else:
        if family == "llama3_rope":
            common["rope_scaling"] = {"rope_type": "llama3", "factor": 8.0,
                "low_freq_factor": 1.0, "high_freq_factor": 4.0,
                "original_max_position_embeddings": 128}
        config = hf.LlamaConfig(**common)
        model = hf.LlamaForCausalLM(config)
    model.config._attn_implementation = "eager"
    model.eval()
    ids = torch.tensor([1, 3, 5, 7, 9, 2])
    with torch.no_grad():
        stock = model(input_ids=ids[None], use_cache=False).logits[0]
        adapter = FrozenDecoder(model)
        ours = adapter.forward(ids).logits
        prefix = adapter.forward(ids[:4])
        transplanted = adapter.import_self_prefix(prefix.canonical_delta, torch.arange(4))
        continuation = adapter.forward(ids[4:], transplanted)
    torch.testing.assert_close(ours, stock, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(continuation.logits, stock[4:], atol=2e-5, rtol=2e-5)
````

### Source 46 — `tests/test_runtime_train_mail.py`

<!-- file: tests/test_runtime_train_mail.py sha256: 50b84332dd83beea5bac7411ee691e5a6253847a80cb39e659a783035bacbc8c -->
````python
from types import MappingProxyType
import pytest
import torch
from telepathy.core.types import KV
from telepathy.core.attention import Gate
from telepathy.core.memory import ForeignView
from telepathy.core.projector import BridgeProjector
from telepathy.runtime.decoder import FrozenDecoder
from telepathy.runtime.toy import ToyModel
from telepathy.runtime.manifest import parameter_digest
from telepathy.train.fit import receiver_kl, fit_mlp, save_projector, load_projector
from telepathy.train.alignment import shared_causal_endpoints
from telepathy.mailbox.protocol import ThoughtWriter, MailboxLedger, MailState, ProbeHead
from telepathy.plugin.control import ControlEvent, Op
from telepathy.eval.metrics import paired_bootstrap, e2_gate, CostLedger


def test_prefill_equals_incremental_and_self_import():
    worker = FrozenDecoder(ToyModel())
    ids = torch.tensor([1, 4, 7, 2, 5, 8], dtype=torch.long)
    full, prefix = worker.forward(ids), worker.forward(ids[:4])
    native = worker.forward(ids[4:], prefix.state)
    imported = worker.import_self_prefix(prefix.canonical_delta, torch.arange(4))
    handoff = worker.forward(ids[4:], imported)
    torch.testing.assert_close(full.logits[4:], native.logits)
    torch.testing.assert_close(handoff.logits, native.logits)
    with pytest.raises(ValueError): worker.import_self_prefix({0: prefix.canonical_delta[0]}, torch.arange(4))


def test_closed_foreign_bank_does_not_alter_native_weights_or_output():
    worker = FrozenDecoder(ToyModel())
    before = parameter_digest(worker.model)
    source = worker.forward(torch.tensor([1, 2, 3]))
    memory = ForeignView(0, torch.arange(3), source.canonical_delta)
    ids = torch.tensor([5, 6])
    baseline, closed = worker.forward(ids), worker.forward(ids, foreign=memory, override=0)
    assert torch.equal(baseline.logits, closed.logits)
    assert parameter_digest(worker.model) == before


def test_gradients_pass_through_frozen_decoder_to_sidecars():
    worker = FrozenDecoder(ToyModel())
    before = parameter_digest(worker.model)
    projector, gate = BridgeProjector((1, 4), (2, 6), "mlp", rank=8), Gate(-1)
    source = KV(torch.randn(4, 1, 4), torch.randn(4, 1, 4))
    projected = projector(source)
    view = ForeignView(0, torch.arange(4), MappingProxyType({1: projected}))
    out = worker.forward(torch.tensor([1, 2, 3]), foreign=view, gates={1: gate})
    teacher = out.logits.detach().clone(); teacher[:, 5] += 4.0
    loss = receiver_kl(out.logits, teacher)
    loss.backward()
    assert gate.logit.grad is not None and gate.logit.grad.abs() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in projector.parameters())
    assert all(p.grad is None and not p.requires_grad for p in worker.model.parameters())
    assert before == parameter_digest(worker.model)


def test_receiver_kl_rejects_detached_or_different_vocab():
    with pytest.raises(ValueError): receiver_kl(torch.zeros(2, 7), torch.zeros(2, 7))
    with pytest.raises(ValueError): receiver_kl(torch.zeros(2, 7, requires_grad=True), torch.zeros(2, 8))


def test_private_mailbox_does_not_modify_parent_and_marker_trains():
    worker, writer = FrozenDecoder(ToyModel()), ThoughtWriter(24)
    state = worker.forward(torch.tensor([1, 2, 3])).state
    old = state.layers[0].k.clone()
    kv = writer.write(worker, state, torch.tensor([4, 5]))
    assert state.length == 3 and torch.equal(old, state.layers[0].k)
    assert all(x.tokens == 3 for x in kv.values())  # marker + two local tokens
    sum(x.v.square().sum() for x in kv.values()).backward()
    assert writer.marker.grad is not None and writer.marker.grad.abs().sum() > 0


def test_mail_attended_is_not_delivered_and_starvation_is_counted():
    ledger = MailboxLedger(capacity=1); ledger.create(3, 0, 5)
    ledger.observe_attention(3, 1, mass=0.9)
    assert ledger.records[3].state == MailState.ATTENDED
    with pytest.raises(OverflowError): ledger.create(4, 1, 5)
    with pytest.raises(ValueError):
        ledger.record_causal_use(3, 2, active_correct=True, ablated_correct=False,
                                 replay_started_before_exposure=False)
    ledger.expire(5)
    assert ledger.records[3].state == MailState.STARVED
    ledger.create(4, 5, 5)
    ledger.record_causal_use(4, 6, active_correct=True, ablated_correct=False,
                             replay_started_before_exposure=True)
    assert ledger.records[4].incorporation == 6


def test_probe_detaches_features_from_models():
    kv = KV(torch.randn(3, 2, 4, requires_grad=True), torch.randn(3, 2, 4, requires_grad=True))
    probe = ProbeHead(2, 4, 7); probe(kv).sum().backward()
    assert kv.k.grad is None and kv.v.grad is None and probe.head.weight.grad is not None


def test_unicode_causal_endpoint_alignment():
    pairs = shared_causal_endpoints("aé🙂z", [(0, 1), (1, 3), (3, 4)], [(0, 2), (2, 3), (3, 4)])
    assert pairs == [(1, 1, 7), (2, 2, 8)]
    assert shared_causal_endpoints("abc", [(0, 1)], [(0, 2)]) == []
    with pytest.raises(ValueError): shared_causal_endpoints("a", [(0, 2)], [(0, 1)])


def test_safe_projector_checkpoint_roundtrip_and_integrity(tmp_path):
    projector = BridgeProjector((2, 4), (1, 6), "mlp", rank=8)
    path = tmp_path / "map"; save_projector(projector, path, {"split": "train"})
    reloaded = load_projector(path)
    kv = KV(torch.randn(3, 2, 4), torch.randn(3, 2, 4))
    torch.testing.assert_close(projector(kv).k, reloaded(kv).k)
    weights = path / "projector.safetensors"
    with weights.open("ab") as handle: handle.write(b"corruption")
    with pytest.raises(ValueError, match="hash"): load_projector(path)


def test_mlp_fitting_runs_with_declared_budget():
    source = KV(torch.randn(64, 1, 4), torch.randn(64, 1, 4))
    target = KV(source.k * 0.5 + 0.1, source.v * 0.2 - 0.1)
    projector = BridgeProjector((1, 4), (1, 4), "mlp", rank=8)
    report = fit_mlp(projector, source, target, steps=12, batch_size=16)
    assert report["steps"] == 12 and report["row_presentations"] == 192
    assert report["last_loss"] >= 0


def test_control_plane_rejects_text_and_extra_fields():
    payload = {"run": "00000000-0000-0000-0000-000000000041", "op": 2, "epoch": 0}
    assert ControlEvent.parse(payload).op == Op.TICK
    with pytest.raises(ValueError): ControlEvent.parse(dict(payload, answer="secret"))
    with pytest.raises(ValueError): ControlEvent.parse(dict(payload, epoch=True))


def test_paired_statistics_and_validity_gate():
    result = paired_bootstrap([1.0] * 100, [0.0] * 100, repetitions=500)
    assert result["ci95"] == [1.0, 1.0]
    assert e2_gate(result, valid_channel_audit=True, preregistered=True) == "PASSED"
    assert e2_gate(result, valid_channel_audit=False, preregistered=True) == "INVALID"
    assert e2_gate(result, valid_channel_audit=True, preregistered=False) == "BLOCKED"
    result = paired_bootstrap([0.0] * 100, [0.0] * 100, repetitions=500)
    assert e2_gate(result, valid_channel_audit=True, preregistered=True) == "FAILED"


def test_energy_unmeasured_is_not_zero():
    ledger = CostLedger(); ledger.validate(); assert ledger.joules is None
    ledger.wire_bytes = -1
    with pytest.raises(ValueError): ledger.validate()


def test_unsupported_rotary_fails_closed():
    model = ToyModel(); model.config.rope_scaling = {"rope_type": "dynamic"}
    with pytest.raises(ValueError): FrozenDecoder(model)
````

### Source 47 — `tests/test_transport_sync.py`

<!-- file: tests/test_transport_sync.py sha256: 202e1760e3a1b64c55d51374001f78edc5f1228c35878e8525e0b4c363c5b464 -->
````python
import hmac
import queue
import socket
import struct
from uuid import UUID
import pytest
import torch
from telepathy.core.types import Delta, KV
from telepathy.core.memory import ForeignKVBank
from telepathy.core.sync import SyncController
from telepathy.transport.wire import FrameCodec, HEADER, MAGIC
from telepathy.transport.backends import InProcTransport, SocketTransport, MCDMATransport
from test_core import identity

SESSION = UUID(int=41)


def delta(epoch=0, direction=0):
    return Delta(SESSION, direction, epoch, epoch, epoch,
                 {0: KV(torch.randn(1, 2, 4), torch.randn(1, 2, 4))})


def test_binary_roundtrip_and_no_plaintext_metadata():
    codec, original = FrameCodec(b"x" * 32), delta()
    frame = codec.encode(original); restored = codec.decode(frame)
    assert frame.startswith(MAGIC)
    assert b"token_ids" not in frame and b"question" not in frame
    assert restored.session == original.session
    torch.testing.assert_close(restored.layers[0].k, original.layers[0].k)


def test_authentication_and_wrong_key_rejected():
    codec = FrameCodec(b"x" * 32); encoded = codec.encode(delta())
    with pytest.raises(ValueError): codec.decode(encoded[:-1] + bytes([encoded[-1] ^ 1]))
    with pytest.raises(ValueError): FrameCodec(b"y" * 32).decode(encoded)


def test_signed_trailing_bytes_and_oversized_dimensions_rejected():
    key, codec = b"x" * 32, FrameCodec(b"x" * 32)
    body = codec.encode(delta())[:-32] + b"untyped"
    with pytest.raises(ValueError): codec.decode(body + hmac.digest(key, body, "sha256"))
    body = HEADER.pack(MAGIC, 1, 0, SESSION.bytes, 0, 0, 0, 900000, 1)
    with pytest.raises(ValueError): codec.decode(body + hmac.digest(key, body, "sha256"))


def test_bounded_inproc_queue_and_closed_guard():
    transport = InProcTransport(FrameCodec(b"x" * 32), capacity=1, timeout=0.01)
    original = delta(); transport.send(original)
    original.layers[0].k.fill_(333)
    with pytest.raises(queue.Full): transport.send(delta(1))
    assert not torch.equal(transport.receive().layers[0].k, original.layers[0].k)
    transport.close()
    with pytest.raises(RuntimeError): transport.send(delta())


def test_socketpair_roundtrip():
    a, b = socket.socketpair()
    tx, rx = SocketTransport(a, FrameCodec(b"x" * 32)), SocketTransport(b, FrameCodec(b"x" * 32))
    try:
        original = delta(); tx.send(original); received = rx.receive()
        torch.testing.assert_close(received.layers[0].v, original.layers[0].v)
    finally:
        tx.close(); rx.close()


def test_actual_loopback_tcp_roundtrip():
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.settimeout(2)
    try:
        listener.bind(("127.0.0.1", 0)); listener.listen(1)
        client = socket.create_connection(listener.getsockname(), timeout=2)
        server, _ = listener.accept()
    except PermissionError:
        listener.close(); pytest.skip("sandbox forbids loopback TCP bind")
    tx, rx = SocketTransport(client, FrameCodec(b"x" * 32)), SocketTransport(server, FrameCodec(b"x" * 32))
    try:
        original = delta(); tx.send(original)
        torch.testing.assert_close(rx.receive().layers[0].k, original.layers[0].k)
    finally:
        tx.close(); rx.close(); listener.close()


def test_truncated_frame_poisons_socket():
    a, b = socket.socketpair(); receiver = SocketTransport(b, FrameCodec(b"x" * 32))
    a.sendall(struct.pack("!I", 100) + b"short"); a.close()
    with pytest.raises(EOFError): receiver.receive()
    assert receiver.closed


def test_mcdma_fails_loudly_instead_of_faking_hardware():
    with pytest.raises(RuntimeError, match="M4 BLOCKED"): MCDMATransport()


def test_causal_scheduler_pins_both_views_before_computation():
    a = ForeignKVBank(SESSION, 1, [(0, 0, identity())])
    b = ForeignKVBank(SESSION, 0, [(0, 0, identity())])
    scheduler, seen = SyncController(a, b), []
    def worker(direction):
        def run(view, epoch):
            seen.append((direction, epoch, None if view is None else view.epoch))
            return delta(epoch, direction)
        return run
    for _ in range(3): scheduler.tick(worker(0), worker(1))
    assert seen == [(0, 0, None), (1, 0, None), (0, 1, 0), (1, 1, 0), (0, 2, 1), (1, 2, 1)]


def test_failed_scheduler_cannot_silently_resume():
    a = ForeignKVBank(SESSION, 1, [(0, 0, identity())]); b = ForeignKVBank(SESSION, 0, [(0, 0, identity())])
    scheduler = SyncController(a, b)
    def fail(view, epoch): raise RuntimeError("injected failure")
    with pytest.raises(RuntimeError): scheduler.tick(fail, fail)
    assert scheduler.failed
    with pytest.raises(RuntimeError, match="poisoned"): scheduler.tick(fail, fail)
````

### Source 48 — `validation/REFERENCE_VALIDATION.json`

<!-- file: validation/REFERENCE_VALIDATION.json sha256: c9301df12bab0d5507b5913e9a039bdfe18c597dfd1ae0db7f6aaeab8950ed68 -->
````json
{
  "issued": "2026-09-18",
  "scope": "CPU correctness reference, random toy decoder, synthetic KV arrays, local loopback TCP",
  "pytest": {
    "passed": 42,
    "failed": 0,
    "errors": 0,
    "skipped_modules": 1,
    "skip_reason": "Optional transformers package unavailable; actual HF architecture tests were not executed"
  },
  "self_handoff": {
    "fixture": "random_toy",
    "status": "PASSED",
    "errors": {
      "self_handoff_max_abs": 0.0,
      "prefill_vs_incremental_max_abs": 2.980232238769531e-07
    },
    "weights_unchanged": true,
    "semantic_transfer": "NOT_EVALUATED"
  },
  "transport_affine_smoke": {
    "fixture": "synthetic_affine_mapping",
    "max_abs_key_error": 4.76837158203125e-07,
    "status": "PASSED",
    "frame_bytes_including_hmac": 1118,
    "tensor_payload_bytes_float32": 1024,
    "receiver_resident_slots": 16,
    "cross_family_language_transfer": "NOT_EVALUATED",
    "mcdma": "NOT_USED",
    "energy_joules": null
  },
  "cli_checks": {
    "fit_ridge_cli": "PASSED",
    "fit_mlp_cli": "PASSED",
    "external_scorer_cli": "PASSED_ON_SYNTHETIC_FIXTURE",
    "unresolved_manifest_guard": "PASSED_EXPECTED_EXIT_2"
  },
  "python_compileall": "PASSED",
  "local_wheel_build": "PASSED_NO_DEPENDENCY_DOWNLOAD",
  "native_contract_header": "DECLARATIONS_ONLY_SYNTAX_CHECKED_NOT_IMPLEMENTED",
  "markdown_materialization": "PASSED_SOURCE_HASHES_AND_EXTRACTED_SUITE_42_PASSED_1_SKIPPED",
  "not_validated": [
    "real pretrained checkpoints",
    "HF optional architecture tests",
    "semantic E1-E4 experiments",
    "CUDA device execution",
    "MLX/Metal execution",
    "native MCDMA integration",
    "OMP plugin integration",
    "trained mailbox addressing",
    "distributed restart/replay service",
    "energy and hardware performance"
  ],
  "embedded_file_count": 48
}
````

---

**End of handoff. Start at M−1. Run the baseline tests before modifying the source.**
