# Explicit experimental worker registration

This opt-in extension registers `drift-experimental/<model_id>` using the actual installed OMP provider API and the private worker subprocess protocol. It does not change `/drift`, ordinary Duo, model weights, serving processes or transport. Names visibly include EXPERIMENTAL, native_state retained/reconstructed/fixture and memory_mode no-link/linked. Memory mode is an owner declaration, not a new verified protocol capability; only a separately qualified backend may claim linked operation.

## Owner configuration

Set `DRIFT_WORKER_CONFIG` to an absolute owner JSON path and `DRIFT_WORKER_CONFIG_SHA256` to its byte SHA256, then load `scripts/omp/experimental_extension.mjs` with OMP `--extension`. Configuration contains version 1, experimental true and one or two workers. Each worker supplies identity `{session,worker,model_id,model_sha256,translator_sha256}`, the five protocol limits, expected `{backend:live|fixture,nativeStates:[retained|reconstructed|fixture]}`, context_window, memory_mode, command `{executable,sha256,args,cwd,env}`, and a nonempty artifacts array of `{path,sha256}`. Use absolute paths; env is explicit and does not inherit OMP credentials. The hash of the entire configuration binds all argv, environment, capabilities and limits. Executable and artifacts are rehashed before spawn. List every mutable local launcher/backend/config dependency; no automatic import-closure guarantee is claimed. Remote command contents and model artifacts remain the launcher's verification responsibility.

Workers start lazily when selected and receive only their own context. Each registration is a one-shot coding session with intermediate tool turns and acknowledged close before its final answer. A subsequent user task needs a new OMP session/config identity. This avoids the installed CLI's asynchronous shutdown-hook race and does not claim engine persistence for reconstructed GLM state. The client now enforces one absolute monotonic wall-clock deadline from open, not a fresh deadline for every request.

## Qualification and cross-language regression

The previous independent fixture implemented its own envelope handling and missed a concrete mismatch: TS started at sequence 1 while Python WorkerSession required sequence 0. The new registration probe uses the actual Python worker_stdio, WorkerSession, contract and artifact verification modules with a tiny backend fixture that implements only generation callbacks. Before the runtime fix both registered selectors failed before backend open; after first-sequence alignment both completed open 1, own_prompt 1, tool_result 1, stream 2, close 1, with exactly one real OMP tool execution each. This is a synthetic backend qualification, not model inference.

```sh
python3 scripts/omp/registration_probe.py --omp /home/example/.local/bin/omp --worker-source /home/example/.codex/worktrees/drift-worker-session/Telepathy
```

The probe copies actual protocol modules into a temporary package, verifies real tiny manifest artifacts, denies network, isolates OMP config/session state, and tests selectors glm-fixture and qwen-fixture against the explicit registration extension. OMP executable SHA256 is `d498da40d577e1ffa681ca8632c2ea40a9f722a08b880412011d37dffee9513a`; pre-fix session SHA256 was `bcd067564033f1ebb7c27300bdd359a952774930aa66bc29917a29472b50c76b`; fixed session SHA256 at the first passing probe was `561cbfe6bca37deb722f2970a600b1836434087e79d678d478403e169d72310d`; stdio SHA256 was `1c173939187bb33b22dbd8d2799f9602f84dbdbe314b13fba41d495c91569d5d`. Measured wall time was 1.154 seconds for both selectors. No model tokens, money or energy were measured.

## Bounded real echo probe: prepared, not run

```sh
python3 scripts/omp/native_probe.py --omp /home/example/.local/bin/omp --owner-config /ABS/owner.json --owner-sha256 OWNER_SHA256 --model OWNER_MODEL_ID
```

The owner config must select backend live, exactly two turns, no more than 64 new tokens per turn (128 total) and no more than 60 seconds. The sole registered model tool returns a fixed echo string; built-ins, task tools, skills/rules and extension discovery are disabled. Generated text and stderr are discarded; the returned report contains aggregate tool/turn/token counts, status, elapsed time and hashes. It does not launch without explicitly running this command. No native model probe was performed by this ticket. This runner is not OS-network-sandboxed because its explicit owner backend may be an SSH process; the model has no network/filesystem tool surface. Controller/worker trust and resource authority must be resolved by the parent before execution.

## Prepared task-local API smoke

`task_smoke.py` creates a new small task directory containing api.py and a visible verify.py, writes a scope manifest outside that directory, and prints an invocation; it never runs OMP. Supply --task-root, --scope-output, --owner-config, --model and --omp. The selected worker must have at most six turns, 8192 session tokens, 60 seconds and 16384 context tokens. The prompt asks for a one-line health API fix and running the visible verification command; this is a development smoke, not a benchmark or hidden evaluation.

`task_tools.mjs` exposes only drift_task_read, drift_task_edit and drift_task_run. Read/edit reject path traversal and symlinks and cap file bytes. Run allows only exact owner-listed argv, verifies the executable hash and uses sandbox-exec with no network, no process forking, writes only inside the task directory, bounded time/output, and explicit runtime read roots. Regression checks actually ran permitted task reads and Python arithmetic while denying a sibling private sentinel. The sandbox denies reads in Users, Volumes, private, Applications, Library, opt and cores except its explicit task/runtime allowances; it still permits operating-system paths needed to launch tools and metadata reads. It is not a universal confidentiality claim: keep private evidence outside the task and allowed runtime roots, retain real directory/process isolation, and do not place hidden tests in the task checkout. General deny-all read policy aborted even /bin/echo on this host and was not represented as working containment. Tool subprocess sandboxing is separate from OMP/controller and model-serving process authority.

## Checks and remaining gates

Plugin baseline 40 passed; final 47 passed, TypeScript clean. Red tests covered unavailable registration, accepted late client operations after the absolute session deadline, and a failing overly broad sandbox before focused fixes. Cross-language red/green is described above. Full reference suite: 407 passed, 3 skipped; next suite: 155 passed, 1 pre-existing expected sparse-indexer failure. Skipped real adapter gates remain BLOCKED.

Next: merge the actual WorkerSession first-sequence fix, configure and verify the real launcher/model/translator artifacts, run the bounded native echo probe under approved resources, and only then execute the prepared task smoke. Retained Studio memory and reconstructed GLM foreign snapshot restoration require backend evidence; synthetic selector success does not establish either.
