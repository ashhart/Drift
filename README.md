# Drift

**A memory link between models.**

An agent can write a handover. Drift explores another route: let its partner read translated working memory.

Drift connects frozen language models through their key-value caches. Model-specific taps capture memory, trained translators map it into a form the receiver can use, and the receiver attends to it alongside its own context. The backbone weights stay frozen.

The goal is a choice in your workflow: ordinary text communication, or a Drift memory link over MCDMA. Duo is the first agent integration; the exchange components and tap CLI live here so other workflows can use the same machinery.

> Experimental. You can use the CLI, extend adapters and run the local reference today. The repository also contains exploratory two-way MCDMA code, but reliable native two-way recall and the complete Duo/subagent workflow are not yet qualified.

[CLI guide](docs/TAP_CLI.md) · [Adapter contract](docs/ADAPTERS.md) · [Exchange API](docs/EXCHANGE_MODULES.md) · [Current progress](docs/agent-progress.md)

## What crosses the link

The designated activation channel carries KV tensors and bounded protocol metadata. It does not carry task text or token IDs. Each direction needs a compatible tap, translator and receiver; matching tensor shapes alone is not enough.

MCDMA moves the memory. Drift handles its representation, ownership, sequence and application. The agent runtime still owns tools, local generation and lifecycle.

This is a channel-specific claim. Setup prompts, locally generated tokens and shared repository files still exist, and any text fallback must be declared. A transport acknowledgement is not proof that a model used the memory correctly.

## Start locally

Use Python 3.11–3.13 in a fresh environment. The dependency pins describe the reference build; use an approved PyTorch build for your host and requalify any variation, without upgrading a serving environment in place.

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test]'

drift --help
drift adapter list
python -m pytest -q
```

The reference suite does not need model downloads, a Studio, Sparks or a running MCDMA daemon. Optional runtime tests can skip when their environment is absent; those skips are blocked qualification gates, not passes.

For a small self-handoff check with the random toy model:

```bash
mkdir -p local/reference
python scripts/self_handoff.py --out local/reference/self_handoff.json
```

That checks cache mechanics, not language understanding or cross-family recall.

## Create your own tap

The CLI inventories two local checkpoints, hashes their inputs and checks the evidence needed to connect them. It does not load the backbone models, train a translator or start remote services.

Replace these paths with your checkpoints and recorded runtime locks. This example uses the built-in Qwen and GLM Torch adapter IDs; their runtime requirements and qualification still apply.

```bash
drift tap create my-pair \
  --source-checkpoint /models/qwen \
  --source-adapter qwen4_exp/torch \
  --source-host local \
  --source-runtime-lock /locks/qwen.lock \
  --target-checkpoint /models/glm \
  --target-adapter glm5_next/torch \
  --target-host local \
  --target-runtime-lock /locks/glm.lock \
  --output ./local/taps/my-pair

drift tap status ./local/taps/my-pair
```

Without qualification reports and translators, this creates a `BLOCKED` project and tells you which gates are missing. Status checks rehash the project artifacts and inputs rather than trusting a saved verdict. Existing output directories are never overwritten.

| Exit code | Meaning |
| --- | --- |
| `0` | `PASSED` for the requested operation |
| `1` | `FAILED` |
| `2` | `BLOCKED`, with missing prerequisites reported |
| `3` | `INVALID` request or evidence |

To emit a runnable local manifest, also supply `--source-qualification`, `--target-qualification`, `--source-translator` and `--target-translator`. Both adapters must qualify against the exact checkpoint and runtime, and both translators must bind to the same `pool.v1` format. Different-host projects remain `BLOCKED`; entering two hostnames does not qualify a distributed run.

Want to add another family? Generate a separate adapter package:

```bash
drift adapter scaffold my_model/torch \
  --model-type my_model \
  --output ./local/drift-adapter-my-model
```

The scaffold deliberately starts with a blocked loader. Implement the [adapter contract](docs/ADAPTERS.md), then qualify it on tiny configurations and real weights. A model name is not a compatibility guarantee, including MiniMax or Nemotron variants.

The [full CLI guide](docs/TAP_CLI.md) covers qualification artifacts, translator inputs and plugin registration.

## Duo and `/drift`

`drift` is the shell CLI. `/drift` is an OMP command registered by the separate `omp-drift` plugin in this repository, not by Duo.

From this repository's root, link the plugin and restart OMP:

```bash
omp plugin link "$PWD/plugin/omp-drift"
```

That adds the command without modifying Duo's source. Configure the local service and authentication described in the [plugin guide](plugin/omp-drift/README.md) before using `/drift start --reference`.

The command currently controls the reference service. It does **not** turn an existing Duo room into a native KV-linked pair. The worker-provider exchange hook is wired locally, but the native runtime-owned coordinator bootstrap and complete two-actor qualification remain open. A text/Drift mode selector must wait for those checks; there is no supported live-mode shortcut.

## Add Drift to another workflow

The reusable exchange core takes a source, link and sink. It owns the cursors, capacity checks and poisoned-session state while runtime adapters own the model caches and transport calls.

Keep that coordinator alive across exchanges. Spawning a new CLI process at each tool boundary loses the state that makes sequencing safe. See the [exchange contracts and integration limits](docs/EXCHANGE_MODULES.md) before connecting a main agent, subagent or another host runtime.

| Code | Responsibility |
| --- | --- |
| [`drift/taps/`](drift/taps/) and [`drift/cli.py`](drift/cli.py) | Checkpoint inventory, qualification checks and tap projects |
| [`drift/adapters/`](drift/adapters/) and [`drift/translate/`](drift/translate/) | Model-specific taps and memory translators |
| [`drift/exchange/`](drift/exchange/) | Stateful exchange contracts, live adapters and coordinator |
| [`drift/serving/`](drift/serving/) | Native worker sessions, cache application and MCDMA integration |
| [`plugin/omp-drift/`](plugin/omp-drift/) | OMP commands, worker provider and lifecycle controls |
| [`drift/eval/`](drift/eval/) and [`tests/`](tests/) | Evaluation machinery and engineering regressions |

## If you are an agent

Read [AGENTS.md](AGENTS.md) and [TELEPATHY_AGENT_ENGINEERING.md](TELEPATHY_AGENT_ENGINEERING.md) before changing this repository. The handoff's embedded source is a historical reference, not a manifest for the current checkout.

1. Read the [latest progress](docs/agent-progress.md) and identify the first failing or unimplemented gate. Preserve the M−1 through M6 sequence.
2. Inspect the actual runtime and integration source. Do not invent provider APIs, model support, hardware guarantees or measurements.
3. Run the baseline tests before code changes, then focused tests and the full suite after a meaningful fix. Keep modules focused and code comments to one line.
4. Keep backbone weights frozen and task text off the live activation channel. Stop on corrupted activations, unexplained native parity failures or poisoned sessions.
5. Keep private activations, credentials and scorer evidence outside agent-readable project trees. Do not start remote jobs, restart services or change drivers without the required ownership and authorization.
6. Record the exact commit, commands, test results, evidence hashes, measured costs and next blocker in `docs/agent-progress.md`. Use `PASSED`, `FAILED`, `BLOCKED` or `INVALID`, and never count a required skip as a pass.

A green local suite does not establish native recall, source ownership or successful agent collaboration. Do not remove a guard to make a demo run.

## Before connecting real machines

Public examples use `.invalid` hostnames, TEST-NET addresses and example paths. Replace them with verified private configuration and pin the deployed code, model and translator hashes.

Measure transport time, cache application and task success separately. Millisecond delivery does not establish millisecond incorporation, and the exploratory asynchronous loop does not qualify the reference's prior-epoch schedule.

Keep receipts, private activations and credentials out of commits and release archives. The original development history is private; publication uses a separately audited source snapshot.
