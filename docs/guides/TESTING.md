# Tester handoff

This handoff is for a tester who already has MCDMA and multiple models on their own hardware. The next step is to identify one supported pair and its serving runtimes, not to copy the original deployment's addresses or run its orchestration scripts unchanged.

Live test status: `NOT_RUN`; engineering admission: `BLOCKED` pending the tester's inventory and a reviewed, bounded run configuration. This document does not mark the friend’s hardware ready or advance M-1 through M6.

## What the tester can do now

Record the following without starting, stopping or reconfiguring a model or MCDMA service:

| Item | Information needed |
| --- | --- |
| Models | Exact checkpoint identifiers and revisions, text-model architecture, quantization and tokenizer revision for one proposed pair |
| Serving software | Runtime name, installed version or source commit, and whether its cache tap/import hooks can be accessed locally |
| Hosts | OS/build, CPU/GPU models, GPU count, available memory, CUDA or Metal runtime and tensor-parallel arrangement |
| MCDMA | Existing source commit/build identifier, transport type, active endpoint identity, registered-region ownership and documented completion/visibility behavior |
| Resources | Owner-approved wall time, model token limits, memory/storage limits and a period when the selected runtimes are idle |
| Access boundary | Who may launch the test and where private model inputs, model outputs and controller evidence may be stored |

Use labels such as `host-a` and `host-b` in anything shared; omit credentials, account identifiers, private addresses, absolute personal paths, prompts, outputs and cache contents. A short model/runtime/hardware summary is enough to select the first integration task; no weight upload is required. Do not invoke a daemon binary with `--help` to discover its version, because some existing binaries start a service instead.

From the reviewed Drift source root and an already approved Python environment, these commands inspect package metadata without loading models:

```bash
python3 --version
python3 -m pip show torch numpy safetensors pytest transformers mlx mlx-lm mlx-vlm vllm
```

Missing packages in that output are inventory facts, not an instruction to install or upgrade the serving environment. `pyproject.toml` specifies Python 3.11 through 3.13 and the CPU reference dependency pins; that environment is distinct from the experimental model runtimes.

## Optional CPU reference smoke

Use an existing approved environment with Drift's reference dependencies and pytest installed; run from the reviewed source root, with `python3` resolving to that environment. These commands use random toy tensors and a toy model, never the tester's checkpoints or MCDMA endpoints.

```bash
umask 077
drift_smoke_dir=$(mktemp -d)
PYTHONPATH=. python3 scripts/doctor.py --out "$drift_smoke_dir/environment.json"
PYTHONPATH=. python3 -m pytest -q tests/test_core.py tests/test_transport_sync.py tests/test_mcdma_mailbox.py
PYTHONPATH=. python3 scripts/self_handoff.py --device cpu --out "$drift_smoke_dir/toy-identity.json"
PYTHONPATH=. python3 scripts/reference_demo.py --out "$drift_smoke_dir/synthetic-transfer.json"
```

The temporary directory holds only this smoke run's reports and is not uploaded automatically. Run the commands individually and stop after the first error. A passing synthetic-transfer report explicitly says `mcdma: NOT_USED`, and the self-handoff report says `semantic_transfer: NOT_EVALUATED`; the mailbox tests use fake regions and do not validate the installed daemon. These checks establish reference mechanics only, not model compatibility or live telepathy.

## Compatibility boundaries in this checkout

| Code path | What exists | What it does not establish |
| --- | --- | --- |
| `drift/runtime/decoder.py` | Dense `drift_toy`, `qwen3` and `llama` support with explicit rotary/layout restrictions | Support for arbitrary hybrid architectures, quantization or the tester's serving process |
| `drift/adapters/` | PyTorch and MLX source adapters for `qwen4_exp` and `glm5_next` architecture identifiers | Qualification of every similarly named checkpoint or runtime release |
| `drift/runtime/builders.py` | Real manifest-loader branches for `qwen4_exp/torch` and `glm5_next/torch` | A universal loader for every identifier listed in the registry |
| `scripts/live/studio_mcdma_loop.py` | The existing deployment-specific Qwen/GLM two-way experiment | A portable launcher for another pair or another hardware layout |
| `drift/transport/backends.py` | In-process and socket reference backends; the generic `MCDMATransport` constructor deliberately refuses construction | Automatic selection of the separate experimental MCDMA mailbox path |

No MiniMax or Nemotron adapter is registered in this checkout. Having those models and a working MCDMA link does not yet supply their canonical cache taps, receiver integration or trained directional translators. Adapter source presence and registry entries are not qualification evidence; a skipped required HF test leaves the real-adapter gate `BLOCKED`.

The existing live experiment imports oMLX and MLX-VLM internals, selects a particular local Qwen checkpoint, uses explicit layer/head mappings and both directions' fitted translator artifacts, and expects a matching GLM connector and two-rank mailbox setup. It also embeds deployment-specific paths. The coordinators read the Studio's MCDMA legs from `DRIFT_MCDMA_LINKS`, one `target/source` address pair per rank with the head first, and refuse a linked run without them. The repository holds no real address. Changing the SSH host and link environment variables alone does not make that experiment portable.

## Before a live test can be offered

First match the tester's exact architecture/runtime pair to real tap and receiver hooks, pin its source/config/tokenizer/weight identities and establish native parity with foreign memory disabled. Keep both model backbones frozen. Inspect the tester's actual MCDMA API and region contracts before integration; a successful byte transfer does not establish GPU visibility or correct cache application.

Prepare a private, host-specific run manifest with verified target identity, exclusive approved mailbox ownership, no competing bridge or inference job, bounded startup and wall deadlines, token limits and session-specific cleanup. Confirm both directional translators match the selected checkpoints and layout, and verify per-rank cache-application receipts separately from transport acknowledgements. Do not reuse the original pair's translator files for another model pair without evidence.

A supervised exploratory test should keep startup costs separate from steady-state transfer/application timing and include no-link and one-way controls before interpreting a two-way result. Keep answer keys, raw outputs and private cache evidence outside model-readable repositories and tools; return only approved aggregate status and timing to an assistant. This is not a held-out scientific evaluation.

The original `demo_appointment.py` and `mcdma_loop_test.py` are not the tester's live command: they can stage files and launch remote processes, and their deployment assumptions require review. No portable live invocation is supplied until the inventory, adapter path and safety checks have been validated. If a bounded test later times out or loses contact, treat remote cleanup as unconfirmed until independently checked; do not infer that local SSH exit stopped every remote worker, restart services or stop shared daemons as recovery.

The immediate handoff back to the maintainer is the proposed pair's exact checkpoint/runtime names and hardware layout, plus the installed MCDMA build identifier. That determines whether the next work is qualifying an existing adapter or implementing a missing one.
