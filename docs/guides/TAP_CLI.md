# User-created taps

The `drift` command creates a fail-closed two-model tap project without loading either model or changing its weights.
It hashes each local checkpoint, tokenizer set, and runtime lock, checks adapter availability, validates supplied level-2 qualification evidence, validates both translators, and writes an immutable project inventory.
Unknown architectures produce a usable `BLOCKED` project with exact next gates rather than pretending shape compatibility is enough.

## Create a tap

Install this repository in an isolated environment, then run:

```bash
drift tap create minimax-nemotron \
  --source-checkpoint /models/minimax-m3 \
  --source-adapter minimax_m3/torch \
  --source-host local \
  --source-runtime-lock /locks/minimax.lock \
  --target-checkpoint /models/nemotron-lighting-3 \
  --target-adapter nemotron_lighting_3/torch \
  --target-host local \
  --target-runtime-lock /locks/nemotron.lock \
  --output ./taps/minimax-nemotron
```

Exit code `2` means the project was created but remains `BLOCKED` by missing adapters, real-weight qualification, or translators.
Exit code `3` means the request or stored evidence is `INVALID`.
The command never overwrites an existing output directory.

Run `drift tap status ./taps/minimax-nemotron` to rehash every generated artifact and detect edits.

## Add a model architecture

List built-in and installed adapter loaders with `drift adapter list`.
For an architecture that is not installed, create a separate adapter package:

```bash
drift adapter scaffold minimax_m3/torch \
  --model-type minimax_m3 \
  --output ./drift-adapter-minimax-m3
```

The scaffold registers a `drift.adapters` entry point and starts with a deliberate `BLOCKED` loader.
Implement the adapter contract in `docs/guides/ADAPTERS.md`, then pass tiny-config and real-weight qualification before supplying its report to `tap create`.
The CLI does not infer cache boundaries, rotary behavior, or foreign-attention injection from a model name.

## Compile a runnable local tap

Pass `--source-qualification`, `--target-qualification`, `--source-translator`, and `--target-translator` after both adapters qualify.
A level-2 report has to set `passed` to true and bind the checkpoint's config, weight manifest, tokenizer manifest, runtime lock, adapter id, and `model_type`.
A known adapter also has to admit that `model_type`: `qwen4_exp` admits `qwen4_exp_text`, and `glm5_next` admits `glm5_next_text`.
Both translator directories must be the `pool.v1` artifact written by `save_translator`, for the same pool, and the CLI loads them before it will call the project passed.
When those gates pass, the project status is `PASSED` and the CLI writes `run.json` plus registry entries for the existing service builder.
`drift tap status` recomputes that verdict from the inventory files. A `tap.json` whose status disagrees with the evidence is `INVALID`.

Two different hosts remain `BLOCKED` in the CLI because it does not create portable native-worker profiles or launch a remote pair.
The separately configured native GLM/Qwen coordinator has bounded lifecycle evidence, described in [the native guide](NATIVE_OWNER_EXCHANGE.md); that does not qualify arbitrary CLI host assignments.
The tap project still records both host assignments, but it does not emit a misleading runnable manifest.
