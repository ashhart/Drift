# Documentation

Start with [current status](STATUS.md) for supported paths, measured results and
limits, or the [roadmap](ROADMAP.md) for the next engineering and research gates.

## Use Drift

- [Tap CLI](guides/TAP_CLI.md): inventory checkpoints, supply translators and qualify a local tap.
- [Transcript-backed memory](guides/TRANSCRIPT_MEMORY.md): let local GLM read API messages and transfer its latent memory to Qwen.
- [Model adapters](guides/ADAPTERS.md): implement and qualify a model/runtime integration.
- [Test your own pair](guides/TESTING.md): collect an inventory and run the local reference safely.
- [Native GLM/Qwen exchange](guides/NATIVE_OWNER_EXCHANGE.md): owner setup, staged application and tested lifecycle scope.
- [OMP plugin](../plugin/omp-drift/README.md): install the command and configure the reference service.

## Develop and operate

- [Technical reference](reference/README.md): exchange, model-cache, worker, OMP and service contracts.
- [Evaluation](evaluation/README.md): interpretation of results, controls and trial requirements.
- [Research](research/README.md): architecture studies and proposals, not compatibility promises.
- [Agent execution contract](../AGENTS.md) and [engineering specification](../TELEPATHY_AGENT_ENGINEERING.md).
- [Current engineering record](agent-progress.md): the latest ticket and evidence pointers.

[Historical records](history/README.md) preserve earlier results and decisions;
their old commands and status labels are not current operating instructions.
Private notebooks, host inventories, raw activations and scoring evidence remain
outside this repository.
