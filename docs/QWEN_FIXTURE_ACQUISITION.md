# Controlled Qwen source fixture

Prepared implementation only: no Studio acquisition is recorded here. This driver
runs one fixed public note, `API: GET /health returns HTTP 200.`, as at most 32 raw
local tokenizer tokens with no special tokens and no generated tokens. It reuses
unchanged `studio_tap_qwen.py`, including its guard's 140 GiB estimated need, 48 GiB
reclaimable reserve, 148 GiB MLX limit and 4 GiB cache limit. These are configuration
bounds, not allocator guarantees or measured model memory.

Stage the following source package under a private directory outside repositories,
using the Studio's existing oMLX interpreter and Python paths: `scripts/qualify_live/fixture.py`,
`fixture_child.py`, `scripts/live/studio_tap_qwen.py`, and the imported package modules
`fixture_host`, `fixture_supervision`, `fixture_receipt`, `qwen_fixture`, `live_publication`,
`omlx_cache`, `studio_guard`, `drift.translate.stacked`, plus normal package initializers.
Include the shared `worker_supervisor` and `worker_supervisor_io` from commit
`98fbe13d0c72e14c64985f4c2c1d25cd136488ff`; this ticket does not duplicate their process logic.
Stage and verify the frozen reverse translator SHA-256
`36c352797d936aca6468d4066c1be93fea99ab0185f7505e731874dc5e7be8b9`.
Freeze the staged source hashes and invocation before dispatch.

```sh
PYTHONPATH="$OMLX_SITE:$OMLX_RESOURCES:." "$OMLX_PYTHON" scripts/qualify_live/fixture.py \
  --private-root "$PRIVATE_FIXTURES" \
  --checkpoint "$EXISTING_QWEN_CHECKPOINT" \
  --translator "$PRIVATE_REVERSE_V3" \
  --tap-script scripts/live/studio_tap_qwen.py
```

The variables denote the already inspected Studio environment and private paths;
this command does not install or change software. The private root must have mode
0700 and no repository ancestor. Every invocation creates a new one-use claim,
with raw IDs, source taps and output only under its `raw` directory. Nothing is
published, and generated/model text is neither returned nor retained.

The parent owns only the control pipe's writer. The separate on-host supervisor
owns the child process group and closes it on parent loss, abort or deadline,
reporting group termination/reaping through a separate evidence pipe. The complete
acquisition has a 300-second outer budget, with at most 280 seconds allocated to
supervision; the shared supervisor reserves its cleanup time inside that allocation.
A missing or late termination receipt cannot pass. Operating-system process launch,
termination and file I/O remain dependencies; incomplete evidence is reported INVALID.

The child tokenizes and fingerprints the public source, tokenizer/config and tap
script, invokes the existing tap script once, waits for the model process to exit,
selects the final canonical row, then applies reverse-v3 at gain one and repeats that
translated row twelve times. Output is eleven GLM layers with twelve rows of 512
values each. Full weight identity is explicitly NOT_MEASURED; config/tokenizer/index
hashes are a minimal fingerprint and do not replace an existing checkpoint manifest.

Only aggregate source/output/translator hashes, shapes, token counts, timings,
supervisor evidence and pre/post lock/OS-memory metadata are printed and written to
`receipt.json`. Raw files stay private. Post-exit lock availability and OS reclaimable
memory do not prove CUDA/Metal allocator release. Preserve a failed claim and never
retry inference automatically. A successful fixture can supply the single-publication
GLM qualification, but does not establish provenance, causal collaboration or coding benefit.

Local tests use synthetic arrays and translator doubles, plus the actual shared
supervisor running harmless local Python children; they never load Qwen or GLM.
