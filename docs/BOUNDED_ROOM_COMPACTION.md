# Bounded worker history and automatic compaction

The third native room failed after OMP compacted its history between provider calls. Its safe event metadata reported an estimated 17,303-token context against the GLM provider's advertised 16,384-token window; the unchanged cumulative worker allowance was 65,536. The history-preserving provider then rejected the changed transcript with CONTEXT. Native model output was not inspected to diagnose this.

The public deterministic fixture reproduces the failure using the actual installed OMP and unchanged stock Duo. Its parent reports 9,214 then 17,303 input tokens; its child reports 7,001 initial input tokens, with advertised windows 16,384 and 8,192 and cumulative allowances 65,536 and 20,000. Automatic compaction enabled yields one compaction, CONTEXT and UNKNOWN, zero guard blocks and no completed child. Disabling compaction yields both completed workers and the public API verification pass, with the same usage, limits and tools. The fixture's usage is synthetic and is not a model cost measurement.

New bounded room preparations write `compaction.enabled: false` to their isolated agent `config.yml` before hashing the launch bundle. The API development fixture inherits the same pinned file. It prevents automatic transcript replacement while these workers retain their exact history; it does not truncate history, widen advertised context, raise a worker budget or change global OMP settings. Native worker and supervisor limits still terminate an over-budget run. This setting belongs to both auxiliary development arms; the unweakened ordinary-Duo primary benchmark is separate and remains unrun.

Reproduction command, adding `--enable-compaction` for the expected failing control:

```sh
.venv/bin/python scripts/omp/compaction_probe.py \
  --omp /home/example/.local/bin/omp \
  --source /opt/drift \
  --duo /home/example/Documents/Codex/2026-08-25/i-added-omp-harness-and-want/omp-local-duo \
  --python /Library/Frameworks/Python.framework/Versions/3.11/bin/python3
```

The preparation regression requires the exact file to be included in the frozen source pins and rejects tampering. Actual OMP synthetic red/green and focused tests passed; native confirmation requires a fresh evidence directory and unused worker stages. Full-check counts and hashes are in the progress record. No backbone parameters, remote serving processes or frozen experiment data changed.
