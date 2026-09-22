# GLM snapshot owner beside native own input

The opt-in child command is `python -m drift.serving.glm_owner_worker --config FILE --config-sha256 SHA256`. Run it under the existing `worker_stdio_launcher` with a fresh private termination receipt and its independently bounded wall clock. The child is the real `glm_restore_factory` session, bound through `bind_snapshot_bank`; there is no fixture fallback or alternate backend CLI. Its ordinary stdin/stdout remains the existing WorkerSession own-input protocol. The only new endpoint is a mode 0600, one-client Unix socket under a mode 0700 owner-only root outside Git repositories. Use a short root because Unix socket path lengths are limited.

The whole configuration is pinned before decoding and includes the existing `memory_mode: linked_snapshot`, manifests, restoration snapshot, SSH route and native budgets. Add these two fields:

```json
{
  "owner_control": {
    "root": "/private/owner-run",
    "socket_name": "owner.sock",
    "evidence_name": "bank.json"
  },
  "owner_bank": {
    "root": "/private/owner-run/incoming",
    "session": "fixed-omp-session",
    "recipe_path": "/private/pins/reverse-v3-manifest.json",
    "recipe_sha256": "<original reverse-v3 manifest digest>",
    "max_rows": 12,
    "max_bytes": 1048576,
    "max_versions": 2,
    "max_total_bytes": 2097152,
    "initial": {
      "v": 1,
      "session": "fixed-omp-session",
      "version": 1,
      "path": "initial.npz",
      "sha256": "<initial snapshot digest>",
      "rows": 12,
      "source_worker": "qwen",
      "target_worker": "glm",
      "recipe_sha256": "<original reverse-v3 manifest digest>"
    }
  }
}
```

Directories must already exist with owner-only permissions. Initial path, rows and digest must match the existing restoration configuration, and its target must be the worker's exact ID. Layer layouts and ownership labels derive from restoration. The original reverse recipe is separately hashed; it is not equated with a wrapper translator-manifest digest or the forward export recipe. The bank's fixed session also pins the ordinary own-protocol session. Native input/output/session budgets remain enforced by the existing WorkerSession and backend.

After startup, keep one owner connection open until own close. Send exactly `{"op":"publish","snapshot":FRAME}` followed by a newline, where FRAME has the same fields as initial, the next consecutive version, and a new staged leaf filename under the bank root. The reply is `{"op":"published", ...SnapshotBank.receipt()}` with no raw path, arrays or text. A Qwen source session is separate provenance; never substitute it for this GLM bank session. No arbitrary status or file-read operations exist. Append/tap are not accepted here, and Qwen's existing append/tap contract is unchanged.

Publication creates a validated immutable copy immediately. The current GLM turn keeps the version captured at its preparation boundary; only a subsequent turn can select the newer version. A publication receipt therefore proves storage admission, not use by a running request. The owner must park both actual OMP workers at the declared tool boundary, publish the next version, and then release the tool result for the intended second turn.

Malformed input, owner EOF or abort poison the bank, wake the ordinary input loop, and invoke existing owned backend cancellation/cleanup. The private aggregate bank receipt records selected versions for completed turns and the existing outbox receipts, including their manifest hashes when export is enabled. The separate supervisor receipt proves local child reaping; neither replaces the other's evidence. A missing receipt, cleanup failure, late deadline or incomplete native turn is not a success. The child deadline is capped at 60 seconds, while the external supervisor independently reserves its cleanup interval; choose both budgets before dispatch.

Synthetic subprocess regressions use the real WorkerSession, bank, turn binding, socket framing and supervisor with a fake native transport. They prove first-turn v1 and second-turn v2 selection despite a mid-turn update, and fail-closed cleanup; they establish no language transfer, TP atomicity, causal influence, GPU allocation release, or same-UID filesystem sandbox. Other processes with the owner's UID remain outside the Unix permission boundary, so the declared OMP/tool isolation is still required.
