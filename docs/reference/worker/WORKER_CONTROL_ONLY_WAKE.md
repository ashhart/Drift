# Control-only worker wake

The stock Duo controller can wake a completed worker with only developer messages, including peer-authored Hub updates. These are declared text communication in the experimental no-link/text-Duo arm, not activation-only evidence. The provider sends a bounded own-control snapshot before streaming; it does not replay or invent a user message.

Four regression cases failed before this change: WorkerSession rejected the wake, GLM omitted the pending control from reconstructed input, Studio generated without appending the native control suffix, and Studio consequently failed to exercise its native-prefix rejection. The core now admits one stream for a pending accepted control, clears readiness on consumption, and continues to reject an unprompted repeated stream. Existing tool-result, turn, token, byte, and absolute-deadline bounds remain.

GLM appends the pending control to its canonical private transcript before token counting and foreign-memory preparation; every reconstructed prefill remains charged. Studio appends through the existing checkpoint codec and exact-prefix guard, without cache reset, and charges the suffix. Native checkpoint templates may reject mid-conversation system messages; that remains CAPABILITY/BLOCKED and is not established by synthetic tokenizer tests.

Baseline: 525 passed, 3 skipped in 34.17 seconds. Four new tests initially failed; focused worker/backend tests then passed 36/36 in 0.12 seconds; full reference suite passed 529 with 3 skips in 33.72 seconds. Commands used the existing primary .venv Python with PYTHONPATH=. in this isolated worktree. No model inference or hardware energy measurement occurred, and skipped architecture gates remain blocked.
