# Historical records

These documents preserve prior decisions, results and failed attempts; their
status labels and commands are not current instructions. Use
[current status](../STATUS.md) and the [native guide](../guides/NATIVE_OWNER_EXCHANGE.md)
for the supported path.

| Record | Why retained |
| --- | --- |
| [Engineering log through 22 September](engineering-2026-09-22.md) | Exact commits, commands, costs, failed attempts and evidence hashes |
| [Build snapshot](BUILD_STATUS.md) | Earlier component-level results and limits |
| [Release assessment](RELEASE_READINESS.md) | Earlier workflow gaps and formal evaluation outcomes |
| [D0-D7 roadmap](ROADMAP-2026-09-20.md) | Original product/research proposal |
| [SSH baseline instructions](LIVE-SSH.md) | Reproduction context for old experiments, not the MCDMA launch path |
| [Original engineering plan](original_plan.md) | Historical input referenced by the binding engineering handoff |

The former `docs/FRONTIER.md` was a 16-line placeholder, not the original
experiment notebook. Its valid warnings now live in STATUS; the original
private notebooks and host records remain outside this repository.
Historical references to its numbered sections refer to that earlier notebook,
not to a missing public benchmark dataset.

Frozen preregistration files retain their original references to avoid changing
their identity. Existing Git history also retains former paths; this cleanup
does not rewrite published history or alter historical experiment verdicts.

The pinned GLM connector also retains its original source bytes, including an
old comment referring to `docs/GLM_PREFIX_REUSE.md`; that reference now maps to
[prefix reuse](../reference/glm/GLM_PREFIX_REUSE.md). Editing that comment would
invalidate the deployed bundle's hash without changing its behavior.
The engineering handoff's embedded source appendix is likewise unchanged and
describes its original extracted file layout, not the current documentation tree.
