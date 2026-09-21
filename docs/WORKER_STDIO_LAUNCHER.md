# Owned private-stdio launcher

`python -m drift.serving.worker_stdio_launcher --evidence /private/new-receipt.json --wall-seconds 60 -- command ...` creates separate supervisor control and evidence descriptors on the model host, relays private stdin/stdout through the existing supervisor, and retains only its aggregate termination receipt.

The launcher alone holds the control writer, so its death closes that channel; ordinary SSH input EOF also stops the worker through the supervisor's input relay. The supervisor has its own process session, owns the model child's process group, and applies the existing TERM/KILL/reap policy. A failed or late cleanup remains a failure; process reaping does not prove device allocator release or survival of supervisor SIGKILL or host failure.

Cold model load and any startup prefill are inside the selected wall limit, which cannot exceed 60 seconds. At the default budget the supervisor gets 59.5 seconds, including its existing 6.25-second cleanup reserve, leaving about 53.25 seconds for startup and protocol work. The outer OMP echo runner's 70-second subprocess wait is a cleanup allowance, not extra model-generation time. Studio's four-token startup prefill must be reported separately from OMP's turn usage; it generates no startup tokens.

The evidence path is exclusively created with mode 0600 before any child starts; reuse fails without launching. The caller must place it in its private audit directory, pin the command and manifests, leave Studio's guard unchanged, and omit activation configuration for a no-link run. Tests exercise actual child-process private-byte relay, EOF cleanup, ignored-TERM deadline escalation, launcher death, and refusal to overwrite existing evidence; they make no model or linked-collaboration claim.
