# Scoped live-session cancellation

Engineering verdict: PASSED for the synthetic HTTP/child-process cancellation
gate; actual vLLM request termination and device-memory release remain BLOCKED
pending a bounded real-server qualification.

The Spark runner accepts `--control-stdin`. In this mode a small supervisor
owns the original runner as its only child and keeps controller traffic separate
from model output. A typed `{"op":"abort"}` line, controller EOF, malformed or
oversized control, SIGTERM or SIGHUP terminates that child and reaps it, escalating
to kill after a bounded wait. The operating system then closes the child's HTTP
socket. The supervisor withholds the successful completion event until the child
exits successfully and the controller remains connected.

The coordinator copies `live_session_control.py` beside the Spark runner,
enables supervised mode and owns its stdin pipe. On failure it sends abort and
closes the pipe, immediately interrupts other owned processes, allows bounded
supervisor cleanup and only then escalates the local SSH process. The original
unsupervised CLI behavior remains available when the flag is absent, so existing
callers with stdin at EOF do not start cancelling themselves.

Tests use a loopback HTTP streaming server and synthetic child processes. They
observe peer-visible disconnection, absence of a successful done event after
cancellation, and reaping of a child that ignores TERM. Coordinator regressions
verify helper deployment arguments, supervised launch and stdin ownership in
both linked and no-link conditions. No real model or private activation is used.

Read-only inspection of the installed vLLM source found the cancellation route
from a closed/cancelled generation stream through `AsyncLLM.abort` into engine
request abortion. Its explicit abort endpoint is mounted only in development
mode, and global pause affects unrelated requests, so neither is used here.
Socket closure in a synthetic test does not prove this entire chain works in
the live server, nor that all tensor-parallel ranks released resources.

Before calling live cancellation qualified, run one explicitly bounded owned
request, cancel while decoding, verify the request is absent on every relevant
worker and memory returns to its declared baseline, and verify an unrelated
request was unaffected. Do not restart serving processes or enable development
routers as a substitute for that evidence.
