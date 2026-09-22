# Bounded metrics fetch

`fetch_metrics(url, timeout=remaining)` fetches only an explicit IPv4 loopback
HTTP `/metrics` endpoint and returns at most 2 MiB of UTF-8 text to the internal
parser. Redirects and environment proxies are disabled. The caller must not log
the returned exposition; exceptions contain fixed messages only.

One owned Python subprocess performs the HTTP request without inherited environment
variables, stdin, stderr, user site packages or subprocess descendants. The parent
uses one absolute deadline across request completion and cleanup, reserving up to
0.2 seconds inside that deadline to kill and reap the process. A continuously
trickling response cannot extend the socket timeout indefinitely. Process launch
and termination still depend on the operating system: failure to confirm reaping
produces `MetricFetchError` instead of a successful bounded-cleanup claim.

The regression reproduces the old socket-only timeout accepting a response after
its deadline, then tests the bounded implementation against real loopback trickling
body and stalled-header servers. These tests exercise HTTP/process mechanics only;
they do not contact a model, establish server cancellation or measure GPU memory.
