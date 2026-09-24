# MCDMA recovery after a broken pull

Status: deployed 23 September 2026 on the Studio and both Sparks; tested on the hardware. Engineering evidence, not a
research claim.

## What failed

On the evening of 23 September the drop-in's first pull failed with `ERR <peer> control` for the head Spark after the link had sat idle for
about ten hours. The Studio's `handoffd` then reported the head Spark down, and every reconnect failed. The network was
fine: both control ports answered and ping was clean. Both Spark daemons were asleep inside sessions whose connections
had died without closing. A Spark daemon serves one control connection at a time and blocks on it with no timeout or
keepalive, so it never accepted again; the Studio's reconnects waited in its four-slot accept backlog, and the Studio's
own handshake had no timeout either. Recovery took restarting all three daemons by hand.

## The change

A patch against the deployed `handoffd.c`, one source for the Studio (Apple `librdma` and the MCDMA provider) and the
Sparks (`libibverbs`). The daemon's source is not public, so the patch stays with it, in the development repository as
`scripts/handoffd/handoffd-recovery.patch`; its SHA-256 is recorded below. The change:

- A Spark watches its listening socket during a session. A new connection whose first line is a `HELLO` replaces the
  session, since one Studio talks to each Spark; a connection that sends anything else is closed and the session goes
  on. Control connections use TCP keepalive: 30 s idle, then 3 probes 10 s apart.
- The Studio bounds its connect and handshake at 5 s and every later control reply at 15 s, and uses the same
  keepalive. A pull whose `SERVE` exchange fails before any data moves reconnects and asks once more.
- `TEST_ORPHAN name` makes the Studio forget a peer's control connection without closing it, as a connection that died
  silently leaves the Spark's end. It exists for this test.

Nothing changes in the RDMA data path or the command protocol otherwise. The build warnings are the same ten
format-truncation warnings the original gives.

## Test

`scripts/live/handoffd_recovery_test.py` writes a random 64 MiB blob into each Spark's handoff root, records its
SHA-256, and drives the Studio's daemon through `drift/serving/handoffd_client.py`, hashing what each pull wrote into
shared memory. It breaks pulls two ways and restarts no daemon:

- orphan: the Studio forgets its control connection without closing it, then pulls again;
- stall: the Spark's daemon is paused with SIGSTOP before a pull, so that pull must fail within the Studio's bounds,
  then resumed with SIGCONT and pulled from again.

Before, with the new Studio daemon and the second Spark still on the old one: the baseline pull matched its SHA-256 at
48.5 Gbit/s; after an orphaned connection the next pull failed (`connect`, bounded at 5.0 s by the new Studio code,
where the old one waited forever), and the old Spark daemon stayed in the dead session until it was restarted.

After, all three daemons built from the patched source:

| Spark | Step | Result | Seconds | SHA-256 matches |
| --- | --- | --- | --- | --- |
| head | baseline | OK, 46.5 Gbit/s | 0.18 | yes |
| head | orphan, then pull (3 times) | OK, 46.0 to 47.8 Gbit/s | 0.18 to 0.19 | yes, all 3 |
| head | pull while its daemon is paused | ERR connect, bounded | 20.0 | no data |
| head | pull after it resumes | OK, 47.8 Gbit/s | 0.17 | yes |
| second | baseline | OK, 46.7 Gbit/s | 0.18 | yes |
| second | orphan, then pull (3 times) | OK, 43.4 to 47.7 Gbit/s | 0.18 | yes, all 3 |
| second | pull while its daemon is paused | ERR connect, bounded | 20.0 | no data |
| second | pull after it resumes | OK, 45.5 Gbit/s | 0.18 | yes |

After a deliberately broken pull, the next pull lands byte for byte with no daemon restarted. The paused case fails in
20.0 s: 15 s for the unanswered `SERVE`, then 5 s for the reconnect's unanswered handshake.

Two limits remain. A Spark that is paused for longer than the Studio waits still costs that pull, and recovery relies on
one Studio per Spark: a second Studio's `HELLO` would replace the first one's session.

SHA-256: original `handoffd.c` `6ff1c8380eb3a62a88f22839f30e8fc0a2d5df3a08575ef26a6bba006cbca503`; patched source
`28650b1181b2b1afcb74de8090ecaf807eb9700c804b37f2285a5cafd6d702f4`, reproduced by applying the patch
`91e780ef3623b03f4099dd4a0c1a742e3859b8632c9bf04f90f3fac133b6441a`; Studio binary
`d38f421621aafbd03a1e40049f10efabfeda49d725c9117971c2413ea09c7618`; both Spark binaries
`f7e8cd3ba977a103e4774e6ef86bfb7744fb7211b3fe8d487462fcc4a5d3a9b5`; reports, private local copies: before
`b01d45c5f3d753a590b05a115416cc4194a5507e71e721334c986fbd0e28948a`, after
`c05d16e2161fbb6fe651fbcdecc6c6ff4da605bb2361261f71239c0f9f7c734c`.

## The Studio daemon needs a live session

On the morning of 24 September, after about ten idle hours, every pull from the head Spark failed at once with
`ERR <peer> connect`, and the Studio reported that Spark down. Nothing reached the Spark: its log gained no line per
attempt. The TCP port answered from a new shell on the Studio, and the Spark's daemon was idle and listening.

The cause was the session, not the daemon or the link. The Studio daemon had been started from an SSH session, and that
session ended when the agent session driving it restarted. macOS applies local network privacy to a process once its
session is gone: in a background process whose SSH session had ended, Apple's `/usr/bin/nc` still connected to the
Spark's port, while a third-party binary (Python) got `EHOSTUNREACH` at once. The daemon's connects failed the same way
and, before this change, said nothing. The pull the daemon already held open did not need a new connection, which is why
the second Spark stayed up until its own connection closed.

The daemon has to run inside a live session, as its "session-attached start" log line intends, or the owner has to grant
it local network access. Build 3 of the patched source logs why a connect fails (`<peer>: connect: <error>`, and the
resolver and socket failures before it), so this shows up by name next time. The Studio daemon was stopped with
`SHUTDOWN` on its socket and restarted from build 3 inside a kept-alive SSH session; the Sparks keep their earlier build,
since the change touches only the Studio's side of a connection.

The recovery test on build 3, same steps:

| Spark | Step | Result | Seconds | SHA-256 matches |
| --- | --- | --- | --- | --- |
| head | baseline | OK, 48.2 Gbit/s | 0.05 | yes |
| head | orphan, then pull (3 times) | OK, 47.9 to 48.9 Gbit/s | 0.18 | yes, all 3 |
| head | pull while its daemon is paused | ERR connect, bounded | 20.0 | no data |
| head | pull after it resumes | OK, 47.3 Gbit/s | 0.17 | yes |
| second | baseline | OK, 40.4 Gbit/s | 0.06 | yes |
| second | orphan, then pull (3 times) | OK, 41.2 to 47.3 Gbit/s | 0.18 to 0.19 | yes, all 3 |
| second | pull while its daemon is paused | ERR connect, bounded | 20.0 | no data |
| second | pull after it resumes | OK, 45.2 Gbit/s | 0.17 | yes |

SHA-256: build 3 source `6c436670c860768393dc7cc957739fb614ea931854cba15bb7b03b43d2266ecc`, reproduced by applying the
updated patch `d2b7ca57914079d4c249906a81bf77d557040d6d84b1da3cf61a186d6e7db455`; Studio binary
`397cf6fd1e2892acac8f91dd4a3d5442b63295b5f68c19b3cee5da7d63584e1f`; report, private local copy,
`7ce1fd34b1be3279edd930db3906daf3383fa75bfb0a6bcb845fd5f569ea3f76`.

