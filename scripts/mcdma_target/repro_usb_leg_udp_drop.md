# Minimal reproduction: the USB-C leg drops datagrams of certain sizes, Spark to Studio

No MCDMA code is involved. Measured 2026-09-21 on the spark-a.invalid leg (192.0.2.1 to 192.0.2.40); the MCDMA symptom is identical on
the spark-b.invalid leg.

Listener on the Studio (12 seconds, then prints what never arrived):

```bash
python3 -c 'import socket,time
s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM); s.bind(("192.0.2.40",18793)); s.settimeout(1.0)
got,end=set(),time.time()+12
while time.time()<end:
    try: got.add(len(s.recvfrom(65535)[0]))
    except socket.timeout: pass
print(sorted(set(range(660,760))-got))'
```

Sender on spark-a.invalid, within those 12 seconds (every size three times, 2 ms apart):

```bash
python3 -c 'import socket,time
s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM); s.bind(("192.0.2.1",0))
for r in range(3):
    for n in range(660,760): s.sendto(bytes(n),("192.0.2.40",18793)); time.sleep(0.002)'
```

Result: UDP payloads of 700, 701, 702 and 703 bytes never arrive; every other size from 660 to 759 does. The same holds
1,024 bytes higher (1,724 tried) and 2,048 higher (2,747 and 2,748 tried). In frame terms: Ethernet frames of 742 to 745
bytes, plus any multiple of 1,024, vanish in the Spark-to-Studio direction.

What it does to MCDMA: a one-sided READ of n bytes is answered by one datagram of n + 24 bytes, so reads with
n mod 1024 in 676 to 679 (675 intermittently) never complete and the client times out after 2 s
(`scripts/mcdma_target/repro_read_size_fault.py`: 24 of 70 sizes fail over the leg, 0 of 70 when the client runs on the
Spark itself). Writes of the same sizes succeed, and the write acknowledgement is a fixed small datagram. 1.2 million reads
of other sizes had no loss. Not determined: which side drops the frame (the Spark's userspace NCM driver on transmit, or the
Mac's NCM function on receive) and whether the Studio-to-Spark direction has a band of its own at other sizes.

## Full map, both directions (every UDP payload size from 1 to 8,984 bytes, sent twice, spark-a.invalid leg)

| Direction | Sizes that never arrived |
| --- | --- |
| Spark to Studio | 36 of 8,984: exactly 700 to 703 plus k x 1,024 for every k from 0 to 8 (700-703, 1724-1727, 2748-2751, ... 8892-8895) |
| Studio to Spark | none |

So the loss is one-directional, exactly periodic in 1,024 bytes and four sizes wide. Hypothesis only, not tested: the
transfer that carries such a frame comes out as an exact multiple of the 1,024-byte USB bulk packet size and is sent
without a terminating zero-length packet, so the receiver never completes it; four adjacent frame sizes share one padded
transfer length. Drift's mailbox avoids the band on reads (`mcdma_mailbox._safe_read`); its writes travel in the direction
that loses nothing.
