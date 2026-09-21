"""Marker training on dense toys; the peer CLI on two loopback processes."""
from __future__ import annotations
import json
import os
import subprocess
import sys
from pathlib import Path
import torch
from drift.adapters.toy import DenseAdapter
from drift.core.attention import Gate
from drift.mailbox.stream import MailWriter
from drift.runtime.decoder import FrozenDecoder
from drift.runtime.toy import ToyModel
from drift.train.mailbox_train import MailExample, train_marker
from drift.translate.pool import Layout, PoolFormat, Translator

FMT = PoolFormat("pool.v1", 3, 8, "ee" * 32)


def test_marker_training_raises_mail_mass_and_moves_only_sidecars():
    torch.manual_seed(0)
    sender = DenseAdapter(FrozenDecoder(ToyModel(seed=1, kvheads=2, dim=6, layers=3)))
    receiver = DenseAdapter(FrozenDecoder(ToyModel(seed=2, kvheads=1, dim=4, layers=3)))
    writer = Translator("s", Layout("kv_split", 2, 6), {0: 0, 1: 1, 2: 2}, FMT)
    reader = Translator("r", Layout("kv_split", 1, 4), {0: 0, 1: 1, 2: 2}, FMT)
    with torch.no_grad():
        for p in list(writer.parameters()) + list(reader.parameters()):
            p.mul_(0.2)
    mail = MailWriter(sender.decoder.model.config.hidden_size, max_tokens=4)
    gates = {i: Gate(-3.0) for i in range(3)}
    g = torch.Generator().manual_seed(2)
    examples = [MailExample(torch.randint(1, 40, (4,), generator=g), torch.randint(1, 40, (2,), generator=g),
                            torch.randint(1, 40, (2,), generator=g), torch.randint(1, 40, (4,), generator=g),
                            torch.randint(1, 40, (2,), generator=g)) for _ in range(5)]
    marker_before = mail.marker.detach().clone()
    ledger = train_marker(sender, receiver, mail, writer, reader, gates, examples, steps=30, learning_rate=5e-2)
    assert ledger.steps == 30 and ledger.sender_mail_tokens == 60
    first = sum(l["mass"] for l in ledger.losses[:5]) / 5
    last = sum(l["mass"] for l in ledger.losses[-5:]) / 5
    assert last > first                               # the mail block gets noticed more
    assert not torch.equal(mail.marker.detach(), marker_before)
    assert all(p.grad is None for p in sender.decoder.model.parameters())
    assert all(p.grad is None for p in receiver.decoder.model.parameters())


def test_peer_cli_couples_two_processes_over_loopback(tmp_path):
    manifest = {"kind": "toy", "session_uuid": "00000000-0000-0000-0000-00000000004d", "setup_tokenizer": "bytes",
                "pool": {"levels": 3, "width": 8, "fingerprint": "ab" * 32}, "window": {"sinks": 2, "recent": 6},
                "setup_text": {"A": "task one", "B": "task two"},
                "members": [{"name": "A", "writer": 1, "mail_writer": 11, "seed": 1, "toy": {"seed": 1, "kvheads": 2, "dim": 6, "layers": 3}},
                            {"name": "B", "writer": 2, "mail_writer": 12, "seed": 2, "toy": {"seed": 2, "kvheads": 1, "dim": 4, "layers": 3}}]}
    path = tmp_path / "run.json"
    path.write_text(json.dumps(manifest))
    import socket
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    env = {**os.environ, "PYTHONPATH": ".", "DRIFT_LINK_SECRET": "x" * 40}
    common = [sys.executable, "scripts/peer_serve.py", "--manifest", str(path), "--port", str(port), "--epochs", "3", "--timeout", "60"]
    server = subprocess.Popen(common + ["--member", "A", "--role", "server", "--audit-root", str(tmp_path / "A")], env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    import time
    time.sleep(2.5)
    client = subprocess.run(common + ["--member", "B", "--role", "client", "--audit-root", str(tmp_path / "B")], env=env, capture_output=True, text=True, timeout=120)
    out, err = server.communicate(timeout=120)
    assert server.returncode == 0, err
    assert client.returncode == 0, client.stderr
    a = json.loads((tmp_path / "A" / "peer-A.json").read_text())
    b = json.loads((tmp_path / "B" / "peer-B.json").read_text())
    assert a["epochs"] == 3 == b["epochs"] and not a["poisoned"] and not b["poisoned"]
    assert a["wire_bytes_sent"] == b["wire_bytes_received"] and a["wire_bytes_received"] == b["wire_bytes_sent"]
    assert a["manifest_sha256"] == b["manifest_sha256"] and len(a["generated_ids"]) == 3
