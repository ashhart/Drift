"""Two-process lockstep over loopback TCP, and the level-2 parity runner on a tiny checkpoint."""
from __future__ import annotations
import json
import multiprocessing as mp
import subprocess
import sys
from pathlib import Path
from uuid import UUID
import pytest
import torch
from drift.adapters.toy import DenseAdapter
from drift.core.pool import PoolBank
from drift.runtime.decoder import FrozenDecoder
from drift.runtime.hive import HiveController
from drift.runtime.peer import PeerLink, PeerRunner, connect_pair
from drift.runtime.toy import ToyModel
from drift.runtime.worker import Worker
from drift.translate.pool import Layout, PoolFormat, Translator
from drift.transport.wire2 import FrameCodec2

SESSION = UUID(int=31)
FMT = PoolFormat("pool.v1", 3, 8, "bb" * 32)
SHAPES = {"A": dict(seed=1, kvheads=2, dim=6, layers=3), "B": dict(seed=2, kvheads=1, dim=4, layers=3)}
IDS = {"A": 1, "B": 2}


def member(name: str) -> Worker:
    torch.manual_seed(IDS[name])
    adapter = DenseAdapter(FrozenDecoder(ToyModel(**SHAPES[name])))
    d = adapter.descriptor
    t = Translator(name, Layout("kv_split", d.kv_heads, d.head_dim), {l: l for l in d.kv_layers}, FMT)
    with torch.no_grad():
        for p in t.parameters():
            p.mul_(0.3)
    others = {v for k, v in IDS.items() if k != name}
    return Worker(name, IDS[name], SESSION, adapter, t, PoolBank(SESSION, t, IDS[name], others, sinks=2, recent=6), override=0.5)


def inputs(seed: int, epochs: int) -> list[dict[str, torch.Tensor]]:
    g = torch.Generator().manual_seed(seed)
    return [{n: torch.randint(1, 30, (2,), generator=g) for n in ("A", "B")} for _ in range(epochs)]


def _peer_process(name: str, port: int, role: str, schedule, queue) -> None:
    import socket
    torch.set_num_threads(1)
    if role == "server":
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", port))
        listener.listen(1)
        queue.put(("port", listener.getsockname()[1]))
        sock, _ = listener.accept()
        listener.close()
    else:
        sock = socket.create_connection(("127.0.0.1", port), timeout=30)
    runner = PeerRunner(member(name), PeerLink(sock, FrameCodec2(b"p" * 32)))
    outputs = []
    with torch.no_grad():
        for step in schedule:
            outputs.append(runner.tick(step[name]).output.tolist())
    queue.put(("done", name, outputs, runner.wire_bytes_sent, runner.wire_bytes_received))
    runner.link.close()


def test_two_processes_lockstep_over_tcp_match_the_in_process_hive():
    schedule = inputs(9, 4)
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    server = ctx.Process(target=_peer_process, args=("A", 0, "server", schedule, queue))
    server.start()
    kind, port = queue.get(timeout=60)
    assert kind == "port"
    client = ctx.Process(target=_peer_process, args=("B", port, "client", schedule, queue))
    client.start()
    results = {}
    for _ in range(2):
        item = queue.get(timeout=120)
        results[item[1]] = item
    server.join(30)
    client.join(30)
    assert server.exitcode == 0 and client.exitcode == 0
    # Same schedule in-process must give the same outputs: the wire changes nothing.
    controller = HiveController({"A": member("A"), "B": member("B")}, FrameCodec2(b"p" * 32))
    with torch.no_grad():
        expected = [controller.tick(step) for step in schedule]
    for name in ("A", "B"):
        _, _, outputs, sent, received = results[name]
        assert sent > 0 and received > 0
        for step_out, report in zip(outputs, expected):
            torch.testing.assert_close(torch.tensor(step_out), report[name].output, rtol=1e-5, atol=1e-5)


def test_peer_runner_poisons_on_disconnect_and_bad_epoch():
    server, client = connect_pair()
    a = PeerRunner(member("A"), PeerLink(server, FrameCodec2(b"p" * 32), timeout=5))
    client.close()
    with pytest.raises((EOFError, OSError)):
        a.tick(torch.tensor([1, 2]))
    assert a.failed and a.worker.poisoned
    with pytest.raises(RuntimeError, match="poisoned"):
        a.tick(torch.tensor([1, 2]))


@pytest.mark.skipif(not Path(".venv-next/bin/python").exists(), reason="needs the transformers 5.17 environment")
def test_level2_parity_runner_on_a_tiny_saved_checkpoint(tmp_path):
    """The level-2 runner refuses without a preregistered tolerance, then passes on tiny weights."""
    code = """
import sys, torch
sys.path.insert(0, "tests")
from test_adapters_next import make_qwen
stock, _, _ = make_qwen()
stock.save_pretrained(sys.argv[1])
"""
    ckpt = tmp_path / "ckpt"
    subprocess.run([".venv-next/bin/python", "-c", code, str(ckpt)], check=True, capture_output=True, env={"PYTHONPATH": "."})
    prereg = tmp_path / "prereg.json"
    prereg.write_text(json.dumps({"parity": {"level2_real_weight_tolerance": "UNRESOLVED"}}))
    cmd = [".venv-next/bin/python", "scripts/parity_real.py", "--family", "qwen4_exp", "--checkpoint", str(ckpt),
           "--prereg", str(prereg), "--registry", str(tmp_path / "registry"), "--model-id", "qwen-tiny",
           "--lengths", "1,7", "--seeds", "0"]
    blocked = subprocess.run(cmd, capture_output=True, text=True, env={"PYTHONPATH": "."})
    assert blocked.returncode == 2 and "BLOCKED" in blocked.stderr
    prereg.write_text(json.dumps({"parity": {"level2_real_weight_tolerance": 2e-5}}))
    run = subprocess.run(cmd, capture_output=True, text=True, env={"PYTHONPATH": "."})
    assert run.returncode == 0, run.stdout + run.stderr
    report = json.loads((tmp_path / "registry" / "qwen-tiny" / "qualification" / "level2.json").read_text())
    assert report["status"] == "PASSED" and report["worst_max_abs"] <= 2e-5 and report["weights_unchanged"]
    assert all(r["hard_off_bit_identical"] for r in report["records"]) and len(report["records"]) == 2
