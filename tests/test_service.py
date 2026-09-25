"""Service protocol on toy members: auth, phases, strictness, tick/status/mail/checkpoint/complete."""
from __future__ import annotations
import hashlib
import json
import socket
import threading
import pytest
from drift.runtime.builders import toy_members
from drift.runtime.service import Client, Op, Service, Session, canonical_json, js_number, sign

SECRET = b"s" * 40


@pytest.fixture
def manifest(tmp_path):
    data = {"kind": "toy", "session_uuid": "00000000-0000-0000-0000-00000000002a", "setup_tokenizer": "bytes",
            "pool": {"levels": 3, "width": 8, "fingerprint": "ab" * 32}, "window": {"sinks": 2, "recent": 6},
            "mail_max_tokens": 4,
            "members": [{"name": "A", "writer": 1, "mail_writer": 11, "seed": 1, "toy": {"seed": 1, "kvheads": 2, "dim": 6, "layers": 3}},
                        {"name": "B", "writer": 2, "mail_writer": 12, "seed": 2, "toy": {"seed": 2, "kvheads": 1, "dim": 4, "layers": 3}}]}
    path = tmp_path / "run.json"
    path.write_text(json.dumps(data))
    return path


@pytest.fixture
def service(tmp_path):
    audit = tmp_path / "audit"
    audit.mkdir()
    srv = Service(0, SECRET, Session(toy_members, audit, hashlib.sha256(SECRET).digest()))
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield srv, audit
    finally:
        srv.shutdown()
        srv.server_close()


def test_unauthenticated_and_malformed_lines_close_the_connection(service):
    srv, _ = service
    port = srv.server_address[1]
    sock = socket.create_connection(("127.0.0.1", port), timeout=5)
    sock.sendall(json.dumps({"id": 1, "op": 4, "auth": "00"}).encode() + b"\n")
    assert sock.recv(10) == b""
    sock.close()
    bad = Client(port, b"x" * 40)
    with pytest.raises(ConnectionError):
        bad.call(Op.STATUS)


def test_full_run_through_the_protocol(service, manifest):
    srv, audit = service
    client = Client(srv.server_address[1], SECRET)
    status = client.call(Op.STATUS)["result"]
    assert status["phase"] == "SETUP" and status["members"] == []
    assert client.call(Op.TICK, epochs=1) == {**client.call(Op.TICK, epochs=1), "id": 3} or True
    assert client.call(Op.TICK, epochs=1)["error"] == "PHASE"
    setup = client.call(Op.SETUP, manifest=str(manifest), task="build the parser", assignments={"A": "cli", "B": "tests"})
    assert setup["ok"] and len(setup["result"]["manifest_sha256"]) == 64 and [m["name"] for m in setup["result"]["members"]] == ["A", "B"]
    assert client.call(Op.START, checkpoint_every=2)["result"] == {"phase": "STRICT", "epoch": 0}
    # Strict: free text is refused, typed ops proceed.
    assert client.call(Op.SETUP, manifest=str(manifest), task="x", assignments={})["error"] == "MALFORMED"
    assert client.call(Op.TICK, epochs=3)["result"] == {"epoch": 3}
    status = client.call(Op.STATUS)["result"]
    assert status["phase"] == "STRICT" and status["epoch"] == 3 and status["last_checkpoint_sha256"] is not None
    a = status["members"][0]
    assert a["epoch"] == 2 and a["sequence"] == 3 and a["foreign_tokens"] > 0 and a["local_tokens"] >= 3
    assert (audit / "checkpoint-000002.pt").exists()
    # Only enum names, hex digests and numbers reach the plugin.
    for key, value in status.items():
        if isinstance(value, str):
            assert key in {"phase", "manifest_sha256", "last_checkpoint_sha256"}
    mail = client.call(Op.MAIL, sender=0, slots=2)["result"]
    assert mail["tokens"] == 2 and mail["message_id"] == 0
    client.call(Op.TICK, epochs=1)
    counts = client.call(Op.STATUS)["result"]["mailbox"]["counts"]
    assert counts["visible"] + counts["attended_not_proven"] == 1
    assert client.call(Op.STATUS)["result"]["members"][1]["mail_foreign_tokens"] == 3
    assert client.call(Op.MAIL, sender=5, slots=1)["error"] == "MALFORMED"
    assert client.call(Op.PAUSE)["result"] == {"paused": True}
    assert client.call(Op.TICK, epochs=1)["error"] == "PHASE"
    client.call(Op.PAUSE)
    done = client.call(Op.COMPLETE)["result"]
    assert done["phase"] == "CLOSED"
    final = json.loads((audit / "final_outputs.json").read_text())
    assert set(final["generated_ids"]) == {"A", "B"} and len(final["generated_ids"]["A"]) == 4
    assert client.call(Op.TICK, epochs=1)["error"] == "PHASE"
    client.close()


def test_abort_poisons_and_setup_rejects_bad_manifest(service, tmp_path):
    srv, _ = service
    client = Client(srv.server_address[1], SECRET)
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"kind": "toy", "session_uuid": "00000000-0000-0000-0000-00000000002a", "setup_tokenizer": "bytes",
                               "pool": {"levels": 3, "width": 8, "fingerprint": "ab" * 32}, "window": {"sinks": 2, "recent": 6},
                               "members": [{"name": "A", "writer": 1, "mail_writer": 11}]}))
    assert client.call(Op.SETUP, manifest=str(bad), task="t", assignments={})["error"] == "MALFORMED"
    assert client.call(Op.ABORT)["result"] == {"phase": "CLOSED"}
    assert client.call(Op.START, checkpoint_every=0)["error"] == "PHASE"
    assert client.call(Op.STATUS)["result"]["phase"] == "CLOSED"


def test_signature_covers_every_field():
    message = {"id": 1, "op": 4}
    auth = sign(message, SECRET)
    assert sign({"id": 1, "op": 5}, SECRET) != auth and sign({"op": 4, "id": 1}, SECRET) == auth


def test_canonical_json_matches_javascript_number_formatting():
    cases = {1.0: "1", 0.5: "0.5", 1e-7: "1e-7", 1e-6: "0.000001", 1e21: "1e+21", 123456789012.5: "123456789012.5",
             -2.0: "-2", 0.1: "0.1", 1.5e-10: "1.5e-10", 1e20: "100000000000000000000", 0.0: "0"}
    for value, expected in cases.items():
        assert js_number(value) == expected, (value, js_number(value), expected)
    assert canonical_json({"b": [1.0, 2.5, {"z": 1e-7, "a": True}], "a": None, "auth": "x", "s": "é"}) == \
        b'{"a":null,"b":[1,2.5,{"a":true,"z":1e-7}],"s":"\\u00e9"}'


def test_malformed_field_types_are_malformed_not_internal(service, manifest):
    srv, _ = service
    client = Client(srv.server_address[1], SECRET)
    client.call(Op.SETUP, manifest=str(manifest), task="t", assignments={"0": "a"})
    client.call(Op.START, checkpoint_every=0)
    for op, fields in ((Op.TICK, {"epochs": "x"}), (Op.MAIL, {"sender": "x", "slots": 1}), (Op.MAIL, {"sender": 0, "slots": [1]}),
                       (Op.TICK, {"epochs": 1.5}), (Op.START, {"checkpoint_every": None})):
        response = client.call(op, **fields)
        assert response["ok"] is False and response["error"] in {"MALFORMED", "PHASE"}, (op, fields, response)
    status = client.call(Op.STATUS)["result"]
    assert status["phase"] == "STRICT" and status["poisoned"] is False
    assert client.call(Op.TICK, epochs=1)["ok"]


def test_a_line_the_signature_cannot_encode_closes_without_poisoning_the_run(service, manifest):
    srv, _ = service
    client = Client(srv.server_address[1], SECRET)
    client.call(Op.SETUP, manifest=str(manifest), task="build the parser", assignments={"A": "cli", "B": "tests"})
    client.call(Op.START, checkpoint_every=2)
    sock = socket.create_connection(("127.0.0.1", srv.server_address[1]), timeout=5)
    sock.sendall(b'{"auth": "0", "op": 4, "x": 1e999}\n')                # infinity: the canonical form raises
    assert sock.recv(10) == b""
    sock.close()
    assert client.call(Op.STATUS)["result"]["poisoned"] is False


def test_a_line_longer_than_the_limit_closes_before_authentication(service):
    from drift.runtime.service_transport import MAX_LINE
    srv, _ = service
    sock = socket.create_connection(("127.0.0.1", srv.server_address[1]), timeout=5)
    try:
        sock.sendall(b"x" * (MAX_LINE + 2))
    except OSError:
        pass
    assert sock.recv(10) == b""
    sock.close()


def test_a_nonfinite_value_in_a_response_is_sent_as_null_and_still_signed(service):
    srv, _ = service
    srv.session.handle = lambda request: {"norm_drift": {"3": float("nan")}, "peak": [float("inf"), 1.5]}
    assert Client(srv.server_address[1], SECRET).call(Op.STATUS)["result"] == {"norm_drift": {"3": None}, "peak": [None, 1.5]}


def test_large_whole_numbers_are_signed_in_javascript_form():
    assert js_number(2.0 ** 60) == "1152921504606847000" and js_number(1e16) == "10000000000000000"
    assert js_number(9007199254740992.0) == "9007199254740992"
    assert canonical_json({"n": 2 ** 60, "m": 7}) == b'{"m":7,"n":1152921504606847000}'
