"""The Drift client of the owner's RDMA daemon speaks its socket protocol and opens no verbs objects itself."""
import os, socket, threading
import pytest
from drift.serving.handoffd_client import HandoffdClient, HandoffdError


def serve(path, replies):
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); server.bind(path); server.listen(1)
    seen = []

    def run():
        for reply in replies:
            conn, _ = server.accept()
            seen.append(conn.recv(4096).decode()); conn.sendall(reply.encode()); conn.close()
        server.close()

    thread = threading.Thread(target=run, daemon=True); thread.start()
    return seen, thread


def test_pull_sends_the_daemons_command_and_parses_its_timing(tmp_path):
    path = f"/tmp/drift-test-{os.getpid()}.sock"
    try:
        seen, thread = serve(path, ["OK spark-a.invalid 1048576 2000000 500000 4.19 100000 3000000\nEND\n", "ERR spark-a.invalid no such file\nEND\n", "OK spark-a.invalid 1 1 1\n"])
        client = HandoffdClient(path, timeout_s=5)
        result = client.pull("spark-a.invalid", "/dev/shm/glm53-handoff/drift-abc/rank0.bin", offset=4096)
        assert seen[0] == "PULL spark-a.invalid /dev/shm/glm53-handoff/drift-abc/rank0.bin shm:4096 1\n" and result.bytes == 1048576 and result.gbit_s == 4.19 and result.job_ns == 3000000
        with pytest.raises(HandoffdError, match="no such file"):
            client.pull("spark-a.invalid", "/dev/shm/glm53-handoff/drift-abc/rank0.bin")
        with pytest.raises(HandoffdError, match="before END"):
            client.pull("spark-a.invalid", "/dev/shm/glm53-handoff/drift-abc/rank0.bin")
        thread.join(timeout=5)
    finally:
        if os.path.exists(path):
            os.unlink(path)
    for bad in (("spark-a.invalid;rm", "/dev/shm/x", 0), ("spark-a.invalid", "/dev/shm/../etc/passwd", 0), ("spark-a.invalid", "/dev/shm/a b", 0), ("spark-a.invalid", "/dev/shm/x", 100)):
        with pytest.raises(ValueError):
            HandoffdClient(path).pull(bad[0], bad[1], offset=bad[2])


def test_blob_in_memory_parses_like_the_file(tmp_path):
    import json, struct
    import numpy as np
    from drift.serving.glm53_handoff import MAGIC, pack_fp8_ds_mla, read_latents
    latents = np.random.default_rng(0).standard_normal((5, 512)).astype(np.float32)
    rows = np.zeros((64, 656), np.uint8); rows[:5, :528] = pack_fp8_ds_mla(latents)
    header = json.dumps({"format": "glm53-handoff-raw-v1", "cache_config": {"cache_dtype": "fp8_ds_mla"}, "export_from": 0, "n_tokens": 5,
                         "tensors": [{"layer": "language_model.model.layers.3.self_attn.attn", "shape": [1, 64, 656], "dtype": "uint8", "offset": 0, "nbytes": rows.nbytes}]}).encode()
    blob = (MAGIC + struct.pack("<Q", len(header)) + header).ljust(4096, b"\0") + rows.tobytes()
    path = tmp_path / "rank0.bin"; path.write_bytes(blob)
    from_file, from_memory = read_latents(path), read_latents(memoryview(blob))
    assert list(from_file) == list(from_memory) == [3]
    np.testing.assert_array_equal(from_file[3], from_memory[3])


def test_orphan_sends_the_test_command_and_checks_the_reply(tmp_path):
    path = f"/tmp/drift-test-orphan-{os.getpid()}.sock"
    try:
        seen, thread = serve(path, ["ORPHANED spark-a.invalid\nEND\n", "ERR - bad TEST_ORPHAN\nEND\n"])
        client = HandoffdClient(path, timeout_s=5)
        client.test_orphan("spark-a.invalid")
        assert seen[0] == "TEST_ORPHAN spark-a.invalid\n"
        with pytest.raises(HandoffdError, match="bad TEST_ORPHAN"):
            client.test_orphan("spark-a.invalid")
        thread.join(timeout=5)
    finally:
        if os.path.exists(path):
            os.unlink(path)
    with pytest.raises(ValueError):
        HandoffdClient(path).test_orphan("spark-a.invalid;rm")


def test_pull_file_names_the_local_file_in_the_command(tmp_path):
    path = f"/tmp/drift-test-file-{os.getpid()}.sock"
    try:
        seen, thread = serve(path, ["OK spark-a.invalid 4096 1000 100 32.77 10 2000\nEND\n"])
        result = HandoffdClient(path, timeout_s=5).pull_file("spark-a.invalid", "/dev/shm/glm53-handoff/x/rank0.bin", "/tmp/drift-pull/p3.bin", unlink=False)
        assert seen[0] == "PULL spark-a.invalid /dev/shm/glm53-handoff/x/rank0.bin /tmp/drift-pull/p3.bin 0\n" and result.bytes == 4096
        thread.join(timeout=5)
    finally:
        if os.path.exists(path):
            os.unlink(path)
    with pytest.raises(ValueError):
        HandoffdClient(path).pull_file("spark-a.invalid", "/dev/shm/x", "/tmp/../etc/x")
