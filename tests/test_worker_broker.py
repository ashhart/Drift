import json
import select
import sys
import pytest
from test_worker_activation import append_frame


def fixture(tmp_path):
    root = tmp_path / 'private'; root.mkdir(mode=0o700)
    memory = root / 'memory'; memory.mkdir(mode=0o700)
    script = tmp_path / 'worker.py'
    script.write_text('''import json,sys
import numpy as np
from drift.serving.worker_native_gate import NativeGate
from drift.serving.worker_activation_server import ActivationServer
config=json.load(open(sys.argv[1])); state={'own_slots':[0,1],'foreign_total':0}
def handle(command):
 if command['op']=='append':
  state['foreign_total']+=2
  return {'appended':2}
 if command['op']=='tap':
  np.savez(command['out'],k0=np.ones((2,1,2),dtype=np.float16),v0=np.ones((2,1,2),dtype=np.float16))
  return {'tapped':2,'next_first':2}
 return {'foreign_total':state['foreign_total']}
gate=NativeGate(handle,state,lambda:None)
server=ActivationServer(config['activation'],gate,{'k0':(1,2),'v0':(1,2)},lambda:None) if 'activation' in config else None
if server: server.start()
print(json.dumps({'ready':True,'activation':bool(server)}),flush=True)
try:
 for line in sys.stdin:
  value=json.loads(line)
  print(json.dumps(gate({'op':'observe'}) if value=={'op':'observe'} else {'error':'OWN_PROTOCOL'}),flush=True)
finally:
 if server: server.close()
''')
    activation = {'session': 'session-one', 'source_worker': 'glm', 'target_worker': 'qwen', 'memory_root': str(memory), 'max_bytes': 65536, 'max_rows': 4, 'max_total_rows': 4}
    return root, memory, [sys.executable, str(script), '{worker_config}'], activation


def read(process):
    assert select.select([process.stdout], [], [], 2)[0]
    return json.loads(process.stdout.readline())


@pytest.mark.parametrize('abort', [False, True])
def test_broker_owns_real_separate_fd_and_closes_both_endpoints(tmp_path, abort):
    from drift.serving.worker_broker import launch_owner_worker
    root, memory, command, activation = fixture(tmp_path)
    worker = launch_owner_worker(command, {'worker': 'qwen'}, root=root, activation=activation, wall_seconds=4, stop_timeout=.1)
    peer = worker.activation.socket.dup(); peer.settimeout(1)
    try:
        assert read(worker.process) == {'ready': True, 'activation': True}
        worker.process.stdin.write(b'{"op":"observe"}\n'); worker.process.stdin.flush()
        assert read(worker.process) == {'foreign_total': 0}
        frame = append_frame(memory)
        reply = worker.activation.append(frame['path'], frame['sha256'], frame['rows'])
        assert reply['op'] == 'appended' and reply['rows'] == 2
        assert (reply['source_worker'], reply['target_worker']) == ('glm', 'qwen')
        tapped = worker.activation.tap('own.npz', 0, 2)
        assert (tapped['source_worker'], tapped['target_worker']) == ('qwen', 'glm')
        assert tapped['rows'] == 2 and tapped['next_first'] == 2
        worker.process.stdin.write(b'{"op":"append"}\n'); worker.process.stdin.flush()
        assert read(worker.process) == {'error': 'OWN_PROTOCOL'}
        worker.process.stdin.write(b'{"op":"observe"}\n'); worker.process.stdin.flush()
        assert read(worker.process) == {'foreign_total': 2}
        receipt = worker.finish(abort=abort)
        assert receipt['child_reaped'] and not receipt['process_group_alive']
        assert worker.activation.socket.fileno() == -1 and peer.recv(1) == b''
        assert not worker.config_path.exists()
    finally:
        worker.finish(abort=True); peer.close()


def test_default_launcher_has_no_activation_channel_and_rejects_owner_fd_override(tmp_path):
    from drift.serving.worker_broker import launch_owner_worker
    root, memory, command, activation = fixture(tmp_path)
    worker = launch_owner_worker(command, {'worker': 'qwen'}, root=root, wall_seconds=4, stop_timeout=.1)
    assert worker.activation is None and read(worker.process)['activation'] is False
    assert worker.finish()['child_reaped']
    with pytest.raises(ValueError):
        launch_owner_worker(command, {'worker': 'qwen'}, root=root, activation={**activation, 'fd': 123}, wall_seconds=4, stop_timeout=.1)
    with pytest.raises(ValueError):
        launch_owner_worker(command, {'worker': 'qwen', 'activation': activation}, root=root, wall_seconds=4, stop_timeout=.1)


def test_explicit_own_input_eof_still_collects_cleanup_receipt(tmp_path):
    from drift.serving.worker_broker import launch_owner_worker
    root, memory, command, activation = fixture(tmp_path)
    worker = launch_owner_worker(command, {'worker': 'qwen'}, root=root, activation=activation, wall_seconds=4, stop_timeout=.1)
    assert read(worker.process)['ready']
    pinned = json.loads(worker.config_path.read_text())
    assert type(pinned['activation']['fd']) is int and 'fd' not in activation
    assert worker.config_path.stat().st_mode & 0o777 == 0o400
    worker.process.stdin.close()
    assert worker.finish()['child_reaped']
