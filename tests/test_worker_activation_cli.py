import json
import os
import select
import socket
import subprocess
import sys
from test_worker_activation import append_frame


def test_supervised_child_receives_activation_on_separate_inherited_fd(tmp_path):
    root = tmp_path / 'memory'; root.mkdir(mode=0o700)
    peer, child_socket = socket.socketpair()
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'fd': child_socket.fileno(), 'session': 'session-one', 'source_worker': 'glm', 'target_worker': 'qwen', 'memory_root': str(root), 'max_bytes': 65536, 'max_rows': 4, 'max_total_rows': 4}))
    fixture = tmp_path / 'fixture.py'
    fixture.write_text('''import json,sys
from drift.serving.worker_native_gate import NativeGate
from drift.serving.worker_activation_server import ActivationServer
state = {'cache': object(), 'own_slots': [], 'foreign_total': 0}
def handle(command):
 if command['op'] == 'append':
  state['foreign_total'] += 2
  return {'appended': 2}
 return {'foreign_total': state['foreign_total']}
gate = NativeGate(handle, state, lambda: None)
server = ActivationServer(json.load(open(sys.argv[1])), gate, {'k0': (1,2), 'v0': (1,2)}, lambda: None)
server.start()
print(json.dumps({'ready': True}), flush=True)
try:
 for line in sys.stdin:
  if line.strip() == 'quit': break
  print(json.dumps(gate({'op': 'observe'})), flush=True)
finally: server.close()
''')
    control_r, control_w = os.pipe(); evidence_r, evidence_w = os.pipe()
    own_r, own_w = os.pipe(); out_r, out_w = os.pipe()
    inherited = (control_r, evidence_w, own_r, out_w, child_socket.fileno())
    process = subprocess.Popen([sys.executable, '-m', 'drift.serving.worker_supervisor', '--control-fd', str(control_r), '--evidence-fd', str(evidence_w), '--input-fd', str(own_r), '--output-fd', str(out_w), '--activation-fd', str(child_socket.fileno()), '--wall-seconds', '4', '--stop-timeout', '0.1', '--', sys.executable, str(fixture), str(config)], pass_fds=inherited, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    child_socket.close(); peer.settimeout(2)
    try:
        assert select.select([out_r], [], [], 2)[0]
        assert json.loads(os.read(out_r, 4096)) == {'ready': True}
        os.write(own_w, b'observe\n')
        assert select.select([out_r], [], [], 2)[0]
        assert json.loads(os.read(out_r, 4096)) == {'foreign_total': 0}
        peer.sendall(json.dumps(append_frame(root)).encode() + b'\n')
        assert json.loads(peer.recv(4096))['op'] == 'appended'
        os.write(own_w, b'observe\n')
        assert select.select([out_r], [], [], 2)[0]
        assert json.loads(os.read(out_r, 4096)) == {'foreign_total': 2}
        os.write(own_w, b'quit\n')
        stdout, stderr = process.communicate(timeout=3)
        assert process.returncode == 0 and not stdout and not stderr
        receipt = json.loads(os.read(evidence_r, 4096))
        assert receipt['child_reaped'] and not receipt['process_group_alive']
        assert peer.recv(1) == b''
    finally:
        if process.poll() is None: process.kill(); process.wait()
        peer.close()
        for fd in (control_r, control_w, evidence_r, evidence_w, own_r, own_w, out_r, out_w): os.close(fd)
