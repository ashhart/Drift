"""Validate owner-pinned room-only budgets and commands before any worker dispatch."""
import hashlib
import json
from pathlib import Path
import re

LIMITS = dict(max_input_bytes=262144, max_output_tokens=512, max_session_tokens=20000, max_turns=6, deadline_ms=50000)
TOKEN = re.compile(r'[A-Za-z0-9_.-]{1,96}')
SHA = re.compile(r'[a-f0-9]{64}')
ENV = {'PATH', 'HOME', 'SSH_AUTH_SOCK', 'PYTHONPATH', 'PYTHONDONTWRITEBYTECODE', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pinned(item, maximum=None):
    path = Path(item['path'])
    if not path.is_absolute() or not SHA.fullmatch(item['sha256']) or not path.is_file(): raise ValueError('ROOM_PIN')
    if maximum is not None and path.stat().st_size > maximum: raise ValueError('ROOM_SIZE')
    if digest(path) != item['sha256']: raise ValueError('ROOM_PIN')
    return path.resolve(strict=True)


def command(value):
    pinned(dict(path=value['executable'], sha256=value['sha256']))
    if not Path(value['cwd']).is_absolute() or not Path(value['cwd']).is_dir(): raise ValueError('ROOM_COMMAND')
    if not isinstance(value['args'], list) or any(type(arg) is not str for arg in value['args']) or len(json.dumps(value).encode()) > 16384: raise ValueError('ROOM_COMMAND')
    if set(value['env']) - ENV or any(type(item) is not str for item in value['env'].values()): raise ValueError('ROOM_ENV')
    for artifact in value.get('artifacts', []): pinned(artifact)


def validate_entry(entry, *, development=False):
    identity = entry['identity']
    if any(type(number) is not int for number in entry['limits'].values()): raise ValueError('ROOM_LIMIT')
    maximum = entry['limits'].get('max_session_tokens')
    allowed = (20000, 65536) if entry['expected']['nativeStates'] == ['reconstructed'] else (20000,)
    if development and entry['expected']['nativeStates'] == ['reconstructed'] and entry['context_window'] == 16384:
        allowed += (10 * (16384 + 512),)
    fixed = {**LIMITS, 'max_session_tokens': maximum}
    if development: fixed.update(max_turns=10, deadline_ms=170000)
    if maximum not in allowed or entry['limits'] != fixed: raise ValueError('ROOM_LIMIT')
    if entry['memory_mode'] != 'no-link' or entry['experimental_multi_turn'] is not True: raise ValueError('ROOM_LIMIT')
    if entry['expected']['backend'] != 'live' or entry['expected']['nativeStates'] not in (['retained'], ['reconstructed']): raise ValueError('ROOM_BACKEND')
    if any(not TOKEN.fullmatch(identity[key]) for key in ('session', 'worker', 'model_id')): raise ValueError('ROOM_IDENTITY')
    if any(not SHA.fullmatch(identity[key]) for key in ('model_sha256', 'translator_sha256')): raise ValueError('ROOM_PIN')
    if type(entry['context_window']) is not int or not 512 <= entry['context_window'] <= 131072: raise ValueError('ROOM_LIMIT')


def tree_digest(root):
    paths = sorted((root/'src').rglob('*')) + [root/'agents/duo-peer.md']
    return hashlib.sha256(json.dumps([(str(path.relative_to(root)), digest(path)) for path in paths if path.is_file()], separators=(',', ':')).encode()).hexdigest()


def load(path, expected, *, development=False):
    raw_path = pinned(dict(path=str(path), sha256=expected), 65536)
    value = json.loads(raw_path.read_text())
    purposes = ('restricted-no-link-room', 'restricted-api-development') if development else ('restricted-no-link-room',)
    if set(value) != {'version', 'owner', 'omp', 'duo', 'parent', 'child', 'cleanup', 'purpose'} or value['version'] != 1 or value['purpose'] not in purposes: raise ValueError('ROOM_PROFILE')
    owner_path = pinned(value['owner'], 262144); owner = json.loads(owner_path.read_text())
    if owner.get('version') != 1 or owner.get('experimental') is not True or len(owner.get('workers', [])) != 2: raise ValueError('ROOM_OWNER')
    ids = set(); workers = set(); sessions = set()
    for entry in owner['workers']:
        validate_entry(entry, development=value['purpose'] == 'restricted-api-development'); command(entry['command'])
        if not entry.get('artifacts'): raise ValueError('ROOM_PIN')
        for artifact in entry['artifacts']: pinned(artifact)
        ids.add('drift-experimental/' + entry['identity']['model_id']); workers.add(entry['identity']['worker']); sessions.add(entry['identity']['session'])
    if len(ids) != 2 or len(workers) != 2 or len(sessions) != 2 or {value['parent'], value['child']} != ids: raise ValueError('ROOM_ROUTE')
    pinned(value['omp'])
    duo = Path(value['duo']['path'])
    if not duo.is_absolute() or tree_digest(duo) != value['duo']['sha256']: raise ValueError('ROOM_DUO_PIN')
    if len(value['cleanup']) != 2 or {item['worker'] for item in value['cleanup']} != workers: raise ValueError('ROOM_CLEANUP')
    for item in value['cleanup']:
        command(item['command'])
        identity = next(entry['identity'] for entry in owner['workers'] if entry['identity']['worker'] == item['worker'])
        if item['session'] != identity['session'] or not item['command'].get('artifacts'): raise ValueError('ROOM_CLEANUP_PIN')
    if value['cleanup'][0]['command'] == value['cleanup'][1]['command']: raise ValueError('ROOM_CLEANUP_PIN')
    return value, owner
