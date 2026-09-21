"""Prepare the public API task with identical tool privileges for both worker roles."""
import json
from pathlib import Path
import shutil
from native_room_profile import digest
from native_room_stage import prepare as room_prepare
from task_smoke import prepare as task_prepare

PROMPT = ('Use Duo to fix api.py: GET /health must return 200 JSON {"status":"ok"}; missing paths must remain JSON 404. '
          'Each response has a hard 512-token limit: keep tool arguments and Hub messages compact, and do not repeat this task specification. '
          'Create one todo and invite duo-peer named DuoPeer to read api.py and verify.py and propose a split. '
          'Follow PROPOSE then AGREE before work; aim for DuoPeer to implement and verify while you review. '
          'Use drift_task_write to replace api.py and drift_task_verify to check it; report DONE through Hub, review and finish. '
          'Use only assigned task tools and pinned room coordination.')


def prepare(profile, evidence, python, port):
    if type(port) is not int or not 1024 <= port <= 65535: raise ValueError('DEVELOPMENT_PORT')
    prepared = room_prepare(profile, evidence)
    scratch = Path(prepared['cwd']); task = scratch/'task'
    scope = task_prepare(task, python.resolve(strict=True)); scope['port'] = port
    verifier = task/'verify.py'
    verifier.write_text(verifier.read_text().replace('("127.0.0.1", 0)', f'("127.0.0.1", {port})'))
    scope['runs'][0]['artifacts'][0]['sha256'] = digest(verifier)
    scope_path = evidence/'task-scope.json'; scope_path.write_text(json.dumps(scope))
    here = Path(__file__).resolve().parent; target = scratch/'scripts/omp'
    for name in ('development_extension.mjs', 'development_guard.mjs', 'development_tools.mjs', 'development_verify.mjs', 'task_scope.mjs'):
        shutil.copyfile(here/name, target/name)
    for folder in ('agents', '.omp/agents'): shutil.copytree(scratch/folder, task/folder)
    room_scope = dict(root=str(task), parent=profile['parent'], child=profile['child'], run=evidence.name)
    (scratch/'scope.json').write_text(json.dumps(room_scope))
    command = prepared['command']; command[command.index('--cwd')+1] = str(task)
    command[command.index(str(target/'native_room_extension.mjs'))] = str(target/'development_extension.mjs')
    command[-1] = PROMPT
    if profile.get('purpose') == 'restricted-api-development': command[command.index('--max-time')+1] = '170'
    prepared.update(cwd=str(task), source_root=str(scratch), prompt_bytes=len(PROMPT.encode()), qualification='restricted-api-development', channels=['peer-text', 'shared-public-artifact'])
    prepared['env'].update(DRIFT_TASK_CONFIG=str(scope_path), DRIFT_TASK_CONFIG_SHA256=digest(scope_path))
    prepared['pins'] = {str(path.relative_to(task)): digest(path) for path in task.rglob('*') if path.is_file()}
    prepared['source_pins'] = {str(path): digest(path) for path in scratch.rglob('*') if path.is_file() and not path.is_relative_to(task)}
    prepared['source_pins'][str(scope_path)] = digest(scope_path)
    return prepared
