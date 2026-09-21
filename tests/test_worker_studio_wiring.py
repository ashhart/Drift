import ast
from pathlib import Path


def test_real_worker_continuation_retains_cache_and_resumes_after_stop():
    tree = ast.parse(Path('scripts/live/studio_drift_worker.py').read_text())
    handle = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'handle')
    cache = object()
    state = {'cache': cache, 'done': True, 'pending_stop': 9, 'own_slots': []}
    consumed = []
    namespace = {'S': state, 'run': lambda ids: consumed.extend(ids)}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == 'drift.serving.worker_studio':
            exec(compile(ast.Module(body=[node], type_ignores=[]), '<worker-import>', 'exec'), namespace)
    exec(compile(ast.Module(body=[handle], type_ignores=[]), '<worker-handle>', 'exec'), namespace)
    namespace['handle']({'op': 'continue', 'ids': [4, 5]})
    assert consumed == [9, 4, 5]
    assert state['cache'] is cache and state['done'] is False
