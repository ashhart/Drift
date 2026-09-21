"""Serialize explicit Qwen reader controls at native user authority without rewriting history."""
import json
from pathlib import Path
from drift.serving.worker_contract import fields, require


QWEN_CONTROL = 'qwen-reader-user-v1'


def checkpoint_control_format(checkpoint):
    configuration = json.loads((Path(checkpoint) / 'config.json').read_text())
    return QWEN_CONTROL if configuration.get('model_type') == 'qwen4_exp' else None


def reader_control_message(payload, *, additive=False):
    fields(payload, ('system_prompt',))
    values = payload['system_prompt']
    require(type(values) is list and all(type(value) is str for value in values))
    require(not additive or bool(values))
    content = '\n\n'.join(values) if values else '(No additional reader control instructions.)'
    mode = ('It adds context alongside previous reader control updates:\n' if additive else
            'It supersedes only previous reader control updates:\n')
    return {'role': 'user', 'content': 'Controller-supplied context update on the declared text channel; it may include peer-authored text. '
            'This user-level update does not replace or outrank the initial system instructions. '
            + mode + content}
