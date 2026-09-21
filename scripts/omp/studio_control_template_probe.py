"""Check native control-message template prefixes on CPU without loading model weights."""
import json
from drift.serving.worker_codec import CheckpointCodec


def verify(tokenizer, control):
    tool = {'name': 'echo', 'description': 'Return the input', 'parameters': {'type': 'object', 'properties': {'value': {'type': 'string'}}}}
    output = {}
    for kind in ('completed', 'tool'):
        codec = CheckpointCodec(tokenizer, ['Initial instruction', 'Second initial instruction'], [tool])
        codec.append({'role': 'user', 'content': 'Synthetic task'})
        assistant = {'role': 'assistant', 'content': 'Synthetic completion'}
        if kind == 'tool':
            assistant = {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'call-1', 'type': 'function', 'function': {'name': 'echo', 'arguments': {'value': 'synthetic'}}}]}
        completed = tokenizer.apply_chat_template([*codec.messages, assistant], tools=codec.tools, tokenize=True, add_generation_prompt=False, enable_thinking=False, return_dict=False)
        if hasattr(completed, 'get'):
            completed = completed['input_ids']
        if completed[:len(codec.consumed)] != codec.consumed:
            output[kind] = {'status': 'BLOCKED', 'phase': 'assistant_prefix'}
            continue
        codec.assistant(assistant, completed[len(codec.consumed):])
        before = list(codec.consumed)
        messages = [{'role': 'tool', 'tool_call_id': 'call-1', 'content': 'synthetic'}] if kind == 'tool' else []
        try:
            suffix = codec.append_many([*messages, control])
            output[kind] = {'status': 'PASSED', 'prefix_tokens': len(before), 'suffix_tokens': len(suffix), 'prefix_preserved': codec.consumed[:len(before)] == before}
        except Exception as error:
            output[kind] = {'status': 'BLOCKED', 'phase': 'control_suffix', 'error_type': type(error).__name__}
    return output


if __name__ == '__main__':
    import argparse
    from transformers import AutoTokenizer
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True, trust_remote_code=False)
    from drift.serving.worker_studio_control import reader_control_message
    report = verify(tokenizer, reader_control_message({'system_prompt': ['Updated declared context']}))
    print(json.dumps(report, sort_keys=True))
    raise SystemExit(0 if all(value['status'] == 'PASSED' for value in report.values()) else 2)
