"""Exercise real tokenizer/parser continuation on synthetic native assistant serialization only."""
import argparse
import json
from drift.serving.worker_assistant import parsed_assistant
from drift.serving.worker_codec import CheckpointCodec


def verify(tokenizer, parse_tools):
    tool = {'name': 'echo', 'description': 'Echo value', 'parameters': {'type': 'object', 'properties': {'value': {'type': 'string'}}}}
    assistant = {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'fixture', 'type': 'function', 'function': {'name': 'echo', 'arguments': {'value': 'synthetic'}}}]}
    rows = []
    for variant in ('canonical', 'whitespace', 'leading_whitespace', 'trailing_whitespace', 'extra_thinking', 'semantic_tamper', 'own_input_tamper'):
        codec = CheckpointCodec(tokenizer, ['Initial instruction'], [tool])
        codec.append({'role': 'user', 'content': 'Synthetic tool task'})
        rendered = tokenizer.apply_chat_template([*codec.messages, assistant], tools=codec.tools, tokenize=False, add_generation_prompt=False, enable_thinking=False)
        prefix = tokenizer.decode(codec.consumed, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if not rendered.startswith(prefix): raise RuntimeError('INITIAL_PREFIX')
        raw = rendered[len(prefix):].split('<|im_end|>')[0]
        if variant == 'whitespace': raw = raw.replace('\n</tool_call>', '\n  \n</tool_call>')
        if variant == 'leading_whitespace': raw = '\n  '+raw
        if variant == 'trailing_whitespace': raw += '\n  \n'
        if variant == 'extra_thinking': raw = '<think>Synthetic thought</think>\n' + raw
        ids = tokenizer.encode(raw, add_special_tokens=False)
        text = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        cleaned, calls = parse_tools(text, tokenizer, codec.tools)
        message, events = parsed_assistant(cleaned, calls)
        codec.assistant(message, ids, tokenizer.convert_tokens_to_ids('<|im_end|>'), raw_text=text)
        before = list(codec.consumed)
        if variant == 'semantic_tamper': codec.messages[-1]['tool_calls'][0]['function']['arguments']['value'] = 'changed'
        if variant == 'own_input_tamper': codec.messages[0]['content'] = 'changed'
        try:
            codec.append({'role': 'tool', 'tool_call_id': calls[0].id, 'content': 'synthetic'})
            outcome = 'PASSED'
        except Exception as error:
            outcome = 'REJECTED' if str(error) == 'CAPABILITY' else type(error).__name__
        rows.append({'variant': variant, 'outcome': outcome, 'tool_calls': len(events), 'arguments_match': events[0]['arguments'] == {'value': 'synthetic'}, 'thinking_prefix_present': '</think>' in prefix, 'native_prefix_preserved': codec.consumed[:len(before)] == before})
    return rows


def main():
    from transformers import AutoTokenizer
    from mlx_vlm.tool_parsers import _infer_tool_parser, load_tool_module
    from omlx.api.tool_calling import parse_tool_calls
    parser = argparse.ArgumentParser(); parser.add_argument('--checkpoint', required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True, trust_remote_code=False)
    module = load_tool_module(_infer_tool_parser(tokenizer.chat_template))
    tokenizer.has_tool_calling = True
    tokenizer.tool_call_start, tokenizer.tool_call_end = module.tool_call_start, module.tool_call_end
    tokenizer.tool_parser = module.parse_tool_call
    report = verify(tokenizer, parse_tool_calls)
    print(json.dumps(report, sort_keys=True))
    expected = ['PASSED', 'PASSED', 'REJECTED', 'REJECTED', 'PASSED', 'REJECTED', 'REJECTED']
    return 0 if [row['outcome'] for row in report] == expected and all(row['arguments_match'] and row['native_prefix_preserved'] for row in report) else 2


if __name__ == '__main__': raise SystemExit(main())
