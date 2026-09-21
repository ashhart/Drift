"""Use the supplied checkpoint tokenizer and reject any rewritten native prefix."""
from copy import deepcopy
from collections.abc import Mapping
from drift.serving.worker_contract import require
from drift.serving.worker_assistant_history import AssistantHistory


class CheckpointCodec:
    def __init__(self, tokenizer, system_prompt, tools, *, max_tokens=65536):
        require(type(system_prompt) is list and all(type(value) is str for value in system_prompt))
        require(type(max_tokens) is int and 0 < max_tokens <= 65536, 'LIMIT')
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.messages = [{'role': 'system', 'content': '\n\n'.join(system_prompt)}] if system_prompt else []
        self.tools = [{'type': 'function', 'function': deepcopy(tool)} for tool in tools]
        self.consumed = []
        self.assistants = AssistantHistory()

    def append(self, message):
        return self.append_many([message])

    def append_many(self, messages):
        candidate = [*self.messages, *deepcopy(messages)]
        ids = self.tokenizer.apply_chat_template(self.assistants.render(candidate), tools=self.tools or None,
                                                 tokenize=True, add_generation_prompt=True,
                                                 enable_thinking=False, return_dict=False)
        if isinstance(ids, Mapping):
            ids = ids.get('input_ids')
        require(type(ids) is list)
        require(0 < len(ids) <= self.max_tokens, 'LIMIT')
        require(all(type(x) is int and 0 <= x < 2**31 for x in ids))
        ids = list(ids)
        require(ids[:len(self.consumed)] == self.consumed, 'CAPABILITY')
        suffix = ids[len(self.consumed):]
        require(bool(suffix))
        self.messages = candidate
        self.consumed = ids
        return suffix

    def assistant(self, message, generated_ids, stop_id=None, *, raw_text=None):
        require(type(message) is dict and message.get('role') == 'assistant')
        require(type(generated_ids) is list and all(type(x) is int and x >= 0 for x in generated_ids))
        require(stop_id is None or (type(stop_id) is int and stop_id >= 0))
        if raw_text is not None:
            require(type(raw_text) is str and self.tokenizer.decode(generated_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False) == raw_text, 'CAPABILITY')
            self.assistants.remember(len(self.messages), message, raw_text)
        self.messages.append(deepcopy(message))
        self.consumed.extend(generated_ids)
        if stop_id is not None:
            self.consumed.append(stop_id)
