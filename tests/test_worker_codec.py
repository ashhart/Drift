from collections import UserDict
import pytest
from drift.serving.worker_codec import CheckpointCodec
from drift.serving.worker_contract import WorkerError


class Tokenizer:
    def __init__(self, result):
        self.result = result

    def apply_chat_template(self, messages, **kwargs):
        self.messages, self.options = messages, kwargs
        return self.result


def test_ordered_initial_system_blocks_are_one_native_system_message():
    tokenizer = Tokenizer([1, 2])
    codec = CheckpointCodec(tokenizer, ['first\nblock', '', 'third block'], [])
    assert codec.append({'role': 'user', 'content': 'own input'}) == [1, 2]
    assert tokenizer.messages[0] == {'role': 'system', 'content': 'first\nblock\n\n\n\nthird block'}
    assert len([m for m in tokenizer.messages if m['role'] == 'system']) == 1
    assert tokenizer.options['return_dict'] is False


def test_batch_encoding_normalizes_only_one_bounded_integer_sequence():
    tokenizer = Tokenizer(UserDict(input_ids=[1, 2], attention_mask=[1, 1]))
    codec = CheckpointCodec(tokenizer, [], [], max_tokens=3)
    assert codec.append({'role': 'user', 'content': 'own input'}) == [1, 2]
    assert tokenizer.messages[0]['role'] == 'user'


@pytest.mark.parametrize('value', [[], [[1, 2]], [True], [-1], [2**31], [1.0], [1, 2, 3, 4], (1, 2)])
def test_invalid_batch_encoding_sequences_fail_before_committing_history(value):
    tokenizer = Tokenizer(UserDict(input_ids=value))
    codec = CheckpointCodec(tokenizer, ['system'], [], max_tokens=3)
    original = list(codec.messages)
    with pytest.raises(WorkerError):
        codec.append({'role': 'user', 'content': 'own input'})
    assert codec.messages == original and codec.consumed == []


def test_normalization_preserves_strict_prefix_and_mid_system_failures():
    tokenizer = Tokenizer(UserDict(input_ids=[1, 2]))
    codec = CheckpointCodec(tokenizer, [], [])
    codec.append({'role': 'user', 'content': 'own input'})
    tokenizer.result = UserDict(input_ids=[1, 9, 3])
    with pytest.raises(WorkerError, match='CAPABILITY'):
        codec.append({'role': 'system', 'content': 'new control'})
    assert codec.consumed == [1, 2]
    assert tokenizer.messages[-1]['role'] == 'system'


def test_template_rejection_is_not_silently_rewritten():
    class Reject(Tokenizer):
        def apply_chat_template(self, messages, **kwargs):
            raise ValueError('template rejects this role')
    codec = CheckpointCodec(Reject(None), [], [])
    with pytest.raises(ValueError):
        codec.append({'role': 'system', 'content': 'unsupported midstream control'})
    assert codec.messages == []
