"""Metrics samples are synthetic Prometheus exposition, never live model data."""
import pytest

from drift.serving.live_metrics import MetricsError, MetricsSnapshot, parse_metrics

MODEL = 'GLM-5.3-Flash-EXL3'
NAMES = ('vllm:num_requests_running', 'vllm:num_requests_waiting', 'vllm:kv_cache_usage_perc')


def exposition(values=(0, 0, 0.0), labels=None):
    labels = labels or f'engine="0",model_name="{MODEL}"'
    return '\n'.join(f'{name}{{{labels}}} {value}' for name, value in zip(NAMES, values))


def test_complete_target_metrics_ignore_unrelated_series_and_waiting_reasons():
    text = '# comment\n' + exposition((1, 0, 0.25)) + '\n' + exposition((9, 9, 9), 'engine="1",model_name="other"')
    text += '\nvllm:num_requests_waiting_by_reason{engine="0",reason="scheduler"} 999'
    assert parse_metrics(text) == MetricsSnapshot(1, 0, 0.25, MODEL, '0')


@pytest.mark.parametrize('fault', ['missing', 'duplicate', 'extra_label_duplicate', 'nan', 'inf', 'negative', 'fractional_count', 'missing_labels', 'duplicate_label', 'invalid_escape'])
def test_incomplete_or_ambiguous_metrics_never_become_zero(fault):
    text = exposition()
    if fault == 'missing': text = '\n'.join(text.splitlines()[:2])
    if fault == 'duplicate': text += '\n' + text.splitlines()[0]
    if fault == 'extra_label_duplicate': text += '\n' + exposition((0,), f'engine="0",model_name="{MODEL}",replica="other"')
    if fault == 'nan': text = exposition((0, 0, 'NaN'))
    if fault == 'inf': text = exposition((0, 'Inf', 0))
    if fault == 'negative': text = exposition((-1, 0, 0))
    if fault == 'fractional_count': text = exposition((0.5, 0, 0))
    if fault == 'missing_labels': text = '\n'.join(f'{name} 0' for name in NAMES)
    if fault == 'duplicate_label': text = exposition(labels=f'engine="0",engine="0",model_name="{MODEL}"')
    if fault == 'invalid_escape': text = exposition(labels='engine="0",model_name="bad\\tmodel"')
    with pytest.raises(MetricsError):
        parse_metrics(text)


def test_label_escapes_and_optional_timestamps_are_parsed():
    model = 'synthetic "model"\\revision\nline'
    labels = 'model_name="synthetic \\"model\\"\\\\revision\\nline",engine="0"'
    assert parse_metrics(exposition((1, 0, '1e-3'), labels), model).kv_usage == 0.001
    assert parse_metrics('\n'.join(line + ' 123' for line in exposition().splitlines())).running == 0


@pytest.mark.parametrize('values', [(True, 0, 0), (0, -1, 0), (0, 0, float('nan')), (1.5, 0, 0)])
def test_manually_constructed_snapshots_are_also_validated(values):
    with pytest.raises(MetricsError):
        MetricsSnapshot(*values, MODEL, '0')
