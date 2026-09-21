"""Parse complete target-engine scheduler and KV-cache metrics using only the standard library."""
import math
import re
from dataclasses import dataclass

NAMES = ('vllm:num_requests_running', 'vllm:num_requests_waiting', 'vllm:kv_cache_usage_perc')
LINE = re.compile(r'([^\s{]+)(?:\{(.*)\})?\s+(\S+)(?:\s+(\S+))?\s*')
LABEL = re.compile(r'\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*"((?:[^"\\]|\\["\\n])*)"\s*')


class MetricsError(ValueError):
    """Required target metrics are absent, invalid or ambiguous."""


@dataclass(frozen=True)
class MetricsSnapshot:
    running: float
    waiting: float
    kv_usage: float
    model: str
    engine: str

    def __post_init__(self):
        for value in (self.running, self.waiting, self.kv_usage):
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise MetricsError('metric values must be finite and nonnegative')
        if not float(self.running).is_integer() or not float(self.waiting).is_integer():
            raise MetricsError('request counts must be integers')
        if not isinstance(self.model, str) or not self.model or not isinstance(self.engine, str) or not self.engine:
            raise MetricsError('target model and engine are required')


def _labels(text):
    labels = {}
    while text:
        match = LABEL.match(text)
        if match is None or match[1] in labels:
            raise MetricsError('malformed or duplicate metric label')
        value = re.sub(r'\\(["\\n])', lambda item: '\n' if item[1] == 'n' else item[1], match[2])
        labels[match[1]] = value
        text = text[match.end():]
        if text:
            if not text.startswith(','):
                raise MetricsError('invalid metric label separator')
            text = text[1:].lstrip()
    return labels


def parse_metrics(text, model='GLM-5.3-Flash-EXL3', engine='0'):
    """Select one exact sample per required metric; never infer missing zeros."""
    if not isinstance(text, str) or len(text.encode('utf-8')) > 2 * 1024 * 1024:
        raise MetricsError('metrics text is invalid or oversized')
    values = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        name = re.split(r'[\s{]', line, maxsplit=1)[0]
        if name not in NAMES:
            continue
        match = LINE.fullmatch(line)
        if match is None or match[2] is None:
            raise MetricsError('malformed required metric sample')
        labels = _labels(match[2])
        if not {'engine', 'model_name'} <= labels.keys():
            raise MetricsError('required metric lacks target labels')
        if labels['engine'] != engine or labels['model_name'] != model:
            continue
        if name in values:
            raise MetricsError('ambiguous target metric sample')
        try:
            values[name] = float(match[3])
            if match[4] is not None and not math.isfinite(float(match[4])):
                raise ValueError
        except ValueError:
            raise MetricsError('invalid target metric value') from None
    if set(values) != set(NAMES):
        raise MetricsError('target metrics are incomplete')
    return MetricsSnapshot(*(values[name] for name in NAMES), model, engine)
