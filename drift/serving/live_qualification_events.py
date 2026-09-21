"""Aggregate-only limits and event accounting for an owned cancellation probe."""
from dataclasses import dataclass
import math


class QualificationError(RuntimeError):
    """A fixed error code, never model output or an underlying exception message."""


@dataclass(frozen=True)
class Limits:
    max_new: int = 256
    max_prompt_tokens: int = 4096
    max_events: int = 1024
    max_line_bytes: int = 65536
    max_output_bytes: int = 262144
    max_seconds: float = 180
    stop_timeout: float = 2

    def __post_init__(self):
        for name in ('max_new', 'max_prompt_tokens', 'max_events', 'max_line_bytes', 'max_output_bytes'):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError('qualification count limits must be positive integers')
        for value in (self.max_seconds, self.stop_timeout):
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError('qualification time limits must be finite and positive')


@dataclass(frozen=True)
class RequestEvidence:
    started: bool
    active: bool
    abort_sent: bool
    cancelled: bool
    completed: bool
    concurrent: bool = False


class EventCounts:
    def __init__(self, limits, reserved_tokens=0):
        self.limits = limits
        self.reserved_tokens = reserved_tokens
        self.events = self.started = self.output = self.nonempty = self.output_bytes = 0
        self.prompt_tokens = self.done = self.cancelled = 0
        self.terminal = False

    def accept(self, event):
        self.events += 1
        if self.events > self.limits.max_events:
            raise QualificationError('EVENT_LIMIT')
        if not isinstance(event, dict) or self.terminal:
            raise QualificationError('EVENT_PROTOCOL')
        if event.get('started') is True:
            if self.started or set(event) - {'started', 'own_prompt_tokens', 'span_start', 't'}:
                raise QualificationError('START_PROTOCOL')
            tokens = event.get('own_prompt_tokens')
            if type(tokens) is not int or not 0 <= tokens <= self.limits.max_prompt_tokens - self.reserved_tokens:
                raise QualificationError('PROMPT_TOKEN_LIMIT')
            self.started = 1
            self.prompt_tokens = tokens
        elif 'text' in event:
            if not self.started or set(event) - {'text', 't'} or not isinstance(event['text'], str):
                raise QualificationError('OUTPUT_PROTOCOL')
            size = len(event['text'].encode('utf-8'))
            self.output += 1
            self.nonempty += int(size > 0)
            self.output_bytes += size
            if self.output_bytes > self.limits.max_output_bytes:
                raise QualificationError('OUTPUT_BYTE_LIMIT')
        elif event.get('done') is True and not set(event) - {'done', 't'}:
            if not self.started:
                raise QualificationError('DONE_PROTOCOL')
            self.done = 1
            self.terminal = True
        elif event.get('cancelled') is True and set(event) == {'cancelled'}:
            self.cancelled = 1
            self.terminal = True
        else:
            raise QualificationError('CHILD_EVENT_FAILURE')

    def report(self):
        return {'events': self.events, 'started_events': self.started, 'output_chunks': self.output,
                'nonempty_output_chunks': self.nonempty, 'output_bytes': self.output_bytes,
                'prompt_tokens': self.prompt_tokens, 'reserved_prompt_tokens': self.reserved_tokens, 'done_events': self.done,
                'cancelled_events': self.cancelled, 'requested_max_new': self.limits.max_new,
                'actual_generated_tokens': None}
