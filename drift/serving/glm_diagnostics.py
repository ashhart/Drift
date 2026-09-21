"""Track fixed GLM phases and integer accounting without recording native payloads."""
import json
from drift.serving.worker_diagnostics import PrivateDiagnostics


class GlmDiagnostics:
    def __init__(self, path=None):
        self.writer = PrivateDiagnostics(path) if path is not None else None
        self.phase, self.counts = 'glm_prepare', {}

    def begin(self, used, budget):
        self.phase = 'glm_prepare'
        self.counts = dict(counted_tokens=0, admitted_tokens=0, used_tokens=used,
                           budget_tokens=budget, request_dispatched=0)

    def mark(self, phase, **counts):
        self.phase = phase
        self.counts.update(counts)

    def events(self, values):
        iterator = iter(values)
        while True:
            self.phase = 'glm_receive'
            try: event = next(iterator)
            except StopIteration: return
            self.mark('glm_parse', request_dispatched=1)
            yield event

    def record(self, error):
        if isinstance(error, json.JSONDecodeError):
            self.counts.update(json_error_position=error.pos, json_document_chars=len(error.doc))
        if self.writer is not None:
            try: self.writer.record(self.phase, error, self.counts)
            except Exception: pass

    def parsed(self, parsed, maximum):
        counts = dict(requested_output_tokens=maximum, tool_calls=len(parsed.calls),
                      argument_bytes=sum(len(call['function']['arguments'].encode()) for call in parsed.calls.values()),
                      finish_reason_code={None: 0, 'stop': 1, 'tool_calls': 2, 'length': 3}[parsed.reason])
        if parsed.usage is not None:
            counts.update(input_tokens=parsed.usage['input_tokens'], generated_tokens=parsed.usage['output_tokens'])
        self.mark('glm_parse', **counts)

    def protocol_failure(self, phase, error, counts):
        if self.writer is not None:
            self.writer.record(phase, error, counts)

    def close(self):
        if self.writer is not None: self.writer.close()
