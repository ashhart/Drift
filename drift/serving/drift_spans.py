"""Scheduler-side tracking of Drift spans on vLLM: validate a request's span, follow its blocks, release it once computed.

kv_transfer_params name a span [drift_span_start, drift_span_start + drift_tokens) and ask for "drift_inject" (write
memory over placeholders before own tokens at drift_own_start see them) and/or "drift_tap" (read the span back once it is
computed). Injecting requests need a cache_salt the server has not seen, so their blocks never serve another request's
prefix. A step that is not safe to apply carries a reason instead of being dropped, so every rank reports it.
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Any, Callable
try:
    from glm_allocation_lifecycle import merge_blocks
except ImportError:
    from drift.serving.glm_allocation_lifecycle import merge_blocks

_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,96}$")


@dataclass
class SpanStep:
    request_id: str
    kind: str                                                       # "inject" or "tap"
    name: str
    tokens: int = 0
    start: int = 0
    own_start: int = 0
    block_ids: tuple[tuple[int, ...], ...] = ()
    failed: str = ""
    hit: int = -1                                                   # vLLM's prefix-cache hit at first scheduling


def _int(params: dict, key: str, default: int = 0) -> int:
    value = params.get(key, default)
    if type(value) is not int:
        raise ValueError(f"{key} must be an integer")
    return value


class SpanTracker:
    def __init__(self, alignment: int, mark: Callable[[str, str, dict], None]):
        self.alignment, self.mark = alignment, mark
        self.pending: dict[str, list[SpanStep]] = {}
        self.applied: dict[str, str] = {}                           # request id -> memory already written
        self.salts: set[str] = set()

    def admit(self, request: Any) -> list[SpanStep]:
        params = getattr(request, "kv_transfer_params", None) or {}
        kinds = [kind for kind in ("inject", "tap") if params.get(f"drift_{kind}")]
        if not kinds:
            return []
        prompt_len = len(getattr(request, "prompt_token_ids", None) or [])
        salt = getattr(request, "cache_salt", None)
        steps = []
        for kind in kinds:
            step = SpanStep(request.request_id, kind, str(params[f"drift_{kind}"]))
            try:
                step.tokens, step.start = _int(params, "drift_tokens"), _int(params, "drift_span_start")
                step.own_start = _int(params, "drift_own_start", step.start + step.tokens)
            except ValueError as error:
                step.failed = str(error)
            if step.failed:
                pass
            elif not _NAME.match(step.name):
                step.name, step.failed = "invalid-name", f"drift_{kind} must match [A-Za-z0-9_.-]{{1,96}}"
            elif not 0 <= step.start < step.start + step.tokens <= step.own_start <= prompt_len or (kind == "inject" and step.own_start == prompt_len):
                step.failed = f"need 0 <= span start < span end <= own start, inside the prompt ({prompt_len} tokens)"
            elif step.start % self.alignment or step.tokens % self.alignment:
                step.failed = f"the span must start and end on a multiple of {self.alignment} tokens"
            elif kind == "inject" and (not salt or salt in self.salts):
                step.failed = "an injecting request needs a cache_salt this server has not seen, so its blocks are never shared"
            steps.append(step)
        if salt:
            self.salts.add(salt)
        if "inject" in kinds:
            request.skip_reading_prefix_cache = True                # the span must be computed in this request's blocks
        self.pending[request.request_id] = steps
        return steps

    def advance(self, scheduler_output: Any) -> list[SpanStep]:
        for request_id in getattr(scheduler_output, "preempted_req_ids", ()) or ():
            for step in self.pending.get(request_id, ()):
                step.block_ids = ()
            name = self.applied.pop(request_id, None)
            if name is not None:                                    # the recomputed span would hold placeholders again
                self.mark(name, "preempted", {"request_id": request_id})
        ready: list[SpanStep] = []

        def progress(request_id: str, additions: Any, replace: bool, before: int, scheduled: int) -> None:
            after, waiting = before + scheduled, []
            for step in self.pending.pop(request_id):
                step.block_ids = merge_blocks(step.block_ids, additions, replace)
                if step.hit < 0 or replace:
                    step.hit = int(before)
                if after < step.start + step.tokens and not step.failed:
                    waiting.append(step)                            # the span is still being computed
                    continue
                if not step.failed and step.kind == "inject" and after > step.own_start:
                    step.failed = (f"engine step covered positions {before}..{after}, past own start {step.own_start}; "
                                   "own tokens would have attended to placeholder rows")
                if not step.failed and step.kind == "inject" and step.hit > step.start:
                    step.failed = f"a prefix-cache hit of {step.hit} tokens covered the span start {step.start}"
                if not step.failed and step.kind == "inject":
                    self.applied[request_id] = step.name
                ready.append(step)
            if waiting:
                self.pending[request_id] = waiting

        for new in scheduler_output.scheduled_new_reqs:
            if new.req_id in self.pending:
                progress(new.req_id, new.block_ids, True, new.num_computed_tokens, scheduler_output.num_scheduled_tokens.get(new.req_id, 0))
        cached = scheduler_output.scheduled_cached_reqs
        for index, request_id in enumerate(cached.req_ids):
            if request_id in self.pending:
                progress(request_id, cached.new_block_ids[index], request_id in cached.resumed_req_ids,
                         cached.num_computed_tokens[index], scheduler_output.num_scheduled_tokens.get(request_id, 0))
        return sorted(ready, key=lambda step: step.kind != "inject")   # a request's write lands before its own tap reads

    def forget(self, request_id: str) -> None:
        self.pending.pop(request_id, None)
        self.applied.pop(request_id, None)
