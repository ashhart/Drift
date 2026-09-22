"""GLM cache injection and live receiver hooks for the owner’s vLLM connector."""
from __future__ import annotations
import importlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import numpy as np
try:
    from glm_prefill_boundary import boundary, guard_initial
except ImportError:
    from drift.serving.glm_prefill_boundary import boundary, guard_initial

_base = importlib.import_module(os.environ.get("DRIFT_GLM53_BASE", "glm53_handoff_connector"))
try:
    from glm53_handoff import pack_fp8_ds_mla      # deployed flat next to this file
except ImportError:                                               # inside the Drift repository
    from drift.serving.glm53_handoff import pack_fp8_ds_mla
try:
    from glm_rank_failure import any_rank_failed
    from glm_worker_save import save_step
except ImportError:
    from drift.serving.glm_rank_failure import any_rank_failed
    from drift.serving.glm_worker_save import save_step
try:
    from glm_allocation_lifecycle import invalidate_preempted, live_blocks, merge_blocks
except ImportError:
    from drift.serving.glm_allocation_lifecycle import invalidate_preempted, live_blocks, merge_blocks
try:
    from live_receiver_glm import LiveReceiverError, apply_live_step
except ImportError:
    from drift.serving.live_receiver_glm import LiveReceiverError, apply_live_step
logger = getattr(_base, "logger", None) or __import__("logging").getLogger(__name__)
try:
    from live_tap_finish import finish_stream, processed_frontier
except ImportError:
    from drift.serving.live_tap_finish import finish_stream, processed_frontier
_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,96}$")
_ATTN = re.compile(r"^language_model\.model\.layers\.(\d+)\.self_attn\.attn$")
LATENT, PACKED = 512, 528


@dataclass
class InjectStep:
    request_id: str
    name: str
    tokens: int
    own_start: int = 0
    block_ids: tuple[tuple[int, ...], ...] = ()
    failed: str = ""


@dataclass
class LiveStep:
    """One live engine step with reserved incoming memory and verified outgoing positions."""
    request_id: str
    name: str
    reserve: int
    reserve_start: int = 0                # the reserved span is positions [reserve_start, reserve_start + reserve)
    blocks: tuple[tuple[int, ...], ...] = ()   # the request's blocks per cache group (the worker picks its MLA latent group)
    before: int = 0                       # scheduled cursor, which can include unfinished async work
    after: int = 0
    apply: tuple[int, ...] = ()           # sequence numbers of memory files every rank must write after this step
    tap: bool = True
    failed: str = ""
    failed_code: str = ""                 # allowlisted numeric diagnostics for scheduler-side refusals (see LiveReceiverError.fields)
    failed_details: tuple = ()
    verified: int = 0


@dataclass
class DriftMetadata(_base.Glm53HandoffMetadata):
    inject: list[InjectStep] = field(default_factory=list)
    live: list[LiveStep] = field(default_factory=list)


class DriftGlm53Connector(_base.Glm53HandoffConnector):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._tp_pending: dict[str, InjectStep] = {}              # scheduler: requests whose placeholders are not computed yet
        self._tp_dir = Path(self._output_path) / "tp-inject"
        self._live: dict[str, dict] = {}                          # scheduler: live sessions by request id
        self._live_worker: dict[str, dict] = {}                   # worker: per session {filled, tapped}
        self._live_in, self._live_out = Path(self._output_path) / "tp-live-in", Path(self._output_path) / "tp-live-out"
        self._live_tap_every = int(self._kv_transfer_config.get_from_extra_config("drift_tap_every", 8))

    # Scheduler side ------------------------------------------------------
    def on_new_request(self, request: Any) -> None:
        super().on_new_request(request)
        params = getattr(request, "kv_transfer_params", None) or {}
        session = params.get("drift_session")
        if session:
            reserve = int(params.get("drift_reserve", 0) or 0)
            prompt_len = len(getattr(request, "prompt_token_ids", None) or [])
            first = int(params.get("drift_reserve_start", 0) or 0)
            state = {"name": str(session), "reserve": reserve, "start": first, "tap": bool(params.get("drift_tap", True)), "blocks": (), "next_seq": 0, "failed": ""}
            if not _NAME.match(state["name"]):
                state["name"], state["failed"] = "invalid-name", "drift_session must match [A-Za-z0-9_.-]{1,96}"
            elif not (first >= 0 and reserve >= 0 and first + reserve < prompt_len):
                state["failed"] = f"reserved span {first}..{first + reserve} must lie inside the prompt ({prompt_len} tokens) and leave own tokens after it"
            reuse = params.get("drift_prefix_reuse")
            if reuse is not None and reuse is not True:
                state["failed"] = state["failed"] or "drift_prefix_reuse must be the literal true"
            if reuse is True and not state["failed"] and getattr(request, "skip_writing_prefix_cache", False) is not True:
                state["failed"], state["code"], state["details"] = "prefix reuse requires the engine's resolved no-store flag", "NO_STORE", ()
            if reuse is True and not state["failed"]:
                state["reuse"] = True                             # reads allowed before the span; guard at first scheduling (docs/GLM_PREFIX_REUSE.md)
            else:
                request.skip_reading_prefix_cache = True          # the reserved span must live in THIS request's blocks
            self._live[request.request_id] = state
            state['request'] = request
            try:
                stop = boundary(params)
                if stop is not None:
                    if stop >= prompt_len: raise ValueError('own input must follow the prefill boundary')
                    state['prefill_boundary'] = stop
            except ValueError:
                state['failed'] = 'invalid initial prefill boundary'
            logger.info("drift live %s: reserved span %d..%d of %d prompt tokens", state["name"], first, first + reserve, prompt_len)
        name = params.get("drift_inject")
        if not name:
            return
        tokens = int(params.get("drift_tokens", 0) or 0)
        step = InjectStep(request_id=request.request_id, name=str(name), tokens=tokens, own_start=int(params.get("drift_own_start", tokens) or tokens))
        prompt_len = len(getattr(request, "prompt_token_ids", None) or [])
        if not _NAME.match(step.name):
            step.name, step.failed = "invalid-name", "drift_inject must match [A-Za-z0-9_.-]{1,96}"
        elif not 0 < step.tokens <= step.own_start < prompt_len:
            step.failed = f"need 0 < drift_tokens ({step.tokens}) <= drift_own_start ({step.own_start}) < prompt length ({prompt_len}): leave at least one own token"
        request.skip_reading_prefix_cache = True                  # placeholders must be computed in THIS request's blocks
        self._tp_pending[request.request_id] = step
        logger.info("drift inject %s: %d memory positions, own prompt from %d, %d prompt tokens", step.name, step.tokens, step.own_start, prompt_len)

    def build_connector_meta(self, scheduler_output: Any) -> Any:
        inherited = super().build_connector_meta(scheduler_output)
        invalidate_preempted(self, scheduler_output)
        ready: list[InjectStep] = []

        def merge(step: InjectStep, additions: Any, replace: bool = False) -> None:
            step.block_ids = merge_blocks(step.block_ids, additions, replace)

        def progress(request_id: str, step: InjectStep, before: int, scheduled: int) -> None:
            after = before + scheduled
            if after < step.tokens and not step.failed:
                return                                            # placeholders still being computed
            if not step.failed and after > step.own_start:
                step.failed = (f"engine step covered positions {before}..{after}, past the placeholder boundary [{step.tokens}, {step.own_start}]; "
                               "own tokens would have attended to placeholder latents")
            ready.append(self._tp_pending.pop(request_id))

        for new in scheduler_output.scheduled_new_reqs:
            step = self._tp_pending.get(new.req_id)
            if step is not None:
                merge(step, new.block_ids, replace=True)
                progress(new.req_id, step, new.num_computed_tokens, scheduler_output.num_scheduled_tokens.get(new.req_id, 0))
        cached = scheduler_output.scheduled_cached_reqs
        for index, request_id in enumerate(cached.req_ids):
            step = self._tp_pending.get(request_id)
            if step is not None:
                merge(step, cached.new_block_ids[index], replace=request_id in cached.resumed_req_ids)
                progress(request_id, step, cached.num_computed_tokens[index], scheduler_output.num_scheduled_tokens.get(request_id, 0))
        live: list[LiveStep] = []

        def live_step(request_id: str, additions: Any, replace: bool, before: int, scheduled: int) -> None:
            state = self._live[request_id]
            if "hit" not in state:                                 # first scheduling: `before` is vLLM's own prefix-cache hit
                state["hit"] = int(before)
                if state.get("reuse") and before > state["start"] and not state["failed"]:
                    state["failed"], state["code"], state["details"] = "prefix cache hit reaches the reserved span", "PREFIX_HIT", (("hit", int(before)), ("start", int(state["start"])))
            elif replace and not state["failed"]:                  # rescheduled after preemption: the span's written contents are gone
                state["failed"], state["code"], state["details"] = "request was preempted; a fresh session is required", "PREEMPTED", (("computed", int(before)),)
            if any((self._live_out / state["name"]).glob("error.rank*")):
                state["failed"] = "live rank failed; a fresh session is required"
            live_blocks(state, additions, replace)
            verified = 0
            if not state['failed'] and state['tap']:
                try:
                    verified = processed_frontier(state['request'], before)
                except ValueError:
                    state['failed'] = 'live tap scheduler frontier is not settled'
            apply = []
            guard_initial(state, before, before+scheduled,
                          (self._live_in / state['name'] / '000000.npz').is_file())
            if not state["failed"] and before + scheduled >= state["start"] + state["reserve"]:   # the reserved span exists from this step on
                while len(apply) < 4 and (self._live_in / state["name"] / f"{state['next_seq']:06d}.npz").exists():
                    apply.append(state["next_seq"]); state["next_seq"] += 1
            live.append(LiveStep(request_id, state["name"], state["reserve"], state["start"], state["blocks"], int(before), int(before + scheduled), tuple(apply), state["tap"], state["failed"],
                                 state.get("code", ""), state.get("details", ()), verified))

        for new in scheduler_output.scheduled_new_reqs:
            if new.req_id in self._live:
                live_step(new.req_id, new.block_ids, True, new.num_computed_tokens, scheduler_output.num_scheduled_tokens.get(new.req_id, 0))
        for index, request_id in enumerate(cached.req_ids):
            if request_id in self._live:
                live_step(request_id, cached.new_block_ids[index], request_id in cached.resumed_req_ids, cached.num_computed_tokens[index],
                          scheduler_output.num_scheduled_tokens.get(request_id, 0))
        return DriftMetadata(requests=list(getattr(inherited, "requests", []) or []), inject=ready, live=live)

    def _live_finished(self, request: Any) -> None:
        state = self._live.pop(request.request_id, None)
        if state is not None:                                     # scheduler process, head host: tell the reader of the taps that no more will come
            try:
                folder = self._live_out / state["name"]
                finish_stream(folder, state, request)
            except OSError:
                logger.error("drift live %s: cannot write the finished marker", state["name"])

    def request_finished(self, request: Any, block_ids: Any):
        self._live_finished(request)
        self._tp_pending.pop(request.request_id, None)
        return super().request_finished(request, block_ids)

    def request_finished_all_groups(self, request: Any, block_ids: Any):
        self._live_finished(request)
        self._tp_pending.pop(request.request_id, None)
        return super().request_finished_all_groups(request, block_ids)

    # Worker side ---------------------------------------------------------
    def _drift_rank_failed(self, failed):
        return any_rank_failed(failed, self._tp()[1])

    def wait_for_save(self):
        metadata = self._get_connector_metadata()
        for step in getattr(metadata, "inject", None) or []:
            save_step(self, step, False, logger)
        for step in getattr(metadata, "live", None) or []:
            save_step(self, step, True, logger)
        if getattr(metadata, "requests", None):
            super().wait_for_save()

    def _live_targets(self) -> dict:
        targets = {}
        for layer_name, cache in self._kv_caches.items():
            match = _ATTN.match(layer_name)
            if match and self._layer_to_group.get(layer_name) == self._mla_group:
                part = cache[0] if isinstance(cache, (list, tuple)) else cache
                targets[int(match.group(1))] = part
        if not targets:
            raise RuntimeError("no target MLA attention layers registered")
        return targets

    def _live_index(self, part: Any, blocks: tuple, start: int, stop: int):
        import torch
        group = self._mla_group
        if group is None or group < 0 or group >= len(blocks):
            raise LiveReceiverError('BLOCK_GROUP', group=-1 if group is None else int(group), groups=len(blocks))
        rows, _ = self._page_rows(part, blocks[group])
        slots = int(part.shape[1])
        if start < 0 or stop < start or stop > len(rows) * slots:
            raise LiveReceiverError('PAGE_RANGE', start=int(start), stop=int(stop), pages=len(rows), slots=slots)
        positions = torch.arange(start, stop, device=part.device)
        return torch.as_tensor(rows, device=part.device, dtype=torch.long)[positions // slots], positions % slots

    def _live_step(self, step: LiveStep) -> None:
        apply_live_step(self, step)

    def _tp_report(self, name: str, rank: int, kind: str, payload: dict) -> None:
        self._tp_dir.mkdir(parents=True, exist_ok=True)
        tmp = self._tp_dir / f".{name}.rank{rank}.{kind}.tmp"
        tmp.write_text(json.dumps(payload))
        os.replace(tmp, self._tp_dir / f"{name}.rank{rank}.{kind}")

    def _drift_write(self, step: InjectStep) -> dict:
        import torch
        started = time.perf_counter()
        group = self._mla_group
        if group is None or group >= len(step.block_ids):
            raise RuntimeError("no MLA latent group or no blocks for the request")
        entries = np.load(self._tp_dir / f"{step.name}.npz")
        targets = {}
        for layer_name, cache in self._kv_caches.items():
            match = _ATTN.match(layer_name)
            if match and self._layer_to_group.get(layer_name) == group:
                targets[int(match.group(1))] = cache
        if not targets:
            raise RuntimeError("no target MLA attention layers registered")
        missing = [layer for layer in targets if f"l{layer}" not in entries.files]
        if missing:
            raise RuntimeError(f"entries are missing layers {missing}; a partial memory is never written")
        on_gpu = any((c[0] if isinstance(c, (list, tuple)) else c).is_cuda for c in targets.values())
        if on_gpu:
            torch.cuda.synchronize()                              # the step that computed the placeholders has finished
        written = 0
        for layer, cache in sorted(targets.items()):
            latents = np.asarray(entries[f"l{layer}"], dtype=np.float32)
            if latents.ndim != 2 or latents.shape[1] != LATENT or not 0 < latents.shape[0] <= step.tokens:
                raise RuntimeError(f"layer {layer}: entries {latents.shape}, expected [1..{step.tokens}, {LATENT}]")
            latents = latents[np.arange(step.tokens) % latents.shape[0]]          # tile: NoPE attention, so copies are position-free (a log(copies) prior)
            part = cache[0] if isinstance(cache, (list, tuple)) else cache
            if part.dtype != torch.uint8 or part.ndim != 3 or part.shape[-1] < PACKED:
                raise RuntimeError(f"layer {layer}: cache {tuple(part.shape)} {part.dtype} is not fp8_ds_mla pages")
            rows, _ = self._page_rows(part, step.block_ids[group])
            slots = int(part.shape[1])
            if len(rows) * slots < step.tokens:
                raise RuntimeError("the request's pages do not cover the placeholder positions")
            need = -(-step.tokens // slots)
            index = torch.as_tensor(rows[:need], device=part.device, dtype=torch.long)
            pages = part.index_select(0, index)                   # [need, slots, 656], a copy
            packed = torch.from_numpy(pack_fp8_ds_mla(latents)).to(part.device)
            pages.view(-1, pages.shape[-1])[: step.tokens, :PACKED] = packed        # the rope slot (unused, NoPE) is left as computed
            part.index_copy_(0, index, pages)
            written += 1
        if on_gpu:
            torch.cuda.synchronize()
        report = {"tokens": step.tokens, "layers": written, "seconds": round(time.perf_counter() - started, 4)}
        logger.info("drift inject %s: wrote %d positions into %d layers in %.3f s", step.name, step.tokens, written, report["seconds"])
        return report
