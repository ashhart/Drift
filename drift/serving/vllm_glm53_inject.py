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
try:
    import glm_query_capture as query_capture
except ImportError:
    import drift.serving.glm_query_capture as query_capture
try:
    import glm_hidden_capture as hidden_capture
except ImportError:
    import drift.serving.glm_hidden_capture as hidden_capture
try:
    from glm_state_compute import layer_modules
except ImportError:
    from drift.serving.glm_state_compute import layer_modules
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
    state_blob: str = ""                  # handoff id whose exported recurrent state each rank writes at the reserve's end
    state_rows: bool = False              # advance that state by the memory's per-layer inputs in state.npz


@dataclass
class CaptureStep:
    """This step's share of a capture's prompt rows; every step holding rows must hold the request alone."""
    request_id: str
    name: str
    positions: tuple[int, ...] = ()
    before: int = 0                                                  # the step's first computed position
    failed: str = ""
    final: bool = True                                               # the capture's last rows: write the file after this step


@dataclass
class DriftMetadata(_base.Glm53HandoffMetadata):
    inject: list[InjectStep] = field(default_factory=list)
    live: list[LiveStep] = field(default_factory=list)
    capture: list[CaptureStep] = field(default_factory=list)


class DriftGlm53Connector(_base.Glm53HandoffConnector):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._tp_pending: dict[str, InjectStep] = {}              # scheduler: requests whose placeholders are not computed yet
        self._tp_dir = Path(self._output_path) / "tp-inject"
        self._live: dict[str, dict] = {}                          # scheduler: live sessions by request id
        self._live_worker: dict[str, dict] = {}                   # worker: per session {filled, tapped}
        self._live_in, self._live_out = Path(self._output_path) / "tp-live-in", Path(self._output_path) / "tp-live-out"
        self._live_tap_every = int(self._kv_transfer_config.get_from_extra_config("drift_tap_every", 8))
        self._query_capture = bool(self._kv_transfer_config.get_from_extra_config("drift_query_capture", False))
        self._hidden_capture = bool(self._kv_transfer_config.get_from_extra_config("drift_hidden_capture", False))
        self._capture_pending: dict[str, CaptureStep] = {}               # scheduler: capture requests not yet computed
        self._capture_dir = Path(self._output_path) / "drift-capture"

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
            blob = params.get("drift_state_blob")
            if blob is not None and not state["failed"]:
                if not isinstance(blob, str) or not _NAME.match(blob):
                    state["failed"] = "drift_state_blob must match [A-Za-z0-9_.-]{1,96}"
                elif state.get("prefill_boundary") != first + reserve:
                    state["failed"] = "a recurrent state needs drift_prefill_boundary at the reserve's end"
                else:
                    state["state_blob"] = blob
            rows = params.get("drift_state_rows")
            if rows is not None and not state["failed"]:
                if rows is not True or "state_blob" not in state:
                    state["failed"] = "drift_state_rows must be the literal true, with drift_state_blob naming the head's state"
                else:
                    state["state_rows"] = True
            logger.info("drift live %s: reserved span %d..%d of %d prompt tokens", state["name"], first, first + reserve, prompt_len)
        self._admit_capture(request, params)
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
                                 state.get("code", ""), state.get("details", ()), verified, state.get("state_blob", ""),
                                 state.get("state_rows", False)))

        for new in scheduler_output.scheduled_new_reqs:
            if new.req_id in self._live:
                live_step(new.req_id, new.block_ids, True, new.num_computed_tokens, scheduler_output.num_scheduled_tokens.get(new.req_id, 0))
        for index, request_id in enumerate(cached.req_ids):
            if request_id in self._live:
                live_step(request_id, cached.new_block_ids[index], request_id in cached.resumed_req_ids, cached.num_computed_tokens[index],
                          scheduler_output.num_scheduled_tokens.get(request_id, 0))
        return DriftMetadata(requests=list(getattr(inherited, "requests", []) or []), inject=ready, live=live,
                             capture=self._capture_ready(scheduler_output))

    def _admit_capture(self, request: Any, params: dict) -> None:
        name = params.get("drift_capture")
        if not name:
            return
        rows = params.get("drift_capture_rows")
        prompt_len = len(getattr(request, "prompt_token_ids", None) or [])
        step = CaptureStep(request.request_id, str(name))
        if not (self._query_capture or self._hidden_capture):
            step.failed = "this server was not started with drift_query_capture or drift_hidden_capture"
        elif not _NAME.match(step.name):
            step.name, step.failed = "invalid-name", "drift_capture must match [A-Za-z0-9_.-]{1,96}"
        elif not (isinstance(rows, list) and 0 < len(rows) <= 512 and all(type(v) is int for v in rows)
                  and rows == sorted(set(rows)) and 0 <= rows[0] and rows[-1] < prompt_len):
            step.failed = f"drift_capture_rows must be 1..512 increasing positions inside the prompt ({prompt_len} tokens)"
        else:
            step.positions = tuple(rows)
        request.skip_reading_prefix_cache = True                  # every captured row must be computed in this request's step
        self._capture_pending[request.request_id] = step

    def _capture_ready(self, scheduler_output: Any) -> list[CaptureStep]:
        ready = []
        total = getattr(scheduler_output, "total_num_scheduled_tokens", None)
        cached = scheduler_output.scheduled_cached_reqs
        scheduled = [(new.req_id, new.num_computed_tokens) for new in scheduler_output.scheduled_new_reqs]
        scheduled += [(request_id, cached.num_computed_tokens[i]) for i, request_id in enumerate(cached.req_ids)]
        for request_id, before in scheduled:
            pending = self._capture_pending.get(request_id)
            if pending is None:
                continue
            after = before + scheduler_output.num_scheduled_tokens.get(request_id, 0)
            rows = tuple(p for p in pending.positions if before <= p < after)
            if not pending.failed and pending.positions and pending.positions[0] < before:
                pending.failed = f"capture row {pending.positions[0]} was computed in a step the connector did not see"
            if not pending.failed and not rows:
                continue                                          # the rows come in a later step
            if not pending.failed and total is not None and total != after - before:
                pending.failed = "a capture step must hold this request alone"
            left = tuple(p for p in pending.positions if p >= after)
            ready.append(CaptureStep(request_id, pending.name, rows, before, pending.failed, final=bool(pending.failed) or not left))
            if pending.failed or not left:
                self._capture_pending.pop(request_id)
            else:
                pending.positions = left
        return ready

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
        self._capture_pending.pop(request.request_id, None)
        return super().request_finished(request, block_ids)

    def request_finished_all_groups(self, request: Any, block_ids: Any):
        self._live_finished(request)
        self._tp_pending.pop(request.request_id, None)
        self._capture_pending.pop(request.request_id, None)
        return super().request_finished_all_groups(request, block_ids)

    # Worker side ---------------------------------------------------------
    def _drift_rank_failed(self, failed):
        return any_rank_failed(failed, self._tp()[1])

    def register_kv_caches(self, kv_caches: dict[str, Any]):
        super().register_kv_caches(kv_caches)
        if self._query_capture:
            logger.info("drift query capture: MLAAttention wrapper %s", "installed" if query_capture.install() else "unavailable")
        if self._hidden_capture:
            logger.info("drift hidden capture: KDA wrapper %s", "installed" if hidden_capture.install() else "unavailable")

    def start_load_kv(self, forward_context: Any, **kwargs: Any) -> None:
        super().start_load_kv(forward_context, **kwargs)
        if not getattr(self, "_kda_modules", None) and forward_context is not None:
            self._kda_modules = layer_modules(getattr(forward_context, "no_compile_layers", None))
        steps = getattr(self._get_connector_metadata(), "capture", None) or []
        query_capture.CAPTURE.clear()
        hidden_capture.CAPTURE.clear()
        live = [step for step in steps if not step.failed]
        if self._query_capture and live:
            query_capture.CAPTURE.layers = frozenset(self._capture_layers().values())
            query_capture.CAPTURE.active = True
        if self._hidden_capture and live and self._tp()[0] == 0:           # every rank holds the same layer input
            import torch
            hidden_capture.CAPTURE.rows = torch.as_tensor([p - live[0].before for p in live[0].positions], dtype=torch.long)
            hidden_capture.CAPTURE.active = True

    def _capture_layers(self) -> dict[int, str]:
        names = {}
        for layer_name in self._kv_caches:
            match = _ATTN.match(layer_name)
            if match and self._layer_to_group.get(layer_name) == self._mla_group:
                names[int(match.group(1))] = layer_name
        return names

    def _finish_captures(self, steps: list) -> None:
        rank = self._tp()[0]
        parts = self.__dict__.setdefault("_capture_parts", {})       # worker: capture name -> rows gathered so far
        try:
            for step in steps:
                try:
                    if step.failed:
                        raise RuntimeError(step.failed)
                    part = parts.setdefault(step.name, {"positions": [], "queries": {}, "hidden": {}, "scale": None, "heads": None})
                    part["positions"] += list(step.positions)
                    if self._query_capture:
                        for layer, name in sorted(self._capture_layers().items()):
                            record = query_capture.CAPTURE.records.get(name)
                            if record is None:
                                raise RuntimeError(f"layer {layer} recorded no query this step")
                            latent, part["scale"] = query_capture.latent_queries(record, [p - step.before for p in step.positions])
                            part["queries"].setdefault(layer, []).append(latent)
                            part["heads"] = latent.shape[1]
                    if self._hidden_capture and rank == 0:
                        expected = len([g for g in self._layer_to_group.values() if g in self._state_groups])
                        records = hidden_capture.CAPTURE.records
                        if len(records) != expected:
                            raise RuntimeError(f"{len(records)} of {expected} recurrent layers recorded their input; the server must run eagerly")
                        for layer, value in records.items():
                            part["hidden"].setdefault(layer, []).append(value)
                    if not step.final:
                        continue
                    parts.pop(step.name, None)
                    report = {"rows": len(part["positions"])}
                    if self._query_capture:
                        layers = {layer: np.concatenate(values) for layer, values in part["queries"].items()}
                        query_capture.save(self._capture_dir / f"{step.name}.rank{rank}.npz", layers, part["scale"], rank * part["heads"], part["positions"])
                        report.update(layers=len(layers), heads=part["heads"])
                    if self._hidden_capture and rank == 0:
                        import torch
                        records = {layer: torch.cat(values) for layer, values in part["hidden"].items()}
                        hidden_capture.save(self._capture_dir / f"{step.name}.hidden.npz", records, part["positions"])
                        report.update(hidden_layers=len(records))
                    self._capture_mark(step.name, rank, "done", report)
                except Exception as error:
                    parts.pop(step.name, None)
                    logger.error("drift capture %s rank %d failed: %s", step.name, rank, error)
                    self._capture_mark(step.name, rank, "error", {"error": str(error)[:500]})
        finally:
            query_capture.CAPTURE.clear()
            hidden_capture.CAPTURE.clear()

    def _capture_mark(self, name: str, rank: int, kind: str, payload: dict) -> None:
        self._capture_dir.mkdir(parents=True, exist_ok=True)
        tmp = self._capture_dir / f".{name}.rank{rank}.{kind}.tmp"
        tmp.write_text(json.dumps(payload))
        os.replace(tmp, self._capture_dir / f"{name}.rank{rank}.{kind}")

    def wait_for_save(self):
        metadata = self._get_connector_metadata()
        if getattr(metadata, "capture", None):
            self._finish_captures(metadata.capture)
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
