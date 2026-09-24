"""Drift memory injection for Qwen3.8-Flash-Next on vLLM, on top of the owner's handoff connector.

A request opts in with kv_transfer_params {"drift_inject": name, "drift_tokens": T, "drift_span_start": S,
"drift_own_start": O} and a cache_salt this server has not seen before. Positions [S, S + T) hold placeholder
tokens; once the engine step that computes them has run, every rank overwrites that span's full-attention K and V,
and the compressed selector keys when supplied, from <handoff_path>/drift-inject/<name>.npz. That step must not
reach past O, so no own token attends to placeholder rows. Recurrent layers keep the state the placeholders gave
them: only the attention path carries Drift memory. The owner's export behaviour is unchanged.
"""
from __future__ import annotations
import importlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import numpy as np
try:
    from qwen38_pages import HEAD_DIM, INDEX_DIM, attention_layers, compressed_layers, rotate, write_compressed, write_kv
    from drift_spans import SpanStep, SpanTracker
except ImportError:
    from drift.serving.qwen38_pages import HEAD_DIM, INDEX_DIM, attention_layers, compressed_layers, rotate, write_compressed, write_kv
    from drift.serving.drift_spans import SpanStep, SpanTracker

_base = importlib.import_module(os.environ.get("DRIFT_QWEN38_BASE", "qwen38_handoff_connector"))
logger = getattr(_base, "logger", None) or __import__("logging").getLogger(__name__)
RATIO = 4                                                           # tokens per compressed selector group


@dataclass
class DriftQwen38Metadata(_base.Qwen38HandoffMetadata):
    inject: list[SpanStep] = field(default_factory=list)


class DriftQwen38Connector(_base.Qwen38HandoffConnector):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._drift_dir = Path(self._output_path) / "drift-inject"
        self._spans = SpanTracker(RATIO, self._drift_mark)
        extra = self._kv_transfer_config
        text = getattr(self._vllm_config.model_config.hf_config, "text_config", None) or self._vllm_config.model_config.hf_config
        head_dim = int(getattr(text, "head_dim", HEAD_DIM))
        self._rope_theta = float(extra.get_from_extra_config("drift_rope_theta", getattr(text, "rope_theta", 0.0)))
        self._rotary_dim = int(extra.get_from_extra_config("drift_rotary_dim", round(head_dim * float(getattr(text, "partial_rotary_factor", 1.0)))))
        if head_dim != HEAD_DIM or self._rope_theta <= 0 or not 0 < self._rotary_dim <= HEAD_DIM:
            raise ValueError("Drift Qwen connector needs head_dim 256, a positive rope_theta and a rotary_dim within it")

    # Scheduler side ------------------------------------------------------
    def on_new_request(self, request: Any) -> None:
        super().on_new_request(request)
        for step in self._spans.admit(request):
            if step.kind == "tap" and not step.failed:
                step.failed = "this connector reads spans through the owner's qwen38_handoff export, not drift_tap"
            logger.info("drift %s %s: span %d..%d, own prompt from %d%s", step.kind, step.name, step.start, step.start + step.tokens,
                        step.own_start, f" (refused: {step.failed})" if step.failed else "")

    def build_connector_meta(self, scheduler_output: Any) -> Any:
        inherited = super().build_connector_meta(scheduler_output)
        ready = self._spans.advance(scheduler_output)
        return DriftQwen38Metadata(requests=list(getattr(inherited, "requests", []) or []), inject=ready)

    def request_finished(self, request: Any, block_ids: Any):
        self._spans.forget(request.request_id)
        return super().request_finished(request, block_ids)

    def request_finished_all_groups(self, request: Any, block_ids: Any):
        self._spans.forget(request.request_id)
        return super().request_finished_all_groups(request, block_ids)

    # Worker side ---------------------------------------------------------
    def wait_for_save(self):
        metadata = self._get_connector_metadata()
        for step in getattr(metadata, "inject", None) or []:
            rank = self._tp()[0]
            try:
                if step.failed:
                    raise RuntimeError(step.failed)
                self._drift_mark(step.name, f"rank{rank}.done", self._drift_write(step))
            except Exception as error:                              # the engine keeps serving; the client sees the error file
                logger.error("drift inject %s rank %d failed: %s", step.name, rank, error)
                self._drift_mark(step.name, f"rank{rank}.error", {"error": str(error)[:500]})
        if getattr(metadata, "requests", None):
            super().wait_for_save()

    def _drift_mark(self, name: str, kind: str, payload: dict) -> None:
        self._drift_dir.mkdir(parents=True, exist_ok=True)
        tmp = self._drift_dir / f".{name}.{kind}.tmp"
        tmp.write_text(json.dumps(payload))
        os.replace(tmp, self._drift_dir / f"{name}.{kind}")

    def _drift_write(self, step: SpanStep) -> dict:
        import torch
        started = time.perf_counter()
        entries = np.load(self._drift_dir / f"{step.name}.npz")
        rank, world = self._tp()
        attention, selector = attention_layers(self._kv_caches), compressed_layers(self._kv_caches)
        if not attention:
            raise RuntimeError("no full-attention layers registered")
        missing = [layer for layer in attention if f"v{layer}" not in entries.files or not {f"k{layer}", f"kr{layer}"} & set(entries.files)]
        if missing:
            raise RuntimeError(f"entries are missing layers {missing}; a partial memory is never written")
        with_selector = any(f"c{layer}" in entries.files for layer in attention)
        if with_selector and sorted(layer for layer in attention if f"c{layer}" in entries.files) != sorted(attention):
            raise RuntimeError("compressed selector keys must cover every layer or none")
        if with_selector and sorted(selector) != sorted(attention):
            raise RuntimeError("the compressed selector caches do not match the attention layers")
        positions = np.arange(step.start, step.start + step.tokens)
        plan = []
        for layer, name in sorted(attention.items()):              # validate everything before the first write
            part = self._kv_caches[name]
            part = part[0] if isinstance(part, (list, tuple)) else part
            heads = int(part.shape[1])
            if heads * world != entries[f"v{layer}"].shape[1]:
                raise RuntimeError(f"layer {layer}: {entries[f'v{layer}'].shape[1]} memory heads for {world} ranks of {heads}")
            own = slice(rank * heads, (rank + 1) * heads)
            values = np.asarray(entries[f"v{layer}"], dtype=np.float32)[:, own]
            if f"kr{layer}" in entries.files:
                keys = np.asarray(entries[f"kr{layer}"], dtype=np.float32)[:, own]
            else:
                keys = rotate(np.asarray(entries[f"k{layer}"], dtype=np.float32)[:, own], positions, self._rope_theta, self._rotary_dim)
            if keys.shape != (step.tokens, heads, HEAD_DIM) or values.shape != keys.shape or not (np.isfinite(keys).all() and np.isfinite(values).all()):
                raise RuntimeError(f"layer {layer}: memory {keys.shape} is not {step.tokens} finite rows of {heads} x {HEAD_DIM}")
            group = self._layer_to_group[name]
            if group >= len(step.block_ids):
                raise RuntimeError(f"layer {layer}: no blocks for cache group {group}")
            rows, _ = self._page_rows(part, step.block_ids[group])
            compressed = None
            if with_selector:
                compressed = np.asarray(entries[f"c{layer}"], dtype=np.float32)
                if compressed.shape != (step.tokens // RATIO, INDEX_DIM) or not np.isfinite(compressed).all():
                    raise RuntimeError(f"layer {layer}: selector keys {compressed.shape} are not {step.tokens // 4} finite rows")
            plan.append((layer, part, rows, keys, values, compressed))
        on_gpu = any(part.is_cuda for _, part, *_ in plan)
        if on_gpu:
            torch.cuda.synchronize()                                # the step that computed the placeholders has finished
        for layer, part, rows, keys, values, compressed in plan:
            write_kv(part, rows, step.start, keys, values)
            if compressed is not None:
                target = self._kv_caches[selector[layer]]
                target = target[0] if isinstance(target, (list, tuple)) else target
                selector_rows, _ = self._page_rows(target, step.block_ids[self._layer_to_group[selector[layer]]])
                write_compressed(target, selector_rows, step.start // RATIO, compressed)
        if on_gpu:
            torch.cuda.synchronize()
        report = {"tokens": step.tokens, "span_start": step.start, "layers": len(plan), "selector": with_selector,
                  "rank": rank, "seconds": round(time.perf_counter() - started, 4)}
        logger.info("drift inject %s: wrote %d positions into %d layers (selector %s) in %.3f s", step.name, step.tokens,
                    len(plan), with_selector, report["seconds"])
        return report
