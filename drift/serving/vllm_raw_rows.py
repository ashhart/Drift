"""Drift connector that moves a span's cache entries as raw bytes, for models whose memory lives in compressed MLA caches.

Built for DeepSeek V4 on vLLM, where a span far behind the sliding window survives only as compressed entries: one per
compress_ratio tokens in each compressed MLA cache and in the sparse indexer's cache. Memory tensors are the caches whose
spec class is listed in extra config "drift_spec_classes" (default MLAAttentionSpec); sliding-window and compressor-state
caches are left alone. MLA keeps one latent head, replicated on every tensor-parallel rank, so every rank writes the same
rows and taps its own copy. Rows keep the cache's own byte format, so an identity round trip needs no decoding; memory
from another model would need rows packed in that format first. Scheduling rules are the shared ones in drift_spans.
"""
from __future__ import annotations
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1, KVConnectorMetadata, SupportsHMA
try:
    from drift_spans import SpanStep, SpanTracker
    from raw_rows import load, read_rows, save, span_entries, write_rows
except ImportError:
    from drift.serving.drift_spans import SpanStep, SpanTracker
    from drift.serving.raw_rows import load, read_rows, save, span_entries, write_rows
logger = __import__("logging").getLogger("vllm.drift_raw_rows")


@dataclass
class DriftRowsMetadata(KVConnectorMetadata):
    steps: list[SpanStep] = field(default_factory=list)


class DriftRawRowConnector(KVConnectorBase_V1, SupportsHMA):
    def __init__(self, vllm_config: Any, role: Any, kv_cache_config: Any = None):
        super().__init__(vllm_config, role, kv_cache_config)
        extra = self._kv_transfer_config
        self._path = Path(extra.get_from_extra_config("drift_path", "/dev/shm/drift-rows"))
        classes = tuple(extra.get_from_extra_config("drift_spec_classes", ["MLAAttentionSpec"]))
        self._num_blocks = int(getattr(kv_cache_config, "num_blocks", 0) or 0)
        self._memory: dict[str, tuple[int, int, int]] = {}         # layer -> (group, ratio, entries per logical block)
        for group_id, group in enumerate(getattr(kv_cache_config, "kv_cache_groups", ()) or ()):
            per_layer = getattr(group.kv_cache_spec, "kv_cache_specs", None)
            for name in group.layer_names:
                spec = per_layer[name] if isinstance(per_layer, dict) and name in per_layer else group.kv_cache_spec
                if type(spec).__name__ in classes:
                    ratio = max(1, int(getattr(spec, "compress_ratio", 1) or 1))
                    self._memory[name] = (group_id, ratio, int(spec.block_size) // ratio)
        alignment = max([ratio for _, ratio, _ in self._memory.values()] or [1])
        self._spans = SpanTracker(alignment, self._mark)
        self._kv_caches: dict[str, Any] = {}
        logger.info("drift raw rows: %d memory tensors, ratios %s, span alignment %d, path %s", len(self._memory),
                    sorted({ratio for _, ratio, _ in self._memory.values()}), alignment, self._path)

    # Scheduler side ------------------------------------------------------
    def on_new_request(self, request: Any) -> None:
        for step in self._spans.admit(request):
            logger.info("drift %s %s: span %d..%d, own prompt from %d%s", step.kind, step.name, step.start, step.start + step.tokens,
                        step.own_start, f" (refused: {step.failed})" if step.failed else "")

    def get_num_new_matched_tokens(self, request: Any, num_computed_tokens: int) -> tuple[int | None, bool]:
        return 0, False

    def update_state_after_alloc(self, request: Any, blocks: Any, num_external_tokens: int) -> None:
        return None

    def build_connector_meta(self, scheduler_output: Any) -> KVConnectorMetadata:
        return DriftRowsMetadata(steps=self._spans.advance(scheduler_output))

    def request_finished(self, request: Any, block_ids: Any) -> tuple[bool, dict | None]:
        self._spans.forget(request.request_id)
        return False, None

    def request_finished_all_groups(self, request: Any, block_ids: Any) -> tuple[bool, dict | None]:
        self._spans.forget(request.request_id)
        return False, None

    # Worker side ---------------------------------------------------------
    def register_kv_caches(self, kv_caches: dict[str, Any]) -> None:
        self._kv_caches = kv_caches
        missing = sorted(set(self._memory) - set(kv_caches))
        logger.info("drift raw rows: %d caches registered%s", len(kv_caches), f", missing memory tensors {missing}" if missing else "")

    def start_load_kv(self, forward_context: Any, **kwargs: Any) -> None:
        return None

    def wait_for_layer_load(self, layer_name: str) -> None:
        return None

    def save_kv_layer(self, layer_name: str, kv_layer: Any, attn_metadata: Any, **kwargs: Any) -> None:
        return None                                                 # rows are read from the registered caches after the step

    def wait_for_save(self) -> None:
        rank = self._rank()
        for step in getattr(self._get_connector_metadata(), "steps", None) or []:
            try:
                if step.failed:
                    raise RuntimeError(step.failed)
                report = self._write(step) if step.kind == "inject" else self._tap(step, rank)
                self._mark(step.name, f"{step.kind}.rank{rank}.done", report)
            except Exception as error:                              # the engine keeps serving; the client sees the error file
                logger.error("drift %s %s rank %d failed: %s", step.kind, step.name, rank, error)
                self._mark(step.name, f"{step.kind}.rank{rank}.error", {"error": str(error)[:500]})

    @staticmethod
    def _rank() -> int:
        try:
            from vllm.distributed import get_tensor_model_parallel_rank
            return int(get_tensor_model_parallel_rank())
        except Exception:
            return int(os.environ.get("RANK", "0"))

    def _mark(self, name: str, kind: str, payload: dict) -> None:
        folder = self._path / "drift-marks"
        folder.mkdir(parents=True, exist_ok=True)
        tmp = folder / f".{name}.{kind}.tmp"
        tmp.write_text(json.dumps(payload))
        os.replace(tmp, folder / f"{name}.{kind}")

    def _part(self, name: str) -> Any:
        part = self._kv_caches[name]
        return part[0] if isinstance(part, (list, tuple)) else part

    def _pages(self, name: str, part: Any, step: SpanStep) -> tuple[list[int], int]:
        group, ratio, per_block = self._memory[name]
        if group >= len(step.block_ids):
            raise RuntimeError(f"{name}: no blocks for cache group {group}")
        rows = int(part.shape[0])
        split = rows // self._num_blocks if self._num_blocks and rows % self._num_blocks == 0 else 1
        if self._num_blocks and rows not in (self._num_blocks, split * self._num_blocks):
            raise RuntimeError(f"{name}: cache dim0 {rows} is not a multiple of {self._num_blocks} blocks")
        if int(part.shape[1]) * split != per_block:
            raise RuntimeError(f"{name}: {part.shape[1]} slots x {split} pages per block, expected {per_block} entries per block")
        return [b * split + p for b in step.block_ids[group] for p in range(split)], ratio

    def _plan(self, step: SpanStep) -> list[tuple[str, Any, list[int], int, int, int]]:
        plan = []
        for name in sorted(self._memory):
            if name not in self._kv_caches:
                raise RuntimeError(f"memory tensor {name} is not registered")
            part = self._part(name)
            rows, ratio = self._pages(name, part, step)
            first, count = span_entries(step.start, step.tokens, ratio)
            plan.append((name, part, rows, ratio, first, count))
        if not plan:
            raise RuntimeError("no memory tensors registered")
        return plan

    def _tap(self, step: SpanStep, rank: int) -> dict:
        import torch
        started = time.perf_counter()
        plan = self._plan(step)
        if any(part.is_cuda for _, part, *_ in plan):
            torch.cuda.synchronize()
        rows = {name: (ratio, read_rows(part, pages, first, count)) for name, part, pages, ratio, first, count in plan}
        save(self._path / "drift-tap" / f"{step.name}.rank{rank}.npz", rows)
        return {"tensors": len(rows), "bytes": int(sum(r.nbytes for _, r in rows.values())), "seconds": round(time.perf_counter() - started, 4)}

    def _write(self, step: SpanStep) -> dict:
        import torch
        started = time.perf_counter()
        plan = self._plan(step)
        memory = load(self._path / "drift-inject" / f"{step.name}.npz")
        missing = sorted(name for name, *_ in plan if name not in memory)
        if missing or len(memory) != len(plan):
            raise RuntimeError(f"memory names {len(memory)} tensors against {len(plan)}, missing {missing[:3]}; a partial memory is never written")
        for name, part, _, ratio, _, count in plan:                  # validate everything before the first write
            got_ratio, data = memory[name]
            if got_ratio != ratio or data.shape[0] != count or data.shape[1] != part[0, 0].numel() * part.element_size():
                raise RuntimeError(f"{name}: memory rows {data.shape} at ratio {got_ratio} do not fit {count} entries at ratio {ratio}")
        on_gpu = any(part.is_cuda for _, part, *_ in plan)
        if on_gpu:
            torch.cuda.synchronize()                                # the step that computed the placeholders has finished
        for name, part, pages, _, first, _ in plan:
            write_rows(part, pages, first, memory[name][1])
        if on_gpu:
            torch.cuda.synchronize()
        return {"tensors": len(plan), "tokens": step.tokens, "span_start": step.start, "seconds": round(time.perf_counter() - started, 4)}
