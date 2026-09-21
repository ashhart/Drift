"""Drift KV connector for vLLM (KV connector v1 API). Imported only inside vLLM.

Launch (example):
  vllm serve MODEL --kv-transfer-config '{
    "kv_connector": "DriftConnector",
    "kv_connector_module_path": "drift.serving.vllm_connector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
        "exchange_root": "/var/lib/drift/exchange",
        "placeholder_token_id": 151654,
        "kv_layers": {"model.layers.3.self_attn.attn": 3, "model.layers.7.self_attn.attn": 7},
        "rope_theta": 10000000.0, "rotary_dim": 64, "layout": "blocks_first"
    }}'

Per request, clients pass `kv_transfer_params`:
  {"drift_session": "<uuid>", "drift_inject": "<name>" | null, "drift_tap": "<name>" | null}
- inject: the prompt begins with N placeholder tokens; `<exchange_root>/<session>/inject/<name>`
  holds N canonical entries per KV-bearing layer (N a multiple of the block size). They are
  written into the placeholder slots; vLLM skips computing those tokens.
- tap: after prefill, canonical entries for the request's REAL prompt tokens are written to
  `<exchange_root>/<session>/tap/<name>`.
Anything else is a stock request. The translator sidecar owns everything between tap and inject.

Qualified so far: logic against a mocked vLLM (tests/test_vllm_connector.py). NOT yet qualified on a
live server: hybrid recurrent-state initialisation for the placeholder span, tensor-parallel head
sharding, MTP speculative decoding, prefix-cache interaction (use a per-session cache_salt).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole
from drift.core.types import KV
from drift.serving.core import LayerSpec, RopeSpec, inject_request, plan_request, shard_heads, tap_layer
from drift.serving.exchange import load_entries, ready, save_entries

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.request import Request


@dataclass
class DriftReqMeta:
    request_id: str
    session: str
    inject: str | None
    tap: str | None
    placeholders: int
    slot_mapping: torch.Tensor         # slots for every prompt token, in order


@dataclass
class DriftMetadata(KVConnectorMetadata):
    requests: list[DriftReqMeta] = field(default_factory=list)


def _slots(block_ids: list[int], block_size: int, tokens: int) -> torch.Tensor:
    blocks = torch.tensor(block_ids, dtype=torch.long)
    slots = (torch.arange(block_size)[None, :] + blocks[:, None] * block_size).flatten()
    if slots.numel() < tokens:
        raise ValueError("allocated blocks do not cover the prompt")
    return slots[:tokens]


class DriftConnector(KVConnectorBase_V1):
    def __init__(self, vllm_config: Any, role: KVConnectorRole, kv_cache_config: Any = None):
        super().__init__(vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config)
        cfg = self._kv_transfer_config.get_from_extra_config
        self._block_size = int(vllm_config.cache_config.block_size)
        self._root = Path(cfg("exchange_root", "/tmp/drift-exchange"))
        self._placeholder = int(cfg("placeholder_token_id", -1))
        if self._placeholder < 0:
            raise ValueError("kv_connector_extra_config.placeholder_token_id is required")
        layers = cfg("kv_layers", None)
        if not isinstance(layers, dict) or not layers:
            raise ValueError("kv_connector_extra_config.kv_layers must map server layer names to model layer indices")
        theta, rotary_dim = cfg("rope_theta", None), cfg("rotary_dim", None)
        rope = RopeSpec(float(theta), int(rotary_dim)) if theta is not None and rotary_dim else None
        layout = cfg("layout", "blocks_first")
        cache_dtype = str(getattr(vllm_config.cache_config, "cache_dtype", "auto"))
        self._specs = [LayerSpec(name, int(index), layout, None if layout == "mla" else rope, cache_dtype)
                       for name, index in layers.items()]
        for spec in self._specs:
            spec.check()
        parallel = getattr(vllm_config, "parallel_config", None)
        self._tp_world = int(getattr(parallel, "tensor_parallel_size", 1) or 1)
        self._tp_rank = int(cfg("tp_rank_override", -1))
        self._pending: dict[str, DriftReqMeta] = {}          # scheduler side
        self._tapped: dict[str, dict[int, Any]] = {}             # worker side, per request

    # ------------------------------------------------------------------ scheduler side
    @staticmethod
    def _params(request: Any) -> dict:
        params = getattr(request, "kv_transfer_params", None) or {}
        return params if isinstance(params, dict) and params.get("drift_session") else {}

    def get_num_new_matched_tokens(self, request: "Request", num_computed_tokens: int) -> tuple[int | None, bool]:
        params = self._params(request)
        inject = params.get("drift_inject")
        if not inject:
            return 0, False
        directory = self._root / str(params["drift_session"]) / "inject"
        if not ready(directory, inject):
            raise ValueError("drift_inject names entries that are not in the exchange")
        import json
        available = int(json.loads((directory / f"{inject}.json").read_text())["tokens"])
        plan = plan_request(list(request.prompt_token_ids or []), self._placeholder, available,
                            already_computed=num_computed_tokens, block_size=self._block_size)
        return plan.matched, False

    def update_state_after_alloc(self, request: "Request", blocks: Any, num_external_tokens: int) -> None:
        params = self._params(request)
        if not params:
            return
        placeholders = num_external_tokens if params.get("drift_inject") else 0
        self._pending[request.request_id] = DriftReqMeta(
            request.request_id, str(params["drift_session"]), params.get("drift_inject"),
            params.get("drift_tap"), placeholders, torch.empty(0, dtype=torch.long))

    def build_connector_meta(self, scheduler_output: "SchedulerOutput") -> KVConnectorMetadata:
        meta = DriftMetadata()
        for new_req in scheduler_output.scheduled_new_reqs:
            pending = self._pending.pop(new_req.req_id, None)
            if pending is None:
                continue
            tokens = len(new_req.prompt_token_ids or [])
            pending.slot_mapping = _slots(list(new_req.block_ids[0]), self._block_size, tokens)
            meta.requests.append(pending)
        return meta

    def request_finished(self, request: "Request", block_ids: Any) -> tuple[bool, dict | None]:
        self._pending.pop(request.request_id, None)
        return False, None

    # ------------------------------------------------------------------ worker side
    def _rank(self) -> int:
        if self._tp_rank >= 0:
            return self._tp_rank
        try:
            from vllm.distributed import get_tensor_model_parallel_rank
            return int(get_tensor_model_parallel_rank())
        except Exception:
            return 0

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, DriftMetadata):
            return
        caches = {}
        for spec in self._specs:
            layer = forward_context.no_compile_layers.get(spec.name)
            cache = getattr(layer, "kv_cache", None)
            if cache is None:
                raise ValueError(f"server has no KV cache for configured layer {spec.name}")
            caches[spec.name] = cache[0] if isinstance(cache, (list, tuple)) else cache
        for request in metadata.requests:
            if not request.inject or request.placeholders == 0:
                continue
            entries, _, _ = load_entries(self._root / request.session / "inject", request.inject)
            if self._tp_world > 1:
                entries = {k: (shard_heads(v, self._rank(), self._tp_world) if isinstance(v, KV) else v) for k, v in entries.items()}
            inject_request(self._specs, caches, request.slot_mapping[: request.placeholders], entries)

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor, attn_metadata: Any, **kwargs: Any) -> None:
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, DriftMetadata):
            return
        spec = next((s for s in self._specs if s.name == layer_name), None)
        if spec is None:
            return
        for request in metadata.requests:
            if not request.tap:
                continue
            real = request.slot_mapping[request.placeholders:]
            if real.numel() == 0:
                continue
            positions = torch.arange(request.placeholders, request.placeholders + real.numel())
            self._tapped.setdefault(request.request_id, {})[spec.model_layer] = tap_layer(spec, kv_layer, real, positions)

    def wait_for_save(self) -> None:
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, DriftMetadata):
            return
        for request in metadata.requests:
            entries = self._tapped.pop(request.request_id, None)
            if not request.tap or not entries:
                continue
            if set(entries) != {s.model_layer for s in self._specs}:
                raise ValueError("tap is incomplete across KV-bearing layers; refusing a partial publication")
            tokens = request.slot_mapping.numel() - request.placeholders
            name = request.tap if self._tp_world == 1 else f"{request.tap}.rank{self._rank()}"
            save_entries(self._root / request.session / "tap", name, entries,
                         torch.arange(tokens), {"request": request.request_id[:64], "placeholders": request.placeholders,
                                                "tp_rank": self._rank(), "tp_world": self._tp_world})
