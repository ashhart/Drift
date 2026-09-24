"""Record GLM's latent-space queries at flagged prompt positions, the targets for distilling the reverse translator.

GLM's MLA attention has no positional part, so a query reads its latent cache as softmax(scale * (q W_UK) . c) c. A
class-level wrapper on vLLM's MLAAttention.forward_impl keeps each target layer's per-rank query while a flagged step
runs; afterwards the flagged rows become latent queries q W_UK [rows, heads, kv_lora_rank]. Each rank holds its own
heads, so every rank writes its own file.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np


@dataclass
class QueryCapture:
    active: bool = False
    layers: frozenset = frozenset()
    records: dict = field(default_factory=dict)                     # layer name -> (q, W_UK_T, scale, nope dims)

    def clear(self) -> None:
        self.active, self.records = False, {}


CAPTURE = QueryCapture()


def _wrap(original):
    def forward_impl(self, q, *args, **kwargs):
        if CAPTURE.active and getattr(self, "layer_name", None) in CAPTURE.layers:
            CAPTURE.records[self.layer_name] = (q.detach().clone(), self.W_UK_T, float(self.scale), int(self.qk_nope_head_dim))
        return original(self, q, *args, **kwargs)
    forward_impl._drift_query_capture = True
    return forward_impl


def install(cls=None) -> bool:
    """Wrap MLAAttention.forward_impl once per process; False when vLLM's class cannot be imported."""
    if cls is None:
        try:
            from vllm.model_executor.layers.attention.mla_attention import MLAAttention as cls
        except ImportError:
            return False
    if not getattr(cls.forward_impl, "_drift_query_capture", False):
        cls.forward_impl = _wrap(cls.forward_impl)
    return True


def latent_queries(record, rows: list[int]) -> tuple[np.ndarray, float]:
    """The step's rows as float32 latent queries [len(rows), heads, kv_lora_rank], with the softmax scale."""
    import torch
    q, w_uk_t, scale, nope = record
    if w_uk_t is None or q.ndim != 3 or not rows or min(rows) < 0 or max(rows) >= q.shape[0] or w_uk_t.shape[:2] != (q.shape[1], nope):
        raise ValueError(f"query rows {rows[:3]}... or W_UK_T {None if w_uk_t is None else tuple(w_uk_t.shape)} do not fit q {tuple(q.shape)}")
    index = torch.as_tensor(rows, device=q.device)
    latent = torch.einsum("nhp,hpl->nhl", q.index_select(0, index)[:, :, :nope].float(), w_uk_t.float())
    return latent.cpu().numpy(), scale


def save(path: Path, layers: dict[int, np.ndarray], scale: float, head_offset: int, positions: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "wb") as handle:
        np.savez(handle, scale=np.float32(scale), head_offset=np.int32(head_offset), positions=np.asarray(positions, np.int32),
                 **{f"l{layer}": value.astype(np.float16) for layer, value in layers.items()})
    tmp.replace(path)
