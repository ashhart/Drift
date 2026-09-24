"""Record the input of GLM's recurrent layers at flagged prompt positions, to fit a translated recurrent state.

Every KDA layer gets one mixed 4096-wide input per token, the same on each tensor-parallel rank, and derives its
keys, values, decay and learning rate from it with its own projections. A class-level wrapper on the layer's
forward keeps the flagged rows while a flagged step runs. The wrapper sits inside compiled code, so it records
only on a server started with --enforce-eager; a compiled server leaves the records empty and the step fails.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import re
import numpy as np

_LAYER = re.compile(r"(?:^|\.)layers\.(\d+)\.self_attn$")


@dataclass
class HiddenCapture:
    active: bool = False
    rows: object = None                                             # step-relative row indices, a tensor on the layer's device
    records: dict = field(default_factory=dict)                     # layer index -> [rows, hidden] tensor

    def clear(self) -> None:
        self.active, self.rows, self.records = False, None, {}


CAPTURE = HiddenCapture()


def _wrap(original):
    def forward(self, hidden_states, positions, *args, **kwargs):
        match = _LAYER.search(getattr(self, "prefix", ""))
        if CAPTURE.active and match and getattr(self, "prefix", "").startswith("language_model."):
            rows = CAPTURE.rows.to(hidden_states.device)
            CAPTURE.records[int(match.group(1))] = hidden_states.index_select(0, rows).detach().clone()
        return original(self, hidden_states, positions, *args, **kwargs)
    forward._drift_hidden_capture = True
    return forward


def install(cls=None) -> bool:
    """Wrap the KDA layer's forward once per process; False when vLLM's class cannot be imported."""
    if cls is None:
        try:
            from vllm.models.glm5next.nvidia.kda import Glm5NextLinearAttention as cls
        except ImportError:
            return False
    if not getattr(cls.forward, "_drift_hidden_capture", False):
        cls.forward = _wrap(cls.forward)
    return True


def save(path: Path, records: dict, positions: list[int]) -> None:
    """One npz per request: positions and h{layer} float16 [rows, hidden] for every recorded layer."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "wb") as handle:
        np.savez(handle, positions=np.asarray(positions, np.int32),
                 **{f"h{layer}": value.float().cpu().numpy().astype(np.float16) for layer, value in sorted(records.items())})
    tmp.replace(path)
