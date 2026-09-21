from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Callable


Loader = Callable[[str, str], object]


@dataclass(frozen=True)
class AdapterRecord:
    adapter_id: str
    installed: bool
    provider: str


def _qwen(checkpoint: str, device: str) -> object:
    from drift.adapters.qwen4_exp import Qwen4ExpAdapter

    return Qwen4ExpAdapter.from_local(checkpoint, device)


def _glm(checkpoint: str, device: str) -> object:
    from drift.adapters.glm5_next import Glm5NextAdapter

    return Glm5NextAdapter.from_local(checkpoint, device)


BUILTIN_LOADERS: dict[str, Loader] = {
    "qwen4_exp/torch": _qwen,
    "glm5_next/torch": _glm,
}

# model_type strings the adapter modules themselves reject on any other value.
ADAPTER_MODEL_TYPES = {
    "drift_toy/torch": "drift_toy",
    "qwen3/torch": "qwen3",
    "llama/torch": "llama",
    "qwen4_exp/torch": "qwen4_exp_text",
    "qwen4_exp/mlx": "qwen4_exp_text",
    "glm5_next/torch": "glm5_next_text",
    "glm5_next/mlx": "glm5_next_text",
}

KNOWN_ADAPTERS = {
    "drift_toy/torch",
    "qwen3/torch",
    "llama/torch",
    "qwen4_exp/torch",
    "glm5_next/torch",
    "qwen4_exp/mlx",
    "glm5_next/mlx",
}


def plugin_entry_points() -> dict[str, metadata.EntryPoint]:
    return {entry.name: entry for entry in metadata.entry_points(group="drift.adapters")}


def adapter_records() -> list[AdapterRecord]:
    plugins = plugin_entry_points()
    ids = KNOWN_ADAPTERS | set(plugins)
    records = []
    for adapter_id in sorted(ids):
        if adapter_id in BUILTIN_LOADERS:
            records.append(AdapterRecord(adapter_id, True, "built-in"))
        elif adapter_id in plugins:
            records.append(AdapterRecord(adapter_id, True, plugins[adapter_id].value))
        else:
            records.append(AdapterRecord(adapter_id, False, "declared"))
    return records


def installed_adapter_ids() -> set[str]:
    return {record.adapter_id for record in adapter_records() if record.installed}


def known_adapter_ids() -> set[str]:
    return {record.adapter_id for record in adapter_records()}


def load_adapter(adapter_id: str, checkpoint: str | Path, device: str) -> object:
    if adapter_id in BUILTIN_LOADERS:
        return BUILTIN_LOADERS[adapter_id](str(checkpoint), device)
    entry = plugin_entry_points().get(adapter_id)
    if entry is None:
        raise RuntimeError(f"no installed loader for adapter {adapter_id}")
    loader = entry.load()
    return loader(str(checkpoint), device)
