"""Model registry (docs/guides/ADAPTERS.md): one entry per checkpoint with its adapter id,
pinned hashes, qualification evidence and fitted translators. Entries are written
only from evidence files; nothing here fabricates a hash."""
from __future__ import annotations
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from drift.adapters.catalog import known_adapter_ids


@dataclass(frozen=True)
class ModelEntry:
    model_id: str
    adapter: str
    checkpoint_repo: str | None
    checkpoint_revision: str | None
    config_sha256: str | None
    weights_manifest_sha256: str | None
    tokenizer_manifest_sha256: str | None
    runtime_lock_sha256: str
    quantization: str
    host: str
    qualification_level: int                # 0 none, 1 tiny-config parity, 2 real-weight parity, 3 cross-runtime
    qualification_sha256: str | None

    def check(self) -> None:
        if self.adapter not in known_adapter_ids():
            raise ValueError("unknown adapter id")
        if not 0 <= self.qualification_level <= 3:
            raise ValueError("bad qualification level")
        if self.qualification_level >= 1 and not self.qualification_sha256:
            raise ValueError("a qualification level needs its evidence hash")
        if self.qualification_level >= 2 and not (self.config_sha256 and self.weights_manifest_sha256):
            raise ValueError("real-weight qualification needs pinned config and weight hashes")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_entry(root: Path, entry: ModelEntry) -> Path:
    entry.check()
    directory = root / entry.model_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "qualification").mkdir(exist_ok=True)
    (directory / "translators").mkdir(exist_ok=True)
    path = directory / "model.json"
    path.write_text(json.dumps(asdict(entry), indent=2) + "\n")
    return path


def read_entry(root: Path, model_id: str) -> ModelEntry:
    entry = ModelEntry(**json.loads((root / model_id / "model.json").read_text()))
    entry.check()
    return entry


def usable_for(entry: ModelEntry, stage: str) -> bool:
    """Which stages an entry may enter: real-weight stages need level >= 2."""
    needed = {"tiny": 1, "M0": 2, "M1": 2, "M2": 2, "M3": 2, "M4": 3, "M5": 3}[stage]
    return entry.qualification_level >= needed
