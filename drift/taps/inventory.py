from __future__ import annotations

import hashlib
import json
from pathlib import Path


WEIGHT_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth"}
TOKENIZER_NAMES = {
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "vocab.txt",
    "merges.txt",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def regular_file(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular file without symlinks")
    return path.resolve()


def manifest(files: list[Path], root: Path) -> tuple[str | None, list[dict]]:
    if not files:
        return None, []
    rows = []
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
        regular_file(path, "checkpoint artifact")
        relative = path.relative_to(root).as_posix()
        item = {"path": relative, "bytes": path.stat().st_size, "sha256": file_sha256(path)}
        rows.append(item)
        digest.update(json.dumps(item, sort_keys=True, separators=(",", ":")).encode())
    return digest.hexdigest(), rows


def inspect_checkpoint(checkpoint: Path, runtime_lock: Path) -> dict:
    if checkpoint.is_symlink() or not checkpoint.is_dir():
        raise ValueError("checkpoint must be a directory without symlinks")
    checkpoint = checkpoint.resolve()
    lock = regular_file(runtime_lock, "runtime lock")
    config_path = regular_file(checkpoint / "config.json", "checkpoint config")
    try:
        config = json.loads(config_path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError("checkpoint config must be valid JSON") from error
    model_type = config.get("model_type")
    if not isinstance(model_type, str) or not model_type:
        raise ValueError("checkpoint config needs a model_type")
    weights = [path for path in checkpoint.iterdir() if path.is_file() and path.suffix in WEIGHT_SUFFIXES]
    tokenizers = [path for path in checkpoint.iterdir() if path.is_file() and path.name in TOKENIZER_NAMES]
    weights_hash, weight_rows = manifest(weights, checkpoint)
    tokenizer_hash, tokenizer_rows = manifest(tokenizers, checkpoint)
    return {
        "checkpoint": str(checkpoint),
        "model_type": model_type,
        "config_sha256": file_sha256(config_path),
        "weights_manifest_sha256": weights_hash,
        "tokenizer_manifest_sha256": tokenizer_hash,
        "runtime_lock": str(lock),
        "runtime_lock_sha256": file_sha256(lock),
        "weights": weight_rows,
        "tokenizer": tokenizer_rows,
    }
