from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from drift.adapters.catalog import ADAPTER_MODEL_TYPES
from drift.taps.inventory import file_sha256, regular_file


NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,62}$")
ADAPTER_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,62}/[a-z0-9][a-z0-9._-]{0,31}$")


def validate_name(value: str, label: str) -> str:
    pattern = ADAPTER_PATTERN if label == "adapter" else NAME_PATTERN
    if not pattern.fullmatch(value):
        raise ValueError(f"invalid {label} name")
    return value


def _whole(value: object, minimum: int) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        return None
    return value


def qualification(path: Path | None, inventory: dict, adapter_id: str) -> tuple[dict | None, str | None]:
    if path is None:
        return None, "real-weight qualification is missing"
    path = regular_file(path, "qualification")
    try:
        report = json.loads(path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError("qualification must be valid JSON") from error
    level = _whole(report.get("level"), 2)
    if report.get("passed") is not True or level is None:
        return None, "real-weight qualification has not passed"
    identity = {
        "config_sha256": inventory.get("config_sha256"),
        "weights_manifest_sha256": inventory.get("weights_manifest_sha256"),
        "tokenizer_manifest_sha256": inventory.get("tokenizer_manifest_sha256"),
        "runtime_lock_sha256": inventory.get("runtime_lock_sha256"),
        "model_type": inventory.get("model_type"),
        "adapter": adapter_id,
    }
    for key, expected in identity.items():
        if not expected or report.get(key) != expected:
            return None, f"qualification {key} does not match the checkpoint"
    return {"path": str(path), "sha256": file_sha256(path), "level": level}, None


def _pool(pool: object):
    from drift.translate.pool import PoolFormat

    if not isinstance(pool, dict):
        return None
    levels, width, fingerprint = _whole(pool.get("levels"), 1), _whole(pool.get("width"), 1), pool.get("fingerprint")
    if pool.get("version") != "pool.v1" or levels is None or width is None or not isinstance(fingerprint, str) or not fingerprint:
        return None
    return PoolFormat("pool.v1", levels, width, fingerprint)


def translator(path: Path | None) -> tuple[dict | None, str | None]:
    if path is None:
        return None, "translator is missing"
    if path.is_symlink() or not path.is_dir():
        raise ValueError("translator must be a directory without symlinks")
    manifest_path = regular_file(path / "translator.json", "translator manifest")
    weights_path = regular_file(path / "translator.safetensors", "translator weights")
    try:
        data = json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError("translator manifest must be valid JSON") from error
    if data.get("format") != 1 or file_sha256(weights_path) != data.get("sha256"):
        return None, "translator artifact is invalid"
    pool = _pool(data.get("pool"))
    if pool is None:
        return None, "translator pool format is invalid"
    try:
        from drift.translate.pool import load_translator

        loaded = load_translator(path, pool)
    except Exception:
        return None, "translator artifact is invalid"
    if loaded.member != data.get("member") or loaded.pool.fingerprint != pool.fingerprint:
        return None, "translator artifact is invalid"
    digest = hashlib.sha256(manifest_path.read_bytes() + weights_path.read_bytes()).hexdigest()
    return {"path": str(path.resolve()), "sha256": digest, "pool": data["pool"], "member": loaded.member}, None


def admits_model(adapter_id: str, model_type: str) -> bool:
    expected = ADAPTER_MODEL_TYPES.get(adapter_id)
    return expected is None or expected == model_type
