from __future__ import annotations

import hashlib
import json
from pathlib import Path

from drift.adapters.catalog import installed_adapter_ids
from drift.taps.artifacts import artifact_hashes
from drift.taps.inventory import file_sha256, inspect_checkpoint
from drift.taps.project import shared_pool
from drift.taps.validation import admits_model, qualification, translator


INVENTORY_KEYS = {
    "checkpoint",
    "model_type",
    "config_sha256",
    "weights_manifest_sha256",
    "tokenizer_manifest_sha256",
    "runtime_lock",
    "runtime_lock_sha256",
    "weights",
    "tokenizer",
}


def translator_hash(path: Path) -> str:
    manifest = path / "translator.json"
    weights = path / "translator.safetensors"
    return hashlib.sha256(manifest.read_bytes() + weights.read_bytes()).hexdigest()


def member_inputs_intact(member: dict) -> bool:
    current = inspect_checkpoint(Path(member["checkpoint"]), Path(member["runtime_lock"]))
    if any(current.get(key) != member.get(key) for key in INVENTORY_KEYS):
        return False
    qualification = member.get("qualification")
    if qualification and file_sha256(Path(qualification["path"])) != qualification["sha256"]:
        return False
    translator = member.get("translator")
    if translator and translator_hash(Path(translator["path"])) != translator["sha256"]:
        return False
    return True


def recorded_members(root: Path) -> dict | None:
    try:
        return {
            "source": json.loads((root / "inventory" / "source.json").read_text()),
            "target": json.loads((root / "inventory" / "target.json").read_text()),
        }
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def member_ready(member: dict, installed: set[str]) -> bool:
    if not member.get("weights_manifest_sha256") or not member.get("tokenizer_manifest_sha256"):
        return False
    adapter = member.get("adapter")
    if not isinstance(adapter, str) or adapter not in installed or not admits_model(adapter, str(member.get("model_type"))):
        return False
    qualification_record, translator_record = member.get("qualification"), member.get("translator")
    if not isinstance(qualification_record, dict) or not isinstance(translator_record, dict):
        return False
    try:
        _, qual_reason = qualification(Path(qualification_record["path"]), member, adapter)
        _, trans_reason = translator(Path(translator_record["path"]))
    except (OSError, ValueError, KeyError, TypeError):
        return False
    return qual_reason is None and trans_reason is None


def evidence_passed(models: dict, installed: set[str]) -> bool:
    if models["source"].get("host") != models["target"].get("host"):
        return False
    pool, blocker = shared_pool(models)
    if blocker or pool is None:
        return False
    return all(member_ready(member, installed) for member in models.values())


def tap_status(root: Path) -> dict:
    manifest_path = root / "tap.json"
    if root.is_symlink() or not root.is_dir() or manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("tap path must contain a regular tap.json")
    try:
        manifest = json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError("tap.json must be valid JSON") from error
    expected = manifest.get("artifacts")
    artifacts_intact = isinstance(expected, dict) and expected == artifact_hashes(root)
    recorded = recorded_members(root)
    inventory_matches = recorded is not None and recorded == manifest.get("models")
    try:
        inputs_intact = inventory_matches and all(member_inputs_intact(member) for member in recorded.values())
        passed = inputs_intact and evidence_passed(recorded, installed_adapter_ids())
    except (KeyError, OSError, TypeError, ValueError):
        inputs_intact, passed = False, False
    recomputed = "PASSED" if passed else "BLOCKED"
    claim_matches = manifest.get("status") == recomputed
    intact = artifacts_intact and inputs_intact and inventory_matches and claim_matches
    status = recomputed if intact else "INVALID"
    return {
        "status": status,
        "tap": manifest.get("name"),
        "path": str(root.resolve()),
        "artifacts_intact": artifacts_intact,
        "inputs_intact": inputs_intact,
        "blockers": manifest.get("blockers", []),
    }
