from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from uuid import uuid4

from drift.adapters.catalog import installed_adapter_ids
from drift.taps.artifacts import artifact_hashes, write_json
from drift.taps.inventory import inspect_checkpoint
from drift.taps.validation import admits_model, qualification, translator, validate_name


def inspect_member(role: str, spec: dict, installed: set[str]) -> tuple[dict, list[str]]:
    inventory = inspect_checkpoint(spec["checkpoint"], spec["runtime_lock"])
    adapter_id = validate_name(spec["adapter"], "adapter")
    blockers = []
    if inventory["weights_manifest_sha256"] is None:
        blockers.append(f"{role} checkpoint weights are missing")
    if inventory["tokenizer_manifest_sha256"] is None:
        blockers.append(f"{role} tokenizer artifacts are missing")
    if adapter_id not in installed:
        blockers.append(f"{role} adapter {adapter_id} is not installed")
    if not admits_model(adapter_id, inventory["model_type"]):
        blockers.append(f"{role} adapter {adapter_id} does not admit model type {inventory['model_type']}")
    qualified, reason = qualification(spec.get("qualification"), inventory, adapter_id)
    if reason:
        blockers.append(f"{role} {reason}")
    translated, reason = translator(spec.get("translator"))
    if reason:
        blockers.append(f"{role} {reason}")
    member = {
        **inventory,
        "adapter": adapter_id,
        "host": spec["host"],
        "quantization": spec.get("quantization", "none"),
        "qualification": qualified,
        "translator": translated,
    }
    return member, blockers


def shared_pool(models: dict) -> tuple[dict | None, str | None]:
    source = models["source"]["translator"]
    target = models["target"]["translator"]
    if source is None or target is None:
        return None, None
    if source["pool"] != target["pool"]:
        return None, "translators use different pool formats"
    if not isinstance(source["member"], str) or not isinstance(target["member"], str):
        return None, "translators need member names"
    if source["member"] == target["member"]:
        return None, "translators need distinct member names"
    return source["pool"], None


def run_manifest(name: str, models: dict, pool: dict) -> dict:
    return {
        "schema": "drift.run.v2",
        "kind": "real",
        "stage": "M1",
        "session_uuid": str(uuid4()),
        "setup_tokenizer": "bytes",
        "registry_root": None,
        "device": "cpu",
        "pool": pool,
        "window": {"sinks": 4, "recent": 128},
        "members": [
            {
                "name": models["source"]["translator"]["member"],
                "writer": 1,
                "mail_writer": 11,
                "registry_entry": f"{name}-source",
                "checkpoint_dir": models["source"]["checkpoint"],
                "translator_dir": models["source"]["translator"]["path"],
                "gated": True,
            },
            {
                "name": models["target"]["translator"]["member"],
                "writer": 2,
                "mail_writer": 12,
                "registry_entry": f"{name}-target",
                "checkpoint_dir": models["target"]["checkpoint"],
                "translator_dir": models["target"]["translator"]["path"],
                "gated": True,
            },
        ],
    }


def registry_entry(name: str, role: str, member: dict) -> dict:
    return {
        "model_id": f"{name}-{role}",
        "adapter": member["adapter"],
        "checkpoint_repo": None,
        "checkpoint_revision": None,
        "config_sha256": member["config_sha256"],
        "weights_manifest_sha256": member["weights_manifest_sha256"],
        "tokenizer_manifest_sha256": member["tokenizer_manifest_sha256"],
        "runtime_lock_sha256": member["runtime_lock_sha256"],
        "quantization": member["quantization"],
        "host": member["host"],
        "qualification_level": member["qualification"]["level"],
        "qualification_sha256": member["qualification"]["sha256"],
    }


def create_tap(name: str, source: dict, target: dict, output: Path) -> dict:
    validate_name(name, "tap")
    if output.exists() or output.is_symlink():
        raise ValueError("output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        installed = installed_adapter_ids()
        source_model, source_blockers = inspect_member("source", source, installed)
        target_model, target_blockers = inspect_member("target", target, installed)
        models = {"source": source_model, "target": target_model}
        pool, pool_blocker = shared_pool(models)
        blockers = source_blockers + target_blockers + ([pool_blocker] if pool_blocker else [])
        if source_model["host"] != target_model["host"]:
            blockers.append("remote two-host coordination is not qualified")
        write_json(temporary / "inventory" / "source.json", source_model)
        write_json(temporary / "inventory" / "target.json", target_model)
        if not blockers:
            run = run_manifest(name, models, pool)
            run["registry_root"] = str((output.resolve() / "registry"))
            write_json(temporary / "run.json", run)
            for role, member in models.items():
                write_json(temporary / "registry" / f"{name}-{role}" / "model.json", registry_entry(name, role, member))
        status = "PASSED" if not blockers else "BLOCKED"
        tap = {
            "schema": "drift.tap.v1",
            "name": name,
            "status": status,
            "models": models,
            "pool": pool,
            "blockers": blockers,
            "artifacts": artifact_hashes(temporary),
        }
        write_json(temporary / "tap.json", tap)
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary)
        raise
    return {"status": status, "tap": name, "path": str(output.resolve()), "blockers": blockers}

