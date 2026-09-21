"""Member builders for the service. `toy_members` is the reference; `from_manifest`
constructs real adapters from registry entries and loads their fitted translators,
refusing anything unqualified for the stage."""
from __future__ import annotations
from pathlib import Path
from uuid import UUID
import torch
from drift.adapters.catalog import load_adapter
from drift.adapters.toy import DenseAdapter
from drift.core.attention import Gate
from drift.core.pool import PoolBank
from drift.mailbox.stream import MailWriter
from drift.registry import read_entry, usable_for
from drift.runtime.decoder import FrozenDecoder
from drift.runtime.toy import ToyModel
from drift.runtime.worker import Worker
from drift.translate.pool import Layout, PoolFormat, Translator, load_translator


def toy_members(manifest: dict) -> dict[str, Worker]:
    session = UUID(manifest["session_uuid"])
    fmt = PoolFormat("pool.v1", int(manifest["pool"]["levels"]), int(manifest["pool"]["width"]), manifest["pool"]["fingerprint"])
    members = manifest["members"]
    writers = {m["name"]: int(m["writer"]) for m in members}
    mail_ids = {m["name"]: int(m["mail_writer"]) for m in members}
    workers = {}
    for spec in members:
        name = spec["name"]
        torch.manual_seed(int(spec.get("seed", 0)))
        adapter = DenseAdapter(FrozenDecoder(ToyModel(**spec.get("toy", {}))))
        d = adapter.descriptor
        level_map = {layer: layer % fmt.levels for layer in d.kv_layers[:fmt.levels]}
        translator = Translator(name, Layout("kv_split", d.kv_heads, d.head_dim), level_map, fmt)
        with torch.no_grad():
            for p in translator.parameters():
                p.mul_(0.3)
        others = {w for n, w in writers.items() if n != name}
        mail_others = {w for n, w in mail_ids.items() if n != name}
        workers[name] = Worker(
            name, writers[name], session, adapter, translator,
            PoolBank(session, translator, writers[name], others, sinks=int(manifest["window"]["sinks"]), recent=int(manifest["window"]["recent"])),
            override=float(spec.get("override", 0.5)), mail_writer=mail_ids[name],
            mail_bank=PoolBank(session, translator, mail_ids[name], mail_others, sinks=0, recent=int(manifest["window"]["recent"]), multiple_per_epoch=True),
            mail=MailWriter(adapter.decoder.model.config.hidden_size, max_tokens=int(manifest.get("mail_max_tokens", 8))))
    return workers


def from_manifest(manifest: dict) -> dict[str, Worker]:
    """Real members: registry entries must be qualified for the manifest's stage."""
    if manifest.get("kind") == "toy":
        return toy_members(manifest)
    registry = Path(manifest["registry_root"])
    stage = manifest.get("stage", "M0")
    session = UUID(manifest["session_uuid"])
    fmt = PoolFormat("pool.v1", int(manifest["pool"]["levels"]), int(manifest["pool"]["width"]), manifest["pool"]["fingerprint"])
    members = manifest["members"]
    writers = {m["name"]: int(m["writer"]) for m in members}
    mail_ids = {m["name"]: int(m["mail_writer"]) for m in members}
    device = manifest.get("device", "cpu")
    workers = {}
    for spec in members:
        name = spec["name"]
        entry = read_entry(registry, spec["registry_entry"])
        if not usable_for(entry, stage):
            raise RuntimeError(f"BLOCKED: {entry.model_id} is qualified to level {entry.qualification_level}, stage {stage} needs more")
        adapter = load_adapter(entry.adapter, spec["checkpoint_dir"], device)
        translator = load_translator(Path(spec["translator_dir"]), fmt, device)
        if translator.member != name:
            raise RuntimeError("translator member mismatch")
        others = {w for n, w in writers.items() if n != name}
        mail_others = {w for n, w in mail_ids.items() if n != name}
        gates = {layer: Gate(float(spec.get("gate_initial_logit", -6.0))) for layer in translator.level_map} if spec.get("gated", False) else None
        hidden = adapter.embed(torch.tensor([0])).shape[-1]
        workers[name] = Worker(
            name, writers[name], session, adapter, translator,
            PoolBank(session, translator, writers[name], others, sinks=int(manifest["window"]["sinks"]), recent=int(manifest["window"]["recent"])),
            gates=gates, override=None if gates else float(spec.get("override", 0.5)), mail_writer=mail_ids[name],
            mail_bank=PoolBank(session, translator, mail_ids[name], mail_others, sinks=0, recent=int(manifest["window"]["recent"]), multiple_per_epoch=True),
            mail=MailWriter(hidden, max_tokens=int(manifest.get("mail_max_tokens", 8)), marker_ple_id=spec.get("marker_ple_id")))
    return workers
