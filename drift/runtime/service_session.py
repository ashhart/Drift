"""Reference worker session lifecycle and typed operation dispatch."""
from __future__ import annotations
import hashlib
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping
import torch
from drift.eval.detectors import DriftTracker, gate_saturation
from drift.mailbox.stream import MailboxController
from drift.runtime.hive import HiveController
from drift.runtime.worker import Worker
from drift.runtime.service_protocol import Op, ServiceError, _int
from drift.transport.wire2 import FrameCodec2


@dataclass
class Session:
    builder: Callable[[dict], dict[str, Worker]]
    audit_root: Path
    codec_key: bytes
    phase: str = "SETUP"
    manifest: dict | None = None
    manifest_sha256: str | None = None
    controller: HiveController | None = None
    mailbox: MailboxController = field(default_factory=lambda: MailboxController(ttl=8, slots=8, per_epoch_cap=2))
    last_local: dict[str, torch.Tensor] = field(default_factory=dict)
    generated: dict[str, list[int]] = field(default_factory=dict)
    trackers: dict[str, DriftTracker] = field(default_factory=dict)
    last_checkpoint_sha256: str | None = None
    checkpoint_every: int = 0
    paused: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)

    # -- helpers --------------------------------------------------------------
    def _require(self, *phases: str) -> None:
        if self.phase not in phases:
            raise ServiceError("PHASE")
        if self.controller is not None and self.controller.failed:
            raise ServiceError("POISONED")

    def _members(self) -> list[str]:
        return list(self.controller.workers) if self.controller else []

    # -- ops ------------------------------------------------------------------
    def setup(self, manifest_path: str, task: str, assignments: Mapping[str, str]) -> dict:
        self._require("SETUP")
        path = Path(manifest_path)
        if not path.is_file():
            raise ServiceError("MALFORMED")
        raw = path.read_bytes()
        try:
            self.manifest = json.loads(raw)
        except json.JSONDecodeError:
            raise ServiceError("MALFORMED")
        self.manifest_sha256 = hashlib.sha256(raw).hexdigest()
        try:
            workers = self.builder(self.manifest)
        except (KeyError, ValueError, TypeError):
            raise ServiceError("MALFORMED")
        if len(workers) < 2:
            raise ServiceError("MALFORMED")
        self.controller = HiveController(workers, FrameCodec2(self.codec_key))
        tokenizer = self.manifest.get("setup_tokenizer", "bytes")
        if tokenizer != "bytes":
            raise ServiceError("MALFORMED")
        names = list(workers)
        # The plugin keys assignments by member enum index; names are accepted too.
        by_name = {(names[int(k)] if str(k).isdigit() and int(k) < len(names) else k): v for k, v in assignments.items()}
        for name, worker in workers.items():
            text = (task + "\n" + by_name.get(name, "")).encode()
            ids = torch.tensor([b % worker.adapter.decoder.model.config.vocab_size if hasattr(worker.adapter, "decoder") else b for b in text][:64] or [1], dtype=torch.long)
            self.last_local[name] = ids
            self.generated[name] = []
            self.trackers[name] = DriftTracker()
        (self.audit_root / "setup.json").write_text(json.dumps({"manifest_sha256": self.manifest_sha256, "task": task, "assignments": dict(assignments)}, indent=2))
        return {"manifest_sha256": self.manifest_sha256, "members": [{"index": i, "name": n, "writer": w.writer} for i, (n, w) in enumerate(workers.items())]}

    def start(self, checkpoint_every: int) -> dict:
        self._require("SETUP")
        if self.controller is None or checkpoint_every < 0:
            raise ServiceError("MALFORMED")
        self.checkpoint_every, self.phase = checkpoint_every, "STRICT"
        return {"phase": self.phase, "epoch": 0}

    def tick(self, epochs: int) -> dict:
        self._require("STRICT")
        if epochs < 1 or self.paused:
            raise ServiceError("MALFORMED" if epochs < 1 else "PHASE")
        for _ in range(epochs):
            with torch.no_grad():
                reports = self.controller.tick(dict(self.last_local))
            for name, report in reports.items():
                token = int(report.output[-1].argmax())
                self.generated[name].append(token)
                self.last_local[name] = torch.tensor([token], dtype=torch.long)
                self.trackers[name].observe(report.canonical_norms, report.foreign_mass)
                worker = self.controller.workers[name]
                if worker.mail_bank is not None and report.mail_foreign_tokens:
                    mass = {layer: m[report.native_foreign_tokens:] for layer, m in report.entry_mass.items()}
                    self.mailbox.observe(name, report.epoch, worker.mail_bank.pin(), mass)
            self.mailbox.expire(self.controller.epoch)
            if self.checkpoint_every and self.controller.epoch % self.checkpoint_every == 0:
                self.checkpoint()
        return {"epoch": self.controller.epoch}

    def mail(self, sender_index: int, slots: int) -> dict:
        self._require("STRICT")
        names = self._members()
        if not 0 <= sender_index < len(names) or slots < 1:
            raise ServiceError("MALFORMED")
        name = names[sender_index]
        worker = self.controller.workers[name]
        if worker.mail is None:
            raise ServiceError("MALFORMED")
        recent = self.generated[name][-min(slots, worker.mail.max_tokens):]
        if not recent:
            raise ServiceError("BUDGET")
        try:
            record = self.mailbox.create(name, self.controller.epoch - 1, len(recent))
        except OverflowError:
            raise ServiceError("BUDGET")
        self.controller.post_mail(name, torch.tensor(recent, dtype=torch.long), self.mailbox, record.id)
        return {"message_id": record.id, "slot": record.slot, "tokens": len(recent)}

    def checkpoint(self) -> dict:
        self._require("STRICT")
        data = self.controller.checkpoint()
        blob = json.dumps({"epoch": data["epoch"], "members": {n: {"counters": vars(w["counters"])} for n, w in data["workers"].items()}}, sort_keys=True).encode()
        digest = hashlib.sha256(blob).hexdigest()
        torch.save(data, self.audit_root / f"checkpoint-{self.controller.epoch:06d}.pt")
        self.last_checkpoint_sha256 = digest
        return {"epoch": self.controller.epoch, "sha256": digest}

    def pause(self) -> dict:
        self._require("STRICT")
        self.paused = not self.paused
        return {"paused": self.paused}

    def abort(self) -> dict:
        if self.controller is not None:
            self.controller.failed = True
        self.phase = "CLOSED"
        return {"phase": self.phase}

    def complete(self) -> dict:
        self._require("STRICT")
        (self.audit_root / "final_outputs.json").write_text(json.dumps({
            "manifest_sha256": self.manifest_sha256, "epoch": self.controller.epoch,
            "generated_ids": self.generated, "mailbox": self.mailbox.metrics(),
            "detectors": {n: t.report() for n, t in self.trackers.items()}}, indent=2))
        self.phase = "CLOSED"
        return {"phase": self.phase, "epoch": self.controller.epoch}

    def status(self) -> dict:
        members = []
        for index, (name, worker) in enumerate((self.controller.workers if self.controller else {}).items()):
            c = worker.counters
            report = self.trackers[name].report() if name in self.trackers else {}
            members.append({
                "index": index, "writer": worker.writer, "epoch": c.epoch, "sequence": c.sequence,
                "source_position": c.source_position, "local_tokens": c.local_tokens,
                "mail_sent": c.mail_sequence, "foreign_tokens": 0 if worker.bank.pin() is None else int(worker.bank.pin().positions.numel()),
                "mail_foreign_tokens": 0 if worker.mail_bank is None or worker.mail_bank.pin() is None else int(worker.mail_bank.pin().positions.numel()),
                "gate": {str(k): v for k, v in gate_saturation(worker.gates or {}).items()},
                "mass_mean": {str(k): v for k, v in report.get("mass_mean", {}).items()},
            })
        detectors = {"nonfinite": any(t.report()["nonfinite"] for t in self.trackers.values())}
        for name, tracker in self.trackers.items():
            rep = tracker.report()
            detectors[f"norm_drift_{name}"] = {str(k): v for k, v in rep["norm_drift"].items()}
        return {"phase": self.phase, "epoch": 0 if self.controller is None else self.controller.epoch,
                "poisoned": bool(self.controller and self.controller.failed), "paused": self.paused,
                "members": members, "mailbox": {"counts": self.mailbox.metrics()["counts"]},
                "detectors": detectors, "manifest_sha256": self.manifest_sha256,
                "last_checkpoint_sha256": self.last_checkpoint_sha256}

    # -- dispatch -------------------------------------------------------------
    def handle(self, request: Mapping[str, Any]) -> dict:
        try:
            op = Op(int(request.get("op")))
        except (TypeError, ValueError):
            raise ServiceError("UNKNOWN_OP")
        with self.lock:
            if self.phase == "STRICT":
                for key, value in request.items():
                    if isinstance(value, str) and key not in ("auth",):
                        raise ServiceError("MALFORMED")
            if op is Op.SETUP:
                assignments = request.get("assignments") or {}
                if not isinstance(request.get("manifest", ""), str) or not isinstance(request.get("task", ""), str) \
                        or not isinstance(assignments, dict) or not all(isinstance(v, str) for v in assignments.values()):
                    raise ServiceError("MALFORMED")
                return self.setup(request.get("manifest", ""), request.get("task", ""), assignments)
            if op is Op.START:
                return self.start(_int(request, "checkpoint_every", 0))
            if op is Op.TICK:
                return self.tick(_int(request, "epochs", 1))
            if op is Op.STATUS:
                return self.status()
            if op is Op.PAUSE:
                return self.pause()
            if op is Op.CHECKPOINT:
                return self.checkpoint()
            if op is Op.ABORT:
                return self.abort()
            if op is Op.COMPLETE:
                return self.complete()
            if op is Op.MAIL:
                return self.mail(_int(request, "sender", -1), _int(request, "slots", 0))
            raise ServiceError("UNKNOWN_OP")
