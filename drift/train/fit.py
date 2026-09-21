from __future__ import annotations
import hashlib
import json
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from safetensors.torch import save_file, load_file
from drift.core.projector import BridgeProjector
from drift.core.types import KV


def receiver_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                temperature: float = 1.0) -> torch.Tensor:
    """Both logits MUST be from the RECEIVER vocabulary at aligned task positions."""
    if student_logits.shape != teacher_logits.shape or temperature <= 0:
        raise ValueError("receiver logits must align; never KL across different vocabularies")
    if not student_logits.requires_grad:
        raise ValueError("student graph is detached; frozen backbones still need autograd")
    logp = F.log_softmax(student_logits.float() / temperature, dim=-1)
    q = F.softmax(teacher_logits.detach().float() / temperature, dim=-1)
    return F.kl_div(logp, q, reduction="batchmean") * temperature**2


def fit_mlp(projector: BridgeProjector, source: KV, target: KV,
            steps: int = 200, batch_size: int = 64, learning_rate: float = 1e-3,
            seed: int = 11) -> dict:
    """KV regression stage only; task/behavior training is a separate build gate."""
    source.check()
    target.check()
    if projector.kind != "mlp" or source.tokens != target.tokens:
        raise ValueError("MLP and aligned training rows required")
    if min(steps, batch_size) <= 0 or learning_rate <= 0:
        raise ValueError("invalid budget")
    device = next(projector.parameters()).device
    if source.k.device != device or target.k.device != device:
        raise ValueError("put training tensors on the projector device explicitly")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    optimizer = torch.optim.AdamW(projector.parameters(), lr=learning_rate, weight_decay=1e-4)
    started, losses = time.perf_counter(), []
    projector.train()
    for _ in range(steps):
        rows = torch.randint(source.tokens, (batch_size,), generator=generator).to(device)
        predicted = projector(KV(source.k[rows].detach(), source.v[rows].detach()))
        loss = (F.mse_loss(predicted.k.float(), target.k[rows].detach().float())
                + F.mse_loss(predicted.v.float(), target.v[rows].detach().float()))
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(projector.parameters(), max_norm=1.0, error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach()))
    projector.eval()
    return {"kind": "kv_regression_only", "steps": steps, "row_presentations": steps * batch_size,
            "unique_input_rows": source.tokens, "seed": seed,
            "wall_seconds": time.perf_counter() - started,
            "first_loss": losses[0], "last_loss": losses[-1],
            "trainable_parameters": sum(p.numel() for p in projector.parameters())}


def save_projector(projector: BridgeProjector, directory: Path, provenance: dict) -> None:
    directory.mkdir(parents=True, exist_ok=False)
    tensors = {name: value.detach().cpu().contiguous() for name, value in projector.state_dict().items()}
    weights = directory / "projector.safetensors"
    save_file(tensors, str(weights))
    metadata = {"format": 1, "source": list(projector.source), "target": list(projector.target),
                "kind": projector.kind, "rank": projector.rank,
                "sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
                "provenance": provenance}
    (directory / "projector.json").write_text(json.dumps(metadata, indent=2) + "\n")


def load_projector(directory: Path, device: str = "cpu") -> BridgeProjector:
    metadata = json.loads((directory / "projector.json").read_text())
    if set(metadata) != {"format", "source", "target", "kind", "rank", "sha256", "provenance"}:
        raise ValueError("unknown projector manifest fields")
    if metadata["format"] != 1:
        raise ValueError("unknown projector artifact version")
    weights = directory / "projector.safetensors"
    if hashlib.sha256(weights.read_bytes()).hexdigest() != metadata["sha256"]:
        raise ValueError("projector checkpoint hash mismatch")
    for shape in (metadata["source"], metadata["target"]):
        if (len(shape) != 2 or any(type(n) is not int for n in shape)
                or not 1 <= shape[0] <= 128 or not 1 <= shape[1] <= 1024
                or shape[0] * shape[1] > 4096):
            raise ValueError("unsafe projector shape")
    if type(metadata["rank"]) is not int or not 1 <= metadata["rank"] <= 1024:
        raise ValueError("unsafe rank")
    hs, ds = metadata["source"]; ht, dt = metadata["target"]; r = metadata["rank"]
    estimated = (2 * (hs * ds * ht * dt + ht * dt) if metadata["kind"] == "ridge"
                 else 2 * (ht * hs + ds * dt + dt + 2 * ds + ds * r + r + r * dt + dt))
    if estimated > 20_000_000:
        raise ValueError("projector exceeds the reference allocation budget")
    projector = BridgeProjector(tuple(metadata["source"]), tuple(metadata["target"]),
                                metadata["kind"], metadata["rank"])
    projector.load_state_dict(load_file(str(weights), device="cpu"), strict=True)
    return projector.to(device).eval()
