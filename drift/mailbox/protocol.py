from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import torch
from torch import nn
from drift.core.types import KV
from drift.runtime.decoder import FrozenDecoder, NativeState


class ThoughtWriter(nn.Module):
    """Token-conditioned v0: local tokens become activations on a PRIVATE branch.

    Not a direct latent writer and not tokenless cognition. The controller must
    ensure the text/IDs originated locally, not from an answer key or a peer tool.
    This function does not serialize anything; only returned canonical KV may
    enter the declared transport, after normal session/window validation.
    """
    def __init__(self, hidden_size: int, max_tokens: int = 32):
        super().__init__()
        self.marker = nn.Parameter(torch.zeros(1, hidden_size))
        self.max_tokens = max_tokens

    def write(self, decoder: FrozenDecoder, private_parent: NativeState,
              local_ids: torch.Tensor) -> dict[int, KV]:
        if local_ids.ndim != 1 or not 1 <= local_ids.numel() <= self.max_tokens:
            raise ValueError("mail token budget exceeded")
        embeddings = decoder.model.model.embed_tokens(local_ids.to(decoder.device))
        embeddings = torch.cat((self.marker.to(embeddings), embeddings))
        output = decoder.forward(None, private_parent.clone(), embeddings=embeddings)
        # No native caller cache is updated; training may backpropagate into marker.
        return dict(output.canonical_delta)


class MailState(str, Enum):
    PENDING = "pending"
    ATTENDED = "attended_not_proven"
    INCORPORATED = "causally_incorporated"
    STARVED = "expired_without_incorporation"


@dataclass
class MailRecord:
    created: int
    expires: int
    state: MailState = MailState.PENDING
    first_attention: int | None = None
    incorporation: int | None = None


class MailboxLedger:
    """Experimenter-side ledger. Models cannot read it or receive its verdicts."""
    def __init__(self, capacity: int = 32):
        if capacity <= 0:
            raise ValueError("positive capacity required")
        self.capacity, self.records = capacity, {}

    def create(self, message_id: int, epoch: int, ttl: int) -> None:
        if min(message_id, epoch) < 0 or ttl <= 0 or message_id in self.records:
            raise ValueError("invalid or duplicate mail ID")
        active = sum(r.state in {MailState.PENDING, MailState.ATTENDED} for r in self.records.values())
        if active >= self.capacity:
            raise OverflowError("mailbox storm/backpressure")
        self.records[message_id] = MailRecord(epoch, epoch + ttl)

    def observe_attention(self, message_id: int, epoch: int, mass: float,
                          threshold: float = 0.01) -> None:
        record = self.records[message_id]
        if not 0 <= mass <= 1 or not record.created <= epoch < record.expires:
            raise ValueError("bad observation time or attention mass")
        if mass >= threshold and record.state == MailState.PENDING:
            record.first_attention, record.state = epoch, MailState.ATTENDED

    def record_causal_use(self, message_id: int, epoch: int, *,
                          active_correct: bool, ablated_correct: bool,
                          replay_started_before_exposure: bool) -> None:
        record = self.records[message_id]
        if not record.created <= epoch < record.expires:
            raise ValueError("outside delivery window")
        if not replay_started_before_exposure:
            raise ValueError("late masking cannot establish a clean causal counterfactual")
        if active_correct and not ablated_correct:
            record.state, record.incorporation = MailState.INCORPORATED, epoch

    def expire(self, epoch: int) -> None:
        for record in self.records.values():
            if epoch >= record.expires and record.state in {MailState.PENDING, MailState.ATTENDED}:
                record.state = MailState.STARVED


class ProbeHead(nn.Module):
    """Diagnostic label probe, NOT a guaranteed natural-language thought decoder.

    Labels, training data and outputs belong to the external audit process.
    """
    def __init__(self, kv_heads: int, head_dim: int, labels: int):
        super().__init__()
        self.head = nn.Linear(2 * kv_heads * head_dim, labels)

    def forward(self, kv: KV) -> torch.Tensor:
        features = torch.cat((kv.k.detach().mean(0).flatten(), kv.v.detach().mean(0).flatten()))
        return self.head(features)
