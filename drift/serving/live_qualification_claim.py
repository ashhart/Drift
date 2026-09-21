"""Persist a one-use claim for each cancellation qualification session."""
from dataclasses import dataclass
from pathlib import Path
import uuid

from drift.serving.live_qualification_events import QualificationError


@dataclass(frozen=True)
class SessionClaim:
    name: str
    path: Path

    def spend(self):
        if self.path.name != self.name or not self.path.is_dir():
            raise QualificationError('INVALID_SESSION_CLAIM')
        try:
            with (self.path / 'started').open('x') as marker:
                marker.write('owned cancellation qualification\n')
        except FileExistsError:
            raise QualificationError('SESSION_REUSE') from None


def claim_session(private_ledger_root):
    root = Path(private_ledger_root)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    name = 'qualification-' + uuid.uuid4().hex
    path = root / name
    try:
        path.mkdir(mode=0o700)
    except FileExistsError:
        raise QualificationError('SESSION_REUSE') from None
    return SessionClaim(name, path)
