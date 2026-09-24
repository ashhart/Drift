"""Name the Studio's MCDMA leg to each GLM rank from an operator-supplied list that stays out of the repository."""
from __future__ import annotations
import ipaddress

RANKS: tuple[str, ...] = ("spark-a.invalid", "spark-b.invalid")        # receipt labels, head first
HEAD = RANKS[0]


def parse_links(text: str) -> dict[str, tuple[str, str]]:
    """'target/source,target/source', head first -> rank label -> (rank target address, Studio source address)."""
    pairs = [leg.split("/") for leg in text.split(",")]
    if len(pairs) != len(RANKS) or any(len(pair) != 2 for pair in pairs):
        raise ValueError(f"expected {len(RANKS)} MCDMA legs as target/source, head first")
    legs = [(target, source) for target, source in pairs]
    for leg in legs:
        for address in leg:
            ipaddress.IPv4Address(address)
    if len({target for target, _ in legs}) != len(legs):
        raise ValueError("each rank needs its own MCDMA target")
    return {rank: leg for rank, leg in zip(RANKS, legs)}
