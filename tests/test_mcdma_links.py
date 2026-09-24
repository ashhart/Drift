"""The Studio's MCDMA legs come from the operator, head first, and a malformed list fails before any connection."""
from pathlib import Path
import re

import pytest

from drift.serving.mcdma_links import HEAD, RANKS, parse_links

ROOT = Path(__file__).resolve().parents[1]
LINKS = "192.0.2.1/192.0.2.40,198.51.100.1/198.51.100.40"
STUDIO_SCRIPTS = ("scripts/live/studio_mcdma_loop.py", "scripts/live/studio_mcdma_reverse.py", "scripts/mcdma_target/studio_roundtrip.py")


def test_legs_map_to_ranks_head_first():
    legs = parse_links(LINKS)
    assert list(legs) == list(RANKS) and HEAD == RANKS[0]
    assert legs[HEAD] == ("192.0.2.1", "192.0.2.40")
    assert legs[RANKS[1]] == ("198.51.100.1", "198.51.100.40")


@pytest.mark.parametrize("text", [
    "",
    "192.0.2.1/192.0.2.40",
    LINKS + ",203.0.113.1/203.0.113.40",
    "192.0.2.1,198.51.100.1",
    "192.0.2.1/192.0.2.40/192.0.2.41,198.51.100.1/198.51.100.40",
    "spark-a.invalid/192.0.2.40,198.51.100.1/198.51.100.40",
    "192.0.2.1/192.0.2.40,192.0.2.1/198.51.100.40",
    "192.0.2.1/192.0.2.40, 198.51.100.1/198.51.100.40",
])
def test_a_malformed_list_is_refused(text):
    with pytest.raises(ValueError):
        parse_links(text)


@pytest.mark.parametrize("path", STUDIO_SCRIPTS)
def test_studio_scripts_take_their_legs_from_the_operator(path):
    source = (ROOT / path).read_text()
    assert "parse_links(args.links)" in source
    assert not re.search(r"\b\d{1,3}(\.\d{1,3}){3}\b", source)
