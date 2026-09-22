import re
from pathlib import Path
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]


def prose(text):
    lines, fence = [], None
    for line in text.splitlines():
        marker = re.match(r"^\s*(`{3,}|~{3,})", line)
        if marker:
            token = marker[1]
            if fence is None:
                fence = token
            elif token[0] == fence[0] and len(token) >= len(fence):
                fence = None
            continue
        if fence is None:
            lines.append(line)
    return "\n".join(lines)


def test_documentation_has_one_current_status_and_a_reader_index():
    docs = ROOT / "docs"
    assert {path.name for path in docs.glob("*.md")} == {
        "README.md", "STATUS.md", "ROADMAP.md", "agent-progress.md",
    }
    assert not (docs / "FRONTIER.md").exists()
    assert "STATUS.md" in (docs / "README.md").read_text()
    assert "docs/STATUS.md" in (ROOT / "README.md").read_text()


def test_local_documentation_links_resolve():
    paths = [*(path for path in ROOT.glob("*.md") if path.name != "TELEPATHY_AGENT_ENGINEERING.md"),
             ROOT / "plugin/omp-drift/README.md",
             *sorted((ROOT / "docs").rglob("*.md"))]
    broken = []
    for path in paths:
        for target in re.findall(r"\]\(([^)\s]+)(?:\s+[^)]*)?\)", prose(path.read_text())):
            parsed = urlsplit(target.strip("<>"))
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            destination = path.parent / unquote(parsed.path)
            if not destination.exists():
                broken.append((str(path.relative_to(ROOT)), target))
    assert not broken, broken


def test_historical_records_are_not_current_instructions():
    history = ROOT / "docs/history"
    assert history.is_dir()
    for path in history.glob("*.md"):
        if path.name == "README.md":
            continue
        assert "historical" in "\n".join(path.read_text().splitlines()[:8]).lower(), path.name
