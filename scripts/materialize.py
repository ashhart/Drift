"""Extract hashed source blocks from the standalone engineering Markdown file.

Usage: python materialize.py TELEPATHY_AGENT_ENGINEERING.md ./drift
Refuses nonempty destinations, path traversal, duplicate files and bad hashes.
"""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path, PurePosixPath
import re


def extract(document: Path, destination: Path) -> int:
    text = document.read_text(encoding="utf-8")
    pattern = re.compile(
        r'<!-- file: ([^\n]+) sha256: ([0-9a-f]{64}) -->\n'
        r'````[^\n]*\n(.*?)\n````(?=\n)', re.DOTALL)
    files = {}
    for match in pattern.finditer(text):
        name, expected, body = match.groups()
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name or not name:
            raise ValueError(f"unsafe source path: {name}")
        if name in files:
            raise ValueError(f"duplicate source: {name}")
        payload = (body + "\n").encode("utf-8")
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError(f"source hash mismatch: {name}")
        files[name] = payload
    if not files:
        raise ValueError("no extractable source blocks")
    if destination.is_symlink() or (destination.exists() and any(destination.iterdir())):
        raise FileExistsError("destination must be a new or empty nonsymlink directory")
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    for name, payload in files.items():
        target = root.joinpath(*PurePosixPath(name).parts)
        if not target.resolve().is_relative_to(root):
            raise ValueError("source escapes destination")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    return len(files)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("document", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(f"Extracted {extract(args.document, args.destination)} verified files")
