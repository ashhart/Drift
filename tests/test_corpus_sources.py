import json
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/live/build_corpus.py'


def test_repository_corpus_builds_without_archived_handoff(tmp_path, monkeypatch):
    text = ('This synthetic paragraph tests current repository corpus input. ' * 8).strip()
    (tmp_path / 'README.md').write_text(text)
    (tmp_path / 'AGENTS.md').write_text('Contract')
    (tmp_path / 'docs').mkdir()
    (tmp_path / 'drift').mkdir()
    class Tokenizer:
        @classmethod
        def from_file(cls, path):
            return cls()
        def encode(self, value, **kwargs):
            self.text = value
            offsets = [(i, min(i + 4, len(value))) for i in range(0, len(value), 4)]
            return SimpleNamespace(ids=list(range(len(offsets))), offsets=offsets)
        def decode(self, ids, **kwargs):
            return self.text
    monkeypatch.setitem(sys.modules, 'tokenizers', SimpleNamespace(Tokenizer=Tokenizer))
    monkeypatch.chdir(tmp_path)
    output = tmp_path / 'corpus.json'
    monkeypatch.setattr(sys, 'argv', [str(SCRIPT), '--out', str(output), '--heldout', '1'])
    runpy.run_path(str(SCRIPT), run_name='__main__')
    records = json.loads(output.read_text())['records']
    assert len(records) == 1 and records[0]['source'] == 'README.md'
    assert records[0]['split'] == 'heldout' and records[0]['aligned']
    assert json.loads(output.with_suffix('.ids.json').read_text())['records'][0]['id'] == records[0]['id']
