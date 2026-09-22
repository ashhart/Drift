"""Every relative JavaScript dependency must survive isolated boundary staging."""
import importlib
from pathlib import Path
import re


def test_boundary_stage_contains_its_relative_module_closure(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / 'scripts' / 'omp'))
    module = importlib.import_module('provider_boundary_setup')
    module.copy_boundary_modules(tmp_path)
    for path in tmp_path.glob('*.mjs'):
        for name in re.findall(r"from ['\"]\./([^'\"]+\.mjs)['\"]", path.read_text()):
            assert (tmp_path / name).is_file(), (path.name, name)
