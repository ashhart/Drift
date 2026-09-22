from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_plugin_ci_provisions_the_required_seatbelt_runtime():
    workflow = (ROOT / '.github/workflows/tests.yml').read_text()
    plugin = workflow.split('\n  plugin:', 1)[1]
    assert 'runs-on: macos-15' in plugin
    assert 'uses: actions/setup-python@v5' in plugin
    assert 'python-version: "3.11.9"' in plugin
    assert 'test -x /usr/bin/sandbox-exec' in plugin
    assert 'test -x /Library/Frameworks/Python.framework/Versions/3.11/bin/python3' in plugin


def test_pure_python_stage_fixtures_use_the_current_interpreter():
    for name in ('test_development_api.py', 'test_development_budget.py'):
        source = (ROOT / 'tests' / name).read_text()
        assert '/Library/Frameworks/' not in source
        assert 'Path(sys.executable)' in source
