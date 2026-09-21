"""Explicit recipe selection never silently changes legacy CLI defaults."""
import sys
from pathlib import Path

import pytest

from drift.serving.live_options import parse_run_options


def options(monkeypatch, *extra):
    monkeypatch.setattr(sys, 'argv', ['drift_loop.py', '--scenario', 'scenario.json', '--out', 'run', *extra])
    return parse_run_options()[0]


def test_legacy_cli_defaults_are_unchanged(monkeypatch):
    args = options(monkeypatch)
    assert args.forward_recipe == 'legacy' and args.forward == Path('local/live/stacked2.npz')
    assert args.gain_forward == 1.5 and args.forward_manifest is None


def test_explicit_recipe_gets_its_default_gain_from_manifest(monkeypatch):
    args = options(monkeypatch, '--forward-recipe', 'v4', '--forward-manifest', 'frozen.json')
    assert args.forward is None and args.gain_forward is None
    assert args.forward_manifest == Path('frozen.json')


@pytest.mark.parametrize('extra', [
    ['--forward-recipe', 'v4'], ['--forward-manifest', 'frozen.json'],
    ['--forward-recipe', 'fanout', '--forward-manifest', 'frozen.json', '--forward', 'other.npz'],
    ['--gain-forward', 'nan'], ['--gain-reverse', 'inf']])
def test_ambiguous_or_invalid_recipe_options_are_rejected(monkeypatch, extra):
    with pytest.raises(SystemExit) as failure:
        options(monkeypatch, *extra)
    assert failure.value.code == 2
