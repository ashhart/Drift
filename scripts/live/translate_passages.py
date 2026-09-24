"""Turn a sender's passage features into GLM memory: MLA rows from a row translator, KDA inputs from a state translator.

Reads one npz per passage with the sender's per-token features as x: Qwen's taps from studio_tap_passages.py, or
DeepSeek V4's from dsv4_token_features.py. Writes one npz per passage: l{layer} float16 [tokens, 512] for GLM's MLA layers
and h{layer} float16 [tokens, 4096] for its KDA layers, one row per sender token. The row translator is Qwen's reverse
stacked reader (stacked3_rev.npz), any translator fit_state_translator.py wrote with GLM's l{layer} as outputs, a
translator with one flat output of every MLA layer's latent in layer order (a shared-space pair, drift/translate/hub.py),
or with --context-reader a Qwen-to-GLM contextual reader, which reads the whole passage in windows. Run it on the Studio,
never on the laptop.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from drift.translate import ridge_map
from drift.translate.stacked import StackedReader

GL, QL = tuple(3 + 4 * i for i in range(11)), tuple(3 + 4 * i for i in range(12))
parser = argparse.ArgumentParser()
parser.add_argument("--taps", type=Path, required=True)
parser.add_argument("--rows", type=Path, help="stacked3_rev.npz for Qwen, or a fitted translator whose outputs are GLM's l{layer}")
parser.add_argument("--context-reader", type=Path, help="a Qwen-to-GLM contextual reader folder, in place of --rows")
parser.add_argument("--reader-window", type=int, default=1536)
parser.add_argument("--state", type=Path, help="the state translator from fit_state_translator.py; without it only rows are written")
parser.add_argument("--limit", type=int, default=100000)
parser.add_argument("--gain-power", type=float, default=1.0, help="rescaling of the state translator's shrunk outputs")
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()


def row_reader(path: Path):
    """Sender features [tokens, width] -> {GLM MLA layer: rows}."""
    if "basis" in np.load(path).files:
        translator = ridge_map.load(path)
        if sorted(translator["layers"]) == [0] and translator["layers"][0][0].shape[-1] == 512 * len(GL):
            return lambda x: split(ridge_map.apply(translator, x)[0])
        if sorted(translator["layers"]) != list(GL):
            raise SystemExit("the row translator's outputs are not GLM's MLA layers")
        return lambda x: ridge_map.apply(translator, x)
    rev = StackedReader.load(path, QL, GL, kv_heads=0, head_dim=512)
    return lambda x: rev.read({l: x[:, i * 1024:(i + 1) * 1024] for i, l in enumerate(QL)}, 1.0)


def split(flat: np.ndarray) -> dict:
    """[tokens, 11 x 512] in layer order -> {GLM MLA layer: [tokens, 512]}."""
    return {layer: flat[:, i * 512:(i + 1) * 512] for i, layer in enumerate(GL)}


if (args.rows is None) == (args.context_reader is None):
    raise SystemExit("give --rows or --context-reader")
if args.context_reader:
    from drift.translate.context_reader import ContextRows
    reader = ContextRows(args.context_reader, window=args.reader_window)
    read_rows = lambda x: split(reader.read(x))
else:
    read_rows = row_reader(args.rows)
state = ridge_map.load(args.state, args.gain_power) if args.state else None
args.out.mkdir(parents=True, exist_ok=True)
for path in sorted(args.taps.glob("*.npz"))[: args.limit]:
    x = np.load(path)["x"].astype(np.float32)
    rows, hidden = read_rows(x), ridge_map.apply(state, x) if state else {}
    np.savez(args.out / path.name, **{f"l{l}": np.asarray(rows[l], np.float16) for l in GL}, **{f"h{l}": v.astype(np.float16) for l, v in hidden.items()})
print({"passages": len(list(args.out.glob("*.npz")))})
