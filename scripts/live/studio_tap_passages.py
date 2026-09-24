"""Tap Qwen's full-attention K/V over passages, in the framing GLM's captures use. Run with oMLX's bundled interpreter.

Each passage follows Qwen's chat template up to @@DRIFT@@, the same system text as the GLM captures, and is saved as
one npz: the passage tokens' character offsets and x, float16 [tokens, 12 layers x (K + V)], the flat feature the loop
already publishes to GLM's translators. With --linear it also saves h{layer}, float16 [tokens, 2560], each
linear-attention layer's input, the targets for a GLM-to-Qwen recurrent state.
"""
import argparse, json, time
from pathlib import Path
import numpy as np
from omlx.patches.mlx_vlm_qwen4_exp_compat import apply_mlx_vlm_qwen4_exp_compat_patch
apply_mlx_vlm_qwen4_exp_compat_patch()
import mlx.core as mx
from mlx_vlm.utils import load_model
from omlx.engine.vlm import _force_qwen4_exp_sanitize_on_load
from tokenizers import Tokenizer
from drift.serving.omlx_cache import Rope, tap
from drift.serving.studio_guard import acquire

LINK = ("You are linked to another AI model through a shared memory. A document it has read is in your memory, not in this chat; "
        "you were never shown it as text. @@DRIFT@@ Answer the user's question from what you recall from that shared memory. "
        "Give the answer directly.")
QL = tuple(3 + 4 * i for i in range(12))
parser = argparse.ArgumentParser()
parser.add_argument("--triples", type=Path)
parser.add_argument("--split", choices=["train", "val"])
parser.add_argument("--items", type=Path, help="alternatively, a json list of {id, passage}; files are named by the id before its first hyphen")
parser.add_argument("--limit", type=int, default=100000)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--checkpoint", default="~/.omlx/models/Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP")
parser.add_argument("--head-text", type=Path, help="chat-template text to read before each passage instead of the default framing")
parser.add_argument("--linear", action="store_true", help="also save each linear-attention layer's input as h{layer}, the targets for GLM-to-Qwen state")
args = parser.parse_args()
acquire("studio_tap_passages.py", need_gb=150)
ck = Path(args.checkpoint).expanduser()
cfg = json.loads((ck / "config.json").read_text()); t = cfg.get("text_config", cfg); rp = t.get("rope_parameters") or {}
rope = Rope(float(rp.get("rope_theta", 1e7)), int(t["head_dim"] * float(rp.get("partial_rotary_factor", 1.0))))
tok = Tokenizer.from_file(str(ck / "tokenizer.json"))
with _force_qwen4_exp_sanitize_on_load(ck):
    model = load_model(ck)
lm = model.language_model
LINEAR = {id(layer.linear_attn): i for i, layer in enumerate(lm.model.layers) if getattr(layer, "is_linear", False)}
CAPTURE = {"span": None, "records": {}}
if args.linear:
    from mlx_vlm.models.qwen4_exp.language import Qwen4ExpGatedDeltaNet
    original = Qwen4ExpGatedDeltaNet.__call__

    def recording(self, inputs, *rest, **options):
        if CAPTURE["span"] is not None and id(self) in LINEAR:
            CAPTURE["records"][LINEAR[id(self)]] = inputs[0, CAPTURE["span"][0]:CAPTURE["span"][1]]
        return original(self, inputs, *rest, **options)                # oMLX passes speculative-verify options too

    Qwen4ExpGatedDeltaNet.__call__ = recording
prompt = f"<|im_start|>system\n{LINK}<|im_end|>\n"
HEAD = tok.encode(args.head_text.read_text() if args.head_text else prompt[:prompt.index("@@DRIFT@@")], add_special_tokens=False).ids
args.out.mkdir(parents=True, exist_ok=True)
passages = {}
if args.items:
    for item in json.loads(args.items.read_text()):
        passages.setdefault(item["id"].split("-")[0], item["passage"])
elif args.triples and args.split:
    for triple in json.loads(args.triples.read_text())[args.split]:
        passages.setdefault(triple["pid"], triple["passage"])
else:
    raise SystemExit("give --items, or --triples with --split")
done, started, written = {p.stem for p in args.out.glob("*.npz")}, time.time(), 0
for pid, text in list(passages.items())[: args.limit]:
    if pid in done:
        continue
    encoded = tok.encode(text, add_special_tokens=False)
    ids = HEAD + encoded.ids
    cache = lm.make_cache()
    CAPTURE.update(span=(len(HEAD), len(ids)) if args.linear else None, records={})
    mx.eval(lm(mx.array(np.asarray(ids, dtype=np.int32))[None], cache=cache).logits[0, -1])
    CAPTURE["span"] = None
    if args.linear and len(CAPTURE["records"]) != len(LINEAR):
        raise SystemExit(f"captured {len(CAPTURE['records'])} of {len(LINEAR)} linear layers")
    rows = tap(cache, QL, rope, len(HEAD), len(ids))
    n = len(encoded.ids)
    x = np.concatenate([np.concatenate((rows[l][0].reshape(n, -1), rows[l][1].reshape(n, -1)), axis=1) for l in QL], axis=1)
    linear = {f"h{i}": np.array(v.astype(mx.float16)) for i, v in CAPTURE["records"].items()}
    np.savez(args.out / f"{pid}.npz", offsets=np.asarray(encoded.offsets, np.int32), head=np.int32(len(HEAD)), x=x.astype(np.float16), **linear)
    written += 1
    if written % 50 == 0:
        print(json.dumps({"done": written, "seconds": round(time.time() - started, 1)}), flush=True)
print(json.dumps({"split": args.split, "written": written, "seconds": round(time.time() - started, 1)}))
