"""Rebuild the memory Qwen published in a live appointment run: its prompt rows, tapped and translated as the loop does.

Run on the Studio with oMLX's interpreter, the run folder copied next to the loop's out folder. For each scenario Qwen
reads its own prompt in the loop's chat layout, its full-attention K/V over the whole prompt is tapped, and the reverse
translator turns it into GLM latents, float16 like the loop's publication. The copies are left to the reader, as the
bridge tiles them. With these rows in GLM's reserve, an export reproduces the context GLM wrote its reply in.
--features also keeps Qwen's tapped features as x, the input translate_passages.py turns into rows and a state. With
--items instead of --run, the chats come from appointment_items.py and each file is named by its item's id.
"""
import argparse, json
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
from drift.translate.stacked import StackedReader

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=Path, help="the demo_appointment.py output folder")
parser.add_argument("--items", type=Path, help="alternatively, appointment_items.py output")
parser.add_argument("--rows", type=Path, default=Path("local/live/stacked3_rev.npz"), help="the loop's reverse translator")
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--checkpoint", default="~/.omlx/models/Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP")
parser.add_argument("--features", type=Path, help="also write Qwen's tapped features per scenario here, as translate_passages.py reads them")
args = parser.parse_args()
if bool(args.run) == bool(args.items):
    parser.error("give --run or --items")
acquire("studio_replay_memory.py", need_gb=150)
GL, QL = tuple(3 + 4 * i for i in range(11)), tuple(3 + 4 * i for i in range(12))
ck = Path(args.checkpoint).expanduser()
cfg = json.loads((ck / "config.json").read_text()); t = cfg.get("text_config", cfg); rp = t.get("rope_parameters") or {}
rope = Rope(float(rp.get("rope_theta", 1e7)), int(t["head_dim"] * float(rp.get("partial_rotary_factor", 1.0))))
tok = Tokenizer.from_file(str(ck / "tokenizer.json"))
with _force_qwen4_exp_sanitize_on_load(ck):
    model = load_model(ck)
lm = model.language_model
rev = StackedReader.load(args.rows, QL, GL, kv_heads=0, head_dim=512)
args.out.mkdir(parents=True, exist_ok=True)
if args.run:
    chats = [(f"r{row['scenario']}", json.loads((args.run / f"s{row['scenario']}.qwen.json").read_text()))
             for row in json.loads((args.run / "report.json").read_text())["rows"]]
else:
    chats = [(item["id"], item["qwen"]) for item in json.loads(args.items.read_text())]
for name, messages in chats:
    prompt = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages) + "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    ids = tok.encode(prompt, add_special_tokens=False).ids
    cache = lm.make_cache()
    mx.eval(lm(mx.array(np.asarray(ids, dtype=np.int32))[None], cache=cache).logits[0, -1])
    own = tap(cache, QL, rope, 0, len(ids))
    per_layer = {l: np.concatenate((own[l][0].reshape(len(ids), -1), own[l][1].reshape(len(ids), -1)), axis=1).astype(np.float32) for l in QL}
    latents = rev.read(per_layer, 1.0)
    np.savez(args.out / f"{name}.npz", **{f"l{l}": np.asarray(latents[l], dtype=np.float16) for l in GL})
    if args.features:
        args.features.mkdir(parents=True, exist_ok=True)
        np.savez(args.features / f"{name}.npz", x=np.concatenate([per_layer[l] for l in QL], axis=1).astype(np.float16))
print(json.dumps({"scenarios": len(list(args.out.glob("*.npz")))}))
