"""Studio (oMLX runtime): long-document Drift. GLM read each document on the Sparks; here its raw entries are translated ON
THE READER'S GPU (K/V and, new, the sparse-selector keys), appended to a fresh cache, and Qwen answers needle questions.

Conditions per question:
  text          the document is in the prompt (time to prefill measured once per document)
  own_real      Qwen's own K/V and own selector keys (the ceiling for any cache-only memory: recurrent state is fresh)
  drift_keys    translated K/V + TRANSLATED selector keys            <- the intervention
  drift_zero    translated K/V + zero selector keys                  <- what Drift did before
  wrong_memory  another document's drift_keys memory
  drift_kv_own_keys / own_kv_drift_keys   DIAGNOSTICS (need the text-side token alignment, so not a Drift channel): which half fails?
  none          question only
Every stage is timed separately; nothing is summed into a headline number here."""
import argparse, json, time
from pathlib import Path
import numpy as np
from omlx.patches.mlx_vlm_qwen4_exp_compat import apply_mlx_vlm_qwen4_exp_compat_patch
apply_mlx_vlm_qwen4_exp_compat_patch()
import mlx.core as mx
from tokenizers import Tokenizer
from mlx_vlm.utils import load_model
from omlx.engine.vlm import _force_qwen4_exp_sanitize_on_load
from drift.serving.omlx_cache import Rope, append_entries, kv_layer_indices, tap
from drift.translate.fanout import CorrectedFanoutReader, FanoutReader
from drift.translate.index_keys import IndexKeyReader
from drift.translate.mlx_reader import MlxForwardReader
from drift.translate.stacked import StackedReader

parser = argparse.ArgumentParser()
parser.add_argument("--docs", type=Path, required=True)
parser.add_argument("--taps", type=Path, required=True)
parser.add_argument("--artifacts", type=Path, default=Path("local/live"))
parser.add_argument("--index", default="index3.npz")
parser.add_argument("--gain-power", type=float, default=1.5)
parser.add_argument("--index-gain-power", type=float, default=1.0)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--label", default="EXPLORATORY")
parser.add_argument("--only", nargs="*", help="run only these conditions (development sweeps)")
parser.add_argument("--sweep-gain", type=float, nargs="*", default=[], help="extra drift conditions drift@<gain power>, translated selector keys")
parser.add_argument("--variants", nargs="*", default=[], help="development: extra drift conditions name=stacked:<npz> or name=context:<npz> (base translator only, no fan-out, CPU)")
parser.add_argument("--base", default="stacked3.npz")
parser.add_argument("--fanout", default="fanout3.npz")
parser.add_argument("--correction", default="v4_correction.safetensors", help="'none' = the untrained fan-out reader")
args = parser.parse_args()
GL, QL = tuple(3 + 4 * i for i in range(11)), tuple(3 + 4 * i for i in range(12))
ck = Path("~/.omlx/models/Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP").expanduser()
cfg = json.loads((ck / "config.json").read_text()); t = cfg.get("text_config", cfg); rp = t.get("rope_parameters") or {}
rope = Rope(float(rp.get("rope_theta", 1e7)), int(t["head_dim"] * float(rp.get("partial_rotary_factor", 1.0))))
tok = Tokenizer.from_file(str(ck / "tokenizer.json"))
from drift.serving.studio_guard import acquire
acquire("studio_longctx_drift.py", need_gb=170)
with _force_qwen4_exp_sanitize_on_load(ck):
    model = load_model(ck)
lm = model.language_model
STOP = {tok.token_to_id("<|im_end|>"), tok.token_to_id("<|endoftext|>")}
base = StackedReader.load(args.artifacts / args.base, GL, QL, kv_heads=2, head_dim=256)
if args.fanout == "none":                                                     # development: base translator only, one reader row per writer token
    fan = FanoutReader(base, np.zeros((base.input_mean.shape[0], 1), np.float32), np.zeros((1, 1), np.float32), np.zeros(1, np.float32), 0.0, {}, {}, "none")
else:
    fan = FanoutReader.load(args.artifacts / args.fanout, base)
if args.correction == "none":
    width = 2 * base.kv_heads * base.head_dim * len(QL)
    corrected = CorrectedFanoutReader(fan, np.zeros((base.input_mean.shape[0], 1), np.float32), np.zeros((1, width), np.float32), np.zeros((len(QL), 2), np.float32), "none")
else:
    corrected = CorrectedFanoutReader.load(args.artifacts / args.correction, fan)
index = IndexKeyReader.load(args.artifacts / args.index, QL, json.loads(str(np.load(args.artifacts / args.index)["meta"]))["base_sha256"] if args.fanout == "none" else base.sha256)
reader = MlxForwardReader(corrected, index, args.gain_power, args.index_gain_power)
sweep = {g: MlxForwardReader(corrected, index, g, args.index_gain_power) for g in args.sweep_gain}
from drift.translate.context import ContextFanoutReader, ContextReader
variants = {}
for spec in args.variants:                                                     # name=stacked:<npz> | name=context:<npz> | name=full:<context npz>+<fan-out npz>+<selector npz>
    name, rest = spec.split("=", 1); kind, path = rest.split(":", 1)
    if kind == "full":
        c, f, i, *corr = (Path(x) for x in path.split("+"))                     # optional 4th part: a binding-training correction
        ctx = ContextReader.load(c, GL, QL, 2, 256)
        full = ContextFanoutReader.load(f, ctx)
        variants[name] = (full.with_correction(corr[0]) if corr else full, IndexKeyReader.load(i, QL, ctx.sha256))
    else:
        variants[name] = ContextReader.load(Path(path), GL, QL, 2, 256) if kind == "context" else StackedReader.load(Path(path), GL, QL, kv_heads=2, head_dim=256)


def clone(cache):
    fresh = lm.make_cache()
    for new, old in zip(fresh, cache):
        state = old.state
        new.state = type(state)(mx.array(x) if isinstance(x, mx.array) else x for x in state) if isinstance(state, (list, tuple)) else state
        if hasattr(old, "meta_state"):
            try:
                new.meta_state = old.meta_state
            except Exception:
                pass
    return fresh


def step(ids, cache, start):
    positions = mx.arange(start, start + len(ids), dtype=mx.int32)[None]
    logits = lm(mx.array(np.asarray(ids, dtype=np.int32))[None], cache=cache, position_ids=positions).logits[0, -1]
    mx.eval(logits)
    return logits


def prefill(ids, cache, start=0, chunk=2048):
    for a in range(0, len(ids), chunk):
        last = step(ids[a:a + chunk], cache, start + a)
    return last


def answer(prompt_ids, cache, start, limit=32):
    t0 = time.time(); logits = prefill(prompt_ids, cache, start); first = time.time() - t0
    out, pos = [], start + len(prompt_ids)
    for _ in range(limit):
        nxt = int(mx.argmax(logits).item())
        if nxt in STOP:
            break
        out.append(nxt); logits = step([nxt], cache, pos); pos += 1
    return tok.decode(out), first


def remember(entries, keys, rows):
    cache = lm.make_cache(); t0 = time.time()
    append_entries(cache, entries, rope, np.arange(rows), index.index_dim, dtype=mx.bfloat16, index_keys=keys)
    mx.eval([cache[l].keys for l in QL])
    return cache, time.time() - t0


chat = lambda user: tok.encode(f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", add_special_tokens=False).ids
docs = json.loads(args.docs.read_text())["docs"]
load = lambda d: {l: z[f"l{l}"] for z in [np.load(args.taps / f"{d['id']}.npz")] for l in GL}
reader.read({l: x[:64] for l, x in load(docs[0]).items()})                    # warm the kernels so timings are steady-state
report = {"label": args.label, "index": args.index, "index_sha256": index.sha256, "gain_power": args.gain_power, "index_gain_power": args.index_gain_power, "docs": []}
args.out.parent.mkdir(parents=True, exist_ok=True)
check = None
for n, d in enumerate(docs):
    other = next(o for o in docs[n + 1:] + docs[:n] if o["target"] == d["target"] and o["id"] != d["id"])
    latents = load(d)
    t0 = time.time(); entries, keys, order, which = reader.read(latents); t_translate = time.time() - t0; rows = len(order)
    if check is None:                                                         # the GPU reader must agree with the NumPy reference
        ref = corrected.read({l: x[:512].astype(np.float32) for l, x in latents.items()}, args.gain_power)
        got = reader.read({l: x[:512] for l, x in latents.items()})[0]
        check = max(float(np.abs(ref[l][0] - got[l][0].astype(np.float32)).max() / np.abs(ref[l][0]).max()) for l in QL)
        report["mlx_vs_numpy_max_rel_error"] = check
    wrong_entries, wrong_keys, wrong_order, _ = reader.read(load(other)); wrong_rows = len(wrong_order)
    doc_ids = tok.encode(d["text"], add_special_tokens=False).ids
    native = lm.make_cache(); t0 = time.time(); prefill(doc_ids, native); t_prefill = time.time() - t0
    own = tap(native, list(QL), rope, 0, len(doc_ids))
    own_keys = {l: np.array(native[l].index_keys[0, :len(doc_ids)].astype(mx.float32)) for l in QL}
    sel = [float(np.mean([np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9) for a, b in zip(own_keys[l][:: max(1, len(doc_ids) // 256)], keys[l][:: max(1, rows // 256)])])) for l in QL] if rows == len(doc_ids) else None
    diagnostics = {}
    if len(d.get("writer_of_reader", [])) == len(doc_ids):
        w_of_r = np.asarray(d["writer_of_reader"]); last_row = np.cumsum(np.bincount(order, minlength=int(w_of_r.max()) + 1)) - 1      # row of (t, j = 0)
        readers_of = {}
        for i, tkn in enumerate(w_of_r):
            readers_of.setdefault(int(tkn), []).append(i)
        nearest = lambda tkn: readers_of.get(tkn) or readers_of[min(readers_of, key=lambda u: abs(u - tkn))]
        reader_of_row = np.asarray([nearest(int(tkn))[max(0, len(nearest(int(tkn))) - 1 - int(j))] for tkn, j in zip(order, which)])
        diagnostics = {"drift_k_own_v": ({l: (entries[l][0], own[l][1][reader_of_row]) for l in QL}, keys, rows),           # addressing translated, content own
                       "own_k_drift_v": ({l: (own[l][0][reader_of_row], entries[l][1]) for l in QL}, keys, rows),           # addressing own, content translated
                       "drift_kv_own_keys": (entries, {l: own_keys[l][reader_of_row] for l in QL}, rows),
                       "own_kv_drift_keys": (own, {l: keys[l][last_row[w_of_r]] for l in QL}, len(doc_ids))}
        sel = [float(np.mean([np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9) for a, b in zip(own_keys[l][::16].astype(np.float32), keys[l][last_row[w_of_r]][::16].astype(np.float32))])) for l in QL]
    for name, r in variants.items():
        if isinstance(r, tuple):
            t1 = time.time(); e, o, _, xc = r[0].read_rows({l: x.astype(np.float32) for l, x in latents.items()}, args.gain_power); ik = r[1].read(xc, o, args.index_gain_power)
            diagnostics[name] = ({l: (k.astype(np.float16), v.astype(np.float16)) for l, (k, v) in e.items()}, ik, len(o)); cpu_translate = round(time.time() - t1, 2)
            continue
        e = r.read({l: x.astype(np.float32) for l, x in latents.items()}, args.gain_power)
        stack = np.concatenate([latents[l].astype(np.float32) for l in GL], 1) - base.input_mean
        diagnostics[name] = ({l: (k.astype(np.float16), v.astype(np.float16)) for l, (k, v) in e.items()}, index.read(stack), stack.shape[0])
    for g, r in sweep.items():
        e, k, o, _ = r.read(latents); diagnostics[f"drift@{g}"] = (e, k, len(o))
    head = tok.encode("<|im_start|>user\n" + d["text"], add_special_tokens=False).ids
    text_cache = lm.make_cache(); prefill(head, text_cache)
    del native
    hits, timing, qrows = {}, {"append": [], "first_token_memory": [], "first_token_text_tail": []}, []
    for q in d["questions"]:
        tail = tok.encode(f"\n\n{q['question']}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", add_special_tokens=False).ids
        texts = {}
        if not args.only or "text" in args.only:
            texts["text"], ft = answer(tail, clone(text_cache), len(head)); timing["first_token_text_tail"].append(ft)
        for name, (e, k, r) in {n_: v_ for n_, v_ in {"own_real": (own, own_keys, len(doc_ids)), "drift_keys": (entries, keys, rows), "drift_zero": (entries, None, rows), "wrong_memory": (wrong_entries, wrong_keys, wrong_rows), **diagnostics}.items() if not args.only or n_ in args.only}.items():
            cache, ta = remember(e, k, r)
            texts[name], ft = answer(chat(q["question"]), cache, r)
            if name == "drift_keys":
                timing["append"].append(ta); timing["first_token_memory"].append(ft)
        if not args.only or "none" in args.only:
            texts["none"], _ = answer(chat(q["question"]), lm.make_cache(), 0)
        for c, text in texts.items():
            hits[c] = hits.get(c, 0) + int(q["answer"].lower() in text.lower())
        qrows.append({**q, **texts})
    med = lambda v: round(float(np.median(v)), 4) if v else None
    row = {"id": d["id"], "reader_tokens": len(doc_ids), "writer_tokens": int(next(iter(latents.values())).shape[0]), "memory_rows": rows, "questions": len(d["questions"]),
           "accuracy": {c: round(v / len(d["questions"]), 3) for c, v in hits.items()},
           "seconds": {"text_prefill": round(t_prefill, 2), "translate_on_gpu": round(t_translate, 3), "append": med(timing["append"]),
                       "question_first_token_with_memory": med(timing["first_token_memory"]), "question_first_token_after_text": med(timing["first_token_text_tail"])},
           "bytes": {"writer_entries_float16": int(sum(x.nbytes for x in latents.values())), "text_utf8": len(d["text"].encode())},
           "selector_key_cosine_vs_own": sel, "rows": qrows}
    report["docs"].append(row); print(json.dumps({k: v for k, v in row.items() if k != "rows"}), flush=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    del text_cache, own, own_keys, entries, keys, wrong_entries, wrong_keys
report["peak_gb"] = round(mx.get_peak_memory() / 2**30, 1)
args.out.write_text(json.dumps(report, indent=2) + "\n")
