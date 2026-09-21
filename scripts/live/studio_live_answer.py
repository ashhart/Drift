"""Reader side of a live exchange (Studio, oMLX runtime). Qwen answers a question under several
memory conditions; greedy decoding. The passage text is only present for the explicit controls."""
import argparse, json, time
from pathlib import Path
import numpy as np
from omlx.patches.mlx_vlm_qwen4_exp_compat import apply_mlx_vlm_qwen4_exp_compat_patch
apply_mlx_vlm_qwen4_exp_compat_patch()
import mlx.core as mx
from tokenizers import Tokenizer
from mlx_vlm.utils import load_model
from omlx.engine.vlm import _force_qwen4_exp_sanitize_on_load
from drift.serving.omlx_cache import Rope, inject_prefix, kv_layer_indices, tap

parser = argparse.ArgumentParser()
parser.add_argument("--jobs", type=Path, required=True, help="json list: {id, question, answer?, memory (npz), wrong_memory?, passage? (controls only)}")
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--max-new", type=int, default=48)
parser.add_argument("--checkpoint", default="~/.omlx/models/Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP")
args = parser.parse_args()
ck = Path(args.checkpoint).expanduser()
cfg = json.loads((ck / "config.json").read_text()); t = cfg.get("text_config", cfg); rp = t.get("rope_parameters") or {}
rope = Rope(float(rp.get("rope_theta", 1e7)), int(t["head_dim"] * float(rp.get("partial_rotary_factor", 1.0))))
tok = Tokenizer.from_file(str(ck / "tokenizer.json"))
from drift.serving.studio_guard import acquire
acquire('studio_live_answer.py', need_gb=150)        # one model process at a time, preflight memory check, hard MLX limits
with _force_qwen4_exp_sanitize_on_load(ck):
    model = load_model(ck)
lm = model.language_model
STOP = {tok.token_to_id("<|im_end|>"), tok.token_to_id("<|endoftext|>")}


def step(ids, cache):
    out = lm(mx.array(np.asarray(ids, dtype=np.int32))[None], cache=cache).logits[0]
    mx.eval(out)
    return np.array(out.astype(mx.float32))


probe = lm.make_cache(); step([1, 2, 3, 4], probe)
INDEX_DIM = int(probe[kv_layer_indices(probe)[0]].index_keys.shape[-1])


def with_memory(entries):
    cache = lm.make_cache()
    n = next(iter(entries.values()))[0].shape[0]
    inject_prefix(cache, entries, rope, dtype=mx.bfloat16, index_keys={l: mx.zeros((1, n, INDEX_DIM), dtype=mx.bfloat16) for l in kv_layer_indices(cache)})
    return cache


def load(path, limit=None):
    z = np.load(path)
    return {int(k[1:]): (z[k][:limit].astype(np.float32), z["v" + k[1:]][:limit].astype(np.float32)) for k in z.files if k.startswith("k")}


def chat(question, context=None):
    user = question if context is None else f"{context}\n\n{question}"
    return tok.encode(f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", add_special_tokens=False).ids


def answer(prompt, cache, gold=None):
    logits = step(prompt, cache)[-1]
    out = {}
    if gold is not None:                                         # log-prob of the reference answer, teacher-forced on a copy
        import copy
        gold_ids = tok.encode(gold, add_special_tokens=False).ids
        side, last, total = copy.deepcopy(cache), logits, 0.0
        for g in gold_ids:
            z = last - last.max(); total += float(z[g] - np.log(np.exp(z).sum()))
            last = step([g], side)[-1]
        out["answer_logprob"] = total
    generated = []
    for _ in range(args.max_new):
        nxt = int(logits.argmax())
        if nxt in STOP:
            break
        generated.append(nxt)
        logits = step([nxt], cache)[-1]
    out["text"] = tok.decode(generated)
    return out


results, started = [], time.time()
for job in json.loads(args.jobs.read_text()):
    q, gold = job["question"], job.get("answer")
    row = {"id": job["id"], "question": q, "answer": gold}
    row["no_memory"] = answer(chat(q), lm.make_cache(), gold)
    if job.get("memory"):
        memory = load(job["memory"])
        row["drift"] = answer(chat(q), with_memory(memory), gold)
        row["drift"]["entries"] = int(next(iter(memory.values()))[0].shape[0])
    for name, path in (job.get("memories") or {}).items():
        row[name] = answer(chat(q), with_memory(load(path)), gold)
    if job.get("wrong_memory"):
        row["wrong_memory"] = answer(chat(q), with_memory(load(job["wrong_memory"])), gold)
    if job.get("passage"):                                       # controls: need the text, so they are NOT the drift channel
        ids = tok.encode(job["passage"], add_special_tokens=False).ids
        native = lm.make_cache(); step(ids, native)
        row["own_kv_only"] = answer(chat(q), with_memory(tap(native, kv_layer_indices(native), rope, 0, len(ids))), gold)
        row["text_in_prompt"] = answer(chat(q, job["passage"]), lm.make_cache(), gold)
    results.append(row)
    args.out.write_text(json.dumps({"seconds": round(time.time() - started, 1), "results": results}, indent=2) + "\n")
print(json.dumps({"jobs": len(results), "seconds": round(time.time() - started, 1)}))
