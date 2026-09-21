"""Studio (oMLX runtime), reader only, no partner model: how does cache-injected memory behave BEYOND the sparse selector's
budget, and what does injection cost against reading the same text? EXPLORATORY diagnostic for roadmap D3.

One synthetic document per length (unique thing/place facts), needle questions at spread depths, conditions:
  text          the document is prefilled as text (time measured), then the question
  own_real      the reader's own canonical K/V AND its raw selector keys are injected into a fresh cache, then the question
  own_zero      the same K/V with ZERO selector keys (what Drift does today)
  none          question only"""
import argparse, copy, json, random, time
from pathlib import Path
import numpy as np
from omlx.patches.mlx_vlm_qwen4_exp_compat import apply_mlx_vlm_qwen4_exp_compat_patch
apply_mlx_vlm_qwen4_exp_compat_patch()
import mlx.core as mx
from tokenizers import Tokenizer
from mlx_vlm.utils import load_model
from omlx.engine.vlm import _force_qwen4_exp_sanitize_on_load
from drift.serving.omlx_cache import Rope, append_entries, kv_layer_indices, tap

parser = argparse.ArgumentParser()
parser.add_argument("--lengths", type=int, nargs="+", default=[1000, 2000, 4000, 8000, 16000])
parser.add_argument("--questions", type=int, default=12)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
ck = Path("~/.omlx/models/Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP").expanduser()
cfg = json.loads((ck / "config.json").read_text()); t = cfg.get("text_config", cfg); rp = t.get("rope_parameters") or {}
rope = Rope(float(rp.get("rope_theta", 1e7)), int(t["head_dim"] * float(rp.get("partial_rotary_factor", 1.0))))
tok = Tokenizer.from_file(str(ck / "tokenizer.json"))
from drift.serving.studio_guard import acquire
acquire("studio_longctx_probe.py", need_gb=160)
with _force_qwen4_exp_sanitize_on_load(ck):
    model = load_model(ck)
lm = model.language_model
STOP = {tok.token_to_id("<|im_end|>"), tok.token_to_id("<|endoftext|>")}
THINGS = "compressor,weather buoy,projector,espresso machine,forklift,telescope mirror,server rack,kiln,defibrillator,piano,beehive,water pump,printing press,drone,generator,freezer,microscope,loom,sailboat mast,tractor".split(",")
PLACES = [f"{a} {b}" for a in "Harbour,Lyngen,Riverside,Castlefield,Quarry,Pier,Clinic,Bakery,Ferry,Mill,Thistle,Saltmarsh,Pennywell,Gorse,Larkspur,Whinfell,Alder,Tamar,Kestrel,Dockside".split(",") for b in ("depot", "annex")]
NAMES = "Marta Ingrid Tomas Aisha Kenji Lucia Pavel Noor Emeka Sofia Callum Priya Mateo Hana Dmitri Amara Jonas Leila Rafael Yuki".split()
COLOURS = "red blue green yellow orange purple black white grey brown pink teal".split()


def document(target_tokens, rng):
    combos = [(a, b) for a in THINGS for b in PLACES]; rng.shuffle(combos)
    facts, parts, count = [], [], 0
    for thing, place in combos:
        f = {"thing": thing, "place": place, "code": str(rng.randint(1000, 9999)), "weight": str(rng.randint(12, 980)), "colour": rng.choice(COLOURS), "person": rng.choice(NAMES)}
        text = f"The {thing} at the {place} is looked after by {f['person']}. Its access code is {f['code']}. It weighs {f['weight']} kilograms and its casing is {f['colour']}. "
        parts.append(text); facts.append(f); count += len(tok.encode(text, add_special_tokens=False).ids)
        if count >= target_tokens:
            break
    return "".join(parts), facts


def clone(cache):
    """Branch a prefilled cache without deepcopy (the selector's pooled state holds a function once blocks exist)."""
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
    logits, out, pos = prefill(prompt_ids, cache, start), [], start + len(prompt_ids)
    for _ in range(limit):
        nxt = int(mx.argmax(logits).item())
        if nxt in STOP:
            break
        out.append(nxt); logits = step([nxt], cache, pos); pos += 1
    return tok.decode(out)


chat = lambda user: tok.encode(f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", add_special_tokens=False).ids
report = {"label": "EXPLORATORY", "lengths": []}
args.out.parent.mkdir(parents=True, exist_ok=True)
for length in args.lengths:
    rng = random.Random(length)
    doc, facts = document(length, rng)
    doc_ids = tok.encode(doc, add_special_tokens=False).ids
    n = len(doc_ids)
    native = lm.make_cache()
    t0 = time.time(); prefill(doc_ids, native); t_prefill = time.time() - t0
    layers = kv_layer_indices(native)
    entries = tap(native, layers, rope, 0, n)
    index = {l: np.array(native[l].index_keys[0, :n].astype(mx.float32)) for l in layers}
    index_dim = next(iter(index.values())).shape[1]
    head = tok.encode("<|im_start|>user\n" + doc, add_special_tokens=False).ids
    text_cache = lm.make_cache(); prefill(head, text_cache)
    picks = [facts[int(round(i * (len(facts) - 1) / (args.questions - 1)))] for i in range(args.questions)]
    hits, t_inject = {"text": 0, "own_real": 0, "own_zero": 0, "none": 0}, []
    for f in picks:
        question = f"What is the access code of the {f['thing']} at the {f['place']}? Answer with the code only."
        tail = tok.encode(f"\n\n{question}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", add_special_tokens=False).ids
        hits["text"] += f["code"] in answer(tail, clone(text_cache), len(head))
        for name, keys in (("own_real", index), ("own_zero", None)):
            cache = lm.make_cache()
            t0 = time.time(); append_entries(cache, entries, rope, np.arange(n), index_dim, dtype=mx.bfloat16, index_keys=keys)
            mx.eval([cache[l].keys for l in layers]); t_inject.append(time.time() - t0)
            hits[name] += f["code"] in answer(chat(question), cache, n)
        hits["none"] += f["code"] in answer(chat(question), lm.make_cache(), 0)
    row = {"tokens": n, "facts": len(facts), "questions": len(picks), "accuracy": {k: round(v / len(picks), 3) for k, v in hits.items()},
           "seconds": {"text_prefill": round(t_prefill, 2), "inject_median": round(float(np.median(t_inject)), 3)}, "prefill_tokens_per_second": round(n / t_prefill)}
    report["lengths"].append(row); print(json.dumps(row), flush=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    del native, text_cache, entries, index
report["peak_gb"] = round(mx.get_peak_memory() / 2**30, 1)
args.out.write_text(json.dumps(report, indent=2) + "\n")
