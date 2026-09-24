"""The joining side of a drop-in: Qwen attaches to GLM's cache pulled over MCDMA and answers without reading the project.

Run on the Studio with oMLX's interpreter, handoffd up. Each job is one project context: where spark_dropin_export.py
left GLM's export, the span its context tokens take, the context text and its questions. handoffd RDMA-reads the export
into shared memory; the context's GLM latents are translated on the GPU into Qwen rows (the loop's row stack, with its
selector keys, or with --context-reader a trained contextual reader) and linear-attention inputs (the state translator). For each question a fresh Qwen prefills only its own
head, attaches the rows and advances its state, then reads the question. The same question is answered from text too:
the project files read in the prompt. Writes one JSON line per question with both answers and both times to first token,
and one per context with the pull and translation costs. This process opens no verbs objects.
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
from drift.serving.glm53_handoff import read_latents
from drift.serving.handoffd_client import HandoffdClient
from drift.serving.mcdma_forward import validate_translation
from drift.serving.omlx_cache import Rope, append_entries, selector_block_reset
from drift.serving.studio_guard import acquire
from drift.translate import ridge_map
from drift.translate.context_reader import ContextRows
from drift.translate.fanout import CorrectedFanoutReader, FanoutReader
from drift.translate.index_keys import IndexKeyReader
from drift.translate.mlx_reader import MlxForwardReader
from drift.translate.qwen_rows import split_rows, zero_selector_keys
from drift.translate.stacked import StackedReader

SYSTEM = ("You are linked to another AI model through a shared memory that fills while you work. What your partner knows "
          "and writes arrives in that memory, never in this chat. Use it as your own recollection.")
FRAME = "Project files, from our shared memory:"
parser = argparse.ArgumentParser()
parser.add_argument("--jobs", type=Path, required=True, help="json list of {id, peer, remote, start, tokens, text, questions: [{id, question, answer}]}")
parser.add_argument("--state", type=Path, required=True, help="the GLM-to-Qwen state translator")
parser.add_argument("--artifacts", type=Path, default=Path("local/live"))
parser.add_argument("--socket", default="/tmp/handoffd.sock")
parser.add_argument("--max-new", type=int, default=120)
parser.add_argument("--budget", type=int, help="attend densely up to this many tokens: every full-attention layer's sparse-selection budget is raised to it")
parser.add_argument("--context-reader", type=Path, help="a trained contextual reader's folder: GLM's rows come from it instead of the loop's stack")
parser.add_argument("--reader-window", type=int, help="read longer contexts in overlapping windows of this many tokens, as long as the reader's training windows")
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
if args.context_reader is not None and not args.budget:
    raise SystemExit("a contextual reader writes no selector keys: raise --budget past the memory so it is attended densely")
client = HandoffdClient(args.socket)
print("handoffd:", client.status(), flush=True)                      # fail before loading the model if the daemon is not up
acquire("studio_dropin.py", need_gb=150)
ck = Path("~/.omlx/models/Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP").expanduser()
cfg = json.loads((ck / "config.json").read_text()); t = cfg.get("text_config", cfg); rp = t.get("rope_parameters") or {}
rope = Rope(float(rp.get("rope_theta", 1e7)), int(t["head_dim"] * float(rp.get("partial_rotary_factor", 1.0))))
tok = Tokenizer.from_file(str(ck / "tokenizer.json"))
with _force_qwen4_exp_sanitize_on_load(ck):
    model = load_model(ck)
lm = model.language_model


def dense_up_to(budget):
    """Every full-attention layer attends the whole prefix while it fits the budget, so appended memory needs no selector keys."""
    for layer in lm.model.layers:
        if not getattr(layer, "is_linear", False):
            indexer = layer.self_attn.indexer
            indexer.token_budget, indexer.block_topk = budget, budget // indexer.compress_ratio


if args.budget:
    dense_up_to(args.budget)
GL, QL, INDEX_DIM = tuple(3 + 4 * i for i in range(11)), tuple(3 + 4 * i for i in range(12)), 128
fwd, CONTEXT = None, None
if args.context_reader is not None:
    CONTEXT = ContextRows(args.context_reader, window=args.reader_window)
    CONTEXT.read(np.zeros((8, len(GL) * 512), np.float32))            # warm the kernels
else:
    stack = StackedReader.load(args.artifacts / "stacked3.npz", GL, QL, kv_heads=2, head_dim=256)
    fwd = MlxForwardReader(CorrectedFanoutReader.load(args.artifacts / "v4_correction.safetensors", FanoutReader.load(args.artifacts / "fanout3.npz", stack)),
                           IndexKeyReader.load(args.artifacts / "index3.npz", QL, stack.sha256), 1.5)
    fwd.read({l: np.zeros((8, 512), np.float32) for l in GL})         # warm the kernels
state = ridge_map.load(args.state)
MEAN, BASIS = mx.array(state["mean"]), mx.array(state["basis"])
LAYERS = {index: (mx.array(weight * gain[None, :]), mx.array(bias)) for index, (weight, bias, gain) in state["layers"].items()}
STOP = {tok.token_to_id("<|im_end|>"), tok.token_to_id("<|endoftext|>")}
encode = lambda text: tok.encode(text, add_special_tokens=False).ids
HEAD = encode(f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{FRAME}\n")
tail = lambda question: encode(f"\n\n{question}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")


def run(cache, ids, start):
    logits = lm(mx.array(np.asarray(ids, dtype=np.int32))[None], cache=cache, position_ids=mx.arange(start, start + len(ids), dtype=mx.int32)[None]).logits[0, -1]
    mx.eval(logits)
    return logits


def generate(cache, logits, position):
    out = []
    while len(out) < args.max_new:
        token = int(mx.argmax(logits).item())
        if token in STOP:
            break
        out.append(token)
        logits = run(cache, [token], position)
        position += 1
    return tok.decode(out)


def translate(latents):
    """GLM latents of the context -> Qwen rows with selector keys, and each linear-attention layer's inputs, on the GPU."""
    n, features = len(next(iter(latents.values()))), np.concatenate([latents[l] for l in sorted(latents)], axis=1)
    if CONTEXT is not None:                                            # attended densely, so its rows need no selector keys
        entries, keys, order = split_rows(CONTEXT.read(features), QL), zero_selector_keys(n, QL, INDEX_DIM), np.arange(n)
    else:
        entries, keys, order, _ = fwd.read(latents)
    validate_translation(entries, keys, order, n, QL, INDEX_DIM)
    reduced = (mx.array(features) - MEAN) @ BASIS
    inputs = {index: (reduced @ weight + bias)[None].astype(mx.bfloat16) for index, (weight, bias) in LAYERS.items()}
    mx.eval(list(inputs.values()))
    return entries, keys, np.asarray(order, dtype=np.int64), inputs


def attach(memory, n):
    """A fresh Qwen: its own head, then the translated rows at the context's positions and the state advanced over them."""
    entries, keys, order, inputs = memory
    cache = lm.make_cache()
    run(cache, HEAD, 0)
    append_entries(cache, entries, rope, len(HEAD) + order, INDEX_DIM, dtype=mx.bfloat16, index_keys=keys)
    for layer in QL:
        reset = selector_block_reset(cache[layer])
        if reset is not None:
            reset()
    for index, value in inputs.items():
        lm.model.layers[index].linear_attn(value, mask=None, cache=cache[index])
    mx.eval([part for index in inputs for part in (cache[index][0], cache[index][1]) if part is not None])
    return cache, len(HEAD) + n


args.out.parent.mkdir(parents=True, exist_ok=True)
with args.out.open("w") as sink:
    for job in json.loads(args.jobs.read_text()):
        started = time.time(); pulled = client.pull(job["peer"], job["remote"], offset=0, unlink=True); pull_s = time.time() - started
        started = time.time()
        latents = {l: v[job["start"]:job["start"] + job["tokens"]] for l, v in read_latents(client.view(job["peer"], 0, pulled.bytes)).items()}
        parse_s = time.time() - started
        started = time.time(); memory = translate(latents); translate_s = time.time() - started
        sink.write(json.dumps({"context": job["id"], "rows": str(args.context_reader) if CONTEXT is not None else "stack", "bytes": pulled.bytes, "pull_s": round(pull_s, 4), "rdma_gbit_s": pulled.gbit_s,
                               "parse_s": round(parse_s, 4), "translate_s": round(translate_s, 4), "glm_tokens": job["tokens"]}) + "\n")
        context_ids = encode(job["text"])
        for q in job["questions"]:
            started = time.time(); cache, position = attach(memory, job["tokens"]); logits = run(cache, tail(q["question"]), position)
            drift_first = time.time() - started
            drift = generate(cache, logits, position + len(tail(q["question"])))
            started = time.time(); cache = lm.make_cache(); ids = HEAD + context_ids + tail(q["question"]); logits = run(cache, ids, 0)
            text_first = time.time() - started
            text = generate(cache, logits, len(ids))
            sink.write(json.dumps({"context": job["id"], "id": q["id"], "answer": q["answer"], "drift": drift, "text": text,
                                   "drift_first_s": round(drift_first, 4), "text_first_s": round(text_first, 4),
                                   "drift_prefill_tokens": len(HEAD) + len(tail(q["question"])), "text_prefill_tokens": len(ids)}) + "\n")
            sink.flush()
        print(json.dumps({"context": job["id"], "questions": len(job["questions"]), "pull_s": round(pull_s, 4), "translate_s": round(translate_s, 4)}), flush=True)
