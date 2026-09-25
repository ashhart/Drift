"""The payoff of attaching a cache instead of reading, at long contexts, with one reader feeding several joiners.

Run on the Studio with oMLX's interpreter, handoffd up. Each context (payoff_contexts.py) carries needle questions, one
per joiner. Four ways a joiner gets the context:
  none   no context: its head and question only, the control for needles a model answers from pretraining;
  text   a fresh Qwen prefills its head, the context and its question;
  same   a resident Qwen reads the head and the context once and exports its full-attention rows and recurrent state;
         each joiner prefills its head, writes the rows into its cache, takes the state, and prefills its question;
  glm    GLM's export of the same context (spark_dropin_export.py) is pulled over MCDMA into a file (or was, by
         payoff_prepare.sh, whose pull the job records; a long context comes as a prefix export and a delta export,
         stitched here), translated once by the contextual reader in windows and the
         state translator, and advanced into a template's recurrent state once; each joiner prefills its head, writes
         the translated rows, takes the template's state, and prefills its question.
Prefill runs in chunks, as a server does; only the last chunk's last logits are computed.
Writes one JSON line per context and way with the one-time costs, and one per joiner with its time to first token, the
tokens it prefilled, and whether its answer names the needle. Both backbones stay frozen.

  PYTHONPATH=. $OMLX_PY scripts/live/studio_payoff.py --contexts out/payoff_contexts.json --jobs out/payoff_jobs.json \\
      --context-reader out/reader_answer3fg --state out/reader_answer3fg/state.npz --out out/payoff.jsonl
"""
import argparse, json, re, time
from pathlib import Path
import numpy as np
from omlx.patches.mlx_vlm_qwen4_exp_compat import apply_mlx_vlm_qwen4_exp_compat_patch
apply_mlx_vlm_qwen4_exp_compat_patch()
import mlx.core as mx
from mlx_vlm.utils import load_model
from omlx.engine.vlm import _force_qwen4_exp_sanitize_on_load
from tokenizers import Tokenizer
from drift.eval.answer_match import matches
from drift.serving.glm53_delta import read_latents_span, stitch
from drift.serving.glm53_handoff import read_latents
from drift.serving.handoffd_client import HandoffdClient
from drift.serving.omlx_cache import Rope, append_entries, copy_nonkv_state, selector_block_reset, tap
from drift.serving.studio_guard import acquire
from drift.translate import ridge_map
from drift.translate.context_reader import ContextRows
from drift.translate.qwen_rows import split_rows
from mlx.utils import tree_flatten

SYSTEM = ("You are linked to another AI model through a shared memory that fills while you work. What your partner knows "
          "and writes arrives in that memory, never in this chat. Use it as your own recollection.")
FRAME = "Project files, from our shared memory:"
parser = argparse.ArgumentParser()
parser.add_argument("--contexts", type=Path, required=True, help="json list of {id, text, questions: [{id, question, answer}]}")
parser.add_argument("--jobs", type=Path, help="json list of {id, peer, remote or local, start, tokens, glm_read_s}: GLM's exports, for the glm way")
parser.add_argument("--context-reader", type=Path, help="the contextual reader for the glm way")
parser.add_argument("--state", type=Path, help="the GLM-to-Qwen state translator for the glm way")
parser.add_argument("--reader-window", type=int, default=1536)
parser.add_argument("--ways", default="none,text,same,glm")
parser.add_argument("--chunk", type=int, default=2048, help="prefill chunk, tokens")
parser.add_argument("--joiners", type=int, default=5)
parser.add_argument("--max-new", type=int, default=80)
parser.add_argument("--budget", type=int, default=262144, help="attend densely up to this many tokens, every way alike")
parser.add_argument("--pull-dir", type=Path, default=Path("/tmp/drift-pull"))
parser.add_argument("--socket", default="/tmp/handoffd.sock")
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
ways = args.ways.split(",")
if "glm" in ways and not (args.jobs and args.context_reader and args.state):
    raise SystemExit("the glm way needs --jobs, --context-reader and --state")
client = HandoffdClient(args.socket, timeout_s=600) if "glm" in ways else None
if client:
    print("handoffd:", client.status(), flush=True)
acquire("studio_payoff.py", need_gb=150)
ck = Path("~/.omlx/models/Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP").expanduser()
cfg = json.loads((ck / "config.json").read_text()); t = cfg.get("text_config", cfg); rp = t.get("rope_parameters") or {}
rope = Rope(float(rp.get("rope_theta", 1e7)), int(t["head_dim"] * float(rp.get("partial_rotary_factor", 1.0))))
tok = Tokenizer.from_file(str(ck / "tokenizer.json"))
with _force_qwen4_exp_sanitize_on_load(ck):
    model = load_model(ck)
lm = model.language_model
for layer in lm.model.layers:                                            # dense attention over the whole memory, every way alike
    if not getattr(layer, "is_linear", False):
        layer.self_attn.indexer.token_budget, layer.self_attn.indexer.block_topk = args.budget, args.budget // layer.self_attn.indexer.compress_ratio
QL, INDEX_DIM = tuple(i for i, layer in enumerate(lm.model.layers) if not getattr(layer, "is_linear", False)), 128
LINEAR = tuple(i for i, layer in enumerate(lm.model.layers) if getattr(layer, "is_linear", False))
STOP = {tok.token_to_id("<|im_end|>"), tok.token_to_id("<|endoftext|>")}
encode = lambda text: tok.encode(text, add_special_tokens=False).ids
HEAD = encode(f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{FRAME}\n")
tail = lambda question: encode(f"\n\n{question}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
CONTEXT = ContextRows(args.context_reader, window=args.reader_window) if "glm" in ways else None
if "glm" in ways:
    state = ridge_map.load(args.state)
    MEAN, BASIS = mx.array(state["mean"]), mx.array(state["basis"])
    STATE_LAYERS = {index: (mx.array(weight * gain[None, :]), mx.array(bias)) for index, (weight, bias, gain) in state["layers"].items()}


def evaluate(cache):
    mx.eval([a for c in cache for _, a in tree_flatten(getattr(c, "state", None)) if isinstance(a, mx.array)])


def run(cache, ids, start):
    """Prefill ids at positions start.. in chunks; the logits after the last one."""
    for offset in range(0, len(ids), args.chunk):
        part = ids[offset:offset + args.chunk]
        out = lm(mx.array(np.asarray(part, dtype=np.int32))[None], cache=cache, position_ids=mx.arange(start + offset, start + offset + len(part), dtype=mx.int32)[None])
        if offset + args.chunk < len(ids):
            evaluate(cache)
    logits = out.logits[0, -1]
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


def settle(cache):
    """The selector's block state after appended rows, and every layer's arrays evaluated."""
    for layer in QL:
        reset = selector_block_reset(cache[layer])
        if reset is not None:
            reset()
    mx.eval([a for layer in LINEAR for a in (cache[layer][0], cache[layer][1]) if a is not None])


def join(entries, n, source_state, question):
    """A joiner: its head, the memory's rows at the context's positions, the memory's recurrent state, its question."""
    started = time.time()
    cache = lm.make_cache()
    run(cache, HEAD, 0)
    append_entries(cache, entries, rope, np.arange(len(HEAD), len(HEAD) + n), INDEX_DIM, dtype=mx.bfloat16)
    copy_nonkv_state(source_state, cache, QL)
    settle(cache)
    attach_s = time.time() - started
    logits = run(cache, tail(question), len(HEAD) + n)
    first = time.time() - started
    return cache, logits, attach_s, first


args.out.parent.mkdir(parents=True, exist_ok=True)
jobs = {j["id"]: j for j in json.loads(args.jobs.read_text())} if args.jobs else {}
with args.out.open("w") as sink:
    def write(row):
        sink.write(json.dumps(row) + "\n"); sink.flush()
        print(json.dumps({k: v for k, v in row.items() if k not in ("answer_text",)})[:300], flush=True)

    for ctx in json.loads(args.contexts.read_text()):
        ids, questions = encode(ctx["text"]), ctx["questions"][:args.joiners]
        write({"context": ctx["id"], "way": "context", "qwen_tokens": len(ids), "chars": len(ctx["text"])})
        if "none" in ways:
            for q in questions:
                started = time.time()
                cache = lm.make_cache()
                logits = run(cache, HEAD + tail(q["question"]), 0)
                first = time.time() - started
                answer = generate(cache, logits, len(HEAD) + len(tail(q["question"])))
                write({"context": ctx["id"], "way": "none", "id": q["id"], "first_s": round(first, 3), "prefill_tokens": len(HEAD) + len(tail(q["question"])),
                       "hit": matches(q["answer"], q["question"], answer), "answer_text": answer})
                del cache
        if "text" in ways:
            for q in questions:
                started = time.time()
                cache = lm.make_cache()
                logits = run(cache, HEAD + ids + tail(q["question"]), 0)
                first = time.time() - started
                answer = generate(cache, logits, len(HEAD) + len(ids) + len(tail(q["question"])))
                write({"context": ctx["id"], "way": "text", "id": q["id"], "first_s": round(first, 3), "prefill_tokens": len(HEAD) + len(ids) + len(tail(q["question"])),
                       "hit": matches(q["answer"], q["question"], answer), "answer_text": answer})
                del cache
        if "same" in ways:
            started = time.time()
            resident = lm.make_cache()
            run(resident, HEAD + ids, 0)
            read_s = time.time() - started
            started = time.time()
            rows = tap(resident, QL, rope, len(HEAD), len(HEAD) + len(ids))
            export_s = time.time() - started
            row_bytes = sum(k.nbytes + v.nbytes for k, v in rows.values()) // 2        # as bfloat16 on a wire
            write({"context": ctx["id"], "way": "same", "one_time": {"resident_read_s": round(read_s, 3), "export_s": round(export_s, 3),
                   "prefill_tokens": len(HEAD) + len(ids), "row_bytes": int(row_bytes)}})
            for q in questions:
                cache, logits, attach_s, first = join(rows, len(ids), resident, q["question"])
                answer = generate(cache, logits, len(HEAD) + len(ids) + len(tail(q["question"])))
                write({"context": ctx["id"], "way": "same", "id": q["id"], "first_s": round(first, 3), "attach_s": round(attach_s, 3),
                       "prefill_tokens": len(HEAD) + len(tail(q["question"])), "hit": matches(q["answer"], q["question"], answer), "answer_text": answer})
                del cache
            del resident, rows
        if "glm" in ways and ctx["id"] in jobs:
            job = jobs[ctx["id"]]
            if job.get("local"):                                        # pulled over MCDMA beforehand, one export at a time
                local, pull = Path(job["local"]), {"pull_s": job["pull"]["client_s"], "bytes": job["pull"]["bytes"], "rdma_gbit_s": job["pull"]["gbit_s"], "pulled": "before"}
                if job.get("local_delta"):                              # a long read in two: a prefix export and a delta export
                    pull.update(pull_s=round(pull["pull_s"] + job["pull_delta"]["client_s"], 3), bytes=pull["bytes"] + job["pull_delta"]["bytes"],
                                rdma_gbit_s=round((job["pull"]["gbit_s"] + job["pull_delta"]["gbit_s"]) / 2, 2), glm_reads=2)
            else:
                args.pull_dir.mkdir(parents=True, exist_ok=True)
                local = args.pull_dir / f"{ctx['id']}.bin"
                started = time.time(); pulled = client.pull_file(job["peer"], job["remote"], str(local), unlink=True)
                pull = {"pull_s": round(time.time() - started, 3), "bytes": pulled.bytes, "rdma_gbit_s": pulled.gbit_s, "pulled": "now"}
            started = time.time()
            whole = read_latents(str(local))
            if job.get("local_delta"):
                first, delta = read_latents_span(job["local_delta"])
                whole = stitch(whole, first, delta)
            latents = {l: np.asarray(v[job["start"]:job["start"] + job["tokens"]], dtype=np.float32) for l, v in whole.items()}
            parse_s = time.time() - started
            started = time.time()
            features = np.concatenate([latents[l] for l in sorted(latents)], axis=1)
            entries = split_rows(CONTEXT.read(features), QL)
            reduced = (mx.array(features) - MEAN) @ BASIS
            template = lm.make_cache()
            run(template, HEAD, 0)
            for offset in range(0, len(reduced), args.chunk):                # the state advances a chunk at a time, like a prefill
                part = reduced[offset:offset + args.chunk]
                for index, (weight, bias) in STATE_LAYERS.items():
                    lm.model.layers[index].linear_attn((part @ weight + bias)[None].astype(mx.bfloat16), mask=None, cache=template[index])
                mx.eval([a for layer in LINEAR for a in (template[layer][0], template[layer][1]) if a is not None])
            translate_s = time.time() - started
            n = job["tokens"]
            write({"context": ctx["id"], "way": "glm", "one_time": {"glm_read_s": job.get("glm_read_s"), **pull, "parse_s": round(parse_s, 3),
                   "translate_and_state_s": round(translate_s, 3), "glm_tokens": n}})
            for q in questions:
                cache, logits, attach_s, first = join(entries, n, template, q["question"])
                answer = generate(cache, logits, len(HEAD) + n + len(tail(q["question"])))
                write({"context": ctx["id"], "way": "glm", "id": q["id"], "first_s": round(first, 3), "attach_s": round(attach_s, 3),
                       "prefill_tokens": len(HEAD) + len(tail(q["question"])), "hit": matches(q["answer"], q["question"], answer), "answer_text": answer})
                del cache
            del template, entries, latents, features
            if not job.get("local"):
                local.unlink(missing_ok=True)
