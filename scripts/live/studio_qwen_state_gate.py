"""Own-cache gate for Qwen: do its own attention rows plus its own recurrent state reproduce reading the text?

Run on the Studio with oMLX's bundled interpreter. Per question a prompt frames a span with @@DRIFT@@, and four arms
answer greedily:
  text        the passage's Qwen tokens stand in the span.
  none        the span is empty.
  rows        the head is prefilled, the passage's own K/V rows from a separate read of head plus passage are appended
              at the passage's positions, then the rest of the prompt follows.
  rows_state  as rows, and every linear-attention layer also advances its state by the passage's own inputs to that
              layer, captured in the same read, by calling the layer itself on them.
With --glm-latents, t_rows places GLM's translated rows at the span; with --forward-state as well, t_rows_state adds the
translated state and t_state takes the translated state alone, with the span's positions left empty. With --dsv4-features,
the d_ arms do the same from DeepSeek V4's cache: d_rows with --dsv4-rows, d_state with --dsv4-state, d_rows_state with
both. Items whose passage has no DeepSeek features skip those arms.
Selector keys for appended rows are zeros, which stays exact while the context is within the selector's budget.
"""
import argparse, json, time
from pathlib import Path
import numpy as np
from omlx.patches.mlx_vlm_qwen4_exp_compat import apply_mlx_vlm_qwen4_exp_compat_patch
apply_mlx_vlm_qwen4_exp_compat_patch()
import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_vlm.utils import load_model
from mlx_vlm.models.qwen4_exp.language import Qwen4ExpGatedDeltaNet
from omlx.engine.vlm import _force_qwen4_exp_sanitize_on_load
from tokenizers import Tokenizer
from drift.serving.omlx_cache import Rope, append_entries, selector_block_reset, tap
from drift.serving.studio_guard import acquire
from drift.serving.mcdma_forward import validate_translation
from drift.translate.fanout import CorrectedFanoutReader, FanoutReader
from drift.translate.index_keys import IndexKeyReader
from drift.translate import ridge_map
from drift.translate.mlx_reader import MlxForwardReader
from drift.translate.qwen_rows import join_rows, split_rows
from drift.translate.stacked import StackedReader

LINK = ("You are linked to another AI model through a shared memory. A document it has read is in your memory, not in this chat; "
        "you were never shown it as text. @@DRIFT@@ Answer the user's question from what you recall from that shared memory. "
        "Give the answer directly.")
QL, INDEX_DIM = tuple(3 + 4 * i for i in range(12)), 128
parser = argparse.ArgumentParser()
parser.add_argument("--items", type=Path, required=True, help="json list of {id, passage, question, answer}")
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--max-new", type=int, default=80)
parser.add_argument("--temperature", type=float, default=0.0, help="0 decodes greedily; above 0 samples, reproducibly from --seed")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--budget", type=int, help="attend densely up to this many tokens: every full-attention layer's sparse-selection budget is raised to it")
parser.add_argument("--checkpoint", default="~/.omlx/models/Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP")
parser.add_argument("--glm-latents", type=Path, help="GLM's MLA latents per passage: adds t_rows, GLM's translated rows at the span, as the loop translates them")
parser.add_argument("--artifacts", type=Path, default=Path("local/live"))
parser.add_argument("--forward-state", type=Path, help="GLM-to-Qwen state translator: adds t_rows_state, translated rows plus translated state")
parser.add_argument("--glm-rows", type=Path, help="a fitted GLM-to-Qwen rows translator for the t_ arms, in place of the loop's stack")
parser.add_argument("--context-reader", type=Path, help="a trained contextual reader's folder for the t_ arms' rows, in place of both")
parser.add_argument("--reader-window", type=int, help="read longer contexts in overlapping windows of this many tokens, as long as the reader's training windows")
parser.add_argument("--dsv4-features", type=Path, help="DeepSeek V4 per-token features per passage, from dsv4_token_features.py")
parser.add_argument("--dsv4-rows", type=Path, help="DeepSeek-to-Qwen rows translator, one output of Qwen's full-attention K/V")
parser.add_argument("--dsv4-state", type=Path, help="DeepSeek-to-Qwen state translator, Qwen's linear-attention inputs")
parser.add_argument("--glm-rows-correction", type=Path, help="a rows correction from studio_train_memory_answer.py, added to GLM's translated rows")
parser.add_argument("--dsv4-rows-correction", type=Path, help="the same, added to DeepSeek's translated rows")
parser.add_argument("--arms", help="comma-separated arms to run, such as t_rows,t_rows_state; default every arm the options allow")
args = parser.parse_args()
KNOWN = {"text", "none", "rows", "rows_state", "t_rows", "t_rows_state", "t_state", "d_rows", "d_state", "d_rows_state"}
ARMS = set(args.arms.split(",")) if args.arms else KNOWN
if ARMS - KNOWN:
    raise SystemExit(f"unknown arms: {sorted(ARMS - KNOWN)}")
acquire("studio_qwen_state_gate.py", need_gb=150)
ck = Path(args.checkpoint).expanduser()
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
LINEAR = {id(layer.linear_attn): i for i, layer in enumerate(lm.model.layers) if getattr(layer, "is_linear", False)}
GL = tuple(3 + 4 * i for i in range(11))
fwd = None
if args.glm_latents is not None:                                        # the loop's own forward translator stack
    base = StackedReader.load(args.artifacts / "stacked3.npz", GL, QL, kv_heads=2, head_dim=256)
    fwd = MlxForwardReader(CorrectedFanoutReader.load(args.artifacts / "v4_correction.safetensors", FanoutReader.load(args.artifacts / "fanout3.npz", base)),
                           IndexKeyReader.load(args.artifacts / "index3.npz", QL, base.sha256), 1.5)
STOP = {tok.token_to_id("<|im_end|>"), tok.token_to_id("<|endoftext|>")}
CAPTURE = {"span": None, "records": {}}
original = Qwen4ExpGatedDeltaNet.__call__


def recording(self, inputs, *rest, **options):
    if CAPTURE["span"] is not None and id(self) in LINEAR:
        start, stop = CAPTURE["span"]
        CAPTURE["records"][LINEAR[id(self)]] = inputs[:, start:stop]
    return original(self, inputs, *rest, **options)                # oMLX passes speculative-verify options too


Qwen4ExpGatedDeltaNet.__call__ = recording


def run(cache, ids, start):
    logits = lm(mx.array(np.asarray(ids, dtype=np.int32))[None], cache=cache, position_ids=mx.arange(start, start + len(ids), dtype=mx.int32)[None]).logits[0, -1]
    mx.eval(logits)
    return logits


mx.random.seed(args.seed)


def flush(cache):
    """Evaluates every cache array; unread under a dense budget, the selector's raw keys stay lazy and exhaust Metal's shared events."""
    mx.eval([a for c in cache for _, a in tree_flatten(c.state) if isinstance(a, mx.array)])


def generate(cache, logits, position):
    out = []
    while len(out) < args.max_new:
        pick = mx.argmax(logits) if args.temperature <= 0 else mx.random.categorical(logits.astype(mx.float32) / args.temperature)
        token = int(pick.item())
        if token in STOP:
            break
        out.append(token)
        logits = run(cache, [token], position)
        position += 1
        if len(out) % 256 == 0:
            flush(cache)
    return tok.decode(out)


def framing(question):
    prompt = f"<|im_start|>system\n{LINK}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    a = prompt.index("@@DRIFT@@")
    encode = lambda text: tok.encode(text, add_special_tokens=False).ids
    return encode(prompt[:a]), encode(prompt[a + len("@@DRIFT@@"):])


def as_inputs(outputs):
    return {index: mx.array(value[None].astype(np.float32)).astype(mx.bfloat16) for index, value in outputs.items()}


STATE, G_ROWS, D_ROWS, D_STATE = (None if path is None else ridge_map.load(path) for path in (args.forward_state, args.glm_rows, args.dsv4_rows, args.dsv4_state))
apply_translator = ridge_map.apply


STATE_GPU = None
if STATE is not None:                                                  # on the GPU: a long context's translation would dominate on the CPU
    STATE_GPU = {"mean": mx.array(STATE["mean"]), "basis": mx.array(STATE["basis"]),
                 "layers": {index: (mx.array(weight * gain[None, :]), mx.array(bias)) for index, (weight, bias, gain) in STATE["layers"].items()}}


def translated_inputs(latents):
    """GLM latents in GLM's token order -> each linear-attention layer's input, as the loop's advance computes it."""
    reduced = (mx.array(np.concatenate([latents[l] for l in sorted(latents)], axis=1)) - STATE_GPU["mean"]) @ STATE_GPU["basis"]
    return {index: (reduced @ weight + bias)[None].astype(mx.bfloat16) for index, (weight, bias) in STATE_GPU["layers"].items()}


def load_correction(path):
    if path is None:
        return None
    z = np.load(path)
    return {key: z[key].astype(np.float32) for key in ("mean", "basis", "down", "up")}


G_FIX, D_FIX = load_correction(args.glm_rows_correction), load_correction(args.dsv4_rows_correction)
CONTEXT = None
if args.context_reader is not None:
    from drift.translate.context_reader import ContextRows
    CONTEXT = ContextRows(args.context_reader, window=args.reader_window)


def correction_of(fix, features, order):
    """A trained rows correction for each appended row, from the sender token it translates."""
    reduced = (features - fix["mean"]) @ fix["basis"]
    return (reduced[np.asarray(order)] @ fix["down"]) @ fix["up"]


def ridge_entries(translator, features, fix=None):
    """Sender features -> Qwen's canonical K/V per full-attention layer, from a rows translator's one output."""
    x = apply_translator(translator, features)[0]
    return split_rows(x + correction_of(fix, features, np.arange(len(features))) if fix else x, QL)


def answer(head, passage, tail, arm, rows=None, inputs=None, latents=None):
    started, cache = time.time(), lm.make_cache()
    if arm in ("text", "none"):
        ids = head + (passage if arm == "text" else []) + tail
        logits = run(cache, ids, 0)
        first = time.time() - started                                  # time to first token: the whole prompt is prefilled
        return {"text": generate(cache, logits, len(ids)), "seconds": round(time.time() - started, 2), "first_s": round(first, 4), "prefill_tokens": len(ids)}
    run(cache, head, 0)
    n = len(passage)
    if arm == "t_state":                                               # no rows: only the recurrent layers carry GLM's message
        n = len(next(iter(latents.values())))
    elif arm == "d_state":
        n = len(latents)
    elif arm in ("d_rows", "d_rows_state"):                            # DeepSeek's rows over DeepSeek's own token span
        n = len(latents)
        append_entries(cache, ridge_entries(D_ROWS, latents, D_FIX), rope, np.arange(len(head), len(head) + n), index_dim=INDEX_DIM, dtype=mx.bfloat16)
    elif arm in ("t_rows", "t_rows_state") and CONTEXT is not None:     # rows read from the whole context by the trained reader
        n = len(next(iter(latents.values())))
        rows = CONTEXT.read(np.concatenate([latents[l] for l in sorted(latents)], axis=1))
        append_entries(cache, split_rows(rows, QL), rope, np.arange(len(head), len(head) + n), index_dim=INDEX_DIM, dtype=mx.bfloat16)
    elif arm in ("t_rows", "t_rows_state") and G_ROWS is not None:
        n = len(next(iter(latents.values())))
        features = np.concatenate([latents[l] for l in sorted(latents)], axis=1)
        append_entries(cache, ridge_entries(G_ROWS, features, G_FIX), rope, np.arange(len(head), len(head) + n), index_dim=INDEX_DIM, dtype=mx.bfloat16)
    elif arm in ("t_rows", "t_rows_state"):                             # GLM's rows, translated as the loop does, over GLM's own token span
        n = len(next(iter(latents.values())))
        entries, keys, order, _ = fwd.read(latents)
        validate_translation(entries, keys, order, n, QL, INDEX_DIM)
        if G_FIX is not None:                                        # the stack's rows plus the trained correction
            entries = split_rows(join_rows(entries, QL) + correction_of(G_FIX, np.concatenate([latents[l] for l in sorted(latents)], axis=1), order), QL)
        append_entries(cache, entries, rope, len(head) + np.asarray(order, dtype=np.int64), INDEX_DIM, dtype=mx.bfloat16, index_keys=keys)
    else:
        append_entries(cache, rows, rope, np.arange(len(head), len(head) + n), index_dim=INDEX_DIM)
    for layer in QL:
        reset = selector_block_reset(cache[layer])
        if reset is not None:
            reset()
    if arm in ("t_rows_state", "t_state"):
        inputs = translated_inputs(latents)
    if arm in ("d_rows_state", "d_state"):
        inputs = as_inputs(apply_translator(D_STATE, latents))
    if arm in ("rows_state", "t_rows_state", "t_state", "d_rows_state", "d_state"):
        for index, value in inputs.items():
            lm.model.layers[index].linear_attn(value, mask=None, cache=cache[index])      # advances the window and the state
        mx.eval([part for index in inputs for part in (cache[index][0], cache[index][1]) if part is not None])
    position = len(head) + n
    logits = run(cache, tail, position)
    first = time.time() - started                                      # head prefill, memory attach and state advance, then the question
    return {"text": generate(cache, logits, position + len(tail)), "seconds": round(time.time() - started, 2), "first_s": round(first, 4),
            "prefill_tokens": len(head) + len(tail)}


results = json.loads(args.out.read_text())["results"] if args.out.exists() else []
done, t0 = {r["id"] for r in results}, time.time()
for item in json.loads(args.items.read_text()):
    if item["id"] in done:
        continue
    if "head_text" in item:                                          # a custom framing: the memory sits between these two chat-template texts
        head, tail = (tok.encode(item[key], add_special_tokens=False).ids for key in ("head_text", "tail_text"))
    else:
        head, tail = framing(item["question"])
    passage = tok.encode(item["passage"], add_special_tokens=False).ids
    read = lm.make_cache()
    CAPTURE.update(span=(len(head), len(head) + len(passage)), records={})
    run(read, head + passage, 0)
    CAPTURE["span"] = None
    inputs = dict(CAPTURE["records"])
    mx.eval(list(inputs.values()))
    if len(inputs) != len(LINEAR):
        raise SystemExit(f"captured {len(inputs)} of {len(LINEAR)} linear layers")
    rows = tap(read, QL, rope, len(head), len(head) + len(passage))
    row = {"id": item["id"], "question": item["question"], "answer": item.get("answer"), "passage_tokens": len(passage), "head": len(head)}
    for arm in ("text", "none", "rows", "rows_state"):
        if arm in ARMS:
            row[arm] = answer(head, passage, tail, arm, rows, inputs)
    if fwd is not None:
        z = np.load(args.glm_latents / f"{item['id'].split('-')[0]}.npz")
        latents = {int(k[1:]): z[k].astype(np.float32) for k in z.files if k[0] == "l" and k[1:].isdigit()}
        for arm in ("t_rows", "t_rows_state", "t_state"):
            if arm in ARMS and (arm == "t_rows" or STATE is not None):
                row[arm] = answer(head, passage, tail, arm, latents=latents)
    feature_path = args.dsv4_features / f"{item['id'].split('-')[0]}.npz" if args.dsv4_features else None
    if feature_path is not None and feature_path.exists():
        features = np.load(feature_path)["x"].astype(np.float32)
        for arm, ready in (("d_rows", D_ROWS), ("d_state", D_STATE), ("d_rows_state", D_ROWS and D_STATE)):
            if ready is not None and arm in ARMS:
                row[arm] = answer(head, passage, tail, arm, latents=features)
    results.append(row)
    args.out.write_text(json.dumps({"seconds": round(time.time() - t0, 1), "results": results}, indent=2) + "\n")
    print(json.dumps({"id": item["id"], **{arm: row[arm]["text"][:60] for arm in ("text", "rows", "t_rows", "t_rows_state") if arm in row}}), flush=True)
print(json.dumps({"items": len(results), "seconds": round(time.time() - t0, 1)}))
