"""Answer-level training of translated memory for Qwen on the Studio (oMLX runtime), both backbones frozen.

Teacher: Qwen3.8-Flash reading a passage as text in the loop's follow-up layout, its system prompt, the framing line, the
passage, then the question. Student: the same model given the sender's cache instead: translated rows appended at the
passage's positions, then each linear-attention layer advancing its state by translated inputs. The sender is GLM (its
MLA latents, rows from the loop's row stack) or DeepSeek V4 (per-token features from dsv4_token_features.py, rows from a
fitted rows translator). The fitted ridge translators stay fixed. Trained on top: for the state, a low-rank correction
(one shared down-projection, an up-projection per layer) and a per-layer log-scale; with --rows-rank, a low-rank
correction of the rows as well. Loss: KL(teacher || student) on the teacher's answer tokens. Gradients reach the
corrections through Qwen's own layers, on the training-mode path studio_train_answer_level.py verified.

The state correction is linear in the reduced features, so it is written back as an ordinary translator (gain 1) that
the loop and the gates load unchanged; the rows correction is saved beside it.

  R=/Applications/oMLX.app/Contents/Resources
  PYTHONPATH="$R/Python/framework-mlx-base/lib/python3.11/site-packages:$R:." $R/Python/cpython-3.11/bin/python3 \\
      scripts/live/studio_train_memory_answer.py --triples out/ftrain_triples.json --latents out/ftrain --val-latents out/fwd48 \\
      --state out/state_g2q_r0.1.npz --out out/memory_answer_run1
"""
import argparse, json, os, random, shutil, time
for _switch in ("OMLX_QWEN4_EAGER_DISPATCH", "OMLX_QWEN4_HC_FUSED", "OMLX_QWEN4_HC_HYBRID"):
    os.environ[_switch] = "0"              # oMLX's fused and eager paths have no backward pass; set before its modules import
from pathlib import Path
import numpy as np
from omlx.patches.mlx_vlm_qwen4_exp_compat import apply_mlx_vlm_qwen4_exp_compat_patch
apply_mlx_vlm_qwen4_exp_compat_patch()
import mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
from tokenizers import Tokenizer
from mlx_vlm.utils import load_model
from omlx.engine.vlm import _force_qwen4_exp_sanitize_on_load
from drift.serving import mlx_grad_compat
mlx_grad_compat.apply()
from drift.serving.mcdma_forward import validate_translation
from drift.serving.omlx_cache import Rope, selector_block_reset
from drift.serving.studio_guard import acquire
from drift.translate import ridge_map
from drift.translate.fanout import CorrectedFanoutReader, FanoutReader
from drift.translate.index_keys import IndexKeyReader
from drift.translate.mlx_reader import MlxForwardReader
from drift.translate.stacked import StackedReader

SYSTEM = ("You are linked to another AI model through a shared memory that fills while you work. What your partner knows "
          "and writes arrives in that memory, never in this chat. Use it as your own recollection.")    # the loop's Qwen system prompt
FRAME = "My partner's message, from our shared memory:"                                               # the loop's framing line, by default
parser = argparse.ArgumentParser()
parser.add_argument("--triples", type=Path, required=True, help="json: {train: [...], val: [...]} of {pid, passage, question, answer}")
parser.add_argument("--latents", type=Path, required=True, help="GLM latents per training passage, l{layer} arrays")
parser.add_argument("--val-latents", type=Path, required=True)
parser.add_argument("--state", type=Path, required=True, help="the fitted ridge state translator to correct")
parser.add_argument("--artifacts", type=Path, default=Path("local/live"), help="the loop's forward row stack, for GLM")
parser.add_argument("--rows-translator", type=Path, help="a fitted rows translator in place of the stack, as for DeepSeek")
parser.add_argument("--rows-rank", type=int, default=0, help="rank of a trained correction of the rows; 0 keeps them fixed")
parser.add_argument("--context-reader", type=Path, help="a pretrained contextual reader to fine-tune on answers; its rows.npz is the fixed rows translator")
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--rank", type=int, default=32)
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--accumulate", type=int, default=4)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--answer-tokens", type=int, default=32)
parser.add_argument("--eval-every", type=int, default=50)
parser.add_argument("--eval-items", type=int, default=60)
parser.add_argument("--eval-tokens", type=int, default=48, help="answer length at evaluation; long enough to reach the answer")
parser.add_argument("--eval-only", action="store_true")
parser.add_argument("--frame", default=FRAME, help="the line that opens the memory's block, for teacher and student alike")
parser.add_argument("--budget", type=int, help="attend densely up to this many tokens, for teacher and student alike")
parser.add_argument("--cache-passages", type=int, default=64, help="passages whose translated memory stays cached; the oldest are dropped")
parser.add_argument("--seed", type=int, default=0, help="orders the training triples; a later phase takes a new one so it meets new examples")
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
acquire("studio_train_memory_answer.py", need_gb=185)
ck = Path("~/.omlx/models/Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP").expanduser()
cfg = json.loads((ck / "config.json").read_text()); t = cfg.get("text_config", cfg); rp = t.get("rope_parameters") or {}
rope = Rope(float(rp.get("rope_theta", 1e7)), int(t["head_dim"] * float(rp.get("partial_rotary_factor", 1.0))))
tok = Tokenizer.from_file(str(ck / "tokenizer.json"))
with _force_qwen4_exp_sanitize_on_load(ck):
    model = load_model(ck)
lm = model.language_model
print("plain rotary modules:", mlx_grad_compat.use_plain_rotary(model), flush=True)
if args.budget:                                                         # every full-attention layer attends the whole prefix up to the budget
    for layer in lm.model.layers:
        if not getattr(layer, "is_linear", False):
            layer.self_attn.indexer.token_budget = args.budget
            layer.self_attn.indexer.block_topk = args.budget // layer.self_attn.indexer.compress_ratio
GL, QL, INDEX_DIM = tuple(3 + 4 * i for i in range(11)), tuple(3 + 4 * i for i in range(12)), 128
fwd = None
if args.rows_translator is None:                                        # GLM's rows come from the loop's own row stack
    stack = StackedReader.load(args.artifacts / "stacked3.npz", GL, QL, kv_heads=2, head_dim=256)
    fwd = MlxForwardReader(CorrectedFanoutReader.load(args.artifacts / "v4_correction.safetensors", FanoutReader.load(args.artifacts / "fanout3.npz", stack)),
                           IndexKeyReader.load(args.artifacts / "index3.npz", QL, stack.sha256), 1.5)
ridge = ridge_map.load(args.state)
if args.context_reader and args.rows_translator:
    raise SystemExit("a context reader brings its own rows translator")
reader, READER_CS, READER_TS, READER_SPREAD = None, None, None, None
READER_INPUT, READER_IM = "reduced", None                               # a reader of GLM's full latents keeps them per passage
if args.context_reader:                                                 # its fixed rows translator becomes the base the reader corrects
    from drift.translate.context_reader import ContextReader
    reader_config = json.loads((args.context_reader / "config.json").read_text())
    reader = ContextReader(**reader_config["model"])
    reader.load_weights(str(args.context_reader / "reader.safetensors"))
    scales = np.load(args.context_reader / "scales.npz")
    READER_CS, READER_TS = mx.array(scales["component"]), mx.array(scales["target"])
    READER_INPUT = reader_config.get("input", "reduced")
    READER_IM = mx.array(scales["input_mean"]) if "input_mean" in scales.files else mx.zeros(scales["component"].shape)
    if (args.context_reader / "gain.npz").exists():                     # the spread gain stays fixed; the reader learns under it
        spread = np.load(args.context_reader / "gain.npz")
        READER_SPREAD = (mx.array(spread["centre"]), mx.array(spread["gain"]))
    args.rows_translator = args.context_reader / "rows.npz"
rows_translator = ridge_map.load(args.rows_translator) if args.rows_translator else None
if ridge["phases"] != 1 or (rows_translator and rows_translator["phases"] != 1):
    raise SystemExit("a phased translator cannot be corrected here")
if rows_translator and not (np.array_equal(rows_translator["mean"], ridge["mean"]) and np.array_equal(rows_translator["basis"], ridge["basis"])):
    raise SystemExit("the rows and state translators must share one reduction, as fits on the same sender features do")
LAYERS = sorted(ridge["layers"])
W = mx.array(np.stack([ridge["layers"][l][0] * ridge["layers"][l][2][None, :] for l in LAYERS]))   # gain folded into the weight
Bias = mx.array(np.stack([ridge["layers"][l][1] for l in LAYERS]))
STOP = {tok.token_to_id("<|im_end|>"), tok.token_to_id("<|endoftext|>")}
encode = lambda text: tok.encode(text, add_special_tokens=False).ids
HEAD = encode(f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{args.frame}\n")
tail = lambda question: encode(f"\n\n{question}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")


ROW_WIDTH = len(QL) * 1024                                               # per full-attention layer: 2 heads x 256 of K, then of V
SCALE = None                                                            # per reduced component, the training spread; set before training


class Correction(nn.Module):
    def __init__(self, rank, rows_rank):
        super().__init__()
        self.down = mx.random.normal((W.shape[1], rank)) * (1.0 / np.sqrt(W.shape[1]))
        self.up = mx.zeros((len(LAYERS), rank, W.shape[2]))
        self.log_scale = mx.zeros((len(LAYERS),))
        if rows_rank:
            self.rows_down = mx.random.normal((W.shape[1], rows_rank)) * (1.0 / np.sqrt(W.shape[1]))
            self.rows_up = mx.zeros((rows_rank, ROW_WIDTH))
        if reader is not None:
            self.reader = reader

    def __call__(self, z):
        """z: reduced sender features [T, rank_in] -> each linear layer's input [L, T, width]."""
        corrected = (z[None] @ W + Bias[:, None, :]) * mx.exp(self.log_scale)[:, None, None]
        return corrected + (((z / SCALE) @ self.down)[None] @ self.up)          # corrections see standardised components

    def rows(self, z, order, base, full=None):
        """The fixed rows [R, ROW_WIDTH], each from sender token order[r], plus the trained corrections if any."""
        base = base.astype(mx.float32)
        if "reader" in self:                                            # the reader sees the whole context, then each row takes its token's output
            seen = full.astype(mx.float32) if READER_INPUT == "full" else z
            base = base + self.reader((seen - READER_IM) / READER_CS)[order] * READER_TS
            if READER_SPREAD is not None:
                centre, gain = READER_SPREAD
                base = centre + gain * (base - centre)
        return base + ((z[order] / SCALE) @ self.rows_down) @ self.rows_up if "rows_up" in self else base


memory = {}


def memory_of(folder, pid):
    """Per passage: reduced sender features, fixed rows, selector keys, row order, token count and, for a reader of full
    latents, the latents themselves; all constants."""
    if pid not in memory:
        z = np.load(folder / f"{pid}.npz")
        if "x" in z.files:                                              # DeepSeek: one feature row per token
            if rows_translator is None:
                raise SystemExit("per-token features need --rows-translator")
            features = z["x"].astype(np.float32)
        else:                                                           # GLM: its MLA latents in layer order
            latents = {int(k[1:]): z[k].astype(np.float32) for k in z.files if k[0] == "l" and k[1:].isdigit()}
            features = np.concatenate([latents[l] for l in sorted(latents)], axis=1)
        n = len(features)
        if rows_translator is not None:
            base, order = ridge_map.apply(rows_translator, features)[0], np.arange(n)
            keys = np.zeros((n, INDEX_DIM), np.float32)
        else:
            entries, selector, order, _ = fwd.read(latents)
            validate_translation(entries, selector, order, n, QL, INDEX_DIM)
            base = np.concatenate([np.concatenate((entries[l][0].reshape(len(order), -1), entries[l][1].reshape(len(order), -1)), axis=1) for l in QL], axis=1)
            keys = np.stack([np.asarray(selector[l], np.float32) for l in QL])
        reduced = (features - ridge["mean"]) @ ridge["basis"]
        full = mx.array(features.astype(np.float16)) if READER_INPUT == "full" else None
        memory[pid] = (mx.array(reduced), mx.array(base.astype(np.float16)), keys, np.asarray(order, dtype=np.int64), n, full)   # half the memory
        for old in list(memory)[:max(0, len(memory) - max(1, args.cache_passages))]:
            del memory[old]                                             # dicts keep insertion order, so the oldest go first
    return memory[pid]


def run(ids, cache, start):
    return lm(mx.array([ids]), cache=cache, position_ids=mx.arange(start, start + len(ids), dtype=mx.int32)[None]).logits[0]


def student_cache(corr, folder, pid):
    reduced, base, keys, order, n, full = memory_of(folder, pid)
    cache = lm.make_cache()
    run(HEAD, cache, 0)
    rows = corr.rows(reduced, mx.array(order.astype(np.int32)), base, full).reshape(len(order), len(QL), 2, 2, 256)
    positions = len(HEAD) + order
    for j, layer in enumerate(QL):                                       # appended after the head at the passage's positions
        k = rope.apply(rows[:, j, 0].transpose(1, 0, 2)[None], positions)
        cache[layer].update_and_fetch(k.astype(mx.bfloat16), rows[:, j, 1].transpose(1, 0, 2)[None].astype(mx.bfloat16))
        selector = keys if keys.ndim == 2 else keys[j]
        cache[layer].update_indexer(mx.array(selector)[None].astype(mx.bfloat16), mx.array(positions.astype(np.int32))[None])
    for layer in QL:
        reset = selector_block_reset(cache[layer])
        if reset is not None:
            reset()
    inputs = corr(reduced)
    for j, layer in enumerate(LAYERS):
        lm.model.layers[layer].linear_attn(inputs[j][None].astype(mx.bfloat16), mask=None, cache=cache[layer])
    return cache, len(HEAD) + n


def generate(ids, cache, start, limit):
    logits, out, position = run(ids, cache, start)[-1], [], start + len(ids)
    for _ in range(limit):
        token = int(mx.argmax(logits).item())
        if token in STOP:
            break
        out.append(token); logits = run([token], cache, position)[-1]; position += 1
    return out


def teacher_ids(item):
    return HEAD + encode(item["passage"]) + tail(item["question"])


TEXT_HITS = {}                                                          # the text arm never changes, so each item is read once


def evaluate(corr, items, label):
    model.eval()
    hits = {"memory": 0, "text": 0}
    for item in items:
        cache, start = student_cache(corr, args.val_latents, item["pid"])
        hits["memory"] += item["answer"].lower() in tok.decode(generate(tail(item["question"]), cache, start, args.eval_tokens)).lower()
        key = (item["pid"], item["question"])
        if key not in TEXT_HITS:
            TEXT_HITS[key] = item["answer"].lower() in tok.decode(generate(teacher_ids(item), lm.make_cache(), 0, args.eval_tokens)).lower()
        hits["text"] += TEXT_HITS[key]
    result = {key: round(value / len(items), 3) for key, value in hits.items()}
    print(label, "answers", result, flush=True)
    return result


def export(corr, folder):
    """The state correction folded into an ordinary translator with gain 1, and any rows correction beside it; both take
    raw reduced features, so the standardisation is folded into their down-projections."""
    spread = np.array(SCALE)[:, None]
    weights, biases = ridge_map.fold(np.array(W), np.array(Bias), np.array(corr.log_scale), np.array(corr.down) / spread, np.array(corr.up))
    np.savez(folder / "state.npz", mean=ridge["mean"], basis=ridge["basis"], layers=np.asarray(LAYERS, np.int32),    # full precision: a resumed run checks the reduction is shared
             **{f"W{l}": weights[j].astype(np.float16) for j, l in enumerate(LAYERS)}, **{f"b{l}": biases[j].astype(np.float32) for j, l in enumerate(LAYERS)},
             **{f"gain{l}": np.ones(W.shape[2], np.float32) for l in LAYERS})
    if "reader" in corr:                                                 # a loadable reader folder: weights, config, rows translator, scales
        corr.reader.save_weights(str(folder / "reader.safetensors"))
        for name in ("config.json", "rows.npz", "scales.npz", "gain.npz"):
            if (args.context_reader / name).exists():
                shutil.copyfile(args.context_reader / name, folder / name)
    if "rows_up" in corr:
        np.savez(folder / "rows_correction.npz", mean=ridge["mean"], basis=ridge["basis"].astype(np.float16),
                 down=(np.array(corr.rows_down) / spread).astype(np.float32), up=np.array(corr.rows_up).astype(np.float32))


data = json.loads(args.triples.read_text())
train = [x for x in data["train"] if (args.latents / f"{x['pid']}.npz").exists()]
val = [x for x in data["val"] if (args.val_latents / f"{x['pid']}.npz").exists()][: args.eval_items]
random.Random(args.seed).shuffle(train)
print("train", len(train), "val", len(val), flush=True)
sample = np.concatenate([np.array(memory_of(args.latents, pid)[0]) for pid in list(dict.fromkeys(x["pid"] for x in train))[:64]])
SCALE = mx.array(np.maximum(sample.std(0), 1e-6).astype(np.float32))
corr = Correction(args.rank, args.rows_rank)
mx.eval(corr.parameters())
history = [{"step": 0, **evaluate(corr, val, "step 0")}]
if args.eval_only:
    raise SystemExit
teachers = {}


def teacher(item):
    key = (item["pid"], item["question"])
    if key not in teachers:
        model.eval()
        teachers[key] = generate(teacher_ids(item), lm.make_cache(), 0, args.answer_tokens)
    return teachers[key]


def loss_fn(corr, pid, question_ids, answer_ids, teacher_logp):
    cache, start = student_cache(corr, args.latents, pid)
    logits = run(question_ids + answer_ids, cache, start)[-len(answer_ids) - 1:-1].astype(mx.float32)
    logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    return (mx.exp(teacher_logp) * (teacher_logp - logp)).sum(-1).mean()


step_fn, opt = nn.value_and_grad(corr, loss_fn), optim.Adam(learning_rate=args.lr)
acc, losses, used, started, i = None, [], 0, time.time(), 0
while used < args.steps * args.accumulate:
    item = train[i % len(train)]; i += 1
    answer_ids = teacher(item)
    if len(answer_ids) < 2 or item["answer"].lower() not in tok.decode(answer_ids).lower():
        continue
    memory_of(args.latents, item["pid"])                              # the row reader evaluates, so it runs outside the gradient
    model.train()
    t_logits = run(teacher_ids(item) + answer_ids, lm.make_cache(), 0)[-len(answer_ids) - 1:-1].astype(mx.float32)
    teacher_logp = mx.stop_gradient(t_logits - mx.logsumexp(t_logits, axis=-1, keepdims=True))
    loss, grads = step_fn(corr, item["pid"], tail(item["question"]), answer_ids, teacher_logp)
    acc = grads if acc is None else nn.utils.tree_map(lambda a, g: a + g, acc, grads)
    mx.eval(loss, acc); losses.append(float(loss)); used += 1
    if used % args.accumulate == 0:
        opt.update(corr, nn.utils.tree_map(lambda g: g / args.accumulate, acc)); mx.eval(corr.parameters(), opt.state); acc = None
        done = used // args.accumulate
        if done % 10 == 0:
            print(f"update {done} | KL {np.mean(losses[-10 * args.accumulate:]):.4f} | scale {float(mx.exp(corr.log_scale).mean()):.3f} | "
                  f"{round(time.time() - started)} s | peak {mx.get_peak_memory() / 2**30:.0f} GB", flush=True)
        if done % args.eval_every == 0:
            history.append({"step": done, "kl": float(np.mean(losses[-args.eval_every * args.accumulate:])), **evaluate(corr, val, f"update {done}")})
            export(corr, args.out); (args.out / "history.json").write_text(json.dumps(history, indent=1))
export(corr, args.out)
(args.out / "history.json").write_text(json.dumps(history, indent=1))
print(json.dumps({"updates": used // args.accumulate, "seconds": round(time.time() - started, 1)}))
