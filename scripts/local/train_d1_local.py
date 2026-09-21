"""Answer-level training on the SPLIT-KNOWLEDGE task (conj / sum / hop), both backbones frozen, local pair.
The partner's document A reaches the reader only as translated cache entries; the reader's own note B is text.
Teacher = the same reader with both documents as labelled text. Scoring needs the right fact from the right source,
so this measures whether answer-level training fixes source confusion and loudness, not just recall.

(derived from train_answer_level.py)
Answer-level training of a cache translator with BOTH backbones frozen (offline prototype on the local pair).

Teacher = the reader given the passage as TEXT. Student = the same reader given only the writer's cache entries,
translated and injected as a KV prefix. The translator is the ridge map plus a trainable low-rank correction and
per-layer K/V loudness; the loss is KL(teacher || student) over the teacher's own answer tokens. Nothing in either
model is updated. (XKV-style answer-loss training of gated cache translation, here through a 4-bit MLX reader.)
  PYTHONPATH=. .venv-next/bin/python scripts/local/train_answer_level.py --steps 400 --out local/local_pair/run1"""
import argparse, hashlib, json, random, sys, time
from pathlib import Path
import numpy as np
import mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
from mlx_lm import load
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "live"))
from questions_v3 import QUESTION
from drift.serving.omlx_cache import Rope

parser = argparse.ArgumentParser()
parser.add_argument("--reader", default=str(Path("~/.omlx/models/Vontra/Qwen3.8-27B-MLX-4bit").expanduser()))
parser.add_argument("--ridge", type=Path, default=Path("local/local_pair/ridge.npz"))
parser.add_argument("--writer-taps", type=Path, default=Path("local/local_pair/taps_writer"))
parser.add_argument("--train", type=Path, default=Path("local/local_pair/d1/train.json"))
parser.add_argument("--val", type=Path, default=Path("local/local_pair/d1/val.json"))
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--rank", type=int, default=64)
parser.add_argument("--steps", type=int, default=400)
parser.add_argument("--accumulate", type=int, default=4)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--answer-tokens", type=int, default=64)
parser.add_argument("--eval-every", type=int, default=100)
parser.add_argument("--eval-items", type=int, default=90)
parser.add_argument("--gain-power", type=float, default=1.0)
parser.add_argument("--eval-only", action="store_true")
parser.add_argument("--load", type=Path, help="trained correction weights to evaluate or continue from")
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
model, tok = load(args.reader)
cfg = json.loads((Path(args.reader) / "config.json").read_text()); t = cfg.get("text_config", cfg); rp = t.get("rope_parameters") or {}
rope = Rope(float(rp.get("rope_theta", 1e7)), int(t["head_dim"] * float(rp.get("partial_rotary_factor", 1.0))))
heads, dim = int(t["num_key_value_heads"]), int(t["head_dim"])
z = np.load(args.ridge); meta = json.loads(str(z["meta"]))
R_LAYERS, WIDTH = meta["reader_layers"], meta["reader_width"]
X_MEAN, W0, B0 = mx.array(z["x_mean"]), mx.array(z["W"]), mx.array(z["b"])
GAIN = mx.array(z["gain"]) ** args.gain_power
STOP = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}
LINK = ("You are linked to another AI model through a shared memory. Your partner's document is in your memory, not in this chat; you were never shown it as text. "
        "The user message contains YOUR OWN note. Keep the two sources apart and use both.")
chat = lambda user, system=None: (f"<|im_start|>system\n{system}<|im_end|>\n" if system else "") + f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
SHORT = " Answer in one short sentence, giving the value(s) directly."
student_prompt = lambda it: chat(f"Your own note: {it['note']}\n\nQuestion: {it['question']}{SHORT}", LINK)
teacher_prompt = lambda it: chat(f"Document from your partner: {it['a']['text']}\n\nYour own note: {it['note']}\n\nQuestion: {it['question']}{SHORT}")
import re
has = lambda text, v: re.search(rf"(?<![\w]){re.escape(v.lower())}(?![\w])", text.lower().replace(",", "")) is not None
correct = lambda text, it: all(has(text, v) for v in it["must"]) and not any(has(text, v) for v in it["must_not"])
stack = lambda f: np.concatenate([np.concatenate((f[k].reshape(len(f[k]), -1), f["v" + k[1:]].reshape(len(f[k]), -1)), 1) for k in sorted((n for n in f.files if n.startswith("k")), key=lambda n: int(n[1:]))], 1).astype(np.float32)


class Correction(nn.Module):
    """y = (x W0 + (x A) B) * gain * exp(loud) + b : low-rank answer-level correction and per-layer K / V loudness."""
    def __init__(self, d_in, d_out, rank, layers):
        super().__init__()
        self.A = mx.random.normal((d_in, rank)) * (1.0 / np.sqrt(d_in))
        self.B = mx.zeros((rank, d_out))
        self.loud = mx.zeros((layers, 2))

    def __call__(self, x):
        y = (x @ W0 + (x @ self.A) @ self.B) * GAIN
        y = y.reshape(x.shape[0], len(R_LAYERS), 2, WIDTH // 2) * mx.exp(5.0 * self.loud)[None, :, :, None]   # x5: loudness should move faster than the map
        return y.reshape(x.shape[0], -1) + B0


def triples(path):
    return [dict(it, pid=it["a"]["id"]) for it in json.loads(path.read_text()) if (args.writer_taps / f"{it['a']['id']}.npz").exists()]


def inject(cache, y, n):
    positions = np.arange(n)
    y = y.reshape(n, len(R_LAYERS), 2, heads, dim)
    for j, layer in enumerate(R_LAYERS):
        k = rope.apply(y[:, j, 0].transpose(1, 0, 2)[None], positions)
        cache[layer].update_and_fetch(k.astype(mx.bfloat16), y[:, j, 1].transpose(1, 0, 2)[None].astype(mx.bfloat16))


def logits_of(ids, cache):
    out = model(mx.array([ids]), cache=cache)
    return (out.logits if hasattr(out, "logits") else out)[0]


def generate(prompt_ids, cache, limit):
    model.eval()
    logits, out = logits_of(prompt_ids, cache)[-1], []
    for _ in range(limit):
        nxt = int(mx.argmax(logits).item())
        if nxt in STOP:
            break
        out.append(nxt); logits = logits_of([nxt], cache)[-1]
    return out


def memory_cache(corr, item):
    x = mx.array(stack(np.load(args.writer_taps / f"{item['pid']}.npz"))) - X_MEAN
    cache = model.make_cache(); inject(cache, corr(x), x.shape[0])
    return cache


def evaluate(corr, items, label):
    kinds = sorted({it["kind"] for it in items}); hits = {c: {k: 0 for k in kinds} for c in ("memory", "text", "own_note_only")}
    for it in items:
        q = tok.encode(student_prompt(it), add_special_tokens=False)
        texts = {"memory": tok.decode(generate(q, memory_cache(corr, it), 96)), "own_note_only": tok.decode(generate(q, model.make_cache(), 96)),
                 "text": tok.decode(generate(tok.encode(teacher_prompt(it), add_special_tokens=False), model.make_cache(), 96))}
        for c, v in texts.items():
            hits[c][it["kind"]] += correct(v, it)
    n = {k: sum(it["kind"] == k for it in items) for k in kinds}
    result = {c: {**{k: round(hits[c][k] / n[k], 3) for k in kinds}, "all": round(sum(hits[c].values()) / len(items), 3)} for c in hits}
    print(label, "EM", json.dumps(result), flush=True)
    return result


rng = random.Random(0)
train, val = triples(args.train), triples(args.val)
rng.shuffle(train); val = val[: args.eval_items]
print("train triples", len(train), "| val triples", len(val), flush=True)
corr = Correction(W0.shape[0], W0.shape[1], args.rank, len(R_LAYERS))
if args.load:
    corr.load_weights(str(args.load))
mx.eval(corr.parameters())
history = [{"step": 0, "em": evaluate(corr, val, "step 0 (ridge only)")}]
if args.eval_only:
    raise SystemExit
teacher_cache = {}


def teacher(item):
    """The reader's own greedy answer when it is given the passage as text (cached), and its logits over that answer."""
    key = (item["pid"], item["question"])
    if key not in teacher_cache:
        prompt = tok.encode(teacher_prompt(item), add_special_tokens=False)
        teacher_cache[key] = (prompt, generate(prompt, model.make_cache(), args.answer_tokens))
    return teacher_cache[key]


def loss_fn(corr, x, q_ids, a_ids, teacher_logp):
    cache = model.make_cache(); inject(cache, corr(x), x.shape[0])
    logits = logits_of(q_ids + a_ids, cache)[-len(a_ids) - 1:-1].astype(mx.float32)
    logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    return (mx.exp(teacher_logp) * (teacher_logp - logp)).sum(-1).mean()


step_fn = nn.value_and_grad(corr, loss_fn)
opt = optim.Adam(learning_rate=args.lr)
acc, losses, t0 = None, [], time.time()
for step in range(1, args.steps * args.accumulate + 1):
    item = train[(step - 1) % len(train)]
    t_prompt, a_ids = teacher(item)
    if len(a_ids) < 2 or not correct(tok.decode(a_ids), item):
        continue                                                    # only distil answers the teacher itself gets right
    model.train()                                                   # differentiable path for the linear-attention layers (no dropout in this model)
    t_logits = logits_of(t_prompt + a_ids, model.make_cache())[-len(a_ids) - 1:-1].astype(mx.float32)
    teacher_logp = mx.stop_gradient(t_logits - mx.logsumexp(t_logits, axis=-1, keepdims=True))
    x = mx.array(stack(np.load(args.writer_taps / f"{item['pid']}.npz"))) - X_MEAN
    q_ids = tok.encode(student_prompt(item), add_special_tokens=False)
    loss, grads = step_fn(corr, x, q_ids, a_ids, teacher_logp)
    acc = grads if acc is None else nn.utils.tree_map(lambda a, g: a + g, acc, grads)
    mx.eval(loss, acc); losses.append(float(loss))
    if step % args.accumulate == 0:
        opt.update(corr, nn.utils.tree_map(lambda g: g / args.accumulate, acc)); mx.eval(corr.parameters(), opt.state); acc = None
        done = step // args.accumulate
        if done % 10 == 0:
            print(f"update {done} | KL {np.mean(losses[-40:]):.4f} | log-loudness K/V mean {5 * float(corr.loud[:, 0].mean()):+.3f}/{5 * float(corr.loud[:, 1].mean()):+.3f} | {round(time.time() - t0)} s", flush=True)
        if done % args.eval_every == 0:
            history.append({"step": done, "kl": float(np.mean(losses[-100:])), "em": evaluate(corr, val, f"update {done}")})
            corr.save_weights(str(args.out / "correction.safetensors"))
            (args.out / "history.json").write_text(json.dumps(history, indent=1))
