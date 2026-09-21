"""PROVENANCE training on the real pair (Studio, oMLX runtime), both backbones frozen. Derived from studio_train_answer_level.py.

Task: split knowledge. The partner's document A reaches Qwen only as GLM's translated cache entries; Qwen's own note B is
text. Teacher = the same Qwen with BOTH documents as labelled text ("Document from your partner" / "Your own note").
Trainable, on top of the frozen base + fan-out translator: the rank-r correction, per-layer K/V LOUDNESS (the gate: how
strongly borrowed memory speaks next to the reader's own context) and a per-layer SOURCE TAG, a learned offset added to
every foreign key and value so that borrowed entries occupy a recognisable region of the reader's cache. Loss:
KL(teacher || student) over the teacher's own answer tokens. Scoring needs the right fact from the right source.

(original header follows)
Answer-level training of the GLM -> Qwen translator on the Studio (oMLX runtime), both backbones frozen.
Gradient path verified on 2026-09-20 (scripts/live/studio_grad_probe.py: 1.1 s per backward pass, 133.7 GB peak): oMLX's three
switches below, training mode (plain-ops linear attention), drift.serving.mlx_grad_compat (async_eval no-op, stop_gradient on index ops, plain rotary).

Teacher = Qwen3.8-Flash given the passage as text. Student = the same model given GLM's cache entries, translated by
base map (stacked3) + frozen fan-out residuals + a TRAINABLE rank-r correction and per-layer K/V loudness, injected as
a KV prefix (zero selector keys, fresh recurrent state). Loss = KL(teacher || student) on the teacher's answer tokens.

  R=/Applications/oMLX.app/Contents/Resources
  PYTHONPATH="$R/Python/framework-mlx-base/lib/python3.11/site-packages:$R:." $R/Python/cpython-3.11/bin/python3 \\
      scripts/live/studio_train_answer_level.py --triples local/studio/train/triples.json --taps local/studio/train/taps_glm --out local/studio/train/run1"""
import argparse, json, random, time
import os
for _switch in ("OMLX_QWEN4_EAGER_DISPATCH", "OMLX_QWEN4_HC_FUSED", "OMLX_QWEN4_HC_HYBRID"):
    os.environ[_switch] = "0"              # oMLX's own switches: fused / eager paths have no backward pass; must be set before its modules import
from pathlib import Path
import numpy as np
from omlx.patches.mlx_vlm_qwen4_exp_compat import apply_mlx_vlm_qwen4_exp_compat_patch
apply_mlx_vlm_qwen4_exp_compat_patch()
import mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
from tokenizers import Tokenizer
from mlx_vlm.utils import load_model
from omlx.engine.vlm import _force_qwen4_exp_sanitize_on_load
from drift.serving import mlx_grad_compat
mlx_grad_compat.apply()                                          # differentiable forward without editing oMLX's files
from drift.serving.omlx_cache import Rope, kv_layer_indices
from drift.translate.fanout import FanoutReader
from drift.translate.stacked import StackedReader

parser = argparse.ArgumentParser()
parser.add_argument("--train", type=Path, required=True, help="json list of split-knowledge items (make_d1_local.py format)")
parser.add_argument("--val", type=Path, required=True)
parser.add_argument("--init", type=Path, help="start from a trained lookup correction (A, B, loud); the tag always starts at zero")
parser.add_argument("--taps", type=Path, required=True)
parser.add_argument("--base", type=Path, default=Path("local/studio/translators/stacked3.npz"))
parser.add_argument("--fanout", type=Path, default=Path("local/studio/translators/fanout3.npz"))
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--gain-power", type=float, default=1.5)
parser.add_argument("--rank", type=int, default=64)
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--accumulate", type=int, default=2)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--answer-tokens", type=int, default=48)
parser.add_argument("--eval-every", type=int, default=100)
parser.add_argument("--eval-items", type=int, default=90)
parser.add_argument("--load", type=Path)
parser.add_argument("--suffix", default=" Answer in one short sentence, giving the value(s) directly.", help="appended to every question for teacher and student; empty = the live-qa-v3 format (preregistered for v4)")
parser.add_argument("--eval-only", action="store_true")
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
ck = Path("~/.omlx/models/Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP").expanduser()
cfg = json.loads((ck / "config.json").read_text()); t = cfg.get("text_config", cfg); rp = t.get("rope_parameters") or {}
rope = Rope(float(rp.get("rope_theta", 1e7)), int(t["head_dim"] * float(rp.get("partial_rotary_factor", 1.0))))
tok = Tokenizer.from_file(str(ck / "tokenizer.json"))
from drift.serving.studio_guard import acquire
acquire('studio_train_answer_level.py', need_gb=185)        # one model process at a time, preflight memory check, hard MLX limits
with _force_qwen4_exp_sanitize_on_load(ck):
    model = load_model(ck)
lm = model.language_model
print('plain rotary modules:', mlx_grad_compat.use_plain_rotary(model), flush=True)
GL, QL = tuple(3 + 4 * i for i in range(11)), tuple(3 + 4 * i for i in range(12))
base = StackedReader.load(args.base, GL, QL, kv_heads=2, head_dim=256)
fan = FanoutReader.load(args.fanout, base)
W0 = mx.array(np.concatenate([base.weights[l] for l in QL], axis=1)); B0 = mx.array(np.concatenate([base.biases[l] for l in QL]))
GAIN = mx.array(np.concatenate([base.gains[l] for l in QL])) ** args.gain_power
BASIS, RES = mx.array(fan.basis), {j: mx.array(r) for j, r in fan.residual.items()}
probe = lm.make_cache(); mx.eval(lm(mx.array([[1, 2, 3, 4]]), cache=probe).logits)
KV = kv_layer_indices(probe); INDEX_DIM = int(probe[KV[0]].index_keys.shape[-1])
STOP = {tok.token_to_id("<|im_end|>"), tok.token_to_id("<|endoftext|>")}
import re
SHORT = args.suffix
LINK = ("You are linked to another AI model through a shared memory. Your partner's document is in your memory, not in this chat; you were never shown it as text. "
        "The user message contains YOUR OWN note. Keep the two sources apart and use both.")


def chat(user, system=None):
    head = f"<|im_start|>system\n{system}<|im_end|>\n" if system else ""
    return tok.encode(f"{head}<|im_start|>user\n{user}{SHORT}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", add_special_tokens=False).ids


student_prompt = lambda it: chat(f"Your own note: {it['note']}\n\nQuestion: {it['question']}", LINK)
teacher_prompt = lambda it: chat(f"Document from your partner: {it['a']['text']}\n\nYour own note: {it['note']}\n\nQuestion: {it['question']}")
has = lambda text, v: re.search(rf"(?<![\w]){re.escape(v.lower())}(?![\w])", text.lower().replace(",", "")) is not None
correct = lambda text, it: all(has(text, v) for v in it["must"]) and not any(has(text, v) for v in it["must_not"])


class Correction(nn.Module):
    def __init__(self, d_in, d_out, rank):
        super().__init__()
        self.A = mx.random.normal((d_in, rank)) * (1.0 / np.sqrt(d_in)); self.B = mx.zeros((rank, d_out)); self.loud = mx.zeros((len(QL), 2))
        self.tag = mx.zeros((len(QL), 2, 512))                   # learned source marker: one offset per layer for keys and for values

    def __call__(self, x, order, extra):
        """x: centred GLM rows [T, 5632]; order: reader row -> writer row (fan-out); extra: frozen fan-out residual [T', out]."""
        xo = x[order]
        y = (xo @ W0 + extra + (xo @ self.A) @ self.B) * GAIN
        y = y.reshape(y.shape[0], len(QL), 2, 512) * mx.exp(5.0 * self.loud)[None, :, :, None] + self.tag[None]
        return y.reshape(y.shape[0], -1) + B0


def memory_inputs(pid):
    z = np.load(args.taps / f"{pid}.npz")
    x = np.concatenate([z[f"l{l}"].astype(np.float32) for l in GL], axis=1) - base.input_mean
    counts = fan.spans(x); order, which = [], []
    for row, n in enumerate(counts):
        for j in range(int(n) - 1, -1, -1):
            order.append(row); which.append(j)
    order, which = np.asarray(order), np.asarray(which)
    extra = np.zeros((len(order), W0.shape[1]), np.float32)
    zb = x @ fan.basis
    for j in np.unique(which[which > 0]):
        extra[which == j] = zb[order[which == j]] @ fan.residual[int(j)]
    return mx.array(x), mx.array(order.astype(np.int32)), mx.array(extra)


def memory_cache(corr, pid):
    x, order, extra = memory_inputs(pid)
    y = corr(x, order, extra); n = y.shape[0]
    y = y.reshape(n, len(QL), 2, 2, 256); positions = np.arange(n)
    cache = lm.make_cache()
    for j, layer in enumerate(KV):
        k = rope.apply(y[:, j, 0].transpose(1, 0, 2)[None], positions)
        cache[layer].update_and_fetch(k.astype(mx.bfloat16), y[:, j, 1].transpose(1, 0, 2)[None].astype(mx.bfloat16))
        cache[layer].update_indexer(mx.zeros((1, n, INDEX_DIM), dtype=mx.bfloat16), mx.array(positions.astype(np.int32))[None])
    return cache, n


def logits_of(ids, cache, start):
    positions = mx.arange(start, start + len(ids), dtype=mx.int32)[None]
    return lm(mx.array([ids]), cache=cache, position_ids=positions).logits[0]


def generate(ids, cache, start, limit):
    model.eval()
    logits, out, pos = logits_of(ids, cache, start)[-1], [], start + len(ids)
    for _ in range(limit):
        nxt = int(mx.argmax(logits).item())
        if nxt in STOP:
            break
        out.append(nxt); logits = logits_of([nxt], cache, pos)[-1]; pos += 1
    return out


def evaluate(corr, items, label):
    kinds = sorted({it["kind"] for it in items}); hits = {c: {k: 0 for k in kinds} for c in ("memory", "text", "own_note_only")}
    for it in items:
        cache, n = memory_cache(corr, it["pid"])
        outs = {"memory": tok.decode(generate(student_prompt(it), cache, n, 64)), "text": tok.decode(generate(teacher_prompt(it), lm.make_cache(), 0, 64)),
                "own_note_only": tok.decode(generate(student_prompt(it), lm.make_cache(), 0, 64))}
        for c, v in outs.items():
            hits[c][it["kind"]] += correct(v, it)
    count = {k: sum(it["kind"] == k for it in items) for k in kinds}
    result = {c: {**{k: round(hits[c][k] / count[k], 3) for k in kinds}, "all": round(sum(hits[c].values()) / len(items), 3)} for c in hits}
    print(label, "EM", json.dumps(result), flush=True)
    return result


load_items = lambda path: [dict(it, pid=it["a"]["id"]) for it in json.loads(path.read_text()) if (args.taps / f"{it['a']['id']}.npz").exists()]
train, val = load_items(args.train), load_items(args.val)[: args.eval_items]
random.Random(0).shuffle(train)
print("train", len(train), "val", len(val), flush=True)
corr = Correction(W0.shape[0], W0.shape[1], args.rank)
if args.init:
    corr.load_weights(str(args.init), strict=False)               # A, B, loud from the lookup run; the source tag stays zero
if args.load:
    corr.load_weights(str(args.load))
mx.eval(corr.parameters())
history = [{"step": 0, "em": evaluate(corr, val, "step 0")}]
if args.eval_only:
    raise SystemExit
teachers = {}


def teacher(it):
    key = (it["pid"], it["question"])
    if key not in teachers:
        prompt = teacher_prompt(it)
        teachers[key] = (prompt, generate(prompt, lm.make_cache(), 0, args.answer_tokens))
    return teachers[key]


def loss_fn(corr, pid, q_ids, a_ids, teacher_logp):
    cache, n = memory_cache(corr, pid)
    logits = logits_of(q_ids + a_ids, cache, n)[-len(a_ids) - 1:-1].astype(mx.float32)
    logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    return (mx.exp(teacher_logp) * (teacher_logp - logp)).sum(-1).mean()


step_fn, opt = nn.value_and_grad(corr, loss_fn), optim.Adam(learning_rate=args.lr)
acc, losses, used, t0, i = None, [], 0, time.time(), 0
while used < args.steps * args.accumulate:
    it = train[i % len(train)]; i += 1
    t_prompt, a_ids = teacher(it)
    if len(a_ids) < 2 or not correct(tok.decode(a_ids), it):
        continue
    model.train()
    t_logits = logits_of(t_prompt + a_ids, lm.make_cache(), 0)[-len(a_ids) - 1:-1].astype(mx.float32)
    teacher_logp = mx.stop_gradient(t_logits - mx.logsumexp(t_logits, axis=-1, keepdims=True))
    loss, grads = step_fn(corr, it["pid"], student_prompt(it), a_ids, teacher_logp)
    acc = grads if acc is None else nn.utils.tree_map(lambda a, g: a + g, acc, grads)
    mx.eval(loss, acc); losses.append(float(loss)); used += 1
    if used % args.accumulate == 0:
        opt.update(corr, nn.utils.tree_map(lambda g: g / args.accumulate, acc)); mx.eval(corr.parameters(), opt.state); acc = None
        done = used // args.accumulate
        if done % 10 == 0:
            print(f"update {done} | KL {np.mean(losses[-40:]):.4f} | log-loudness K/V {5 * float(corr.loud[:, 0].mean()):+.3f}/{5 * float(corr.loud[:, 1].mean()):+.3f} | tag rms {float(mx.sqrt((corr.tag * corr.tag).mean())):.4f} | {round(time.time() - t0)} s | peak {mx.get_peak_memory() / 2**30:.0f} GB", flush=True)
        if done % args.eval_every == 0:
            history.append({"step": done, "kl": float(np.mean(losses[-100:])), "em": evaluate(corr, val, f"update {done}")})
            corr.save_weights(str(args.out / "correction.safetensors")); (args.out / "history.json").write_text(json.dumps(history, indent=1))
